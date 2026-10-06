"""
Análise estatística complementar dos experimentos de TLR (requisitos -> classes Java, iTrust).

Entradas:
  - data/processed/predictions_consolidated.csv  (predições por par das 15 combinações)
  - data/raw/requirements, data/raw/test_cases, data/raw/ground_truth.txt

Saídas (results/analysis/):
  - metrics_full.csv      : matriz de confusão, P, R, F1, F2 e IC 95% (bootstrap) das 15 combinações
  - paired_tests.csv      : McNemar exato + bootstrap pareado de F1 (referência vs. demais), correção de Holm
  - baselines.csv         : baselines clássicos (TF-IDF, BM25, LSI) nas 858 pares e no espaço completo
  - sensitivity.csv       : sensibilidade de Precisão/F1 à proporção positivos:negativos
  - cost_table.csv        : custos (faturamento OpenRouter), tokens estimados de entrada e falhas

Reprodutibilidade: todas as sementes aleatórias são fixas (42).
"""
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize

BASE = Path(__file__).resolve().parent.parent
sys.path.append(str(Path(__file__).resolve().parent))

from data_loader import (  # noqa: E402
    load_requirements,
    load_test_cases,
    load_ground_truth,
    get_stratified_evaluation_dataset,
)
from prompt_templates import get_zero_shot_prompt, get_few_shot_prompt, get_cot_prompt  # noqa: E402

PRED_CSV = BASE / "data" / "processed" / "predictions_consolidated.csv"
OUT = BASE / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BOOT = 10_000

MODEL_LABEL = {
    "anthropic/claude-3.5-sonnet": "Claude Sonnet 4",
    "openai/gpt-4o-mini": "GPT-4o-mini",
    "deepseek/deepseek-chat": "DeepSeek V3",
    "perplexity/llama-3.1-sonar-large-128k-chat": "Perplexity Sonar",
    "gemini-3.6-flash": "Gemini Flash",
}
STRAT_LABEL = {"Zero-Shot": "Zero-shot", "Few-Shot": "Few-shot", "Chain-of-Thought (CoT)": "CoT"}
STRAT_ORDER = ["Zero-shot", "Few-shot", "CoT"]
MODEL_ORDER = ["Claude Sonnet 4", "GPT-4o-mini", "DeepSeek V3", "Perplexity Sonar", "Gemini Flash"]
REF = ("GPT-4o-mini", "Few-shot")

# Custo faturado no painel do OpenRouter (Activity > Spend by Model). Gemini: Free Tier (US$ 0).
BILLED_USD = {"Claude Sonnet 4": 17.91, "Perplexity Sonar": 12.94, "DeepSeek V3": 1.45,
              "GPT-4o-mini": 0.72, "Gemini Flash": 0.0}


# ----------------------------------------------------------------------------- métricas
def prf(tp, fp, fn):
    tp, fp, fn = np.asarray(tp, float), np.asarray(fp, float), np.asarray(fn, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f2 = np.where(5 * tp + 4 * fn + fp > 0, 5 * tp / (5 * tp + 4 * fn + fp), 0.0)
    return p, r, f1, f2


def counts(y, yhat):
    y, yhat = np.asarray(y, int), np.asarray(yhat, int)
    return (int(((y == 1) & (yhat == 1)).sum()), int(((y == 0) & (yhat == 1)).sum()),
            int(((y == 0) & (yhat == 0)).sum()), int(((y == 1) & (yhat == 0)).sum()))


def boot_metrics(y, yhat, idx):
    yt, yp = y[idx], yhat[idx]
    tp = (yt & yp).sum(1)
    fp = ((1 - yt) & yp).sum(1)
    fn = (yt & (1 - yp)).sum(1)
    return prf(tp, fp, fn)


def ci(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


# ----------------------------------------------------------------------------- baselines IR
JAVA_STOP = set("""import package public private protected static final void return new class interface extends
implements throws throw try catch finally if else for while do switch case break continue this super null true false
int long boolean double float char byte short string object list arraylist hashmap map set exception sql
integer date get override serializable abstract synchronized instanceof""".split())
STOP = set(ENGLISH_STOP_WORDS) | JAVA_STOP


def stem(w):
    for suf in ("ations", "ation", "ings", "ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def tokenize(text):
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    toks = re.findall(r"[A-Za-z]{2,}", text)
    return [stem(t.lower()) for t in toks if t.lower() not in STOP]


def bm25_matrix(q_tokens, d_tokens, k1=1.5, b=0.75):
    N = len(d_tokens)
    avgdl = np.mean([len(d) for d in d_tokens])
    df = {}
    for d in d_tokens:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    tfs = [pd.Series(d).value_counts().to_dict() for d in d_tokens]
    scores = np.zeros((len(q_tokens), N))
    for qi, q in enumerate(q_tokens):
        for t in set(q):
            if t not in df:
                continue
            idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)
            for di, d in enumerate(d_tokens):
                f = tfs[di].get(t, 0)
                if f:
                    scores[qi, di] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avgdl))
    return scores


def best_threshold(s, y):
    o = np.argsort(-s, kind="stable")
    s_, y_ = s[o], y[o]
    tp = np.cumsum(y_)
    fp = np.cumsum(1 - y_)
    P = y.sum()
    f1 = 2 * tp / (2 * tp + fp + (P - tp))
    last = np.r_[s_[1:] != s_[:-1], True]
    f1 = np.where(last, f1, -1.0)
    i = int(np.argmax(f1))
    return float(s_[i]), float(f1[i])


def build_baseline_scores():
    reqs = load_requirements()
    codes = load_test_cases()
    req_ids = sorted(reqs)
    code_ids = sorted(codes)
    qt = [tokenize(reqs[r]) for r in req_ids]
    dt = [tokenize(codes[c]) for c in code_ids]

    vec = TfidfVectorizer(analyzer=lambda x: x, sublinear_tf=True)
    X = vec.fit_transform(qt + dt)
    R, C = X[: len(qt)], X[len(qt):]
    tfidf = (R @ C.T).toarray()

    svd = TruncatedSVD(n_components=100, random_state=SEED)
    Z = normalize(svd.fit_transform(X))
    lsi = Z[: len(qt)] @ Z[len(qt):].T

    bm = bm25_matrix(qt, dt)
    bm = bm / np.maximum(bm.max(axis=1, keepdims=True), 1e-9)  # normaliza por requisito
    return req_ids, code_ids, {"TF-IDF (cosseno)": tfidf, "BM25": bm, "LSI (SVD, k=100)": lsi}


# ----------------------------------------------------------------------------- principal
def main():
    rng = np.random.default_rng(SEED)
    pred = pd.read_csv(PRED_CSV)
    pred["Modelo"] = pred["model"].map(MODEL_LABEL)
    pred["Estratégia"] = pred["strategy"].map(STRAT_LABEL)
    n = 858

    ref_df = get_stratified_evaluation_dataset().reset_index(drop=True)
    y = ref_df["ground_truth"].to_numpy(int)
    req_of = ref_df["req_id"].to_numpy()
    test_of = ref_df["test_id"].to_numpy()

    configs = {}
    api_err = {}
    for (m, s), d in pred.groupby(["Modelo", "Estratégia"]):
        d = d.sort_values("pair_id")
        assert len(d) == n and (d["ground_truth"].to_numpy() == y).all()
        configs[(m, s)] = d["prediction"].to_numpy(int)
        api_err[(m, s)] = int(d["api_error"].sum())

    idx = rng.integers(0, n, size=(N_BOOT, n), dtype=np.int16)

    # ---------------- métricas e ICs
    rows = []
    boots = {}
    for m in MODEL_ORDER:
        for s in STRAT_ORDER:
            yh = configs[(m, s)]
            tp, fp, tn, fn = counts(y, yh)
            p, r, f1, f2 = (float(v) for v in prf(tp, fp, fn))
            bp, br, bf1, bf2 = boot_metrics(y, yh, idx)
            boots[(m, s)] = bf1
            rows.append({
                "Modelo": m, "Estratégia": s, "VP": tp, "FP": fp, "VN": tn, "FN": fn,
                "Precisão": p, "Revocação": r, "F1": f1, "F2": f2,
                "F1_IC_inf": ci(bf1)[0], "F1_IC_sup": ci(bf1)[1],
                "Prec_IC_inf": ci(bp)[0], "Prec_IC_sup": ci(bp)[1],
                "Rev_IC_inf": ci(br)[0], "Rev_IC_sup": ci(br)[1],
                "F2_IC_inf": ci(bf2)[0], "F2_IC_sup": ci(bf2)[1],
                "Taxa_pred_positiva": (tp + fp) / n,
                "TPR": tp / (tp + fn), "FPR": fp / (fp + tn),
                "Pares_com_erro_API": api_err[(m, s)],
            })
    met = pd.DataFrame(rows)
    met.to_csv(OUT / "metrics_full.csv", index=False, encoding="utf-8")
    print(met[["Modelo", "Estratégia", "VP", "FP", "VN", "FN", "Precisão", "Revocação", "F1", "F2",
               "F1_IC_inf", "F1_IC_sup", "Pares_com_erro_API"]].round(4).to_string())

    # ---------------- baselines IR
    req_ids, code_ids, mats = build_baseline_scores()
    r_i = {r: i for i, r in enumerate(req_ids)}
    c_i = {c: i for i, c in enumerate(code_ids)}
    gt_set = load_ground_truth()
    Y_full = np.zeros((len(req_ids), len(code_ids)), int)
    for r, c in gt_set:
        Y_full[r_i[r], c_i[c]] = 1
    assert Y_full.sum() == 286
    full_pairs = Y_full.size
    full_pos = int(Y_full.sum())

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    base_rows = []
    base_preds = {}
    for name, M in mats.items():
        s858 = np.array([M[r_i[r], c_i[t]] for r, t in zip(req_of, test_of)])
        # (a) limiar escolhido por validação cruzada (5 folds) -> sem vazamento
        yh = np.zeros(n, int)
        for tr, te in skf.split(s858.reshape(-1, 1), y):
            thr, _ = best_threshold(s858[tr], y[tr])
            yh[te] = (s858[te] >= thr).astype(int)
        base_preds[name] = yh
        tp, fp, tn, fn = counts(y, yh)
        p, r, f1, f2 = (float(v) for v in prf(tp, fp, fn))
        bp, br, bf1, bf2 = boot_metrics(y, yh, idx)
        boots[("Baseline", name)] = bf1
        # (b) limiar ótimo no próprio conjunto (limite superior otimista)
        thr_all, f1_or = best_threshold(s858, y)
        yh_or = (s858 >= thr_all).astype(int)
        tp_o, fp_o, tn_o, fn_o = counts(y, yh_or)
        po, ro, f1o, f2o = (float(v) for v in prf(tp_o, fp_o, fn_o))
        # (c) espaço completo (29.606 pares): limiar de (b) aplicado a todos os pares e ótimo no espaço completo
        flat_s, flat_y = M.ravel(), Y_full.ravel()
        yh_full = (flat_s >= thr_all).astype(int)
        tpf, fpf, tnf, fnf = counts(flat_y, yh_full)
        pf, rf, f1f, f2f = (float(v) for v in prf(tpf, fpf, fnf))
        thr_full, f1_full_or = best_threshold(flat_s, flat_y)
        ap_full = float(average_precision_score(flat_y, flat_s))
        base_rows.append({
            "Método": name,
            "CV_VP": tp, "CV_FP": fp, "CV_VN": tn, "CV_FN": fn,
            "CV_Precisão": p, "CV_Revocação": r, "CV_F1": f1, "CV_F2": f2,
            "CV_F1_IC_inf": ci(bf1)[0], "CV_F1_IC_sup": ci(bf1)[1],
            "Ótimo858_VP": tp_o, "Ótimo858_FP": fp_o, "Ótimo858_VN": tn_o, "Ótimo858_FN": fn_o,
            "Ótimo858_Precisão": po, "Ótimo858_Revocação": ro, "Ótimo858_F1": f1o,
            "Full_limiar858_VP": tpf, "Full_limiar858_FP": fpf, "Full_limiar858_Precisão": pf,
            "Full_limiar858_Revocação": rf, "Full_limiar858_F1": f1f,
            "Full_F1_ótimo": f1_full_or, "Full_AP": ap_full,
            "Full_pares": full_pairs, "Full_positivos": full_pos,
        })
    base = pd.DataFrame(base_rows)
    base.to_csv(OUT / "baselines.csv", index=False, encoding="utf-8")
    print("\n=== BASELINES ===")
    print(base.round(4).T.to_string())

    # ---------------- testes pareados (referência vs. demais)
    ref_pred = configs[REF]
    comps = []
    others = [(m, s) for m in MODEL_ORDER for s in STRAT_ORDER if (m, s) != REF]
    for key in others:
        comps.append((f"{key[0]} / {key[1]}", configs[key], boots[key]))
    for name, yh in base_preds.items():
        comps.append((f"Baseline: {name}", yh, boots[("Baseline", name)]))
    ref_correct = (ref_pred == y)
    ref_f1_boot = boots[REF]
    prow = []
    for label, yh, bf1 in comps:
        oc = (yh == y)
        b = int((ref_correct & ~oc).sum())   # referência acerta, outro erra
        c = int((~ref_correct & oc).sum())   # referência erra, outro acerta
        p_mc = mcnemar_exact(b, c)
        delta = ref_f1_boot - bf1
        tp, fp, tn, fn = counts(y, yh)
        f1_other = float(prf(tp, fp, fn)[2])
        f1_ref = float(prf(*[counts(y, ref_pred)[i] for i in (0, 1, 3)])[2])
        prow.append({
            "Comparação": f"{REF[0]} / {REF[1]}  vs  {label}",
            "F1_ref": f1_ref, "F1_outro": f1_other, "ΔF1": f1_ref - f1_other,
            "ΔF1_IC_inf": ci(delta)[0], "ΔF1_IC_sup": ci(delta)[1],
            "Prop_boot_ΔF1<=0": float((delta <= 0).mean()),
            "McNemar_b(ref acerta, outro erra)": b, "McNemar_c(ref erra, outro acerta)": c,
            "p_McNemar": p_mc,
        })
    pt = pd.DataFrame(prow)
    pt["p_Holm"] = holm(pt["p_McNemar"].to_numpy())
    pt["Significativo_5%"] = pt["p_Holm"] < 0.05
    pt.to_csv(OUT / "paired_tests.csv", index=False, encoding="utf-8")
    print("\n=== TESTES PAREADOS ===")
    print(pt.round(4).to_string())

    # ---------------- sensibilidade ao desbalanceamento (projeção a partir de TPR/FPR)
    neg_total = full_pairs - full_pos
    ratios = [2, 5, 10, 50, round(neg_total / full_pos, 1)]
    srows = []

    def project(tpr, fpr, ratio):
        tp = tpr * full_pos
        fp = fpr * ratio * full_pos
        fn = (1 - tpr) * full_pos
        p, r, f1, f2 = prf(tp, fp, fn)
        return float(p), float(r), float(f1), float(f2)

    for _, r in met.iterrows():
        for ratio in ratios:
            p, rc, f1, f2 = project(r["TPR"], r["FPR"], ratio)
            srows.append({"Modelo": r["Modelo"], "Estratégia": r["Estratégia"], "Proporção_pos:neg": f"1:{ratio:g}",
                          "Precisão": p, "Revocação": rc, "F1": f1, "F2": f2})
    for name, yh in base_preds.items():
        tp, fp, tn, fn = counts(y, yh)
        for ratio in ratios:
            p, rc, f1, f2 = project(tp / (tp + fn), fp / (fp + tn), ratio)
            srows.append({"Modelo": "Baseline", "Estratégia": name, "Proporção_pos:neg": f"1:{ratio:g}",
                          "Precisão": p, "Revocação": rc, "F1": f1, "F2": f2})
    sens = pd.DataFrame(srows)

    # IC por bootstrap estratificado (positivos e negativos reamostrados separadamente)
    def strat_ci(yh, ratio, B=N_BOOT):
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        rp = rng.choice(pos, size=(B, len(pos)))
        rn = rng.choice(neg, size=(B, len(neg)))
        tpr = yh[rp].mean(1)
        fpr = yh[rn].mean(1)
        tp = tpr * full_pos
        fp = fpr * ratio * full_pos
        fn = (1 - tpr) * full_pos
        _, _, f1, _ = prf(tp, fp, fn)
        p = np.where(tp + fp > 0, tp / (tp + fp), 0)
        return ci(p), ci(f1)

    ci_rows = []
    for key in [REF, ("GPT-4o-mini", "Chain-of-Thought".replace("Chain-of-Thought", "CoT")), ("GPT-4o-mini", "Zero-shot"),
                ("Claude Sonnet 4", "Few-shot"), ("Gemini Flash", "Few-shot")]:
        for ratio in ratios:
            (pl, ph), (fl, fh) = strat_ci(configs[key], ratio)
            ci_rows.append({"Modelo": key[0], "Estratégia": key[1], "Proporção_pos:neg": f"1:{ratio:g}",
                            "Prec_IC_inf": pl, "Prec_IC_sup": ph, "F1_IC_inf": fl, "F1_IC_sup": fh})
    pd.DataFrame(ci_rows).to_csv(OUT / "sensitivity_ci.csv", index=False, encoding="utf-8")
    sens.to_csv(OUT / "sensitivity.csv", index=False, encoding="utf-8")
    print("\n=== SENSIBILIDADE (F1 projetado) ===")
    print(sens.pivot_table(index=["Modelo", "Estratégia"], columns="Proporção_pos:neg", values="F1").round(3).to_string())
    print(pd.DataFrame(ci_rows).round(3).to_string())

    # ---------------- custo e tokens (estimativa de entrada = caracteres / 4)
    prompt_fns = {"Zero-shot": get_zero_shot_prompt, "Few-shot": get_few_shot_prompt, "CoT": get_cot_prompt}
    tok_rows = []
    for s, fn_ in prompt_fns.items():
        chars = np.array([len(fn_(rq, tx)) for rq, tx in zip(ref_df["req_text"], ref_df["test_text"])])
        tok_rows.append({"Estratégia": s, "Tokens_entrada_est_total_por_modelo": int(chars.sum() / 4),
                         "Tokens_entrada_est_médio_por_par": float(chars.mean() / 4),
                         "Tokens_entrada_est_mediana": float(np.median(chars) / 4),
                         "Tokens_entrada_est_p95": float(np.percentile(chars, 95) / 4)})
    tok = pd.DataFrame(tok_rows)
    tok.to_csv(OUT / "tokens_entrada_estimados.csv", index=False, encoding="utf-8")
    print("\n=== TOKENS DE ENTRADA ESTIMADOS ===")
    print(tok.round(1).to_string())

    in_tot = int(tok["Tokens_entrada_est_total_por_modelo"].sum())
    crow = []
    for m in MODEL_ORDER:
        sub = met[met["Modelo"] == m]
        best = sub.sort_values("F1", ascending=False).iloc[0]
        cost = BILLED_USD[m]
        crow.append({
            "Modelo": m,
            "Requisições_previstas": 3 * n,
            "Tokens_entrada_estimados": in_tot,
            "Custo_faturado_USD": cost,
            "Custo_médio_por_requisição_USD": cost / (3 * n),
            "Custo_por_1000_pares_USD": cost / (3 * n) * 1000,
            "Pares_com_erro_API": int(sub["Pares_com_erro_API"].sum()),
            "Melhor_estratégia": best["Estratégia"],
            "Melhor_F1": float(best["F1"]),
            "F1_por_USD": (float(best["F1"]) / cost) if cost > 0 else np.nan,
        })
    cost_df = pd.DataFrame(crow)
    cost_df.to_csv(OUT / "cost_table.csv", index=False, encoding="utf-8")
    print("\n=== CUSTO ===")
    print(cost_df.round(4).to_string())

    # ---------------- cobertura de positivos (difíceis) entre as 15 combinações
    pos_idx = np.where(y == 1)[0]
    found = np.array([configs[k][pos_idx] for k in configs])  # (15, 286)
    n_found = found.sum(0)
    print("\nPositivos nunca encontrados por nenhuma das 15 combinações:", int((n_found == 0).sum()))
    print("Positivos encontrados por pelo menos 1 combinação:", int((n_found > 0).sum()))
    print("Positivos encontrados por todas as 15:", int((n_found == 15).sum()))
    union_best = np.zeros(len(pos_idx), bool)
    for m in MODEL_ORDER:
        for s in STRAT_ORDER:
            union_best |= configs[(m, s)][pos_idx].astype(bool)
    print("Revocação da união (OR) das 15 combinações:", union_best.mean())


if __name__ == "__main__":
    main()
