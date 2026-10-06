"""
Análise qualitativa/quantitativa dos erros (falsos positivos e falsos negativos).

Saídas em results/analysis/:
  - fp_reference.csv / fn_reference.csv : listas de FP e FN da melhor configuração LLM (GPT-4o-mini, Few-shot)
  - fp_all_configs.csv                  : FPs de todas as combinações (agregados por par)
  - fn_factors.csv                      : fatores associados aos falsos negativos (multirrótulo)
  - model_patterns.csv                  : padrões de erro por modelo
  - class_type_recall.csv               : revocação por tipo de classe Java
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.append(str(Path(__file__).resolve().parent))

from analysis import (  # noqa: E402
    MODEL_LABEL, STRAT_LABEL, MODEL_ORDER, STRAT_ORDER, REF, tokenize, build_baseline_scores,
)
from data_loader import load_requirements, load_test_cases, load_ground_truth  # noqa: E402

OUT = BASE / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)


def class_type(name):
    base = name.replace(".java", "")
    for suf in ["Action", "DAO", "BeanLoader", "BeanValidator", "Bean", "Loader", "Validator", "Exception",
                "Factor", "Risks", "Test"]:
        if base.endswith(suf):
            return {"BeanLoader": "Loader", "BeanValidator": "Validator", "Factor": "Fator de risco",
                    "Risks": "Fator de risco"}.get(suf, suf)
    return "Outras"


def main():
    pred = pd.read_csv(BASE / "data" / "processed" / "predictions_consolidated.csv")
    pred["Modelo"] = pred["model"].map(MODEL_LABEL)
    pred["Estratégia"] = pred["strategy"].map(STRAT_LABEL)

    reqs = load_requirements()
    codes = load_test_cases()
    gt = load_ground_truth()
    links_per_req = pd.Series([r for r, _ in gt]).value_counts()
    links_per_cls = pd.Series([c for _, c in gt]).value_counts()

    req_ids, code_ids, mats = build_baseline_scores()
    r_i = {r: i for i, r in enumerate(req_ids)}
    c_i = {c: i for i, c in enumerate(code_ids)}
    tfidf = mats["TF-IDF (cosseno)"]

    d = pred[(pred["Modelo"] == REF[0]) & (pred["Estratégia"] == REF[1])].copy()
    d["tfidf"] = [tfidf[r_i[r], c_i[t]] for r, t in zip(d.req_id, d.test_id)]
    d["tipo_classe"] = d.test_id.map(class_type)
    d["req_chars"] = d.req_id.map(lambda r: len(reqs[r]))
    d["code_chars"] = d.test_id.map(lambda c: len(codes[c]))
    d["links_do_req"] = d.req_id.map(lambda r: int(links_per_req.get(r, 0)))
    d["links_da_classe"] = d.test_id.map(lambda c: int(links_per_cls.get(c, 0)))
    d["jaccard"] = [
        len(set(tokenize(reqs[r])) & set(tokenize(codes[c]))) / max(1, len(set(tokenize(reqs[r])) | set(tokenize(codes[c]))))
        for r, c in zip(d.req_id, d.test_id)
    ]

    fp = d[(d.ground_truth == 0) & (d.prediction == 1)]
    fn = d[(d.ground_truth == 1) & (d.prediction == 0)]
    tp = d[(d.ground_truth == 1) & (d.prediction == 1)]
    tn = d[(d.ground_truth == 0) & (d.prediction == 0)]
    cols = ["pair_id", "req_id", "test_id", "tipo_classe", "tfidf", "jaccard", "req_chars", "code_chars",
            "links_do_req", "links_da_classe"]
    fp[cols].to_csv(OUT / "fp_reference.csv", index=False, encoding="utf-8")
    fn[cols].to_csv(OUT / "fn_reference.csv", index=False, encoding="utf-8")

    print("=== Configuração de referência:", REF, "===")
    print("FP:", len(fp), "FN:", len(fn), "TP:", len(tp), "TN:", len(tn))
    print("\nFP (ref):")
    print(fp[cols].round(3).to_string())

    print("\n--- Médias (TP vs FN, ref) ---")
    for nm, g in [("TP", tp), ("FN", fn), ("FP", fp), ("TN", tn)]:
        print(nm, "tfidf=%.3f jaccard=%.3f code_chars(med)=%d req_chars(med)=%d links_req(med)=%.1f links_cls(med)=%.1f" % (
            g.tfidf.mean(), g.jaccard.mean(), g.code_chars.median(), g.req_chars.median(),
            g.links_do_req.median(), g.links_da_classe.median()))

    # revocação por tipo de classe (apenas positivos)
    pos = d[d.ground_truth == 1]
    ct = pos.groupby("tipo_classe").agg(positivos=("prediction", "size"), encontrados=("prediction", "sum"))
    ct["revocação"] = ct["encontrados"] / ct["positivos"]
    ct = ct.sort_values("positivos", ascending=False)
    ct.to_csv(OUT / "class_type_recall.csv", encoding="utf-8")
    print("\n--- Revocação por tipo de classe (ref) ---")
    print(ct.round(3).to_string())

    # revocação por tamanho do código (quartis)
    pos = pos.copy()
    pos["q_code"] = pd.qcut(pos.code_chars, 4, labels=["Q1 (menor)", "Q2", "Q3", "Q4 (maior)"])
    print("\n--- Revocação por quartil de tamanho da classe (ref) ---")
    print(pos.groupby("q_code", observed=True).agg(n=("prediction", "size"), rev=("prediction", "mean"),
                                                   med_chars=("code_chars", "median")).round(3).to_string())
    pos["q_tfidf"] = pd.qcut(pos.tfidf, 4, labels=["Q1 (menor sobreposição)", "Q2", "Q3", "Q4 (maior)"])
    print("\n--- Revocação por quartil de similaridade lexical (TF-IDF) (ref) ---")
    print(pos.groupby("q_tfidf", observed=True).agg(n=("prediction", "size"), rev=("prediction", "mean"),
                                                    med_tfidf=("tfidf", "median")).round(3).to_string())
    pos["req_links_grp"] = pd.cut(pos.links_do_req, [0, 1, 2, 4, 100], labels=["1", "2", "3-4", "5+"])
    print("\n--- Revocação por nº de classes ligadas ao requisito (ref) ---")
    print(pos.groupby("req_links_grp", observed=True).agg(n=("prediction", "size"), rev=("prediction", "mean")).round(3).to_string())
    pos["req_suffix"] = pos.req_id.str.extract(r"UC\d+([ES]?)\d*")[0].replace({"": "UC (fluxo principal)", "S": "S", "E": "E"})
    print("\n--- Revocação por tipo de artefato de requisito (ref) ---")
    print(pos.groupby("req_suffix").agg(n=("prediction", "size"), rev=("prediction", "mean")).round(3).to_string())

    # fatores associados a FN (multirrótulo)
    q_low = pos.tfidf.quantile(0.25)
    big = pos.code_chars.quantile(0.75)
    support_types = {"Bean", "Loader", "Validator", "Exception", "Fator de risco", "Outras"}
    fn = fn.copy()
    fn["f_sem_vocab"] = fn.tfidf <= q_low
    fn["f_codigo_extenso"] = fn.code_chars >= big
    fn["f_distribuida"] = fn.links_do_req >= 3
    fn["f_apoio"] = fn.tipo_classe.isin(support_types)
    fn["f_classe_compartilhada"] = fn.links_da_classe >= 5
    fac = pd.DataFrame({
        "fator": ["Baixa similaridade lexical (1º quartil de TF-IDF)", "Código extenso (4º quartil de tamanho)",
                  "Implementação distribuída (requisito com 3+ classes)", "Classe de apoio (Bean/Loader/Validator/Exception/outras)",
                  "Classe compartilhada (ligada a 5+ requisitos)"],
        "n_FN": [int(fn.f_sem_vocab.sum()), int(fn.f_codigo_extenso.sum()), int(fn.f_distribuida.sum()),
                 int(fn.f_apoio.sum()), int(fn.f_classe_compartilhada.sum())],
    })
    fac["% dos FN"] = fac.n_FN / len(fn)
    fac.to_csv(OUT / "fn_factors.csv", index=False, encoding="utf-8")
    print("\n--- Fatores associados aos FN (ref, n=%d) ---" % len(fn))
    print(fac.round(3).to_string())
    print("FN com nenhum dos fatores:", int((~(fn.f_sem_vocab | fn.f_codigo_extenso | fn.f_distribuida | fn.f_apoio | fn.f_classe_compartilhada)).sum()))
    print("q_low tfidf =", round(q_low, 4), "| big code chars =", int(big))

    # FPs por todas as combinações
    allfp = pred[(pred.ground_truth == 0) & (pred.prediction == 1)].copy()
    agg = allfp.groupby(["req_id", "test_id"]).agg(n_configs=("Modelo", "size"),
                                                   configs=("Modelo", lambda s: ", ".join(sorted(set(s))))).reset_index()
    agg = agg.sort_values("n_configs", ascending=False)
    agg.to_csv(OUT / "fp_all_configs.csv", index=False, encoding="utf-8")
    print("\n--- FPs mais recorrentes (de 15 combinações) ---")
    print(agg.head(12).to_string())
    print("Total de pares negativos que viraram FP em alguma combinação:", len(agg))
    print("FPs por tipo de classe (todas combinações):")
    print(allfp.test_id.map(class_type).value_counts().to_string())

    # padrões por modelo
    rows = []
    for m in MODEL_ORDER:
        sub = pred[pred.Modelo == m]
        for s in STRAT_ORDER:
            x = sub[sub["Estratégia"] == s]
            n_pos_pred = int(x.prediction.sum())
            rows.append({"Modelo": m, "Estratégia": s, "Pares_preditos_SIM": n_pos_pred,
                         "Taxa_SIM": n_pos_pred / len(x),
                         "FP": int(((x.ground_truth == 0) & (x.prediction == 1)).sum()),
                         "FN": int(((x.ground_truth == 1) & (x.prediction == 0)).sum()),
                         "Erros_API": int(x.api_error.sum())})
    mp = pd.DataFrame(rows)
    mp.to_csv(OUT / "model_patterns.csv", index=False, encoding="utf-8")
    print("\n--- Padrões por modelo ---")
    print(mp.round(3).to_string())

    # classes que mais concentram FN na referência
    print("\n--- Classes com mais FN (ref) ---")
    print(fn.test_id.value_counts().head(8).to_string())
    hubs4 = ["AuthDAO.java", "PatientDAO.java", "GetUserNameAction.java", "EmailUtil.java"]
    print("Elos positivos nas 4 classes mais ligadas:", int(pos.test_id.isin(hubs4).sum()),
          "| FN nelas:", int(fn.test_id.isin(hubs4).sum()))

    # revocação por segmento em cada combinação (classes "hub" e tipo de requisito)
    seg_rows = []
    hub = set(links_per_cls[links_per_cls >= 5].index)
    allp = pred[pred.ground_truth == 1].copy()
    allp["hub"] = allp.test_id.isin(hub)
    allp["req_tipo"] = allp.req_id.str.extract(r"UC\d+([ES]?)\d*")[0].replace({"": "UC"})
    for (m, s), x in allp.groupby(["Modelo", "Estratégia"]):
        seg_rows.append({
            "Modelo": m, "Estratégia": s,
            "Rev_classes_hub(5+ reqs)": x[x.hub].prediction.mean(), "n_hub": int(x.hub.sum()),
            "Rev_classes_não_hub": x[~x.hub].prediction.mean(), "n_não_hub": int((~x.hub).sum()),
            "Rev_req_E": x[x.req_tipo == "E"].prediction.mean(), "n_E": int((x.req_tipo == "E").sum()),
            "Rev_req_S": x[x.req_tipo == "S"].prediction.mean(), "n_S": int((x.req_tipo == "S").sum()),
        })
    seg = pd.DataFrame(seg_rows)
    seg["Modelo"] = pd.Categorical(seg["Modelo"], MODEL_ORDER, ordered=True)
    seg["Estratégia"] = pd.Categorical(seg["Estratégia"], STRAT_ORDER, ordered=True)
    seg = seg.sort_values(["Modelo", "Estratégia"])
    seg.to_csv(OUT / "model_recall_segments.csv", index=False, encoding="utf-8")
    print("\n--- Revocação por segmento (todas as combinações) ---")
    print(seg.round(3).to_string())

    # sensibilidade ao Few-shot (ganho de F1 e revocação ZS -> FS)
    print("\n--- Ganho Zero-shot -> Few-shot (F1 e Revocação) ---")
    met = pd.read_csv(OUT / "metrics_full.csv")
    piv = met.pivot(index="Modelo", columns="Estratégia", values="F1")
    pivr = met.pivot(index="Modelo", columns="Estratégia", values="Revocação")
    g = pd.DataFrame({"ΔF1 FS-ZS": piv["Few-shot"] - piv["Zero-shot"], "ΔF1 CoT-ZS": piv["CoT"] - piv["Zero-shot"],
                      "ΔRev FS-ZS": pivr["Few-shot"] - pivr["Zero-shot"]})
    print(g.round(3).to_string())

    # pares FN e FP exemplos: imprime textos curtos para inspeção manual
    def show(rid, cid, n=600):
        print("\n[REQ %s]\n%s\n[CLASSE %s] (%d chars)\n%s\n" % (rid, reqs[rid][:n], cid, len(codes[cid]), codes[cid][:n]))

    print("\n=========== INSPEÇÃO MANUAL: FPs da referência ===========")
    for _, r in fp.iterrows():
        show(r.req_id, r.test_id, 500)


if __name__ == "__main__":
    main()
