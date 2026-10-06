"""
Reconstrói as predições por par (15 combinações modelo x estratégia) a partir dos logs
de execução e valida o resultado contra o CSV oficial de resultados.

Uso único (script de recuperação): depende dos logs locais das execuções originais.
Saída: data/processed/predictions_consolidated.csv
"""
import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
LOG_DIR = Path(r"C:\Users\Lucas Luis\.gemini\antigravity-ide\brain\ddf49f81-4058-486c-96e9-27320f91b9d8\.system_generated\tasks")
RESULTS = BASE / "data" / "processed" / "multimodel_prompt_results_full.csv"
OUT = BASE / "data" / "processed" / "predictions_consolidated.csv"

sys.path.append(str(BASE / "src"))
from data_loader import get_stratified_evaluation_dataset  # noqa: E402

re_header = re.compile(r"^\[([^\]]+)\]\[([^\]]+)\] Par #(\d+)/858: \[([^\]]+)\] <---> \[([^\]]+)\]")
re_gt = re.compile(r"^Ground Truth :\s+(\d)")
re_resp = re.compile(r"^Resposta IA\s*:\s*(.*)\[Predi..o:\s*(\d)\]$")
re_rerun_par = re.compile(r"^Avaliando par (\d+)/858")
re_rerun_pred = re.compile(r"^Resposta -> Predi..o:\s*(\d)")


def read_lines(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [ln.rstrip("\n").rstrip("\r").strip() for ln in f]


def parse_main_log(path):
    """Formato do pipeline principal (src/main.py)."""
    rec = {}
    cur = None
    for line in read_lines(path):
        m = re_header.search(line)
        if m:
            cur = {"model": m.group(1), "strategy": m.group(2), "pair_id": int(m.group(3)),
                   "req_id": m.group(4), "test_id": m.group(5), "gt": None, "err": False}
            continue
        if cur is None:
            continue
        g = re_gt.search(line)
        if g:
            cur["gt"] = int(g.group(1))
            continue
        if "[ERRO" in line and "API" in line:
            cur["err"] = True
        r = re_resp.search(line)
        if r and cur["gt"] is not None:
            preview = r.group(1).strip()
            err = cur["err"] or preview.startswith("ERRO")
            rec[(cur["model"], cur["strategy"], cur["pair_id"])] = {
                "model": cur["model"], "strategy": cur["strategy"], "pair_id": cur["pair_id"],
                "req_id": cur["req_id"], "test_id": cur["test_id"], "ground_truth": cur["gt"],
                "prediction": int(r.group(2)), "api_error": bool(err), "source": path.name,
            }
            cur = None
    return rec


def parse_rerun_log(path):
    """Formato dos scripts de reexecução (finish_*.py): 'Avaliando par n/858' + 'Resposta -> Predição: x'."""
    out = {}
    par, err = None, False
    for line in read_lines(path):
        m = re_rerun_par.search(line)
        if m:
            par, err = int(m.group(1)), False
            continue
        if "[ERRO" in line and "API" in line:
            err = True
        p = re_rerun_pred.search(line)
        if p and par is not None:
            out[par] = (int(p.group(1)), err)
            par = None
    return out


def main():
    logs = sorted(LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime)
    data = {}
    for lg in logs:
        data.update(parse_main_log(lg))

    df = pd.DataFrame(list(data.values()))
    print("Observações após log principal:", len(df))

    # Reexecuções documentadas nos scripts de recuperação
    pplx = ("perplexity/llama-3.1-sonar-large-128k-chat", "Chain-of-Thought (CoT)")
    gem = ("gemini-3.6-flash", "Chain-of-Thought (CoT)")

    ref = get_stratified_evaluation_dataset().reset_index(drop=True)

    rer418 = parse_rerun_log(LOG_DIR / "task-418.log")
    rer502 = parse_rerun_log(LOG_DIR / "task-502.log")
    print("task-418 pares:", len(rer418), "| task-502 pares:", len(rer502))

    def patch(df, key, rer, only_from=None):
        for pid, (pred, err) in rer.items():
            if only_from and pid < only_from:
                continue
            mask = (df.model == key[0]) & (df.strategy == key[1]) & (df.pair_id == pid)
            if mask.sum() == 0:
                row = ref.iloc[pid - 1]
                df.loc[len(df)] = {"model": key[0], "strategy": key[1], "pair_id": pid,
                                   "req_id": row["req_id"], "test_id": row["test_id"],
                                   "ground_truth": int(row["ground_truth"]), "prediction": pred,
                                   "api_error": err, "source": "rerun"}
            else:
                df.loc[mask, ["prediction", "api_error", "source"]] = [pred, err, "rerun"]
        return df

    df = patch(df, pplx, rer418, only_from=728)
    df = patch(df, gem, rer502)

    # Completa req_id/test_id e confere ground truth contra a base oficial (semente 42)
    ref["pair_id"] = ref.index + 1
    df = df.merge(ref[["pair_id", "req_id", "test_id", "ground_truth"]], on="pair_id", suffixes=("", "_ref"))
    assert (df.req_id == df.req_id_ref).all() and (df.test_id == df.test_id_ref).all(), "pares fora de ordem"
    assert (df.ground_truth == df.ground_truth_ref).all(), "ground truth divergente"
    df = df.drop(columns=["req_id_ref", "test_id_ref", "ground_truth_ref"])

    # Validação contra CSV oficial
    res = pd.read_csv(RESULTS)
    ok_all = True
    for _, r in res.iterrows():
        d = df[(df.model == r["model"]) & (df.strategy == r["strategy"])]
        tp = int(((d.ground_truth == 1) & (d.prediction == 1)).sum())
        fp = int(((d.ground_truth == 0) & (d.prediction == 1)).sum())
        tn = int(((d.ground_truth == 0) & (d.prediction == 0)).sum())
        fn = int(((d.ground_truth == 1) & (d.prediction == 0)).sum())
        ok = (len(d) == 858 and (tp, fp, tn, fn) == (r.tp, r.fp, r.tn, r.fn))
        ok_all &= ok
        print(f"{r['model'][:28]:28s} {r['strategy'][:10]:10s} n={len(d)} "
              f"obs=({tp},{fp},{tn},{fn}) csv=({r.tp},{r.fp},{r.tn},{r.fn}) {'OK' if ok else 'DIVERGE'} "
              f"erros_api={int(d.api_error.sum())}")
    print("TODAS AS 15 COMBINAÇÕES CONFEREM COM O CSV OFICIAL:", ok_all)
    df = df.sort_values(["model", "strategy", "pair_id"]).reset_index(drop=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print("Salvo:", OUT, df.shape)


if __name__ == "__main__":
    main()
