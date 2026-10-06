import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from data_loader import load_requirements, load_test_cases, load_ground_truth

reqs = load_requirements(); codes = load_test_cases(); gt = load_ground_truth()
rl = pd.Series({k: len(v) for k, v in reqs.items()}); cl = pd.Series({k: len(v) for k, v in codes.items()})
print("REQ chars: mean %.0f median %.0f min %d max %d" % (rl.mean(), rl.median(), rl.min(), rl.max()))
print("CODE chars: mean %.0f median %.0f min %d max %d" % (cl.mean(), cl.median(), cl.min(), cl.max()))
print("lines code median", pd.Series({k: v.count("\n")+1 for k, v in codes.items()}).median())
lpr = pd.Series([r for r, _ in gt]).value_counts(); lpc = pd.Series([c for _, c in gt]).value_counts()
print("reqs com >=1 elo:", len(lpr), "de", len(reqs), "| sem elo:", len(reqs) - len(lpr))
print("classes com >=1 elo:", len(lpc), "de", len(codes), "| sem elo:", len(codes) - len(lpc))
print("elos por req (entre os que têm): média %.2f mediana %.1f máx %d" % (lpr.mean(), lpr.median(), lpr.max()))
print("elos por classe (entre as que têm): média %.2f mediana %.1f máx %d" % (lpc.mean(), lpc.median(), lpc.max()))
print("top classes:", lpc.head(6).to_dict())
ids = pd.Series(list(reqs))
import re
t = ids.map(lambda s: "E" if re.match(r"UC\d+E", s) else ("S" if re.match(r"UC\d+S", s) else "UC"))
print("tipos de req:", t.value_counts().to_dict())
print("\n--- exemplo UC10E1 ---"); print(reqs["UC10E1.txt"]); print([c for r, c in gt if r == "UC10E1.txt"])
print("\n--- GetUserNameAction.java ---"); print(codes["GetUserNameAction.java"])
print("\n--- FN exemplos ---")
pr = pd.read_csv(Path(__file__).resolve().parent.parent / "results" / "analysis" / "fn_reference.csv")
print(pr.sort_values("tfidf").head(6)[["req_id", "test_id", "tfidf", "code_chars", "links_do_req", "links_da_classe"]].to_string())
print(pr.sort_values("links_da_classe", ascending=False).head(6)[["req_id", "test_id", "tfidf", "code_chars", "links_do_req", "links_da_classe"]].to_string())
print(pr[pr.req_id.str.contains("E")].head(6)[["req_id", "test_id", "tfidf", "code_chars", "links_do_req", "links_da_classe"]].to_string())
for rid in ["UC22E1.txt", "UC19S1.txt"]:
    if rid in reqs:
        print("\n", rid, reqs[rid][:500], [c for r, c in gt if r == rid])
