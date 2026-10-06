"""
Gráficos complementares para o capítulo de Resultados (lê results/analysis/*.csv).
Saída: results/figures/*.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE = Path(__file__).resolve().parent.parent
A = BASE / "results" / "analysis"
FIG = BASE / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MODELS = ["Claude Sonnet 4", "GPT-4o-mini", "DeepSeek V3", "Perplexity Sonar", "Gemini Flash"]
STRATS = ["Zero-shot", "Few-shot", "CoT"]
COL = {"Zero-shot": "#4C72B0", "Few-shot": "#55A868", "CoT": "#C44E52"}
sns.set_theme(style="whitegrid")


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=300, bbox_inches="tight")
    plt.close()
    print("salvo:", FIG / name)


def main():
    met = pd.read_csv(A / "metrics_full.csv")
    base = pd.read_csv(A / "baselines.csv")
    sens = pd.read_csv(A / "sensitivity.csv")
    cost = pd.read_csv(A / "cost_table.csv")

    # 1) F1 com IC 95% e baselines
    fig, ax = plt.subplots(figsize=(12, 6.3))
    w = 0.26
    for j, s in enumerate(STRATS):
        sub = met[met["Estratégia"] == s].set_index("Modelo").loc[MODELS]
        x = np.arange(len(MODELS)) + (j - 1) * w
        f1 = sub["F1"].to_numpy() * 100
        err = np.vstack([f1 - sub["F1_IC_inf"].to_numpy() * 100, sub["F1_IC_sup"].to_numpy() * 100 - f1])
        ax.bar(x, f1, w, yerr=err, capsize=3, color=COL[s], edgecolor="black", linewidth=0.7, label=s)
        for xi, v in zip(x, f1):
            ax.text(xi, 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")
    styles = {"TF-IDF (cosseno)": (":", "#555555"), "BM25": ("--", "#8c564b"), "LSI (SVD, k=100)": ("-.", "#17becf")}
    for _, r in base.iterrows():
        ls, c = styles[r["Método"]]
        ax.axhline(r["CV_F1"] * 100, ls=ls, color=c, lw=1.6, label=f"Baseline {r['Método']}: {r['CV_F1']*100:.1f}%")
    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 85)
    ax.set_ylabel("F1-score (%)")
    ax.set_xlabel("Modelo")
    ax.legend(loc="upper right", fontsize=8.5, ncol=2, frameon=True)
    save("f1_ic95_baselines.png")

    # 2) Heatmap F1
    piv = met.pivot(index="Modelo", columns="Estratégia", values="F1").loc[MODELS, STRATS] * 100
    err = met.pivot(index="Modelo", columns="Estratégia", values="Pares_com_erro_API").loc[MODELS, STRATS]
    annot = piv.round(1).astype(str)
    for m in MODELS:
        for s in STRATS:
            if err.loc[m, s] > 0:
                annot.loc[m, s] += "*"
    fig, ax = plt.subplots(figsize=(7, 4.6))
    sns.heatmap(piv, annot=annot, fmt="", cmap="YlGnBu", vmin=0, vmax=75, cbar_kws={"label": "F1-score (%)"}, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_xlabel("Estratégia de prompt")
    ax.set_ylabel("")
    save("heatmap_f1.png")

    # 3) Custo x desempenho (melhor F1 de cada modelo)
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for _, r in cost.iterrows():
        x = r["Custo_faturado_USD"]
        y = r["Melhor_F1"] * 100
        ax.scatter(x, y, s=170, edgecolor="black", zorder=3)
        ax.annotate(f"{r['Modelo']}\n({r['Melhor_estratégia']})", (x, y), textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xscale("symlog", linthresh=0.5)
    ax.set_xlim(-0.05, 40)
    ax.set_xticks([0, 0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0\n(Free Tier)", "0,5", "1", "2", "5", "10", "20"])
    ax.set_ylim(15, 80)
    ax.set_xlabel("Custo total faturado no OpenRouter (US$, escala simétrica logarítmica)")
    ax.set_ylabel("Melhor F1-score do modelo (%)")
    save("custo_vs_f1.png")

    # 4) Matriz de confusão da melhor configuração
    best = met[(met["Modelo"] == "GPT-4o-mini") & (met["Estratégia"] == "Few-shot")].iloc[0]
    cm = np.array([[best["VN"], best["FP"]], [best["FN"], best["VP"]]], int)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Previsto: NÃO", "Previsto: SIM"], yticklabels=["Real: sem elo", "Real: com elo"],
                annot_kws={"fontsize": 15, "fontweight": "bold"})
    save("matriz_confusao_melhor.png")

    # 5) Distribuição dos tipos de erro
    fp_cat = pd.Series({"Vocabulário semelhante,\ncomportamento diferente": 4,
                        "Código de apoio\n(validador/carregador)": 3,
                        "Requisito transversal\n(registro de transações)": 2,
                        "Classe compartilhada\nde acesso a dados": 2,
                        "Funcionalidade vizinha /\nrelação indireta": 2}).sort_values()
    fnf = pd.read_csv(A / "fn_factors.csv").sort_values("% dos FN")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
    axes[0].barh(fp_cat.index, fp_cat.values, color="#C44E52", edgecolor="black")
    for i, v in enumerate(fp_cat.values):
        axes[0].text(v + 0.05, i, str(v), va="center", fontweight="bold")
    axes[0].set_xlabel("Nº de falsos positivos (total: 13)")
    axes[0].set_title("Falsos positivos (GPT-4o-mini, Few-shot)", fontsize=11)
    short = {
        "Classe compartilhada (ligada a 5+ requisitos)": "Classe ligada a 5+ requisitos",
        "Implementação distribuída (requisito com 3+ classes)": "Requisito ligado a 3+ classes",
        "Baixa similaridade lexical (1º quartil de TF-IDF)": "Baixa similaridade lexical\n(1º quartil)",
        "Código extenso (4º quartil de tamanho)": "Classe extensa\n(4º quartil de tamanho)",
        "Classe de apoio (Bean/Loader/Validator/Exception/outras)": "Classe de apoio\n(Validator/Exception/outras)",
    }
    axes[1].barh([short[f] for f in fnf["fator"]], fnf["% dos FN"] * 100, color="#4C72B0", edgecolor="black")
    for i, v in enumerate(fnf["% dos FN"] * 100):
        axes[1].text(v + 1, i, f"{v:.1f}%", va="center", fontweight="bold")
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("% dos 130 falsos negativos (fatores não excludentes)")
    axes[1].set_title("Falsos negativos (GPT-4o-mini, Few-shot)", fontsize=11)
    save("distribuicao_erros.png")

    # 6) Sensibilidade à proporção positivos:negativos
    ratio_order = ["1:2", "1:5", "1:10", "1:50", "1:102.5"]
    xs = [2, 5, 10, 50, 102.5]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    series = [("GPT-4o-mini", "Few-shot", "#2ca02c", "-", "o"), ("GPT-4o-mini", "CoT", "#98df8a", "-", "o"),
              ("Claude Sonnet 4", "Few-shot", "#1f77b4", "-", "s"), ("Gemini Flash", "Few-shot", "#d62728", "-", "^"),
              ("DeepSeek V3", "CoT", "#ff7f0e", "-", "D")]
    for m, s, c, ls, mk in series:
        d = sens[(sens["Modelo"] == m) & (sens["Estratégia"] == s)].set_index("Proporção_pos:neg").loc[ratio_order]
        ax.plot(xs, d["F1"] * 100, color=c, ls=ls, marker=mk, lw=2, label=f"{m} ({s})")
    for meth, c, ls in [("TF-IDF (cosseno)", "#555555", ":"), ("BM25", "#8c564b", "--"), ("LSI (SVD, k=100)", "#17becf", "-.")]:
        d = sens[(sens["Modelo"] == "Baseline") & (sens["Estratégia"] == meth)].set_index("Proporção_pos:neg").loc[ratio_order]
        ax.plot(xs, d["F1"] * 100, color=c, ls=ls, marker="x", lw=1.8, label=f"Baseline {meth}")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(["1:2\n(experimento)", "1:5", "1:10", "1:50", "1:102,5\n(iTrust completo)"])
    ax.set_xlabel("Proporção entre pares com elo e sem elo (pares negativos por positivo)")
    ax.set_ylabel("F1-score projetado (%)")
    ax.set_ylim(0, 75)
    ax.legend(fontsize=8.5, loc="upper right")
    save("sensibilidade_desbalanceamento.png")

    # 7) LLM vs baselines: P, R, F1, F2
    items = [("GPT-4o-mini\n(Few-shot)", met[(met.Modelo == "GPT-4o-mini") & (met["Estratégia"] == "Few-shot")].iloc[0], "llm"),
             ("Claude Sonnet 4\n(Few-shot)", met[(met.Modelo == "Claude Sonnet 4") & (met["Estratégia"] == "Few-shot")].iloc[0], "llm"),
             ("Gemini Flash\n(Few-shot)", met[(met.Modelo == "Gemini Flash") & (met["Estratégia"] == "Few-shot")].iloc[0], "llm")]
    for _, r in base.iterrows():
        items.append((r["Método"].replace(" (", "\n("), pd.Series({"Precisão": r["CV_Precisão"], "Revocação": r["CV_Revocação"],
                                                                "F1": r["CV_F1"], "F2": r["CV_F2"]}), "ir"))
    fig, ax = plt.subplots(figsize=(12, 5.4))
    mets = ["Precisão", "Revocação", "F1", "F2"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#8172B3"]
    w = 0.2
    for j, mt in enumerate(mets):
        vals = [it[1][mt] * 100 for it in items]
        xs_ = np.arange(len(items)) + (j - 1.5) * w
        ax.bar(xs_, vals, w, color=colors[j], edgecolor="black", linewidth=0.6, label=mt)
        for xi, v in zip(xs_, vals):
            ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(np.arange(len(items)))
    ax.set_xticklabels([it[0] for it in items], fontsize=9)
    ax.axvline(2.5, color="gray", ls="--")
    ax.text(1, 106, "LLMs", ha="center", fontweight="bold")
    ax.text(4, 106, "Baselines clássicos (limiar por validação cruzada)", ha="center", fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_ylabel("%")
    ax.legend(loc="lower right", ncol=4, fontsize=9)
    save("comparativo_baselines.png")


if __name__ == "__main__":
    main()
