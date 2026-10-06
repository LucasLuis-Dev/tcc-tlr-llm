import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Adiciona o diretório 'src' ao sys.path
sys.path.append(str(Path(__file__).resolve().parent))

# Diretórios de entrada e saída
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "processed" / "multimodel_prompt_results_full.csv"
OUTPUT_DIR = BASE_DIR / "results" / "figures"

# Mapeamento de nomes oficiais dos modelos para rótulos legíveis
MODEL_LABEL_MAP = {
    "anthropic/claude-3.5-sonnet": "Claude Sonnet 4",
    "openai/gpt-4o-mini": "GPT-4o-mini",
    "deepseek/deepseek-chat": "DeepSeek V3",
    "perplexity/llama-3.1-sonar-large-128k-chat": "Perplexity Sonar",
    "gemini-3.6-flash": "Gemini Flash"
}

# Ordem padronizada de estratégias
STRATEGY_ORDER = ["Zero-Shot", "Few-Shot", "Chain-of-Thought (CoT)"]


def parse_metric_value(val) -> float:
    """
    Converte valores de métrica para float na escala de 0 a 100%.
    Trata:
    - Strings com símbolo de % e vírgula (ex: '85,71%' -> 85.71)
    - Floats na escala de 0.0 a 1.0 (ex: 0.8571 -> 85.71)
    - Floats já na escala de 0 a 100.
    """
    if pd.isna(val):
        return 0.0
    if isinstance(val, str):
        val_clean = val.replace("%", "").replace(",", ".").strip()
        try:
            num = float(val_clean)
        except ValueError:
            return 0.0
        return num if num > 1.0 else num * 100.0
    elif isinstance(val, (int, float)):
        num = float(val)
        return num if num > 1.0 else num * 100.0
    return 0.0


def load_and_clean_data(csv_path: Path) -> pd.DataFrame:
    """
    Carrega o arquivo CSV de resultados e aplica a limpeza e normalização das métricas.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo de resultados não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)

    # Aplica normalização das colunas de métricas
    df["precision_pct"] = df["precision"].apply(parse_metric_value)
    df["recall_pct"] = df["recall"].apply(parse_metric_value)
    df["f1_pct"] = df["f1"].apply(parse_metric_value)

    # Adiciona rótulos limpos dos modelos
    df["model_clean"] = df["model"].map(lambda m: MODEL_LABEL_MAP.get(m, m))

    return df


def plot_f1_score_comparative(df: pd.DataFrame, output_dir: Path):
    """
    Gráfico 1: Comparativo de F1-Score por Modelo e Estratégia de Prompt (Grouped Barplot).
    Evidencia o salto de desempenho do GPT-4o-mini com Chain-of-Thought (CoT).
    """
    plt.figure(figsize=(12, 6.5))
    sns.set_theme(style="whitegrid")

    palette = {
        "Zero-Shot": "#4C72B0",
        "Few-Shot": "#55A868",
        "Chain-of-Thought (CoT)": "#C44E52"
    }

    ax = sns.barplot(
        data=df,
        x="model_clean",
        y="f1_pct",
        hue="strategy",
        hue_order=STRATEGY_ORDER,
        palette=palette,
        edgecolor="black",
        linewidth=0.8
    )

    # Rótulos de valor no topo de cada barra
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.1f}%",
                (p.get_x() + p.get_width() / 2.0, height + 1.2),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold"
            )

    plt.title("Comparativo de F1-Score por Modelo de IA e Estratégia de Prompt (TLR)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Modelo de Inteligência Artificial", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("F1-Score (%)", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylim(0, 105)

    plt.legend(title="Estratégia de Prompt", title_fontsize="11", fontsize="10", loc="upper right", frameon=True)
    plt.tight_layout()

    out_file = output_dir / "f1_score_comparative.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Gráfico 1 salvo]: {out_file}")


def plot_precision_vs_recall(df: pd.DataFrame, output_dir: Path):
    """
    Gráfico 2: Trade-off entre Precisão e Revocação (Scatterplot).
    Mostra o agrupamento de alta precisão/baixa revocação vs. o ponto de equilíbrio do GPT-4o-mini (CoT).
    """
    plt.figure(figsize=(11, 7))
    sns.set_theme(style="whitegrid")

    # Filtra modelos com valores de predição válidos para destaque
    df_valid = df.copy()

    model_colors = {
        "Claude Sonnet 4": "#1f77b4",
        "GPT-4o-mini": "#2ca02c",
        "DeepSeek V3": "#ff7f0e",
        "Perplexity Sonar": "#9467bd",
        "Gemini Flash": "#d62728"
    }

    markers = {
        "Zero-Shot": "o",
        "Few-Shot": "s",
        "Chain-of-Thought (CoT)": "^"
    }

    ax = sns.scatterplot(
        data=df_valid,
        x="precision_pct",
        y="recall_pct",
        hue="model_clean",
        style="strategy",
        style_order=STRATEGY_ORDER,
        palette=model_colors,
        markers=markers,
        s=180,
        edgecolor="black",
        linewidth=1.2,
        alpha=0.9
    )

    # Anotações para pontos-chave de interesse científico
    annotations = [
        ("GPT-4o-mini (Few-shot)", "GPT-4o-mini", "Few-Shot", (-65, -25)),
        ("Perplexity (Few-Shot)", "Perplexity Sonar", "Few-Shot", (-75, 10)),
        ("Claude Sonnet 4 (Few-Shot)", "Claude Sonnet 4", "Few-Shot", (-110, -25)),
    ]

    for label, model_name, strategy_name, offset in annotations:
        row = df_valid[(df_valid["model_clean"] == model_name) & (df_valid["strategy"] == strategy_name)]
        if not row.empty:
            x_val = row["precision_pct"].values[0]
            y_val = row["recall_pct"].values[0]
            ax.annotate(
                label,
                xy=(x_val, y_val),
                xytext=(x_val + offset[0], y_val + offset[1]),
                textcoords="data",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color="black", lw=1.2),
                fontsize=9.5,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85)
            )

    plt.title("Trade-off entre Precisão e Revocação por Modelo e Estratégia de Prompt", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Precisão - Precision (%)", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("Revocação - Recall (%)", fontsize=12, fontweight="bold", labelpad=10)
    plt.xlim(-5, 108)
    plt.ylim(-5, 105)

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, title="Modelo & Estratégia", frameon=True)
    plt.tight_layout()

    out_file = output_dir / "precision_vs_recall.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Gráfico 2 salvo]: {out_file}")


def main():
    print("=" * 70)
    print("  GERANDO GRÁFICOS DE VISUALIZAÇÃO EXPERIMENTAL (TLR)")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Carrega e limpa os dados
    df = load_and_clean_data(CSV_PATH)

    print("\n[Dados carregados e normalizados]:")
    print(df[["model_clean", "strategy", "precision_pct", "recall_pct", "f1_pct"]].to_string())
    print("-" * 70)

    # Gera os dois gráficos solicitados
    plot_f1_score_comparative(df, OUTPUT_DIR)
    plot_precision_vs_recall(df, OUTPUT_DIR)

    print("=" * 70)
    print("Geração de gráficos finalizada com sucesso!")
    print(f"Imagens salvas no diretório: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
