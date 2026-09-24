import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd

# Adiciona o diretório 'src' ao sys.path para importações diretas
sys.path.append(str(Path(__file__).resolve().parent))

# Garante suporte adequado a UTF-8 no terminal Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from data_loader import get_stratified_evaluation_dataset
from api_clients import call_openrouter, call_gemini
from prompt_templates import get_few_shot_prompt
from evaluate import calculate_metrics

# Lista oficial dos modelos a serem avaliados via OpenRouter
OPENROUTER_MODELS = [
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4o-mini",
    "deepseek/deepseek-chat",
    "perplexity/llama-3.1-sonar-large-128k-chat"
]


def parse_prediction(response_text: str) -> int:
    """
    Normaliza e converte a resposta textual da IA em valor binário:
    - 1 para SIM (existe elo de rastreabilidade)
    - 0 para NÃO (não existe elo)
    """
    cleaned = response_text.strip().upper()

    if "SIM" in cleaned and "NÃO" not in cleaned and "NAO" not in cleaned:
        return 1
    elif "NÃO" in cleaned or "NAO" in cleaned:
        return 0
    elif "YES" in cleaned and "NO" not in cleaned:
        return 1
    elif "NO" in cleaned:
        return 0

    # Fallback conservador para saídas ambíguas ou erros
    return 0


def evaluate_model(model_name: str, df_subset: pd.DataFrame) -> Dict[str, Any]:
    """
    Executa a inferência Few-Shot para um modelo específico via OpenRouter
    e calcula as métricas com base no gabarito real.

    :param model_name: Identificador oficial do modelo no OpenRouter.
    :param df_subset: Subconjunto padronizado de pares para avaliação.
    :return: Dicionário contendo as métricas de Precision, Recall, F1 e Matriz de Confusão.
    """
    print("\n" + "=" * 70, flush=True)
    print(f"  AVALIANDO MODELO: {model_name} (Estratégia Few-Shot)", flush=True)
    print("=" * 70, flush=True)

    y_true = []
    y_pred = []

    for index, row in df_subset.iterrows():
        req_id = row["req_id"]
        req_text = row["req_text"]
        test_id = row["test_id"]
        test_text = row["test_text"]
        ground_truth = int(row["ground_truth"])

        # Monta o prompt Few-shot
        prompt = get_few_shot_prompt(req_text, test_text)

        print("-" * 60, flush=True)
        print(f"[{model_name}] Par #{len(y_true) + 1}/{len(df_subset)}: [{req_id}] <---> [{test_id}]", flush=True)
        print(f"Ground Truth : {'1 (SIM - Elo Existe)' if ground_truth == 1 else '0 (NÃO - Sem Elo)'}", flush=True)
        print(f"Enviando requisição ao OpenRouter ({model_name})...", flush=True)

        try:
            raw_response = call_openrouter(prompt, model_name=model_name)
        except Exception as e:
            raw_response = f"ERRO: {e}"

        pred_bin = parse_prediction(raw_response)

        y_true.append(ground_truth)
        y_pred.append(pred_bin)

        print(f"Resposta IA  : {raw_response} -> [Predição: {pred_bin}]", flush=True)

    # Cálculo e exibição das métricas deste modelo
    metrics = calculate_metrics(y_true, y_pred)
    return metrics


def run_pipeline(models: Optional[List[str]] = None, max_eval: Optional[int] = 50) -> Dict[str, Dict[str, Any]]:
    """
    Executa o pipeline de validação TLR multimodelo sobre o benchmark estratificado do iTrust.
    Todos os modelos avaliam rigorosamente os mesmos pares para assegurar comparabilidade científica.

    :param models: Lista de modelos para avaliação. Se None, utiliza OPENROUTER_MODELS.
    :param max_eval: Quantidade de instâncias a serem avaliadas (padrão = 50).
    :return: Dicionário mapeando model_name para suas respectivas métricas.
    """
    print("=" * 75, flush=True)
    print("  INICIANDO PIPELINE DE AVALIAÇÃO MULTIMODELO (TLR) - iTrust BENCHMARK", flush=True)
    print("=" * 75, flush=True)

    # 1. Validação de credenciais do OpenRouter
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        print("[AVISO]: A variável OPENROUTER_API_KEY não está preenchida no arquivo .env.", flush=True)
        print("Configure a chave no arquivo .env para permitir as chamadas ao OpenRouter.\n", flush=True)
        return {}

    # 2. Carrega a base oficial de avaliação (858 pares: 286 positivos, 572 negativos)
    df_eval = get_stratified_evaluation_dataset()
    total_disponivel = len(df_eval)

    # Garante que o mesmo subconjunto seja utilizado para todos os modelos
    if max_eval is not None and max_eval < total_disponivel:
        df_subset = df_eval.head(max_eval).copy()
        print(f"\n[AMOSTRA ESTRATIFICADA]: Avaliando as primeiras {max_eval} de {total_disponivel} instâncias para todos os modelos.\n", flush=True)
    else:
        df_subset = df_eval.copy()
        print(f"\n[AVALIAÇÃO COMPLETA]: Executando todas as {total_disponivel} instâncias para todos os modelos.\n", flush=True)

    target_models = models if models is not None else OPENROUTER_MODELS
    results_summary: Dict[str, Dict[str, Any]] = {}

    # 3. Iteração sobre a lista de modelos
    for model_name in target_models:
        metrics = evaluate_model(model_name, df_subset)
        results_summary[model_name] = metrics

    # 4. Exibição do quadro comparativo final consolidado
    print("\n" + "=" * 80)
    print("          QUADRO COMPARATIVO CONSOLIDADO (MULTIMODELO - FEW-SHOT)")
    print("=" * 80)
    print(f"{'Modelo':<45} | {'Precisão':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 80)
    for model, m in results_summary.items():
        p = m["precision"]
        r = m["recall"]
        f1 = m["f1"]
        print(f"{model:<45} | {p * 100:>8.2f}% | {r * 100:>8.2f}% | {f1 * 100:>8.2f}%")
    print("=" * 80 + "\n")

    return results_summary


if __name__ == "__main__":
    # Executa a homologação multimodelo com os mesmos 50 pares via OpenRouter
    run_pipeline(max_eval=50)

