import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
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
from prompt_templates import get_zero_shot_prompt, get_few_shot_prompt, get_cot_prompt
from evaluate import calculate_metrics

# Lista oficial dos modelos a serem avaliados via OpenRouter
OPENROUTER_MODELS = [
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4o-mini",
    "deepseek/deepseek-chat",
    "perplexity/llama-3.1-sonar-large-128k-chat"
]

# Lista padrão unificada incluindo o Gemini
DEFAULT_MODELS = OPENROUTER_MODELS + ["gemini-3.6-flash"]

# Dicionário mapeando as estratégias de prompt oficiais
PROMPT_STRATEGIES: Dict[str, Callable[[str, str], str]] = {
    "Zero-Shot": get_zero_shot_prompt,
    "Few-Shot": get_few_shot_prompt,
    "Chain-of-Thought (CoT)": get_cot_prompt
}


def dispatch_inference(prompt: str, model_name: str) -> str:
    """
    Encaminha a requisição para o provedor apropriado (Google Gemini nativo ou OpenRouter).
    """
    if "gemini" in model_name.lower() and "/" not in model_name:
        return call_gemini(prompt, model_name=model_name)
    else:
        return call_openrouter(prompt, model_name=model_name, max_tokens=400)


def parse_prediction(response_text: str) -> int:
    """
    Normaliza e converte a resposta textual da IA em valor binário:
    - 1 para SIM (existe elo de rastreabilidade)
    - 0 para NÃO (não existe elo)

    Compatível com Zero-Shot, Few-Shot e Chain-of-Thought (CoT),
    inspecionando prioritariamente a conclusão nas linhas finais do texto gerado.
    """
    if not response_text or not isinstance(response_text, str):
        return 0

    raw = response_text.strip()
    if raw.startswith("ERRO") or raw.startswith("ERRO_REQUISICAO"):
        return 0

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return 0

    # Para CoT e respostas explicativas, a decisão final está nas últimas linhas
    for line in reversed(lines[-3:]):
        cleaned = line.upper().replace("*", "").replace(".", "").replace(":", " ").replace("-", " ")
        tokens = cleaned.split()

        has_sim = any(t in ("SIM", "YES") for t in tokens) or "SIM" in cleaned or "YES" in cleaned
        has_nao = any(t in ("NÃO", "NAO", "NO") for t in tokens) or "NÃO" in cleaned or "NAO" in cleaned

        if has_sim and not has_nao:
            return 1
        elif has_nao and not has_sim:
            return 0
        elif has_sim and has_nao:
            # Se ambos ocorrem na mesma linha, desempata pela palavra-chave de conclusão
            if "RESPOSTA SIM" in cleaned or cleaned.endswith("SIM") or cleaned.endswith("YES"):
                return 1
            elif "RESPOSTA NÃO" in cleaned or "RESPOSTA NAO" in cleaned or cleaned.endswith("NÃO") or cleaned.endswith("NAO"):
                return 0

    # Fallback de busca no texto completo
    cleaned_all = raw.upper()
    if cleaned_all.endswith("SIM") or cleaned_all.endswith("SIM."):
        return 1
    elif cleaned_all.endswith("NÃO") or cleaned_all.endswith("NÃO.") or cleaned_all.endswith("NAO"):
        return 0

    return 0


def evaluate_model_strategy(
    model_name: str,
    strategy_name: str,
    prompt_func: Callable[[str, str], str],
    df_subset: pd.DataFrame
) -> Dict[str, Any]:
    """
    Executa a inferência para um modelo específico com uma estratégia de prompt selecionada
    e calcula as métricas com base no gabarito real.

    :param model_name: Identificador oficial do modelo.
    :param strategy_name: Nome da estratégia (Zero-Shot, Few-Shot, CoT).
    :param prompt_func: Função construtora do template de prompt.
    :param df_subset: Subconjunto padronizado de pares para avaliação.
    :return: Dicionário contendo as métricas de Precision, Recall, F1 e Matriz de Confusão.
    """
    print("\n" + "=" * 75, flush=True)
    print(f"  MODELO: {model_name} | ESTRATÉGIA: {strategy_name}", flush=True)
    print("=" * 75, flush=True)

    y_true = []
    y_pred = []

    for index, row in df_subset.iterrows():
        req_id = row["req_id"]
        req_text = row["req_text"]
        test_id = row["test_id"]
        test_text = row["test_text"]
        ground_truth = int(row["ground_truth"])

        # Monta o prompt conforme a estratégia
        prompt = prompt_func(req_text, test_text)

        print("-" * 60, flush=True)
        print(f"[{model_name}][{strategy_name}] Par #{len(y_true) + 1}/{len(df_subset)}: [{req_id}] <---> [{test_id}]", flush=True)
        print(f"Ground Truth : {'1 (SIM - Elo Existe)' if ground_truth == 1 else '0 (NÃO - Sem Elo)'}", flush=True)
        print(f"Enviando requisição ({model_name})...", flush=True)

        try:
            raw_response = dispatch_inference(prompt, model_name=model_name)
        except Exception as e:
            raw_response = f"ERRO: {e}"

        pred_bin = parse_prediction(raw_response)

        y_true.append(ground_truth)
        y_pred.append(pred_bin)

        # Exibe resposta sintetizada (primeiras e últimas linhas se muito longa)
        preview_response = raw_response.strip().replace("\n", " ")
        if len(preview_response) > 120:
            preview_response = preview_response[:100] + "... " + preview_response[-20:]
        print(f"Resposta IA  : {preview_response} -> [Predição: {pred_bin}]", flush=True)

        # Pausa para respeitar limites de requisições por minuto (RPM)
        time.sleep(0.5)

    # Cálculo e exibição das métricas desta combinação
    metrics = calculate_metrics(y_true, y_pred)
    return metrics


def run_pipeline(
    models: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
    max_eval: Optional[int] = 50
) -> Dict[tuple, Dict[str, Any]]:
    """
    Executa o pipeline de validação TLR cruzando modelos e estratégias de prompt (Zero-Shot, Few-Shot e CoT).
    Todos os modelos e estratégias avaliam rigorosamente os mesmos pares para assegurar comparabilidade científica.

    :param models: Lista de modelos para avaliação. Se None, utiliza DEFAULT_MODELS.
    :param strategies: Lista de nomes de estratégias (chaves de PROMPT_STRATEGIES). Se None, executa todas.
    :param max_eval: Quantidade de instâncias a serem avaliadas (padrão = 50).
    :return: Dicionário mapeando (model_name, strategy_name) para suas respectivas métricas.
    """
    print("=" * 80, flush=True)
    print("  INICIANDO PIPELINE DE AVALIAÇÃO MULTIMODELO & MULTIESTRATÉGIA (TLR)", flush=True)
    print("=" * 80, flush=True)

    # 1. Carrega a base oficial de avaliação (858 pares: 286 positivos, 572 negativos)
    df_eval = get_stratified_evaluation_dataset()
    total_disponivel = len(df_eval)

    # Garante que o mesmo subconjunto seja utilizado em todas as baterias
    if max_eval is not None and max_eval < total_disponivel:
        df_subset = df_eval.head(max_eval).copy()
        print(f"\n[AMOSTRA ESTRATIFICADA]: Avaliando as primeiras {max_eval} de {total_disponivel} instâncias.\n", flush=True)
        out_csv = Path(__file__).resolve().parent.parent / "data" / "processed" / "multimodel_prompt_results.csv"
    else:
        df_subset = df_eval.copy()
        print(f"\n[AVALIAÇÃO COMPLETA]: Executando todas as {total_disponivel} instâncias.\n", flush=True)
        out_csv = Path(__file__).resolve().parent.parent / "data" / "processed" / "multimodel_prompt_results_full.csv"

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    target_models = models if models is not None else DEFAULT_MODELS
    target_strategies = {
        name: PROMPT_STRATEGIES[name]
        for name in (strategies if strategies is not None else PROMPT_STRATEGIES.keys())
        if name in PROMPT_STRATEGIES
    }

    # Carrega progresso anterior para evitar reprocessamento
    completed_combos = set()
    if out_csv.exists():
        try:
            existing_df = pd.read_csv(out_csv)
            for _, row in existing_df.iterrows():
                completed_combos.add((row["model"], row["strategy"]))
            print(f"\n[Retomando Progresso]: {len(completed_combos)} combinações já processadas encontradas em {out_csv.name}.\n", flush=True)
        except Exception as e:
            print(f"Erro ao ler CSV existente: {e}")

    results_summary: Dict[tuple, Dict[str, Any]] = {}

    # 2. Iteração cruzada: Modelo x Estratégia de Prompt
    for model_name in target_models:
        for strategy_name, prompt_func in target_strategies.items():
            if (model_name, strategy_name) in completed_combos:
                print(f"[{model_name} | {strategy_name}] Já avaliado anteriormente. Pulando...", flush=True)
                continue

            metrics = evaluate_model_strategy(model_name, strategy_name, prompt_func, df_subset)
            results_summary[(model_name, strategy_name)] = metrics
            
            # Exporta progressivamente (Checkpoint)
            new_record = pd.DataFrame([{
                "model": model_name,
                "strategy": strategy_name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "tp": metrics["confusion_matrix"]["tp"],
                "fp": metrics["confusion_matrix"]["fp"],
                "tn": metrics["confusion_matrix"]["tn"],
                "fn": metrics["confusion_matrix"]["fn"]
            }])
            
            if not out_csv.exists() or out_csv.stat().st_size == 0:
                new_record.to_csv(out_csv, index=False, encoding="utf-8")
            else:
                new_record.to_csv(out_csv, mode="a", header=False, index=False, encoding="utf-8")
                
            print(f"[Checkpoint] Salvo resultado para {model_name} - {strategy_name}\n", flush=True)

    print("\n[Execução Finalizada] Todos os modelos e estratégias foram processados.\n")
    return results_summary


if __name__ == "__main__":
    # Executa a avaliação completa (858 pares) com todos os modelos e estratégias
    run_pipeline(max_eval=None)

