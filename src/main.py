import os
import sys
from pathlib import Path
from typing import Optional

# Adiciona o diretório 'src' ao sys.path para importações diretas
sys.path.append(str(Path(__file__).resolve().parent))

from data_loader import get_stratified_evaluation_dataset
from api_clients import call_gemini
from prompt_templates import get_few_shot_prompt
from evaluate import calculate_metrics


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
    elif "YES" in cleaned:
        return 1
    elif "NO" in cleaned:
        return 0
    
    # Fallback conservador para saídas ambíguas ou erros
    return 0


def run_pipeline(max_eval: Optional[int] = 10):
    """
    Executa o pipeline oficial de validação de TLR com o dataset estratificado (858 pares)
    e computa as métricas clássicas (Precision, Recall, F1 e Matriz de Confusão).
    
    :param max_eval: Limite de pares para avaliação no loop.
                     Padrão = 10 para testes rápidos e econômicos.
                     Defina como None para rodar toda a base oficial de 858 pares.
    """
    print("=" * 70, flush=True)
    print("  INICIANDO PIPELINE DE AVALIAÇÃO (TLR) - iTrust BENCHMARK", flush=True)
    print("=" * 70, flush=True)

    # 1. Carrega a base oficial de avaliação (858 pares: 286 positivos, 572 negativos)
    df_eval = get_stratified_evaluation_dataset()
    total_disponivel = len(df_eval)

    # Aplica o limite se configurado
    if max_eval is not None and max_eval < total_disponivel:
        df_subset = df_eval.head(max_eval).copy()
        print(f"\n[MODO TESTE RÁPIDO]: Avaliando as primeiras {max_eval} de {total_disponivel} instâncias.", flush=True)
        print("Para avaliar toda a base oficial (858 instâncias), defina max_eval=None.\n", flush=True)
    else:
        df_subset = df_eval.copy()
        print(f"\n[AVALIAÇÃO COMPLETA]: Executando todas as {total_disponivel} instâncias da base oficial.\n", flush=True)

    # 2. Verifica se a chave da API está configurada
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("[AVISO]: A variável GEMINI_API_KEY não está preenchida no arquivo .env.", flush=True)
        print("Configure a chave no arquivo .env para permitir as chamadas à API.\n", flush=True)
        return

    # 3. Acumuladores de métricas
    y_true = []
    y_pred = []

    # 4. Loop de inferência
    for index, row in df_subset.iterrows():
        req_id = row["req_id"]
        req_text = row["req_text"]
        test_id = row["test_id"]
        test_text = row["test_text"]
        ground_truth = int(row["ground_truth"])

        # Monta o prompt Few-shot
        prompt = get_few_shot_prompt(req_text, test_text)

        print("-" * 60, flush=True)
        print(f"Par #{len(y_true) + 1}/{len(df_subset)}: [{req_id}] <---> [{test_id}]", flush=True)
        print(f"Ground Truth : {'1 (SIM - Elo Existe)' if ground_truth == 1 else '0 (NÃO - Sem Elo)'}", flush=True)
        print("Enviando requisição ao Gemini...", flush=True)

        try:
            raw_response = call_gemini(prompt)
        except Exception as e:
            raw_response = f"ERRO: {e}"

        pred_bin = parse_prediction(raw_response)
        
        y_true.append(ground_truth)
        y_pred.append(pred_bin)

        print(f"Resposta IA  : {raw_response} -> [Predição: {pred_bin}]", flush=True)

    # 5. Cálculo e exibição das métricas finais
    calculate_metrics(y_true, y_pred)


if __name__ == "__main__":
    # Executa o primeiro experimento com 50 pares da amostra estratificada
    run_pipeline(max_eval=50)
