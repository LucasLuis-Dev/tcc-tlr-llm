import os
import sys
from pathlib import Path

# Adiciona o diretório 'src' ao sys.path para importações diretas
sys.path.append(str(Path(__file__).resolve().parent))

from data_loader import get_mock_dataset
from api_clients import call_gemini


def run_pipeline():
    """
    Executa o pipeline principal com Mock Dataset e inferência Zero-shot via Gemini.
    """
    print("=" * 70)
    print("  INICIANDO PIPELINE DE RASTREABILIDADE (TLR) - EXPERIMENTO INICIAL")
    print("=" * 70)

    # Verifica se a chave do Gemini está configurada no .env
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("\n[AVISO]: A variável GEMINI_API_KEY não está preenchida no arquivo .env.")
        print("Para realizar as chamadas reais à API, configure sua chave no arquivo .env.\n")
        return

    # 1. Carrega o mock dataset simulado
    df = get_mock_dataset()
    print(f"Total de pares para avaliação: {len(df)}\n")

    # 2. Itera sobre cada par e executa a classificação Zero-shot
    for index, row in df.iterrows():
        req_id = row["req_id"]
        req_text = row["req_text"]
        test_id = row["test_id"]
        test_text = row["test_text"]
        ground_truth = row["ground_truth"]

        # Monta o prompt Zero-shot
        prompt = (
            f"Dado o requisito: '{req_text}' e o caso de teste: '{test_text}', "
            f"eles possuem relação de rastreabilidade? Responda apenas SIM ou NÃO."
        )

        print("-" * 50)
        print(f"Par #{index + 1}: [{req_id}] <--> [{test_id}]")
        print(f"Requisito : {req_text}")
        print(f"Teste     : {test_text}")
        print("Enviando requisição para o Gemini...")

        # 3. Chama a API do Gemini
        try:
            prediction = call_gemini(prompt)
        except Exception as e:
            prediction = f"ERRO: {e}"

        print(f"Resposta IA  : {prediction}")
        print(f"Ground Truth : {'1 (SIM)' if ground_truth == 1 else '0 (NÃO)'}")


if __name__ == "__main__":
    run_pipeline()
