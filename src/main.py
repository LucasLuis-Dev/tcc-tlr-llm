import os
import sys
from pathlib import Path

# Adiciona o diretório 'src' ao sys.path para importações diretas
sys.path.append(str(Path(__file__).resolve().parent))

from data_loader import load_itrust_dataset
from api_clients import call_gemini
from prompt_templates import get_zero_shot_prompt


def run_pipeline(sample_size: int = 2):
    """
    Executa o pipeline principal com o dataset real processado do iTrust e inferência Zero-shot via Gemini.
    
    :param sample_size: Quantidade de pares para avaliar (padrão 2 para validação rápida e econômica da API).
                        Defina como None para avaliar o dataset completo de 29.606 pares.
    """
    print("=" * 70, flush=True)
    print("  INICIANDO PIPELINE DE RASTREABILIDADE (TLR) - iTrust BENCHMARK", flush=True)
    print("=" * 70, flush=True)

    # 1. Carrega o dataset do iTrust (com amostragem balanceada para teste rápido)
    df = load_itrust_dataset(sample_size=sample_size)
    print(f"Total de pares selecionados para avaliação: {len(df)}", flush=True)
    print(f"Pares com elo verdadeiro (1): {df['ground_truth'].sum()} | Sem elo (0): {(df['ground_truth'] == 0).sum()}\n", flush=True)

    # Verifica se a chave do Gemini está configurada no .env
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("[AVISO]: A variável GEMINI_API_KEY não está preenchida no arquivo .env.", flush=True)
        print("Preencha sua chave no arquivo .env para executar chamadas reais à API do Gemini.\n", flush=True)
        print("Prévia dos pares que serão avaliados:", flush=True)
        print(df[["req_id", "test_id", "ground_truth"]].head(sample_size or 5), flush=True)
        return

    # 2. Itera sobre cada par e executa a classificação Zero-shot
    for index, row in df.iterrows():
        req_id = row["req_id"]
        req_text = row["req_text"]
        test_id = row["test_id"]
        test_text = row["test_text"]
        ground_truth = row["ground_truth"]

        # Gera o prompt Zero-shot estruturado
        prompt = get_zero_shot_prompt(req_text, test_text)

        print("-" * 60, flush=True)
        print(f"Par #{index + 1}: [{req_id}] <---> [{test_id}]", flush=True)
        print(f"Ground Truth : {'1 (SIM - Elo Existe)' if ground_truth == 1 else '0 (NÃO - Sem Elo)'}", flush=True)
        print("Enviando prompt para a API do Gemini...", flush=True)

        # 3. Chama a API do Gemini
        try:
            prediction = call_gemini(prompt)
        except Exception as e:
            prediction = f"ERRO: {e}"

        print(f"Resposta IA  : {prediction}", flush=True)


if __name__ == "__main__":
    run_pipeline(sample_size=2)
