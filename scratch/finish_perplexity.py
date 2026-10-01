import sys
import os
import re
from pathlib import Path
import pandas as pd
import time

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from evaluate import calculate_metrics
from data_loader import get_stratified_evaluation_dataset
from prompt_templates import get_cot_prompt
from api_clients import call_openrouter
from main import parse_prediction

log_path = Path(r"C:\Users\Lucas Luis\.gemini\antigravity-ide\brain\ddf49f81-4058-486c-96e9-27320f91b9d8\.system_generated\tasks\task-387.log")

y_true = []
y_pred = []

with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

current_gt = None
is_perplexity = False

for line in lines:
    line = line.strip()
    
    if "[perplexity/llama-3.1-sonar-large-128k-chat][Chain-of-Thought (CoT)]" in line:
        is_perplexity = True
        
    if is_perplexity:
        m_gt = re.search(r'^Ground Truth :\s+(\d)', line)
        if m_gt:
            current_gt = int(m_gt.group(1))
            
        m_pred = re.search(r'\[Predição: (\d)\]$', line)
        if m_pred and current_gt is not None:
            if "402 Client Error" in line or "Insufficient credits" in line:
                break # pare no primeiro erro 402
            pred = int(m_pred.group(1))
            y_true.append(current_gt)
            y_pred.append(pred)
            current_gt = None
            is_perplexity = False # reseta para esperar o próximo cabeçalho

print(f"Recuperados {len(y_pred)} predições válidas do log.")

df_eval = get_stratified_evaluation_dataset()
missing_start = len(y_pred)

if missing_start == 858:
    print("Nenhum par faltando!")
    sys.exit(0)

df_missing = df_eval.iloc[missing_start:]
print(f"Faltam {len(df_missing)} pares para avaliar.")

for index, row in df_missing.iterrows():
    req_text = row["req_text"]
    test_text = row["test_text"]
    ground_truth = int(row["ground_truth"])
    
    prompt = get_cot_prompt(req_text, test_text)
    print(f"Avaliando par {missing_start + 1}/858...")
    
    try:
        raw_response = call_openrouter(prompt, "perplexity/llama-3.1-sonar-large-128k-chat", max_tokens=400)
    except Exception as e:
        raw_response = f"ERRO: {e}"
        
    pred = parse_prediction(raw_response)
    print(f"Resposta -> Predição: {pred}")
    
    y_true.append(ground_truth)
    y_pred.append(pred)
    missing_start += 1
    time.sleep(0.5)

metrics = calculate_metrics(y_true, y_pred)
print(f"Novas métricas: F1={metrics['f1']:.4f}")

out_csv = Path(__file__).resolve().parent.parent / "data" / "processed" / "multimodel_prompt_results_full.csv"
df_res = pd.read_csv(out_csv)

df_res = df_res[~((df_res['model'] == 'perplexity/llama-3.1-sonar-large-128k-chat') & (df_res['strategy'] == 'Chain-of-Thought (CoT)'))]

new_row = {
    "model": "perplexity/llama-3.1-sonar-large-128k-chat",
    "strategy": "Chain-of-Thought (CoT)",
    "precision": metrics["precision"],
    "recall": metrics["recall"],
    "f1": metrics["f1"],
    "tp": metrics["confusion_matrix"]["tp"],
    "fp": metrics["confusion_matrix"]["fp"],
    "tn": metrics["confusion_matrix"]["tn"],
    "fn": metrics["confusion_matrix"]["fn"]
}
df_res = pd.concat([df_res, pd.DataFrame([new_row])], ignore_index=True)
df_res.to_csv(out_csv, index=False, encoding="utf-8")
print("CSV atualizado com sucesso!")
