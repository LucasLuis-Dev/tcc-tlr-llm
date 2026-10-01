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
from api_clients import call_gemini
from main import parse_prediction

df_eval = get_stratified_evaluation_dataset()
y_true = []
y_pred = []

print(f"Iniciando avaliação Chain-of-Thought para gemini-3.6-flash (Total: {len(df_eval)} pares)")

for index, row in df_eval.iterrows():
    req_text = row["req_text"]
    test_text = row["test_text"]
    ground_truth = int(row["ground_truth"])
    
    prompt = get_cot_prompt(req_text, test_text)
    print(f"Avaliando par {index + 1}/858...")
    
    try:
        raw_response = call_gemini(prompt, "gemini-3.6-flash")
    except Exception as e:
        raw_response = f"ERRO: {e}"
        
    pred = parse_prediction(raw_response)
    print(f"Resposta -> Predição: {pred}")
    
    y_true.append(ground_truth)
    y_pred.append(pred)
    time.sleep(0.5)

metrics = calculate_metrics(y_true, y_pred)
print(f"Novas métricas CoT: F1={metrics['f1']:.4f}")

out_csv = Path(__file__).resolve().parent.parent / "data" / "processed" / "multimodel_prompt_results_full.csv"
df_res = pd.read_csv(out_csv)

# Remove old row
df_res = df_res[~((df_res['model'] == 'gemini-3.6-flash') & (df_res['strategy'] == 'Chain-of-Thought (CoT)'))]

# Add new row
new_row = {
    "model": "gemini-3.6-flash",
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
