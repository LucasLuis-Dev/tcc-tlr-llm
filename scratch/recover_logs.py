import re
import os
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from evaluate import calculate_metrics

log_path = Path(r"C:\Users\Lucas Luis\.gemini\antigravity-ide\brain\ddf49f81-4058-486c-96e9-27320f91b9d8\.system_generated\tasks\task-287.log")
out_csv = Path(__file__).resolve().parent.parent / "data" / "processed" / "multimodel_prompt_results_full.csv"

with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

data = {} # (model, strategy) -> {'y_true': [], 'y_pred': []}

current_model = None
current_strategy = None
current_gt = None

for line in lines:
    line = line.strip()
    
    # Match: [anthropic/claude-3.5-sonnet][Zero-Shot] Par #1/858:...
    m_header = re.search(r'^\[(.*?)\]\[(.*?)\] Par #(\d+)/858:', line)
    if m_header:
        current_model = m_header.group(1)
        current_strategy = m_header.group(2)
        if (current_model, current_strategy) not in data:
            data[(current_model, current_strategy)] = {'y_true': [], 'y_pred': []}
            
    m_gt = re.search(r'^Ground Truth :\s+(\d)', line)
    if m_gt:
        current_gt = int(m_gt.group(1))
        
    m_pred = re.search(r'\[Predição: (\d)\]$', line)
    if m_pred and current_model and current_strategy and current_gt is not None:
        pred = int(m_pred.group(1))
        data[(current_model, current_strategy)]['y_true'].append(current_gt)
        data[(current_model, current_strategy)]['y_pred'].append(pred)
        current_gt = None # reset for next

records = []
for (model, strategy), vals in data.items():
    if len(vals['y_true']) == 858:
        metrics = calculate_metrics(vals['y_true'], vals['y_pred'])
        records.append({
            "model": model,
            "strategy": strategy,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "tp": metrics["confusion_matrix"]["tp"],
            "fp": metrics["confusion_matrix"]["fp"],
            "tn": metrics["confusion_matrix"]["tn"],
            "fn": metrics["confusion_matrix"]["fn"]
        })
        print(f"Recuperado com sucesso: {model} - {strategy} (858 pares)")
    else:
        print(f"Ignorado (incompleto): {model} - {strategy} ({len(vals['y_true'])} pares)")

if records:
    df_res = pd.DataFrame(records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Salvo em {out_csv}")
else:
    print("Nenhum modelo completo recuperado.")
