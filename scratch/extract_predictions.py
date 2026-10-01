import os
import re
import pandas as pd
from pathlib import Path

# Pasta onde ficam os logs das tasks
log_dir = Path(r"C:\Users\Lucas Luis\.gemini\antigravity-ide\brain\ddf49f81-4058-486c-96e9-27320f91b9d8\.system_generated\tasks")
out_csv = Path(r"c:\Users\Lucas Luis\Documents\tcc-tlr-llm\data\processed\predictions_raw.csv")

data = []

# Padrões Regex
re_header = re.compile(r"^\[([^\]]+)\]\[([^\]]+)\] Par #(\d+)/858")
re_gt = re.compile(r"^Ground Truth :\s+(\d)")
re_pred = re.compile(r"\[Predi..o:\s*(\d)\]$")

# Processar todos os arquivos .log do diretório
for log_file in log_dir.glob("*.log"):
    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        current_model = None
        current_strategy = None
        current_par = None
        current_gt = None
        
        for line in f:
            line = line.strip()
            
            m_header = re_header.search(line)
            if m_header:
                current_model = m_header.group(1)
                current_strategy = m_header.group(2)
                current_par = int(m_header.group(3))
                continue
                
            m_gt = re_gt.search(line)
            if m_gt and current_par is not None:
                current_gt = int(m_gt.group(1))
                continue
                
            m_pred = re_pred.search(line)
            if m_pred and current_gt is not None:
                pred = int(m_pred.group(1))
                data.append({
                    "model": current_model,
                    "strategy": current_strategy,
                    "pair_id": current_par, # 1-indexed
                    "ground_truth": current_gt,
                    "prediction": pred
                })
                # Reset para o próximo
                current_model = None
                current_strategy = None
                current_par = None
                current_gt = None

df = pd.DataFrame(data)
# Pode haver duplicações de execuções canceladas, vamos manter as últimas
df = df.drop_duplicates(subset=["model", "strategy", "pair_id"], keep="last")

df.to_csv(out_csv, index=False)
print(f"Extraídas {len(df)} predições para {out_csv}")
