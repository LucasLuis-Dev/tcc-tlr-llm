import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

text_intro = """\
# Análise Qualitativa de Erros (TCC TLR)
Neste notebook, vamos investigar as falhas cometidas pelos modelos na avaliação final (Falsos Positivos e Falsos Negativos).
Temos os seguintes objetivos de pesquisa:
1. **Falsos Positivos (Alucinações):** Onde modelos de alta precisão (como o Claude 3.5 Sonnet ou DeepSeek) falharam?
2. **Falsos Negativos (Omissões):** Quais elos o GPT-4o-mini (nosso campeão) deixou passar no Few-Shot?
"""

code_setup = """\
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('../src')
from data_loader import get_stratified_evaluation_dataset

# Configurações visuais
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Carrega a base de instâncias oficiais (req_text, test_text, ground_truth)
df_itrust = get_stratified_evaluation_dataset()
df_itrust['pair_id'] = df_itrust.index + 1

# Carrega a base de predições que o Antigravity extraiu dos logs
df_preds = pd.read_csv('../data/processed/predictions_raw.csv')

# Junta as duas bases (Merge)
df = df_preds.merge(df_itrust, on=['pair_id', 'ground_truth'], how='left')

print(f"Total de predições carregadas: {len(df)}")
df.head()
"""

text_fp = """\
## 1. Investigação de Falsos Positivos (FP) - "Alucinações"
Modelos como **Claude 3.5 Sonnet** e **DeepSeek** apresentaram Precisão altíssima (muitas vezes acima de 95%). Isso significa que eles raramente "alucinam" um elo. 
Vamos isolar as raras vezes em que eles preveram `1` quando a realidade era `0`.
"""

code_fp = """\
# Filtra Falsos Positivos (Previu 1, mas Real é 0)
df_fp = df[(df['prediction'] == 1) & (df['ground_truth'] == 0)]

# Vamos olhar os Falsos Positivos apenas do Claude e DeepSeek
modelos_alta_precisao = ['anthropic/claude-3.5-sonnet', 'deepseek/deepseek-chat']
df_fp_high_prec = df_fp[df_fp['model'].isin(modelos_alta_precisao)]

print(f"Total de Falsos Positivos nos modelos de alta precisão: {len(df_fp_high_prec)}")
display(df_fp_high_prec.groupby(['model', 'strategy']).size().reset_index(name='FP Count'))
"""

code_fp_view = """\
# Imprime um exemplo de Falso Positivo para análise semântica
if not df_fp_high_prec.empty:
    amostra_fp = df_fp_high_prec.iloc[0]
    print(f"MODELO: {amostra_fp['model']} | ESTRATÉGIA: {amostra_fp['strategy']}")
    print("="*80)
    print(f"REQUISITO:\\n{amostra_fp['req_text']}")
    print("-" * 80)
    print(f"CASO DE TESTE:\\n{amostra_fp['test_text'][:1000]}... [truncado]")
"""

text_fn = """\
## 2. Investigação de Falsos Negativos (FN) - "Omissões"
O **GPT-4o-mini** foi o campeão de Revocação, principalmente usando **Few-Shot**.
Vamos isolar as instâncias que ele *perdeu* (Previu `0` quando a realidade era `1`) para entender as limitações atuais dessa técnica.
"""

code_fn = """\
# Filtra Falsos Negativos (Previu 0, mas Real é 1)
df_fn = df[(df['prediction'] == 0) & (df['ground_truth'] == 1)]

# Isola o campeão: GPT-4o-mini (Few-Shot)
df_fn_gpt = df_fn[(df_fn['model'] == 'openai/gpt-4o-mini') & (df_fn['strategy'] == 'Few-Shot')]

print(f"Total de Falsos Negativos do GPT-4o-mini (Few-Shot): {len(df_fn_gpt)} de 286 elos reais possíveis.")
"""

code_fn_view = """\
# Imprime um exemplo de Falso Negativo para análise
if not df_fn_gpt.empty:
    amostra_fn = df_fn_gpt.iloc[0]
    print(f"MODELO: {amostra_fn['model']} | ESTRATÉGIA: {amostra_fn['strategy']}")
    print("="*80)
    print(f"REQUISITO:\\n{amostra_fn['req_text']}")
    print("-" * 80)
    print(f"CASO DE TESTE:\\n{amostra_fn['test_text'][:1000]}... [truncado]")
"""

text_viz = """\
## 3. Visualização: Tipos de Erros por Estratégia
Qual estratégia induz mais alucinações (FP) e qual induz mais omissões (FN)?
"""

code_viz = """\
df['error_type'] = 'Correto'
df.loc[(df['prediction'] == 1) & (df['ground_truth'] == 0), 'error_type'] = 'Falso Positivo'
df.loc[(df['prediction'] == 0) & (df['ground_truth'] == 1), 'error_type'] = 'Falso Negativo'

# Agrupa por estratégia e tipo de erro
erros_por_estrategia = df[df['error_type'] != 'Correto'].groupby(['strategy', 'error_type']).size().reset_index(name='count')

plt.figure(figsize=(10, 6))
sns.barplot(data=erros_por_estrategia, x='strategy', y='count', hue='error_type', palette=['#ff7f0e', '#d62728'])
plt.title("Volume de Erros (FP e FN) por Estratégia de Prompt", fontsize=14)
plt.ylabel("Quantidade de Erros Acumulados")
plt.xlabel("Estratégia")
plt.legend(title="Tipo de Erro")
plt.show()
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_markdown_cell(text_fp),
    nbf.v4.new_code_cell(code_fp),
    nbf.v4.new_code_cell(code_fp_view),
    nbf.v4.new_markdown_cell(text_fn),
    nbf.v4.new_code_cell(code_fn),
    nbf.v4.new_code_cell(code_fn_view),
    nbf.v4.new_markdown_cell(text_viz),
    nbf.v4.new_code_cell(code_viz)
]

out_file = Path(r"C:\Users\Lucas Luis\Documents\tcc-tlr-llm\notebooks\02_analise_erros.ipynb")
out_file.parent.mkdir(parents=True, exist_ok=True)

with open(out_file, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook gerado em: {out_file}")
