# TCC TLR LLM - Recuperação de Links de Rastreabilidade com LLMs

Projeto de Trabalho de Conclusão de Curso focado na **Recuperação de Links de Rastreabilidade (Traceability Link Recovery - TLR)** entre artefatos de software (como requisitos e código-fonte/classes) utilizando **Modelos de Linguagem de Larga Escala (LLMs)** aplicados ao benchmark **iTrust**.

---

## Estrutura do Projeto

```text
tcc-tlr-llm/
├── data/
│   ├── raw/                # Dados brutos do benchmark (não versionados no git)
│   └── processed/          # Dados processados e prontos para modelagem/experimentos
├── notebooks/
│   └── 01_exploracao_dataset.ipynb  # Notebook de análise exploratória
├── src/
│   ├── api_clients.py      # Integração com APIs de LLMs (Gemini, OpenRouter)
│   ├── evaluate.py         # Cálculo de métricas (Precision, Recall, F1, etc.)
│   └── main.py             # Pipeline principal de execução dos experimentos
├── .env                    # Chaves de API e variáveis de ambiente locais (não versionado)
├── .gitignore              # Configuração de arquivos ignorados pelo Git
├── requirements.txt        # Dependências do projeto Python
└── README.md               # Documentação do projeto
```

---

## Configuração do Ambiente

1. **Clonar o repositório:**
   ```bash
   git clone https://github.com/LucasLuis-Dev/tcc-tlr-llm.git
   cd tcc-tlr-llm
   ```

2. **Criar e ativar o ambiente virtual (recomendado):**
   ```bash
   python -m venv venv
   # No Windows:
   .\venv\Scripts\activate
   # No Linux/macOS:
   source venv/bin/activate
   ```

3. **Instalar dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar variáveis de ambiente:**
   No arquivo `.env`, preencha suas credenciais de API:
   ```env
   GEMINI_API_KEY=sua_chave_gemini_aqui
   OPENROUTER_API_KEY=sua_chave_openrouter_aqui
   ```
