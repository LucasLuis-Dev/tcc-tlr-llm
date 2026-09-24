import os
import requests
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def call_gemini(prompt: str, model_name: str = "gemini-3.6-flash") -> str:
    """
    Envia uma requisição para a API do Google Gemini (rota Google AI Studio).
    
    :param prompt: Texto do prompt a ser enviado.
    :param model_name: Nome do modelo (padrão: gemini-3.6-flash).
    :return: Texto da resposta gerada pelo modelo.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY não foi encontrada no arquivo .env.")

    # Remove prefixo 'models/' caso seja passado acidentalmente
    clean_model = model_name.replace("models/", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    import time
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            # Extrai o texto da resposta
            candidates = data.get("candidates", [])
            if not candidates:
                return "Nenhuma resposta retornada pela API do Gemini."
                
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                return "Conteúdo vazio retornado pelo Gemini."
                
            return parts[0].get("text", "").strip()

        except requests.exceptions.RequestException as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            if status_code in (503, 429) and attempt < max_retries:
                wait_seconds = attempt * 3
                print(f"[Aviso Gemini API]: Erro temporário ({status_code}). Tentativa {attempt}/{max_retries}. Aguardando {wait_seconds}s...")
                time.sleep(wait_seconds)
                continue

            print(f"[ERRO Gemini API]: Falha na requisição HTTP - {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"[Detalhes]: {e.response.text}")
            return f"ERRO_REQUISICAO: {e}"


def call_openrouter(
    prompt: str,
    model_name: str = "openai/gpt-4o-mini",
    max_tokens: int = 400,
    temperature: float = 0.0
) -> str:
    """
    Envia uma requisição para o agregador OpenRouter com retry e backoff exponencial.
    
    :param prompt: Texto do prompt a ser enviado.
    :param model_name: Identificador do modelo no OpenRouter (ex: 'openai/gpt-4o-mini',
                       'anthropic/claude-3.5-sonnet', 'deepseek/deepseek-chat',
                       'perplexity/llama-3.1-sonar-large-128k-chat').
    :param max_tokens: Limite máximo de tokens de resposta (padrão: 400 para permitir raciocínio CoT).
    :param temperature: Temperatura para controle de determinismo (padrão: 0.0).
    :return: Texto da resposta gerada pelo modelo.
    """
    import time

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY não foi encontrada no arquivo .env.")

    # Mapeamento para garantir compatibilidade caso os identificadores oficiais tenham sido atualizados no OpenRouter
    model_aliases = {
        "anthropic/claude-3.5-sonnet": "anthropic/claude-sonnet-4",
        "perplexity/llama-3.1-sonar-large-128k-chat": "perplexity/sonar",
    }
    target_model = model_aliases.get(model_name, model_name)

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/LucasLuis-Dev/tcc-tlr-llm",
        "X-Title": "TCC TLR LLM"
    }
    payload = {
        "model": target_model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    max_retries = 4
    base_delay = 2

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)

            # Caso receba 404/400 tentando modelo original, tenta alias de fallback se aplicável
            if response.status_code in (400, 404) and target_model != model_aliases.get(model_name):
                fallback = model_aliases.get(model_name)
                if fallback:
                    payload["model"] = fallback
                    target_model = fallback
                    continue

            response.raise_for_status()
            data = response.json()

            # Extrai o texto da resposta
            choices = data.get("choices", [])
            if not choices:
                return "Nenhuma resposta retornada pelo OpenRouter."

            msg = choices[0].get("message", {})
            content = msg.get("content")
            if not content:
                content = msg.get("reasoning") or msg.get("reasoning_content") or ""

            return content.strip()

        except requests.exceptions.RequestException as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            is_transient = (
                status_code in (429, 500, 502, 503, 504)
                or isinstance(e, (requests.exceptions.Timeout, requests.exceptions.ConnectionError))
            )
            if is_transient and attempt < max_retries:
                wait_seconds = base_delay * (2 ** (attempt - 1))
                print(
                    f"[Aviso OpenRouter - {target_model}]: Falha temporária ({status_code or 'Rede'}). "
                    f"Tentativa {attempt}/{max_retries}. Aguardando {wait_seconds}s...",
                    flush=True
                )
                time.sleep(wait_seconds)
                continue

            print(f"[ERRO OpenRouter API - {target_model}]: Falha na requisição HTTP - {e}", flush=True)
            if hasattr(e, "response") and e.response is not None:
                print(f"[Detalhes]: {e.response.text}", flush=True)
            return f"ERRO_REQUISICAO: {e}"

