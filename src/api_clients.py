import os
import requests
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def call_gemini(prompt: str, model_name: str = "gemini-1.5-flash") -> str:
    """
    Envia uma requisição para a API do Google Gemini (rota Google AI Studio).
    
    :param prompt: Texto do prompt a ser enviado.
    :param model_name: Nome do modelo (padrão: gemini-1.5-flash).
    :return: Texto da resposta gerada pelo modelo.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY não foi encontrada no arquivo .env.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
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

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
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
        print(f"[ERRO Gemini API]: Falha na requisição HTTP - {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"[Detalhes]: {e.response.text}")
        return f"ERRO_REQUISICAO: {e}"


def call_openrouter(prompt: str, model_name: str = "anthropic/claude-3.5-sonnet") -> str:
    """
    Envia uma requisição para o agregador OpenRouter.
    
    :param prompt: Texto do prompt a ser enviado.
    :param model_name: Identificador do modelo no OpenRouter (ex: 'anthropic/claude-3.5-sonnet',
                       'openai/gpt-4o-mini', 'deepseek/deepseek-chat', etc.).
    :return: Texto da resposta gerada pelo modelo.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY não foi encontrada no arquivo .env.")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/LucasLuis-Dev/tcc-tlr-llm",
        "X-Title": "TCC TLR LLM"
    }
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Extrai o texto da resposta
        choices = data.get("choices", [])
        if not choices:
            return "Nenhuma resposta retornada pelo OpenRouter."
            
        return choices[0].get("message", {}).get("content", "").strip()

    except requests.exceptions.RequestException as e:
        print(f"[ERRO OpenRouter API]: Falha na requisição HTTP - {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"[Detalhes]: {e.response.text}")
        return f"ERRO_REQUISICAO: {e}"
