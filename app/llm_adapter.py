from __future__ import annotations

import json
import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

# Provider selection. "ollama" (default, local dev) or "deepseek" (CI / remote).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

# Ollama (local)
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "dummy")

# DeepSeek (remote, OpenAI-compatible)
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")


_SYSTEM_PROMPT = (
    "你是一个金融日报编辑。"
    "你必须输出合法 JSON，且只输出 JSON。"
    "不要输出 markdown，不要输出额外解释。"
)


def _active_provider_config() -> tuple[str, str, str]:
    """Return (base_url, model, api_key) for the currently selected provider."""
    if LLM_PROVIDER == "deepseek":
        if not DEEPSEEK_API_KEY:
            raise RuntimeError(
                "LLM_PROVIDER=deepseek but DEEPSEEK_API_KEY is not set"
            )
        return DEEPSEEK_API_BASE, DEEPSEEK_MODEL, DEEPSEEK_API_KEY
    return OLLAMA_API_BASE, OLLAMA_MODEL, OLLAMA_API_KEY


def local_llm_callable(prompt: str, timeout: int = 60) -> str:
    """
    Call the configured LLM (Ollama locally, or a remote OpenAI-compatible
    provider like DeepSeek) through /chat/completions. Returns a JSON string.
    """
    base, model, api_key = _active_provider_config()
    url = f"{base}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"].strip()

    try:
        parsed = json.loads(content)
        return json.dumps(parsed, ensure_ascii=False)
    except Exception:
        return content


def check_llm_available() -> bool:
    """
    Provider-aware health check. Replaces the old localhost-only Ollama ping.
    - ollama: GET the base host root
    - deepseek (or any remote): presence of API key is enough; skip a network
      probe so we don't burn a request on every preflight.
    """
    if LLM_PROVIDER == "deepseek":
        return bool(DEEPSEEK_API_KEY)

    try:
        health_url = OLLAMA_API_BASE.replace("/v1", "")
        resp = requests.get(health_url, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


if __name__ == "__main__":
    test_prompt = """
请输出 JSON:
{
  "happened": "...",
  "why": "..."
}
新闻标题：Fed rate cut bets revived
新闻摘要：Odds for a reduction jumped to about 43%.
新闻来源：Reuters
""".strip()

    print(f"[provider={LLM_PROVIDER}]")
    print(local_llm_callable(test_prompt))
