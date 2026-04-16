from __future__ import annotations

import json
import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "dummy")


def local_llm_callable(prompt: str, timeout: int = 60) -> str:
    """
    Call local Ollama through its OpenAI-compatible /chat/completions endpoint.
    Must return a JSON string like:
    {"happened": "...", "why": "..."}
    """
    url = f"{OLLAMA_API_BASE}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OLLAMA_API_KEY}",
    }

    payload: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一个金融日报编辑。"
                    "你必须输出合法 JSON，且只输出 JSON。"
                    "不要输出 markdown，不要输出额外解释。"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"].strip()

    # 尝试确保返回的是 JSON 字符串
    try:
        parsed = json.loads(content)
        return json.dumps(parsed, ensure_ascii=False)
    except Exception:
        # 给上层 fallback 机制处理
        return content


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

    print(local_llm_callable(test_prompt))