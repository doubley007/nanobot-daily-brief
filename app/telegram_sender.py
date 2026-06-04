from __future__ import annotations

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

MAX_RETRIES = 3
RETRY_BASE_DELAY = 5  # seconds; delays: 5, 10, 20


def send_to_telegram(text: str, timeout: int = 15) -> dict:
    """
    Send a message to Telegram with exponential-backoff retry.

    Retries up to MAX_RETRIES times on transient failures (network errors,
    5xx server errors). Raises on permanent failures (4xx).
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not bot_token:
        raise ValueError("Missing TELEGRAM_BOT_TOKEN in .env")
    if not chat_id:
        raise ValueError("Missing TELEGRAM_CHAT_ID in .env")

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    # Telegram has a 4096-char limit per message; split if needed
    chunks = _split_message(text, max_len=4000)
    last_result = {}

    for chunk in chunks:
        payload = {"chat_id": chat_id, "text": chunk}
        last_result = _send_with_retry(url, payload, timeout)

    return last_result


def _send_with_retry(url: str, payload: dict, timeout: int) -> dict:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            # Permanent client error — no point retrying
            if 400 <= response.status_code < 500:
                response.raise_for_status()
            # Server error — worth retrying
            if response.status_code >= 500:
                raise requests.exceptions.HTTPError(
                    f"Server error {response.status_code}", response=response
                )
            return response.json()
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError) as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(f"Telegram send attempt {attempt} failed: {e}. "
                      f"Retrying in {delay}s...")
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _split_message(text: str, max_len: int = 4000) -> list[str]:
    """Split long messages into chunks that fit Telegram's 4096-char limit."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # Try to split at a newline near the limit
        split_pos = text.rfind("\n", 0, max_len)
        if split_pos == -1:
            split_pos = max_len
        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip("\n")
    return chunks


if __name__ == "__main__":
    result = send_to_telegram("Test: Telegram send succeeded.")
    print(result)
