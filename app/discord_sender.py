"""
Minimal Discord webhook sender for the self-curated #prediction-markets feed.

Reads DISCORD_PREDICTION_MARKETS_WEBHOOK_URL from .env and POSTs plain text
messages to it. Kept deliberately narrow — this module owns the transport;
feed-building / formatting lives elsewhere.

Usage:
    from discord_sender import send_to_discord
    send_to_discord("hello from nanobot")
"""
from __future__ import annotations

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

MAX_RETRIES = 3
RETRY_BASE_DELAY = 5  # seconds; delays: 5, 10, 20
DISCORD_CONTENT_LIMIT = 2000  # Discord hard cap per message


def send_to_discord(content: str, timeout: int = 15) -> dict:
    """
    Send a plain-text message to the configured Discord webhook.

    Discord enforces a 2000-char limit per message, so long payloads are
    split on newlines where possible. Retries transient failures with
    exponential backoff; raises on permanent 4xx.
    """
    webhook_url = os.getenv("DISCORD_PREDICTION_MARKETS_WEBHOOK_URL", "").strip()
    if not webhook_url:
        raise ValueError(
            "Missing DISCORD_PREDICTION_MARKETS_WEBHOOK_URL in .env"
        )

    chunks = _split_message(content, max_len=DISCORD_CONTENT_LIMIT - 50)
    last_result: dict = {}
    for chunk in chunks:
        last_result = _send_with_retry(webhook_url, {"content": chunk}, timeout)
    return last_result


def _send_with_retry(url: str, payload: dict, timeout: int) -> dict:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # `wait=true` makes Discord return the created message JSON
            response = requests.post(
                url, json=payload, params={"wait": "true"}, timeout=timeout
            )
            if 400 <= response.status_code < 500:
                response.raise_for_status()
            if response.status_code >= 500:
                raise requests.exceptions.HTTPError(
                    f"Server error {response.status_code}", response=response
                )
            # Webhooks without wait=true return 204 No Content; guard the json()
            if response.status_code == 204 or not response.content:
                return {"status": response.status_code}
            return response.json()
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError) as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(f"Discord send attempt {attempt} failed: {e}. "
                      f"Retrying in {delay}s...")
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _split_message(text: str, max_len: int) -> list[str]:
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        split_pos = text.rfind("\n", 0, max_len)
        if split_pos == -1:
            split_pos = max_len
        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip("\n")
    return chunks


if __name__ == "__main__":
    result = send_to_discord("prediction markets feed — webhook test ✅")
    print(result)
