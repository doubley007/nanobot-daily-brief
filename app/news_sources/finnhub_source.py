from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

FINNHUB_BASE_URL = "https://finnhub.io/api/v1/news"
DEFAULT_CATEGORY = "general"


def _get_api_key() -> str:
    api_key = os.getenv("FINNHUB_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing FINNHUB_API_KEY in .env")
    return api_key


def fetch_from_finnhub(
    category: str = DEFAULT_CATEGORY,
    limit: int = 20,
    timeout: int = 15,
) -> list[dict[str, Any]]:
    """
    Return a normalized list of dict items:
    {
        "title": str,
        "summary": str,
        "source": str,
        "category": str,
        "url": str | None,
        "published_at": str | None,
    }
    """
    api_key = _get_api_key()

    params = {
        "category": category,
        "token": api_key,
    }

    response = requests.get(FINNHUB_BASE_URL, params=params, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    if not isinstance(data, list):
        return []

    results: list[dict[str, Any]] = []
    for item in data[:limit]:
        title = (item.get("headline") or "").strip()
        if not title:
            continue

        summary = (item.get("summary") or "").strip() or "no summary available"
        source = (item.get("source") or "Unknown").strip()
        url = item.get("url")
        published_at = item.get("datetime")

        results.append(
            {
                "title": title,
                "summary": summary,
                "source": source,
                "category": "general",
                "url": url,
                "published_at": str(published_at) if published_at is not None else None,
            }
        )

    return results


if __name__ == "__main__":
    try:
        news = fetch_from_finnhub(limit=5)
        print(f"Fetched {len(news)} items from Finnhub.")
        for idx, item in enumerate(news, start=1):
            print(f"{idx}. {item['title']}")
            print(f"   source={item['source']}")
            print(f"   summary={item['summary'][:120]}")
            print()
    except Exception as e:
        print(f"ERROR: {e}")