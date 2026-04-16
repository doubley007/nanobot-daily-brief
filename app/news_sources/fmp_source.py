from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

FMP_STOCK_NEWS_URL = "https://financialmodelingprep.com/stable/news/stock"
FMP_GENERAL_NEWS_URL = "https://financialmodelingprep.com/stable/news/general-latest"


def _get_api_key() -> str:
    api_key = os.getenv("FMP_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing FMP_API_KEY in .env")
    return api_key


def _normalize_fmp_item(item: dict[str, Any], category: str) -> dict[str, Any] | None:
    title = (item.get("title") or item.get("headline") or "").strip()
    if not title:
        return None

    summary = (item.get("text") or item.get("summary") or "").strip() or "暂无摘要"
    source = (item.get("site") or item.get("source") or "Unknown").strip()
    url = item.get("url")
    published_at = item.get("publishedDate") or item.get("date")

    return {
        "title": title,
        "summary": summary,
        "source": source,
        "category": category,
        "url": url,
        "published_at": str(published_at) if published_at is not None else None,
    }


def fetch_from_fmp_stock_news(
    limit: int = 20,
    timeout: int = 15,
) -> list[dict[str, Any]]:
    try:
        api_key = _get_api_key()
    except Exception as e:
        print(f"WARNING fmp stock news skipped: {e}")
        return []

    params = {
        "apikey": api_key,
        "page": 0,
        "limit": limit,
    }

    try:
        response = requests.get(FMP_STOCK_NEWS_URL, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.HTTPError as e:
        print(f"WARNING fmp stock news fetch failed: {e}")
        return []
    except Exception as e:
        print(f"WARNING fmp stock news fetch failed: {e}")
        return []

    data = response.json()
    if not isinstance(data, list):
        return []

    results: list[dict[str, Any]] = []
    for item in data:
        normalized = _normalize_fmp_item(item, category="equity")
        if normalized is not None:
            results.append(normalized)

    return results


def fetch_from_fmp_general_news(
    limit: int = 20,
    timeout: int = 15,
) -> list[dict[str, Any]]:
    try:
        api_key = _get_api_key()
    except Exception as e:
        print(f"WARNING fmp general news skipped: {e}")
        return []

    params = {
        "apikey": api_key,
        "page": 0,
        "limit": limit,
    }

    try:
        response = requests.get(FMP_GENERAL_NEWS_URL, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.HTTPError as e:
        print(f"WARNING fmp general news fetch failed: {e}")
        return []
    except Exception as e:
        print(f"WARNING fmp general news fetch failed: {e}")
        return []

    data = response.json()
    if not isinstance(data, list):
        return []

    results: list[dict[str, Any]] = []
    for item in data:
        normalized = _normalize_fmp_item(item, category="general")
        if normalized is not None:
            results.append(normalized)

    return results


if __name__ == "__main__":
    stock_news = fetch_from_fmp_stock_news(limit=5)
    general_news = fetch_from_fmp_general_news(limit=5)

    print(f"Stock news: {len(stock_news)}")
    for idx, item in enumerate(stock_news, start=1):
        print(f"{idx}. {item['title']}")

    print(f"\nGeneral news: {len(general_news)}")
    for idx, item in enumerate(general_news, start=1):
        print(f"{idx}. {item['title']}")