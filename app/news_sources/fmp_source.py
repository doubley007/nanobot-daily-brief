from __future__ import annotations

import logging
import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

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

    summary = (item.get("text") or item.get("summary") or "").strip() or "no summary available"
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


def _fetch_fmp(
    url: str,
    label: str,
    category: str,
    limit: int,
    timeout: int,
) -> list[dict[str, Any]]:
    try:
        api_key = _get_api_key()
    except ValueError as e:
        logger.info("FMP %s skipped: %s", label, e)
        return []

    params = {"apikey": api_key, "page": 0, "limit": limit}

    try:
        response = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as e:
        logger.warning("FMP %s network error: %s", label, e)
        return []

    if response.status_code == 402:
        logger.info("FMP %s unavailable on current plan (HTTP 402) — skipping", label)
        return []
    if response.status_code == 401:
        logger.warning("FMP %s auth failed (HTTP 401) — check FMP_API_KEY", label)
        return []
    if response.status_code == 429:
        logger.warning("FMP %s rate limited (HTTP 429) — skipping", label)
        return []
    if not response.ok:
        logger.warning("FMP %s fetch failed: HTTP %s", label, response.status_code)
        return []

    try:
        data = response.json()
    except ValueError as e:
        logger.warning("FMP %s returned invalid JSON: %s", label, e)
        return []

    if not isinstance(data, list):
        return []

    results: list[dict[str, Any]] = []
    for item in data:
        normalized = _normalize_fmp_item(item, category=category)
        if normalized is not None:
            results.append(normalized)
    return results


def fetch_from_fmp_stock_news(
    limit: int = 20,
    timeout: int = 15,
) -> list[dict[str, Any]]:
    return _fetch_fmp(FMP_STOCK_NEWS_URL, "stock news", "equity", limit, timeout)


def fetch_from_fmp_general_news(
    limit: int = 20,
    timeout: int = 15,
) -> list[dict[str, Any]]:
    return _fetch_fmp(FMP_GENERAL_NEWS_URL, "general news", "general", limit, timeout)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    stock_news = fetch_from_fmp_stock_news(limit=5)
    general_news = fetch_from_fmp_general_news(limit=5)

    print(f"Stock news: {len(stock_news)}")
    for idx, item in enumerate(stock_news, start=1):
        print(f"{idx}. {item['title']}")

    print(f"\nGeneral news: {len(general_news)}")
    for idx, item in enumerate(general_news, start=1):
        print(f"{idx}. {item['title']}")
