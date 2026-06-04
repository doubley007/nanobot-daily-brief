from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
DEFAULT_TOPICS = "economy_macro,financial_markets,finance,real_estate,manufacturing"


def _get_api_key() -> str:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing ALPHA_VANTAGE_API_KEY in .env")
    return api_key


def _normalize_alpha_item(item: dict[str, Any]) -> dict[str, Any] | None:
    title = (item.get("title") or "").strip()
    if not title:
        return None

    summary = (item.get("summary") or "").strip() or "no summary available"
    source = (item.get("source") or "Alpha Vantage").strip()
    url = item.get("url")
    published_at = item.get("time_published")

    # Alpha Vantage provides overall_sentiment_label — pass it through so
    # risk_monitor can skip the LLM call for AV articles.
    av_sentiment = (item.get("overall_sentiment_label") or "").strip().upper()
    # Normalize to NEGATIVE / POSITIVE / NEUTRAL
    if av_sentiment in ("Bearish", "Somewhat-Bearish"):
        av_sentiment = "NEGATIVE"
    elif av_sentiment in ("Bullish", "Somewhat-Bullish"):
        av_sentiment = "POSITIVE"
    else:
        av_sentiment = "NEUTRAL"

    topics = item.get("topics") or []
    topic_names = []
    if isinstance(topics, list):
        for t in topics:
            if isinstance(t, dict):
                name = (t.get("topic") or "").strip().lower()
                if name:
                    topic_names.append(name)

    joined_topics = " ".join(topic_names)

    if any(k in joined_topics for k in ["economy", "macro", "financial_markets", "finance", "real_estate"]):
        category = "macro"
    elif "earnings" in joined_topics:
        category = "equity"
    else:
        category = "general"

    return {
        "title": title,
        "summary": summary,
        "source": source,
        "category": category,
        "url": url,
        "published_at": str(published_at) if published_at is not None else None,
        "av_sentiment": av_sentiment,
    }


def fetch_from_alpha_vantage(
    limit: int = 20,
    timeout: int = 20,
    topics: str = DEFAULT_TOPICS,
) -> list[dict[str, Any]]:
    """
    Use Alpha Vantage NEWS_SENTIMENT endpoint.
    Return normalized dict items.
    """
    try:
        api_key = _get_api_key()
    except Exception as e:
        print(f"WARNING alpha vantage skipped: {e}")
        return []

    params = {
        "function": "NEWS_SENTIMENT",
        "topics": topics,
        "limit": limit,
        "apikey": api_key,
    }

    try:
        response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.HTTPError as e:
        print(f"WARNING alpha vantage fetch failed: {e}")
        return []
    except Exception as e:
        print(f"WARNING alpha vantage fetch failed: {e}")
        return []

    data = response.json()
    feed = data.get("feed", [])
    if not isinstance(feed, list):
        return []

    results: list[dict[str, Any]] = []
    for item in feed:
        normalized = _normalize_alpha_item(item)
        if normalized is not None:
            results.append(normalized)

    return results


if __name__ == "__main__":
    news = fetch_from_alpha_vantage(limit=5)
    print(f"Fetched {len(news)} items from Alpha Vantage.")
    for idx, item in enumerate(news, start=1):
        print(f"{idx}. {item['title']}")
        print(f"   source={item['source']}")
        print(f"   category={item['category']}")
        print(f"   summary={item['summary'][:120]}")
        print()