"""
把 news_fetcher 抓回来的 RawNewsItem 写进 Knowledge Store。

不依赖 LLM：只做轻量 asset_tag 标注 + 情绪关键词启发式；
如果以后想上 LLM 摘要/重要性打分，在 enrich_news_doc() 接一层即可。
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Iterable

from assistant.asset_taxonomy import ASSETS
from assistant.rag.store import KnowledgeStore, NewsDoc, default_store

logger = logging.getLogger(__name__)


_BULLISH_WORDS = [
    "rally", "surge", "gain", "jump", "rise", "boost", "soar",
    "record high", "bullish", "upgrade", "beat", "breakout",
    "oversold", "accumulate", "strong buy", "beat expectations",
    "better than expected", "buying opportunity", "new high",
    "上涨", "大涨", "利好", "看多", "超预期", "抄底", "新高", "突破",
]
_BEARISH_WORDS = [
    "plunge", "drop", "fall", "decline", "slump", "tumble",
    "record low", "bearish", "downgrade", "miss", "warning", "crash",
    "overbought", "overvalued", "resistance", "correction", "extended",
    "don't chase", "wait for pullback", "take profit", "reduce exposure",
    "miss expectations", "guidance cut",
    "下跌", "大跌", "利空", "看空", "不及预期", "追高风险", "等回调", "过热",
]


def _hash_id(source: str, title: str, published_at: str | None) -> str:
    key = f"{source}::{title}::{published_at or ''}".encode("utf-8")
    return hashlib.sha1(key).hexdigest()[:20]


def _parse_ts(value: str | None) -> float:
    """news_fetcher 的 published_at 是字符串；转 unix 秒，失败就回退到现在。"""
    if not value:
        return datetime.now(tz=timezone.utc).timestamp()
    # 常见 ISO 格式，多数 feed 都是 2024-01-05T12:34:56Z 或 epoch
    try:
        if value.isdigit():
            return float(value)
        # yfinance 等常用 ISO 形式，支持 'Z' 结尾
        v = value.replace("Z", "+00:00")
        return datetime.fromisoformat(v).timestamp()
    except Exception:
        return datetime.now(tz=timezone.utc).timestamp()


# Category → default asset tags for macro news that lacks explicit asset names.
# A "rates" article about the Fed doesn't mention "gold" but is highly relevant to it.
_CATEGORY_DEFAULT_TAGS: dict[str, list[str]] = {
    "macro":          ["sp500", "gold", "usd"],
    "rates":          ["sp500", "gold", "usd", "sgd"],
    "equities":       ["sp500", "nasdaq"],
    "crypto":         ["bitcoin", "ethereum"],
    "singapore_local": ["sti", "dbs", "ocbc", "uob", "sgd"],
    "commodities":    ["gold", "oil", "silver", "copper"],
}


def detect_asset_tags(text: str, category: str = "") -> list[str]:
    tags: list[str] = []
    lower = (text or "").lower()
    for spec in ASSETS:
        if any(t.lower() in lower for t in spec.aliases + spec.keywords):
            tags.append(spec.id)
    # If no explicit asset found, fall back to category defaults
    if not tags and category:
        tags = list(_CATEGORY_DEFAULT_TAGS.get(category, []))
    return tags


def quick_sentiment(text: str) -> str:
    lower = (text or "").lower()
    bull = sum(1 for w in _BULLISH_WORDS if w in lower)
    bear = sum(1 for w in _BEARISH_WORDS if w in lower)
    if bull and bear:
        return "mixed"
    if bull > bear:
        return "bullish"
    if bear > bull:
        return "bearish"
    return "neutral"


def quick_importance(text: str) -> float:
    """非常轻量的重要性评估：命中多资产/多话题分高。"""
    score = 0.0
    lower = (text or "").lower()
    for keyword, weight in [
        ("fed", 0.2), ("rate cut", 0.3), ("rate hike", 0.3),
        ("recession", 0.25), ("war", 0.2), ("sanction", 0.2),
        ("cpi", 0.2), ("earnings", 0.1), ("default", 0.25),
        ("降息", 0.3), ("加息", 0.3), ("通胀", 0.2),
    ]:
        if keyword in lower:
            score += weight
    return min(1.0, score)


def raw_items_to_docs(raw_items: Iterable) -> list[NewsDoc]:
    """
    raw_items 期望是 news_fetcher.RawNewsItem 或任意 duck-typed 对象。
    只要有 title/summary/source/url/published_at/category 字段即可。
    """
    docs: list[NewsDoc] = []
    for item in raw_items:
        title = getattr(item, "title", "") or ""
        summary = getattr(item, "summary", "") or ""
        source = getattr(item, "source", "") or "unknown"
        url = getattr(item, "url", "") or ""
        published_raw = getattr(item, "published_at", "") or ""
        category = getattr(item, "category", "") or ""
        text = f"{title}\n{summary}"
        ts = _parse_ts(published_raw)
        asset_tags = detect_asset_tags(text, category=category)
        topic_tags = [category] if category else []

        docs.append(NewsDoc(
            id=_hash_id(source, title, published_raw),
            source=source,
            title=title,
            published_at=ts,
            raw_text=text,
            summary=summary,
            asset_tags=asset_tags,
            topic_tags=topic_tags,
            sentiment=quick_sentiment(text),
            importance_score=quick_importance(text),
            why_it_matters=summary[:200],
            url=url,
        ))
    return docs


def index_news(
    raw_items: Iterable,
    store: KnowledgeStore | None = None,
) -> int:
    store = store or default_store()
    docs = raw_items_to_docs(raw_items)
    n = store.upsert_news(docs)
    logger.info("news_indexer: wrote %d docs", n)
    return n
