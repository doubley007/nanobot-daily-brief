"""RAG store 读写 + 检索 基本正确性测试。"""
from __future__ import annotations

import time

from assistant.rag.store import NewsDoc, CommunityDoc, default_store
from assistant.rag.retriever import Retriever


def _mk_news(idx: int, asset: str, hours_ago: float,
             title: str = "Gold rally") -> NewsDoc:
    return NewsDoc(
        id=f"n{idx}", source="test", title=title,
        published_at=time.time() - hours_ago * 3600,
        raw_text=title, summary=title,
        asset_tags=[asset], topic_tags=[],
        sentiment="bullish", importance_score=0.5,
        why_it_matters="", url="",
    )


def _mk_community(idx: int, asset: str, hours_ago: float,
                  text: str = "gold looks good") -> CommunityDoc:
    return CommunityDoc(
        id=f"c{idx}", platform="reddit", channel_or_group="wsb",
        author="", published_at=time.time() - hours_ago * 3600,
        raw_text=text, normalized_text=text.lower(), summary=text,
        asset_tags=[asset],
        bullish_bearish_label="bullish", emotion_label="bullish_optimism",
        confidence=0.8, engagement_score=50, url="",
    )


def test_upsert_and_count():
    store = default_store()
    store.upsert_news([_mk_news(1, "gold", 2)])
    store.upsert_community([_mk_community(1, "gold", 3)])
    counts = store.count()
    assert counts["news"] == 1
    assert counts["community"] == 1


def test_upsert_is_idempotent():
    store = default_store()
    store.upsert_news([_mk_news(1, "gold", 2)])
    store.upsert_news([_mk_news(1, "gold", 2)])  # same id
    assert store.count()["news"] == 1


def test_retriever_window_filter():
    store = default_store()
    store.upsert_news([
        _mk_news(1, "gold", hours_ago=2, title="Gold rally today"),
        _mk_news(2, "gold", hours_ago=400, title="Old gold story"),   # out of 72h
    ])
    retriever = Retriever(store=store)
    news = retriever.retrieve_news(asset="gold", window_hours=72, top_k=10)
    titles = [n.title for n in news]
    assert "Gold rally today" in titles
    assert "Old gold story" not in titles


def test_retriever_asset_keyword_fallback():
    """没打 asset_tag 也能被关键词命中。"""
    store = default_store()
    doc = _mk_news(1, "other", hours_ago=2, title="Gold jumps on Fed bets")
    doc.asset_tags = []          # deliberately missing
    store.upsert_news([doc])
    news = Retriever(store=store).retrieve_news(asset="gold", window_hours=72)
    assert len(news) == 1
