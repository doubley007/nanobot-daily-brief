"""投资场景情绪聚合器单测。"""
from __future__ import annotations

import time

from assistant.rag.store import CommunityDoc
from assistant.sentiment_aggregator import aggregate


def _doc(idx: int, bb: str, emotion: str) -> CommunityDoc:
    return CommunityDoc(
        id=f"c{idx}", platform="reddit", channel_or_group="wsb",
        author="", published_at=time.time(),
        raw_text="gold looks good", normalized_text="gold",
        summary="",
        asset_tags=["gold"],
        bullish_bearish_label=bb, emotion_label=emotion,
        confidence=0.8, engagement_score=10, url="",
    )


def test_empty_docs_neutral():
    agg = aggregate([], asset="gold")
    assert agg.post_count == 0
    assert agg.overall_bias == "neutral"


def test_bullish_majority():
    docs = ([_doc(i, "bullish", "bullish_optimism") for i in range(7)]
            + [_doc(i + 7, "bearish", "bearish_panic") for i in range(2)]
            + [_doc(i + 9, "neutral", "neutral") for i in range(1)])
    agg = aggregate(docs, asset="gold")
    assert agg.overall_bias == "bullish"
    assert agg.bullish_ratio > agg.bearish_ratio


def test_fomo_implies_bullish_when_bb_neutral():
    """如果 bullish_bearish_label 是 neutral，但情绪是 fomo_chasing，
    聚合器应视为多头。"""
    docs = [_doc(i, "neutral", "fomo_chasing") for i in range(8)]
    agg = aggregate(docs, asset="gold")
    assert agg.overall_bias == "bullish"
    assert agg.fomo_ratio == 1.0


def test_crowded_trade_risk_detection():
    docs = ([_doc(i, "bullish", "bullish_optimism") for i in range(7)]
            + [_doc(i + 7, "neutral", "fomo_chasing") for i in range(3)])
    agg = aggregate(docs, asset="gold")
    # 10 条里 7 bullish + 3 fomo（fomo 也算 bullish）-> 100% bull, 30% fomo
    assert agg.crowded_trade_risk in ("medium", "high")


def test_panic_detected():
    docs = [_doc(i, "bearish", "bearish_panic") for i in range(5)]
    agg = aggregate(docs, asset="gold")
    assert agg.overall_bias == "bearish"
    assert agg.panic_ratio >= 0.8
