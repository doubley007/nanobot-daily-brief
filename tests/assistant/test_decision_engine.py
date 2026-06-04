"""决策引擎核心逻辑测试 —— 不依赖 LLM，只验证骨架规则。"""
from __future__ import annotations

import time

from assistant.decision_engine import make_decision, Decision
from assistant.rag.store import NewsDoc, CommunityDoc
from assistant.sentiment_aggregator import aggregate, AggregatedSentiment
from assistant.trend_signals import trend_from_values


def _news(idx: int, title: str, sent: str = "bullish",
          importance: float = 0.5) -> NewsDoc:
    return NewsDoc(
        id=f"n{idx}", source="t", title=title,
        published_at=time.time(), raw_text=title, summary=title,
        asset_tags=["gold"], topic_tags=[],
        sentiment=sent, importance_score=importance,
        why_it_matters="", url="",
    )


def _com(idx: int, bb: str = "bullish",
         emotion: str = "bullish_optimism") -> CommunityDoc:
    return CommunityDoc(
        id=f"c{idx}", platform="reddit", channel_or_group="wsb",
        author="", published_at=time.time(),
        raw_text="gold is going up", normalized_text="gold",
        summary="",
        asset_tags=["gold"],
        bullish_bearish_label=bb, emotion_label=emotion,
        confidence=0.8, engagement_score=10, url="",
    )


def test_no_evidence_returns_hold_wait():
    """彻底没证据时必须 hold_wait 而非瞎猜。"""
    d = make_decision(
        asset="gold", news=[], community_docs=[],
        trend=trend_from_values("gold", r7=None, r30=None),
        sentiment_aggregate=aggregate([], asset="gold"),
    )
    assert d.action == "hold_wait"
    assert d.confidence == "low"


def test_strong_bull_returns_buy():
    news = [_news(i, "Rate cut bets strengthen, gold rally") for i in range(5)]
    docs = [_com(i) for i in range(15)]
    d = make_decision(
        asset="gold", news=news, community_docs=docs,
        trend=trend_from_values("gold", r7=0.02, r30=0.04),
    )
    assert d.action in ("buy", "buy_small")


def test_crowded_downgrade():
    """情绪极端一致 + 过热 -> 即使 news bullish 也只能 buy_small 或 avoid_chasing。"""
    news = [_news(i, "Gold record high, rate cut bets") for i in range(5)]
    docs = ([_com(i, "bullish", "fomo_chasing") for i in range(8)]
            + [_com(i + 8, "bullish", "strong_conviction") for i in range(6)])
    agg = aggregate(docs, asset="gold")
    d = make_decision(
        asset="gold", news=news, community_docs=docs,
        trend=trend_from_values("gold", r7=0.06, r30=0.15),   # 过热
        sentiment_aggregate=agg,
    )
    assert d.action in ("buy_small", "avoid_chasing")


def test_bearish_news_vs_bullish_crowd_divergence():
    """新闻偏空 + 社区疯狂看多 -> confidence 应该降到 low。"""
    news = [_news(i, "Stronger dollar pressures gold, rate hike bets",
                  sent="bearish", importance=0.5) for i in range(4)]
    docs = [_com(i, "bullish", "fomo_chasing") for i in range(10)]
    d = make_decision(
        asset="gold", news=news, community_docs=docs,
        trend=trend_from_values("gold", r7=-0.02, r30=-0.01),
    )
    assert d.confidence == "low"


def test_decision_has_evidence_fields():
    news = [_news(1, "Rate cut, gold rally")]
    docs = [_com(i) for i in range(12)]
    d = make_decision(
        asset="gold", news=news, community_docs=docs,
        trend=trend_from_values("gold", r7=0.02, r30=0.04),
    )
    assert isinstance(d, Decision)
    assert "news" in d.evidence
    assert "community_aggregate" in d.evidence
    assert "trend" in d.evidence
    assert d.thesis
    assert d.one_line_advice
    assert d.risks
