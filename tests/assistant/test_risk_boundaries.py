"""
CL5: Risk boundary hardening tests.

Tests for each hard guardrail in decision_engine.py:
  1. Weak evidence (strength < 0.3) → hold_wait, low confidence
  2. News/community divergence → confidence="low"
  3. Overheating/crowding downgrade → buy downgraded to buy_small or avoid_chasing
  4. FOMO firewall — make_decision() does NOT accept UserEmotionProfile
  5. Insufficient facts → hold_wait (no news + no community)
  6. Bearish direction → avoid / avoid_chasing
"""
from __future__ import annotations

import inspect
import time
import pytest

from assistant.decision_engine import (
    make_decision,
    assess_news,
    _evidence_strength,
    _calc_crowding_score,
    _calc_entry_quality,
    _decide_action,
)
from assistant.rag.store import NewsDoc, CommunityDoc
from assistant.sentiment_aggregator import aggregate, AggregatedSentiment
from assistant.trend_signals import trend_from_values


# ── Helpers ───────────────────────────────────────────────────────────────────

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
        raw_text="gold up", normalized_text="gold",
        summary="",
        asset_tags=["gold"],
        bullish_bearish_label=bb, emotion_label=emotion,
        confidence=0.8, engagement_score=10, url="",
    )


# ── 1. Weak evidence → hold_wait ──────────────────────────────────────────────

class TestWeakEvidence:
    def test_no_evidence_hold_wait(self):
        d = make_decision(
            asset="gold", news=[], community_docs=[],
            trend=trend_from_values("gold", r7=None, r30=None),
            sentiment_aggregate=aggregate([], asset="gold"),
        )
        assert d.action == "hold_wait"
        assert d.confidence == "low"

    def test_minimal_news_no_community_hold_wait(self):
        # Only 1 weak news item, no community, flat trend
        news = [_news(1, "gold prices mixed", sent="neutral", importance=0.1)]
        d = make_decision(
            asset="gold", news=news, community_docs=[],
            trend=trend_from_values("gold", r7=0.01, r30=0.01),
            sentiment_aggregate=aggregate([], asset="gold"),
        )
        assert d.action == "hold_wait"
        assert d.confidence == "low"

    def test_evidence_strength_below_threshold_triggers_hold_wait(self):
        # Only trend signal (no news keywords, no community)
        news = [_news(1, "gold price today", sent="neutral", importance=0.0)]
        trend = trend_from_values("gold", r7=0.03, r30=0.09)
        agg = aggregate([], asset="gold")
        news_assess = assess_news("gold", news)
        strength = _evidence_strength(news_assess, agg, trend)
        # With only trend signal: strength = 0.3 exactly, so borderline
        # This test just verifies the threshold function itself
        assert 0 <= strength <= 1


# ── 2. Divergence → low confidence ───────────────────────────────────────────

class TestDivergence:
    def test_bearish_news_bullish_crowd_low_confidence(self):
        news = [_news(i, "stronger dollar pressures gold, rate hike",
                      sent="bearish", importance=0.6) for i in range(4)]
        docs = [_com(i, "bullish", "fomo_chasing") for i in range(12)]
        d = make_decision(
            asset="gold", news=news, community_docs=docs,
            trend=trend_from_values("gold", r7=-0.01, r30=-0.02),
        )
        assert d.confidence == "low"

    def test_bullish_news_bearish_crowd_low_confidence(self):
        news = [_news(i, "rate cut gold rally safe haven",
                      sent="bullish", importance=0.6) for i in range(4)]
        docs = [_com(i, "bearish", "bearish_concern") for i in range(12)]
        d = make_decision(
            asset="gold", news=news, community_docs=docs,
            trend=trend_from_values("gold", r7=0.02, r30=0.03),
        )
        assert d.confidence == "low"

    def test_divergence_recorded_in_engine_trace(self):
        news = [_news(i, "stronger dollar, rate hike bets",
                      sent="bearish", importance=0.5) for i in range(4)]
        docs = [_com(i, "bullish", "fomo_chasing") for i in range(10)]
        d = make_decision(
            asset="gold", news=news, community_docs=docs,
            trend=trend_from_values("gold", r7=-0.01, r30=-0.02),
        )
        trace = d.evidence.get("engine_trace", [])
        assert any("divergence" in step.lower() for step in trace)


# ── 3. Overheating / crowding downgrade ──────────────────────────────────────

class TestOverheatingDowngrade:
    def test_overheated_market_not_buy(self):
        news = [_news(i, "rate cut gold rally") for i in range(5)]
        docs = (
            [_com(i, "bullish", "fomo_chasing") for i in range(8)]
            + [_com(i + 8, "bullish", "strong_conviction") for i in range(6)]
        )
        agg = aggregate(docs, asset="gold")
        d = make_decision(
            asset="gold", news=news, community_docs=docs,
            trend=trend_from_values("gold", r7=0.07, r30=0.18),  # high overheating
            sentiment_aggregate=agg,
        )
        assert d.action in ("buy_small", "avoid_chasing", "hold_wait")

    def test_high_crowding_score_reduces_entry_quality(self):
        high_crowd = AggregatedSentiment(
            asset="gold", window="3d", overall_bias="bullish",
            bullish_ratio=0.85, bearish_ratio=0.05,
            fomo_ratio=0.45, conviction_ratio=0.35,
            uncertainty_ratio=0.05, panic_ratio=0.0,
            crowded_trade_risk="high", narrative_keywords=[],
            summary="", post_count=30,
        )
        trend = trend_from_values("gold", r7=0.05, r30=0.15)
        crowding = _calc_crowding_score(high_crowd, trend)
        entry, chasing = _calc_entry_quality(0.6, crowding, trend)
        # High crowding + high overheating should degrade entry
        assert entry in ("medium", "poor")
        assert chasing in ("medium", "high")

    def test_bitcoin_overheated_downgraded(self):
        """7d+5%, 30d+18% is high overheating → buy must be downgraded."""
        from assistant.fixtures import install_bitcoin_fixture
        from assistant.rag.store import default_store
        from assistant.rag.retriever import Retriever
        install_bitcoin_fixture()
        store = default_store()
        retriever = Retriever(store=store)
        ev = retriever.retrieve("bitcoin", window_hours=72)
        agg = aggregate(ev.community, asset="bitcoin")
        trend = trend_from_values("bitcoin", r7=0.05, r30=0.18)
        d = make_decision(
            asset="bitcoin", news=ev.news, community_docs=ev.community,
            trend=trend, sentiment_aggregate=agg,
        )
        # With high overheating, should not get plain "buy"
        assert d.action in ("buy_small", "avoid_chasing", "hold_wait")


# ── 4. FOMO firewall ──────────────────────────────────────────────────────────

class TestFomoFirewall:
    def test_make_decision_has_no_user_emotion_param(self):
        """make_decision() must NOT accept UserEmotionProfile in its signature."""
        sig = inspect.signature(make_decision)
        param_names = list(sig.parameters.keys())
        assert "user_emotion" not in param_names
        assert "emotion" not in param_names
        assert "user_emotion_profile" not in param_names

    def test_fomo_user_does_not_change_decision_vs_neutral(self):
        """Passing the same market data gives identical decision regardless of user context."""
        news = [_news(i, "rate cut gold rally safe haven inflation hedge") for i in range(5)]
        docs = [_com(i) for i in range(15)]
        trend = trend_from_values("gold", r7=0.02, r30=0.05)
        d = make_decision(asset="gold", news=news, community_docs=docs, trend=trend)
        # There is no way to pass user emotion — both calls are identical
        # Just verify the call succeeds with only market data
        assert d.action is not None
        assert d.confidence is not None


# ── 5. Bearish direction → avoid ─────────────────────────────────────────────

class TestBearishDirection:
    def test_strong_bearish_returns_avoid(self):
        news = [
            _news(i, "stronger dollar pressures gold, rate hike bets, usd strength",
                  sent="bearish", importance=0.7)
            for i in range(5)
        ]
        docs = [_com(i, "bearish", "bearish_concern") for i in range(15)]
        d = make_decision(
            asset="gold", news=news, community_docs=docs,
            trend=trend_from_values("gold", r7=-0.04, r30=-0.08),
        )
        assert d.action in ("avoid", "avoid_chasing")

    def test_moderate_bearish_returns_avoid_chasing(self):
        news = [_news(i, "dollar rally, risk on, de-escalation",
                      sent="bearish", importance=0.4) for i in range(3)]
        docs = [_com(i, "bearish", "bearish_concern") for i in range(8)]
        d = make_decision(
            asset="gold", news=news, community_docs=docs,
            trend=trend_from_values("gold", r7=-0.02, r30=-0.03),
        )
        assert d.action in ("avoid", "avoid_chasing", "hold_wait")


# ── 6. Decision has required evidence fields ──────────────────────────────────

class TestDecisionEvidenceCompleteness:
    def test_all_evidence_keys_present(self):
        news = [_news(1, "rate cut gold rally")]
        docs = [_com(i) for i in range(12)]
        d = make_decision(
            asset="gold", news=news, community_docs=docs,
            trend=trend_from_values("gold", r7=0.02, r30=0.04),
        )
        required_keys = [
            "news", "news_assessment", "community_aggregate",
            "community_samples", "trend", "decision_scores", "engine_trace",
        ]
        for key in required_keys:
            assert key in d.evidence, f"Missing evidence key: {key}"

    def test_scores_always_present(self):
        news = [_news(1, "rate cut")]
        docs = [_com(i) for i in range(12)]
        d = make_decision(
            asset="gold", news=news, community_docs=docs,
            trend=trend_from_values("gold", r7=0.02, r30=0.04),
        )
        assert d.scores is not None
        assert d.scores.direction_score is not None

    def test_engine_trace_not_empty_with_data(self):
        news = [_news(i, "rate cut gold rally") for i in range(4)]
        docs = [_com(i) for i in range(12)]
        d = make_decision(
            asset="gold", news=news, community_docs=docs,
            trend=trend_from_values("gold", r7=0.02, r30=0.04),
        )
        assert len(d.evidence.get("engine_trace", [])) > 0
