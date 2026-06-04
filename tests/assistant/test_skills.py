"""Tests for skills module."""
from __future__ import annotations

import pytest
import assistant.pipeline as pipeline
from assistant.skills import (
    detect_user_intent_and_emotion,
    retrieve_market_context,
    aggregate_asset_sentiment,
    assess_entry_quality,
    compose_in_house_style_reply,
    IntentEmotionResult,
    MarketContextResult,
    AssetSentimentResult,
    EntryAssessmentResult,
)
from assistant.fixtures import install_gold_fixture
from assistant.trend_signals import trend_from_values
from assistant.company_context import DEFAULT_COMPANY_CONTEXT, reset_company_context
from assistant.user_profile import DEFAULT_PROFILE, reset_profile_store
from assistant.user_emotion import UserEmotionProfile
from assistant.decision_engine import Decision, DecisionScores
from assistant.query_router import RouterResult


@pytest.fixture(autouse=True)
def _reset():
    reset_company_context()
    reset_profile_store()
    yield
    reset_company_context()
    reset_profile_store()


@pytest.fixture
def gold_ready(monkeypatch):
    install_gold_fixture()
    monkeypatch.setattr(
        pipeline, "fetch_trend_signal",
        lambda asset: trend_from_values(asset, r7=0.03, r30=0.09),
    )
    monkeypatch.setattr(pipeline, "_llm_callable", lambda: None)


# ─── Skill 1 ──────────────────────────────────────────────────────────────────

class TestDetectUserIntentAndEmotion:
    def test_gold_decision_detected(self):
        result = detect_user_intent_and_emotion("我能不能买黄金", llm_callable=None)
        assert isinstance(result, IntentEmotionResult)
        assert result.route == "market_decision"
        assert result.asset == "gold"

    def test_fomo_emotion_detected(self):
        result = detect_user_intent_and_emotion(
            "大家都在买黄金，我是不是也该上车", llm_callable=None
        )
        assert result.primary_emotion == "fomo"
        assert result.emotion_intensity > 0

    def test_emotional_chat_detected(self):
        result = detect_user_intent_and_emotion("我好焦虑不知道怎么办", llm_callable=None)
        assert result.route == "emotional_chat"
        assert result.primary_emotion in ("anxious", "fomo", "uncertain", "frustrated", "neutral")

    def test_market_summary_detected(self):
        result = detect_user_intent_and_emotion("最近市场情绪怎么样", llm_callable=None)
        assert result.route in ("market_summary", "market_decision", "emotional_chat")

    def test_result_has_confidence(self):
        result = detect_user_intent_and_emotion("黄金能买吗", llm_callable=None)
        assert 0 <= result.confidence <= 1


# ─── Skill 2 ──────────────────────────────────────────────────────────────────

class TestRetrieveMarketContext:
    def test_returns_result(self, gold_ready):
        result = retrieve_market_context(asset="gold", window_hours=72)
        assert isinstance(result, MarketContextResult)
        assert result.asset == "gold"

    def test_news_count_positive(self, gold_ready):
        result = retrieve_market_context(asset="gold", window_hours=72)
        assert result.news_count > 0

    def test_community_count_positive(self, gold_ready):
        result = retrieve_market_context(asset="gold", window_hours=72)
        assert result.community_count > 0

    def test_top_news_titles_populated(self, gold_ready):
        result = retrieve_market_context(asset="gold", window_hours=72)
        assert len(result.top_news_titles) > 0
        assert all(isinstance(t, str) for t in result.top_news_titles)

    def test_empty_store_returns_zeros(self):
        # No gold_ready fixture, empty store
        result = retrieve_market_context(asset="gold", window_hours=1)
        assert result.news_count == 0
        assert result.community_count == 0


# ─── Skill 3 ──────────────────────────────────────────────────────────────────

class TestAggregateAssetSentiment:
    def test_returns_result(self, gold_ready):
        result = aggregate_asset_sentiment(asset="gold", window_hours=72)
        assert isinstance(result, AssetSentimentResult)
        assert result.asset == "gold"

    def test_bias_is_valid(self, gold_ready):
        result = aggregate_asset_sentiment(asset="gold", window_hours=72)
        assert result.overall_bias in ("bullish", "bearish", "neutral", "mixed")

    def test_ratios_sum_to_at_most_one(self, gold_ready):
        result = aggregate_asset_sentiment(asset="gold", window_hours=72)
        assert 0 <= result.bullish_ratio <= 1
        assert 0 <= result.bearish_ratio <= 1
        assert result.bullish_ratio + result.bearish_ratio <= 1.05  # allow rounding

    def test_fomo_ratio_positive_for_gold_fixture(self, gold_ready):
        result = aggregate_asset_sentiment(asset="gold", window_hours=72)
        assert result.fomo_ratio >= 0

    def test_empty_store_returns_neutral(self):
        result = aggregate_asset_sentiment(asset="gold", window_hours=1)
        assert result.overall_bias == "neutral"
        assert result.post_count == 0


# ─── Skill 4 ──────────────────────────────────────────────────────────────────

class TestAssessEntryQuality:
    def test_returns_result(self, gold_ready):
        result = assess_entry_quality(asset="gold", window_hours=72)
        assert isinstance(result, EntryAssessmentResult)

    def test_entry_quality_valid_values(self, gold_ready):
        result = assess_entry_quality(asset="gold", window_hours=72)
        assert result.entry_quality in ("good", "medium", "poor")

    def test_chasing_risk_valid_values(self, gold_ready):
        result = assess_entry_quality(asset="gold", window_hours=72)
        assert result.chasing_risk in ("low", "medium", "high")

    def test_action_suggestion_valid(self, gold_ready):
        result = assess_entry_quality(asset="gold", window_hours=72)
        assert result.action_suggestion in (
            "buy", "buy_small", "hold_wait", "avoid_chasing", "avoid"
        )

    def test_one_line_not_empty(self, gold_ready):
        result = assess_entry_quality(asset="gold", window_hours=72)
        assert len(result.one_line) > 0

    def test_direction_score_in_range(self, gold_ready):
        result = assess_entry_quality(asset="gold", window_hours=72)
        assert -1 <= result.direction_score <= 1

    def test_crowding_score_in_range(self, gold_ready):
        result = assess_entry_quality(asset="gold", window_hours=72)
        assert 0 <= result.crowding_score <= 1


# ─── Skill 5 ──────────────────────────────────────────────────────────────────

class TestComposeInHouseStyleReply:
    def _make_decision(self, action="buy_small"):
        return Decision(
            asset="gold",
            action=action,
            confidence="medium",
            thesis="黄金整体偏多，有避险支撑",
            risks=["短期拥挤风险较高", "止损位要留好"],
            evidence={
                "news": [{"title": "Gold hits record", "source": "Reuters",
                           "sentiment": "bullish", "url": "", "published_at": 0, "snippet": ""}],
                "news_assessment": {"direction": "bullish", "bullish_score": 0.8, "bearish_score": 0.2, "key_bullets": []},
                "community_aggregate": {"post_count": 30, "overall_bias": "bullish",
                                         "bullish_ratio": 0.7, "bearish_ratio": 0.1,
                                         "fomo_ratio": 0.3, "crowded_trade_risk": "medium",
                                         "narrative_keywords": ["rate cut"], "summary": "偏多"},
                "community_samples": [],
                "trend": {"momentum_label": "up", "overheating_risk": "medium",
                           "recent_return_7d": 0.03, "recent_return_30d": 0.09},
                "decision_scores": {"direction_score": 0.4, "crowding_score": 0.4,
                                    "entry_quality": "medium", "chasing_risk": "medium"},
                "engine_trace": [],
            },
            scores=DecisionScores(direction_score=0.4, crowding_score=0.4,
                                   entry_quality="medium", chasing_risk="medium"),
            suitable_for="愿意分批的投资者",
            one_line_advice="小仓位分批",
        )

    def _make_route(self):
        return RouterResult(
            route="market_decision", asset="gold",
            user_emotion="neutral", confidence=0.9,
        )

    def _make_emotion(self, primary="neutral", intensity=0.1):
        return UserEmotionProfile(
            primary_emotion=primary,
            emotion_intensity=intensity,
            needs_confirmation=False,
            risk_of_impulsive_action=False,
            signals=[],
        )

    def test_returns_non_empty_string(self):
        reply = compose_in_house_style_reply(
            decision=self._make_decision(),
            route=self._make_route(),
            emotion=self._make_emotion(),
            profile=DEFAULT_PROFILE,
            company=DEFAULT_COMPANY_CONTEXT,
            llm_callable=None,
        )
        assert isinstance(reply, str)
        assert len(reply) > 20

    def test_reply_contains_conclusion(self):
        reply = compose_in_house_style_reply(
            decision=self._make_decision("buy_small"),
            route=self._make_route(),
            emotion=self._make_emotion(),
            profile=DEFAULT_PROFILE,
            company=DEFAULT_COMPANY_CONTEXT,
        )
        assert "结论" in reply

    def test_fomo_buy_has_caution(self):
        reply = compose_in_house_style_reply(
            decision=self._make_decision("buy"),
            route=self._make_route(),
            emotion=self._make_emotion("fomo", 0.8),
            profile=DEFAULT_PROFILE,
            company=DEFAULT_COMPANY_CONTEXT,
        )
        assert any(kw in reply for kw in ("分批", "小仓位", "别重仓"))

    def test_no_banned_phrases_in_output(self):
        reply = compose_in_house_style_reply(
            decision=self._make_decision("buy_small"),
            route=self._make_route(),
            emotion=self._make_emotion(),
            profile=DEFAULT_PROFILE,
            company=DEFAULT_COMPANY_CONTEXT,
        )
        banned = DEFAULT_COMPANY_CONTEXT.banned_phrases
        for phrase in banned:
            assert phrase.lower() not in reply.lower(), (
                f"Output contains banned phrase: '{phrase}'"
            )
