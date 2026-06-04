"""Tests for reply_policy module."""
from __future__ import annotations

import pytest
from assistant.reply_policy import (
    check_reply_policy,
    apply_style_constraints,
    get_style_examples_for_prompt,
    STYLE_EXAMPLES,
    PolicyResult,
)
from assistant.company_context import CompanyContext, reset_company_context
from assistant.user_profile import UserProfile
from assistant.user_emotion import UserEmotionProfile
from assistant.decision_engine import Decision, DecisionScores


@pytest.fixture(autouse=True)
def _reset_ctx():
    reset_company_context()
    yield
    reset_company_context()


def _make_emotion(primary="neutral", intensity=0.1, fomo=False):
    return UserEmotionProfile(
        primary_emotion=primary,
        emotion_intensity=intensity,
        needs_confirmation=(primary == "fomo"),
        risk_of_impulsive_action=fomo,
        signals=[],
    )


def _make_decision(action="buy"):
    return Decision(
        asset="gold",
        action=action,
        confidence="medium",
        thesis="黄金整体偏多",
        risks=["止损位要留好"],
        scores=DecisionScores(
            direction_score=0.4, crowding_score=0.3,
            entry_quality="medium", chasing_risk="low",
        ),
    )


def _make_profile(role="retail", is_internal=False, wants_concise=False):
    return UserProfile(
        user_id="1",
        role=role,
        is_internal=is_internal,
        preferred_style="concise" if wants_concise else "analytical",
    )


def _make_company(banned=("取决于你的风险偏好",)):
    return CompanyContext(banned_phrases=banned)


class TestCheckReplyPolicy:
    def test_valid_reply_passes(self):
        result = check_reply_policy(
            reply="✅ 结论：可以参与（黄金）\n\n新闻面整体利多，建议分批入场。",
            decision=_make_decision("buy"),
            emotion=_make_emotion("neutral"),
            profile=_make_profile(),
            company=_make_company(),
        )
        assert result.is_valid
        assert result.violations == []

    def test_banned_phrase_detected(self):
        result = check_reply_policy(
            reply="取决于你的风险偏好，建议咨询专业人士",
            decision=_make_decision("buy"),
            emotion=_make_emotion("neutral"),
            profile=_make_profile(),
            company=_make_company(banned=("取决于你的风险偏好",)),
        )
        assert not result.is_valid
        assert any("banned_phrase" in v for v in result.violations)

    def test_fomo_buy_missing_caution(self):
        result = check_reply_policy(
            reply="✅ 可以买黄金，大家都看多。",
            decision=_make_decision("buy"),
            emotion=_make_emotion("fomo", intensity=0.7),
            profile=_make_profile(),
            company=_make_company(),
        )
        assert not result.is_valid
        assert any("fomo_buy_missing_caution" in v for v in result.violations)

    def test_fomo_buy_with_caution_passes(self):
        result = check_reply_policy(
            reply="✅ 可以买黄金，但请分批入场，不要满仓追。",
            decision=_make_decision("buy"),
            emotion=_make_emotion("fomo", intensity=0.7),
            profile=_make_profile(),
            company=_make_company(),
        )
        assert result.is_valid

    def test_emotion_bias_buy_on_avoid(self):
        result = check_reply_policy(
            reply="你可以买，建议立刻买入。",
            decision=_make_decision("avoid"),
            emotion=_make_emotion("fomo"),
            profile=_make_profile(),
            company=_make_company(),
        )
        assert not result.is_valid
        assert any("emotion_bias" in v for v in result.violations)

    def test_hold_decision_with_neutral_reply_passes(self):
        result = check_reply_policy(
            reply="⏸ 结论：先别动，等更清楚的信号。",
            decision=_make_decision("hold_wait"),
            emotion=_make_emotion("neutral"),
            profile=_make_profile(),
            company=_make_company(),
        )
        assert result.is_valid


class TestApplyStyleConstraints:
    def test_removes_banned_phrases(self):
        company = _make_company(banned=("仅供参考",))
        profile = _make_profile()
        emotion = _make_emotion()
        text = "这份分析仅供参考，请自行判断。"
        result = apply_style_constraints(text, profile, emotion, company)
        assert "仅供参考" not in result

    def test_internal_user_long_reply_truncated(self):
        company = _make_company(banned=())
        profile = _make_profile(is_internal=True)
        emotion = _make_emotion()
        # 10 paragraphs
        text = "\n\n".join([f"段落 {i}" for i in range(10)])
        result = apply_style_constraints(text, profile, emotion, company)
        paragraphs = [p for p in result.split("\n\n") if p.strip()]
        assert len(paragraphs) <= 6

    def test_retail_user_reply_not_truncated(self):
        company = _make_company(banned=())
        profile = _make_profile(role="retail")
        emotion = _make_emotion()
        text = "\n\n".join([f"段落 {i}" for i in range(5)])
        result = apply_style_constraints(text, profile, emotion, company)
        paragraphs = [p for p in result.split("\n\n") if p.strip()]
        assert len(paragraphs) == 5


class TestStyleExamples:
    def test_examples_have_required_fields(self):
        for ex in STYLE_EXAMPLES:
            assert "question" in ex
            assert "good_reply" in ex
            assert "bad_reply" in ex
            assert "emotion" in ex
            assert "decision" in ex

    def test_good_reply_not_contains_banned(self):
        banned = (
            "取决于你的风险偏好",
            "建议咨询专业人士",
            "请自行判断",
            "仅供参考",
            "不构成投资建议",
        )
        for ex in STYLE_EXAMPLES:
            good = ex["good_reply"].lower()
            for phrase in banned:
                assert phrase.lower() not in good, (
                    f"Style example for '{ex['question']}' contains banned phrase: '{phrase}'"
                )

    def test_get_style_examples_returns_string(self):
        result = get_style_examples_for_prompt(emotion="fomo", n=2)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_get_style_examples_filters_by_emotion(self):
        result = get_style_examples_for_prompt(emotion="fomo", n=1)
        assert "FOMO" in result or "fomo" in result.lower() or "怕错过" in result
