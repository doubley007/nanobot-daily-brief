"""
升级版黄金场景端到端测试。

覆盖新增的：
  - company context 注入
  - user profile（内部用户 vs 普通用户）
  - policy 检查
  - skills 层
  - context builder
  - debug trace 格式

所有测试不依赖真实 LLM 和网络。
"""
from __future__ import annotations

import pytest
import assistant.pipeline as pipeline
from assistant.pipeline import answer_question_traced, PipelineTrace
from assistant.fixtures import install_gold_fixture
from assistant.trend_signals import trend_from_values
from assistant.user_profile import UserProfile, get_profile_store, reset_profile_store
from assistant.company_context import reset_company_context


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


# ─── Company context 注入 ──────────────────────────────────────────────────────

class TestCompanyContextInjection:
    def test_trace_has_company(self, gold_ready):
        trace = answer_question_traced("我能不能买黄金")
        assert trace.company is not None
        assert trace.company.company_name

    def test_reply_has_no_banned_phrases(self, gold_ready):
        trace = answer_question_traced("我能不能买黄金")
        banned = trace.company.banned_phrases if trace.company else ()
        for phrase in banned:
            assert phrase.lower() not in trace.reply.lower(), (
                f"Reply contains banned phrase: '{phrase}'"
            )

    def test_context_pkg_company_block(self, gold_ready):
        trace = answer_question_traced("黄金能追吗")
        assert trace.context_pkg is not None
        block = trace.context_pkg.to_prompt_block()
        assert "公司语境" in block


# ─── User profile ─────────────────────────────────────────────────────────────

class TestUserProfileInPipeline:
    def test_trace_has_profile(self, gold_ready):
        trace = answer_question_traced("黄金怎么样", user_id=None)
        assert trace.profile is not None

    def test_default_profile_for_unknown_user(self, gold_ready):
        trace = answer_question_traced("黄金能买吗", user_id=9999999999)
        assert trace.profile.role == "unknown"

    def test_internal_user_profile_loaded(self, gold_ready):
        store = get_profile_store()
        store.set(UserProfile(
            user_id="12345",
            role="pm",
            is_internal=True,
            preferred_style="concise",
        ))
        trace = answer_question_traced("黄金怎么样", user_id=12345)
        assert trace.profile.is_internal is True
        assert trace.profile.role == "pm"

    def test_internal_user_context_block_mentions_internal(self, gold_ready):
        store = get_profile_store()
        store.set(UserProfile(user_id="777", role="insider", is_internal=True))
        trace = answer_question_traced("黄金能追吗", user_id=777)
        assert trace.context_pkg is not None
        block = trace.context_pkg.to_prompt_block()
        assert "内部用户" in block


# ─── Reply policy ──────────────────────────────────────────────────────────────

class TestReplyPolicyInPipeline:
    def test_fomo_buy_reply_has_caution(self, gold_ready):
        trace = answer_question_traced("大家都在买黄金，我是不是也该上")
        # FOMO + buy/buy_small → must have 分批/小仓位
        if trace.decision and trace.decision.action in ("buy", "buy_small"):
            assert any(kw in trace.reply for kw in ("分批", "小仓位", "别重仓", "不要满仓"))

    def test_policy_violations_logged(self, gold_ready):
        trace = answer_question_traced("我能不能买黄金")
        # 不管有没有 violations，列表存在
        assert isinstance(trace.policy_violations, list)

    def test_avoid_reply_has_no_buy_suggestion(self, gold_ready, monkeypatch):
        # 强制 decision 为 avoid
        from assistant.decision_engine import Decision, DecisionScores
        from assistant.rag.store import NewsDoc, CommunityDoc
        from assistant.sentiment_aggregator import AggregatedSentiment
        from assistant.trend_signals import TrendSignal

        fixed_decision = Decision(
            asset="gold", action="avoid", confidence="high",
            thesis="全面利空",
            evidence={
                "news": [], "news_assessment": {"direction": "bearish", "bullish_score": 0,
                                                 "bearish_score": 0.8, "key_bullets": []},
                "community_aggregate": {"post_count": 20, "overall_bias": "bearish",
                                         "bullish_ratio": 0.1, "bearish_ratio": 0.7,
                                         "fomo_ratio": 0.0, "crowded_trade_risk": "low",
                                         "narrative_keywords": [], "summary": "偏空"},
                "community_samples": [], "trend": {}, "decision_scores": {},
                "engine_trace": [],
            },
            risks=["下行风险"],
            scores=DecisionScores(
                direction_score=-0.6, crowding_score=0.2,
                entry_quality="poor", chasing_risk="low",
            ),
        )
        monkeypatch.setattr(pipeline, "make_decision", lambda **_: fixed_decision)
        trace = answer_question_traced("我能不能买黄金", user_id=None)
        # avoid 决策 → reply 不应该说"可以买"
        import re
        assert not re.search(r"(可以买|建议买|立刻买|马上买)", trace.reply)


# ─── Context builder in pipeline ──────────────────────────────────────────────

class TestContextBuilderInPipeline:
    def test_news_and_community_in_trace(self, gold_ready):
        trace = answer_question_traced("我能不能买黄金")
        assert trace.context_pkg is not None
        assert len(trace.context_pkg.news) > 0
        assert len(trace.context_pkg.community) > 0

    def test_debug_dict_has_all_fields(self, gold_ready):
        trace = answer_question_traced("黄金能买吗")
        d = trace.context_pkg.to_debug_dict()
        for key in ("route", "asset", "user_emotion", "profile",
                    "company_context_injected", "retrieved_news_count",
                    "retrieved_community_count"):
            assert key in d, f"Missing key: {key}"

    def test_build_time_in_meta(self, gold_ready):
        trace = answer_question_traced("黄金能买吗")
        assert "context_build_ms" in trace.meta
        assert trace.meta["context_build_ms"] >= 0


# ─── Core gold scenarios (regression) ─────────────────────────────────────────

class TestGoldScenariosRegression:
    """Regression: 这些测试在升级后仍然必须通过。"""

    def test_gold_buy_question_end_to_end(self, gold_ready):
        trace = answer_question_traced("我能不能买黄金？大家都在买，我是不是也该上")
        assert trace.route.route == "market_decision"
        assert trace.route.asset == "gold"
        assert trace.route.user_emotion == "fomo"
        assert trace.decision is not None
        assert trace.decision.action in ("buy_small", "buy", "avoid_chasing")
        assert trace.aggregate is not None
        assert trace.aggregate.overall_bias == "bullish"
        assert trace.aggregate.post_count > 10
        assert "结论" in trace.reply
        assert "黄金" in trace.reply
        assert "取决于你的风险偏好" not in trace.reply
        assert "请咨询专业" not in trace.reply
        assert any(kw in trace.reply for kw in ("分批", "小仓位", "别重仓"))

    def test_gold_chase_question(self, gold_ready):
        trace = answer_question_traced("黄金还能追吗")
        assert trace.route.route == "market_decision"
        assert trace.route.asset == "gold"
        assert trace.decision is not None
        assert trace.decision.thesis

    def test_market_summary_for_gold(self, gold_ready):
        trace = answer_question_traced("最近黄金市场怎么样？")
        assert trace.route.route in ("market_summary", "market_decision")
        assert trace.route.asset == "gold"
        assert "黄金" in trace.reply

    def test_emotional_path_no_decision(self, gold_ready):
        trace = answer_question_traced("我好焦虑，不知道该怎么办")
        assert trace.route.route == "emotional_chat"
        assert trace.decision is None
        assert "结论：" not in trace.reply

    def test_fomo_community_question(self, gold_ready):
        trace = answer_question_traced("大家都在买，我是不是也该上")
        assert trace.route.user_emotion == "fomo"
        # FOMO + buy → 分批提醒
        if trace.decision and trace.decision.action in ("buy", "buy_small"):
            assert any(kw in trace.reply for kw in ("分批", "小仓位", "FOMO", "怕错过"))
