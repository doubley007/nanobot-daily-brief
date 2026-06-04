"""
端到端比特币场景：fixture -> retrieve -> aggregate -> decide -> compose.

验证框架不只服务于黄金——相同的 pipeline 对另一个资产也能：
  1. 正确识别资产（bitcoin）
  2. 聚合出有意义的情绪（bullish，有 FOMO 信号）
  3. 产出有明确结论的回复
  4. 在偏热场景下给出 buy_small / avoid_chasing（而不是 buy）
  5. 路由一条情绪性"踏空"消息时走 emotional_chat 而非 decision
"""
from __future__ import annotations

import pytest

import assistant.pipeline as pipeline
from assistant.fixtures import install_bitcoin_fixture
from assistant.pipeline import answer_question_traced
from assistant.trend_signals import trend_from_values


@pytest.fixture
def btc_ready(monkeypatch):
    install_bitcoin_fixture()
    # BTC fixture: 7d=+5%, 30d=+18% -> momentum=up, overheating=high
    _trend = lambda asset: trend_from_values(asset, r7=0.05, r30=0.18)
    monkeypatch.setattr(pipeline, "fetch_trend_signal", _trend)
    # context_builder also imports fetch_trend_signal directly, patch it too
    import assistant.context_builder as _cb
    monkeypatch.setattr(_cb, "fetch_trend_signal", _trend)
    monkeypatch.setattr(pipeline, "_llm_callable", lambda: None)


def test_bitcoin_buy_question_routing(btc_ready):
    trace = answer_question_traced("比特币现在能买吗？大家都说快涨了")
    assert trace.route.route == "market_decision"
    assert trace.route.asset == "bitcoin"


def test_bitcoin_aggregated_sentiment(btc_ready):
    trace = answer_question_traced("btc 还能追吗")
    assert trace.aggregate is not None
    assert trace.aggregate.post_count > 10
    # fixture has ~65% bullish posts
    assert trace.aggregate.overall_bias == "bullish"
    # FOMO posts are present
    assert trace.aggregate.fomo_ratio > 0.0


def test_bitcoin_overheated_downgrade(btc_ready):
    """BTC fixture has 7d+5%, 30d+18% and crowded community -> should not get plain 'buy'."""
    trace = answer_question_traced("能不能买比特币？")
    assert trace.decision is not None
    # Overheated + crowded = should be downgraded from buy
    assert trace.decision.action in ("buy_small", "avoid_chasing", "hold_wait")
    # entry_quality and chasing_risk must be in evidence
    scores = trace.decision.evidence.get("decision_scores", {})
    assert scores.get("chasing_risk") in ("medium", "high")


def test_bitcoin_decision_reply_has_conclusion(btc_ready):
    trace = answer_question_traced("我该不该上比特币？")
    assert trace.decision is not None
    reply = trace.reply
    assert "结论" in reply
    assert "比特币" in reply
    assert "取决于你的风险偏好" not in reply
    assert "请咨询专业" not in reply


def test_bitcoin_fomo_user_gets_addendum(btc_ready):
    """FOMO 用户问比特币时，reply_composer 应附加冷静建议。"""
    trace = answer_question_traced(
        "比特币大家都在买，我怕错过，能不能上车？"
    )
    assert trace.route.user_emotion == "fomo"
    assert trace.decision is not None
    # Even if action is buy/buy_small, FOMO addendum should appear
    if trace.decision.action in ("buy", "buy_small"):
        reply = trace.reply
        assert "分批" in reply or "别重仓" in reply or "小仓位" in reply


def test_bitcoin_emotional_bypass(btc_ready):
    """纯情绪宣泄（无明确决策询问）不应触发决策链路。"""
    # 无资产关键词、无决策动作词 -> emotional_chat
    trace = answer_question_traced("我最近亏麻了，整个人都崩溃了，睡不着")
    assert trace.route.route == "emotional_chat"
    assert trace.decision is None
    assert "结论：" not in trace.reply


# ── CL2 additions: offline bitcoin pipeline completeness ─────────────────────

def test_bitcoin_company_context_injected(btc_ready):
    """Company context must always be present in bitcoin traces."""
    trace = answer_question_traced("比特币能买吗")
    assert trace.company is not None
    assert trace.company.company_name


def test_bitcoin_user_profile_present(btc_ready):
    """User profile fallback must work for unknown user on bitcoin query."""
    trace = answer_question_traced("比特币涨了，我还能追吗", user_id=None)
    assert trace.profile is not None
    assert trace.profile.role == "unknown"


def test_bitcoin_policy_violations_list(btc_ready):
    """policy_violations is always a list, even when empty."""
    trace = answer_question_traced("比特币现在能买吗")
    assert isinstance(trace.policy_violations, list)


def test_bitcoin_reply_no_banned_phrases(btc_ready):
    """Reply must not contain company banned phrases."""
    trace = answer_question_traced("比特币能不能买")
    if trace.company:
        for phrase in trace.company.banned_phrases:
            assert phrase.lower() not in trace.reply.lower()


def test_bitcoin_context_build_time_recorded(btc_ready):
    """Pipeline must record context build time in ms."""
    trace = answer_question_traced("能不能买比特币")
    assert "context_build_ms" in trace.meta
    assert trace.meta["context_build_ms"] >= 0


def test_bitcoin_evidence_populated(btc_ready):
    """Bitcoin decision must have news and community evidence populated."""
    trace = answer_question_traced("比特币现在适合入场吗")
    assert trace.context_pkg is not None
    assert len(trace.context_pkg.news) > 0
    assert len(trace.context_pkg.community) > 0


def test_bitcoin_decision_scores_present(btc_ready):
    """Decision must have scores with all required fields."""
    trace = answer_question_traced("比特币能买吗")
    assert trace.decision is not None
    sc = trace.decision.scores
    assert sc is not None
    assert sc.direction_score is not None
    assert sc.entry_quality in ("good", "medium", "poor")
    assert sc.chasing_risk in ("low", "medium", "high")
