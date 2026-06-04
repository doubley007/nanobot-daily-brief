"""
端到端黄金场景：fixture -> retrieve -> aggregate -> decide -> compose.

不依赖真实 LLM（强制 llm_callable=None）也不依赖 yfinance（monkey-patch
fetch_trend_signal）。这个测试就是用户要的核心 demo 链路的 pytest 保险丝。
"""
from __future__ import annotations

import pytest

import assistant.pipeline as pipeline
from assistant.fixtures import install_gold_fixture
from assistant.pipeline import answer_question_traced
from assistant.trend_signals import trend_from_values


@pytest.fixture
def gold_ready(monkeypatch):
    install_gold_fixture()
    monkeypatch.setattr(
        pipeline, "fetch_trend_signal",
        lambda asset: trend_from_values(asset, r7=0.03, r30=0.09),
    )
    # 强制不走 LLM，确保 test 可 reproducible
    monkeypatch.setattr(pipeline, "_llm_callable", lambda: None)


def test_gold_buy_question_end_to_end(gold_ready):
    trace = answer_question_traced(
        "我能不能买黄金？大家都在买，我是不是也该上"
    )
    # 路由应识别为 market_decision + gold + fomo
    assert trace.route.route == "market_decision"
    assert trace.route.asset == "gold"
    assert trace.route.user_emotion == "fomo"

    # 决策应该有明确结论
    assert trace.decision is not None
    assert trace.decision.action in ("buy_small", "buy", "avoid_chasing")

    # 聚合应该识别出偏多 + 有 FOMO 信号
    assert trace.aggregate is not None
    assert trace.aggregate.overall_bias == "bullish"
    assert trace.aggregate.post_count > 10

    # 回复必须包含结论和依据
    reply = trace.reply
    assert "结论" in reply
    assert "黄金" in reply
    # 不能出现模糊免责话术
    assert "取决于你的风险偏好" not in reply
    assert "请咨询专业" not in reply
    # 应该给 FOMO 用户额外的"别重仓"提醒
    assert "分批" in reply or "小仓位" in reply or "别重仓" in reply


def test_gold_can_chase_question(gold_ready):
    trace = answer_question_traced("黄金还能追吗")
    assert trace.route.route == "market_decision"
    assert trace.route.asset == "gold"
    assert trace.decision is not None
    assert trace.decision.thesis


def test_market_summary_for_gold(gold_ready):
    trace = answer_question_traced("最近黄金市场怎么样？")
    assert trace.route.route in ("market_summary", "market_decision")
    assert trace.route.asset == "gold"
    assert "黄金" in trace.reply


def test_emotional_path_does_not_run_decision(gold_ready):
    trace = answer_question_traced("我好焦虑，不知道该怎么办")
    assert trace.route.route == "emotional_chat"
    assert trace.decision is None
    # 应给共情 + 引导下一步，不应出现具体决策结论
    assert "结论：" not in trace.reply
