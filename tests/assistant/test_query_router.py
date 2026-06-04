"""规则层路由单测 —— 不依赖 LLM。"""
from __future__ import annotations

from assistant.query_router import route_query


def test_market_decision_gold_fomo():
    r = route_query("我能不能买黄金？怕错过这波")
    assert r.route == "market_decision"
    assert r.asset == "gold"
    # FOMO 模式可能被识别成 fomo 或 uncertain；两者都表示犹豫且需要明确答案
    assert r.user_emotion in ("fomo", "uncertain", "anxious")
    assert r.confidence >= 0.7


def test_market_decision_nvidia():
    r = route_query("英伟达现在还能追吗")
    assert r.route == "market_decision"
    assert r.asset == "nvidia"


def test_market_summary_market_general():
    r = route_query("今天市场怎么了？有什么新闻")
    assert r.route == "market_summary"


def test_emotional_chat_no_asset():
    r = route_query("我好焦虑，睡不着")
    assert r.route == "emotional_chat"
    assert r.user_emotion in ("anxious", "neutral")
    assert r.asset is None


def test_emotional_chat_frustration():
    r = route_query("我亏麻了怎么办")
    assert r.route == "emotional_chat"
    assert r.user_emotion == "frustrated"


def test_empty_input_is_emotional():
    r = route_query("")
    assert r.route == "emotional_chat"
    assert r.confidence < 0.5


def test_confirm_seeking_with_asset():
    r = route_query("大家都看多黄金，我是不是也该上")
    assert r.route == "market_decision"
    assert r.asset == "gold"
