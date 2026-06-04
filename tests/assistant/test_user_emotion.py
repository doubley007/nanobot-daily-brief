"""用户情绪识别器单测。"""
from __future__ import annotations

from assistant.user_emotion import analyze_user_emotion


def test_fomo_with_impulsive():
    p = analyze_user_emotion("怕错过，要不要 all in 黄金")
    assert p.primary_emotion == "fomo"
    assert p.needs_confirmation
    assert p.risk_of_impulsive_action  # 'all in' 触发


def test_frustrated_loss():
    p = analyze_user_emotion("我亏麻了，套得很深，心态崩了")
    assert p.primary_emotion == "frustrated"
    assert p.emotion_intensity >= 0.3


def test_neutral_input():
    p = analyze_user_emotion("今天天气不错")
    assert p.primary_emotion == "neutral"
    assert not p.needs_confirmation


def test_seeking_confirmation():
    p = analyze_user_emotion("大家都看多黄金，对不对")
    assert p.needs_confirmation
