"""
Tests for CL4: enhanced /debug, /why command, and --explain demo flag.

Verifies:
  - _format_debug_block contains all expected fields
  - _format_why_block shows decision skeleton reasoning steps
  - /why fallback message when no trace cached
"""
from __future__ import annotations

import pytest
import assistant.pipeline as pipeline
from assistant.pipeline import answer_question_traced
from assistant.fixtures import install_gold_fixture
from assistant.trend_signals import trend_from_values
from assistant.telegram_bot import _format_debug_block, _format_why_block
from assistant.user_profile import reset_profile_store
from assistant.company_context import reset_company_context


@pytest.fixture(autouse=True)
def _reset():
    reset_company_context()
    reset_profile_store()
    yield
    reset_company_context()
    reset_profile_store()


@pytest.fixture
def gold_trace(monkeypatch):
    install_gold_fixture()
    _trend = lambda asset: trend_from_values(asset, r7=0.03, r30=0.09)
    monkeypatch.setattr(pipeline, "fetch_trend_signal", _trend)
    import assistant.context_builder as _cb
    monkeypatch.setattr(_cb, "fetch_trend_signal", _trend)
    monkeypatch.setattr(pipeline, "_llm_callable", lambda: None)
    return answer_question_traced("我能不能买黄金")


@pytest.fixture
def emotional_trace(monkeypatch):
    install_gold_fixture()
    monkeypatch.setattr(pipeline, "_llm_callable", lambda: None)
    return answer_question_traced("我好焦虑，不知道该怎么办")


# ── _format_debug_block ───────────────────────────────────────────────────────

class TestFormatDebugBlock:
    def test_contains_route(self, gold_trace):
        block = _format_debug_block(gold_trace)
        assert "route=" in block

    def test_contains_asset(self, gold_trace):
        block = _format_debug_block(gold_trace)
        assert "gold" in block

    def test_contains_user_emotion(self, gold_trace):
        block = _format_debug_block(gold_trace)
        assert "user_emotion=" in block

    def test_contains_retrieved_counts(self, gold_trace):
        block = _format_debug_block(gold_trace)
        assert "retrieved:" in block
        assert "news=" in block
        assert "community=" in block

    def test_contains_aggregate_when_market_decision(self, gold_trace):
        block = _format_debug_block(gold_trace)
        assert "aggregate:" in block

    def test_contains_decision_action(self, gold_trace):
        block = _format_debug_block(gold_trace)
        assert "decision:" in block

    def test_contains_build_ms(self, gold_trace):
        block = _format_debug_block(gold_trace)
        assert "build_ms=" in block

    def test_empty_when_no_context_pkg(self):
        class FakeTrace:
            context_pkg = None
        assert _format_debug_block(FakeTrace()) == ""


# ── _format_why_block ─────────────────────────────────────────────────────────

class TestFormatWhyBlock:
    def test_contains_decision_action(self, gold_trace):
        block = _format_why_block(gold_trace)
        assert gold_trace.decision.action in block

    def test_contains_news_section(self, gold_trace):
        block = _format_why_block(gold_trace)
        assert "新闻面" in block

    def test_contains_community_section(self, gold_trace):
        block = _format_why_block(gold_trace)
        assert "社区" in block

    def test_contains_direction_score(self, gold_trace):
        block = _format_why_block(gold_trace)
        assert "方向=" in block

    def test_contains_engine_trace(self, gold_trace):
        block = _format_why_block(gold_trace)
        assert "推理步骤" in block

    def test_contains_entry_quality(self, gold_trace):
        block = _format_why_block(gold_trace)
        assert "入场质量" in block

    def test_no_decision_path_graceful(self, emotional_trace):
        block = _format_why_block(emotional_trace)
        # Should not crash; should explain no decision
        assert "决策链路" in block or "route=" in block

    def test_why_block_not_empty(self, gold_trace):
        block = _format_why_block(gold_trace)
        assert len(block) > 50
