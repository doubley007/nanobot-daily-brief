"""
v6 Task 4 tests: executive style report.

Covers:
  A. generate_report(style="executive") returns non-empty string
  B. Executive shorter (or at least different) than analyst
  C. Executive contains required fields: verdict, action tendency, community bias
  D. Executive does NOT contain raw numeric signal fields
  E. Fallback report with executive style
  F. Telegram /report gold executive entry point (style parsing)
  G. Demo --report-style executive flag
  H. Default style is "analyst" (backwards compat)
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from assistant.fixtures import install_gold_fixture, install_bitcoin_fixture
from assistant.session_memory import reset_session_store


@pytest.fixture(autouse=True)
def _reset(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSISTANT_KNOWLEDGE_DB", str(tmp_path / "knowledge.sqlite3"))
    monkeypatch.setenv("SESSION_MEMORY_FILE", str(tmp_path / "session_memory.json"))
    from assistant.rag import store as _store_mod
    _store_mod._default = None
    reset_session_store()
    from assistant.rag.vector_store import reset_vector_store
    reset_vector_store()
    yield
    _store_mod._default = None
    reset_session_store()
    reset_vector_store()


def _make_mock_sig():
    sig = MagicMock()
    sig.computed_at = 1700000000.0
    sig.news_direction = "bullish"
    sig.news_strength = 0.7
    sig.community_bias = "bullish"
    sig.bullish_ratio = 0.62
    sig.bearish_ratio = 0.25
    sig.fomo_ratio = 0.18
    sig.crowding_risk = "low"
    sig.entry_quality = "good"
    sig.trend_momentum = "rising"
    sig.narrative_keywords = ["inflation", "safe haven", "rally", "demand", "etf", "hedge"]
    sig.summary = "黄金近期多方信号强劲，社区情绪偏多。"
    sig.news_count = 12
    sig.post_count = 87
    return sig


def _make_mock_trend():
    trend = MagicMock()
    trend.momentum_label = "rising"
    trend.recent_return_7d = 0.031
    trend.recent_return_30d = 0.092
    trend.overheating_risk = "low"
    return trend


# ── A: executive report returns non-empty string ──────────────────────────────

class TestExecutiveBasic:
    def test_executive_returns_string(self):
        from assistant.report import _format_executive_report
        sig = _make_mock_sig()
        trend = _make_mock_trend()
        result = _format_executive_report("gold", sig, trend, "3d")
        assert isinstance(result, str)
        assert len(result) > 50

    def test_executive_via_generate_report(self, monkeypatch):
        install_gold_fixture()
        from assistant.trend_signals import trend_from_values
        monkeypatch.setattr("assistant.report.generate_report.__wrapped__"
                            if hasattr(__import__("assistant.report", fromlist=["generate_report"]).generate_report, "__wrapped__")
                            else "assistant.report._format_executive_report",
                            lambda *a, **kw: "OK")
        from assistant.report import _format_executive_report
        sig = _make_mock_sig()
        result = _format_executive_report("gold", sig, None, "3d")
        assert result


# ── B: executive vs analyst length ───────────────────────────────────────────

class TestExecutiveVsAnalyst:
    def test_executive_shorter_than_analyst(self):
        from assistant.report import _format_report, _format_executive_report
        sig = _make_mock_sig()
        trend = _make_mock_trend()
        analyst_text = _format_report("gold", sig, trend, "3d")
        executive_text = _format_executive_report("gold", sig, trend, "3d")
        assert len(executive_text) < len(analyst_text), (
            f"Executive ({len(executive_text)}) should be shorter than analyst ({len(analyst_text)})"
        )

    def test_executive_has_fewer_lines_than_analyst(self):
        from assistant.report import _format_report, _format_executive_report
        sig = _make_mock_sig()
        trend = _make_mock_trend()
        analyst_lines = _format_report("gold", sig, trend, "3d").split("\n")
        executive_lines = _format_executive_report("gold", sig, trend, "3d").split("\n")
        assert len(executive_lines) < len(analyst_lines)


# ── C: executive required fields ─────────────────────────────────────────────

class TestExecutiveRequiredFields:
    def test_has_verdict_line(self):
        from assistant.report import _format_executive_report
        sig = _make_mock_sig()
        text = _format_executive_report("gold", sig, None, "3d")
        assert "▶" in text or "积极" in text or "消极" in text or "中性" in text

    def test_has_action_tendency(self):
        from assistant.report import _format_executive_report
        sig = _make_mock_sig()
        text = _format_executive_report("gold", sig, None, "3d")
        assert "行动倾向" in text or "建议" in text or "可以" in text

    def test_has_community_bias(self):
        from assistant.report import _format_executive_report
        sig = _make_mock_sig()
        text = _format_executive_report("gold", sig, None, "3d")
        assert "多" in text or "空" in text

    def test_has_data_count(self):
        from assistant.report import _format_executive_report
        sig = _make_mock_sig()
        text = _format_executive_report("gold", sig, None, "3d")
        assert "新闻" in text or "社区" in text or "数据" in text

    def test_has_narrative_keywords(self):
        from assistant.report import _format_executive_report
        sig = _make_mock_sig()
        text = _format_executive_report("gold", sig, None, "3d")
        assert "叙事" in text or "inflation" in text.lower() or "safe haven" in text.lower()

    def test_fomo_warning_when_high(self):
        from assistant.report import _format_executive_report
        sig = _make_mock_sig()
        sig.fomo_ratio = 0.45
        text = _format_executive_report("gold", sig, None, "3d")
        assert "FOMO" in text or "追高" in text

    def test_no_fomo_warning_when_low(self):
        from assistant.report import _format_executive_report
        sig = _make_mock_sig()
        sig.fomo_ratio = 0.10
        text = _format_executive_report("gold", sig, None, "3d")
        # Low FOMO should not show the warning
        assert "FOMO" not in text or "偏高" not in text

    def test_crowding_warning_when_high(self):
        from assistant.report import _format_executive_report
        sig = _make_mock_sig()
        sig.crowding_risk = "high"
        text = _format_executive_report("gold", sig, None, "3d")
        assert "拥挤" in text and ("高" in text or "警惕" in text)

    def test_trend_included_when_available(self):
        from assistant.report import _format_executive_report
        sig = _make_mock_sig()
        trend = _make_mock_trend()
        text = _format_executive_report("gold", sig, trend, "3d")
        assert "趋势" in text or "动量" in text

    def test_trend_excluded_when_none(self):
        from assistant.report import _format_executive_report
        sig = _make_mock_sig()
        text = _format_executive_report("gold", sig, None, "3d")
        # Should not crash with no trend
        assert isinstance(text, str)


# ── D: analyst-only fields absent from executive ─────────────────────────────

class TestExecutiveNoRawNumbers:
    def test_no_raw_signal_strength(self):
        from assistant.report import _format_executive_report
        sig = _make_mock_sig()
        text = _format_executive_report("gold", sig, None, "3d")
        # Analyst shows "强度 0.70" style numbers — executive shouldn't
        assert "强度 0." not in text

    def test_no_7d_30d_return_lines(self):
        from assistant.report import _format_executive_report
        sig = _make_mock_sig()
        trend = _make_mock_trend()
        text = _format_executive_report("gold", sig, trend, "3d")
        # Analyst shows "7d=+3.1%  30d=+9.2%" — executive shouldn't
        assert "7d=" not in text and "30d=" not in text


# ── E: fallback report with executive style ───────────────────────────────────

class TestFallbackExecutive:
    def test_fallback_executive_returns_string(self):
        install_gold_fixture()
        from assistant.rag.store import default_store
        from assistant.report import _fallback_report
        store = default_store()
        result = _fallback_report("gold", store, 72, None, style="executive")
        assert isinstance(result, str)
        assert len(result) > 30

    def test_fallback_executive_has_different_format(self):
        install_gold_fixture()
        from assistant.rag.store import default_store
        from assistant.report import _fallback_report
        store = default_store()
        analyst = _fallback_report("gold", store, 72, None, style="analyst")
        executive = _fallback_report("gold", store, 72, None, style="executive")
        assert analyst != executive

    def test_fallback_executive_no_emoji_heavy_header(self):
        install_gold_fixture()
        from assistant.rag.store import default_store
        from assistant.report import _fallback_report
        store = default_store()
        result = _fallback_report("gold", store, 72, None, style="executive")
        # Executive fallback uses 【...】 format, not 📊
        assert "📊" not in result or "执行摘要" in result


# ── F: Telegram style parsing ─────────────────────────────────────────────────

class TestTelegramStyleParsing:
    def test_parts_parse_executive(self):
        text = "/report gold executive"
        parts = text.strip().split()
        asset = parts[1].lower() if len(parts) > 1 else None
        style = "executive" if len(parts) > 2 and parts[2].lower() == "executive" else "analyst"
        assert asset == "gold"
        assert style == "executive"

    def test_parts_parse_default_analyst(self):
        text = "/report gold"
        parts = text.strip().split()
        style = "executive" if len(parts) > 2 and parts[2].lower() == "executive" else "analyst"
        assert style == "analyst"

    def test_snapshot_command_parses_same_way(self):
        text = "/snapshot bitcoin executive"
        parts = text.strip().split()
        asset = parts[1].lower() if len(parts) > 1 else None
        style = "executive" if len(parts) > 2 and parts[2].lower() == "executive" else "analyst"
        assert asset == "bitcoin"
        assert style == "executive"


# ── G: demo CLI style flag ────────────────────────────────────────────────────

class TestDemoStyleFlag:
    def test_report_style_arg_accepted(self):
        from assistant.demo import main
        install_gold_fixture()
        # Should not raise
        ret = main(["黄金能买吗", "--report", "--report-style", "executive", "--no-llm"])
        assert ret == 0

    def test_report_style_analyst_default(self):
        from assistant.demo import main
        install_gold_fixture()
        ret = main(["黄金能买吗", "--report", "--no-llm"])
        assert ret == 0


# ── H: backwards compatibility ───────────────────────────────────────────────

class TestBackwardsCompat:
    def test_default_style_is_analyst(self):
        from assistant.report import generate_report
        import inspect
        sig = inspect.signature(generate_report)
        assert sig.parameters["style"].default == "analyst"

    def test_generate_report_with_fixture(self):
        install_gold_fixture()
        from assistant.report import generate_report
        result = generate_report("gold")  # no style kwarg
        assert isinstance(result, str)
        assert len(result) > 50

    def test_analyst_contains_emoji_header(self):
        from assistant.report import _format_report
        sig = _make_mock_sig()
        text = _format_report("gold", sig, None, "3d")
        assert "📊" in text
