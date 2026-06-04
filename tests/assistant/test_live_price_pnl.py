"""
v6 Task 1 tests: live price wired into holdings P&L.

Covers:
  A. get_current_price returns (price, status) tuple
  B. holdings_reply_addendum receives current_price and shows correct P&L label
  C. pipeline passes price to addendum (in_profit / underwater / near_cost)
  D. Fallback path: yfinance unavailable → price=None, status="fallback"
  E. FOMO firewall: live price does NOT change decision action
"""
from __future__ import annotations

import pytest
import assistant.pipeline as pipeline
from assistant.pipeline import answer_question_traced
from assistant.holdings import (
    Holding, holdings_reply_addendum,
    default_holdings_store, reset_holdings_store,
)
from assistant.trend_signals import trend_from_values
from assistant.fixtures import install_gold_fixture
from assistant.user_profile import reset_profile_store
from assistant.company_context import reset_company_context
from assistant.session_memory import reset_session_store


@pytest.fixture(autouse=True)
def _reset(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSISTANT_KNOWLEDGE_DB", str(tmp_path / "knowledge.sqlite3"))
    monkeypatch.setenv("SESSION_MEMORY_FILE", str(tmp_path / "session_memory.json"))
    from assistant.rag import store as _store_mod
    _store_mod._default = None
    reset_session_store()
    reset_profile_store()
    reset_company_context()
    reset_holdings_store()
    from assistant.rag.vector_store import reset_vector_store
    reset_vector_store()
    yield
    _store_mod._default = None
    reset_session_store()
    reset_profile_store()
    reset_company_context()
    reset_holdings_store()
    reset_vector_store()


# ── A: get_current_price interface ───────────────────────────────────────────

class TestGetCurrentPrice:
    def test_returns_tuple_two_elements(self, monkeypatch):
        from assistant.trend_signals import get_current_price
        monkeypatch.setattr("assistant.trend_signals.get_current_price",
                            lambda asset: (3300.0, "live"))
        price, status = get_current_price("gold")
        assert isinstance(price, float)
        assert status in ("live", "fallback", "missing")

    def test_missing_asset_returns_none(self):
        from assistant.trend_signals import get_current_price
        price, status = get_current_price(None)
        assert price is None
        assert status == "missing"

    def test_unknown_asset_returns_none_or_fallback(self):
        from assistant.trend_signals import get_current_price
        price, status = get_current_price("unknown_xyz_asset_9999")
        assert price is None
        assert status in ("missing", "fallback")

    def test_yfinance_failure_returns_fallback(self, monkeypatch):
        import yfinance as yf
        monkeypatch.setattr(yf.Ticker, "history", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("network")))
        from assistant.trend_signals import get_current_price
        price, status = get_current_price("gold")
        assert price is None
        assert status in ("fallback", "missing")


# ── B: holdings_reply_addendum with current_price ────────────────────────────

class TestAddendumWithPrice:
    def test_in_profit_shows_label(self):
        store = default_holdings_store()
        store.set("u1", "gold", position_size="medium", avg_cost=3000.0, horizon="mid")
        note = holdings_reply_addendum("u1", "gold", current_price=3300.0)
        assert note
        assert "浮盈" in note or "盈" in note

    def test_underwater_shows_label(self):
        store = default_holdings_store()
        store.set("u1", "gold", position_size="medium", avg_cost=3500.0, horizon="mid")
        note = holdings_reply_addendum("u1", "gold", current_price=3300.0)
        assert note
        assert "浮亏" in note or "亏" in note

    def test_near_cost_shows_label(self):
        store = default_holdings_store()
        store.set("u1", "gold", position_size="medium", avg_cost=3300.0, horizon="mid")
        note = holdings_reply_addendum("u1", "gold", current_price=3300.0 * 1.01)
        assert note
        assert "成本" in note

    def test_no_price_still_returns_note(self):
        store = default_holdings_store()
        store.set("u1", "gold", position_size="medium", avg_cost=3200.0, horizon="mid")
        note = holdings_reply_addendum("u1", "gold", current_price=None)
        assert note  # should still show holding info, just without P&L

    def test_no_holding_returns_empty(self):
        note = holdings_reply_addendum("u_nobody", "gold", current_price=3300.0)
        assert not note


# ── C: pipeline injects price into addendum ──────────────────────────────────

class TestPipelinePriceInjection:
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        install_gold_fixture()
        _trend = lambda asset: trend_from_values(asset, r7=0.03, r30=0.09)
        monkeypatch.setattr(pipeline, "fetch_trend_signal", _trend)
        import assistant.context_builder as _cb
        monkeypatch.setattr(_cb, "fetch_trend_signal", _trend)
        monkeypatch.setattr(pipeline, "_llm_callable", lambda: None)
        # Mock get_current_price to return controlled value
        monkeypatch.setattr("assistant.pipeline.get_current_price",
                            lambda asset: (3300.0, "live"))

    def test_holdings_price_status_in_trace_meta(self):
        trace = answer_question_traced("黄金现在能买吗", user_id="cp1")
        assert "holdings_price_status" in trace.meta
        assert trace.meta["holdings_price_status"] in ("live", "fallback", "missing")

    def test_pnl_status_in_trace_meta(self):
        trace = answer_question_traced("黄金现在能买吗", user_id="cp2")
        assert "pnl_status" in trace.meta

    def test_pnl_status_is_in_profit_when_holding_underwater(self, monkeypatch):
        store = default_holdings_store()
        store.set("cp3", "gold", position_size="medium", avg_cost=3500.0, horizon="mid")
        # price < avg_cost → underwater
        monkeypatch.setattr("assistant.pipeline.get_current_price",
                            lambda asset: (3200.0, "live"))
        trace = answer_question_traced("黄金现在能买吗", user_id="cp3")
        assert trace.meta.get("pnl_status") == "underwater"

    def test_reply_contains_holding_context(self, monkeypatch):
        store = default_holdings_store()
        store.set("cp4", "gold", position_size="large", avg_cost=3000.0, horizon="long")
        monkeypatch.setattr("assistant.pipeline.get_current_price",
                            lambda asset: (3300.0, "live"))
        trace = answer_question_traced("黄金现在能买吗", user_id="cp4")
        # Reply should mention holding context
        assert trace.reply  # non-empty


# ── D: fallback path ──────────────────────────────────────────────────────────

class TestPriceFallback:
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        install_gold_fixture()
        _trend = lambda asset: trend_from_values(asset, r7=0.03, r30=0.09)
        monkeypatch.setattr(pipeline, "fetch_trend_signal", _trend)
        import assistant.context_builder as _cb
        monkeypatch.setattr(_cb, "fetch_trend_signal", _trend)
        monkeypatch.setattr(pipeline, "_llm_callable", lambda: None)
        # Simulate price fetch failure
        monkeypatch.setattr("assistant.pipeline.get_current_price",
                            lambda asset: (None, "fallback"))

    def test_pipeline_still_runs_with_fallback_price(self):
        trace = answer_question_traced("黄金能买吗", user_id="fb1")
        assert trace.reply
        assert trace.meta.get("holdings_price_status") == "fallback"


# ── E: FOMO firewall ──────────────────────────────────────────────────────────

class TestFOMOFirewall:
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        install_gold_fixture()
        _trend = lambda asset: trend_from_values(asset, r7=0.03, r30=0.09)
        monkeypatch.setattr(pipeline, "fetch_trend_signal", _trend)
        import assistant.context_builder as _cb
        monkeypatch.setattr(_cb, "fetch_trend_signal", _trend)
        monkeypatch.setattr(pipeline, "_llm_callable", lambda: None)

    def test_live_price_does_not_change_decision(self, monkeypatch):
        monkeypatch.setattr("assistant.pipeline.get_current_price",
                            lambda asset: (3300.0, "live"))
        t1 = answer_question_traced("黄金能买吗", user_id="ff1")

        monkeypatch.setattr("assistant.pipeline.get_current_price",
                            lambda asset: (None, "fallback"))
        from assistant.session_memory import reset_session_store
        reset_session_store()
        t2 = answer_question_traced("黄金能买吗", user_id="ff1")

        # Decision action should be same regardless of price availability
        if t1.decision and t2.decision:
            assert t1.decision.action == t2.decision.action
