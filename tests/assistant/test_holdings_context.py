"""
Tests for Holdings / Portfolio context (Task 2 / v4).

Covers:
  - No position / small / large context blocks
  - Holdings persists to JSON and survives reload
  - Holdings injected into ContextPackage.to_prompt_block()
  - Holdings addendum in pipeline reply (with heavy position → risk warning)
  - Holdings do NOT directly change decision action
  - Telegram command parse helpers
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path

import assistant.pipeline as pipeline
from assistant.pipeline import answer_question_traced
from assistant.holdings import (
    Holding,
    HoldingsStore,
    build_holdings_context_block,
    holdings_reply_addendum,
    default_holdings_store,
    reset_holdings_store,
    parse_setholding_args,
    parse_clearholding_args,
)
from assistant.fixtures import install_gold_fixture
from assistant.trend_signals import trend_from_values
from assistant.user_profile import reset_profile_store
from assistant.company_context import reset_company_context


@pytest.fixture(autouse=True)
def _reset(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSISTANT_KNOWLEDGE_DB", str(tmp_path / "knowledge.sqlite3"))
    monkeypatch.setenv("HOLDINGS_FILE", str(tmp_path / "holdings.json"))
    from assistant.rag import store as _store_mod
    _store_mod._default = None
    reset_holdings_store()
    reset_profile_store()
    reset_company_context()
    yield
    _store_mod._default = None
    reset_holdings_store()
    reset_profile_store()
    reset_company_context()


@pytest.fixture
def gold_ready(monkeypatch):
    install_gold_fixture()
    _trend = lambda asset: trend_from_values(asset, r7=0.03, r30=0.09)
    monkeypatch.setattr(pipeline, "fetch_trend_signal", _trend)
    import assistant.context_builder as _cb
    monkeypatch.setattr(_cb, "fetch_trend_signal", _trend)
    monkeypatch.setattr(pipeline, "_llm_callable", lambda: None)


# ── HoldingsStore CRUD ────────────────────────────────────────────────────────

class TestHoldingsStore:
    def test_set_and_get(self, tmp_path):
        hs = HoldingsStore(path=tmp_path / "h.json")
        hs.set("42", "gold", position_size="small")
        h = hs.get("42", "gold")
        assert h is not None
        assert h.position_size == "small"
        assert h.asset == "gold"

    def test_get_none_for_unknown(self, tmp_path):
        hs = HoldingsStore(path=tmp_path / "h.json")
        assert hs.get("42", "gold") is None

    def test_clear_removes_holding(self, tmp_path):
        hs = HoldingsStore(path=tmp_path / "h.json")
        hs.set("1", "gold", position_size="medium")
        removed = hs.clear("1", "gold")
        assert removed is True
        assert hs.get("1", "gold") is None

    def test_clear_nonexistent_returns_false(self, tmp_path):
        hs = HoldingsStore(path=tmp_path / "h.json")
        assert hs.clear("1", "gold") is False

    def test_get_all_returns_all_holdings(self, tmp_path):
        hs = HoldingsStore(path=tmp_path / "h.json")
        hs.set("5", "gold", position_size="small")
        hs.set("5", "bitcoin", position_size="large")
        holdings = hs.get_all("5")
        assets = {h.asset for h in holdings}
        assert {"gold", "bitcoin"} == assets

    def test_int_user_id(self, tmp_path):
        hs = HoldingsStore(path=tmp_path / "h.json")
        hs.set(99, "gold", "medium")
        h = hs.get(99, "gold")
        assert h is not None
        assert h.position_size == "medium"


class TestHoldingsPersistence:
    def test_survives_reload(self, tmp_path):
        p = tmp_path / "h.json"
        hs1 = HoldingsStore(path=p)
        hs1.set("7", "gold", "large", horizon="mid")
        hs2 = HoldingsStore(path=p)
        h = hs2.get("7", "gold")
        assert h is not None
        assert h.position_size == "large"
        assert h.horizon == "mid"

    def test_schema_version_in_file(self, tmp_path):
        p = tmp_path / "h.json"
        hs = HoldingsStore(path=p)
        hs.set("1", "gold", "small")
        data = json.loads(p.read_text())
        assert "schema_version" in data

    def test_clear_persists(self, tmp_path):
        p = tmp_path / "h.json"
        hs1 = HoldingsStore(path=p)
        hs1.set("3", "gold", "small")
        hs1.clear("3", "gold")
        hs2 = HoldingsStore(path=p)
        assert hs2.get("3", "gold") is None


# ── Context blocks ────────────────────────────────────────────────────────────

class TestHoldingContextBlock:
    def test_no_position_block(self, tmp_path):
        hs = HoldingsStore(path=tmp_path / "h.json")
        hs.set("1", "gold", "none")
        h = hs.get("1", "gold")
        block = h.to_context_block()
        assert "无仓位" in block

    def test_small_position_block(self, tmp_path):
        hs = HoldingsStore(path=tmp_path / "h.json")
        hs.set("1", "gold", "small")
        h = hs.get("1", "gold")
        block = h.to_context_block()
        assert "small" in block

    def test_large_position_block(self, tmp_path):
        hs = HoldingsStore(path=tmp_path / "h.json")
        hs.set("1", "gold", "large")
        h = hs.get("1", "gold")
        block = h.to_context_block()
        assert "large" in block

    def test_none_returned_for_no_holding(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOLDINGS_FILE", str(tmp_path / "h.json"))
        # No holding set — build_holdings_context_block returns hint text
        block = build_holdings_context_block("1", "gold")
        assert "gold" in block


# ── Reply addendum ────────────────────────────────────────────────────────────

class TestHoldingsReplyAddendum:
    def test_empty_for_none_user(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOLDINGS_FILE", str(tmp_path / "h.json"))
        assert holdings_reply_addendum(None, "gold") == ""

    def test_empty_for_no_holding_recorded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOLDINGS_FILE", str(tmp_path / "h.json"))
        # user exists but no holding recorded
        assert holdings_reply_addendum("77", "gold") == ""

    def test_no_position_addendum(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOLDINGS_FILE", str(tmp_path / "h.json"))
        hs = default_holdings_store()
        hs.set("1", "gold", "none")
        text = holdings_reply_addendum("1", "gold")
        assert "没有持仓" in text or "建仓" in text

    def test_medium_position_addendum(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOLDINGS_FILE", str(tmp_path / "h.json"))
        hs = default_holdings_store()
        hs.set("1", "gold", "medium")
        text = holdings_reply_addendum("1", "gold")
        assert "medium" in text or "加仓" in text or "持有" in text

    def test_large_position_risk_warning(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOLDINGS_FILE", str(tmp_path / "h.json"))
        hs = default_holdings_store()
        hs.set("1", "gold", "large")
        text = holdings_reply_addendum("1", "gold")
        assert "重仓" in text or "风险" in text


# ── Holdings in pipeline ──────────────────────────────────────────────────────

class TestHoldingsInPipeline:
    def test_holding_in_context_pkg(self, gold_ready, tmp_path, monkeypatch):
        monkeypatch.setenv("HOLDINGS_FILE", str(tmp_path / "h.json"))
        hs = default_holdings_store()
        hs.set(42, "gold", "small")
        trace = answer_question_traced("我能不能买黄金", user_id=42)
        assert trace.context_pkg is not None
        assert trace.context_pkg.holding is not None
        assert trace.context_pkg.holding.position_size == "small"

    def test_holding_in_prompt_block(self, gold_ready, tmp_path, monkeypatch):
        monkeypatch.setenv("HOLDINGS_FILE", str(tmp_path / "h.json"))
        hs = default_holdings_store()
        hs.set(42, "gold", "large")
        trace = answer_question_traced("我能不能买黄金", user_id=42)
        block = trace.context_pkg.to_prompt_block()
        assert "持仓" in block

    def test_heavy_position_risk_in_reply(self, gold_ready, tmp_path, monkeypatch):
        monkeypatch.setenv("HOLDINGS_FILE", str(tmp_path / "h.json"))
        hs = default_holdings_store()
        hs.set(42, "gold", "large")
        trace = answer_question_traced("我能不能买黄金", user_id=42)
        assert "重仓" in trace.reply or "风险" in trace.reply

    def test_no_holding_no_addendum(self, gold_ready):
        trace = answer_question_traced("我能不能买黄金", user_id=None)
        # No user_id → no holdings lookup → no addendum
        assert trace.context_pkg is not None
        assert trace.context_pkg.holding is None

    def test_holdings_do_not_change_decision_action(self, gold_ready, tmp_path, monkeypatch):
        """Holdings only change wording, not market decision."""
        monkeypatch.setenv("HOLDINGS_FILE", str(tmp_path / "h.json"))
        hs = default_holdings_store()

        trace_no_holding = answer_question_traced("我能不能买黄金", user_id=None)
        hs.set(55, "gold", "large")
        trace_heavy = answer_question_traced("我能不能买黄金", user_id=55)

        # Same market data → same decision action
        if trace_no_holding.decision and trace_heavy.decision:
            assert trace_no_holding.decision.action == trace_heavy.decision.action


# ── Command parse helpers ─────────────────────────────────────────────────────

class TestCommandParsers:
    def test_parse_setholding_basic(self):
        asset, size, cost, horizon = parse_setholding_args("/setholding gold small")
        assert asset == "gold"
        assert size == "small"

    def test_parse_setholding_large(self):
        asset, size, cost, horizon = parse_setholding_args("/setholding bitcoin large")
        assert asset == "bitcoin"
        assert size == "large"

    def test_parse_setholding_default_size(self):
        asset, size, cost, horizon = parse_setholding_args("/setholding gold")
        assert asset == "gold"
        assert size == "small"

    def test_parse_setholding_no_args(self):
        asset, _, _c, _h = parse_setholding_args("/setholding")
        assert asset is None

    def test_parse_clearholding(self):
        asset = parse_clearholding_args("/clearholding gold")
        assert asset == "gold"

    def test_parse_clearholding_no_args(self):
        asset = parse_clearholding_args("/clearholding")
        assert asset is None
