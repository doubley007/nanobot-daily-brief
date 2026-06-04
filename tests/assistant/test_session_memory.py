"""
Tests for session memory (Task 3 / v5).

Covers:
  A. SessionTurn: store and retrieve
  B. Follow-up asset resolution
  C. Follow-up continuity: "那现在还能追吗？" inherits previous asset
  D. Explicit new asset overrides session
  E. Session isolation between users
  F. Session TTL (expired turns not used)
  G. Pipeline: is_followup flag in trace.meta
  H. Persistent JSON storage
"""
from __future__ import annotations

import time
import json
import pytest
from pathlib import Path

import assistant.pipeline as pipeline
from assistant.pipeline import answer_question_traced
from assistant.session_memory import (
    SessionTurn,
    UserSession,
    SessionMemoryStore,
    reset_session_store,
    record_turn,
    resolve_session_context,
    _looks_like_followup,
)
from assistant.fixtures import install_gold_fixture, install_bitcoin_fixture
from assistant.trend_signals import trend_from_values
from assistant.user_profile import reset_profile_store
from assistant.company_context import reset_company_context


@pytest.fixture(autouse=True)
def _reset(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSISTANT_KNOWLEDGE_DB", str(tmp_path / "knowledge.sqlite3"))
    monkeypatch.setenv("VECTOR_INDEX_DIR", str(tmp_path / "vidx"))
    # Redirect session persistence to tmp path so tests don't write to logs/
    monkeypatch.setenv("SESSION_MEMORY_FILE", str(tmp_path / "session_memory.json"))
    from assistant.rag import store as _store_mod
    _store_mod._default = None
    reset_session_store()
    reset_profile_store()
    reset_company_context()
    from assistant.rag.vector_store import reset_vector_store
    reset_vector_store()
    yield
    _store_mod._default = None
    reset_session_store()
    reset_profile_store()
    reset_company_context()
    reset_vector_store()


# ── A: SessionTurn store/retrieve ─────────────────────────────────────────────

class TestSessionTurn:
    def test_push_and_last_turn(self):
        session = UserSession()
        t = SessionTurn(asset="gold", intent="market_decision",
                        emotion="neutral", action="buy_consider", topic="gold:buy")
        session.push(t)
        assert session.last_turn() is t

    def test_max_turns_ring_buffer(self):
        session = UserSession(max_turns=3)
        for i in range(5):
            session.push(SessionTurn(
                asset=f"a{i}", intent="emotional_chat",
                emotion="neutral", action="unknown", topic=f"t{i}"))
        turns = session.recent_turns()
        assert len(turns) == 3
        # Most recent should be last three
        assets = [t.asset for t in turns]
        assert assets == ["a2", "a3", "a4"]

    def test_last_asset_skips_none(self):
        session = UserSession()
        session.push(SessionTurn(asset="gold", intent="market_decision",
                                  emotion="neutral", action="hold", topic="g"))
        session.push(SessionTurn(asset=None, intent="emotional_chat",
                                  emotion="fomo", action="unknown", topic="e"))
        assert session.last_asset() == "gold"

    def test_serialization_round_trip(self):
        session = UserSession()
        session.push(SessionTurn(asset="bitcoin", intent="market_decision",
                                  emotion="fomo", action="avoid", topic="btc:avoid"))
        data = session.to_list()
        restored = UserSession.from_list(data)
        assert restored.last_turn().asset == "bitcoin"
        assert restored.last_turn().action == "avoid"


# ── B/C: Follow-up resolution ─────────────────────────────────────────────────

class TestFollowupResolution:
    def _session_with_gold(self) -> UserSession:
        session = UserSession()
        session.push(SessionTurn(
            asset="gold", intent="market_decision",
            emotion="neutral", action="buy_consider", topic="gold:buy"))
        return session

    def test_followup_inherits_asset(self):
        session = self._session_with_gold()
        ctx = session.resolve_context("那现在还能追吗？", detected_asset=None)
        assert ctx.resolved_asset == "gold"
        assert ctx.is_followup is True

    def test_new_asset_overrides_session(self):
        session = self._session_with_gold()
        ctx = session.resolve_context("比特币怎么样？", detected_asset="bitcoin")
        assert ctx.resolved_asset == "bitcoin"
        assert ctx.is_followup is False

    def test_no_previous_session_no_followup(self):
        session = UserSession()
        ctx = session.resolve_context("那现在还能追吗？", detected_asset=None)
        assert ctx.resolved_asset is None
        assert ctx.is_followup is False

    def test_short_message_detected_as_followup(self):
        session = self._session_with_gold()
        ctx = session.resolve_context("还能追吗", detected_asset=None)
        assert ctx.is_followup is True
        assert ctx.resolved_asset == "gold"

    def test_recent_action_from_last_turn(self):
        session = self._session_with_gold()
        ctx = session.resolve_context("如果我已经买了呢？", detected_asset=None)
        assert ctx.recent_action == "buy_consider"


# ── D: _looks_like_followup heuristic ────────────────────────────────────────

class TestFollowupDetection:
    def test_followup_phrases_detected(self):
        assert _looks_like_followup("那现在还能追吗？")
        assert _looks_like_followup("如果我已经买了呢？")
        assert _looks_like_followup("然后呢")
        # "那比特币呢" is an asset-switch (score=0.5) in v6, not a generic follow-up
        # It resolves to bitcoin as a new asset, so _looks_like_followup returns False
        assert not _looks_like_followup("那比特币呢")

    def test_new_independent_question_not_followup(self):
        assert not _looks_like_followup("黄金今天能买吗，我关注了好久了")
        assert not _looks_like_followup("英伟达最新财报怎么样")

    def test_short_message_is_followup(self):
        assert _looks_like_followup("还能追")
        assert _looks_like_followup("是吗")


# ── E: Session isolation ──────────────────────────────────────────────────────

class TestSessionIsolation:
    def test_different_users_isolated(self):
        store = SessionMemoryStore()
        store.push("user_a", SessionTurn(
            asset="gold", intent="market_decision",
            emotion="neutral", action="buy_consider", topic="g"))
        store.push("user_b", SessionTurn(
            asset="bitcoin", intent="market_decision",
            emotion="fomo", action="avoid", topic="b"))
        ctx_a = store.resolve_context("user_a", "那现在还能追吗？", None)
        ctx_b = store.resolve_context("user_b", "那现在还能追吗？", None)
        assert ctx_a.resolved_asset == "gold"
        assert ctx_b.resolved_asset == "bitcoin"

    def test_clear_user(self):
        store = SessionMemoryStore()
        store.push("user_x", SessionTurn(
            asset="gold", intent="market_decision",
            emotion="neutral", action="hold", topic="g"))
        store.clear_user("user_x")
        ctx = store.resolve_context("user_x", "那呢？", None)
        assert ctx.resolved_asset is None


# ── F: Session TTL ────────────────────────────────────────────────────────────

class TestSessionTTL:
    def test_expired_turns_not_used(self):
        session = UserSession()
        old_turn = SessionTurn(
            asset="gold", intent="market_decision",
            emotion="neutral", action="buy_consider", topic="g",
            ts=time.time() - 7200,  # 2 hours ago
        )
        session._turns.append(old_turn)
        # TTL = 3600s, so this should be excluded
        ctx = session.resolve_context("那还能追吗？", detected_asset=None)
        assert ctx.resolved_asset is None  # expired turn not used

    def test_recent_turn_used(self):
        session = UserSession()
        fresh_turn = SessionTurn(
            asset="gold", intent="market_decision",
            emotion="neutral", action="hold", topic="g",
            ts=time.time() - 300,  # 5 min ago
        )
        session._turns.append(fresh_turn)
        ctx = session.resolve_context("那还能追吗？", detected_asset=None)
        assert ctx.resolved_asset == "gold"


# ── G: Pipeline integration ────────────────────────────────────────────────────

class TestPipelineSession:
    @pytest.fixture(autouse=True)
    def gold_ready(self, monkeypatch):
        install_gold_fixture()
        _trend = lambda asset: trend_from_values(asset, r7=0.03, r30=0.09)
        monkeypatch.setattr(pipeline, "fetch_trend_signal", _trend)
        import assistant.context_builder as _cb
        monkeypatch.setattr(_cb, "fetch_trend_signal", _trend)
        monkeypatch.setattr(pipeline, "_llm_callable", lambda: None)

    def test_is_followup_false_on_first_question(self):
        trace = answer_question_traced("黄金现在能买吗", user_id=1)
        assert "is_followup" in trace.meta
        assert trace.meta["is_followup"] is False

    def test_session_asset_present_in_meta(self):
        trace = answer_question_traced("黄金现在能买吗", user_id=2)
        assert "session_asset" in trace.meta

    def test_followup_inherits_asset_in_pipeline(self):
        # First turn establishes gold context
        reset_session_store()
        t1 = answer_question_traced("黄金现在能买吗", user_id=77)
        assert t1.route.asset == "gold"
        # Second turn is a follow-up with no explicit asset
        t2 = answer_question_traced("那现在还能追吗？", user_id=77)
        # Should resolve gold from session
        assert t2.meta.get("session_asset") == "gold" or \
               t2.route.asset == "gold" or \
               t2.meta.get("is_followup") is True


# ── H: Persistent JSON storage ────────────────────────────────────────────────

class TestSessionPersistence:
    def test_sessions_persist_to_json(self, tmp_path):
        path = tmp_path / "sessions.json"
        store = SessionMemoryStore(path=path)
        store.push("u1", SessionTurn(
            asset="gold", intent="market_decision",
            emotion="neutral", action="hold", topic="g"))
        assert path.exists()
        raw = json.loads(path.read_text())
        assert "u1" in raw["sessions"]

    def test_sessions_survive_reload(self, tmp_path):
        path = tmp_path / "sessions.json"
        store = SessionMemoryStore(path=path)
        store.push("u1", SessionTurn(
            asset="bitcoin", intent="market_decision",
            emotion="fomo", action="avoid", topic="btc"))
        # Reload
        store2 = SessionMemoryStore(path=path)
        ctx = store2.resolve_context("u1", "那还能追吗？", None)
        assert ctx.resolved_asset == "bitcoin"
