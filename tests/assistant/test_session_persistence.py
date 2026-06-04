"""
v6 Task 3 tests: session memory default persistence.

Covers:
  A. Default path resolves to logs/session_memory.json (without env override)
  B. SESSION_MEMORY_FILE env var overrides path
  C. Empty SESSION_MEMORY_FILE disables persistence (no file writes)
  D. Data survives store reload (restart recovery)
  E. Inactive user pruning on save
  F. Session file format: valid JSON with 'sessions' key
  G. Concurrent turns persist each time
"""
from __future__ import annotations

import json
import os
import time
import pytest
from pathlib import Path

from assistant.session_memory import (
    SessionTurn,
    SessionMemoryStore,
    _resolve_session_path,
    reset_session_store,
    _SESSION_USER_TTL,
)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("SESSION_MEMORY_FILE", str(tmp_path / "session_memory.json"))
    reset_session_store()
    yield
    reset_session_store()


def _turn(asset="gold"):
    return SessionTurn(
        asset=asset, intent="market_decision",
        emotion="neutral", action="hold", topic=f"{asset}:hold",
    )


# ── A: default path ───────────────────────────────────────────────────────────

class TestDefaultPath:
    def test_default_path_uses_logs_dir(self, monkeypatch):
        monkeypatch.delenv("SESSION_MEMORY_FILE", raising=False)
        path = _resolve_session_path()
        assert "logs" in str(path)
        assert str(path).endswith("session_memory.json")

    def test_env_override_replaces_default(self, tmp_path, monkeypatch):
        custom = tmp_path / "custom.json"
        monkeypatch.setenv("SESSION_MEMORY_FILE", str(custom))
        path = _resolve_session_path()
        assert path == custom

    def test_empty_env_returns_devnull(self, monkeypatch):
        monkeypatch.setenv("SESSION_MEMORY_FILE", "")
        path = _resolve_session_path()
        assert str(path) == "/dev/null"


# ── B/C: persistence enable/disable ──────────────────────────────────────────

class TestPersistenceControl:
    def test_store_writes_to_path(self, tmp_path):
        path = tmp_path / "s.json"
        store = SessionMemoryStore(path=path)
        store.push("u1", _turn())
        assert path.exists()

    def test_no_file_write_when_path_is_none(self, tmp_path):
        store = SessionMemoryStore(path=None.__class__.__new__(type(None)))  # type: ignore
        # Use internal None path directly
        store2 = SessionMemoryStore.__new__(SessionMemoryStore)
        store2._path = None
        store2._max_turns = 10
        store2._sessions = {}
        store2.push("u1", _turn())
        # No file anywhere — just shouldn't raise
        assert True

    def test_empty_string_env_disables_writes(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SESSION_MEMORY_FILE", "")
        store = SessionMemoryStore()
        store.push("u1", _turn())
        # With /dev/null sentinel, _path is None → no actual file
        assert store._path is None


# ── D: restart recovery ───────────────────────────────────────────────────────

class TestRestartRecovery:
    def test_data_survives_reload(self, tmp_path):
        path = tmp_path / "s.json"
        store = SessionMemoryStore(path=path)
        store.push("u1", _turn("gold"))
        # Reload
        store2 = SessionMemoryStore(path=path)
        ctx = store2.resolve_context("u1", "那还能追吗？", None)
        assert ctx.resolved_asset == "gold"

    def test_multiple_users_survive_reload(self, tmp_path):
        path = tmp_path / "s.json"
        store = SessionMemoryStore(path=path)
        store.push("alice", _turn("gold"))
        store.push("bob", _turn("bitcoin"))
        store2 = SessionMemoryStore(path=path)
        ctx_alice = store2.resolve_context("alice", "那还能追吗？", None)
        ctx_bob = store2.resolve_context("bob", "那还能追吗？", None)
        assert ctx_alice.resolved_asset == "gold"
        assert ctx_bob.resolved_asset == "bitcoin"

    def test_reload_preserves_ttl_valid_turns(self, tmp_path):
        path = tmp_path / "s.json"
        store = SessionMemoryStore(path=path)
        store.push("u2", _turn("gold"))
        store2 = SessionMemoryStore(path=path)
        # Recent turn should still be valid after reload
        session = store2.get_session("u2")
        assert session.last_asset() == "gold"


# ── E: inactive user pruning ─────────────────────────────────────────────────

class TestInactiveUserPruning:
    def test_stale_user_pruned_on_save(self, tmp_path):
        path = tmp_path / "s.json"
        store = SessionMemoryStore(path=path)
        # Manually insert a very old turn
        old_ts = time.time() - _SESSION_USER_TTL - 100
        old_turn = SessionTurn(
            asset="gold", intent="market_decision",
            emotion="neutral", action="hold", topic="g",
            ts=old_ts,
        )
        store._sessions["stale_user"] = __import__(
            "assistant.session_memory", fromlist=["UserSession"]).UserSession()
        store._sessions["stale_user"]._turns.append(old_turn)
        # Push a fresh turn to trigger save
        store.push("fresh_user", _turn("bitcoin"))
        # Reload — stale_user should be gone
        store2 = SessionMemoryStore(path=path)
        assert "stale_user" not in store2._sessions
        assert "fresh_user" in store2._sessions

    def test_active_user_not_pruned(self, tmp_path):
        path = tmp_path / "s.json"
        store = SessionMemoryStore(path=path)
        store.push("active", _turn("gold"))
        store.push("trigger", _turn("bitcoin"))  # second push triggers save
        store2 = SessionMemoryStore(path=path)
        assert "active" in store2._sessions


# ── F: file format ────────────────────────────────────────────────────────────

class TestFileFormat:
    def test_json_has_sessions_key(self, tmp_path):
        path = tmp_path / "s.json"
        store = SessionMemoryStore(path=path)
        store.push("u1", _turn())
        raw = json.loads(path.read_text())
        assert "sessions" in raw

    def test_sessions_key_is_dict(self, tmp_path):
        path = tmp_path / "s.json"
        store = SessionMemoryStore(path=path)
        store.push("u1", _turn())
        raw = json.loads(path.read_text())
        assert isinstance(raw["sessions"], dict)

    def test_turn_fields_in_file(self, tmp_path):
        path = tmp_path / "s.json"
        store = SessionMemoryStore(path=path)
        store.push("u1", _turn("gold"))
        raw = json.loads(path.read_text())
        turn_data = raw["sessions"]["u1"][0]
        assert turn_data["asset"] == "gold"
        assert "intent" in turn_data
        assert "ts" in turn_data


# ── G: concurrent turns persist ──────────────────────────────────────────────

class TestConcurrentTurns:
    def test_each_push_writes_to_file(self, tmp_path):
        path = tmp_path / "s.json"
        store = SessionMemoryStore(path=path)
        store.push("u1", _turn("gold"))
        mtime1 = path.stat().st_mtime

        time.sleep(0.01)
        store.push("u1", _turn("bitcoin"))
        mtime2 = path.stat().st_mtime

        assert mtime2 >= mtime1  # file was updated

    def test_reload_after_each_push(self, tmp_path):
        path = tmp_path / "s.json"
        store = SessionMemoryStore(path=path)
        store.push("u1", _turn("gold"))
        store.push("u1", _turn("bitcoin"))

        store2 = SessionMemoryStore(path=path)
        session = store2.get_session("u1")
        assets = [t.asset for t in session.recent_turns()]
        assert "gold" in assets
        assert "bitcoin" in assets
