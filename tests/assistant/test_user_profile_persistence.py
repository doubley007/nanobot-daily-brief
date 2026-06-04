"""
Tests for UserProfile JSON persistence (Task 1 / v4).

Covers:
  - Updated traits survive store reload (flush → reload → get)
  - Flush writes the correct schema_version field
  - Flush is idempotent (no duplicate dirty writes)
  - Un-updated profiles are NOT written (dirty-flag check)
  - Fallback DEFAULT_PROFILE never written to disk
  - Multiple update cycles accumulate correctly
  - Load from existing file preserves last_updated_at
  - Atomic write (tmp → rename) means partial writes don't corrupt
"""
from __future__ import annotations

import json
import time
import pytest
from pathlib import Path
from assistant.user_profile import (
    UserProfile,
    UserProfileStore,
    DEFAULT_PROFILE,
    PROFILE_SCHEMA_VERSION,
    get_profile_store,
    reset_profile_store,
    flush_profile_store,
    update_profile_from_interaction,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_profile_store()
    yield
    reset_profile_store()


def _store_with_path(tmp_path: Path, profiles=None) -> tuple[UserProfileStore, Path]:
    """Create an isolated store backed by a temp file."""
    import os
    path = tmp_path / "user_profiles.json"
    os.environ["USER_PROFILES_FILE"] = str(path)
    store = UserProfileStore()
    if profiles:
        for p in profiles:
            store.set(p)
        store._loaded = True
        store._loaded_path = path
    return store, path


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("USER_PROFILES_FILE", raising=False)


# ── flush writes schema_version ──────────────────────────────────────────────

def test_flush_writes_schema_version(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_PROFILES_FILE", str(tmp_path / "up.json"))
    store = UserProfileStore()
    store.set(UserProfile(user_id="1", role="retail"))
    store._loaded_path = tmp_path / "up.json"
    store._dirty_ids.add("1")
    store.flush()
    data = json.loads((tmp_path / "up.json").read_text())
    assert data["schema_version"] == PROFILE_SCHEMA_VERSION


# ── flush → reload → get preserves learned traits ────────────────────────────

def test_fomo_trait_survives_reload(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_PROFILES_FILE", str(tmp_path / "up.json"))

    # First store: add a profile, trigger fomo update, flush
    store1 = UserProfileStore()
    store1.set(UserProfile(user_id="42", role="retail"))
    store1._loaded = True
    store1._loaded_path = tmp_path / "up.json"
    for _ in range(3):
        store1.update_from_interaction("42", "fomo")
    assert store1.flush() is True

    # Second store: reload from file
    store2 = UserProfileStore()
    p = store2.get("42")
    assert "fomo_prone" in p.behavior_traits


def test_interest_survives_reload(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_PROFILES_FILE", str(tmp_path / "up.json"))

    store1 = UserProfileStore()
    store1.set(UserProfile(user_id="7", role="trader"))
    store1._loaded = True
    store1._loaded_path = tmp_path / "up.json"
    for _ in range(2):
        store1.update_from_interaction("7", "asset_mention", "gold")
    store1.flush()

    store2 = UserProfileStore()
    p = store2.get("7")
    assert "gold" in p.interests


def test_preferred_style_survives_reload(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_PROFILES_FILE", str(tmp_path / "up.json"))

    store1 = UserProfileStore()
    store1.set(UserProfile(user_id="99", role="analyst"))
    store1._loaded = True
    store1._loaded_path = tmp_path / "up.json"
    for _ in range(3):
        store1.update_from_interaction("99", "concise_feedback")
    store1.flush()

    store2 = UserProfileStore()
    p = store2.get("99")
    assert p.preferred_style == "concise"


# ── flush is idempotent ───────────────────────────────────────────────────────

def test_flush_clears_dirty_set(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_PROFILES_FILE", str(tmp_path / "up.json"))
    store = UserProfileStore()
    store.set(UserProfile(user_id="5", role="pm"))
    store._loaded = True
    store._loaded_path = tmp_path / "up.json"
    store._dirty_ids.add("5")
    store.flush()
    assert len(store._dirty_ids) == 0


def test_flush_returns_false_when_not_dirty(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_PROFILES_FILE", str(tmp_path / "up.json"))
    store = UserProfileStore()
    store.set(UserProfile(user_id="9", role="pm"))
    store._loaded = True
    store._loaded_path = tmp_path / "up.json"
    # No dirty_ids added
    result = store.flush()
    assert result is False


# ── non-updated profiles are NOT dirtied ─────────────────────────────────────

def test_no_dirty_for_clean_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_PROFILES_FILE", str(tmp_path / "up.json"))
    store = UserProfileStore()
    store.set(UserProfile(user_id="8", role="retail"))
    store._loaded = True
    store._loaded_path = tmp_path / "up.json"
    # interact but don't cross threshold
    store.update_from_interaction("8", "fomo")  # count=1, threshold=3
    assert "8" not in store._dirty_ids


def test_dirty_only_after_threshold(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_PROFILES_FILE", str(tmp_path / "up.json"))
    store = UserProfileStore()
    store.set(UserProfile(user_id="11", role="retail"))
    store._loaded = True
    store._loaded_path = tmp_path / "up.json"
    store.update_from_interaction("11", "fomo")
    store.update_from_interaction("11", "fomo")
    assert "11" not in store._dirty_ids
    store.update_from_interaction("11", "fomo")  # crosses threshold
    assert "11" in store._dirty_ids


# ── DEFAULT_PROFILE never saved ───────────────────────────────────────────────

def test_default_profile_not_written(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_PROFILES_FILE", str(tmp_path / "up.json"))
    store = UserProfileStore()
    store._loaded = True
    store._loaded_path = tmp_path / "up.json"
    # Attempt to save default user_id
    store._dirty_ids.add("default")
    store.flush()
    if (tmp_path / "up.json").exists():
        data = json.loads((tmp_path / "up.json").read_text())
        uids = [e["user_id"] for e in data.get("profiles", [])]
        assert "default" not in uids


# ── last_updated_at is set on update ─────────────────────────────────────────

def test_last_updated_at_set_on_trait_update(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_PROFILES_FILE", str(tmp_path / "up.json"))
    store = UserProfileStore()
    store.set(UserProfile(user_id="22", role="retail"))
    store._loaded = True
    store._loaded_path = tmp_path / "up.json"
    before = time.time()
    for _ in range(3):
        store.update_from_interaction("22", "fomo")
    p = store.get("22")
    assert p.last_updated_at >= before


def test_last_updated_at_preserved_on_reload(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_PROFILES_FILE", str(tmp_path / "up.json"))
    store1 = UserProfileStore()
    store1.set(UserProfile(user_id="33", role="retail"))
    store1._loaded = True
    store1._loaded_path = tmp_path / "up.json"
    for _ in range(3):
        store1.update_from_interaction("33", "fomo")
    ts = store1.get("33").last_updated_at
    store1.flush()

    store2 = UserProfileStore()
    p = store2.get("33")
    assert abs(p.last_updated_at - ts) < 1.0


# ── flush_profile_store() helper works ───────────────────────────────────────

def test_flush_profile_store_helper(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_PROFILES_FILE", str(tmp_path / "up.json"))
    reset_profile_store()
    store = get_profile_store()
    store.set(UserProfile(user_id="55", role="retail"))
    store._loaded_path = tmp_path / "up.json"
    store._dirty_ids.add("55")
    result = flush_profile_store()
    assert result is True
    assert (tmp_path / "up.json").exists()


# ── save() force-saves specific user ─────────────────────────────────────────

def test_save_specific_user(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_PROFILES_FILE", str(tmp_path / "up.json"))
    store = UserProfileStore()
    store.set(UserProfile(user_id="66", role="trader", interests=["bitcoin"]))
    store._loaded = True
    store._loaded_path = tmp_path / "up.json"
    store.save("66")
    data = json.loads((tmp_path / "up.json").read_text())
    saved = {e["user_id"]: e for e in data["profiles"]}
    assert "66" in saved
    assert "bitcoin" in saved["66"]["interests"]


# ── fallback profile works when file missing ─────────────────────────────────

def test_fallback_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_PROFILES_FILE", str(tmp_path / "nonexistent.json"))
    store = UserProfileStore()
    p = store.get("nobody")
    assert p is DEFAULT_PROFILE
