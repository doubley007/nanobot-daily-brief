"""Tests for user_profile module."""
from __future__ import annotations

import json
import pytest
from assistant.user_profile import (
    UserProfile,
    UserProfileStore,
    DEFAULT_PROFILE,
    get_user_profile,
    reset_profile_store,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_profile_store()
    yield
    reset_profile_store()


class TestUserProfile:
    def test_default_profile_fields(self):
        p = DEFAULT_PROFILE
        assert p.user_id == "default"
        assert p.role == "unknown"
        assert p.risk_preference == "moderate"
        assert p.preferred_style == "analytical"

    def test_needs_simplified_language_retail(self):
        p = UserProfile(user_id="1", role="retail", is_internal=False)
        assert p.needs_simplified_language is True

    def test_needs_simplified_language_internal(self):
        p = UserProfile(user_id="1", role="retail", is_internal=True)
        assert p.needs_simplified_language is False

    def test_wants_concise_reply_insider(self):
        p = UserProfile(user_id="1", role="insider")
        assert p.wants_concise_reply is True

    def test_wants_concise_reply_unknown(self):
        p = UserProfile(user_id="1", role="unknown")
        assert p.wants_concise_reply is False

    def test_fomo_prone_trait(self):
        p = UserProfile(user_id="1", behavior_traits=["fomo_prone"])
        assert p.fomo_prone is True
        p2 = UserProfile(user_id="2", behavior_traits=["long_term_holder"])
        assert p2.fomo_prone is False

    def test_to_context_block_contains_role(self):
        p = UserProfile(user_id="99", role="analyst", preferred_style="concise",
                        is_internal=True)
        block = p.to_context_block()
        assert "analyst" in block
        assert "内部用户" in block

    def test_to_dict(self):
        p = UserProfile(user_id="42", role="pm", interests=["gold", "macro"])
        d = p.to_dict()
        assert d["user_id"] == "42"
        assert d["role"] == "pm"
        assert "gold" in d["interests"]


class TestUserProfileStore:
    def test_get_unknown_returns_default(self):
        store = UserProfileStore()
        p = store.get("unknown_xyz_123")
        assert p is DEFAULT_PROFILE

    def test_get_none_returns_default(self):
        store = UserProfileStore()
        p = store.get(None)
        assert p is DEFAULT_PROFILE

    def test_set_and_get(self):
        store = UserProfileStore()
        profile = UserProfile(user_id="123", role="trader", is_internal=True)
        store.set(profile)
        retrieved = store.get("123")
        assert retrieved.role == "trader"
        assert retrieved.is_internal is True

    def test_telegram_int_user_id(self):
        store = UserProfileStore()
        profile = UserProfile(user_id="9876543", role="pm")
        store.set(profile)
        # telegram sends int user_id
        retrieved = store.get(9876543)
        assert retrieved.role == "pm"

    def test_load_from_json_file(self, tmp_path, monkeypatch):
        profiles = {
            "profiles": [
                {
                    "user_id": "777",
                    "display_name": "Boss",
                    "role": "insider",
                    "risk_preference": "aggressive",
                    "preferred_style": "concise",
                    "interests": ["gold", "bitcoin"],
                    "behavior_traits": ["high_conviction"],
                    "language": "zh",
                    "is_internal": True,
                }
            ]
        }
        f = tmp_path / "profiles.json"
        f.write_text(json.dumps(profiles))
        monkeypatch.setenv("USER_PROFILES_FILE", str(f))
        store = UserProfileStore()
        p = store.get("777")
        assert p.role == "insider"
        assert p.is_internal is True
        assert "gold" in p.interests

    def test_reload_resets_profiles(self):
        store = UserProfileStore()
        store.set(UserProfile(user_id="abc", role="pm"))
        store.reload()
        p = store.get("abc")
        assert p is DEFAULT_PROFILE


class TestGetUserProfile:
    def test_get_user_profile_fallback(self):
        p = get_user_profile(None)
        assert p is DEFAULT_PROFILE

    def test_get_user_profile_unknown_id(self):
        p = get_user_profile("no_such_user_00000")
        assert p.role == "unknown"
