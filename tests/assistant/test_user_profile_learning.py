"""
Tests for UserProfile interaction-based learning (CL3).

Rules:
  - Only updates profiles that already exist (no auto-creation for unknowns)
  - Conservative thresholds (fomo=3, asset=2, concise=3)
  - Updates are idempotent once threshold is crossed (no duplicate traits)
"""
from __future__ import annotations

import pytest
from assistant.user_profile import (
    UserProfile,
    UserProfileStore,
    get_profile_store,
    reset_profile_store,
    update_profile_from_interaction,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_profile_store()
    yield
    reset_profile_store()


@pytest.fixture
def store_with_user() -> tuple[UserProfileStore, UserProfile]:
    store = get_profile_store()
    p = UserProfile(user_id="42", role="retail")
    store.set(p)
    return store, p


class TestFomoLearning:
    def test_fomo_added_after_threshold(self, store_with_user):
        store, _ = store_with_user
        for _ in range(3):
            store.update_from_interaction("42", "fomo")
        p = store.get("42")
        assert "fomo_prone" in p.behavior_traits

    def test_fomo_not_added_below_threshold(self, store_with_user):
        store, _ = store_with_user
        for _ in range(2):
            store.update_from_interaction("42", "fomo")
        p = store.get("42")
        assert "fomo_prone" not in p.behavior_traits

    def test_fomo_not_duplicated(self, store_with_user):
        store, _ = store_with_user
        for _ in range(6):
            store.update_from_interaction("42", "fomo")
        p = store.get("42")
        assert p.behavior_traits.count("fomo_prone") == 1

    def test_no_update_for_unknown_user(self):
        store = get_profile_store()
        # user "99" was never registered — should not create a new profile
        result = store.update_from_interaction("99", "fomo")
        assert result is False

    def test_no_update_for_none_user(self):
        store = get_profile_store()
        result = store.update_from_interaction(None, "fomo")
        assert result is False


class TestAssetInterestLearning:
    def test_interest_added_after_threshold(self, store_with_user):
        store, _ = store_with_user
        for _ in range(2):
            store.update_from_interaction("42", "asset_mention", "gold")
        p = store.get("42")
        assert "gold" in p.interests

    def test_interest_not_added_below_threshold(self, store_with_user):
        store, _ = store_with_user
        store.update_from_interaction("42", "asset_mention", "gold")
        p = store.get("42")
        assert "gold" not in p.interests

    def test_interest_not_duplicated(self, store_with_user):
        store, _ = store_with_user
        for _ in range(5):
            store.update_from_interaction("42", "asset_mention", "gold")
        p = store.get("42")
        assert p.interests.count("gold") == 1

    def test_different_assets_tracked_separately(self, store_with_user):
        store, _ = store_with_user
        for _ in range(2):
            store.update_from_interaction("42", "asset_mention", "gold")
        for _ in range(2):
            store.update_from_interaction("42", "asset_mention", "bitcoin")
        p = store.get("42")
        assert "gold" in p.interests
        assert "bitcoin" in p.interests

    def test_existing_interest_not_duplicated(self, store_with_user):
        store, _ = store_with_user
        # "gold" already in interests
        store.set(UserProfile(user_id="42", role="retail", interests=["gold"]))
        for _ in range(3):
            store.update_from_interaction("42", "asset_mention", "gold")
        p = store.get("42")
        assert p.interests.count("gold") == 1


class TestConciseLearning:
    def test_concise_style_after_threshold(self, store_with_user):
        store, _ = store_with_user
        for _ in range(3):
            store.update_from_interaction("42", "concise_feedback")
        p = store.get("42")
        assert p.preferred_style == "concise"

    def test_concise_not_changed_below_threshold(self, store_with_user):
        store, _ = store_with_user
        original = store.get("42").preferred_style
        for _ in range(2):
            store.update_from_interaction("42", "concise_feedback")
        p = store.get("42")
        assert p.preferred_style == original


class TestPublicHelper:
    def test_update_profile_from_interaction_helper(self):
        store = get_profile_store()
        store.set(UserProfile(user_id="77", role="trader"))
        for _ in range(3):
            update_profile_from_interaction("77", "fomo")
        p = store.get("77")
        assert "fomo_prone" in p.behavior_traits

    def test_int_user_id_works(self):
        store = get_profile_store()
        store.set(UserProfile(user_id="55", role="retail"))
        for _ in range(3):
            update_profile_from_interaction(55, "fomo")  # int user_id
        p = store.get(55)
        assert "fomo_prone" in p.behavior_traits

    def test_returns_true_on_update(self):
        store = get_profile_store()
        store.set(UserProfile(user_id="33", role="retail"))
        # First 2 calls → below threshold → False
        for _ in range(2):
            result = update_profile_from_interaction("33", "fomo")
        assert result is False
        # 3rd call → threshold hit → True
        result = update_profile_from_interaction("33", "fomo")
        assert result is True
