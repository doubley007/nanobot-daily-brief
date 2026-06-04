"""Tests for context_builder module."""
from __future__ import annotations

import pytest
import assistant.pipeline as pipeline
from assistant.context_builder import build_context, ContextPackage
from assistant.fixtures import install_gold_fixture
from assistant.trend_signals import trend_from_values
from assistant.user_profile import UserProfile, get_profile_store, reset_profile_store
from assistant.company_context import reset_company_context


@pytest.fixture(autouse=True)
def _reset():
    reset_company_context()
    reset_profile_store()
    yield
    reset_company_context()
    reset_profile_store()


@pytest.fixture
def gold_ready(monkeypatch):
    install_gold_fixture()
    monkeypatch.setattr(
        pipeline, "fetch_trend_signal",
        lambda asset: trend_from_values(asset, r7=0.03, r30=0.09),
    )
    monkeypatch.setattr(pipeline, "_llm_callable", lambda: None)


class TestBuildContext:
    def test_returns_context_package(self):
        pkg = build_context("我能不能买黄金", user_id=None, llm_callable=None)
        assert isinstance(pkg, ContextPackage)

    def test_company_context_always_present(self):
        pkg = build_context("hello", user_id=None)
        assert pkg.company is not None
        assert pkg.company.company_name

    def test_profile_default_for_unknown_user(self):
        pkg = build_context("hello", user_id=99999999)
        assert pkg.profile is not None
        assert pkg.profile.role == "unknown"

    def test_profile_loaded_for_known_user(self):
        store = get_profile_store()
        store.set(UserProfile(user_id="42", role="analyst", is_internal=True))
        pkg = build_context("黄金怎么样", user_id=42)
        assert pkg.profile.role == "analyst"
        assert pkg.profile.is_internal is True

    def test_route_detected(self, gold_ready):
        pkg = build_context("我能不能买黄金？", user_id=None, llm_callable=None)
        assert pkg.route.route == "market_decision"
        assert pkg.route.asset == "gold"

    def test_fomo_emotion_detected(self, gold_ready):
        pkg = build_context(
            "大家都在买，我是不是也该上", user_id=None, llm_callable=None
        )
        assert pkg.user_emotion.primary_emotion == "fomo"

    def test_rag_retrieves_news_and_community(self, gold_ready):
        pkg = build_context("黄金能买吗", user_id=None, llm_callable=None)
        assert len(pkg.news) > 0
        assert len(pkg.community) > 0

    def test_emotional_chat_no_rag(self):
        pkg = build_context("我好焦虑，不知道怎么办", user_id=None, llm_callable=None)
        assert pkg.route.route == "emotional_chat"
        # emotional_chat 不做 RAG
        assert len(pkg.news) == 0
        assert len(pkg.community) == 0

    def test_to_debug_dict_structure(self, gold_ready):
        pkg = build_context("黄金能买吗", user_id=None)
        d = pkg.to_debug_dict()
        assert "route" in d
        assert "asset" in d
        assert "user_emotion" in d
        assert "profile" in d
        assert "company_context_injected" in d
        assert d["company_context_injected"] is True
        assert "retrieved_news_count" in d
        assert "retrieved_community_count" in d

    def test_to_prompt_block_contains_company_info(self, gold_ready):
        pkg = build_context("黄金能买吗", user_id=None)
        block = pkg.to_prompt_block()
        assert "公司语境" in block
        assert "用户画像" in block

    def test_build_time_tracked(self, gold_ready):
        pkg = build_context("黄金能买吗", user_id=None)
        assert pkg.build_time_ms >= 0


class TestContextPackagePromptBlock:
    def test_trend_block_included(self, gold_ready):
        pkg = build_context("黄金能买吗", user_id=None)
        if pkg.trend:
            block = pkg.to_prompt_block()
            assert "趋势信号" in block

    def test_evidence_block_when_no_derived(self, gold_ready, monkeypatch):
        # 关闭 derived cache，强制走 evidence block
        pkg = build_context(
            "黄金能买吗", user_id=None,
            use_derived_cache=False,
        )
        block = pkg.to_prompt_block()
        # should mention news or community
        if pkg.news:
            assert "新闻" in block or "gold" in block.lower()
