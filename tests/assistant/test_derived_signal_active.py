"""
CL6: Tests verifying derived signal is actively used in Q&A (not just stored).

Tests:
  - When DerivedSignal is cached, it appears in context_pkg and to_prompt_block
  - trace.meta["derived_signal_status"] shows "hit:<summary>" vs "miss"
  - When cache is stale/empty, derived_signal_status is "miss"
  - The derived context block is included in to_prompt_block() when present
"""
from __future__ import annotations

import time
import pytest
import assistant.pipeline as pipeline
from assistant.pipeline import answer_question_traced
from assistant.rag.derived_signals import (
    DerivedSignal,
    DerivedSignalStore,
    reset_default_derived_store,
)
from assistant.fixtures import install_gold_fixture
from assistant.trend_signals import trend_from_values
from assistant.user_profile import reset_profile_store
from assistant.company_context import reset_company_context


@pytest.fixture(autouse=True)
def _reset(tmp_path, monkeypatch):
    db = tmp_path / "knowledge.sqlite3"
    monkeypatch.setenv("ASSISTANT_KNOWLEDGE_DB", str(db))
    from assistant.rag import store as _store_mod
    _store_mod._default = None
    reset_default_derived_store()
    reset_company_context()
    reset_profile_store()
    yield
    _store_mod._default = None
    reset_default_derived_store()
    reset_company_context()
    reset_profile_store()


@pytest.fixture
def gold_ready(monkeypatch):
    install_gold_fixture()
    _trend = lambda asset: trend_from_values(asset, r7=0.03, r30=0.09)
    monkeypatch.setattr(pipeline, "fetch_trend_signal", _trend)
    import assistant.context_builder as _cb
    monkeypatch.setattr(_cb, "fetch_trend_signal", _trend)
    monkeypatch.setattr(pipeline, "_llm_callable", lambda: None)


def _make_gold_signal(tmp_path: object) -> DerivedSignal:
    return DerivedSignal(
        asset="gold", window="3d",
        computed_at=time.time(),
        news_direction="bullish", news_strength=0.6,
        community_bias="bullish",
        bullish_ratio=0.7, bearish_ratio=0.1,
        fomo_ratio=0.3, uncertainty_ratio=0.1,
        narrative_keywords=["rate cut", "safe haven"],
        crowding_risk="medium",
        trend_momentum="up",
        entry_quality="medium",
        summary="gold: 新闻bullish，社区bullish，入场质量medium，拥挤风险medium",
        post_count=50, news_count=8,
    )


class TestDerivedSignalHit:
    def test_hit_when_fresh_signal_in_store(self, gold_ready, tmp_path):
        from assistant.rag.derived_signals import default_derived_store
        ds = default_derived_store()
        sig = _make_gold_signal(tmp_path)
        ds.upsert(sig)
        trace = answer_question_traced("我能不能买黄金")
        assert trace.meta.get("derived_signal_status", "").startswith("hit:")

    def test_hit_includes_summary_in_status(self, gold_ready, tmp_path):
        from assistant.rag.derived_signals import default_derived_store
        ds = default_derived_store()
        sig = _make_gold_signal(tmp_path)
        ds.upsert(sig)
        trace = answer_question_traced("黄金能追吗")
        status = trace.meta.get("derived_signal_status", "")
        assert status.startswith("hit:")
        assert len(status) > 5  # has content after "hit:"

    def test_derived_signal_in_context_pkg(self, gold_ready, tmp_path):
        from assistant.rag.derived_signals import default_derived_store
        ds = default_derived_store()
        ds.upsert(_make_gold_signal(tmp_path))
        trace = answer_question_traced("我能不能买黄金")
        assert trace.context_pkg is not None
        assert trace.context_pkg.derived_signal is not None

    def test_derived_signal_in_prompt_block(self, gold_ready, tmp_path):
        from assistant.rag.derived_signals import default_derived_store
        ds = default_derived_store()
        ds.upsert(_make_gold_signal(tmp_path))
        trace = answer_question_traced("黄金值得买吗")
        assert trace.context_pkg is not None
        block = trace.context_pkg.to_prompt_block()
        assert "市场派生信号" in block

    def test_derived_signal_context_block_content(self, gold_ready, tmp_path):
        from assistant.rag.derived_signals import default_derived_store
        ds = default_derived_store()
        ds.upsert(_make_gold_signal(tmp_path))
        trace = answer_question_traced("黄金值得买吗")
        block = trace.context_pkg.to_prompt_block()
        assert "bullish" in block
        assert "medium" in block  # entry_quality / crowding_risk


class TestDerivedSignalMiss:
    def test_miss_when_no_cache(self, gold_ready):
        trace = answer_question_traced("我能不能买黄金")
        assert trace.meta.get("derived_signal_status") == "miss"

    def test_miss_when_cache_expired(self, gold_ready, tmp_path):
        from assistant.rag.derived_signals import default_derived_store
        ds = default_derived_store()
        sig = DerivedSignal(
            asset="gold", window="3d",
            computed_at=time.time() - 7200,  # 2h old
            news_direction="bullish", news_strength=0.5,
            community_bias="bullish",
            bullish_ratio=0.6, bearish_ratio=0.2,
            fomo_ratio=0.2, uncertainty_ratio=0.1,
            summary="stale gold signal",
        )
        ds.upsert(sig)
        trace = answer_question_traced("黄金能买吗")
        assert trace.meta.get("derived_signal_status") == "miss"

    def test_miss_for_emotional_chat(self, gold_ready):
        trace = answer_question_traced("我好焦虑，睡不着觉")
        # Emotional chat doesn't do market retrieval
        assert "derived_signal_status" in trace.meta


class TestDerivedSignalStatus:
    def test_status_always_in_meta(self, gold_ready):
        for q in ["我能不能买黄金", "最近黄金怎么样", "我好焦虑"]:
            trace = answer_question_traced(q)
            assert "derived_signal_status" in trace.meta
