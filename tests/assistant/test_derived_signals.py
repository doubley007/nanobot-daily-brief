"""Tests for rag/derived_signals module."""
from __future__ import annotations

import time
import pytest
import assistant.pipeline as pipeline
from assistant.rag.derived_signals import (
    DerivedSignal,
    DerivedSignalStore,
    compute_derived_signal,
    reset_default_derived_store,
)
from assistant.fixtures import install_gold_fixture
from assistant.trend_signals import trend_from_values


@pytest.fixture(autouse=True)
def _isolated_ds_store(tmp_path, monkeypatch):
    db = tmp_path / "knowledge.sqlite3"
    monkeypatch.setenv("ASSISTANT_KNOWLEDGE_DB", str(db))
    from assistant.rag import store as _store_mod
    _store_mod._default = None
    reset_default_derived_store()
    yield
    _store_mod._default = None
    reset_default_derived_store()


@pytest.fixture
def gold_ready(monkeypatch):
    install_gold_fixture()
    monkeypatch.setattr(
        pipeline, "fetch_trend_signal",
        lambda asset: trend_from_values(asset, r7=0.03, r30=0.09),
    )


class TestDerivedSignalStore:
    def test_upsert_and_get(self, tmp_path):
        store = DerivedSignalStore(db_path=tmp_path / "test.db")
        sig = DerivedSignal(
            asset="gold", window="3d",
            computed_at=time.time(),
            news_direction="bullish", news_strength=0.6,
            community_bias="bullish",
            bullish_ratio=0.7, bearish_ratio=0.1,
            fomo_ratio=0.3, uncertainty_ratio=0.1,
            narrative_keywords=["rate cut"],
            crowding_risk="medium",
            trend_momentum="up",
            entry_quality="medium",
            summary="黄金偏多",
            post_count=50, news_count=8,
        )
        store.upsert(sig)
        retrieved = store.get("gold", window="3d", max_age_seconds=3600)
        assert retrieved is not None
        assert retrieved.asset == "gold"
        assert retrieved.news_direction == "bullish"
        assert retrieved.bullish_ratio == pytest.approx(0.7)

    def test_expired_signal_returns_none(self, tmp_path):
        store = DerivedSignalStore(db_path=tmp_path / "test.db")
        sig = DerivedSignal(
            asset="gold", window="3d",
            computed_at=time.time() - 7200,  # 2 hours ago
            news_direction="neutral", news_strength=0.0,
            community_bias="neutral",
            bullish_ratio=0.5, bearish_ratio=0.3,
            fomo_ratio=0.1, uncertainty_ratio=0.1,
            summary="",
        )
        store.upsert(sig)
        # max_age_seconds=3600: 2h old → expired
        result = store.get("gold", window="3d", max_age_seconds=3600)
        assert result is None

    def test_different_windows_separate(self, tmp_path):
        store = DerivedSignalStore(db_path=tmp_path / "test.db")
        for w in ("1d", "3d", "7d"):
            sig = DerivedSignal(
                asset="gold", window=w,
                computed_at=time.time(),
                news_direction="bullish", news_strength=0.5,
                community_bias="bullish",
                bullish_ratio=0.6, bearish_ratio=0.2,
                fomo_ratio=0.2, uncertainty_ratio=0.1,
                summary=f"window={w}",
            )
            store.upsert(sig)
        r = store.get("gold", window="1d")
        assert r is not None
        assert r.summary == "window=1d"

    def test_list_recent(self, tmp_path):
        store = DerivedSignalStore(db_path=tmp_path / "test.db")
        for asset in ("gold", "bitcoin"):
            sig = DerivedSignal(
                asset=asset, window="3d",
                computed_at=time.time(),
                news_direction="neutral", news_strength=0.0,
                community_bias="neutral",
                bullish_ratio=0.5, bearish_ratio=0.3,
                fomo_ratio=0.1, uncertainty_ratio=0.1,
                summary="",
            )
            store.upsert(sig)
        results = store.list_recent()
        assert len(results) == 2


class TestComputeDerivedSignal:
    def test_compute_gold_signal(self, gold_ready):
        sig = compute_derived_signal("gold", window_hours=72)
        assert isinstance(sig, DerivedSignal)
        assert sig.asset == "gold"
        assert sig.news_direction in ("bullish", "bearish", "neutral")
        assert sig.community_bias in ("bullish", "bearish", "neutral", "mixed")
        assert 0 <= sig.bullish_ratio <= 1
        assert 0 <= sig.bearish_ratio <= 1
        assert sig.entry_quality in ("good", "medium", "poor")
        assert sig.crowding_risk in ("low", "medium", "high")
        assert sig.post_count > 0
        assert sig.news_count > 0

    def test_signal_summary_not_empty(self, gold_ready):
        sig = compute_derived_signal("gold", window_hours=72)
        assert len(sig.summary) > 0

    def test_to_context_block_format(self, gold_ready):
        sig = compute_derived_signal("gold", window_hours=72)
        block = sig.to_context_block()
        assert "市场派生信号" in block
        assert "gold" in block
        assert "新闻面" in block
        assert "社区偏向" in block

    def test_to_dict_serializable(self, gold_ready):
        import json
        sig = compute_derived_signal("gold", window_hours=72)
        d = sig.to_dict()
        # should be JSON serializable
        json.dumps(d)
        assert d["asset"] == "gold"
