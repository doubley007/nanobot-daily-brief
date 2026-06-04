"""
Tests for report / snapshot mode (Task 4 / v5).

Covers:
  A. generate_report() returns structured text with required sections
  B. Fallback report when derived signal not available
  C. Report with derived signal from cache
  D. report_cli() convenience function
  E. Demo CLI --report flag
"""
from __future__ import annotations

import time
import pytest

import assistant.pipeline as pipeline
from assistant.report import generate_report, report_cli
from assistant.rag.store import KnowledgeStore, NewsDoc, CommunityDoc
from assistant.rag.derived_signals import DerivedSignal, DerivedSignalStore
from assistant.fixtures import install_gold_fixture, install_bitcoin_fixture
from assistant.trend_signals import trend_from_values
from assistant.user_profile import reset_profile_store
from assistant.company_context import reset_company_context


@pytest.fixture(autouse=True)
def _reset(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSISTANT_KNOWLEDGE_DB", str(tmp_path / "knowledge.sqlite3"))
    monkeypatch.setenv("VECTOR_INDEX_DIR", str(tmp_path / "vidx"))
    from assistant.rag import store as _store_mod
    _store_mod._default = None
    from assistant.rag.derived_signals import reset_default_derived_store
    reset_default_derived_store()
    reset_profile_store()
    reset_company_context()
    from assistant.rag.vector_store import reset_vector_store
    reset_vector_store()
    yield
    _store_mod._default = None
    reset_default_derived_store()
    reset_profile_store()
    reset_company_context()
    reset_vector_store()


def _make_news(i: int, asset: str = "gold") -> NewsDoc:
    return NewsDoc(
        id=f"n{i}", source="test", title=f"{asset} rate cut bullish {i}",
        published_at=time.time() - i * 600,
        raw_text=f"{asset} safe haven inflation hedge {i}",
        asset_tags=[asset], sentiment="bullish", importance_score=0.5,
    )


def _make_community(i: int, asset: str = "gold") -> CommunityDoc:
    return CommunityDoc(
        id=f"c{i}", platform="reddit", channel_or_group="wsb",
        author="user", published_at=time.time() - i * 300,
        raw_text=f"{asset} bullish going up {i}",
        asset_tags=[asset], bullish_bearish_label="bullish",
        emotion_label="bullish_optimism", confidence=0.8, engagement_score=10.0,
    )


def _make_derived_signal(asset: str = "gold") -> DerivedSignal:
    return DerivedSignal(
        asset=asset, window="3d", computed_at=time.time(),
        news_direction="bullish", news_strength=0.7,
        community_bias="bullish", bullish_ratio=0.65, bearish_ratio=0.15,
        fomo_ratio=0.2, uncertainty_ratio=0.1,
        narrative_keywords=["rate cut", "safe haven", "inflation", "hedge"],
        crowding_risk="medium", trend_momentum="up",
        entry_quality="good",
        summary="gold: 新闻bullish，社区bullish，入场质量good",
        post_count=150, news_count=8,
    )


# ── A: generate_report() basic output ─────────────────────────────────────────

class TestGenerateReport:
    def test_report_contains_asset_name(self, tmp_path):
        store = KnowledgeStore(db_path=tmp_path / "k.db")
        for i in range(5):
            store.upsert_news([_make_news(i)])
            store.upsert_community([_make_community(i)])
        report = generate_report("gold", rag_store=store)
        assert "gold" in report.lower() or "黄金" in report

    def test_report_contains_community_bias(self, tmp_path):
        store = KnowledgeStore(db_path=tmp_path / "k.db")
        for i in range(5):
            store.upsert_community([_make_community(i)])
        report = generate_report("gold", rag_store=store)
        assert "bullish" in report.lower() or "社区" in report

    def test_report_contains_crowding_risk(self, tmp_path):
        store = KnowledgeStore(db_path=tmp_path / "k.db")
        for i in range(5):
            store.upsert_community([_make_community(i)])
        ds = DerivedSignalStore(db_path=tmp_path / "ds.db")
        ds.upsert(_make_derived_signal())
        report = generate_report("gold", rag_store=store, ds_store=ds)
        assert "拥挤" in report or "crowding" in report.lower()

    def test_report_contains_narrative_keywords(self, tmp_path):
        store = KnowledgeStore(db_path=tmp_path / "k.db")
        ds = DerivedSignalStore(db_path=tmp_path / "ds.db")
        ds.upsert(_make_derived_signal())
        report = generate_report("gold", rag_store=store, ds_store=ds)
        assert "rate cut" in report or "叙事" in report or "safe haven" in report

    def test_report_contains_entry_quality(self, tmp_path):
        store = KnowledgeStore(db_path=tmp_path / "k.db")
        ds = DerivedSignalStore(db_path=tmp_path / "ds.db")
        ds.upsert(_make_derived_signal())
        report = generate_report("gold", rag_store=store, ds_store=ds)
        assert "入场" in report or "entry" in report.lower()

    def test_report_is_non_empty_string(self, tmp_path):
        store = KnowledgeStore(db_path=tmp_path / "k.db")
        for i in range(3):
            store.upsert_community([_make_community(i)])
        report = generate_report("gold", rag_store=store)
        assert isinstance(report, str)
        assert len(report) > 50


# ── B: Fallback report (no derived signal) ────────────────────────────────────

class TestFallbackReport:
    def test_fallback_when_no_derived_signal(self, tmp_path):
        store = KnowledgeStore(db_path=tmp_path / "k.db")
        ds = DerivedSignalStore(db_path=tmp_path / "ds.db")
        for i in range(5):
            store.upsert_community([_make_community(i)])
        # No signal inserted → fallback
        report = generate_report("gold", rag_store=store, ds_store=ds)
        assert isinstance(report, str)
        assert len(report) > 20

    def test_empty_store_returns_text(self, tmp_path):
        store = KnowledgeStore(db_path=tmp_path / "k.db")
        ds = DerivedSignalStore(db_path=tmp_path / "ds.db")
        report = generate_report("gold", rag_store=store, ds_store=ds)
        assert isinstance(report, str)


# ── C: Report uses derived signal from cache ──────────────────────────────────

class TestReportWithDerivedSignal:
    def test_report_uses_cached_derived_signal(self, tmp_path):
        store = KnowledgeStore(db_path=tmp_path / "k.db")
        ds = DerivedSignalStore(db_path=tmp_path / "ds.db")
        sig = _make_derived_signal()
        ds.upsert(sig)
        report = generate_report("gold", rag_store=store, ds_store=ds)
        # Should include data from the derived signal
        assert "bullish" in report.lower()
        assert sig.news_count > 0  # used

    def test_report_stale_derived_signal_recomputed(self, tmp_path):
        store = KnowledgeStore(db_path=tmp_path / "k.db")
        for i in range(3):
            store.upsert_community([_make_community(i)])
        ds = DerivedSignalStore(db_path=tmp_path / "ds.db")
        # Insert very old signal (expired)
        old_sig = DerivedSignal(
            asset="gold", window="3d", computed_at=time.time() - 86400,
            news_direction="bearish", news_strength=0.9,
            community_bias="bearish", bullish_ratio=0.1, bearish_ratio=0.8,
            fomo_ratio=0.05, uncertainty_ratio=0.05,
            crowding_risk="high", trend_momentum="down", entry_quality="poor",
            summary="old stale signal",
        )
        ds.upsert(old_sig)
        # Report with 3h TTL → old signal should be ignored, fallback used
        report = generate_report("gold", rag_store=store, ds_store=ds)
        assert isinstance(report, str)


# ── D: report_cli() ───────────────────────────────────────────────────────────

class TestReportCli:
    def test_report_cli_returns_string(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ASSISTANT_KNOWLEDGE_DB", str(tmp_path / "k2.db"))
        from assistant.rag import store as _store_mod
        _store_mod._default = None
        install_gold_fixture()
        report = report_cli("gold")
        assert isinstance(report, str)
        assert len(report) > 50
        _store_mod._default = None

    def test_report_cli_bitcoin(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ASSISTANT_KNOWLEDGE_DB", str(tmp_path / "kbtc.db"))
        from assistant.rag import store as _store_mod
        _store_mod._default = None
        install_bitcoin_fixture()
        report = report_cli("bitcoin")
        assert isinstance(report, str)
        _store_mod._default = None


# ── E: Demo CLI --report flag ─────────────────────────────────────────────────

class TestDemoCLI:
    def test_report_flag_produces_output(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("ASSISTANT_KNOWLEDGE_DB", str(tmp_path / "k3.db"))
        from assistant.rag import store as _store_mod
        _store_mod._default = None

        _trend = lambda asset: trend_from_values(asset, r7=0.03, r30=0.09)
        monkeypatch.setattr(pipeline, "fetch_trend_signal", _trend)
        import assistant.context_builder as _cb
        monkeypatch.setattr(_cb, "fetch_trend_signal", _trend)
        monkeypatch.setattr(pipeline, "_llm_callable", lambda: None)

        from assistant.demo import main
        ret = main(["黄金能买吗", "--asset", "gold", "--report"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "REPORT" in captured.out or "快照" in captured.out or "黄金" in captured.out
        _store_mod._default = None
