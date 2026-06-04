"""
Tests for persistent vector index (Task 1 / v5).

Covers:
  A. VectorIndexStore: save, load, stale detection, force_rebuild
  B. Retriever loads/rebuilds index and propagates index_status
  C. Fallback to keyword mode when numpy unavailable
  D. index_status visible in trace.meta
  E. rebuild_indexes() utility
"""
from __future__ import annotations

import time
import pytest
from pathlib import Path

from assistant.rag.vector_store import VectorIndexStore, reset_vector_store
from assistant.rag.retriever import Retriever, RetrievedEvidence
from assistant.rag.store import KnowledgeStore, NewsDoc, CommunityDoc
from assistant.fixtures import install_gold_fixture
import assistant.pipeline as pipeline
from assistant.pipeline import answer_question_traced
from assistant.trend_signals import trend_from_values


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSISTANT_KNOWLEDGE_DB", str(tmp_path / "knowledge.sqlite3"))
    monkeypatch.setenv("VECTOR_INDEX_DIR", str(tmp_path / "vidx"))
    from assistant.rag import store as _store_mod
    _store_mod._default = None
    reset_vector_store()
    yield
    _store_mod._default = None
    reset_vector_store()


def _news(i: int, text: str = "gold rate cut bullish safe haven") -> NewsDoc:
    return NewsDoc(
        id=f"n{i}", source="t", title=text, published_at=time.time(),
        raw_text=text, sentiment="bullish", importance_score=0.5,
    )


# ── A: VectorIndexStore ────────────────────────────────────────────────────────

class TestVectorIndexStore:
    def test_build_and_save(self, tmp_path):
        vs = VectorIndexStore(index_dir=tmp_path / "vidx")
        texts = ["gold rate cut bullish", "bitcoin etf halving", "dollar stronger"]
        ids = ["a", "b", "c"]
        li = vs.load_or_rebuild("news", texts, ids, force_rebuild=True)
        assert li.status == "rebuilt"
        assert li.index.is_available
        assert li.meta.n_docs == 3
        # Index file should exist
        assert (tmp_path / "vidx" / "news.pkl").exists()

    def test_load_from_cache(self, tmp_path):
        vs = VectorIndexStore(index_dir=tmp_path / "vidx")
        texts = ["gold bullish", "bitcoin bearish"]
        ids = ["a", "b"]
        vs.load_or_rebuild("news", texts, ids, force_rebuild=True)

        # Load again — should be cached
        vs2 = VectorIndexStore(index_dir=tmp_path / "vidx")
        li = vs2.load_or_rebuild("news", texts, ids)
        assert li.status == "loaded"

    def test_stale_on_doc_count_change(self, tmp_path):
        vs = VectorIndexStore(index_dir=tmp_path / "vidx")
        texts = ["gold bullish"] * 5
        ids = [f"d{i}" for i in range(5)]
        vs.load_or_rebuild("news", texts, ids, force_rebuild=True)

        # Now pass significantly more docs
        big_texts = ["gold bullish"] * 15
        big_ids = [f"d{i}" for i in range(15)]
        vs2 = VectorIndexStore(index_dir=tmp_path / "vidx")
        li = vs2.load_or_rebuild("news", big_texts, big_ids)
        # 15 vs 5 = 200% change >> 10% threshold → stale or rebuilt
        assert li.status in ("stale", "rebuilt")

    def test_force_rebuild_ignores_cache(self, tmp_path):
        vs = VectorIndexStore(index_dir=tmp_path / "vidx")
        texts = ["gold bullish"]
        ids = ["a"]
        vs.load_or_rebuild("news", texts, ids, force_rebuild=True)
        # Force rebuild again
        li = vs.load_or_rebuild("news", texts, ids, force_rebuild=True)
        assert li.status == "rebuilt"

    def test_empty_corpus(self, tmp_path):
        vs = VectorIndexStore(index_dir=tmp_path / "vidx")
        li = vs.load_or_rebuild("news", [], [])
        assert li.status == "empty"
        assert li.index.query("gold") == []

    def test_invalidate_removes_files(self, tmp_path):
        vs = VectorIndexStore(index_dir=tmp_path / "vidx")
        vs.load_or_rebuild("news", ["gold"], ["a"], force_rebuild=True)
        assert (tmp_path / "vidx" / "news.pkl").exists()
        vs.invalidate("news")
        assert not (tmp_path / "vidx" / "news.pkl").exists()

    def test_fallback_when_numpy_missing(self, tmp_path, monkeypatch):
        import assistant.rag.vector_index as vi
        monkeypatch.setattr(vi, "_NUMPY_AVAILABLE", False)
        vs = VectorIndexStore(index_dir=tmp_path / "vidx")
        li = vs.load_or_rebuild("news", ["gold rate cut"], ["a"],
                                 force_rebuild=True)
        assert li.status == "fallback"
        assert not li.index.is_available
        assert li.index.query("gold") == []


# ── B: Retriever uses persistent index ────────────────────────────────────────

class TestRetrieverPersistentIndex:
    def test_index_status_in_evidence_rebuilt(self, tmp_path):
        store = KnowledgeStore(db_path=tmp_path / "k.db")
        vs = VectorIndexStore(index_dir=tmp_path / "vidx")
        for i in range(5):
            store.upsert_news([_news(i)])
        r = Retriever(store=store, vector_store=vs)
        ev = r.retrieve("gold", window_hours=72, query_text="gold rate cut bullish")
        assert ev.index_status in ("rebuilt", "loaded", "empty")
        assert isinstance(ev.vector_enabled, bool)

    def test_index_reused_on_second_call(self, tmp_path):
        store = KnowledgeStore(db_path=tmp_path / "k.db")
        vs = VectorIndexStore(index_dir=tmp_path / "vidx")
        for i in range(3):
            store.upsert_news([_news(i)])
        r = Retriever(store=store, vector_store=vs)
        ev1 = r.retrieve("gold", window_hours=72, query_text="gold")
        first_status = ev1.index_status
        ev2 = r.retrieve("gold", window_hours=72, query_text="gold rate")
        # Second call reuses same Retriever instance, index should be loaded (cached in vs)
        assert ev2.index_status in (first_status, "loaded", "rebuilt")

    def test_index_status_none_without_query(self, tmp_path):
        store = KnowledgeStore(db_path=tmp_path / "k.db")
        store.upsert_news([_news(1)])
        vs = VectorIndexStore(index_dir=tmp_path / "vidx")
        r = Retriever(store=store, vector_store=vs)
        ev = r.retrieve("gold", window_hours=72)  # no query_text
        assert ev.index_status == "none"
        assert ev.vector_enabled is False

    def test_rebuild_indexes_method(self, tmp_path):
        store = KnowledgeStore(db_path=tmp_path / "k.db")
        for i in range(5):
            store.upsert_news([_news(i)])
        vs = VectorIndexStore(index_dir=tmp_path / "vidx")
        r = Retriever(store=store, vector_store=vs)
        result = r.rebuild_indexes(window_hours=72)
        assert "news" in result
        assert "community" in result
        assert result["news"] in ("rebuilt", "empty")


# ── C: Fallback when numpy missing ────────────────────────────────────────────

class TestRetrieverFallbackNoNumpy:
    def test_keyword_fallback_when_numpy_missing(self, tmp_path, monkeypatch):
        import assistant.rag.vector_index as vi
        monkeypatch.setattr(vi, "_NUMPY_AVAILABLE", False)
        store = KnowledgeStore(db_path=tmp_path / "k.db")
        store.upsert_news([_news(1, "gold rate cut bullish safe haven")])
        vs = VectorIndexStore(index_dir=tmp_path / "vidx")
        r = Retriever(store=store, vector_store=vs)
        ev = r.retrieve("gold", window_hours=72, query_text="gold")
        # Should still return docs via keyword matching
        assert len(ev.news) > 0
        # No vector scoring
        assert not any(s.retrieval_reason == "vector" for s in ev.scored_news)


# ── D: index_status in trace.meta ────────────────────────────────────────────

class TestIndexStatusInTrace:
    @pytest.fixture(autouse=True)
    def gold_ready(self, monkeypatch):
        install_gold_fixture()
        _trend = lambda asset: trend_from_values(asset, r7=0.03, r30=0.09)
        monkeypatch.setattr(pipeline, "fetch_trend_signal", _trend)
        import assistant.context_builder as _cb
        monkeypatch.setattr(_cb, "fetch_trend_signal", _trend)
        monkeypatch.setattr(pipeline, "_llm_callable", lambda: None)

    def test_index_status_in_trace_meta(self):
        trace = answer_question_traced("黄金现在能买吗")
        assert "index_status" in trace.meta
        assert trace.meta["index_status"] in (
            "loaded", "rebuilt", "stale", "empty", "fallback", "none")

    def test_index_status_present_for_market_summary(self):
        trace = answer_question_traced("最近大家都在讨论什么")
        assert "index_status" in trace.meta
