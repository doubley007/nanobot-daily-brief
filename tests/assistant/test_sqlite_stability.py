"""
Task 4 / v4 — SQLite stability and Telegram feed mode tests.

Tests:
  A. SQLite WAL mode is enabled on both KnowledgeStore and DerivedSignalStore
  B. Concurrent-style repeated writes don't corrupt data
  C. busy_timeout is set (no immediate SQLITE_BUSY on mild contention)
  D. Telegram JSONL feed mode: adapter parses TG_FEED_FILE correctly
  E. Missing/empty feed file handled gracefully
"""
from __future__ import annotations

import json
import time
import sqlite3
import threading
import pytest
from pathlib import Path

from assistant.rag.store import KnowledgeStore
from assistant.rag.derived_signals import DerivedSignalStore, DerivedSignal


# ── A/B/C: KnowledgeStore WAL mode ───────────────────────────────────────────

class TestKnowledgeStoreWAL:
    def test_wal_mode_enabled(self, tmp_path):
        store = KnowledgeStore(db_path=tmp_path / "k.db")
        with sqlite3.connect(str(tmp_path / "k.db")) as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_busy_timeout_set(self, tmp_path):
        store = KnowledgeStore(db_path=tmp_path / "k.db")
        with sqlite3.connect(str(tmp_path / "k.db")) as conn:
            timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert timeout >= 1000  # at least 1s

    def test_repeated_upsert_no_corruption(self, tmp_path):
        from assistant.rag.store import NewsDoc
        store = KnowledgeStore(db_path=tmp_path / "k.db")
        doc = NewsDoc(
            id="test_1", source="t", title="Gold rate cut",
            published_at=time.time(), raw_text="rate cut gold",
            sentiment="bullish", importance_score=0.5,
        )
        # Upsert same doc 50 times
        for _ in range(50):
            store.upsert_news([doc])
        counts = store.count()
        assert counts["news"] == 1  # upsert, not insert — no duplication

    def test_concurrent_writes_no_crash(self, tmp_path):
        """Multiple threads writing simultaneously should not raise exceptions."""
        from assistant.rag.store import NewsDoc
        store = KnowledgeStore(db_path=tmp_path / "k.db")
        errors: list[Exception] = []

        def worker(i: int) -> None:
            doc = NewsDoc(
                id=f"doc_{i}", source="t",
                title=f"gold news {i}", published_at=time.time(),
                raw_text=f"gold rate cut {i}", sentiment="bullish",
                importance_score=0.5,
            )
            try:
                store.upsert_news([doc])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent write errors: {errors}"
        counts = store.count()
        assert counts["news"] == 10


class TestDerivedSignalStoreWAL:
    def test_wal_mode_enabled(self, tmp_path):
        store = DerivedSignalStore(db_path=tmp_path / "ds.db")
        with sqlite3.connect(str(tmp_path / "ds.db")) as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_repeated_upsert_idempotent(self, tmp_path):
        store = DerivedSignalStore(db_path=tmp_path / "ds.db")
        sig = DerivedSignal(
            asset="gold", window="3d", computed_at=time.time(),
            news_direction="bullish", news_strength=0.6,
            community_bias="bullish", bullish_ratio=0.7, bearish_ratio=0.1,
            fomo_ratio=0.3, uncertainty_ratio=0.1, summary="test",
        )
        for _ in range(20):
            store.upsert(sig)
        with sqlite3.connect(str(tmp_path / "ds.db")) as conn:
            count = conn.execute("SELECT COUNT(*) FROM derived_signals").fetchone()[0]
        assert count == 1


# ── D/E: Telegram feed file (Mode A) ─────────────────────────────────────────

class TestTelegramFeedMode:
    def _write_feed(self, path: Path, posts: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for p in posts:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    def test_feed_file_parsed_correctly(self, tmp_path, monkeypatch):
        feed = tmp_path / "tg_feed.jsonl"
        self._write_feed(feed, [
            {"id": "1", "channel": "@goldchannel", "text": "黄金涨了，看多",
             "created_utc": time.time() - 3600, "engagement": 42},
            {"id": "2", "channel": "@cryptotalk", "text": "BTC ETF confirmed bullish",
             "created_utc": time.time() - 7200, "engagement": 100},
        ])
        monkeypatch.setenv("TG_FEED_FILE", str(feed))
        from sources.telegram_adapter import TelegramSourceAdapter
        adapter = TelegramSourceAdapter()
        report = adapter.fetch()
        assert report.ok is True
        assert len(report.posts) == 2
        assert report.posts[0].channel == "@goldchannel"
        assert "黄金" in report.posts[0].body

    def test_empty_feed_file(self, tmp_path, monkeypatch):
        feed = tmp_path / "empty.jsonl"
        feed.write_text("")
        monkeypatch.setenv("TG_FEED_FILE", str(feed))
        from sources.telegram_adapter import TelegramSourceAdapter
        adapter = TelegramSourceAdapter()
        report = adapter.fetch()
        assert report.ok is True
        assert len(report.posts) == 0

    def test_missing_feed_file_graceful(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TG_FEED_FILE", str(tmp_path / "nonexistent.jsonl"))
        from sources.telegram_adapter import TelegramSourceAdapter
        adapter = TelegramSourceAdapter()
        report = adapter.fetch()
        # Should not raise; posts may be empty
        assert isinstance(report.posts, list)

    def test_malformed_lines_skipped(self, tmp_path, monkeypatch):
        feed = tmp_path / "bad.jsonl"
        feed.write_text(
            '{"id":"1","channel":"c","text":"ok","created_utc":' + str(time.time()) + '}\n'
            'THIS IS NOT JSON\n'
            '{"id":"3","channel":"c","text":"also ok","created_utc":' + str(time.time()) + '}\n'
        )
        monkeypatch.setenv("TG_FEED_FILE", str(feed))
        from sources.telegram_adapter import TelegramSourceAdapter
        adapter = TelegramSourceAdapter()
        report = adapter.fetch()
        assert report.ok is True
        assert len(report.posts) == 2  # malformed line skipped

    def test_feed_posts_can_be_indexed_to_rag(self, tmp_path, monkeypatch):
        """Feed posts can flow through community_indexer into RAG."""
        feed = tmp_path / "feed.jsonl"
        self._write_feed(feed, [
            {"id": "t1", "channel": "@gold", "text": "gold rate cut bullish safe haven",
             "created_utc": time.time() - 1800},
        ])
        monkeypatch.setenv("TG_FEED_FILE", str(feed))
        monkeypatch.setenv("ASSISTANT_KNOWLEDGE_DB", str(tmp_path / "k.db"))
        from assistant.rag import store as _store_mod
        _store_mod._default = None

        from sources.telegram_adapter import TelegramSourceAdapter
        adapter = TelegramSourceAdapter()
        report = adapter.fetch()
        assert len(report.posts) == 1

        # Index posts into RAG
        from assistant.rag.community_indexer import index_community
        from assistant.rag.community_indexer import unified_posts_to_docs
        # Convert UnifiedPost → index
        n = index_community(report.posts)
        assert n >= 1

        # Verify retrievable
        from assistant.rag.store import default_store
        counts = default_store().count()
        assert counts["community"] >= 1
