"""
Knowledge store —— SQLite 的一层薄封装。

两张表：news / community。每条都存 raw_text + 结构化字段，
供检索层做关键词匹配 + recency 打分。

设计细节：
  - 列用 JSON 字符串保存 asset_tags/topic_tags 列表，读时 json.loads 回来。
    对百万级数据不够快，对 demo 和几千条记录完全够。
  - upsert 用 INSERT OR REPLACE + id 唯一键，允许重复写入不会膨胀。
  - 时间统一存 unix 秒（float），检索层再做窗口过滤。
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


def _resolve_default_db_path() -> Path:
    """每次重新解析 env —— 方便测试用 monkeypatch 覆盖。"""
    env = os.getenv("ASSISTANT_KNOWLEDGE_DB", "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "logs" / "knowledge.sqlite3"


# 保留模块级常量给老代码 import，但不要再依赖它做路径判断 —— 它只在
# 模块第一次被导入时固化。
DEFAULT_DB_PATH = _resolve_default_db_path()


# ─── 数据类 ──────────────────────────────────────────────────────────────────

@dataclass
class NewsDoc:
    id: str
    source: str
    title: str
    published_at: float           # unix seconds
    raw_text: str
    summary: str = ""
    asset_tags: list[str] = field(default_factory=list)
    topic_tags: list[str] = field(default_factory=list)
    sentiment: str = "neutral"      # bullish | bearish | neutral | mixed
    importance_score: float = 0.0
    why_it_matters: str = ""
    url: str = ""

    def to_row(self) -> tuple:
        return (
            self.id, self.source, self.title, self.published_at,
            self.raw_text, self.summary,
            json.dumps(self.asset_tags, ensure_ascii=False),
            json.dumps(self.topic_tags, ensure_ascii=False),
            self.sentiment, self.importance_score,
            self.why_it_matters, self.url,
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "NewsDoc":
        return cls(
            id=row["id"], source=row["source"], title=row["title"],
            published_at=row["published_at"], raw_text=row["raw_text"],
            summary=row["summary"],
            asset_tags=json.loads(row["asset_tags"] or "[]"),
            topic_tags=json.loads(row["topic_tags"] or "[]"),
            sentiment=row["sentiment"],
            importance_score=row["importance_score"],
            why_it_matters=row["why_it_matters"],
            url=row["url"],
        )


@dataclass
class CommunityDoc:
    id: str
    platform: str
    channel_or_group: str
    published_at: float
    raw_text: str
    normalized_text: str = ""
    summary: str = ""
    author: str = ""
    asset_tags: list[str] = field(default_factory=list)
    bullish_bearish_label: str = "neutral"      # bullish|bearish|neutral|mixed
    emotion_label: str = "neutral"              # 投资场景标签（见 sentiment_aggregator）
    confidence: float = 0.0
    engagement_score: float = 0.0
    url: str = ""

    def to_row(self) -> tuple:
        return (
            self.id, self.platform, self.channel_or_group, self.author,
            self.published_at, self.raw_text, self.normalized_text,
            self.summary,
            json.dumps(self.asset_tags, ensure_ascii=False),
            self.bullish_bearish_label, self.emotion_label,
            self.confidence, self.engagement_score, self.url,
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "CommunityDoc":
        return cls(
            id=row["id"], platform=row["platform"],
            channel_or_group=row["channel_or_group"],
            author=row["author"],
            published_at=row["published_at"],
            raw_text=row["raw_text"],
            normalized_text=row["normalized_text"],
            summary=row["summary"],
            asset_tags=json.loads(row["asset_tags"] or "[]"),
            bullish_bearish_label=row["bullish_bearish_label"],
            emotion_label=row["emotion_label"],
            confidence=row["confidence"],
            engagement_score=row["engagement_score"],
            url=row["url"],
        )


# ─── 存储引擎 ────────────────────────────────────────────────────────────────

class KnowledgeStore:
    """
    线程/进程级都足够简单的 SQLite 封装。每次调用临时开连接 + commit，
    避免 daily_job、telegram_bot、测试 三处同时持有长连接的锁竞争。
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else _resolve_default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ─── schema ──────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        # WAL mode: readers don't block writers; multiple processes safe for reads
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS news (
                    id TEXT PRIMARY KEY,
                    source TEXT,
                    title TEXT,
                    published_at REAL,
                    raw_text TEXT,
                    summary TEXT,
                    asset_tags TEXT,
                    topic_tags TEXT,
                    sentiment TEXT,
                    importance_score REAL,
                    why_it_matters TEXT,
                    url TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_news_published_at ON news(published_at);

                CREATE TABLE IF NOT EXISTS community (
                    id TEXT PRIMARY KEY,
                    platform TEXT,
                    channel_or_group TEXT,
                    author TEXT,
                    published_at REAL,
                    raw_text TEXT,
                    normalized_text TEXT,
                    summary TEXT,
                    asset_tags TEXT,
                    bullish_bearish_label TEXT,
                    emotion_label TEXT,
                    confidence REAL,
                    engagement_score REAL,
                    url TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_community_published_at
                    ON community(published_at);
                CREATE INDEX IF NOT EXISTS idx_community_platform
                    ON community(platform);
                """
            )

    # ─── writes ──────────────────────────────────────────────────────────────

    def upsert_news(self, docs: list[NewsDoc]) -> int:
        if not docs:
            return 0
        rows = [d.to_row() for d in docs]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO news
                (id, source, title, published_at, raw_text, summary,
                 asset_tags, topic_tags, sentiment, importance_score,
                 why_it_matters, url)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
            conn.commit()
        return len(rows)

    def upsert_community(self, docs: list[CommunityDoc]) -> int:
        if not docs:
            return 0
        rows = [d.to_row() for d in docs]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO community
                (id, platform, channel_or_group, author, published_at,
                 raw_text, normalized_text, summary, asset_tags,
                 bullish_bearish_label, emotion_label,
                 confidence, engagement_score, url)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
            conn.commit()
        return len(rows)

    # ─── reads ───────────────────────────────────────────────────────────────

    def news_in_window(
        self, since_ts: float, limit: int = 200,
    ) -> list[NewsDoc]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM news WHERE published_at >= ? "
                "ORDER BY published_at DESC LIMIT ?",
                (since_ts, limit),
            )
            return [NewsDoc.from_row(r) for r in cur.fetchall()]

    def community_in_window(
        self, since_ts: float, platform: str | None = None, limit: int = 500,
    ) -> list[CommunityDoc]:
        sql = "SELECT * FROM community WHERE published_at >= ?"
        params: list[Any] = [since_ts]
        if platform:
            sql += " AND platform = ?"
            params.append(platform)
        sql += " ORDER BY published_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            cur = conn.execute(sql, params)
            return [CommunityDoc.from_row(r) for r in cur.fetchall()]

    def count(self) -> dict[str, int]:
        with self._connect() as conn:
            n = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
            c = conn.execute("SELECT COUNT(*) FROM community").fetchone()[0]
        return {"news": n, "community": c}

    def clear(self) -> None:
        """测试用：清空所有表。"""
        with self._connect() as conn:
            conn.execute("DELETE FROM news")
            conn.execute("DELETE FROM community")
            conn.commit()


# ─── 默认 store 单例 ────────────────────────────────────────────────────────

_default: KnowledgeStore | None = None


def default_store() -> KnowledgeStore:
    global _default
    if _default is None:
        _default = KnowledgeStore()
    return _default


def reset_default_store() -> None:
    """测试钩子：让 default_store() 重新初始化（尊重环境变量）。"""
    global _default
    _default = None
