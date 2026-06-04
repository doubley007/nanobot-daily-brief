"""
Derived Signal Layer —— 聚合层 RAG。

把 news + community 的原始检索结果升华成一个结构化 DerivedSignal，
存进 SQLite derived_signals 表，供快速查询。

作用：
  1. 避免每次问答都重跑 aggregate + assess_news（计算贵）
  2. 让 RAG 有双层：原始层（news/community）+ 聚合层（DerivedSignal）
  3. DerivedSignal 可以直接序列化进 prompt 的 context block

DerivedSignal 字段与用户需求一一对应：
  asset / window / news_direction / news_strength / community_bias /
  bullish_ratio / bearish_ratio / fomo_ratio / uncertainty_ratio /
  narrative_keywords / crowding_risk / trend_momentum / entry_quality / summary
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─── 数据类 ──────────────────────────────────────────────────────────────────

@dataclass
class DerivedSignal:
    asset: str                  # e.g. "gold"
    window: str                 # e.g. "3d"
    computed_at: float          # unix timestamp
    news_direction: str         # "bullish" | "bearish" | "neutral"
    news_strength: float        # 0-1, how strong the news signal is
    community_bias: str         # "bullish" | "bearish" | "neutral" | "mixed"
    bullish_ratio: float
    bearish_ratio: float
    fomo_ratio: float
    uncertainty_ratio: float
    narrative_keywords: list[str] = field(default_factory=list)
    crowding_risk: str = "low"          # "low" | "medium" | "high"
    trend_momentum: str = "unknown"     # "up" | "down" | "flat" | "unknown"
    entry_quality: str = "medium"       # "good" | "medium" | "poor"
    summary: str = ""                   # 一句话人读摘要
    post_count: int = 0
    news_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_context_block(self) -> str:
        """生成注入 prompt 的派生信号段。"""
        lines = [
            f"[市场派生信号 - {self.asset} {self.window}]",
            f"新闻面：{self.news_direction}（强度 {self.news_strength:.2f}）",
            f"社区偏向：{self.community_bias}（多 {self.bullish_ratio:.0%} / 空 {self.bearish_ratio:.0%}）",
            f"FOMO 比例：{self.fomo_ratio:.0%}，拥挤风险：{self.crowding_risk}",
            f"趋势动量：{self.trend_momentum}，入场质量：{self.entry_quality}",
        ]
        if self.narrative_keywords:
            lines.append(f"主流叙事：{', '.join(self.narrative_keywords[:4])}")
        if self.summary:
            lines.append(f"摘要：{self.summary}")
        return "\n".join(lines)


# ─── SQLite 持久化 ────────────────────────────────────────────────────────────

def _resolve_db_path() -> Path:
    import os
    env = os.getenv("ASSISTANT_KNOWLEDGE_DB", "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[4] / "logs" / "knowledge.sqlite3"


class DerivedSignalStore:
    """
    存 / 取 DerivedSignal 的 SQLite 封装。
    使用同一个 knowledge.sqlite3 文件（加一张表）。
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else _resolve_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS derived_signals (
                    asset TEXT NOT NULL,
                    window TEXT NOT NULL,
                    computed_at REAL NOT NULL,
                    news_direction TEXT,
                    news_strength REAL,
                    community_bias TEXT,
                    bullish_ratio REAL,
                    bearish_ratio REAL,
                    fomo_ratio REAL,
                    uncertainty_ratio REAL,
                    narrative_keywords TEXT,
                    crowding_risk TEXT,
                    trend_momentum TEXT,
                    entry_quality TEXT,
                    summary TEXT,
                    post_count INTEGER,
                    news_count INTEGER,
                    PRIMARY KEY (asset, window)
                );
            """)

    def upsert(self, sig: DerivedSignal) -> None:
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO derived_signals
                (asset, window, computed_at, news_direction, news_strength,
                 community_bias, bullish_ratio, bearish_ratio, fomo_ratio,
                 uncertainty_ratio, narrative_keywords, crowding_risk,
                 trend_momentum, entry_quality, summary, post_count, news_count)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                sig.asset, sig.window, sig.computed_at,
                sig.news_direction, sig.news_strength,
                sig.community_bias, sig.bullish_ratio, sig.bearish_ratio,
                sig.fomo_ratio, sig.uncertainty_ratio,
                json.dumps(sig.narrative_keywords, ensure_ascii=False),
                sig.crowding_risk, sig.trend_momentum, sig.entry_quality,
                sig.summary, sig.post_count, sig.news_count,
            ))
            conn.commit()

    def get(self, asset: str, window: str = "3d",
            max_age_seconds: float = 3600) -> DerivedSignal | None:
        """返回最新的信号；超过 max_age_seconds 则视为过期。"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM derived_signals WHERE asset=? AND window=?",
                (asset, window),
            ).fetchone()
        if row is None:
            return None
        age = time.time() - row["computed_at"]
        if age > max_age_seconds:
            return None
        return DerivedSignal(
            asset=row["asset"], window=row["window"],
            computed_at=row["computed_at"],
            news_direction=row["news_direction"] or "neutral",
            news_strength=row["news_strength"] or 0.0,
            community_bias=row["community_bias"] or "neutral",
            bullish_ratio=row["bullish_ratio"] or 0.0,
            bearish_ratio=row["bearish_ratio"] or 0.0,
            fomo_ratio=row["fomo_ratio"] or 0.0,
            uncertainty_ratio=row["uncertainty_ratio"] or 0.0,
            narrative_keywords=json.loads(row["narrative_keywords"] or "[]"),
            crowding_risk=row["crowding_risk"] or "low",
            trend_momentum=row["trend_momentum"] or "unknown",
            entry_quality=row["entry_quality"] or "medium",
            summary=row["summary"] or "",
            post_count=row["post_count"] or 0,
            news_count=row["news_count"] or 0,
        )

    def list_recent(self, max_age_seconds: float = 7200) -> list[DerivedSignal]:
        """列出所有未过期的信号（用于 market_summary）。"""
        cutoff = time.time() - max_age_seconds
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM derived_signals WHERE computed_at >= ? ORDER BY computed_at DESC",
                (cutoff,),
            ).fetchall()
        results = []
        for row in rows:
            results.append(DerivedSignal(
                asset=row["asset"], window=row["window"],
                computed_at=row["computed_at"],
                news_direction=row["news_direction"] or "neutral",
                news_strength=row["news_strength"] or 0.0,
                community_bias=row["community_bias"] or "neutral",
                bullish_ratio=row["bullish_ratio"] or 0.0,
                bearish_ratio=row["bearish_ratio"] or 0.0,
                fomo_ratio=row["fomo_ratio"] or 0.0,
                uncertainty_ratio=row["uncertainty_ratio"] or 0.0,
                narrative_keywords=json.loads(row["narrative_keywords"] or "[]"),
                crowding_risk=row["crowding_risk"] or "low",
                trend_momentum=row["trend_momentum"] or "unknown",
                entry_quality=row["entry_quality"] or "medium",
                summary=row["summary"] or "",
                post_count=row["post_count"] or 0,
                news_count=row["news_count"] or 0,
            ))
        return results


# ─── 信号计算器 ───────────────────────────────────────────────────────────────

def refresh_derived_signals_batch(
    assets: list[str],
    window_hours: int = 72,
    rag_store: "Any | None" = None,
    ds_store: "DerivedSignalStore | None" = None,
) -> dict[str, str]:
    """
    Compute and persist DerivedSignals for a list of assets.
    Returns {asset: "ok" | "sparse" | "error:<msg>"}.
    "sparse" means computed but both news_count and post_count are 0 (stored anyway).
    """
    import time as _time
    results: dict[str, str] = {}
    for asset in assets:
        t0 = _time.time()
        try:
            sig = compute_derived_signal(asset, window_hours=window_hours,
                                         store=rag_store, ds_store=ds_store)
            elapsed = _time.time() - t0
            if sig.news_count == 0 and sig.post_count == 0:
                results[asset] = "sparse"
                logger.info("derived_signals: %s sparse (no data) in %.1fs", asset, elapsed)
            else:
                results[asset] = "ok"
                logger.info(
                    "derived_signals: %s ok (news=%d posts=%d) in %.1fs",
                    asset, sig.news_count, sig.post_count, elapsed,
                )
        except Exception as e:
            logger.warning("derived_signals: %s error: %s", asset, e)
            results[asset] = f"error:{e}"
    return results


def compute_derived_signal(
    asset: str,
    window_hours: int = 72,
    store: "Any | None" = None,
    ds_store: "DerivedSignalStore | None" = None,
) -> DerivedSignal:
    """
    从原始 news + community 层计算派生信号，持久化到 DerivedSignalStore 并返回。
    这是 daily_job / 定时刷新 的调用点。
    store     — RAG knowledge store (news + community SQLite)
    ds_store  — DerivedSignalStore for output persistence; defaults to default_derived_store()
    """
    from assistant.rag.retriever import Retriever
    from assistant.rag.store import default_store
    from assistant.sentiment_aggregator import aggregate
    from assistant.decision_engine import assess_news
    from assistant.trend_signals import fetch_trend_signal

    rag_store = store or default_store()
    retriever = Retriever(store=rag_store)

    evidence = retriever.retrieve(
        asset=asset, window_hours=window_hours,
        top_k_news=15, top_k_community=200,
    )
    news_assess = assess_news(asset, evidence.news)
    agg = aggregate(evidence.community, asset=asset, window_hours=window_hours)
    trend = fetch_trend_signal(asset)

    # entry_quality from decision engine
    from assistant.decision_engine import _calc_crowding_score, _calc_entry_quality
    from assistant.decision_engine import community_direction_score, trend_direction_score
    com_dir = community_direction_score(agg)
    trend_dir = trend_direction_score(trend)
    news_dir_n = 1 if news_assess.direction == "bullish" else (-1 if news_assess.direction == "bearish" else 0)
    direction_score = news_dir_n * 0.45 + com_dir * 0.35 + trend_dir * 0.2
    crowding_score = _calc_crowding_score(agg, trend)
    entry_quality, _ = _calc_entry_quality(direction_score, crowding_score, trend)

    window_label = f"{window_hours // 24}d" if window_hours >= 24 else f"{window_hours}h"
    news_strength = min(1.0, abs(news_assess.net_score) / 2.0)

    summary = (
        f"{asset}: 新闻{news_assess.direction}，社区{agg.overall_bias}，"
        f"入场质量{entry_quality}，拥挤风险{agg.crowded_trade_risk}"
    )

    sig = DerivedSignal(
        asset=asset,
        window=window_label,
        computed_at=time.time(),
        news_direction=news_assess.direction,
        news_strength=round(news_strength, 3),
        community_bias=agg.overall_bias,
        bullish_ratio=agg.bullish_ratio,
        bearish_ratio=agg.bearish_ratio,
        fomo_ratio=agg.fomo_ratio,
        uncertainty_ratio=agg.uncertainty_ratio,
        narrative_keywords=agg.narrative_keywords,
        crowding_risk=agg.crowded_trade_risk,
        trend_momentum=trend.momentum_label,
        entry_quality=entry_quality,
        summary=summary,
        post_count=agg.post_count,
        news_count=len(evidence.news),
    )
    # persist to store so subsequent get() calls hit cache
    out_store = ds_store or default_derived_store()
    out_store.upsert(sig)
    return sig


# ─── 默认单例 ─────────────────────────────────────────────────────────────────

_default_ds_store: DerivedSignalStore | None = None


def default_derived_store() -> DerivedSignalStore:
    global _default_ds_store
    if _default_ds_store is None:
        _default_ds_store = DerivedSignalStore()
    return _default_ds_store


def reset_default_derived_store() -> None:
    global _default_ds_store
    _default_ds_store = None
