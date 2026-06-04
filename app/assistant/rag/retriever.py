"""
RAG 检索层 —— 混合检索：keyword + TF-IDF vector + recency rerank。

打分公式（v5 hybrid）：

    score = keyword_hits * 1.0
          + recency_factor * 0.5
          + importance_score * 0.8           # 仅 news
          + engagement_factor * 0.3          # 仅 community
          + vector_score * 0.6               # TF-IDF cosine（embed 可用时）

v5 变化：vector index 改为预构建并持久化（VectorIndexStore），
而非每次查询重建。Retriever 初始化时加载索引，查询时直接使用。
Trace 标记：index_loaded / index_rebuilt / index_stale / index_fallback。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

from assistant.asset_taxonomy import asset_keywords
from assistant.rag.store import (
    CommunityDoc,
    KnowledgeStore,
    NewsDoc,
    default_store,
)
from assistant.rag.vector_store import VectorIndexStore, default_vector_store

logger = logging.getLogger(__name__)


@dataclass
class ScoredNews:
    doc: NewsDoc
    score: float
    retrieval_reason: str  # "asset_tag" | "keyword:<kw>" | "vector" | "recency"


@dataclass
class ScoredCommunity:
    doc: CommunityDoc
    score: float
    retrieval_reason: str


@dataclass
class RetrievedEvidence:
    news: list[NewsDoc]
    community: list[CommunityDoc]
    window_hours: int
    asset: str | None
    # rich scored versions — same docs, just with scores attached
    scored_news: list[ScoredNews] = field(default_factory=list)
    scored_community: list[ScoredCommunity] = field(default_factory=list)
    vector_enabled: bool = False   # True if TF-IDF scoring was applied
    # v5: index status for trace ("loaded"|"rebuilt"|"stale"|"empty"|"fallback"|"none")
    index_status: str = "none"

    def is_empty(self) -> bool:
        return not self.news and not self.community


Kind = Literal["news", "community", "both"]


# ─── embed() — TF-IDF backed ─────────────────────────────────────────────────

def embed(text: str) -> list[float] | None:
    """
    Placeholder kept for API compatibility.
    Actual embedding happens per-query inside Retriever via TFIDFIndex.
    Returns None (the TFIDFIndex builds its matrix from candidate docs on demand).
    """
    return None


# ─── 打分工具 ─────────────────────────────────────────────────────────────────

def _recency_factor(ts: float, now_ts: float, window_hours: int) -> float:
    if ts <= 0:
        return 0.0
    age_hours = max(0.0, (now_ts - ts) / 3600.0)
    if age_hours >= window_hours:
        return 0.0
    return 1.0 - (age_hours / window_hours)


def _keyword_hits(text: str, keywords: list[str]) -> int:
    lower = (text or "").lower()
    return sum(1 for k in keywords if k.lower() in lower)


def _score_news(
    doc: NewsDoc, keywords: list[str], now_ts: float, window_hours: int,
    vector_score: float = 0.0,
) -> float:
    hits = _keyword_hits(f"{doc.title} {doc.raw_text}", keywords)
    if hits == 0 and keywords and vector_score == 0.0:
        return 0.0
    recency = _recency_factor(doc.published_at, now_ts, window_hours)
    return hits * 1.0 + recency * 0.5 + doc.importance_score * 0.8 + vector_score * 0.6


def _score_community(
    doc: CommunityDoc, keywords: list[str], now_ts: float, window_hours: int,
    vector_score: float = 0.0,
) -> float:
    hits = _keyword_hits(doc.raw_text, keywords)
    if hits == 0 and keywords and vector_score == 0.0:
        return 0.0
    recency = _recency_factor(doc.published_at, now_ts, window_hours)
    import math
    eng = math.log1p(max(0.0, doc.engagement_score)) / 8.0
    return hits * 1.0 + recency * 0.5 + eng * 0.3 + vector_score * 0.6


# ─── 对外类 ──────────────────────────────────────────────────────────────────

class Retriever:
    """
    Hybrid retriever: keyword + persistent TF-IDF vector scoring (v5).

    v5 change: vector index is pre-built and persisted via VectorIndexStore.
    On first retrieve(), the store is loaded (or rebuilt if stale/missing).
    Subsequent calls reuse the loaded index — no per-query rebuild.

    Index status is propagated to RetrievedEvidence.index_status:
      "loaded"   — used cached index
      "rebuilt"  — cache was missing/stale, rebuilt now
      "stale"    — used stale cache (doc count drifted, will rebuild next time)
      "empty"    — no docs in corpus
      "fallback" — numpy unavailable, keyword-only mode
      "none"     — no query_text provided, no vector scoring
    """

    def __init__(
        self,
        store: KnowledgeStore | None = None,
        vector_store: VectorIndexStore | None = None,
    ):
        self.store = store or default_store()
        self._vs = vector_store or default_vector_store()
        # Lazily loaded per-kind: {"news": LoadedIndex, "community": LoadedIndex}
        self._loaded: dict[str, object] = {}

    def _get_index(self, kind: str, texts: list[str], doc_ids: list[str]):
        """Load or rebuild the persistent index for `kind`."""
        from assistant.rag.vector_store import LoadedIndex
        li = self._vs.load_or_rebuild(kind, texts, doc_ids)
        self._loaded[kind] = li
        return li

    def _rebuild_index(self, kind: str, texts: list[str], doc_ids: list[str]):
        """Force-rebuild the persistent index for `kind`."""
        li = self._vs.force_rebuild(kind, texts, doc_ids)
        self._loaded[kind] = li
        return li

    # ── news ──────────────────────────────────────────────────────────────────

    def retrieve_news(
        self,
        asset: str | None,
        window_hours: int = 72,
        top_k: int = 8,
        query_text: str | None = None,
    ) -> list[NewsDoc]:
        scored, _ = self._retrieve_news_scored(
            asset, window_hours, top_k, query_text=query_text)
        return [s.doc for s in scored]

    def _retrieve_news_scored(
        self,
        asset: str | None,
        window_hours: int = 72,
        top_k: int = 8,
        query_text: str | None = None,
    ) -> tuple[list[ScoredNews], str]:
        """Returns (scored_news, index_status)."""
        now_ts = time.time()
        since = now_ts - window_hours * 3600
        candidates = self.store.news_in_window(since_ts=since, limit=500)
        kws = asset_keywords(asset) if asset else []

        # Pre-filter to asset-relevant docs
        relevant = []
        for d in candidates:
            text = f"{d.title} {d.raw_text}"
            if asset and asset not in d.asset_tags and not _keyword_hits(text, kws):
                continue
            relevant.append(d)

        # Vector scoring via persistent index
        vector_scores: list[float] = []
        index_status = "none"
        if relevant and query_text:
            texts = [f"{d.title} {d.raw_text}" for d in relevant]
            doc_ids = [d.id for d in relevant]
            li = self._get_index("news", texts, doc_ids)
            index_status = li.status
            if li.index.is_available:
                # If stale: use existing scores but trigger rebuild asynchronously
                # (for simplicity we rebuild inline on stale)
                if li.status == "stale":
                    li = self._rebuild_index("news", texts, doc_ids)
                    index_status = "rebuilt"
                vector_scores = li.index.query(query_text)
                logger.debug("news vector index: status=%s docs=%d",
                             index_status, li.meta.n_docs)

        results: list[ScoredNews] = []
        vector_enabled = bool(vector_scores)
        for i, d in enumerate(relevant):
            text = f"{d.title} {d.raw_text}"
            vscore = vector_scores[i] if vector_enabled and i < len(vector_scores) else 0.0
            score = _score_news(d, kws, now_ts, window_hours, vscore)
            if score == 0.0:
                continue

            if asset and asset in d.asset_tags:
                reason = "asset_tag"
            elif kws and _keyword_hits(text, kws):
                hit_kw = next((k for k in kws if k.lower() in text.lower()), "keyword")
                reason = f"keyword:{hit_kw}"
            elif vscore > 0.1:
                reason = "vector"
            else:
                reason = "recency"
            results.append(ScoredNews(doc=d, score=score, retrieval_reason=reason))

        results.sort(key=lambda s: s.score, reverse=True)
        return results[:top_k], index_status

    # ── community ─────────────────────────────────────────────────────────────

    def retrieve_community(
        self,
        asset: str | None,
        window_hours: int = 72,
        top_k: int = 20,
        platform: str | None = None,
        query_text: str | None = None,
    ) -> list[CommunityDoc]:
        scored, _ = self._retrieve_community_scored(
            asset, window_hours, top_k, platform, query_text=query_text)
        return [s.doc for s in scored]

    def _retrieve_community_scored(
        self,
        asset: str | None,
        window_hours: int = 72,
        top_k: int = 20,
        platform: str | None = None,
        query_text: str | None = None,
    ) -> tuple[list[ScoredCommunity], str]:
        """Returns (scored_community, index_status)."""
        now_ts = time.time()
        since = now_ts - window_hours * 3600
        candidates = self.store.community_in_window(
            since_ts=since, platform=platform, limit=1000,
        )
        kws = asset_keywords(asset) if asset else []

        # Pre-filter
        relevant = []
        for d in candidates:
            if asset and asset not in d.asset_tags and not _keyword_hits(d.raw_text, kws):
                continue
            relevant.append(d)

        # Vector scoring via persistent index
        vector_scores: list[float] = []
        index_status = "none"
        if relevant and query_text:
            texts = [d.raw_text for d in relevant]
            doc_ids = [d.id for d in relevant]
            li = self._get_index("community", texts, doc_ids)
            index_status = li.status
            if li.index.is_available:
                if li.status == "stale":
                    li = self._rebuild_index("community", texts, doc_ids)
                    index_status = "rebuilt"
                vector_scores = li.index.query(query_text)

        results: list[ScoredCommunity] = []
        vector_enabled = bool(vector_scores)
        for i, d in enumerate(relevant):
            vscore = vector_scores[i] if vector_enabled and i < len(vector_scores) else 0.0
            score = _score_community(d, kws, now_ts, window_hours, vscore)
            if score == 0.0:
                continue

            if asset and asset in d.asset_tags:
                reason = "asset_tag"
            elif kws and _keyword_hits(d.raw_text, kws):
                hit_kw = next((k for k in kws if k.lower() in d.raw_text.lower()), "keyword")
                reason = f"keyword:{hit_kw}"
            elif vscore > 0.1:
                reason = "vector"
            else:
                reason = "recency"
            results.append(ScoredCommunity(doc=d, score=score, retrieval_reason=reason))

        results.sort(key=lambda s: s.score, reverse=True)
        return results[:top_k], index_status

    # ── unified interface ─────────────────────────────────────────────────────

    def retrieve(
        self,
        asset: str | None,
        window_hours: int = 72,
        top_k_news: int = 8,
        top_k_community: int = 20,
        kind: Kind = "both",
        query_text: str | None = None,
    ) -> RetrievedEvidence:
        scored_news: list[ScoredNews] = []
        scored_community: list[ScoredCommunity] = []
        news_idx_status = "none"
        comm_idx_status = "none"

        if kind in ("news", "both"):
            scored_news, news_idx_status = self._retrieve_news_scored(
                asset, window_hours, top_k_news, query_text=query_text)
        if kind in ("community", "both"):
            scored_community, comm_idx_status = self._retrieve_community_scored(
                asset, window_hours, top_k_community,
                query_text=query_text)

        vector_enabled = any(s.retrieval_reason == "vector"
                             for s in scored_news + scored_community)
        # Combine index statuses — prefer the more informative one
        _status_priority = {"rebuilt": 4, "loaded": 3, "stale": 2,
                            "fallback": 1, "empty": 0, "none": -1}
        index_status = max(
            [news_idx_status, comm_idx_status],
            key=lambda s: _status_priority.get(s, -1),
        )

        return RetrievedEvidence(
            news=[s.doc for s in scored_news],
            community=[s.doc for s in scored_community],
            window_hours=window_hours,
            asset=asset,
            scored_news=scored_news,
            scored_community=scored_community,
            vector_enabled=vector_enabled,
            index_status=index_status,
        )

    def rebuild_indexes(self, window_hours: int = 72) -> dict[str, str]:
        """
        Force-rebuild both news and community indexes.
        Returns {"news": status, "community": status}.
        """
        now_ts = time.time()
        since = now_ts - window_hours * 3600
        results: dict[str, str] = {}

        news_docs = self.store.news_in_window(since_ts=since, limit=5000)
        texts = [f"{d.title} {d.raw_text}" for d in news_docs]
        ids = [d.id for d in news_docs]
        li = self._vs.force_rebuild("news", texts, ids)
        results["news"] = li.status

        comm_docs = self.store.community_in_window(since_ts=since, limit=10000)
        texts = [d.raw_text for d in comm_docs]
        ids = [d.id for d in comm_docs]
        li = self._vs.force_rebuild("community", texts, ids)
        results["community"] = li.status

        return results
