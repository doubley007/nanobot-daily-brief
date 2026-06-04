"""
Persistent Vector Index Store — 预构建并持久化 TF-IDF 索引。

解决 v4 的问题：每次查询重建 TF-IDF 矩阵（小语料无感知，但语料增长后会变慢）。
v5 改成：
  - 索引构建一次，持久化到 pickle 文件
  - 每次 Retriever 初始化时加载，无需重建
  - 数据更新后支持增量刷新（rebuild_if_stale）
  - Trace 标记：index_loaded / index_rebuilt / index_stale / index_fallback

设计约束：
  - 只依赖 numpy（已在 requirements）+ pickle（stdlib）
  - 索引文件路径由 VECTOR_INDEX_DIR 环境变量控制，默认 logs/
  - 每种索引（news / community）独立文件，互不干扰
  - 没有 numpy 时 fallback 到关键词模式，不崩溃
"""
from __future__ import annotations

import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from assistant.rag.vector_index import TFIDFIndex, _tokenize

logger = logging.getLogger(__name__)

# 索引文件最大年龄（秒）；超过则标记为 stale，建议刷新
INDEX_MAX_AGE_SECONDS = 3600  # 1 hour
# doc 数量变化超过此比例视为 stale
INDEX_STALE_DOC_RATIO = 0.1


def _resolve_index_dir() -> Path:
    env = os.getenv("VECTOR_INDEX_DIR", "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[4] / "logs" / "vector_index"


@dataclass
class IndexMeta:
    """存储在 pickle 文件旁边，记录构建时的元数据。"""
    built_at: float = 0.0
    n_docs: int = 0
    kind: str = ""         # "news" | "community"
    asset: str = ""        # "" = all assets
    vocab_size: int = 0


@dataclass
class LoadedIndex:
    """查询时使用的索引 + 元数据 + 状态标记。"""
    index: TFIDFIndex
    meta: IndexMeta
    status: str  # "loaded" | "rebuilt" | "stale" | "empty" | "fallback"
    # 每条文档的原始 id，供 retriever 做 doc→score 映射
    doc_ids: list[str] = field(default_factory=list)


class VectorIndexStore:
    """
    预构建并持久化 TF-IDF 索引。

    Usage:
        store = VectorIndexStore()
        loaded = store.load_or_rebuild("news", docs_texts, doc_ids)
        # loaded.status in ("loaded", "rebuilt", "stale", "empty", "fallback")
        scores = loaded.index.query(query_text)
        # scores[i] 对应 doc_ids[i]
    """

    def __init__(self, index_dir: str | Path | None = None) -> None:
        self._dir = Path(index_dir) if index_dir else _resolve_index_dir()
        self._dir.mkdir(parents=True, exist_ok=True)

    def _index_path(self, kind: str) -> Path:
        return self._dir / f"{kind}.pkl"

    def _meta_path(self, kind: str) -> Path:
        return self._dir / f"{kind}.meta.pkl"

    def _save(self, kind: str, idx: TFIDFIndex, meta: IndexMeta,
              doc_ids: list[str]) -> None:
        try:
            tmp_idx = self._index_path(kind).with_suffix(".pkl.tmp")
            tmp_meta = self._meta_path(kind).with_suffix(".meta.pkl.tmp")
            with open(tmp_idx, "wb") as f:
                pickle.dump((idx, doc_ids), f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(tmp_meta, "wb") as f:
                pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_idx.replace(self._index_path(kind))
            tmp_meta.replace(self._meta_path(kind))
            logger.debug("VectorIndexStore: saved %s (%d docs)", kind, meta.n_docs)
        except Exception as e:
            logger.warning("VectorIndexStore: save failed for %s: %s", kind, e)

    def _load_raw(self, kind: str) -> tuple[TFIDFIndex, list[str], IndexMeta] | None:
        try:
            idx_path = self._index_path(kind)
            meta_path = self._meta_path(kind)
            if not idx_path.exists() or not meta_path.exists():
                return None
            with open(idx_path, "rb") as f:
                idx, doc_ids = pickle.load(f)
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            return idx, doc_ids, meta
        except Exception as e:
            logger.warning("VectorIndexStore: load failed for %s: %s", kind, e)
            return None

    def _is_stale(self, meta: IndexMeta, current_n_docs: int) -> bool:
        age = time.time() - meta.built_at
        if age > INDEX_MAX_AGE_SECONDS:
            return True
        if meta.n_docs == 0:
            return current_n_docs > 0
        ratio = abs(current_n_docs - meta.n_docs) / meta.n_docs
        return ratio > INDEX_STALE_DOC_RATIO

    def load_or_rebuild(
        self,
        kind: str,
        texts: Sequence[str],
        doc_ids: list[str],
        force_rebuild: bool = False,
    ) -> LoadedIndex:
        """
        Load existing index if fresh, or rebuild from `texts`.

        Returns LoadedIndex with .status indicating what happened:
          "loaded"   — cache hit, used as-is
          "rebuilt"  — cache miss or stale, rebuilt from texts
          "stale"    — loaded but doc count drifted (caller should schedule refresh)
          "empty"    — no docs provided (returns empty index)
          "fallback" — numpy unavailable, index not built
        """
        n = len(texts)
        if n == 0:
            idx = TFIDFIndex()
            idx.build([])
            return LoadedIndex(
                index=idx,
                meta=IndexMeta(built_at=time.time(), n_docs=0, kind=kind),
                status="empty",
                doc_ids=[],
            )

        # Try loading existing index
        if not force_rebuild:
            raw = self._load_raw(kind)
            if raw is not None:
                idx, saved_ids, meta = raw
                stale = self._is_stale(meta, n)
                if not stale:
                    logger.debug("VectorIndexStore: loaded %s (%d docs)", kind, meta.n_docs)
                    return LoadedIndex(index=idx, meta=meta,
                                       status="loaded", doc_ids=saved_ids)
                else:
                    logger.debug(
                        "VectorIndexStore: %s stale (saved=%d, current=%d, age=%.0fs)",
                        kind, meta.n_docs, n, time.time() - meta.built_at,
                    )
                    # Return stale index but mark it — caller may still use it
                    return LoadedIndex(index=idx, meta=meta,
                                       status="stale", doc_ids=saved_ids)

        # Rebuild
        idx = TFIDFIndex()
        idx.build(list(texts))
        if not idx.is_available:
            return LoadedIndex(
                index=idx,
                meta=IndexMeta(built_at=time.time(), n_docs=n, kind=kind),
                status="fallback",
                doc_ids=list(doc_ids),
            )

        meta = IndexMeta(
            built_at=time.time(),
            n_docs=n,
            kind=kind,
            vocab_size=len(idx._vocab),
        )
        self._save(kind, idx, meta, list(doc_ids))
        logger.info("VectorIndexStore: rebuilt %s (%d docs, %d vocab)",
                    kind, n, meta.vocab_size)
        return LoadedIndex(index=idx, meta=meta, status="rebuilt",
                           doc_ids=list(doc_ids))

    def force_rebuild(
        self,
        kind: str,
        texts: Sequence[str],
        doc_ids: list[str],
    ) -> LoadedIndex:
        """Always rebuild, ignoring cache."""
        return self.load_or_rebuild(kind, texts, doc_ids, force_rebuild=True)

    def invalidate(self, kind: str) -> None:
        """Delete stored index files to force next rebuild."""
        for p in (self._index_path(kind), self._meta_path(kind)):
            try:
                if p.exists():
                    p.unlink()
            except Exception as e:
                logger.warning("VectorIndexStore.invalidate: %s: %s", kind, e)


# ─── module-level singleton ───────────────────────────────────────────────────

_default_vs: VectorIndexStore | None = None


def default_vector_store() -> VectorIndexStore:
    global _default_vs
    if _default_vs is None:
        _default_vs = VectorIndexStore()
    return _default_vs


def reset_vector_store() -> None:
    global _default_vs
    _default_vs = None
