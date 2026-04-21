"""
Topic clustering for unified community posts.

Two strategies, picked at runtime:

  1. Embedding-based (preferred). Uses the Ollama embeddings endpoint with
     a small local model. Groups posts by cosine similarity with an
     agglomerative single-link threshold. Produces finer, event-level
     clusters than keyword bucketing.

  2. Keyword-fallback. Reuses community.analysis.classify_post when
     embeddings are unreachable or disabled. Coarser but dependency-free.

Trending detection:
  After clustering we compute heat per cluster (engagement-weighted) and
  flag "rising" clusters whose heat is meaningfully above a platform-wide
  baseline. That's what lets the Community Analyst identify what's
  *heating up*, not just what's loud.
"""
from __future__ import annotations

import logging
import math
import os
import uuid
from collections import defaultdict
from typing import Any

import requests

from community.analysis import classify_post
from community.schema import TopicCluster, UnifiedPost

logger = logging.getLogger(__name__)


# ─── Embeddings (Ollama native endpoint, lightweight) ────────────────────────

OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1").rstrip("/")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
EMBED_TIMEOUT = 20
# Average-link threshold on nomic-embed-text. Calibrated empirically on the
# Reddit hot feed: 0.70 produces 3-5 event-level clusters of 2-4 posts each
# (oil+hormuz, yuan-pricing, portfolio-advice, etc.). Single-link chained
# unrelated posts at this threshold; average-link is the reason it's safe.
SIMILARITY_THRESHOLD = float(os.getenv("COMMUNITY_CLUSTER_THRESHOLD", "0.70"))


_EMBED_FALLBACK_WARNED = False


def _warn_embed_fallback_once(reason: str) -> None:
    global _EMBED_FALLBACK_WARNED
    if _EMBED_FALLBACK_WARNED:
        return
    _EMBED_FALLBACK_WARNED = True
    logger.warning(
        "Community embeddings unavailable (%s). Falling back to keyword "
        "clustering — topic granularity will be coarser. To enable "
        "event-level clusters run: `ollama pull %s`",
        reason, OLLAMA_EMBED_MODEL,
    )


def _embed_batch(texts: list[str]) -> list[list[float]] | None:
    """
    Call Ollama's native /api/embeddings. Returns None on any failure so
    callers can cleanly fall back to keyword clustering. Warns loudly
    exactly once per process so logs don't drown in per-post retries.
    """
    if not texts:
        return []
    base = OLLAMA_API_BASE.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    url = f"{base}/api/embeddings"
    vectors: list[list[float]] = []
    for text in texts:
        try:
            r = requests.post(
                url,
                json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
                timeout=EMBED_TIMEOUT,
            )
            r.raise_for_status()
            vectors.append(r.json().get("embedding", []))
        except Exception as e:
            _warn_embed_fallback_once(str(e)[:80])
            return None
    return vectors


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _agglomerative(vectors: list[list[float]], threshold: float) -> list[list[int]]:
    """
    Average-link agglomerative clustering by cosine similarity.

    Single-link (nearest-neighbor merge) is known to produce long chains when
    the threshold is anywhere near the bulk of the pairwise distribution.
    Average-link only merges when the *mean* similarity between all points
    in two clusters meets the threshold — this resists chaining while
    still being O(n^2) at this scale.
    """
    n = len(vectors)
    if n == 0:
        return []

    # Precompute pairwise sims once
    sims: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = _cosine(vectors[i], vectors[j])
            sims[i][j] = s
            sims[j][i] = s

    clusters: list[list[int]] = [[i] for i in range(n)]

    def _avg_link(a: list[int], b: list[int]) -> float:
        return sum(sims[x][y] for x in a for y in b) / (len(a) * len(b))

    while True:
        best_i, best_j, best_s = -1, -1, threshold
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                s = _avg_link(clusters[i], clusters[j])
                if s >= best_s:
                    best_s = s
                    best_i, best_j = i, j
        if best_i < 0:
            break
        clusters[best_i].extend(clusters[best_j])
        del clusters[best_j]

    return clusters


# ─── Cluster building ────────────────────────────────────────────────────────

def _heat(posts: list[UnifiedPost]) -> float:
    return float(sum(p.engagement_raw for p in posts))


def _pick_rule_label(posts: list[UnifiedPost]) -> str:
    """Pick the most common keyword-bucket label as a fallback topic name."""
    counts: dict[str, int] = defaultdict(int)
    for p in posts:
        topic, _ = classify_post(p.title)
        if topic:
            counts[topic] += 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _top_sample_titles(posts: list[UnifiedPost], n: int = 6) -> list[str]:
    ranked = sorted(posts, key=lambda p: p.engagement_raw, reverse=True)
    return [p.title.strip() for p in ranked[:n] if p.title.strip()]


def _build_cluster(posts: list[UnifiedPost]) -> TopicCluster:
    platforms = sorted({p.platform for p in posts})
    return TopicCluster(
        cluster_id=uuid.uuid4().hex[:12],
        posts=list(posts),
        platforms=platforms,
        rule_label=_pick_rule_label(posts),
        heat_score=_heat(posts),
        sample_titles=_top_sample_titles(posts),
    )


def _embedding_cluster(posts: list[UnifiedPost]) -> list[TopicCluster] | None:
    texts = [p.text for p in posts]
    vectors = _embed_batch(texts)
    if vectors is None:
        return None
    groups = _agglomerative(vectors, SIMILARITY_THRESHOLD)
    return [_build_cluster([posts[i] for i in g]) for g in groups]


def _keyword_cluster(posts: list[UnifiedPost]) -> list[TopicCluster]:
    buckets: dict[str, list[UnifiedPost]] = defaultdict(list)
    for p in posts:
        topic, _ = classify_post(p.title)
        if topic:
            buckets[topic].append(p)
    clusters = [_build_cluster(group) for group in buckets.values()]
    for c in clusters:
        # rule label is the bucket name for keyword-mode
        c.rule_label = next(
            (k for k, v in buckets.items() if v is c.posts), c.rule_label
        )
    return clusters


def cluster_posts(posts: list[UnifiedPost], use_embeddings: bool = True) -> list[TopicCluster]:
    """
    Cluster unified posts into topic groups.

    Tries embedding-based clustering first; silently falls back to keyword
    buckets if embeddings are unavailable. Returns clusters sorted by heat.
    """
    if not posts:
        return []

    clusters: list[TopicCluster] | None = None
    if use_embeddings:
        clusters = _embedding_cluster(posts)
    if clusters is None:
        clusters = _keyword_cluster(posts)

    # A "topic" needs at least two posts. One viral headline is a news item,
    # not a community conversation — rendering it as a topic overstates signal.
    clusters = [c for c in clusters if c.post_count >= 2]

    clusters.sort(key=lambda c: c.heat_score, reverse=True)
    return clusters


# ─── Trending detection ──────────────────────────────────────────────────────

def mark_rising_clusters(
    clusters: list[TopicCluster],
    rise_threshold: float = 1.8,
) -> list[TopicCluster]:
    """
    Flag clusters whose heat is meaningfully above the cohort median.
    Without historical data we use in-run relative heat as a proxy.
    """
    if not clusters:
        return clusters

    heats = sorted(c.heat_score for c in clusters)
    mid = heats[len(heats) // 2] or 1.0
    for c in clusters:
        c.rise_ratio = c.heat_score / mid if mid else 1.0
        c.is_rising = c.rise_ratio >= rise_threshold
    return clusters
