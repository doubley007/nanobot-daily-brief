"""
X (Twitter) community sentiment analysis for financial daily brief.

Uses Twitter API v2 recent search endpoint to find finance-related tweets,
then applies the same shared topic classification and sentiment analysis
pipeline as the Reddit source.

Requires X_BEARER_TOKEN env var.  When the token is not set, gracefully
returns an empty result so the brief pipeline continues without X data.

Filtering is controlled via env vars (see XFilterConfig).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import requests

from community.base import CommunityPost, CommunitySentiment, TopicCluster
from community.analysis import (
    DEFAULT_FINANCE_KEYWORDS,
    SENTIMENT_LABELS_CN,
    classify_post,
)
from community.clustering import cluster_posts, mark_rising_clusters
from community.llm_analyst import dedupe_posts, run_llm_pipeline
from community.normalize import normalize_posts

logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────

REQUEST_TIMEOUT = 15
MAX_RESULTS_PER_QUERY = 100  # Twitter API v2 max for recent search


class XPlanUnavailableError(Exception):
    """Raised when the X API returns 402 — endpoint not on current plan."""

# Cashtags and keywords used in the search query.
# Twitter search supports cashtags ($SPY) and keywords; we combine them
# with OR logic in a single query string.
DEFAULT_SEARCH_QUERIES = [
    "$SPY OR $QQQ OR $DIA OR $TLT OR $VIX",
    "fed rate OR treasury yield OR inflation CPI",
    "recession OR tariff OR earnings report",
]

DEFAULT_PRIORITY_THEMES = [
    "美联储与利率政策",
    "美债与收益率",
    "通胀与物价",
    "衰退与宏观经济",
]

PRIORITY_BOOST_FACTOR = 1.5


@dataclass
class XFilterConfig:
    """
    All filtering knobs in one place. Loaded from env vars with sensible defaults.

    Env vars:
      X_BEARER_TOKEN             Twitter API v2 bearer token (required)
      X_SEARCH_QUERIES           comma-separated search queries
      X_MIN_LIKES                minimum like count (default 10)
      X_MIN_RETWEETS             minimum retweet count (default 3)
      X_FINANCE_KEYWORDS         comma-separated extra keywords (merged with defaults)
      X_PRIORITY_THEMES          comma-separated topic labels in Chinese
      X_PRIORITY_BOOST           float multiplier for priority themes (default 1.5)
      X_TOP_N_TOPICS             max topics to return (default 4)
    """
    bearer_token: str = ""
    search_queries: list[str] = field(default_factory=lambda: list(DEFAULT_SEARCH_QUERIES))
    min_likes: int = 10
    min_retweets: int = 3
    finance_keywords: list[str] = field(default_factory=lambda: list(DEFAULT_FINANCE_KEYWORDS))
    priority_themes: list[str] = field(default_factory=lambda: list(DEFAULT_PRIORITY_THEMES))
    priority_boost: float = PRIORITY_BOOST_FACTOR
    top_n_topics: int = 4


def load_filter_config() -> XFilterConfig:
    """Build config from env vars, falling back to defaults."""
    config = XFilterConfig()

    config.bearer_token = os.getenv("X_BEARER_TOKEN", "").strip()

    env_queries = os.getenv("X_SEARCH_QUERIES", "").strip()
    if env_queries:
        config.search_queries = [q.strip() for q in env_queries.split(",") if q.strip()]

    env_likes = os.getenv("X_MIN_LIKES", "").strip()
    if env_likes:
        config.min_likes = int(env_likes)

    env_retweets = os.getenv("X_MIN_RETWEETS", "").strip()
    if env_retweets:
        config.min_retweets = int(env_retweets)

    env_keywords = os.getenv("X_FINANCE_KEYWORDS", "").strip()
    if env_keywords:
        extra = [k.strip().lower() for k in env_keywords.split(",") if k.strip()]
        config.finance_keywords = list(set(config.finance_keywords + extra))

    env_priority = os.getenv("X_PRIORITY_THEMES", "").strip()
    if env_priority:
        config.priority_themes = [t.strip() for t in env_priority.split(",") if t.strip()]

    env_boost = os.getenv("X_PRIORITY_BOOST", "").strip()
    if env_boost:
        config.priority_boost = float(env_boost)

    env_topn = os.getenv("X_TOP_N_TOPICS", "").strip()
    if env_topn:
        config.top_n_topics = int(env_topn)

    return config


# ─── Fetch tweets ────────────────────────────────────────────────────────────

def _search_recent_tweets(
    query: str,
    bearer_token: str,
    max_results: int = MAX_RESULTS_PER_QUERY,
) -> list[dict[str, Any]]:
    """
    Search recent tweets using Twitter API v2.
    Returns list of tweet data dicts with public_metrics included.
    """
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {bearer_token}"}
    params = {
        "query": f"{query} lang:en -is:retweet",
        "max_results": min(max_results, 100),
        "tweet.fields": "public_metrics,created_at,author_id",
    }

    response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)

    if response.status_code == 402:
        raise XPlanUnavailableError(
            "X recent search unavailable on current plan (HTTP 402)"
        )
    if response.status_code == 429:
        logger.warning("X API rate limited (HTTP 429) — skipping query")
        return []
    if response.status_code == 401:
        logger.warning("X API auth failed (HTTP 401) — check X_BEARER_TOKEN")
        return []

    response.raise_for_status()

    data = response.json()
    return data.get("data", [])


def _parse_tweets(raw_tweets: list[dict[str, Any]]) -> list[CommunityPost]:
    """Convert raw API tweet dicts into CommunityPost objects."""
    posts: list[CommunityPost] = []
    for t in raw_tweets:
        metrics = t.get("public_metrics", {})
        posts.append(CommunityPost(
            title=t.get("text", ""),
            score=metrics.get("like_count", 0),
            num_comments=metrics.get("reply_count", 0),
            url=f"https://x.com/i/status/{t.get('id', '')}",
            subreddit="x",  # reuse field to mark source platform
            created_utc=0.0,
        ))
    return posts


def fetch_x_posts(config: XFilterConfig) -> list[CommunityPost]:
    """Run all search queries and collect tweets as CommunityPost list."""
    all_posts: list[CommunityPost] = []
    seen_ids: set[str] = set()

    for query in config.search_queries:
        try:
            raw = _search_recent_tweets(query, config.bearer_token)
        except XPlanUnavailableError as e:
            logger.info("%s — skipping all X queries this run", e)
            return []
        except Exception as e:
            logger.warning("Failed to search X for '%s': %s", query[:50], e)
            continue

        for t in raw:
            tid = t.get("id", "")
            if tid in seen_ids:
                continue
            seen_ids.add(tid)
        tweets = _parse_tweets([t for t in raw if t.get("id", "") in seen_ids])
        all_posts.extend(tweets)
        logger.info("Fetched %d tweets for query: %s", len(tweets), query[:50])

    return all_posts


# ─── Post-level filtering ────────────────────────────────────────────────────

def _passes_engagement_threshold(post: CommunityPost, config: XFilterConfig) -> bool:
    return post.score >= config.min_likes or post.num_comments >= config.min_retweets


def _is_finance_related(post: CommunityPost, config: XFilterConfig) -> bool:
    text = post.title.lower()
    return any(kw in text for kw in config.finance_keywords)


def filter_posts(posts: list[CommunityPost], config: XFilterConfig) -> list[CommunityPost]:
    """
    Apply post-level filters:
      1. Engagement threshold (likes OR retweets)
      2. Finance relevance (keyword OR topic classifier)
    """
    before = len(posts)

    posts = [p for p in posts if _passes_engagement_threshold(p, config)]
    after_engagement = len(posts)

    kept: list[CommunityPost] = []
    for p in posts:
        if _is_finance_related(p, config):
            kept.append(p)
            continue
        topic, _ = classify_post(p.title)
        if topic:
            kept.append(p)
    posts = kept
    after_finance = len(posts)

    logger.info(
        "X filtered: %d fetched → %d after engagement (min_likes=%d, min_retweets=%d) "
        "→ %d after finance relevance",
        before, after_engagement, config.min_likes, config.min_retweets, after_finance,
    )
    return posts


# ─── Main entry point ────────────────────────────────────────────────────────

def fetch_x_sentiment(
    config: XFilterConfig | None = None,
    llm_callable: "Callable[[str], str] | None" = None,
) -> CommunitySentiment:
    """X sentiment pipeline — fully LLM-interpreted."""
    if config is None:
        config = load_filter_config()

    if not config.bearer_token:
        logger.info("X_BEARER_TOKEN not set — skipping X sentiment")
        return CommunitySentiment(platform="x")

    if llm_callable is None:
        logger.info("X sentiment skipped — LLM not available")
        return CommunitySentiment(platform="x")

    raw_posts = fetch_x_posts(config)
    if not raw_posts:
        logger.warning("No tweets fetched from X")
        return CommunitySentiment(platform="x")

    legacy_posts = filter_posts(raw_posts, config)
    unified = normalize_posts("x", legacy_posts)
    unified = dedupe_posts(unified)
    if not unified:
        logger.warning("All X tweets filtered out")
        return CommunitySentiment(platform="x")

    clusters = cluster_posts(unified)
    clusters = mark_rising_clusters(clusters)

    kept, overall = run_llm_pipeline(
        platform="x",
        posts=unified,
        clusters=clusters,
        top_n_topics=config.top_n_topics,
        llm_callable=llm_callable,
    )

    return CommunitySentiment(
        platform="x",
        trending_topics=kept,
        overall_sentiment=overall,
        post_count=len(unified),
    )


# ─── Standalone test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = load_filter_config()
    print(f"Config: bearer_token={'set' if config.bearer_token else 'NOT SET'}")
    print(f"  search_queries={config.search_queries}")
    print(f"  min_likes={config.min_likes}, min_retweets={config.min_retweets}")
    print(f"  priority_themes={config.priority_themes}")
    print(f"  priority_boost={config.priority_boost}, top_n={config.top_n_topics}")
    print()

    from llm_adapter import local_llm_callable
    result = fetch_x_sentiment(config, llm_callable=local_llm_callable)
    print(f"Platform: {result.platform}")
    print(f"Posts after filtering: {result.post_count}")
    print(f"Overall sentiment: {result.overall_sentiment}")
    print(f"Narrative: {result.narrative}")
    print(f"Topics: {len(result.trending_topics)}")
    print()
    for c in result.trending_topics:
        sent_cn = SENTIMENT_LABELS_CN.get(c.sentiment.label, "中性")
        title = c.headline or c.rule_label
        print(f"【{title}】 {c.post_count} 条讨论 | 情绪{sent_cn}")
        print(f"  争论点：{c.discussion_focus}")
        print(f"  理由：{c.reasons}")
        print(f"  策略含义：{c.market_relevance}")
        print()
