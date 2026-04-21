"""
Reddit community sentiment analysis for financial daily brief.

Uses Reddit's public JSON API (no auth required, no extra dependencies).
Fetches recent hot posts from configured subreddits, detects trending
financial topics via scored multi-keyword matching, performs engagement-
weighted sentiment analysis, and generates per-topic market commentary.

Filtering is controlled via env vars (see RedditFilterConfig).
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
    TOPIC_RULES,
    MIN_TOPIC_SCORE,
    classify_post as _classify_post,
)
from community.clustering import cluster_posts, mark_rising_clusters
from community.llm_analyst import dedupe_posts, run_llm_pipeline
from community.normalize import normalize_posts

logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────

POSTS_PER_SUBREDDIT = 25
REQUEST_TIMEOUT = 10
USER_AGENT = "nanobot-financial-brief/1.0"

DEFAULT_SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "stockmarket",
    "economics",
]

# Finance keywords are defined in community.analysis and imported above.

# Priority themes get a ranking boost in cluster scoring so they surface
# even when they have lower raw engagement than other topics.
DEFAULT_PRIORITY_THEMES = [
    "美联储与利率政策",
    "美债与收益率",
    "通胀与物价",
    "衰退与宏观经济",
]

PRIORITY_BOOST_FACTOR = 1.5  # multiply heat score by this for priority topics


@dataclass
class RedditFilterConfig:
    """
    All filtering knobs in one place. Loaded from env vars with sensible defaults.

    Env vars:
      REDDIT_SUBREDDITS          comma-separated subreddit names
      REDDIT_MIN_SCORE           minimum upvote score (default 5)
      REDDIT_MIN_COMMENTS        minimum comment count (default 2)
      REDDIT_FINANCE_KEYWORDS    comma-separated extra keywords (merged with defaults)
      REDDIT_PRIORITY_THEMES     comma-separated topic labels in Chinese
      REDDIT_PRIORITY_BOOST      float multiplier for priority themes (default 1.5)
      REDDIT_TOP_N_TOPICS        max topics to return (default 4)
    """
    subreddits: list[str] = field(default_factory=lambda: list(DEFAULT_SUBREDDITS))
    min_score: int = 5
    min_comments: int = 2
    finance_keywords: list[str] = field(default_factory=lambda: list(DEFAULT_FINANCE_KEYWORDS))
    priority_themes: list[str] = field(default_factory=lambda: list(DEFAULT_PRIORITY_THEMES))
    priority_boost: float = PRIORITY_BOOST_FACTOR
    top_n_topics: int = 4


def load_filter_config() -> RedditFilterConfig:
    """Build config from env vars, falling back to defaults."""
    config = RedditFilterConfig()

    env_subs = os.getenv("REDDIT_SUBREDDITS", "").strip()
    if env_subs:
        config.subreddits = [s.strip() for s in env_subs.split(",") if s.strip()]

    env_score = os.getenv("REDDIT_MIN_SCORE", "").strip()
    if env_score:
        config.min_score = int(env_score)

    env_comments = os.getenv("REDDIT_MIN_COMMENTS", "").strip()
    if env_comments:
        config.min_comments = int(env_comments)

    env_keywords = os.getenv("REDDIT_FINANCE_KEYWORDS", "").strip()
    if env_keywords:
        extra = [k.strip().lower() for k in env_keywords.split(",") if k.strip()]
        config.finance_keywords = list(set(config.finance_keywords + extra))

    env_priority = os.getenv("REDDIT_PRIORITY_THEMES", "").strip()
    if env_priority:
        config.priority_themes = [t.strip() for t in env_priority.split(",") if t.strip()]

    env_boost = os.getenv("REDDIT_PRIORITY_BOOST", "").strip()
    if env_boost:
        config.priority_boost = float(env_boost)

    env_topn = os.getenv("REDDIT_TOP_N_TOPICS", "").strip()
    if env_topn:
        config.top_n_topics = int(env_topn)

    return config



# Topic classification, sentiment scoring, discussion focus, and market relevance
# are all imported from community.analysis (see imports above).


# ─── Response cache ──────────────────────────────────────────────────────────
#
# Simple TTL-based in-memory cache for Reddit API responses.
# Prevents redundant fetches if fetch_reddit_sentiment is called multiple
# times within the same process (e.g. during testing or retries).

import time as _time

_CACHE_TTL = int(os.getenv("REDDIT_CACHE_TTL", "300"))  # 5 min default
_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}


def _get_cached(key: str) -> list[dict[str, Any]] | None:
    entry = _cache.get(key)
    if entry is None:
        return None
    ts, data = entry
    if _time.time() - ts > _CACHE_TTL:
        del _cache[key]
        return None
    return data


def _set_cache(key: str, data: list[dict[str, Any]]) -> None:
    _cache[key] = (_time.time(), data)


# ─── Fetch posts ─────────────────────────────────────────────────────────────

def _fetch_subreddit_hot(subreddit: str, limit: int = POSTS_PER_SUBREDDIT) -> list[dict[str, Any]]:
    """Fetch hot posts from a subreddit using the public JSON API (with caching)."""
    cache_key = f"{subreddit}:{limit}"
    cached = _get_cached(cache_key)
    if cached is not None:
        logger.info("Using cached response for r/%s", subreddit)
        return cached

    url = f"https://www.reddit.com/r/{subreddit}/hot.json"
    params = {"limit": limit, "raw_json": 1}
    headers = {"User-Agent": USER_AGENT}

    response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    data = response.json()
    children = data.get("data", {}).get("children", [])
    result = [child["data"] for child in children if child.get("kind") == "t3"]
    _set_cache(cache_key, result)
    return result


def _parse_posts(raw_posts: list[dict[str, Any]], subreddit: str) -> list[CommunityPost]:
    posts: list[CommunityPost] = []
    for p in raw_posts:
        if p.get("stickied"):
            continue
        posts.append(CommunityPost(
            title=p.get("title", ""),
            score=p.get("score", 0),
            num_comments=p.get("num_comments", 0),
            url=f"https://reddit.com{p.get('permalink', '')}",
            subreddit=subreddit,
            created_utc=p.get("created_utc", 0.0),
        ))
    return posts


def fetch_reddit_posts(subreddits: list[str]) -> list[CommunityPost]:
    """Fetch hot posts from all configured subreddits."""
    all_posts: list[CommunityPost] = []
    for sub in subreddits:
        try:
            raw = _fetch_subreddit_hot(sub)
            posts = _parse_posts(raw, sub)
            all_posts.extend(posts)
            logger.info("Fetched %d posts from r/%s", len(posts), sub)
        except Exception as e:
            logger.warning("Failed to fetch r/%s: %s", sub, e)

    return all_posts


# ─── Post-level filtering ───────────────────────────────────────────────────

def _passes_engagement_threshold(post: CommunityPost, config: RedditFilterConfig) -> bool:
    return post.score >= config.min_score or post.num_comments >= config.min_comments


def _is_finance_related(post: CommunityPost, config: RedditFilterConfig) -> bool:
    """Check if a post contains any finance keyword."""
    text = post.title.lower()
    return any(kw in text for kw in config.finance_keywords)


def filter_posts(posts: list[CommunityPost], config: RedditFilterConfig) -> list[CommunityPost]:
    """
    Apply all post-level filters in sequence:
      1. Engagement threshold — drop low-signal posts
      2. Finance relevance — keep only posts that match at least one
         finance keyword OR pass topic classification

    Returns filtered list. Logs counts for transparency.
    """
    before = len(posts)

    # Step 1: engagement threshold
    posts = [p for p in posts if _passes_engagement_threshold(p, config)]
    after_engagement = len(posts)

    # Step 2: finance relevance
    # A post passes if it matches a finance keyword OR if topic classification
    # would assign it a topic.  This avoids double-filtering against the topic
    # classifier — some posts use niche vocabulary that keywords miss.
    kept: list[CommunityPost] = []
    for p in posts:
        if _is_finance_related(p, config):
            kept.append(p)
            continue
        topic, _ = _classify_post(p.title)
        if topic:
            kept.append(p)
    posts = kept
    after_finance = len(posts)

    logger.info(
        "Filtered posts: %d fetched → %d after engagement (min_score=%d, min_comments=%d) "
        "→ %d after finance relevance",
        before, after_engagement, config.min_score, config.min_comments, after_finance,
    )
    return posts


def fetch_reddit_sentiment(
    config: RedditFilterConfig | None = None,
    llm_callable: "Callable[[str], str] | None" = None,
) -> CommunitySentiment:
    """
    Reddit sentiment pipeline:
      fetch → dedupe → engagement + finance filter → coarse rule-based
      grouping → LLM analysis per cluster → LLM synthesis.

    The LLM is the sole source of interpretation. When llm_callable is
    None the pipeline short-circuits and the section is hidden.
    """
    if config is None:
        config = load_filter_config()

    if llm_callable is None:
        logger.info("Reddit sentiment skipped — LLM not available")
        return CommunitySentiment(platform="reddit")

    raw_posts = fetch_reddit_posts(config.subreddits)
    if not raw_posts:
        logger.warning("No Reddit posts fetched")
        return CommunitySentiment(platform="reddit")

    legacy_posts = filter_posts(raw_posts, config)
    unified = normalize_posts("reddit", legacy_posts)
    unified = dedupe_posts(unified)
    if not unified:
        logger.warning("All posts filtered out")
        return CommunitySentiment(platform="reddit")

    clusters = cluster_posts(unified)
    clusters = mark_rising_clusters(clusters)

    kept, overall = run_llm_pipeline(
        platform="reddit",
        posts=unified,
        clusters=clusters,
        top_n_topics=config.top_n_topics,
        llm_callable=llm_callable,
    )

    return CommunitySentiment(
        platform="reddit",
        trending_topics=kept,
        overall_sentiment=overall,
        post_count=len(unified),
    )


# ─── Standalone test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = load_filter_config()
    print(f"Config: subreddits={config.subreddits}")
    print(f"  min_score={config.min_score}, min_comments={config.min_comments}")
    print(f"  priority_themes={config.priority_themes}")
    print(f"  priority_boost={config.priority_boost}, top_n={config.top_n_topics}")
    print(f"  finance_keywords count={len(config.finance_keywords)}")
    print()

    from llm_adapter import local_llm_callable
    result = fetch_reddit_sentiment(config, llm_callable=local_llm_callable)
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
