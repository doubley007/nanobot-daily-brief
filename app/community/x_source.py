"""
X (Twitter) community sentiment — curated KOL timeline mode.

Context:
  The `/2/tweets/search/recent` endpoint requires at least the Basic plan
  ($200/month) and returns HTTP 402 on Free. Rather than gate the whole
  X pipeline on a paid plan, we pull tweets from a curated set of finance
  KOL user timelines via `/2/users/:id/tweets`, which is reachable on
  lower tiers (subject to tight rate limits).

Design:
  - A small hand-picked list of handles (overridable via X_KOL_HANDLES).
  - Each handle is resolved to a numeric user_id once and cached on disk
    under ~/.cache/nanobot/x_user_ids.json. Free-tier user-lookup quota
    is extremely low (~25/24h), so aggressive caching is mandatory.
  - Each run pulls the 10 most recent tweets per handle.
  - Engagement filter is lighter than search-mode, because a KOL with
    10 likes is still a curated signal — noise is already filtered by
    the handle list itself.
  - HTTP 402 still raises XPlanUnavailableError (timeline endpoint might
    also require Basic on some accounts); HTTP 429 is treated as a soft
    skip — we've seen partial data is better than no data.

Requires X_BEARER_TOKEN. When unset, returns an empty sentiment.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
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
MAX_TWEETS_PER_HANDLE = 10           # keep calls cheap; KOLs don't post that much
USER_ID_CACHE_PATH = Path.home() / ".cache" / "nanobot" / "x_user_ids.json"


class XPlanUnavailableError(Exception):
    """
    Raised when the X API returns 402. Two common shapes:
      - title="CreditsDepleted" — account credits are 0 (paid plan, empty wallet)
      - generic plan-tier block  — endpoint not on current subscription tier
    The `reason` attribute carries the title (when parsable) so callers can
    surface it in user-facing status lines.
    """
    def __init__(self, message: str, reason: str = ""):
        super().__init__(message)
        self.reason = reason


def _parse_402_reason(response: requests.Response) -> str:
    """Extract a short cause string (e.g. 'CreditsDepleted') from a 402 body."""
    try:
        body = response.json()
    except ValueError:
        return ""
    title = body.get("title") or ""
    return title.strip()


# Curated finance-focused handles. These were picked because they post
# actionable market information most days and their follower bases span
# rates, equities, and macro. Override with X_KOL_HANDLES env var.
DEFAULT_KOL_HANDLES = [
    "nicktimiraos",       # WSJ Fed reporter — rates / FOMC
    "unusual_whales",     # options flow
    "zerohedge",          # fast-moving macro
    "DiMartinoBooth",     # Fed / macro commentary
    "LizAnnSonders",      # Schwab chief strategist — equities / macro
    "biancoresearch",     # rates / yield curve
    "TheTranscript_",     # earnings call highlights
    "BrianTheFOMC",       # curated FOMC commentary
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
    X pipeline configuration. Loaded from env vars with sensible defaults.

    Env vars:
      X_BEARER_TOKEN       Twitter API v2 bearer token (required)
      X_KOL_HANDLES        comma-separated handles without @ (overrides default list)
      X_MIN_LIKES          minimum like count (default 5 — KOL mode is lenient)
      X_MIN_RETWEETS       minimum retweet count (default 1)
      X_FINANCE_KEYWORDS   comma-separated extra keywords (merged with defaults)
      X_PRIORITY_THEMES    comma-separated topic labels in Chinese
      X_PRIORITY_BOOST     float multiplier for priority themes (default 1.5)
      X_TOP_N_TOPICS       max topics to return (default 4)
    """
    bearer_token: str = ""
    kol_handles: list[str] = field(default_factory=lambda: list(DEFAULT_KOL_HANDLES))
    min_likes: int = 5
    min_retweets: int = 1
    finance_keywords: list[str] = field(default_factory=lambda: list(DEFAULT_FINANCE_KEYWORDS))
    priority_themes: list[str] = field(default_factory=lambda: list(DEFAULT_PRIORITY_THEMES))
    priority_boost: float = PRIORITY_BOOST_FACTOR
    top_n_topics: int = 4


def load_filter_config() -> XFilterConfig:
    """Build config from env vars, falling back to defaults."""
    config = XFilterConfig()

    config.bearer_token = os.getenv("X_BEARER_TOKEN", "").strip()

    env_handles = os.getenv("X_KOL_HANDLES", "").strip()
    if env_handles:
        config.kol_handles = [
            h.strip().lstrip("@") for h in env_handles.split(",") if h.strip()
        ]

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


# ─── User-id cache (Free tier user lookup has ~25 calls/24h — cache hard) ────

def _load_user_id_cache() -> dict[str, str]:
    if not USER_ID_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(USER_ID_CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_user_id_cache(cache: dict[str, str]) -> None:
    try:
        USER_ID_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        USER_ID_CACHE_PATH.write_text(json.dumps(cache, indent=2))
    except OSError as e:
        logger.warning("Failed to persist X user-id cache: %s", e)


def _resolve_user_id(handle: str, bearer_token: str, cache: dict[str, str]) -> str | None:
    """
    Return the numeric user_id for a handle. Caches resolved ids on disk so
    we only hit `/2/users/by/username` once per handle per machine.
    """
    handle_key = handle.lower()
    if handle_key in cache:
        return cache[handle_key]

    url = f"https://api.twitter.com/2/users/by/username/{handle}"
    try:
        r = requests.get(
            url,
            headers={"Authorization": f"Bearer {bearer_token}"},
            timeout=REQUEST_TIMEOUT,
        )
    except Exception as e:
        logger.warning("X user lookup for @%s failed: %s", handle, e)
        return None

    if r.status_code == 402:
        reason = _parse_402_reason(r)
        raise XPlanUnavailableError(
            f"X user lookup unavailable (HTTP 402, {reason or 'plan'})",
            reason=reason,
        )
    if r.status_code == 429:
        logger.info("X user lookup rate limited (429) for @%s — skipping", handle)
        return None
    if r.status_code == 404:
        logger.info("X handle @%s not found (404)", handle)
        return None
    if r.status_code != 200:
        logger.warning("X user lookup for @%s returned %d", handle, r.status_code)
        return None

    user_id = r.json().get("data", {}).get("id")
    if user_id:
        cache[handle_key] = user_id
        _save_user_id_cache(cache)
    return user_id


# ─── Timeline fetch ──────────────────────────────────────────────────────────

def _fetch_user_timeline(
    handle: str,
    user_id: str,
    bearer_token: str,
    max_results: int = MAX_TWEETS_PER_HANDLE,
) -> list[dict[str, Any]]:
    """
    Fetch recent tweets from one user's timeline.
    Excludes retweets and replies — we want the KOL's own posts.
    """
    url = f"https://api.twitter.com/2/users/{user_id}/tweets"
    params = {
        "max_results": max(5, min(max_results, 100)),  # API min is 5
        "exclude": "retweets,replies",
        "tweet.fields": "public_metrics,created_at,author_id",
    }
    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {bearer_token}"},
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    if r.status_code == 402:
        reason = _parse_402_reason(r)
        raise XPlanUnavailableError(
            f"X user timeline unavailable (HTTP 402, {reason or 'plan'})",
            reason=reason,
        )
    if r.status_code == 429:
        logger.info("X timeline rate limited (429) for @%s — skipping", handle)
        return []
    if r.status_code == 401:
        logger.warning("X timeline auth failed (401) — check X_BEARER_TOKEN")
        return []
    if r.status_code != 200:
        logger.warning("X timeline for @%s returned %d", handle, r.status_code)
        return []

    tweets = r.json().get("data", []) or []
    # Tag each tweet with the handle so we can populate channel downstream
    for t in tweets:
        t["_handle"] = handle
    return tweets


def _parse_tweets(raw_tweets: list[dict[str, Any]]) -> list[CommunityPost]:
    """
    Convert raw API tweet dicts into CommunityPost objects.
    The `subreddit` field is repurposed as `@handle` so the normalizer
    can surface it as the cluster channel.
    """
    posts: list[CommunityPost] = []
    for t in raw_tweets:
        metrics = t.get("public_metrics", {})
        handle = t.get("_handle", "")
        created_at = t.get("created_at", "")
        # Parse ISO8601 loosely — fall back to 0 on any issue
        created_utc = 0.0
        if created_at:
            try:
                import datetime as _dt
                created_utc = _dt.datetime.fromisoformat(
                    created_at.replace("Z", "+00:00")
                ).timestamp()
            except (ValueError, TypeError):
                created_utc = 0.0

        posts.append(CommunityPost(
            title=t.get("text", ""),
            score=metrics.get("like_count", 0),
            num_comments=metrics.get("reply_count", 0),
            url=f"https://x.com/{handle}/status/{t.get('id', '')}",
            subreddit=f"@{handle}" if handle else "x",
            created_utc=created_utc,
        ))
    return posts


def fetch_x_posts(config: XFilterConfig) -> list[CommunityPost]:
    """
    Walk each configured handle, resolve user_id (cached), pull the
    recent timeline, and return a merged CommunityPost list.
    """
    user_cache = _load_user_id_cache()
    all_posts: list[CommunityPost] = []
    seen_ids: set[str] = set()

    handles_processed = 0
    for handle in config.kol_handles:
        try:
            user_id = _resolve_user_id(handle, config.bearer_token, user_cache)
        except XPlanUnavailableError as e:
            # Hard stop: if user lookup isn't available on this plan, timeline won't be either.
            logger.info("%s — aborting X fetch", e)
            raise

        if not user_id:
            continue

        try:
            raw = _fetch_user_timeline(handle, user_id, config.bearer_token)
        except XPlanUnavailableError as e:
            logger.info("%s — aborting X fetch after @%s", e, handle)
            raise
        except Exception as e:
            logger.warning("Failed to fetch timeline for @%s: %s", handle, e)
            continue

        new_before = len(all_posts)
        for t in raw:
            tid = t.get("id", "")
            if not tid or tid in seen_ids:
                continue
            seen_ids.add(tid)
            all_posts.extend(_parse_tweets([t]))
        handles_processed += 1
        logger.info(
            "Fetched %d tweets from @%s (total now %d)",
            len(all_posts) - new_before, handle, len(all_posts),
        )

        # Free-tier is aggressively rate-limited. A short pause between
        # handles avoids getting the whole batch 429'd.
        time.sleep(0.5)

    logger.info(
        "X timeline pass: %d/%d handles produced data, %d tweets total",
        handles_processed, len(config.kol_handles), len(all_posts),
    )
    return all_posts


# ─── Post-level filtering ────────────────────────────────────────────────────

def _passes_engagement_threshold(post: CommunityPost, config: XFilterConfig) -> bool:
    # KOL timeline is already curated — OR-match keeps low-engagement but
    # substantive posts (e.g. a 3-like Nick Timiraos Fed scoop).
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
    print(f"  kol_handles={config.kol_handles}")
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
