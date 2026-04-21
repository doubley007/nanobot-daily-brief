"""
Discord community sentiment analysis for financial daily brief.

Uses the Discord Bot API to read recent messages from configured channels,
then applies the shared topic classification and sentiment analysis pipeline.

Requires DISCORD_BOT_TOKEN env var.  When the token is not set, gracefully
returns an empty result so the brief pipeline continues without Discord data.

Filtering is controlled via env vars (see DiscordFilterConfig).
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
MESSAGES_PER_CHANNEL = 100  # Discord API max per request

DEFAULT_PRIORITY_THEMES = [
    "美联储与利率政策",
    "美债与收益率",
    "通胀与物价",
    "衰退与宏观经济",
]

PRIORITY_BOOST_FACTOR = 1.5


@dataclass
class DiscordFilterConfig:
    """
    All filtering knobs in one place. Loaded from env vars with sensible defaults.

    Env vars:
      DISCORD_BOT_TOKEN          Discord bot token (required)
      DISCORD_CHANNEL_IDS        comma-separated channel IDs to monitor
      DISCORD_MIN_REACTIONS      minimum reaction count to keep a message (default 3)
      DISCORD_FINANCE_KEYWORDS   comma-separated extra keywords (merged with defaults)
      DISCORD_PRIORITY_THEMES    comma-separated topic labels in Chinese
      DISCORD_PRIORITY_BOOST     float multiplier for priority themes (default 1.5)
      DISCORD_TOP_N_TOPICS       max topics to return (default 4)
    """
    bot_token: str = ""
    channel_ids: list[str] = field(default_factory=list)
    min_reactions: int = 3
    finance_keywords: list[str] = field(default_factory=lambda: list(DEFAULT_FINANCE_KEYWORDS))
    priority_themes: list[str] = field(default_factory=lambda: list(DEFAULT_PRIORITY_THEMES))
    priority_boost: float = PRIORITY_BOOST_FACTOR
    top_n_topics: int = 4


def load_filter_config() -> DiscordFilterConfig:
    """Build config from env vars, falling back to defaults."""
    config = DiscordFilterConfig()

    config.bot_token = os.getenv("DISCORD_BOT_TOKEN", "").strip()

    env_channels = os.getenv("DISCORD_CHANNEL_IDS", "").strip()
    if env_channels:
        config.channel_ids = [c.strip() for c in env_channels.split(",") if c.strip()]

    env_reactions = os.getenv("DISCORD_MIN_REACTIONS", "").strip()
    if env_reactions:
        config.min_reactions = int(env_reactions)

    env_keywords = os.getenv("DISCORD_FINANCE_KEYWORDS", "").strip()
    if env_keywords:
        extra = [k.strip().lower() for k in env_keywords.split(",") if k.strip()]
        config.finance_keywords = list(set(config.finance_keywords + extra))

    env_priority = os.getenv("DISCORD_PRIORITY_THEMES", "").strip()
    if env_priority:
        config.priority_themes = [t.strip() for t in env_priority.split(",") if t.strip()]

    env_boost = os.getenv("DISCORD_PRIORITY_BOOST", "").strip()
    if env_boost:
        config.priority_boost = float(env_boost)

    env_topn = os.getenv("DISCORD_TOP_N_TOPICS", "").strip()
    if env_topn:
        config.top_n_topics = int(env_topn)

    return config


# ─── Fetch messages ──────────────────────────────────────────────────────────

def _fetch_channel_messages(
    channel_id: str,
    bot_token: str,
    limit: int = MESSAGES_PER_CHANNEL,
) -> list[dict[str, Any]]:
    """Fetch recent messages from a Discord channel."""
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    headers = {"Authorization": f"Bot {bot_token}"}
    params = {"limit": min(limit, 100)}

    response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    return response.json()


def _count_reactions(message: dict[str, Any]) -> int:
    """Sum all reaction counts on a message."""
    reactions = message.get("reactions", [])
    return sum(r.get("count", 0) for r in reactions)


def _parse_messages(raw_messages: list[dict[str, Any]], channel_id: str) -> list[CommunityPost]:
    """Convert raw Discord messages into CommunityPost objects."""
    posts: list[CommunityPost] = []
    for msg in raw_messages:
        # Skip bot messages
        author = msg.get("author", {})
        if author.get("bot", False):
            continue

        content = msg.get("content", "").strip()
        if not content:
            continue

        reaction_count = _count_reactions(msg)

        posts.append(CommunityPost(
            title=content,
            score=reaction_count,
            num_comments=0,  # Discord doesn't have threaded replies in the same way
            url=f"https://discord.com/channels/_/{channel_id}/{msg.get('id', '')}",
            subreddit="discord",  # reuse field to mark source platform
            created_utc=0.0,
        ))
    return posts


def fetch_discord_posts(config: DiscordFilterConfig) -> list[CommunityPost]:
    """Fetch messages from all configured channels."""
    all_posts: list[CommunityPost] = []

    for channel_id in config.channel_ids:
        try:
            raw = _fetch_channel_messages(channel_id, config.bot_token)
            posts = _parse_messages(raw, channel_id)
            all_posts.extend(posts)
            logger.info("Fetched %d messages from channel %s", len(posts), channel_id)
        except Exception as e:
            logger.warning("Failed to fetch channel %s: %s", channel_id, e)

    return all_posts


# ─── Post-level filtering ────────────────────────────────────────────────────

def _passes_engagement_threshold(post: CommunityPost, config: DiscordFilterConfig) -> bool:
    return post.score >= config.min_reactions


def _is_finance_related(post: CommunityPost, config: DiscordFilterConfig) -> bool:
    text = post.title.lower()
    return any(kw in text for kw in config.finance_keywords)


def filter_posts(posts: list[CommunityPost], config: DiscordFilterConfig) -> list[CommunityPost]:
    """
    Apply post-level filters:
      1. Engagement threshold (reaction count)
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
        "Discord filtered: %d fetched → %d after engagement (min_reactions=%d) "
        "→ %d after finance relevance",
        before, after_engagement, config.min_reactions, after_finance,
    )
    return posts


# ─── Main entry point ────────────────────────────────────────────────────────

def fetch_discord_sentiment(
    config: DiscordFilterConfig | None = None,
    llm_callable: "Callable[[str], str] | None" = None,
) -> CommunitySentiment:
    """Discord sentiment pipeline — fully LLM-interpreted."""
    if config is None:
        config = load_filter_config()

    if not config.bot_token:
        logger.info("DISCORD_BOT_TOKEN not set — skipping Discord sentiment")
        return CommunitySentiment(platform="discord")

    if not config.channel_ids:
        logger.info("DISCORD_CHANNEL_IDS not set — skipping Discord sentiment")
        return CommunitySentiment(platform="discord")

    if llm_callable is None:
        logger.info("Discord sentiment skipped — LLM not available")
        return CommunitySentiment(platform="discord")

    raw_posts = fetch_discord_posts(config)
    if not raw_posts:
        logger.warning("No messages fetched from Discord")
        return CommunitySentiment(platform="discord")

    legacy_posts = filter_posts(raw_posts, config)
    unified = normalize_posts("discord", legacy_posts)
    unified = dedupe_posts(unified)
    if not unified:
        logger.warning("All Discord messages filtered out")
        return CommunitySentiment(platform="discord")

    clusters = cluster_posts(unified)
    clusters = mark_rising_clusters(clusters)

    kept, overall = run_llm_pipeline(
        platform="discord",
        posts=unified,
        clusters=clusters,
        top_n_topics=config.top_n_topics,
        llm_callable=llm_callable,
    )

    return CommunitySentiment(
        platform="discord",
        trending_topics=kept,
        overall_sentiment=overall,
        post_count=len(unified),
    )


# ─── Standalone test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = load_filter_config()
    print(f"Config: bot_token={'set' if config.bot_token else 'NOT SET'}")
    print(f"  channel_ids={config.channel_ids}")
    print(f"  min_reactions={config.min_reactions}")
    print(f"  priority_themes={config.priority_themes}")
    print(f"  priority_boost={config.priority_boost}, top_n={config.top_n_topics}")
    print()

    from llm_adapter import local_llm_callable
    result = fetch_discord_sentiment(config, llm_callable=local_llm_callable)
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
