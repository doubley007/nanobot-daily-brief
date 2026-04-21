"""
News-to-community linker.

Bridges the news pipeline's topic classification (news_enricher.detect_news_type)
with the community pipeline's topic clusters (reddit_source TopicCluster labels)
to produce per-news-item community reactions.

Design:
  - The two systems use different label spaces. This module defines an explicit
    mapping table so changes to either side don't silently break the link.
  - Matching is keyword-overlap-based: each news item's text is scored against
    each community cluster's keyword set. This handles cases where the news
    item doesn't neatly fit one label but still overlaps a community topic.
  - A match must exceed a minimum score threshold to avoid forced linking.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from community.base import TopicCluster


# ─── Data model ──────────────────────────────────────────────────────────────

@dataclass
class LinkedReaction:
    """A community reaction linked to a specific news item."""
    topic_label: str       # community topic label (Chinese)
    sentiment: str         # "bullish" | "bearish" | "neutral" | "mixed"
    discussion_focus: str  # what the community is debating
    post_count: int
    confidence: float      # 0-1, how strong the match is


# ─── Keyword bridge ──────────────────────────────────────────────────────────
#
# Each community topic has a set of keywords that can appear in news text.
# These are deliberately broader than the reddit_source TOPIC_RULES because
# news headlines use different vocabulary than Reddit post titles.

COMMUNITY_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "美联储与利率政策": [
        "fed", "federal reserve", "fomc", "rate cut", "rate hike",
        "powell", "monetary policy", "interest rate", "central bank",
        "dovish", "hawkish", "rates",
    ],
    "通胀与物价": [
        "inflation", "cpi", "pce", "consumer price", "deflation",
        "stagflation", "price pressure",
    ],
    "美债与收益率": [
        "treasury", "yield", "yields", "bond", "bonds", "10y",
        "10-year", "yield curve", "bond market",
    ],
    "企业财报与盈利": [
        "earnings", "guidance", "revenue", "profit", "forecast",
        "eps", "quarterly report", "earnings beat", "earnings miss",
    ],
    "衰退与宏观经济": [
        "recession", "gdp", "unemployment", "payroll", "jobs report",
        "economic slowdown", "labor market", "employment",
    ],
    "科技与半导体": [
        "semiconductor", "chip", "chips", "nvidia", "nvda", "tsmc",
        "ai chip", "gpu", "artificial intelligence",
    ],
    "关税与贸易": [
        "tariff", "trade war", "trade deal", "sanctions", "import duty",
        "export ban", "trade deficit",
    ],
    "原油与能源": [
        "oil", "crude", "opec", "brent", "wti", "energy",
        "natural gas", "strait of hormuz",
    ],
    "中国经济": [
        "china", "pboc", "renminbi", "yuan", "chinese market",
    ],
    "房地产与信贷": [
        "real estate", "housing", "mortgage", "commercial real estate",
        "property", "home price", "foreclosure",
    ],
    "加密货币": [
        "bitcoin", "btc", "ethereum", "crypto", "cryptocurrency",
    ],
}

# Minimum keyword overlap score to accept a link.
MIN_LINK_SCORE = 2


# ─── Matching ────────────────────────────────────────────────────────────────

def _score_news_against_topic(news_text: str, topic_label: str) -> int:
    """Count how many bridge keywords from a community topic appear in the news text."""
    keywords = COMMUNITY_TOPIC_KEYWORDS.get(topic_label, [])
    text_lower = news_text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def link_news_to_community(
    news_items: list,
    community_topics: list[TopicCluster],
) -> dict[int, LinkedReaction]:
    """
    For each news item (by index), find the best matching community topic.

    Returns a dict mapping news item index → LinkedReaction.
    Items with no strong match are absent from the dict.
    """
    if not community_topics:
        return {}

    # Build a lookup: rule_label → list[TopicCluster]
    # Multiple clusters may share a rule_label (embedding-mode sub-topics)
    topic_lookup: dict[str, list[TopicCluster]] = {}
    for t in community_topics:
        key = t.rule_label or t.headline
        if not key:
            continue
        topic_lookup.setdefault(key, []).append(t)

    links: dict[int, LinkedReaction] = {}
    used_cluster_ids: set[str] = set()

    # Score every (news_item, topic_bucket) pair
    candidates: list[tuple[int, str, int]] = []  # (news_idx, bucket_key, score)

    for idx, item in enumerate(news_items):
        text = f"{item.title} {item.summary}".lower()
        for bucket_key in topic_lookup:
            score = _score_news_against_topic(text, bucket_key)
            if score >= MIN_LINK_SCORE:
                candidates.append((idx, bucket_key, score))

    # Sort by score descending so the strongest links are assigned first
    candidates.sort(key=lambda x: x[2], reverse=True)

    for news_idx, bucket_key, score in candidates:
        if news_idx in links:
            continue
        # Pick the hottest cluster from this bucket that hasn't been used yet
        cluster = next(
            (c for c in sorted(
                topic_lookup[bucket_key],
                key=lambda c: c.heat_score,
                reverse=True,
            ) if c.cluster_id not in used_cluster_ids),
            None,
        )
        if cluster is None:
            continue

        max_possible = len(COMMUNITY_TOPIC_KEYWORDS.get(bucket_key, []))
        confidence = min(score / max(max_possible, 1), 1.0)

        links[news_idx] = LinkedReaction(
            topic_label=cluster.headline or cluster.rule_label,
            sentiment=cluster.sentiment.label,
            discussion_focus=cluster.discussion_focus,
            post_count=cluster.post_count,
            confidence=confidence,
        )
        used_cluster_ids.add(cluster.cluster_id)

    return links


def get_unlinked_topics(
    community_topics: list[TopicCluster],
    linked_reactions: dict[int, LinkedReaction],
) -> list[TopicCluster]:
    """Return community topics that were NOT linked to any news item."""
    linked_headlines = {r.topic_label for r in linked_reactions.values()}
    return [
        t for t in community_topics
        if (t.headline or t.rule_label) not in linked_headlines
    ]
