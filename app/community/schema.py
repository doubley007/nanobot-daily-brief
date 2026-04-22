"""
Unified cross-platform community data model.

This is the single schema that every community source (Reddit, X, Discord,
future platforms) emits. All downstream stages — dedupe, clustering,
sentiment, credibility, Community Analyst — consume this shape.

Design:
  - UnifiedPost is platform-agnostic. Platform-specific engagement signals
    (upvotes vs likes vs reactions) are normalized into `engagement_raw`
    plus a `platform_specific` blob for traceability.
  - TopicCluster is the unit of analysis. Keyword grouping can produce it,
    embedding clustering will also produce it, so the downstream LLM and
    analyst stages don't need to know how it was assembled.
  - SentimentProfile holds both the coarse label (for backward compat with
    the old formatter) and the richer multi-dimensional breakdown.
  - CredibilityProfile is where signal-vs-noise scoring lives.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ─── Platform-agnostic post ──────────────────────────────────────────────────

@dataclass
class UnifiedPost:
    """
    Normalized cross-platform post. Every source converts to this before
    entering the shared pipeline.
    """
    platform: str                  # "reddit" | "x" | "discord"
    post_id: str                   # platform-native ID
    channel: str                   # subreddit name / channel name / search query
    title: str                     # for X this is the tweet text; for Reddit this is title
    body: str = ""                 # optional longer text (selftext / thread intro)
    url: str = ""
    author: str = ""
    created_utc: float = 0.0
    engagement_raw: int = 0        # upvotes + comments, likes + retweets, reactions
    engagement_breakdown: dict[str, int] = field(default_factory=dict)
    platform_specific: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return f"{self.title} {self.body}".strip()


# ─── Multi-dimensional sentiment ─────────────────────────────────────────────

@dataclass
class SentimentProfile:
    """
    Richer sentiment view. Coarse label kept for legacy rendering.
    Dimensions are 0-1 intensities. `dominant_dimension` names the highest
    one so the formatter can surface the most informative signal without
    needing to rank all five every time.
    """
    label: str = "neutral"              # bullish | bearish | neutral | mixed
    optimism: float = 0.0
    fear: float = 0.0
    uncertainty: float = 0.0
    skepticism: float = 0.0
    hype: float = 0.0
    dominant_dimension: str = ""        # "optimism" | "fear" | ... | ""
    intensity: float = 0.0              # dominant dimension's intensity

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "dimensions": {
                "optimism": self.optimism,
                "fear": self.fear,
                "uncertainty": self.uncertainty,
                "skepticism": self.skepticism,
                "hype": self.hype,
            },
            "dominant": self.dominant_dimension,
            "intensity": self.intensity,
        }


# ─── Trend profile (what is "trending" about this cluster) ───────────────────

@dataclass
class TrendProfile:
    """
    Lightweight trend descriptors. Populated deterministically from the
    current window — no historical store required. When a field is
    unknown we fall back to a conservative value rather than guess.

    trend_direction:     "rising" | "stable" | "fading" | "new"
    persistence:         "new" | "continuing" | "short-lived"
    platform_spread:     "reddit-led" | "discord-led" | "x-led"
                         | "cross-platform" | "single-platform"
    discussion_breadth:  "narrow" | "moderate" | "broad"
    """
    trend_direction: str = "stable"
    persistence: str = "continuing"
    platform_spread: str = "single-platform"
    discussion_breadth: str = "narrow"


# ─── Insurance-view framework (observation, not direct instruction) ──────────

@dataclass
class InsuranceFramework:
    """
    Two-layer insurance readout:
      implications: what this *could* mean for different parts of the book
                    (duration, credit, reinvestment yield, rate-sensitive
                    assets). Deliberately framed as implications, not
                    instructions.
      triggers:     what needs to be true before considering a real action,
                    plus what variable to keep watching.

    Raw `insurance_angle` (legacy free-text) is kept for backward
    compatibility; new code should prefer the structured view.
    """
    implications: str = ""
    triggers: str = ""


# ─── Signal-vs-noise scoring ─────────────────────────────────────────────────

@dataclass
class CredibilityProfile:
    """
    Signals that help decide whether a cluster reflects real market info
    or is just emotional noise. All 0-1.

      specificity: does the discussion reference concrete events, numbers,
                   catalysts? (vs generic "stocks are crashing")
      source_diversity: how many distinct channels/authors contribute?
      engagement_quality: healthy engagement vs a single viral meme post
      llm_judgment: LLM's standalone read of whether this is informational
    """
    specificity: float = 0.0
    source_diversity: float = 0.0
    engagement_quality: float = 0.0
    llm_judgment: float = 0.0
    overall: float = 0.0           # weighted composite, 0-1
    is_noise: bool = False         # convenience threshold flag


# ─── Topic cluster (unit of analysis) ────────────────────────────────────────

@dataclass
class TopicCluster:
    """
    A group of posts about the same real-world topic.
    Carries both the raw signal (posts + stats) and the LLM interpretation.
    """
    # Identity & raw
    cluster_id: str
    posts: list[UnifiedPost] = field(default_factory=list)
    platforms: list[str] = field(default_factory=list)
    rule_label: str = ""                # coarse bucket label (keyword-based)
    heat_score: float = 0.0             # engagement-weighted
    is_rising: bool = False             # engagement spike vs baseline
    rise_ratio: float = 1.0             # current / expected engagement

    # LLM-derived interpretation
    headline: str = ""                   # concrete event-driven label
    discussion_focus: str = ""
    reasons: str = ""
    market_relevance: str = ""           # general market angle
    insurance_angle: str = ""            # legacy free-text; kept for back-compat
    insurance_framework: InsuranceFramework = field(default_factory=InsuranceFramework)
    sentiment: SentimentProfile = field(default_factory=SentimentProfile)
    trend: TrendProfile = field(default_factory=TrendProfile)
    credibility: CredibilityProfile = field(default_factory=CredibilityProfile)
    should_include_in_brief: bool = True

    # Sampling for prompts / debug
    sample_titles: list[str] = field(default_factory=list)

    @property
    def post_count(self) -> int:
        return len(self.posts)


# ─── Community Analyst output ────────────────────────────────────────────────

@dataclass
class CommunityAnalystReport:
    """
    The final structured output of the Community Analyst stage.
    Consumed directly by the daily brief formatter.
    """
    headline_topics: list[TopicCluster] = field(default_factory=list)
    noise_topics: list[TopicCluster] = field(default_factory=list)
    sentiment_structure: str = ""        # one-paragraph read of emotional mix
    cross_platform_signal: str = ""      # what multiple platforms agree on
    insurance_angle: str = ""            # legacy free-text; prefer insurance_framework
    insurance_framework: InsuranceFramework = field(default_factory=InsuranceFramework)
    brief_recommendation: str = ""       # what should go into today's brief
    news_social_bridge: str = ""         # 1-3 sentence narrative bridge
    platforms_covered: list[str] = field(default_factory=list)
    total_posts: int = 0
    total_clusters: int = 0              # how many clusters fed into the analyst
    # Per-platform status strings (e.g. "reddit=ok 59贴", "x=plan limit 402")
    platform_status: list[str] = field(default_factory=list)


# ─── Legacy compatibility shim ───────────────────────────────────────────────
#
# Keep the old CommunitySentiment name alive for code that still imports it.
# It now wraps the analyst report + the clusters it was built from.

@dataclass
class CommunitySentiment:
    platform: str = ""
    trending_topics: list[TopicCluster] = field(default_factory=list)
    overall_sentiment: str = "neutral"
    post_count: int = 0
    narrative: str = ""
    summary: str = ""

    # Bridge to analyst report when this came from the unified pipeline
    analyst_report: CommunityAnalystReport | None = None
