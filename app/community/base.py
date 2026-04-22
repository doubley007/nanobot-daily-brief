"""
Backward-compatibility shim. The canonical data model lives in
community.schema. This module re-exports the unified types plus the
legacy CommunityPost so existing imports keep working.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from community.schema import (
    CommunityAnalystReport,
    CommunitySentiment,
    CredibilityProfile,
    InsuranceFramework,
    SentimentProfile,
    TopicCluster,
    TrendProfile,
    UnifiedPost,
)


@dataclass
class CommunityPost:
    """Legacy per-platform post. New code should use UnifiedPost."""
    title: str
    score: int
    num_comments: int
    url: str
    subreddit: str = ""
    created_utc: float = 0.0


class CommunitySource(Protocol):
    """Protocol for future community sources (X, Discord, etc.)."""

    def fetch_sentiment(self) -> CommunitySentiment: ...


__all__ = [
    "CommunityPost",
    "CommunitySource",
    "CommunitySentiment",
    "TopicCluster",
    "UnifiedPost",
    "SentimentProfile",
    "CredibilityProfile",
    "InsuranceFramework",
    "TrendProfile",
    "CommunityAnalystReport",
]
