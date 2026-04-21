"""
Platform → UnifiedPost normalization.

Each community source already produces CommunityPost objects. This module
lifts them into the unified schema once, at the boundary of the new
pipeline, without changing fetcher code.
"""
from __future__ import annotations

from community.base import CommunityPost
from community.schema import UnifiedPost


def reddit_to_unified(p: CommunityPost) -> UnifiedPost:
    return UnifiedPost(
        platform="reddit",
        post_id=p.url.rsplit("/", 2)[-2] if "/" in p.url else "",
        channel=p.subreddit,
        title=p.title,
        url=p.url,
        created_utc=p.created_utc,
        engagement_raw=p.score + p.num_comments * 2,
        engagement_breakdown={"upvotes": p.score, "comments": p.num_comments},
        platform_specific={},
    )


def x_to_unified(p: CommunityPost) -> UnifiedPost:
    # x_source reuses CommunityPost with subreddit="x"
    return UnifiedPost(
        platform="x",
        post_id=p.url.rsplit("/", 1)[-1] if "/" in p.url else "",
        channel="recent_search",
        title=p.title,
        url=p.url,
        created_utc=p.created_utc,
        engagement_raw=p.score + p.num_comments * 3,
        engagement_breakdown={"likes": p.score, "replies": p.num_comments},
        platform_specific={},
    )


def discord_to_unified(p: CommunityPost) -> UnifiedPost:
    return UnifiedPost(
        platform="discord",
        post_id=p.url.rsplit("/", 1)[-1] if "/" in p.url else "",
        channel=p.subreddit,  # discord_source reuses subreddit field for channel_id
        title=p.title,
        url=p.url,
        created_utc=p.created_utc,
        engagement_raw=p.score,
        engagement_breakdown={"reactions": p.score},
        platform_specific={},
    )


_NORMALIZERS = {
    "reddit": reddit_to_unified,
    "x": x_to_unified,
    "discord": discord_to_unified,
}


def normalize_posts(platform: str, posts: list[CommunityPost]) -> list[UnifiedPost]:
    fn = _NORMALIZERS.get(platform)
    if fn is None:
        raise ValueError(f"Unknown platform for normalization: {platform}")
    return [fn(p) for p in posts]
