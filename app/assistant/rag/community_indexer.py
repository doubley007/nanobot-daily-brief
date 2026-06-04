"""
把 UnifiedPost（community/schema.py）写进 Knowledge Store 的 community 表。

职责：
  1. 给每条帖子打 asset_tags（和新闻用同一张 asset_taxonomy）
  2. 给每条帖子打 bullish_bearish_label（粗分类，供后续 bot 展示）
  3. 给每条帖子打投资场景 emotion_label（真正驱动决策的那一套）

投资场景标签在 sentiment_aggregator.classify_emotion_label() 里定义。
这里只做单条标注；跨条聚合在 sentiment_aggregator.aggregate() 里。
"""
from __future__ import annotations

import hashlib
import logging
from typing import Iterable

from assistant.asset_taxonomy import ASSETS
from assistant.rag.store import CommunityDoc, KnowledgeStore, default_store
from assistant.sentiment_labels import (
    classify_bull_bear,
    classify_emotion_label,
)

logger = logging.getLogger(__name__)


def _hash_id(platform: str, post_id: str, url: str) -> str:
    key = f"{platform}::{post_id}::{url}".encode("utf-8")
    return hashlib.sha1(key).hexdigest()[:20]


def detect_asset_tags(text: str) -> list[str]:
    tags: list[str] = []
    lower = (text or "").lower()
    for spec in ASSETS:
        if any(t.lower() in lower for t in spec.aliases + spec.keywords):
            tags.append(spec.id)
    return tags


def unified_posts_to_docs(posts: Iterable) -> list[CommunityDoc]:
    """
    接受 community.schema.UnifiedPost 或任何含 platform/channel/title/
    body/url/created_utc/engagement_raw/author 字段的对象。
    """
    docs: list[CommunityDoc] = []
    for p in posts:
        platform = getattr(p, "platform", "") or "unknown"
        post_id = getattr(p, "post_id", "") or ""
        url = getattr(p, "url", "") or ""
        title = getattr(p, "title", "") or ""
        body = getattr(p, "body", "") or ""
        channel = getattr(p, "channel", "") or ""
        author = getattr(p, "author", "") or ""
        created = float(getattr(p, "created_utc", 0.0) or 0.0)
        engagement = float(getattr(p, "engagement_raw", 0.0) or 0.0)

        text = f"{title}\n{body}".strip()
        asset_tags = detect_asset_tags(text)
        bull_bear, bb_conf = classify_bull_bear(text)
        emotion, em_conf = classify_emotion_label(text)
        confidence = round(max(bb_conf, em_conf), 2)

        docs.append(CommunityDoc(
            id=_hash_id(platform, post_id, url),
            platform=platform,
            channel_or_group=channel,
            author=author,
            published_at=created,
            raw_text=text,
            normalized_text=text.lower(),
            summary=title[:200],
            asset_tags=asset_tags,
            bullish_bearish_label=bull_bear,
            emotion_label=emotion,
            confidence=confidence,
            engagement_score=engagement,
            url=url,
        ))
    return docs


def index_community(
    posts: Iterable,
    store: KnowledgeStore | None = None,
) -> int:
    store = store or default_store()
    docs = unified_posts_to_docs(posts)
    n = store.upsert_community(docs)
    logger.info("community_indexer: wrote %d docs", n)
    return n
