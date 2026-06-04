"""
Twitter / X 适配器。

community.x_source 的管线严重依赖 X 付费 plan + LLM 聚类，对 bot 的原始抓取
来说太重。这里只用 x_source 的底层 fetch_x_posts + filter_posts，得到
UnifiedPost 直接丢给 RAG。聚类/打标全部交给 community_indexer + aggregator。

如果 X_BEARER_TOKEN 未配置或付费 plan 受限，safe_fetch() 会优雅失败。
"""
from __future__ import annotations

import logging
import os

from community.normalize import normalize_posts
from community.x_source import (
    XPlanUnavailableError,
    fetch_x_posts,
    filter_posts,
    load_filter_config,
)
from sources.base import BaseSourceAdapter, FetchReport

logger = logging.getLogger(__name__)


class TwitterSourceAdapter(BaseSourceAdapter):
    platform = "x"

    def is_configured(self) -> bool:
        return bool(os.getenv("X_BEARER_TOKEN", "").strip())

    def fetch(self) -> FetchReport:
        if not self.is_configured():
            return FetchReport(platform=self.platform, ok=False, posts=[],
                               error="X_BEARER_TOKEN not set")
        config = load_filter_config()
        try:
            raw = fetch_x_posts(config)
        except XPlanUnavailableError as e:
            return FetchReport(platform=self.platform, ok=False, posts=[],
                               error=f"x plan blocked: {e}")

        legacy = filter_posts(raw, config)
        unified = normalize_posts("x", legacy)
        return FetchReport(platform=self.platform, ok=True, posts=unified)
