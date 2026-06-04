"""
Reddit 适配器 —— 薄包装 community.reddit_source.fetch_reddit_posts。
"""
from __future__ import annotations

from community.normalize import normalize_posts
from community.reddit_source import fetch_reddit_posts, load_filter_config
from sources.base import BaseSourceAdapter, FetchReport


class RedditSourceAdapter(BaseSourceAdapter):
    platform = "reddit"

    def is_configured(self) -> bool:
        # 公共 JSON API 不需要 key；任何时候都算 configured
        return True

    def fetch(self) -> FetchReport:
        config = load_filter_config()
        posts = fetch_reddit_posts(config.subreddits)
        unified = normalize_posts("reddit", posts)
        return FetchReport(platform=self.platform, ok=True, posts=unified)
