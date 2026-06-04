"""
统一的社区数据源适配层。

历史上 app/community/{reddit,x,discord,stocktwits}_source.py 是按平台写的
抓取器，输出的是 CommunityPost + CommunitySentiment —— 面向日报流程。

sources/ 包在此之上做了一层更通用的 Source Adapter：
  - 统一输出 UnifiedPost[]，喂给 assistant.rag.community_indexer 直接写库
  - 每个平台一个 adapter 类，实现 fetch_posts()
  - 新增 telegram_source（community/ 里没有），作为 bot 数据源

这层主要给 assistant/bot 使用；daily_job 仍然复用老的 community/ 路径，
不做破坏性改动。
"""
from __future__ import annotations

from sources.base import BaseSourceAdapter
from sources.reddit_adapter import RedditSourceAdapter
from sources.discord_adapter import DiscordSourceAdapter
from sources.twitter_adapter import TwitterSourceAdapter
from sources.telegram_adapter import TelegramSourceAdapter

__all__ = [
    "BaseSourceAdapter",
    "RedditSourceAdapter",
    "DiscordSourceAdapter",
    "TwitterSourceAdapter",
    "TelegramSourceAdapter",
    "all_enabled_adapters",
]


def all_enabled_adapters() -> list[BaseSourceAdapter]:
    """返回所有配置可用的 adapter，env 里缺凭证的自动跳过。"""
    adapters: list[BaseSourceAdapter] = []
    for cls in (
        RedditSourceAdapter,
        DiscordSourceAdapter,
        TwitterSourceAdapter,
        TelegramSourceAdapter,
    ):
        a = cls()
        if a.is_configured():
            adapters.append(a)
    return adapters
