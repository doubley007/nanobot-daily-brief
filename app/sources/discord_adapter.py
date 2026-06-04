"""
Discord 适配器 —— 复用 community.discord_source 的拉取逻辑。
"""
from __future__ import annotations

import os

from community.discord_source import fetch_discord_posts, load_filter_config
from community.normalize import normalize_posts
from sources.base import BaseSourceAdapter, FetchReport


class DiscordSourceAdapter(BaseSourceAdapter):
    platform = "discord"

    def is_configured(self) -> bool:
        return bool(
            os.getenv("DISCORD_BOT_TOKEN", "").strip()
            and os.getenv("DISCORD_CHANNEL_IDS", "").strip()
        )

    def fetch(self) -> FetchReport:
        if not self.is_configured():
            return FetchReport(platform=self.platform, ok=False, posts=[],
                               error="missing DISCORD_* env vars")
        config = load_filter_config()
        posts = fetch_discord_posts(config)
        unified = normalize_posts("discord", posts)
        return FetchReport(platform=self.platform, ok=True, posts=unified)
