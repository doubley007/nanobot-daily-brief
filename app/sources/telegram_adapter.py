"""
Telegram 群/频道适配器。

Telegram Bot API 不支持读取历史消息 —— bot 只能收到入群之后的实时推送。
为了让这套 bot 的 RAG 真的能消费 TG 群的内容，本 adapter 支持两种模式：

模式 A（推荐，零新依赖）：TG_FEED_FILE
    指向一个 JSONL 文件。每行一个 JSON：
      {"id": "...", "channel": "...", "author": "...", "text": "...",
       "created_utc": 1712345678, "url": "...", "engagement": 12}
    外部抓取工具（Telethon 脚本 / 手动整理 / Zapier 等）只要把群里的消息
    追加到这个文件就行。Adapter 只负责读取和解析。

模式 B（可选，需要 pip install telethon）：TG_API_ID + TG_API_HASH
    如果装了 telethon 且配了 API ID/HASH，会走 MTProto 直接从指定群拉消息。
    这条路径会在 fetch() 里被优先尝试；失败就回落到模式 A。

env:
    TG_FEED_FILE      JSONL 文件路径
    TG_CHANNELS       逗号分隔的频道/群 username 或 chat_id（模式 B 使用）
    TG_API_ID         Telethon API ID（模式 B 使用）
    TG_API_HASH       Telethon API HASH（模式 B 使用）
    TG_FETCH_LIMIT    每个群拉多少条，默认 50（模式 B 使用）
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from community.schema import UnifiedPost
from sources.base import BaseSourceAdapter, FetchReport

logger = logging.getLogger(__name__)


class TelegramSourceAdapter(BaseSourceAdapter):
    platform = "telegram"

    def is_configured(self) -> bool:
        # 任一模式配齐就算可用
        has_feed = bool(os.getenv("TG_FEED_FILE", "").strip())
        has_mtproto = all(
            os.getenv(k, "").strip()
            for k in ("TG_API_ID", "TG_API_HASH", "TG_CHANNELS")
        )
        return has_feed or has_mtproto

    # ─── 模式 A：JSONL feed 文件 ─────────────────────────────────────────────

    def _fetch_feed_file(self) -> list[UnifiedPost]:
        path = os.getenv("TG_FEED_FILE", "").strip()
        if not path or not Path(path).exists():
            return []
        posts: list[UnifiedPost] = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("TG feed line %d not JSON, skipping", i)
                    continue
                posts.append(UnifiedPost(
                    platform="telegram",
                    post_id=str(item.get("id") or f"{path}:{i}"),
                    channel=str(item.get("channel") or "telegram"),
                    title=str(item.get("text") or "")[:200],
                    body=str(item.get("text") or ""),
                    url=str(item.get("url") or ""),
                    author=str(item.get("author") or ""),
                    created_utc=float(item.get("created_utc") or time.time()),
                    engagement_raw=int(item.get("engagement") or 0),
                    engagement_breakdown={},
                    platform_specific={},
                ))
        return posts

    # ─── 模式 B：Telethon（可选） ────────────────────────────────────────────

    def _fetch_mtproto(self) -> list[UnifiedPost]:
        try:
            # Telethon 是可选依赖；import 失败就回退到模式 A
            from telethon.sync import TelegramClient  # type: ignore
        except Exception:
            logger.info("telethon not installed; falling back to feed file")
            return []

        api_id = os.getenv("TG_API_ID", "").strip()
        api_hash = os.getenv("TG_API_HASH", "").strip()
        channels = [c.strip() for c in os.getenv("TG_CHANNELS", "").split(",") if c.strip()]
        limit_per_channel = int(os.getenv("TG_FETCH_LIMIT", "50"))
        if not (api_id and api_hash and channels):
            return []

        session_path = os.getenv("TG_SESSION", "nanobot_tg")
        posts: list[UnifiedPost] = []
        try:
            with TelegramClient(session_path, int(api_id), api_hash) as client:
                for ch in channels:
                    try:
                        entity = client.get_entity(ch)
                    except Exception as e:
                        logger.warning("TG resolve %s failed: %s", ch, e)
                        continue
                    for msg in client.iter_messages(entity, limit=limit_per_channel):
                        text = (getattr(msg, "message", "") or "").strip()
                        if not text:
                            continue
                        posts.append(UnifiedPost(
                            platform="telegram",
                            post_id=str(msg.id),
                            channel=str(ch),
                            title=text[:200],
                            body=text,
                            url=f"https://t.me/{str(ch).lstrip('@')}/{msg.id}",
                            author=str(getattr(msg, "sender_id", "") or ""),
                            created_utc=msg.date.timestamp() if msg.date else time.time(),
                            engagement_raw=int(getattr(msg, "views", 0) or 0),
                            engagement_breakdown={
                                "views": int(getattr(msg, "views", 0) or 0),
                                "forwards": int(getattr(msg, "forwards", 0) or 0),
                            },
                            platform_specific={},
                        ))
        except Exception as e:
            logger.warning("telethon fetch failed: %s", e)
            return []
        return posts

    def fetch(self) -> FetchReport:
        posts: list[UnifiedPost] = []
        err = ""

        mtproto_posts = self._fetch_mtproto()
        if mtproto_posts:
            posts.extend(mtproto_posts)

        feed_posts = self._fetch_feed_file()
        if feed_posts:
            posts.extend(feed_posts)

        if not posts and not self.is_configured():
            err = "no TG feed configured"

        return FetchReport(platform=self.platform, ok=True, posts=posts, error=err)
