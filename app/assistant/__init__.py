"""
nanobot Telegram 助理包。

这个包在现有 news_fetcher / community / telegram_sender 之上新增了一个对话式
入口，核心是五段式流水线：

    用户消息
       │
       ├─► query_router   把消息分类成 emotional_chat / market_decision / market_summary
       ├─► user_emotion   提取 primary_emotion、needs_confirmation、冲动风险
       ├─► rag            从 News/Community Knowledge Store 拉证据
       ├─► sentiment_aggregator  按资产聚合投资场景情绪（bullish_optimism、fomo 等）
       ├─► decision_engine       融合新闻/社区/趋势/风险信号 → 结构化决策
       └─► reply_composer        根据用户情绪定制措辞，最终回到 Telegram

pipeline.answer_question(text) 是对外的统一入口，telegram_bot 负责 long-poll。
"""
from __future__ import annotations

__all__ = [
    "answer_question",
]


def answer_question(text: str, user_id: str | int | None = None) -> str:
    """Top-level convenience wrapper. Lazy-imports to keep import time cheap."""
    from assistant.pipeline import answer_question as _impl
    return _impl(text, user_id=user_id)
