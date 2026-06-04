"""
RAG 知识层。

把抓到的新闻和社区帖子做成两张统一的知识表，供 bot 查询时检索：

  news       一条新闻一行：结构化字段 + 原文
  community  一条社区帖子一行：统一 schema + 情绪标签

store.KnowledgeStore 是最底层的 SQLite 操作。
news_indexer / community_indexer 负责把现有抓取结果转成知识条目并写入。
retriever 对外提供按资产/时间窗口检索的统一接口。

设计选择：
  - 先用 SQLite + 关键词 + recency 打分，能跑通黄金 demo，留好 embed() 接口
    以后接真向量库时替换 retriever 里的候选召回和打分函数即可。
  - 所有 write 路径都幂等（upsert），允许重复跑抓取而不脏库。
"""
from __future__ import annotations

from assistant.rag.store import (
    KnowledgeStore,
    NewsDoc,
    CommunityDoc,
    default_store,
)
from assistant.rag.retriever import Retriever, RetrievedEvidence

__all__ = [
    "KnowledgeStore",
    "NewsDoc",
    "CommunityDoc",
    "default_store",
    "Retriever",
    "RetrievedEvidence",
]
