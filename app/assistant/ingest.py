"""
把 News / Community 抓到的数据写进 RAG 知识库。

供两种用法：
  1) 作为 Python 模块被 daily_job / cron 调用
  2) 作为 CLI：  python -m assistant.ingest [--news-only|--community-only]

设计：
  - News 走现有 news_fetcher.fetch_financial_news() 拉一批再索引
  - Community 遍历 sources/all_enabled_adapters() 取已配置源
  - 两者都是幂等（store 是 upsert），重复跑不会出问题
"""
from __future__ import annotations

import argparse
import logging
import sys

from assistant.rag.community_indexer import index_community
from assistant.rag.news_indexer import index_news
from assistant.rag.store import default_store

logger = logging.getLogger("assistant.ingest")


def ingest_news(limit: int = 30) -> int:
    from news_fetcher import fetch_financial_news
    raw = fetch_financial_news(limit=limit)
    logger.info("fetched %d news items", len(raw))
    return index_news(raw, store=default_store())


def ingest_community() -> dict[str, int]:
    from sources import all_enabled_adapters
    totals: dict[str, int] = {}
    for adapter in all_enabled_adapters():
        report = adapter.safe_fetch()
        if not report.ok:
            logger.warning("%s: %s", adapter.platform, report.error)
            totals[adapter.platform] = 0
            continue
        n = index_community(report.posts, store=default_store())
        totals[adapter.platform] = n
        logger.info("%s: indexed %d posts", adapter.platform, n)
    return totals


def run(news_only: bool, community_only: bool) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if community_only:
        ingest_community()
        return
    if news_only:
        ingest_news()
        return
    ingest_news()
    ingest_community()
    logger.info("store state: %s", default_store().count())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ingest news/community into RAG")
    parser.add_argument("--news-only", action="store_true")
    parser.add_argument("--community-only", action="store_true")
    args = parser.parse_args(argv)
    run(news_only=args.news_only, community_only=args.community_only)


if __name__ == "__main__":
    main(sys.argv[1:])
