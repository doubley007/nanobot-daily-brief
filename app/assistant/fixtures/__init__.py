"""
Demo fixtures —— 用于即便真实 API 都拿不到数据，也能本地跑通 demo 场景。
目前支持：黄金（gold）、比特币（bitcoin）。
"""
from assistant.fixtures.gold_fixture import (
    install_gold_fixture,
    build_gold_news_docs,
    build_gold_community_docs,
)
from assistant.fixtures.bitcoin_fixture import (
    install_bitcoin_fixture,
    build_bitcoin_news_docs,
    build_bitcoin_community_docs,
)

__all__ = [
    "install_gold_fixture",
    "build_gold_news_docs",
    "build_gold_community_docs",
    "install_bitcoin_fixture",
    "build_bitcoin_news_docs",
    "build_bitcoin_community_docs",
]
