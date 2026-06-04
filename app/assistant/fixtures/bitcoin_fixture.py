"""
比特币场景 demo 数据 —— 为演示用户提问"比特币现在能买吗？" 而准备。

三类数据：
  - 8 条近期新闻：利多主导（ETF 申请、减半、机构入场）+ 混一条监管利空
  - 25 条社区帖子：bullish 约 65%，FOMO/conviction 声量显著
  - 一个 manual 的 TrendSignal：7d +5%、30d +18% → 偏热、过热风险中等偏高

写入时间戳都落在"过去 48 小时"，默认 72h 窗口的 retriever 能查到。
"""
from __future__ import annotations

import time

from assistant.rag.community_indexer import unified_posts_to_docs
from assistant.rag.news_indexer import raw_items_to_docs
from assistant.rag.store import CommunityDoc, KnowledgeStore, NewsDoc, default_store
from community.schema import UnifiedPost


class _RawNewsStub:
    def __init__(self, title: str, summary: str, source: str,
                 url: str, published_at: str, category: str = "markets"):
        self.title = title
        self.summary = summary
        self.source = source
        self.url = url
        self.published_at = published_at
        self.category = category


def build_bitcoin_news_docs() -> list[NewsDoc]:
    now = time.time()

    def iso(offset_hours: float) -> str:
        import datetime as dt
        ts = now - offset_hours * 3600
        return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")

    raw = [
        _RawNewsStub(
            title="Bitcoin ETF approval odds rise as SEC review window nears",
            summary="Analysts see 70%+ probability of a spot Bitcoin ETF approval this cycle, "
                    "citing improved regulatory dialogue and institutional demand.",
            source="Bloomberg", url="https://example.com/btc-etf-1",
            published_at=iso(5),
        ),
        _RawNewsStub(
            title="BlackRock Bitcoin ETF filing moves to final review stage",
            summary="BlackRock's iShares Bitcoin Trust application advanced to final SEC review, "
                    "spurring a wave of institutional buying interest.",
            source="Reuters", url="https://example.com/blackrock-btc",
            published_at=iso(9),
        ),
        _RawNewsStub(
            title="Bitcoin halving countdown: 60 days to next supply cut",
            summary="With the next Bitcoin halving roughly 60 days away, on-chain data shows "
                    "accumulation by long-term holders accelerating.",
            source="CoinDesk", url="https://example.com/btc-halving",
            published_at=iso(15),
        ),
        _RawNewsStub(
            title="MicroStrategy adds 10,000 BTC to corporate treasury",
            summary="MicroStrategy announced another $650M Bitcoin purchase, bringing total "
                    "holdings to over 190,000 BTC.",
            source="CNBC", url="https://example.com/microstrategy-btc",
            published_at=iso(22),
        ),
        _RawNewsStub(
            title="SEC signals stricter crypto exchange oversight",
            summary="SEC Chair reiterated plans for tighter oversight of crypto exchanges, "
                    "raising short-term regulatory risk for the sector.",
            source="WSJ", url="https://example.com/sec-crypto",
            published_at=iso(28),
        ),
        _RawNewsStub(
            title="Bitcoin on-chain metrics: long-term holder supply at 5-year high",
            summary="On-chain analysis shows LTH supply at its highest since 2019, suggesting "
                    "conviction buyers are not selling into the rally.",
            source="Glassnode", url="https://example.com/btc-lth",
            published_at=iso(35),
        ),
        _RawNewsStub(
            title="Fidelity launches Bitcoin IRA product for retail investors",
            summary="Fidelity expanded access to Bitcoin-backed retirement accounts for "
                    "retail clients, signaling mainstream institutional adoption.",
            source="MarketWatch", url="https://example.com/fidelity-btc-ira",
            published_at=iso(40),
        ),
        _RawNewsStub(
            title="Crypto analysts warn BTC overbought near $50K resistance",
            summary="Technical analysts flagged overbought RSI readings and heavy resistance "
                    "near the $50,000 psychological level for Bitcoin.",
            source="CoinTelegraph", url="https://example.com/btc-overbought",
            published_at=iso(46),
        ),
    ]
    return raw_items_to_docs(raw)


def build_bitcoin_community_docs() -> list[CommunityDoc]:
    now = time.time()

    def hour(h: float) -> float:
        return now - h * 3600

    posts = [
        # Reddit bullish / FOMO
        # Reddit bullish / FOMO — use explicit signal terms for classifier
        UnifiedPost(platform="reddit", post_id="br1", channel="wallstreetbets",
                    title="Bitcoin ETF is basically confirmed, very bullish signal",
                    body="Long BTC. BlackRock doesn't lose. Buy the dip before retail floods in.",
                    url="https://reddit.com/r/wsb/br1",
                    created_utc=hour(2), engagement_raw=850),
        UnifiedPost(platform="reddit", post_id="br2", channel="CryptoCurrency",
                    title="Halving in 60 days, BTC rally incoming — bullish setup",
                    body="On-chain says LTH not selling. Looking good. Long term bullish.",
                    url="https://reddit.com/r/CryptoCurrency/br2",
                    created_utc=hour(3), engagement_raw=620),
        UnifiedPost(platform="reddit", post_id="br3", channel="Bitcoin",
                    title="比特币 ETF 通过了就要大涨，上车看多",
                    body="现在进还来得及，做多，别像上次一样踏空",
                    url="https://reddit.com/r/Bitcoin/br3",
                    created_utc=hour(4), engagement_raw=480),
        UnifiedPost(platform="reddit", post_id="br4", channel="CryptoCurrency",
                    title="All in BTC pre-halving, diamond hands, long",
                    body="We are early. Bullish. Institutions still coming.",
                    url="https://reddit.com/r/CryptoCurrency/br4",
                    created_utc=hour(5), engagement_raw=400),
        UnifiedPost(platform="reddit", post_id="br5", channel="investing",
                    title="看多比特币，加仓 BTC ETF",
                    body="ETF approval would normalize BTC. Rally expected. 利好明确.",
                    url="https://reddit.com/r/investing/br5",
                    created_utc=hour(6), engagement_raw=200),
        UnifiedPost(platform="reddit", post_id="br6", channel="wallstreetbets",
                    title="BTC feels crowded but FOMO is real, keep buying",
                    body="大家都在买，我也冲，跟风入场，怕错过",
                    url="https://reddit.com/r/wsb/br6",
                    created_utc=hour(7), engagement_raw=320),
        UnifiedPost(platform="reddit", post_id="br7", channel="Bitcoin",
                    title="Worried BTC is overbought near $50K",
                    body="RSI is extended. I'm waiting for a retest of $45K.",
                    url="https://reddit.com/r/Bitcoin/br7",
                    created_utc=hour(9), engagement_raw=110),
        UnifiedPost(platform="reddit", post_id="br8", channel="CryptoCurrency",
                    title="跟风买比特币，大家都在买，利好消息不断",
                    body="上车，看多，再涨一段",
                    url="https://reddit.com/r/CryptoCurrency/br8",
                    created_utc=hour(10), engagement_raw=270),
        UnifiedPost(platform="reddit", post_id="br9", channel="investing",
                    title="BTC halving cycle is bullish — long term hold, looks good",
                    body="Supply shock + institutional demand = good setup. Buy the dip.",
                    url="https://reddit.com/r/investing/br9",
                    created_utc=hour(11), engagement_raw=180),
        UnifiedPost(platform="reddit", post_id="br10", channel="wallstreetbets",
                    title="梭哈比特币，ETF 就是这轮的催化，全仓做多",
                    body="All in，长期看好，重仓持有",
                    url="https://reddit.com/r/wsb/br10",
                    created_utc=hour(12), engagement_raw=430),
        # X / Twitter
        UnifiedPost(platform="x", post_id="bx1", channel="@cryptoking",
                    title="BTC ETF = institutional floodgates. Bullish. Buy this rally.",
                    url="https://x.com/cryptoking/1",
                    author="cryptoking",
                    created_utc=hour(1), engagement_raw=1200),
        UnifiedPost(platform="x", post_id="bx2", channel="@halving_watch",
                    title="60 days to halving. LTH at 5Y high. Bullish breakout setup.",
                    url="https://x.com/halving_watch/2",
                    author="halving_watch",
                    created_utc=hour(2), engagement_raw=900),
        UnifiedPost(platform="x", post_id="bx3", channel="@btcbear",
                    title="$50K is major resistance. RSI overbought. Don't chase this level.",
                    url="https://x.com/btcbear/3",
                    author="btcbear",
                    created_utc=hour(4), engagement_raw=380),
        UnifiedPost(platform="x", post_id="bx4", channel="@macroalpha",
                    title="BTC halving + ETF = perfect storm. Long BTC. Good time to buy.",
                    url="https://x.com/macroalpha/4",
                    author="macroalpha",
                    created_utc=hour(6), engagement_raw=750),
        UnifiedPost(platform="x", post_id="bx5", channel="@degentrader",
                    title="5x on BTC since Jan. Bullish. Number go up, moon incoming.",
                    url="https://x.com/degentrader/5",
                    author="degentrader",
                    created_utc=hour(8), engagement_raw=550),
        # Discord
        UnifiedPost(platform="discord", post_id="bd1", channel="crypto-general",
                    title="Long BTC right now. ETF news = rally, looks good.",
                    body="I loaded up this morning. Bullish.",
                    url="https://discord.com/channels/x/bd1",
                    created_utc=hour(2), engagement_raw=40),
        UnifiedPost(platform="discord", post_id="bd2", channel="trades",
                    title="Long BTC, strong conviction on halving cycle, diamond hands",
                    body="不selling until $100K. 长期看好，重仓持有.",
                    url="https://discord.com/channels/x/bd2",
                    created_utc=hour(3), engagement_raw=55),
        UnifiedPost(platform="discord", post_id="bd3", channel="crypto-general",
                    title="怕踏空比特币，跟风买入，大家都在买",
                    body="看大家都在买，我也冲，fomo，上车",
                    url="https://discord.com/channels/x/bd3",
                    created_utc=hour(5), engagement_raw=22),
        UnifiedPost(platform="discord", post_id="bd4", channel="hedge-talk",
                    title="BTC near $50K feels extended. Wait for pullback.",
                    body="I'd rather miss the last 10% than buy at the top.",
                    url="https://discord.com/channels/x/bd4",
                    created_utc=hour(7), engagement_raw=18),
        UnifiedPost(platform="discord", post_id="bd5", channel="trades",
                    title="Halving is bullish narrative, ETF is catalyst. Long.",
                    url="https://discord.com/channels/x/bd5",
                    created_utc=hour(9), engagement_raw=33),
        # Telegram
        UnifiedPost(platform="telegram", post_id="bt1", channel="@btcsignals",
                    title="BTC breakout, bullish. ETF approval = good time to buy.",
                    body="Target $55K, stop $46K. Long.",
                    url="https://t.me/btcsignals/1",
                    created_utc=hour(1), engagement_raw=130),
        UnifiedPost(platform="telegram", post_id="bt2", channel="@cryptomacro",
                    title="比特币减半在即，看多，加仓时机",
                    body="利好明确，长期持有",
                    url="https://t.me/cryptomacro/2",
                    created_utc=hour(4), engagement_raw=80),
        UnifiedPost(platform="telegram", post_id="bt3", channel="@btcsignals",
                    title="全力买入做多 BTC，ETF 就是这轮的催化，梭哈",
                    body="All in，重仓",
                    url="https://t.me/btcsignals/3",
                    created_utc=hour(6), engagement_raw=160),
        UnifiedPost(platform="telegram", post_id="bt4", channel="@riskwatch",
                    title="BTC 涨幅过大，追高风险不小",
                    body="建议等回调到 $46K 再入",
                    url="https://t.me/riskwatch/4",
                    created_utc=hour(8), engagement_raw=55),
        UnifiedPost(platform="telegram", post_id="bt5", channel="@cryptomacro",
                    title="机构在买比特币，看多，利好，散户也该上车",
                    url="https://t.me/cryptomacro/5",
                    created_utc=hour(10), engagement_raw=70),
    ]

    return unified_posts_to_docs(posts)


def install_bitcoin_fixture(store: KnowledgeStore | None = None,
                            clear_existing: bool = True) -> tuple[int, int]:
    """
    把比特币 demo 数据写进 store。返回 (news_rows, community_rows)。
    默认先清空 store（测试/demo 环境用）。
    """
    store = store or default_store()
    if clear_existing:
        store.clear()
    news = build_bitcoin_news_docs()
    community = build_bitcoin_community_docs()
    n_news = store.upsert_news(news)
    n_community = store.upsert_community(community)
    return n_news, n_community
