"""
黄金场景 demo 数据 —— 为演示用户提问"我能不能买黄金？"而准备。

三类数据：
  - 8 条近期新闻：偏利多主导、混一条利空、一条中性
  - 30 条社区帖子：bullish 比例约 70%，有明显 FOMO / 拥挤迹象
  - 一个 manual 的 TrendSignal：7d +3%、30d +9% → 趋势向上、过热风险中等

写入时间戳都落在"过去 48 小时"，所以默认 72h 窗口的 retriever 能查到。
"""
from __future__ import annotations

import time

from assistant.rag.community_indexer import unified_posts_to_docs
from assistant.rag.news_indexer import raw_items_to_docs
from assistant.rag.store import CommunityDoc, KnowledgeStore, NewsDoc, default_store
from community.schema import UnifiedPost


# ─── 简易 RawNewsItem duck ───────────────────────────────────────────────────

class _RawNewsStub:
    def __init__(self, title: str, summary: str, source: str,
                 url: str, published_at: str, category: str = "markets"):
        self.title = title
        self.summary = summary
        self.source = source
        self.url = url
        self.published_at = published_at
        self.category = category


def build_gold_news_docs() -> list[NewsDoc]:
    now = time.time()
    # published_at 用 ISO 字符串，news_indexer 会解析
    def iso(offset_hours: float) -> str:
        import datetime as dt
        ts = now - offset_hours * 3600
        return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")

    raw = [
        _RawNewsStub(
            title="Gold hits record high as rate cut bets strengthen",
            summary="Spot gold surged past $2,450 as markets priced in a higher probability of a Fed rate cut in September. Safe haven demand also rose on Middle East tensions.",
            source="Reuters", url="https://example.com/gold-record-1",
            published_at=iso(6),
        ),
        _RawNewsStub(
            title="Fed officials signal dovish pivot on inflation progress",
            summary="Two Federal Reserve officials hinted at policy easing if inflation continues to cool. Treasury yields fell in response, boosting non-yielding gold.",
            source="Bloomberg", url="https://example.com/fed-dovish",
            published_at=iso(10),
        ),
        _RawNewsStub(
            title="Geopolitical tension in Middle East fuels safe haven bid",
            summary="Escalating regional tension sent investors into gold and the yen. Analysts say the safe haven premium could persist through the quarter.",
            source="FT", url="https://example.com/geo-tension",
            published_at=iso(14),
        ),
        _RawNewsStub(
            title="Central banks keep buying gold at record pace",
            summary="Emerging market central banks added another 42 tonnes of gold to reserves last month, continuing a multi-year trend.",
            source="WSJ", url="https://example.com/cb-buying",
            published_at=iso(20),
        ),
        _RawNewsStub(
            title="Dollar firms on strong jobs print, caps gold gains",
            summary="A stronger-than-expected US payroll report lifted the dollar and limited gold's upside. DXY climbed to a two-week high.",
            source="Reuters", url="https://example.com/usd-strength",
            published_at=iso(26),
        ),
        _RawNewsStub(
            title="Goldman raises gold year-end target to $2,700",
            summary="Goldman Sachs lifted its year-end gold forecast citing persistent central bank demand and expected rate cuts.",
            source="CNBC", url="https://example.com/gs-target",
            published_at=iso(34),
        ),
        _RawNewsStub(
            title="ETF gold holdings tick up first time in five months",
            summary="Gold-backed ETFs saw net inflows for the first time since early this year, suggesting retail is rotating back into gold.",
            source="MarketWatch", url="https://example.com/etf-inflow",
            published_at=iso(40),
        ),
        _RawNewsStub(
            title="Analysts flag crowded gold positioning",
            summary="Speculative long positioning in gold futures hit a 3-year high, prompting some strategists to warn of short-term mean-reversion risk.",
            source="Reuters", url="https://example.com/crowded-gold",
            published_at=iso(44),
        ),
    ]
    return raw_items_to_docs(raw)


def build_gold_community_docs() -> list[CommunityDoc]:
    now = time.time()

    def hour(h: float) -> float:
        return now - h * 3600

    posts = [
        # Reddit bullish FOMO
        UnifiedPost(platform="reddit", post_id="r1", channel="wallstreetbets",
                    title="Gold is breaking out, getting in before I miss this move",
                    body="FOMO is real. Everyone on my feed is long gold.",
                    url="https://reddit.com/r/wsb/r1",
                    created_utc=hour(2), engagement_raw=420),
        UnifiedPost(platform="reddit", post_id="r2", channel="wallstreetbets",
                    title="All in on gold calls, follow the rally",
                    body="Rate cuts are coming, gold to the moon.",
                    url="https://reddit.com/r/wsb/r2",
                    created_utc=hour(3), engagement_raw=310),
        UnifiedPost(platform="reddit", post_id="r3", channel="investing",
                    title="Thinking of adding gold as safe haven hedge",
                    body="Inflation hedge + rate cut narrative, makes sense to allocate 5-10%.",
                    url="https://reddit.com/r/investing/r3",
                    created_utc=hour(4), engagement_raw=180),
        UnifiedPost(platform="reddit", post_id="r4", channel="stocks",
                    title="Gold looks bullish here, rate cut bets are real",
                    body="Long GLD and GDX.",
                    url="https://reddit.com/r/stocks/r4",
                    created_utc=hour(5), engagement_raw=250),
        UnifiedPost(platform="reddit", post_id="r5", channel="wallstreetbets",
                    title="大家都在买黄金，我是不是也该上车了",
                    body="怕错过这波，不知道还能不能追",
                    url="https://reddit.com/r/wsb/r5",
                    created_utc=hour(6), engagement_raw=400),
        UnifiedPost(platform="reddit", post_id="r6", channel="investing",
                    title="Gold record high — anyone else worried about crowded trade?",
                    body="Speculative positioning is at 3-year high, I'm cautious.",
                    url="https://reddit.com/r/investing/r6",
                    created_utc=hour(8), engagement_raw=150),
        UnifiedPost(platform="reddit", post_id="r7", channel="economics",
                    title="Dollar strength could cap gold rally near-term",
                    body="Bearish setup if payroll keeps surprising.",
                    url="https://reddit.com/r/economics/r7",
                    created_utc=hour(10), engagement_raw=90),
        UnifiedPost(platform="reddit", post_id="r8", channel="stocks",
                    title="Buy the dip on gold, central banks still accumulating",
                    body="Long term bullish, not worried about the wobble.",
                    url="https://reddit.com/r/stocks/r8",
                    created_utc=hour(11), engagement_raw=220),
        UnifiedPost(platform="reddit", post_id="r9", channel="wallstreetbets",
                    title="梭哈黄金，rate cut 就在 9 月",
                    body="all in，长期持有，钻石手",
                    url="https://reddit.com/r/wsb/r9",
                    created_utc=hour(12), engagement_raw=350),
        UnifiedPost(platform="reddit", post_id="r10", channel="wallstreetbets",
                    title="Gold feels extended, not adding here",
                    body="Not sure this is the spot to chase.",
                    url="https://reddit.com/r/wsb/r10",
                    created_utc=hour(13), engagement_raw=140),
        # X bullish KOL-ish
        UnifiedPost(platform="x", post_id="x1", channel="@macrotrader",
                    title="Gold breakout confirmed. Safe haven + rate cut narrative aligned. Bullish.",
                    url="https://x.com/macrotrader/status/1",
                    author="macrotrader",
                    created_utc=hour(1), engagement_raw=800),
        UnifiedPost(platform="x", post_id="x2", channel="@goldbug",
                    title="Central banks still buying. Retail ETFs turning. This is the setup.",
                    url="https://x.com/goldbug/status/2",
                    author="goldbug",
                    created_utc=hour(2), engagement_raw=600),
        UnifiedPost(platform="x", post_id="x3", channel="@quantfade",
                    title="Speculative positioning in gold now at 3-year high. Be careful chasing.",
                    url="https://x.com/quantfade/status/3",
                    author="quantfade",
                    created_utc=hour(5), engagement_raw=420),
        UnifiedPost(platform="x", post_id="x4", channel="@fxview",
                    title="Stronger dollar could cap gold near-term, but dips are buyable.",
                    url="https://x.com/fxview/status/4",
                    author="fxview",
                    created_utc=hour(7), engagement_raw=220),
        UnifiedPost(platform="x", post_id="x5", channel="@macronews",
                    title="Gold to $2700 per Goldman Sachs. Follow the rally.",
                    url="https://x.com/macronews/status/5",
                    author="macronews",
                    created_utc=hour(9), engagement_raw=700),
        # Discord
        UnifiedPost(platform="discord", post_id="d1", channel="macro-general",
                    title="Anyone still adding gold here? Feels crowded.",
                    body="I'm starting to question the trade.",
                    url="https://discord.com/channels/x/d1",
                    created_utc=hour(3), engagement_raw=30),
        UnifiedPost(platform="discord", post_id="d2", channel="macro-general",
                    title="Long GLD, strong conviction, holding forever.",
                    body="Diamond hands on this one.",
                    url="https://discord.com/channels/x/d2",
                    created_utc=hour(4), engagement_raw=45),
        UnifiedPost(platform="discord", post_id="d3", channel="trades",
                    title="跟风买黄金，怕错过",
                    body="大家都在买，我也冲",
                    url="https://discord.com/channels/x/d3",
                    created_utc=hour(5), engagement_raw=18),
        UnifiedPost(platform="discord", post_id="d4", channel="trades",
                    title="抄底黄金，长期看好",
                    url="https://discord.com/channels/x/d4",
                    created_utc=hour(8), engagement_raw=22),
        UnifiedPost(platform="discord", post_id="d5", channel="hedge-talk",
                    title="Skeptical on this gold rally, feels like late stage",
                    body="Crowded long, watch for reversal.",
                    url="https://discord.com/channels/x/d5",
                    created_utc=hour(12), engagement_raw=28),
        # Telegram feed（模式 A，预设内容）
        UnifiedPost(platform="telegram", post_id="t1", channel="@goldsignals",
                    title="Gold looks great here, buy the dip",
                    body="Target 2500, stop 2380",
                    url="https://t.me/goldsignals/1",
                    created_utc=hour(1), engagement_raw=90),
        UnifiedPost(platform="telegram", post_id="t2", channel="@chinamacro",
                    title="央行持续买金，避险逻辑未变",
                    body="长期看好",
                    url="https://t.me/chinamacro/2",
                    created_utc=hour(3), engagement_raw=70),
        UnifiedPost(platform="telegram", post_id="t3", channel="@goldsignals",
                    title="上车黄金，rate cut 预期起来了",
                    body="跟风党集合",
                    url="https://t.me/goldsignals/3",
                    created_utc=hour(5), engagement_raw=110),
        UnifiedPost(platform="telegram", post_id="t4", channel="@risktalk",
                    title="不确定此时是否追金，positioning 过高",
                    body="不知道还能不能追",
                    url="https://t.me/risktalk/4",
                    created_utc=hour(7), engagement_raw=40),
        UnifiedPost(platform="telegram", post_id="t5", channel="@macronotes",
                    title="看多黄金，利好堆积",
                    url="https://t.me/macronotes/5",
                    created_utc=hour(9), engagement_raw=60),
        # 几条中性 / 轻度看空
        UnifiedPost(platform="reddit", post_id="r11", channel="stocks",
                    title="Gold is just OK here, waiting for retest",
                    body="Not sure I want to chase.",
                    url="https://reddit.com/r/stocks/r11",
                    created_utc=hour(14), engagement_raw=80),
        UnifiedPost(platform="x", post_id="x6", channel="@cashbear",
                    title="Shorting gold here, overbought",
                    url="https://x.com/cashbear/status/6",
                    author="cashbear",
                    created_utc=hour(16), engagement_raw=150),
        UnifiedPost(platform="discord", post_id="d6", channel="hedge-talk",
                    title="Sell gold, de-escalation in Middle East",
                    url="https://discord.com/channels/x/d6",
                    created_utc=hour(18), engagement_raw=10),
        UnifiedPost(platform="reddit", post_id="r12", channel="investing",
                    title="Gold mixed — waiting for CPI print",
                    url="https://reddit.com/r/investing/r12",
                    created_utc=hour(20), engagement_raw=60),
        UnifiedPost(platform="telegram", post_id="t6", channel="@risktalk",
                    title="Gold 现在不知道该追还是等回调",
                    url="https://t.me/risktalk/6",
                    created_utc=hour(22), engagement_raw=30),
    ]

    return unified_posts_to_docs(posts)


def install_gold_fixture(store: KnowledgeStore | None = None,
                         clear_existing: bool = True) -> tuple[int, int]:
    """
    把黄金 demo 数据写进 store。返回 (news_rows, community_rows)。
    默认会先清空 store（测试/demo 环境用）。
    """
    store = store or default_store()
    if clear_existing:
        store.clear()
    news = build_gold_news_docs()
    community = build_gold_community_docs()
    n_news = store.upsert_news(news)
    n_community = store.upsert_community(community)
    return n_news, n_community
