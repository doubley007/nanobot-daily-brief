"""
Self-curated #prediction-markets feed.

Builds a stream of short, structured items aligned with the boss's priority
themes — Singapore real estate, rates/macro, US/HK/SG equities — and posts
them into the user's own Discord channel via the webhook sender.

Why self-curated instead of reading others' channels:
  The nanobot Discord ingestor already exists. The value here is controlling
  *what* flows in. A curated pool lets us apply the three-tier channel
  strategy (news / discussion / vertical) on our own terms, and keeps the
  downstream community analyst fed with high-signal content.

Structure:
  FeedItem        — platform-agnostic input shape
  format_item     — renders a single item to a Discord-friendly message
  publish_items   — posts items, one message per item, via discord_sender
  main()          — runs the mock batch for pipeline smoke-testing

Step 3 will replace `_mock_items()` with real sources (news API filtered to
priority themes, and a distilled take of the community analyst report).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from discord_sender import send_to_discord
from llm_adapter import local_llm_callable

logger = logging.getLogger(__name__)

# Aligned with user memory: Singapore real estate / rates / US-HK-SG equities
# are the boss-priority themes. Categories here drive the emoji + ordering.
CATEGORY_EMOJI = {
    "sg_property": "🏢",
    "rates_macro": "📉",
    "equities": "📈",
    "event": "🗓️",
    "general": "📰",
}

CATEGORY_LABEL_CN = {
    "sg_property": "新加坡房地产",
    "rates_macro": "利率/宏观",
    "equities": "美股/港股/新加坡",
    "event": "关键事件",
    "general": "综合",
}


@dataclass
class FeedItem:
    """
    One curated item for the prediction-markets feed.

    `category` buckets the item into one of the priority themes so the
    downstream nanobot reader can preserve theme structure without re-parsing
    the text. `why_it_matters` is optional but strongly encouraged — it's the
    line that turns a headline into a tradeable observation.
    """
    title: str
    category: str = "general"
    summary: str = ""
    why_it_matters: str = ""
    source: str = ""
    url: str = ""
    tags: list[str] = field(default_factory=list)


def format_item(item: FeedItem) -> str:
    """Render a FeedItem to a single Discord message (markdown)."""
    emoji = CATEGORY_EMOJI.get(item.category, CATEGORY_EMOJI["general"])
    label = CATEGORY_LABEL_CN.get(item.category, CATEGORY_LABEL_CN["general"])

    # Header: emoji + category chip + bold title
    lines: list[str] = [f"{emoji} **[{label}]** {item.title}"]

    if item.summary:
        lines.append(item.summary)

    if item.why_it_matters:
        lines.append(f"> 为何重要：{item.why_it_matters}")

    footer_parts: list[str] = []
    if item.source:
        footer_parts.append(item.source)
    if item.tags:
        footer_parts.append(" ".join(f"#{t}" for t in item.tags))
    if footer_parts:
        lines.append("— " + " · ".join(footer_parts))

    if item.url:
        # Plain URL on its own line; Discord auto-previews and the nanobot
        # reader can extract it cleanly from the message text.
        lines.append(item.url)

    return "\n".join(lines)


def publish_items(
    items: list[FeedItem],
    sender: Callable[[str], dict] = send_to_discord,
) -> int:
    """
    Post each item as its own Discord message. Returns the number sent.
    One message per item keeps the downstream reader's cluster boundaries
    clean — merging multiple items into one message would let unrelated
    themes share a cluster.
    """
    sent = 0
    for item in items:
        content = format_item(item)
        try:
            sender(content)
            sent += 1
        except Exception as e:
            logger.warning("Failed to send feed item %r: %s", item.title[:40], e)
    return sent


# ─── Real sources (Step 3) ───────────────────────────────────────────────────
#
# Two inputs feed the channel:
#   1. News pipeline (news_fetcher.fetch_financial_news) — bucketed,
#      deduped, scored; we keep only items that map to the boss's priority
#      themes (SG property / rates-macro / equities).
#   2. Community analyst report (orchestrator.run_community_analyst) —
#      distilled headline topics that already went through credibility
#      filtering + LLM interpretation.
#
# Both converge on the FeedItem shape so the sender doesn't need to know
# where content came from.

# Bucket → feed category. Buckets not in this map are skipped — the feed is
# curated, so 'general' / low-signal items don't get forwarded.
_BUCKET_TO_CATEGORY = {
    "fixed_income_macro": "rates_macro",
    "credit": "rates_macro",
    "fx_liquidity": "rates_macro",
    "real_estate_loans": "sg_property",  # refined below by keyword
    "private_equity": "equities",
    "listed_equity": "equities",
    "singapore_local": "sg_property",     # refined below by keyword
    "regulation": "rates_macro",
}

# Keyword hints used to refine the Singapore-property vs rates split when
# the bucket alone is ambiguous.
_SG_PROPERTY_HINTS = (
    "real estate", "commercial real estate", "property", "ura", "hdb",
    "reit", "reits", "office market",
)
_RATES_HINTS = (
    "treasury", "yield", "fed", "cpi", "inflation", "rate cut", "bond",
    "mas", "sgd", "usd",
)


# Narrative/clickbait patterns that sometimes squeak past news_fetcher's
# score but have no place in a curated prediction-markets feed.
_FEED_REJECT_PATTERNS = (
    "if you invested",
    "a decade ago",
    "this is how much",
    "it'd be worth now",
    "could have made you",
    "would have turned",
    "christmas",          # seasonal consumer-spending stories
    "holiday spending",
    "shoppers",
)

# Strict priority-theme hints. These are the *direct* signals we accept;
# a story must mention at least one of these in its TITLE (summary is too
# noisy, a passing mention of "yields" in a Tesla story shouldn't qualify).
#
# Singapore / HK / rates / macro / equities — the boss's scope, hard-enforced.
_FEED_TITLE_HINTS = (
    # Singapore property / REITs
    "singapore", "mas ", "ura ", "hdb", "reit", "reits", "sti",
    "commercial real estate", "office market",
    # Rates / macro direct
    "treasury", "yield", "yields", "fed", "federal reserve", "cpi",
    "inflation", "rate cut", "rate cuts", "interest rate", "interest rates",
    "fomc", "boj", "ecb", "payroll", "payrolls",
    # FX
    "sgd", "usd/sgd", "yuan", "yen",
    # Major indices / HK / US markets
    "hong kong", "hkex", "hang seng", "hsi", "s&p 500", "nasdaq",
    "dow jones",
    # Earnings with portfolio relevance — paired with sector below
    "bank earnings", "insurer earnings", "guidance cut", "guidance raised",
)


def _news_category(item) -> str | None:
    """Map a RawNewsItem to a feed category, or None to skip."""
    from news_fetcher import detect_portfolio_impact_bucket

    title_lc = item.title.lower()
    text_lc = f"{item.title} {item.summary}".lower()

    if any(p in text_lc for p in _FEED_REJECT_PATTERNS):
        return None
    # Strict: the title itself must name a priority theme — this kills
    # tangential items whose finance angle only shows up in the summary.
    if not any(h in title_lc for h in _FEED_TITLE_HINTS):
        return None

    bucket = detect_portfolio_impact_bucket(item)
    category = _BUCKET_TO_CATEGORY.get(bucket)
    if category is None:
        return None

    # Refine SG-leaning buckets: only keep sg_property if the item is really
    # about property; otherwise route to rates_macro.
    if category == "sg_property":
        if not any(h in text_lc for h in _SG_PROPERTY_HINTS):
            if any(h in text_lc for h in _RATES_HINTS):
                return "rates_macro"
            return None
    return category


def _news_to_feed_items(limit: int = 10) -> list[FeedItem]:
    from news_fetcher import fetch_financial_news

    raw = fetch_financial_news(limit=limit)
    items: list[FeedItem] = []
    for n in raw:
        category = _news_category(n)
        if category is None:
            continue
        items.append(FeedItem(
            title=n.title,
            category=category,
            summary=(n.summary or "")[:280],
            source=n.source,
            url=n.url or "",
            tags=[category],
        ))
    return items


_COMMUNITY_SG_HINTS_CN = ("新加坡", "房地产", "地产", "reit", "楼市", "房价")
_COMMUNITY_RATES_HINTS_CN = (
    "利率", "美联储", "加息", "降息", "通胀", "cpi", "美债", "国债",
    "收益率", "国库券", "宏观", "美元", "汇率",
)
_COMMUNITY_EQUITY_HINTS_CN = (
    "股市", "美股", "港股", "纳指", "标普", "恒生", "道指", "板块",
    "业绩", "财报", "银行", "保险",
)

_COMMUNITY_EQUITY_HINTS_EN = (
    "s&p", "nasdaq", "dow", "sti", "hsi", "hang seng", "hong kong",
    "earnings", "guidance", "bank", "banks", "insurer", "reit",
    "equity", "equities", "stocks",
)


def _community_category(topic) -> str | None:
    """Map a TopicCluster to a feed category, or None to skip."""
    text_lc = f"{topic.headline} {topic.rule_label} {topic.discussion_focus}".lower()

    if any(h in text_lc for h in _SG_PROPERTY_HINTS) or \
       any(h in text_lc for h in _COMMUNITY_SG_HINTS_CN):
        return "sg_property"
    if any(h in text_lc for h in _RATES_HINTS) or \
       any(h in text_lc for h in _COMMUNITY_RATES_HINTS_CN):
        return "rates_macro"
    if any(h in text_lc for h in _COMMUNITY_EQUITY_HINTS_EN) or \
       any(h in text_lc for h in _COMMUNITY_EQUITY_HINTS_CN):
        return "equities"
    return None


def _community_to_feed_items() -> list[FeedItem]:
    """
    Pull the community analyst's headline topics and convert each into a
    feed item. The analyst has already done the credibility filtering, so
    anything in `headline_topics` is worth surfacing.
    """
    try:
        from community.orchestrator import run_community_analyst
    except Exception as e:
        logger.warning("community orchestrator unavailable: %s", e)
        return []

    _, report = run_community_analyst(llm_callable=local_llm_callable)
    if not report.headline_topics:
        return []

    items: list[FeedItem] = []
    for t in report.headline_topics:
        if t.credibility.is_noise:
            continue
        headline = (t.headline or t.rule_label or "").strip()
        if not headline:
            continue
        category = _community_category(t)
        if category is None:
            continue
        items.append(FeedItem(
            title=f"社区热议：{headline}",
            category=category,
            summary=t.discussion_focus or "",
            why_it_matters=t.market_relevance or t.insurance_angle or "",
            source=f"community · {'/'.join(t.platforms)}",
            tags=["community"] + list(t.platforms),
        ))
    return items


def build_real_items(news_limit: int = 10) -> list[FeedItem]:
    """Assemble today's real-source feed: news first, then community."""
    return _news_to_feed_items(limit=news_limit) + _community_to_feed_items()


# ─── Mock batch (Step 2, kept for smoke-testing) ─────────────────────────────

def _mock_items() -> list[FeedItem]:
    today = datetime.now().strftime("%Y-%m-%d")
    return [
        FeedItem(
            title="URA Q1 private home price index preview",
            category="sg_property",
            summary=(
                "市场普遍预期 Q1 私宅价格环比持平至微升 0.3%，"
                "CCR 高端项目和 OCR 新盘入市节奏是主要看点。"
            ),
            why_it_matters=(
                "若读数超预期，MAS 可能延续宏观审慎措施；保险组合中 "
                "REITs 板块对此最敏感。"
            ),
            source="URA / Street estimates",
            tags=["SG", "property", "REITs"],
        ),
        FeedItem(
            title="10Y UST yield 维持在 4.25% 附近",
            category="rates_macro",
            summary=(
                "美债收益率本周窄幅波动，市场继续消化 FOMC 纪要偏鸽信号；"
                "SOFR 曲线隐含 2026 年内仍有 2 次降息。"
            ),
            why_it_matters=(
                "长端利率若跌破 4.10%，寿险产品再投资收益率承压；若上破 4.45%，"
                "REITs 和成长股首当其冲。"
            ),
            source="Bloomberg / CME FedWatch",
            tags=["rates", "UST", "macro"],
        ),
        FeedItem(
            title="STI 年初至今跑赢 HSI 约 400bp",
            category="equities",
            summary=(
                "新加坡本地银行（DBS/OCBC/UOB）在高息差环境下继续贡献指数，"
                "港股则受科网板块拖累。"
            ),
            why_it_matters=(
                "本地三大行占 STI 权重约 45%，利率曲线走势决定后续是否还能跑赢。"
            ),
            source="SGX / HKEX",
            tags=["STI", "HSI", "banks"],
        ),
        FeedItem(
            title=f"本周关键数据窗口 ({today})",
            category="event",
            summary=(
                "周三：美国 CPI；周四：MAS 半年度货币政策声明；周五："
                "新加坡 Q1 GDP 预估值。"
            ),
            why_it_matters=(
                "三天内会连续重定价利率+汇率预期，适合提前梳理组合的利率/汇率敞口。"
            ),
            tags=["calendar", "SG", "US"],
        ),
    ]


def main(mode: str = "real", dry_run: bool = False) -> None:
    """
    Entry point. `mode` selects the item source:
      - "real" (default): news pipeline + community analyst
      - "mock": static sample for smoke-testing
    `dry_run=True` prints instead of posting to Discord.
    """
    logging.basicConfig(level=logging.INFO)

    if mode == "mock":
        items = _mock_items()
    else:
        items = build_real_items()

    if not items:
        print("No feed items to publish.")
        return

    if dry_run:
        for it in items:
            print("─" * 60)
            print(format_item(it))
        print("─" * 60)
        print(f"[dry-run] would publish {len(items)} items")
        return

    sent = publish_items(items)
    print(f"Published {sent}/{len(items)} items to #prediction-markets")


if __name__ == "__main__":
    import sys

    mode = "real"
    dry_run = False
    for arg in sys.argv[1:]:
        if arg in ("mock", "real"):
            mode = arg
        elif arg in ("--dry-run", "-n"):
            dry_run = True

    main(mode=mode, dry_run=dry_run)
