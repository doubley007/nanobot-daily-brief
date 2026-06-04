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
    "sg_property": "SG Property",
    "rates_macro": "Rates / Macro",
    "equities": "US / HK / SG Equities",
    "event": "Key Event",
    "general": "General",
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
        lines.append(f"> Why it matters: {item.why_it_matters}")

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
            title=f"Community buzz: {headline}",
            category=category,
            summary=t.discussion_focus or "",
            why_it_matters=t.market_relevance or t.insurance_angle or "",
            source=f"community · {'/'.join(t.platforms)}",
            tags=["community"] + list(t.platforms),
        ))
    return items


def build_real_items(news_limit: int = 10) -> list[FeedItem]:
    """Assemble today's real-source feed: news first, then community."""
    return (
        _news_to_feed_items(limit=news_limit)
        + _community_to_feed_items()
        + _polymarket_to_feed_items()
    )


# ─── Polymarket prediction-market source ─────────────────────────────────────
#
# Polymarket's public Gamma API needs no auth. We pull the top-volume active
# markets and keep only ones that (a) touch our SG-insurer priority themes,
# (b) carry real uncertainty (price in 0.05-0.95). Celebrity/novelty
# questions at 0.99/0.01 add no signal even when they pass keyword filters.

_POLYMARKET_ENDPOINT = "https://gamma-api.polymarket.com/markets"
_POLYMARKET_TIMEOUT = 15
_POLYMARKET_LIMIT = 300

_PM_KEYWORDS_MACRO = (
    "fed", "interest rate", "rate cut", "rate hike", "cpi", "inflation",
    "recession", "gdp", "treasury", "yield", "fomc", "jerome powell", "powell",
)
_PM_KEYWORDS_GEO = (
    "iran", "russia", "ukraine", "israel", "tariff", "china", "hong kong",
    "taiwan",
)
_PM_KEYWORDS_SG = ("singapore", "sgd", "mas ")
_PM_KEYWORDS_COMMOD = ("oil", "opec", "brent", "crude", "wti")
_PM_KEYWORDS_BANK = (
    "bank failure", "credit spread", "regional bank", "bank run",
)
_PM_ALL_KEYWORDS = (
    _PM_KEYWORDS_MACRO + _PM_KEYWORDS_GEO + _PM_KEYWORDS_SG
    + _PM_KEYWORDS_COMMOD + _PM_KEYWORDS_BANK
)

# Reject generic politics/celebrity/novelty markets. Polymarket's top volume
# is dominated by 2028 celebrity-election questions that mean nothing for a
# SG insurer; blacklisting specific names stays maintainable.
_PM_REJECT_KEYWORDS = (
    "2028", "kardashian", "lebron", "oprah", "mrbeast", "beto", "clinton",
    "bernie", "pence", "walz", "phil murphy", "ramaswamy", "gabbard",
    "vivek", "greg abbott", "cheney", "stephen smith", "tulsi", "clooney",
    "pope", "mars", "gta vi", "taylor swift", "nba champion", "super bowl",
    "oscar", "grammy", "eurovision",
)

_PM_UNCERTAINTY_MIN = 0.05
_PM_UNCERTAINTY_MAX = 0.95
_PM_MAX_ITEMS = 5  # cap so the feed doesn't drown the daily brief


def _pm_categorize(question: str) -> str:
    """Map a Polymarket question to a feed category."""
    q = question.lower()
    if any(k in q for k in _PM_KEYWORDS_SG):
        return "sg_property"
    if any(k in q for k in _PM_KEYWORDS_MACRO):
        return "rates_macro"
    # Geopolitics and oil both show up as "event" — they're not directly
    # rates/macro but they drive risk premia and commodity channels.
    return "event"


def _pm_uncertain(outcome_prices) -> bool:
    """True when the YES price is in the uncertainty band."""
    if not outcome_prices:
        return False
    try:
        import json as _json
        prices = (
            _json.loads(outcome_prices)
            if isinstance(outcome_prices, str)
            else outcome_prices
        )
        if not prices:
            return False
        yes_price = float(prices[0])
    except (ValueError, TypeError):
        return False
    return _PM_UNCERTAINTY_MIN <= yes_price <= _PM_UNCERTAINTY_MAX


def _pm_format_summary(market: dict) -> str:
    """One-line market summary with YES price + volume + end date."""
    import json as _json
    try:
        prices = _json.loads(market.get("outcomePrices") or "[]")
        yes = float(prices[0]) if prices else None
    except (ValueError, TypeError):
        yes = None

    volume = float(market.get("volume") or 0)
    end_date = (market.get("endDate") or "")[:10]

    bits: list[str] = []
    if yes is not None:
        bits.append(f"YES implied prob {yes * 100:.0f}%")
    if volume:
        bits.append(f"volume ${volume / 1_000_000:.1f}M")
    if end_date:
        bits.append(f"ends {end_date}")
    return " · ".join(bits)


def _fetch_polymarket_markets() -> list[dict]:
    """Fetch top-volume active markets from Polymarket Gamma."""
    import requests

    try:
        resp = requests.get(
            _POLYMARKET_ENDPOINT,
            params={
                "limit": _POLYMARKET_LIMIT,
                "active": "true",
                "closed": "false",
                "order": "volumeNum",
                "ascending": "false",
            },
            timeout=_POLYMARKET_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json() or []
    except Exception as e:
        logger.warning("Polymarket fetch failed: %s", e)
        return []


def _polymarket_to_feed_items() -> list[FeedItem]:
    """Convert Polymarket's top macro/geo markets into FeedItems."""
    markets = _fetch_polymarket_markets()
    if not markets:
        return []

    items: list[FeedItem] = []
    for m in markets:
        question = (m.get("question") or "").strip()
        if not question:
            continue
        q_lower = question.lower()
        if any(r in q_lower for r in _PM_REJECT_KEYWORDS):
            continue
        if not any(k in q_lower for k in _PM_ALL_KEYWORDS):
            continue
        if not _pm_uncertain(m.get("outcomePrices")):
            continue

        category = _pm_categorize(question)
        summary = _pm_format_summary(m)

        # Why-it-matters is a short insurer-facing take. We don't call LLM
        # here — the brief's community analyst will interpret this item
        # once it lands on the Discord channel.
        why_parts = []
        if category == "rates_macro":
            why_parts.append("A shift in rates / macro-path probabilities feeds directly into fixed-income reinvestment yields.")
        elif category == "event":
            why_parts.append("Geopolitical / commodity event probabilities shift risk premia and commodity exposure.")
        elif category == "sg_property":
            why_parts.append("Direct bearing on local property / SG market exposure.")
        why = " · ".join(why_parts)

        slug = m.get("slug") or ""
        url = f"https://polymarket.com/market/{slug}" if slug else ""

        items.append(FeedItem(
            title=question,
            category=category,
            summary=summary,
            why_it_matters=why,
            source="Polymarket",
            url=url,
            tags=["polymarket", category],
        ))

        if len(items) >= _PM_MAX_ITEMS:
            break

    return items


# ─── Mock batch (Step 2, kept for smoke-testing) ─────────────────────────────

def _mock_items() -> list[FeedItem]:
    today = datetime.now().strftime("%Y-%m-%d")
    return [
        FeedItem(
            title="URA Q1 private home price index preview",
            category="sg_property",
            summary=(
                "Street consensus expects Q1 private home prices flat to +0.3% QoQ; "
                "CCR luxury launches and OCR new-launch pipeline are the key swing factors."
            ),
            why_it_matters=(
                "An upside surprise could keep MAS macroprudential measures in place; "
                "within the insurance book, REITs are the most sensitive sleeve."
            ),
            source="URA / Street estimates",
            tags=["SG", "property", "REITs"],
        ),
        FeedItem(
            title="10Y UST yield holding near 4.25%",
            category="rates_macro",
            summary=(
                "Treasury yields are range-bound this week as the market digests dovish FOMC-minutes signals; "
                "the SOFR curve still prices 2 cuts by end-2026."
            ),
            why_it_matters=(
                "A break below 4.10% at the long end pressures life-product reinvestment yields; "
                "a break above 4.45% hits REITs and growth equities first."
            ),
            source="Bloomberg / CME FedWatch",
            tags=["rates", "UST", "macro"],
        ),
        FeedItem(
            title="STI YTD outperforming HSI by ~400bp",
            category="equities",
            summary=(
                "SG local banks (DBS/OCBC/UOB) keep contributing to the index in a higher-NIM regime; "
                "HK equities are weighed down by tech/internet names."
            ),
            why_it_matters=(
                "The three SG banks are roughly 45% of STI weight — the rate-curve trajectory decides whether the outperformance continues."
            ),
            source="SGX / HKEX",
            tags=["STI", "HSI", "banks"],
        ),
        FeedItem(
            title=f"Key data window this week ({today})",
            category="event",
            summary=(
                "Wed: US CPI. Thu: MAS semi-annual monetary policy statement. "
                "Fri: Singapore Q1 GDP advance estimate."
            ),
            why_it_matters=(
                "Three back-to-back data points that will re-price rates and FX expectations — "
                "worth lining up the book's rate / FX exposures ahead of time."
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
