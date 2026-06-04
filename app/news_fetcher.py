from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from news_sources.alpha_vantage_source import fetch_from_alpha_vantage
from news_sources.finnhub_source import fetch_from_finnhub
from news_sources.fmp_source import (
    fetch_from_fmp_general_news,
    fetch_from_fmp_stock_news,
)

logger = logging.getLogger(__name__)


# =========================================================
# Source-normalized raw item
# =========================================================

@dataclass
class RawNewsItem:
    title: str
    summary: str
    source: str
    category: str
    url: Optional[str] = None
    published_at: Optional[str] = None


# =========================================================
# Insurance / portfolio relevant keywords
# =========================================================

# Scored for the Great Eastern SG insurance use case: rates / credit /
# Asia + SG / insurance-regulation lead; US single-stock equity lags.
INSURANCE_RELEVANT_KEYWORDS = {
    "fed": 5,
    "federal reserve": 5,
    "treasury": 5,
    "yield": 4,
    "yields": 4,
    "bond": 3,
    "bonds": 3,
    "cpi": 4,
    "inflation": 4,
    "payroll": 4,
    "employment": 3,
    "default": 5,
    "downgrade": 5,
    "credit": 5,
    "spread": 4,
    "spreads": 4,
    "liquidity": 4,
    "refinancing": 4,
    "bank": 3,
    "banks": 3,
    "insurer": 4,
    "insurance": 4,
    # Singapore / Asia bias — Great Eastern's actual book
    "mas": 7,
    "singapore": 6,
    "sgd": 6,
    "sti": 4,
    "asia": 3,
    "hong kong": 3,
    "malaysia": 3,
    "indonesia": 3,
    "asean": 4,
    "great eastern": 6,
    "ocbc": 4,
    "dbs": 4,
    "uob": 4,
    "usd": 3,
    "fx": 3,
    "foreign exchange": 3,
    "real estate": 4,
    "commercial real estate": 5,
    "property": 3,
    "corporate loans": 5,
    "private equity": 4,
    "buyout": 3,
    "valuation markdown": 4,
    "regulation": 4,
    "capital requirement": 5,
    "solvency": 5,
    # Insurance-book-specific terms
    "life insurance": 5,
    "par fund": 5,
    "participating fund": 5,
    "with-profits": 4,
    "annuity": 4,
    "reinsurance": 4,
    "policyholder": 3,
    "sanctions": 3,
    "tariff": 3,
}

LOW_VALUE_KEYWORDS = {
    "pre-market": -3,
    "stock moved up": -3,
    "stock moved down": -3,
    "drivers behind the movement": -3,
    "price target": -3,
    "buy now": -5,
    "top stock": -5,
    "analyst says": -3,
    "analyst upgrade": -3,
    "analyst downgrade": -3,
    "upgrade": -2,
    "downgrade to sell": -1,
    "top pick": -3,
    "shares rose": -1,
    "shares fell": -1,
    "etf": -2,
    "valuation": -2,
    "upside": -2,
    "what investors need to know": -4,
    # Single-stock technical / ratings noise
    "ai downgrades": -6,
    "ai downgrade": -6,
    "ai upgrades": -6,
    "ai upgrade": -6,
    "tipranks": -5,
    "technical pressure": -4,
    "moving average": -3,
    "resistance level": -3,
    "support level": -3,
    "overbought": -3,
    "oversold": -3,
    "short interest": -3,
    "why is": -3,
    "why are": -3,
    "stock surging": -3,
    "stock jumps": -3,
    "stock soars": -3,
    "stock plunges": -3,
}

NEGATIVE_NOISE_KEYWORDS = {
    "probe": -3,
    "appeal": -3,
    "lawsuit": -3,
    "nominee": -2,
    "celebrity": -4,
    "gossip": -4,
    "ai model": -1,
    "spending plans": -1,
    "meta": -1,
    "peace": -1,
    "relevant parties": -2,
    "hopes": -1,
    "peace talks": -2,
    "diplomatic": -2,
    "statement": -1,
    # Retail / listicle noise
    "here's why": -3,
    "here is why": -3,
    "5 stocks": -4,
    "3 stocks": -4,
    "10 stocks": -4,
    "these stocks": -3,
    "dividend stock": -3,
    "growth stock": -3,
    "value stock": -3,
    "penny stock": -4,
    "retirement": -2,
    "portfolio builder": -3,
    "passive income": -3,
    "how to invest": -4,
    "warren buffett": -2,
    "cathie wood": -3,
    "jim cramer": -3,
    "motley fool": -3,
    "zacks rank": -4,
    "zacks investment": -4,
}

EXCLUDE_KEYWORDS = [
    "opinion",
    "sponsored",
    "advertisement",
    "advertorial",
    "press release",
]

PREFERRED_SOURCES = {
    "Reuters": 3,
    "Bloomberg": 3,
    "Financial Times": 3,
    "The Wall Street Journal": 3,
    "WSJ": 3,
    "Barrons": 2,
    "MarketWatch": 2,
    "CNBC": 1,
    "Alpha Vantage": 1,
    # SG / Asia local sources
    "Business Times SG": 3,
    "Business Times Banking": 3,
    "Business Times Economy": 3,
    "CNA Business": 2,
    "Straits Times Business": 2,
    "Finnhub": 1,
}

REGION_PENALTY_KEYWORDS = {
    "india": -2,
    "new zealand": -2,
    "australia": -1,
}

CORE_BUCKETS = [
    "fixed_income_macro",
    "credit",
    "fx_liquidity",
    "singapore_local",
    "regulation",
    "real_estate_loans",
    "private_equity",
    "listed_equity",
]


# =========================================================
# Normalization
# =========================================================

def _normalize_item(item: dict) -> Optional[RawNewsItem]:
    title = (item.get("title") or "").strip()
    if not title:
        return None

    return RawNewsItem(
        title=title,
        summary=(item.get("summary") or "no summary available").strip(),
        source=(item.get("source") or "Unknown").strip(),
        category=(item.get("category") or "general").strip(),
        url=item.get("url"),
        published_at=item.get("published_at"),
    )


def _is_excluded(item: RawNewsItem) -> bool:
    text = f"{item.title} {item.summary}".lower()
    return any(keyword in text for keyword in EXCLUDE_KEYWORDS)


# =========================================================
# Headline ↔ body directional consistency check
# =========================================================
#
# Goal: drop items whose headline and summary clearly disagree on the
# direction of the story (e.g. "surging" in the title, "slumped" in the
# body). This is a lightweight guard — we only catch unambiguous
# conflicts and leave nuanced cases to downstream ranking.

_POSITIVE_DIRECTION_WORDS = {
    "surge", "surges", "surging", "surged",
    "rally", "rallies", "rallying", "rallied",
    "jump", "jumps", "jumped", "jumping",
    "soar", "soars", "soaring", "soared",
    "gain", "gains", "gaining", "gained",
    "climb", "climbs", "climbing", "climbed",
    "rise", "rises", "rising", "rose",
    "up", "upbeat", "higher",
    "beat", "beats", "beating",
    "boost", "boosts", "boosted",
    "outperform", "outperforms", "outperformed",
}

_NEGATIVE_DIRECTION_WORDS = {
    "fall", "falls", "falling", "fell",
    "drop", "drops", "dropping", "dropped",
    "plunge", "plunges", "plunging", "plunged",
    "sink", "sinks", "sinking", "sank", "sunk",
    "slump", "slumps", "slumping", "slumped",
    "tumble", "tumbles", "tumbling", "tumbled",
    "slide", "slides", "sliding", "slid",
    "crash", "crashes", "crashing", "crashed",
    "decline", "declines", "declining", "declined",
    "lower", "downbeat",
    "miss", "misses", "missed",
    "cut", "cuts", "cutting",
    "weaken", "weakens", "weakened", "weakening",
    "underperform", "underperforms", "underperformed",
}


def _tokens(text: str) -> set[str]:
    cleaned = []
    for ch in text.lower():
        cleaned.append(ch if ch.isalnum() else " ")
    return set("".join(cleaned).split())


def _direction_of(tokens: set[str]) -> str:
    has_pos = bool(tokens & _POSITIVE_DIRECTION_WORDS)
    has_neg = bool(tokens & _NEGATIVE_DIRECTION_WORDS)
    if has_pos and not has_neg:
        return "up"
    if has_neg and not has_pos:
        return "down"
    if has_pos and has_neg:
        return "mixed"
    return "none"


def _has_headline_body_conflict(item: RawNewsItem) -> bool:
    """
    Return True when the headline clearly says one direction but the
    summary/body clearly says the opposite. Mixed-signal cases don't
    trip the filter — we only flag clean up-vs-down conflicts.
    """
    title_dir = _direction_of(_tokens(item.title))
    body = item.summary or ""
    if not body or body == "no summary available":
        return False
    body_dir = _direction_of(_tokens(body))

    if title_dir == "up" and body_dir == "down":
        return True
    if title_dir == "down" and body_dir == "up":
        return True
    return False


def _is_market_relevant_equity_story(item: RawNewsItem) -> bool:
    text = f"{item.title} {item.summary}".lower()

    market_event_keywords = [
        "earnings miss",
        "earnings beat",
        "guidance cut",
        "guidance raised",
        "sector selloff",
        "broad market",
        "index rebalance",
        "systemic",
        "capital raise",
        "liquidity stress",
        "rating downgrade",
        "rating cut",
        "credit loss",
        "loan loss",
        "sector pressure",
        "bank stress",
        "insurance regulation",
        "capital requirement",
    ]

    return any(k in text for k in market_event_keywords)


def _is_low_value_single_stock_story(item: RawNewsItem) -> bool:
    text = f"{item.title} {item.summary}".lower()
    title_lower = item.title.lower()

    # 先做标题级硬过滤，优先级最高
    hard_reject_title_patterns = [
        "how investors may respond to",
        "stock could soar",
        "stock could jump",
        "stock could rally",
        "stock could surge",
        "what investors need to know",
        "drivers behind the movement",
        "unlock new upside",
        "does its",
        "is its",
        "a high-wire act",
        "the real test amid rising risks",
        # ADR / ticker single-stock format: "Company (ADR)", "Company stock (TICKER):"
        "(adr)",
        "stock (br",   # Brazilian ADR tickers e.g. BRGGBRACNPR7
        "stock (us",
        "stock (hk",
        "stock (sg",
        # "Why X weigh on shares / stock" — classic single-stock explanation
        "weigh on shares",
        "weigh on stock",
        "weighs on shares",
        "weighs on stock",
        "faces headwinds from",
        "stock faces headwinds",
        "faces uncertainty amid",
        "stock faces uncertainty",
        # Single-stock ratings/technical noise
        "ai downgrades",
        "ai downgrade",
        "ai upgrades",
        "ai upgrade",
        "technical pressure offsets",
        "solid fundamentals",
        "strong fundamentals",
        "why is ",
        "why are ",
        "why does ",
        "why did ",
        "why this",
        "why that",
        "tipranks",
        "moving average",
        "moving averages",
        "resistance level",
        "support level",
        "key resistance",
        "key support",
        "relative strength",
        "overbought",
        "oversold",
        "price target",
        "stock picks",
        "top picks",
        "stock to buy",
        "stock to watch",
        "stock to own",
        "best stock",
        "worst stock",
        "analyst downgrade",
        "analyst upgrade",
        "raises price target",
        "cuts price target",
        "short interest",
    ]
    if any(p in title_lower for p in hard_reject_title_patterns):
        return True

    noisy_patterns = [
        "stock (",
        "stock:",
        "shares rose",
        "shares fell",
        "price target",
        "double downgrades",
        "buy now",
        "top stock",
        "all-in-one hcm",
        "moved up by",
        "moved down by",
        "pre-market",
        "why the movement",
        "edge hold up",
        "dips in pre-market",
        "weigh on goog",
        "maintains its dominance",
        "focused on innovation and efficiency",
        "driven by underwriting",
        "direct-to-consumer model",
        "machine learning investment",
        "could soar",
        "could jump",
        "could rally",
        "could surge",
        "undervalued",
        "overvalued",
        "bull case",
        "bear case",
        "workflow launch",
        "ai workflow launch",
        "fedramp",
        "approved ai",
        "for federal agencies",
        "launch for federal agencies",
        "may respond to",
        "launch for",
        "strengthens its position",
        "government market position",
    ]

    protected_keywords = [
        "bank stress",
        "bank failure",
        "insurance regulation",
        "treasury",
        "yield",
        "federal reserve",
        "cpi",
        "inflation",
        "mas",
        "singapore",
        "credit",
        "default",
        "downgrade of",
        "real estate",
        "commercial real estate",
        "private equity",
        "solvency",
        "capital requirement",
        "fx",
        "sgd",
        "usd",
        "corporate loans",
        "liquidity",
        "refinancing",
        "sanctions",
        "spread widening",
        # GE / OCBC group / SG insurance competitors — never drop these
        "great eastern",
        "ocbc",
        "oversea-chinese banking",
        "prudential singapore",
        "aia singapore",
        "income insurance",
        "ntuc income",
        "singlife",
        "aviva singapore",
        "manulife singapore",
        "tokio marine singapore",
        "fwd singapore",
        "medical inflation",
        "bancassurance",
        "rbc 2",
        "rbc2",
        "integrated shield",
        "par fund",
    ]

    if any(k in text for k in protected_keywords):
        return False

    if any(p in text for p in noisy_patterns):
        return True

    company_suffixes = [" inc", " corp", " n.v", " plc", " ltd", " holdings"]
    if any(suffix in text for suffix in company_suffixes):
        if any(k in text for k in ["stock", "shares", "upside", "valuation", "buy", "sell"]):
            return True

    if "(" in item.title and ")" in item.title:
        if any(k in text for k in ["launch", "workflow", "investors may respond", "stock", "shares"]):
            return True

    return False


# =========================================================
# Portfolio impact buckets
# =========================================================

def detect_portfolio_impact_bucket(item: RawNewsItem) -> str:
    text = f"{item.title} {item.summary}".lower()

    # TIER-1: GE / OCBC group / direct SG insurance competitors — highest priority
    # Must be checked first so "Great Eastern Q1 profit" doesn't get routed into
    # macro/general buckets by incidental keyword matches.
    if any(
        k in text
        for k in [
            "great eastern",
            "ocbc group",
            "ocbc wealth",
            "ocbc insurance",
            "oversea-chinese banking",
            "prudential singapore",
            "prudential assurance",
            "aia singapore",
            "income insurance",
            "ntuc income",
            "singlife",
            "aviva singapore",
            "manulife singapore",
            "tokio marine singapore",
            "fwd singapore",
            "life insurance association",
            "integrated shield",
            "mas insurance",
            "rbc 2",
            "rbc2",
        ]
    ):
        return "singapore_insurer"

    if any(
        k in text
        for k in [
            "treasury",
            "yield",
            "yields",
            "fed",
            "federal reserve",
            "cpi",
            "inflation",
            "payroll",
            "employment",
            "bond",
            "bonds",
            "interest rate",
            "rate cut",
            "rate cuts",
        ]
    ):
        return "fixed_income_macro"

    if any(
        k in text
        for k in [
            "default",
            "downgrade",
            "credit",
            "spread",
            "spreads",
            "liquidity",
            "refinancing",
        ]
    ):
        return "credit"

    if any(
        k in text
        for k in [
            "usd",
            "sgd",
            "fx",
            "foreign exchange",
            "mas",
            "singapore dollar",
        ]
    ):
        return "fx_liquidity"

    if any(
        k in text
        for k in [
            "real estate",
            "commercial real estate",
            "property",
            "office market",
            "corporate loans",
        ]
    ):
        return "real_estate_loans"

    if any(
        k in text
        for k in [
            "private equity",
            "buyout",
            "fundraising",
            "valuation markdown",
        ]
    ):
        return "private_equity"

    # 只允许真正“市场相关”的银行/保险/指数类 equity story 进入
    if any(k in text for k in ["bank", "banks", "insurance", "insurer"]):
        if _is_market_relevant_equity_story(item):
            return "listed_equity"

    if any(k in text for k in ["equity market", "index", "indexes", "s&p 500", "nasdaq", "dow", "stoxx", "msci world"]):
        return "listed_equity"

    if any(
        k in text
        for k in [
            "regulation",
            "capital requirement",
            "solvency",
            "compliance",
            "sanctions",
            "tariff",
        ]
    ):
        return "regulation"

    if any(k in text for k in ["singapore", "mas"]):
        return "singapore_local"

    return "general"


# =========================================================
# Topic detection for secondary dedup
# =========================================================

TOPIC_KEYWORDS = {
    "fed_rates": ["fed", "federal reserve", "rate cut", "rate cuts"],
    "inflation": ["cpi", "inflation"],
    "treasury_yields": ["treasury", "yield", "yields", "bond"],
    "credit_risk": ["default", "downgrade", "credit", "spread", "liquidity"],
    "banking": ["bank", "banks", "lender", "lenders"],
    "insurance": ["insurer", "insurance"],
    "singapore": ["mas", "singapore"],
    "real_estate": ["real estate", "commercial real estate", "property"],
    "private_equity": ["private equity", "buyout", "fundraising"],
    "regulation": ["regulation", "capital requirement", "solvency", "compliance"],
}


def _detect_topic(item: RawNewsItem) -> str:
    text = f"{item.title} {item.summary}".lower()

    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                return topic

    return "general"


# =========================================================
# Scoring
# =========================================================

def _score_item(item: RawNewsItem) -> int:
    text = f"{item.title} {item.summary}".lower()
    score = 0

    for keyword, weight in INSURANCE_RELEVANT_KEYWORDS.items():
        if keyword in text:
            score += weight

    for keyword, penalty in LOW_VALUE_KEYWORDS.items():
        if keyword in text:
            score += penalty

    for keyword, penalty in NEGATIVE_NOISE_KEYWORDS.items():
        if keyword in text:
            score += penalty

    for keyword, penalty in REGION_PENALTY_KEYWORDS.items():
        if keyword in text:
            score += penalty

    score += PREFERRED_SOURCES.get(item.source, 0)

    bucket = detect_portfolio_impact_bucket(item)
    bucket_bonus = {
        "fixed_income_macro": 4,
        "credit": 4,
        "fx_liquidity": 3,
        "real_estate_loans": 4,
        "private_equity": 3,
        "listed_equity": 1,
        "regulation": 4,
        "singapore_local": 4,
        "general": -6,
    }
    score += bucket_bonus.get(bucket, 0)

    return score


# =========================================================
# Dedup
# =========================================================

def _deduplicate_exact(items: list[RawNewsItem]) -> list[RawNewsItem]:
    seen = set()
    results: list[RawNewsItem] = []

    for item in items:
        normalized_title = item.title.strip().lower()
        if normalized_title in seen:
            continue
        seen.add(normalized_title)
        results.append(item)

    return results


def _deduplicate_by_topic(
    items: list[RawNewsItem],
    max_per_topic: int = 8,
) -> list[RawNewsItem]:
    """
    Keep the top-scoring items per topic (sorted by score DESC).
    max_per_topic=8 gives the RAG store enough coverage per topic while
    still culling obvious duplicates.  The original behaviour (keep 1) is
    preserved for the brief's select_diverse_by_bucket step which runs
    after this and enforces its own diversity limit.
    """
    from collections import defaultdict
    by_topic: dict[str, list[RawNewsItem]] = defaultdict(list)

    for item in items:
        topic = _detect_topic(item)
        by_topic[topic].append(item)

    results: list[RawNewsItem] = []
    for topic_items in by_topic.values():
        topic_items.sort(key=_score_item, reverse=True)
        results.extend(topic_items[:max_per_topic])

    return results


# =========================================================
# Final selection by portfolio bucket diversity
# =========================================================

def select_diverse_by_bucket(items: list[RawNewsItem], top_n: int = 3) -> list[RawNewsItem]:
    selected: list[RawNewsItem] = []
    used_buckets = set()

    # First pass: fixed order on core buckets
    for core_bucket in CORE_BUCKETS:
        for item in items:
            if item in selected:
                continue
            bucket = detect_portfolio_impact_bucket(item)
            if bucket != core_bucket:
                continue
            if bucket in used_buckets:
                continue

            selected.append(item)
            used_buckets.add(bucket)

            if len(selected) >= top_n:
                return selected

    # Second pass: other non-general buckets
    for item in items:
        if item in selected:
            continue
        bucket = detect_portfolio_impact_bucket(item)
        if bucket == "general":
            continue
        if bucket in used_buckets:
            continue

        selected.append(item)
        used_buckets.add(bucket)

        if len(selected) >= top_n:
            return selected

    # Third pass: fill remaining slots if necessary
    for item in items:
        if item in selected:
            continue
        selected.append(item)
        if len(selected) >= top_n:
            return selected

    return selected


# =========================================================
# Main fetch
# =========================================================

def fetch_financial_news(
    limit: int = 10,
    timeout: int = 15,
) -> list[RawNewsItem]:
    """
    Multi-source aggregator for insurance-investment-relevant news.

    Fetches from all 4 sources in parallel via ThreadPoolExecutor,
    then applies: exact dedup → low-value filter → topic dedup →
    score → select by portfolio impact bucket diversity.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from news_sources.rss_source import fetch_from_rss

    sources = {
        "finnhub": lambda: fetch_from_finnhub(limit=20, timeout=timeout),
        "fmp_stock": lambda: fetch_from_fmp_stock_news(limit=20, timeout=timeout),
        "fmp_general": lambda: fetch_from_fmp_general_news(limit=20, timeout=timeout),
        "alpha_vantage": lambda: fetch_from_alpha_vantage(limit=20, timeout=timeout),
        "rss": lambda: fetch_from_rss(limit=120, timeout=timeout),
    }

    raw_dict_items: list[dict] = []

    with ThreadPoolExecutor(max_workers=5) as pool:
        future_to_name = {pool.submit(fn): name for name, fn in sources.items()}
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                raw_dict_items.extend(future.result())
            except Exception as e:
                print(f"WARNING {name} fetch failed: {e}")

    if not raw_dict_items:
        return []

    normalized_items: list[RawNewsItem] = []
    conflict_filtered = 0
    for item in raw_dict_items:
        normalized = _normalize_item(item)
        if normalized is None:
            continue
        if _is_excluded(normalized):
            continue
        if _is_low_value_single_stock_story(normalized):
            continue
        if _has_headline_body_conflict(normalized):
            conflict_filtered += 1
            logger.info(
                "news_fetcher: dropping headline/body-inconsistent item — title=%r",
                normalized.title,
            )
            continue
        normalized_items.append(normalized)

    if conflict_filtered:
        # Mirror the module-level print pattern used for fetch warnings
        # so the count shows up in daily_job stdout/log, not only in the
        # file-logger.
        print(
            f"INFO news_fetcher: headline/body consistency filter dropped "
            f"{conflict_filtered} item(s)"
        )
        logger.info(
            "news_fetcher: headline/body consistency filter dropped %d item(s)",
            conflict_filtered,
        )

    if not normalized_items:
        return []

    normalized_items = _deduplicate_exact(normalized_items)
    normalized_items.sort(key=_score_item, reverse=True)
    normalized_items = _deduplicate_by_topic(normalized_items)
    normalized_items.sort(key=_score_item, reverse=True)

    selected_items = select_diverse_by_bucket(normalized_items, top_n=limit)
    return selected_items


# =========================================================
# Local test
# =========================================================

if __name__ == "__main__":
    news = fetch_financial_news(limit=10)
    print(f"Fetched {len(news)} items.")
    for idx, item in enumerate(news, start=1):
        print(f"{idx}. {item.title}")
        print(f"   source={item.source}")
        print(f"   bucket={detect_portfolio_impact_bucket(item)}")
        print(f"   topic={_detect_topic(item)}")
        print(f"   score={_score_item(item)}")
        print(f"   summary={item.summary[:160]}")
        print(f"   url={item.url}")
        print()