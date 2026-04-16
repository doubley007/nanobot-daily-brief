from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from news_sources.alpha_vantage_source import fetch_from_alpha_vantage
from news_sources.finnhub_source import fetch_from_finnhub
from news_sources.fmp_source import (
    fetch_from_fmp_general_news,
    fetch_from_fmp_stock_news,
)


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
    "mas": 5,
    "singapore": 4,
    "sgd": 4,
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
    "sanctions": 3,
    "tariff": 3,
}

LOW_VALUE_KEYWORDS = {
    "pre-market": -3,
    "stock moved up": -3,
    "stock moved down": -3,
    "drivers behind the movement": -3,
    "price target": -2,
    "buy now": -4,
    "top stock": -4,
    "analyst says": -2,
    "upgrade": -1,
    "downgrade to sell": -1,
    "top pick": -2,
    "shares rose": -1,
    "shares fell": -1,
    "etf": -2,
    "valuation": -2,
    "upside": -2,
    "what investors need to know": -3,
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
}

EXCLUDE_KEYWORDS = [
    "opinion",
    "sponsored",
    "advertisement",
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
        summary=(item.get("summary") or "暂无摘要").strip(),
        source=(item.get("source") or "Unknown").strip(),
        category=(item.get("category") or "general").strip(),
        url=item.get("url"),
        published_at=item.get("published_at"),
    )


def _is_excluded(item: RawNewsItem) -> bool:
    text = f"{item.title} {item.summary}".lower()
    return any(keyword in text for keyword in EXCLUDE_KEYWORDS)


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


def _deduplicate_by_topic(items: list[RawNewsItem]) -> list[RawNewsItem]:
    best_by_topic: dict[str, RawNewsItem] = {}

    for item in items:
        topic = _detect_topic(item)
        if topic not in best_by_topic:
            best_by_topic[topic] = item
            continue

        if _score_item(item) > _score_item(best_by_topic[topic]):
            best_by_topic[topic] = item

    return list(best_by_topic.values())


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

    sources = {
        "finnhub": lambda: fetch_from_finnhub(limit=20, timeout=timeout),
        "fmp_stock": lambda: fetch_from_fmp_stock_news(limit=20, timeout=timeout),
        "fmp_general": lambda: fetch_from_fmp_general_news(limit=20, timeout=timeout),
        "alpha_vantage": lambda: fetch_from_alpha_vantage(limit=20, timeout=timeout),
    }

    raw_dict_items: list[dict] = []

    with ThreadPoolExecutor(max_workers=4) as pool:
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
    for item in raw_dict_items:
        normalized = _normalize_item(item)
        if normalized is None:
            continue
        if _is_excluded(normalized):
            continue
        if _is_low_value_single_stock_story(normalized):
            continue
        normalized_items.append(normalized)

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