"""
Shared rule-based logic for community sentiment sources.

Interpretation (what a topic means, sentiment reasoning, market relevance)
lives entirely in community.llm_analyst. This module keeps only the cheap,
deterministic plumbing:

  - finance keyword list (pre-screen)
  - topic rule buckets (coarse grouping before LLM)
  - engagement-weighted keyword sentiment (used only as a fallback hint
    when the LLM is unavailable)
"""
from __future__ import annotations


# ─── Finance keyword list (shared across all sources) ────────────────────────

DEFAULT_FINANCE_KEYWORDS = [
    "stock", "stocks", "market", "markets", "invest", "trading",
    "portfolio", "s&p", "nasdaq", "dow", "index", "etf",
    "dividend", "hedge", "fund", "asset", "equity", "equities",
    "debt", "loan", "credit", "spread", "liquidity",
    "fed", "fomc", "rate", "rates", "yield", "yields", "bond", "bonds",
    "treasury", "inflation", "cpi", "gdp", "recession", "payroll",
    "earnings", "revenue", "profit", "guidance", "eps",
    "oil", "crude", "opec", "energy", "gold", "commodity",
    "tariff", "trade war", "sanctions",
    "china", "singapore", "mas",
    "bitcoin", "crypto", "ethereum",
    "bank", "banks", "insurance", "insurer", "mortgage",
    "real estate", "housing", "reits",
    "semiconductor", "chip", "chips", "nvidia", "tsmc",
    "ipo", "buyback", "m&a", "merger", "acquisition",
    "volatility", "vix", "options", "futures", "short", "puts", "calls",
    "bull", "bear", "rally", "crash", "selloff", "correction",
]


# ─── Topic detection (scored, multi-keyword) ─────────────────────────────────

TOPIC_RULES: dict[str, list[tuple[str, int]]] = {
    "美联储与利率政策": [
        ("federal reserve", 5), ("fomc", 5), ("rate cut", 4), ("rate hike", 4),
        ("powell", 4), ("monetary policy", 4), ("fed ", 3), ("interest rate", 3),
        ("dovish", 3), ("hawkish", 3), ("taper", 3), ("fed's", 3),
    ],
    "通胀与物价": [
        ("inflation", 5), ("cpi", 5), ("pce", 4), ("consumer price", 4),
        ("deflation", 4), ("stagflation", 4), ("price pressure", 3),
    ],
    "美债与收益率": [
        ("treasury bond", 5), ("treasury yield", 5), ("10-year", 4),
        ("10y", 4), ("bond market", 4), ("yield curve", 4),
        ("treasury", 3), ("bond", 2), ("yields", 3),
    ],
    "企业财报与盈利": [
        ("earnings", 4), ("quarterly report", 4), ("eps", 4),
        ("guidance", 3), ("revenue miss", 4), ("earnings beat", 4),
        ("earnings miss", 4), ("profit warning", 4),
    ],
    "衰退与宏观经济": [
        ("recession", 5), ("gdp", 4), ("unemployment", 4),
        ("payroll", 4), ("jobs report", 4), ("economic slowdown", 4),
        ("soft landing", 4), ("hard landing", 4), ("labor market", 3),
    ],
    "科技与半导体": [
        ("semiconductor", 5), ("tsmc", 5), ("chip", 3), ("chips", 3),
        ("nvidia", 4), ("nvda", 4), ("ai chip", 5), ("gpu", 3),
        ("artificial intelligence", 3), ("openai", 3),
    ],
    "关税与贸易": [
        ("tariff", 5), ("trade war", 5), ("trade deal", 4),
        ("import duty", 4), ("export ban", 4), ("trade deficit", 4),
        ("protectionism", 3), ("sanctions", 3),
    ],
    "原油与能源": [
        ("oil price", 5), ("crude oil", 5), ("opec", 5),
        ("brent", 4), ("wti", 4), ("energy crisis", 4),
        ("strait of hormuz", 5), ("natural gas", 3), ("oil", 2),
    ],
    "中国经济": [
        ("china economy", 5), ("china gdp", 5), ("pboc", 5),
        ("renminbi", 4), ("yuan", 3), ("chinese market", 4),
        ("china", 2),
    ],
    "房地产与信贷": [
        ("commercial real estate", 5), ("housing market", 5),
        ("mortgage rate", 5), ("real estate", 4), ("housing", 3),
        ("home price", 4), ("foreclosure", 4),
    ],
    "加密货币": [
        ("bitcoin", 5), ("btc", 4), ("ethereum", 5), ("eth", 3),
        ("crypto", 4), ("cryptocurrency", 4), ("defi", 3),
    ],
}

MIN_TOPIC_SCORE = 2


def classify_post(title: str) -> tuple[str, int]:
    """
    Return (topic_label, match_score) for a single post/tweet title.
    Returns ("", 0) if no topic reaches MIN_TOPIC_SCORE.
    """
    text = title.lower()
    best_topic = ""
    best_score = 0

    for topic, rules in TOPIC_RULES.items():
        score = sum(weight for kw, weight in rules if kw in text)
        if score > best_score:
            best_score = score
            best_topic = topic

    if best_score < MIN_TOPIC_SCORE:
        return "", 0
    return best_topic, best_score


SENTIMENT_LABELS_CN = {
    "bullish": "bullish",
    "bearish": "bearish",
    "neutral": "neutral",
    "mixed": "mixed",
}
