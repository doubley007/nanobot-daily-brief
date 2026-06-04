from __future__ import annotations

import json
from typing import Callable, Optional

from financial_brief_formatter import NewsItem


def detect_news_type(item: NewsItem) -> str:
    text = f"{item.title} {item.summary}".lower()

    if any(
        k in text
        for k in [
            "fed",
            "federal reserve",
            "cpi",
            "inflation",
            "payroll",
            "rate cut",
            "rate cuts",
            "rates",
            "interest rate",
            "central bank",
        ]
    ):
        return "macro_rates"

    if any(k in text for k in ["treasury", "yield", "yields", "bond", "10y", "10-year"]):
        return "treasury_rates"

    if any(k in text for k in ["default", "downgrade", "credit", "liquidity", "spread"]):
        return "credit_risk"

    if any(k in text for k in ["earnings", "guidance", "revenue", "profit", "forecast"]):
        return "earnings"

    if any(k in text for k in ["mas", "singapore"]):
        return "singapore"

    if any(k in text for k in ["bank", "banks", "lender", "lenders"]):
        return "banking"

    if any(k in text for k in ["insurer", "insurance"]):
        return "insurance"

    if any(k in text for k in ["gold", "oil", "commodity", "commodities"]):
        return "commodities"

    return "general"


def _clean_raw_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = cleaned.replace("  ", " ")
    return cleaned


def fallback_happened(item: NewsItem) -> str:
    """
    Rule-based rewrite of 'what happened'.
    Make it closer to the actual news content, not just generic category text.
    """
    text = f"{item.title} {item.summary}".lower()
    raw_summary = _clean_raw_text(item.summary)
    raw_title = _clean_raw_text(item.title)

    # Fed / rate cut expectation
    if any(k in text for k in ["fed rate cut", "rate cut bets", "rate cuts"]):
        return "Market pricing for Fed rate cuts this year has firmed; rate-futures curves are shifting toward a more dovish path."

    # Treasury / yields
    if any(k in text for k in ["treasury", "yield", "yields", "bond"]):
        return "Treasury yields are moving, reflecting a market reassessment of the inflation and rate path."

    # Inflation
    if any(k in text for k in ["cpi", "inflation"]):
        return "Inflation expectations are shifting; investors are recalibrating their read on price pressure and policy."

    # Employment
    if any(k in text for k in ["payroll", "employment", "labor"]):
        return "Labour-market signals have shifted; investors are reassessing economic resilience and policy tempo."

    # India / New Zealand / regional central banks
    if "india" in text and "holds rates" in text:
        return "The RBI held rates, flagging that geopolitical developments could amplify growth and inflation uncertainty."

    if "new zealand" in text and "holds rates" in text:
        return "The RBNZ held rates and warned it could act more forcefully if geopolitical shocks push inflation higher."

    # Central bank generic
    if any(k in text for k in ["federal reserve", "central bank", "interest rate", "rates"]):
        return "Central-bank signals have shifted; the market is repricing the forward rate path."

    # Credit risk
    if any(k in text for k in ["default", "downgrade", "credit", "liquidity", "spread"]):
        return "A new credit-risk development is in focus; attention turns to its impact on funding conditions and risk appetite."

    # Bank profits / outlook
    if any(k in text for k in ["bank profits", "bank profit", "lender", "lenders"]):
        return "Bank earnings find partial support, though geopolitical risk keeps the market cautious on the outlook."

    # Earnings / guidance
    if any(k in text for k in ["earnings", "guidance", "revenue", "profit", "forecast"]):
        return "New earnings or guidance information has arrived; the market is reassessing the profit trajectory."

    # Singapore
    if any(k in text for k in ["mas", "singapore"]):
        return "A new signal from Singapore's financial environment is worth tracking for local-market impact."

    # Insurance
    if any(k in text for k in ["insurer", "insurance"]):
        return "New operating or policy signals in the insurance industry — relevant for long-duration capital allocation and sector expectations."

    # Commodities
    if any(k in text for k in ["gold", "oil", "commodity", "commodities"]):
        return "Commodity prices are moving; the market is assessing the pass-through to inflation expectations and risk sentiment."

    if any(k in text for k in ["peace", "ceasefire", "diplomatic", "iran war", "middle east"]):
        return "New geopolitical developments; the market is assessing the impact on risk sentiment and asset pricing."

    if raw_summary and raw_summary != "no summary available":
        return raw_summary[:120]

    return raw_title[:120]


def fallback_why(item: NewsItem) -> str:
    news_type = detect_news_type(item)

    mapping = {
        "macro_rates": "Direct bearing on fixed-income valuations and reinvestment yields, plus duration positioning.",
        "treasury_rates": "Yield moves feed through to bond pricing, discount rates, and cross-asset allocation.",
        "credit_risk": "Affects credit-spread view, corporate-bond risk assessment, and associated impairment pressure.",
        "earnings": "Useful for reading the related sector and risk assets, but without industry spillover the portfolio impact is limited.",
        "singapore": "Direct relevance to SG-local allocation, FX translation, and regional market reads.",
        "banking": "Bank-sector moves shape credit conditions, liquidity expectations, and financials positioning.",
        "insurance": "Insurance-sector signals feed into rate sensitivity, long-duration allocation, and industry risk reads.",
        "commodities": "Commodity moves can transmit through inflation expectations into rates, credit, and risk assets.",
        "general": "Helpful for reading today's portfolio risk appetite and the dominant market theme.",
    }

    return mapping.get(news_type, mapping["general"])


def enrich_news_item_with_fallback(item: NewsItem) -> NewsItem:
    item.summary = fallback_happened(item)
    item.why_it_matters = fallback_why(item)
    return item


def build_news_enrichment_prompt(item: NewsItem) -> str:
    return f"""
You are the editor of a daily financial brief for a Singapore insurance investment team. Given the news below, return a JSON object with a concise, professional, non-fluff explanation. Respond in English.

Requirements:
1. Respond in English.
2. Do not copy the original headline verbatim.
3. "happened" — one sentence explaining what happened.
4. "why" — one sentence on the portfolio impact.
5. No empty phrases ("worth watching", "could have an impact"). Always point to a specific transmission channel or direction.
6. Preferred lenses: fixed income, credit, FX / liquidity, Singapore-local, real-estate / loans, private equity, regulatory impact.
7. Output must be valid JSON of the form:
{{
  "happened": "...",
  "why": "..."
}}

News title: {item.title}
News summary: {item.summary}
News source: {item.source}
""".strip()


def enrich_news_item_with_llm(
    item: NewsItem,
    llm_callable: Callable[[str], str],
) -> NewsItem:
    """
    llm_callable: function(prompt: str) -> str
    It should return a JSON string:
    {
      "happened": "...",
      "why": "..."
    }
    """
    prompt = build_news_enrichment_prompt(item)

    try:
        raw_output = llm_callable(prompt)
        data = json.loads(raw_output)

        happened = str(data.get("happened", "")).strip()
        why = str(data.get("why", "")).strip()

        if happened:
            item.summary = happened
        if why:
            item.why_it_matters = why

        if not happened or not why:
            return enrich_news_item_with_fallback(item)

        return item
    except Exception:
        return enrich_news_item_with_fallback(item)


def enrich_news_items(
    items: list[NewsItem],
    llm_callable: Optional[Callable[[str], str]] = None,
) -> list[NewsItem]:
    enriched: list[NewsItem] = []

    for item in items:
        if llm_callable is not None:
            enriched.append(enrich_news_item_with_llm(item, llm_callable))
        else:
            enriched.append(enrich_news_item_with_fallback(item))

    return enriched