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
    Rule-based rewrite of 'what happened' in Chinese.
    Make it closer to the actual news content, not just generic category text.
    """
    text = f"{item.title} {item.summary}".lower()
    raw_summary = _clean_raw_text(item.summary)
    raw_title = _clean_raw_text(item.title)

    # Fed / rate cut expectation
    if any(k in text for k in ["fed rate cut", "rate cut bets", "rate cuts"]):
        return "市场对美联储年内降息的预期有所回升，利率期货定价开始朝更宽松的方向调整。"

    # Treasury / yields
    if any(k in text for k in ["treasury", "yield", "yields", "bond"]):
        return "美债收益率出现波动，反映市场正在重新评估未来通胀和利率走势。"

    # Inflation
    if any(k in text for k in ["cpi", "inflation"]):
        return "通胀相关预期出现变化，市场开始重新调整对价格压力和政策路径的判断。"

    # Employment
    if any(k in text for k in ["payroll", "employment", "labor"]):
        return "就业市场信号出现变化，投资者正在重新评估经济韧性和政策节奏。"

    # India / New Zealand / regional central banks
    if "india" in text and "holds rates" in text:
        return "印度央行维持利率不变，同时提示地缘局势可能加大增长和通胀的不确定性。"

    if "new zealand" in text and "holds rates" in text:
        return "新西兰央行维持利率不变，并警告若地缘冲突推升通胀，后续可能采取更强硬行动。"

    # Central bank generic
    if any(k in text for k in ["federal reserve", "central bank", "interest rate", "rates"]):
        return "央行政策相关信号出现新变化，市场正在重新评估后续利率路径。"

    # Credit risk
    if any(k in text for k in ["default", "downgrade", "credit", "liquidity", "spread"]):
        return "信用风险相关事件出现新进展，市场开始关注其对融资环境和风险偏好的影响。"

    # Bank profits / outlook
    if any(k in text for k in ["bank profits", "bank profit", "lender", "lenders"]):
        return "银行业盈利前景获得部分支撑，但地缘风险仍让市场对后续表现保持谨慎。"

    # Earnings / guidance
    if any(k in text for k in ["earnings", "guidance", "revenue", "profit", "forecast"]):
        return "公司业绩或管理层指引出现新信息，市场正在重新评估其盈利前景。"

    # Singapore
    if any(k in text for k in ["mas", "singapore"]):
        return "新加坡或本地区金融环境出现新信号，值得关注其对本地市场的影响。"

    # Insurance
    if any(k in text for k in ["insurer", "insurance"]):
        return "保险行业出现新的经营或政策信号，可能影响长期资金配置与行业预期。"

    # Commodities
    if any(k in text for k in ["gold", "oil", "commodity", "commodities"]):
        return "大宗商品价格出现新变化，市场正在评估其对通胀预期和风险情绪的影响。"
    
    if any(k in text for k in ["peace", "ceasefire", "diplomatic", "iran war", "middle east"]):
        return "地缘局势相关表态出现新进展，市场正在评估其对风险情绪和资产定价的影响。"

    if raw_summary and raw_summary != "暂无摘要":
        return raw_summary[:120]

    return raw_title[:120]


def fallback_why(item: NewsItem) -> str:
    news_type = detect_news_type(item)

    mapping = {
        "macro_rates": "这会直接影响固定收益组合估值和再投资收益率，也会影响久期配置判断。",
        "treasury_rates": "收益率变化会影响债券价格、贴现率和大类资产配置方向。",
        "credit_risk": "这会影响信用利差判断、企业债风险评估，以及相关减值压力。",
        "earnings": "这有助于判断相关板块和风险资产表现，但除非具有行业外溢性，对整体组合影响相对有限。",
        "singapore": "这对新加坡本地资产配置、外汇换算和区域市场判断更有直接参考意义。",
        "banking": "银行业变化会影响信用环境、流动性预期以及金融板块配置。",
        "insurance": "保险行业相关信号有助于判断利率敏感性、长期资金配置和行业风险。",
        "commodities": "大宗商品价格变化可能通过通胀预期传导至利率、信用和风险资产表现。",
        "general": "这条信息有助于判断当天组合风险偏好和市场主线变化。",
    }

    return mapping.get(news_type, mapping["general"])


def enrich_news_item_with_fallback(item: NewsItem) -> NewsItem:
    item.summary = fallback_happened(item)
    item.why_it_matters = fallback_why(item)
    return item


def build_news_enrichment_prompt(item: NewsItem) -> str:
    return f"""
你是一个服务于新加坡保险投资团队的金融日报编辑。请基于下面的新闻，输出一个 JSON，对新闻做简洁、专业、非空话的中文解释。

要求：
1. 用中文输出
2. 不要照抄原文标题
3. "happened" 用一句话解释“发生了什么”
4. "why" 用一句话解释“对组合影响”
5. 不要写空话，如“值得关注”“可能产生影响”这类泛话，除非说明具体影响方向
6. 优先从固定收益、信用、外汇流动性、新加坡本地、房地产贷款、私募股权、监管影响来解释
7. 输出必须是 JSON，格式如下：
{{
  "happened": "...",
  "why": "..."
}}

新闻标题：{item.title}
新闻摘要：{item.summary}
新闻来源：{item.source}
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