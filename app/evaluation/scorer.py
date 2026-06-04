"""
Rule-based scorer for pipeline evaluation results.

No external LLM calls — scoring is done entirely with keyword matching,
structural checks on the trace, and heuristics.

Scoring breakdown (max 10 points per answer):
  has_clear_conclusion   : 2 pts  — answer contains an explicit action/conclusion word
  has_market_data        : 1 pt   — price or return data present in answer
  has_risk_warning       : 2 pts  — explicit risk mention
  has_action_suggestion  : 1 pt   — actionable next step mentioned
  has_confidence_level   : 1 pt   — confidence hedge or clarity phrase
  direct_answer_used     : 1 pt   — direct_answer was populated (fact questions)
  route_correct          : 1 pt   — route matches expected_route (when specified)
  asset_correct          : 1 pt   — detected asset matches expected_asset (when specified)
  -hallucination_penalty :        — deduct 2 if hallucination risk flagged
"""
from __future__ import annotations

import re
from typing import Any

# ─── keyword sets ──────────────────────────────────────────────────────────────

_CONCLUSION_PATTERNS = [
    # Chinese action words
    "可以参与", "不建议", "先别动", "避免追入", "建议回避",
    "✅", "🟡", "⏸", "⚠️", "🛑",
    "结论", "可以买", "不能买", "别追", "能追", "先等",
    "持有", "减仓", "加仓", "买入", "卖出",
    # English
    "buy", "avoid", "hold", "wait", "bullish", "bearish",
    "recommend", "suggest", "should not", "should", "can buy",
]

_RISK_PATTERNS = [
    "风险", "注意", "谨慎", "小心", "危险", "不确定",
    "risk", "caution", "warning", "volatile", "uncertainty",
    "可能下跌", "可能回调", "需要注意", "拥挤", "过热",
    "risks:", "风险提醒", "⚠", "需警惕",
]

_ACTION_PATTERNS = [
    "建议", "可以", "分批", "止损", "观察", "等待",
    "👉", "action:", "建议操作", "行动",
    "next step", "consider", "分 2", "分 3",
]

_CONFIDENCE_PATTERNS = [
    "置信度", "confidence", "高", "中等", "偏低",
    "较高", "中等", "证据不足", "数据不足",
    "判断置信度", "信心", "high", "medium", "low",
    "当前证据不足", "暂时无法",
]

_MARKET_DATA_PATTERNS = [
    "%", "涨", "跌", "收益", "价格", "收盘", "利率", "yield",
    "return", "price", "+", "-", "bp", "点",
    "7日", "30日", "7d", "30d", "pct", "percent",
]

# Exported for use by run_eval.py to check whether a question is "factual"
_FACT_KEYWORDS = (
    "涨", "跌", "价格", "多少钱", "报价", "现在", "最近", "今天",
    "rising", "falling", "price", "quote", "current", "now", "today", "lately",
    "recent", "how much", "how is",
)

_HALLUCINATION_RISK_PATTERNS = [
    # Specific price claims without data
    r"\$\d+[\.,]\d+",           # e.g. $3,456.78
    r"\d+[\.,]\d{3}美元",       # e.g. 3,456美元
    r"涨了\s*\d+[\.,]?\d*\s*%", # e.g. 涨了5.3%
    r"跌了\s*\d+[\.,]?\d*\s*%",
    # Fabricated source names
    r"据\s*(彭博|路透|华尔街|FT|Bloomberg|Reuters)\s*报道",
    r"根据\s*(最新|权威|消息|数据)\s*显示",
    # Absolute future predictions
    "一定会", "肯定涨", "肯定跌", "必然", "必涨", "必跌",
    "guaranteed", "certainly will", "definitely",
]

_INSUFFICIENT_EVIDENCE_PHRASES = [
    "当前证据不足", "数据不足", "暂时无法获取", "没有检索到",
    "insufficient", "no data", "暂无数据", "无法获取",
]


def _contains_any(text: str, patterns: list[str]) -> bool:
    """
    Return True if any pattern matches.
    Patterns starting with a backslash-d or backslash-dollar are treated as
    regex applied to lowercased text; all others are plain substring matches.
    """
    text_lower = text.lower()
    for p in patterns:
        if p.startswith(("\\d", "\\$", r"\d", r"\$")):
            try:
                if re.search(p, text_lower):
                    return True
            except re.PatternError:
                if p in text_lower:
                    return True
        else:
            if p in text or p in text_lower:
                return True
    return False


def _check_hallucination_risk(answer: str, has_market_data_flag: bool) -> bool:
    """
    Heuristic: flag as hallucination risk if the answer makes specific numeric
    claims (prices / % moves) but the trace had no market data, OR if it uses
    fabricated-source phrases.
    """
    has_specific_numbers = bool(
        re.search(r"\d+[\.,]?\d*\s*%", answer) or
        re.search(r"\$\s*\d+", answer) or
        re.search(r"\d{3,}[\.,]\d{2}", answer)  # e.g. 3,456.78
    )
    fabrication_phrases = [
        "一定会", "肯定涨", "肯定跌", "必然", "必涨", "必跌",
        "据彭博报道", "据路透报道", "据华尔街报道",
        "guaranteed", "certainly will", "definitely will",
    ]
    has_fabrication = any(p in answer for p in fabrication_phrases)
    if has_fabrication:
        return True
    # Specific numbers in answer without any market data from pipeline
    if has_specific_numbers and not has_market_data_flag:
        return True
    return False


def score_result(result: dict[str, Any]) -> dict[str, Any]:
    """
    Score a single evaluation result dict.
    Expects fields: question, category, answer, route, expected_route,
                    expected_asset, asset, direct_answer_present,
                    has_trend_data, citations_count.
    Returns an augmented dict with scoring fields and total_score.
    """
    answer = result.get("answer", "") or ""
    category = result.get("category", "")
    route = result.get("route") or ""
    expected_route = result.get("expected_route") or ""
    expected_asset = result.get("expected_asset")
    detected_asset = result.get("asset")
    direct_answer_present = bool(result.get("direct_answer_present"))
    has_trend_data = bool(result.get("has_trend_data"))
    citations_count = int(result.get("citations_count", 0))

    score = 0
    flags: dict[str, bool] = {}

    # ── has_clear_conclusion (2 pts) ─────────────────────────────────────────
    has_conclusion = _contains_any(answer, _CONCLUSION_PATTERNS)
    flags["has_clear_conclusion"] = has_conclusion
    if has_conclusion:
        score += 2

    # ── has_market_data (1 pt) ───────────────────────────────────────────────
    has_mdata = has_trend_data or _contains_any(answer, _MARKET_DATA_PATTERNS)
    flags["has_market_data"] = has_mdata
    if has_mdata:
        score += 1

    # ── has_risk_warning (2 pts) ─────────────────────────────────────────────
    has_risk = _contains_any(answer, _RISK_PATTERNS)
    flags["has_risk_warning"] = has_risk
    if has_risk:
        score += 2

    # ── has_action_suggestion (1 pt) ─────────────────────────────────────────
    has_action = _contains_any(answer, _ACTION_PATTERNS)
    flags["has_action_suggestion"] = has_action
    if has_action:
        score += 1

    # ── has_confidence_level (1 pt) ──────────────────────────────────────────
    has_conf = _contains_any(answer, _CONFIDENCE_PATTERNS)
    flags["has_confidence_level"] = has_conf
    if has_conf:
        score += 1

    # ── direct_answer_used (1 pt) — only for factual questions ──────────────
    if category == "factual":
        flags["direct_answer_used"] = direct_answer_present
        if direct_answer_present:
            score += 1
    else:
        flags["direct_answer_used"] = False  # N/A for non-factual

    # ── route_correct (1 pt) ─────────────────────────────────────────────────
    if expected_route:
        route_correct = (route == expected_route)
        flags["route_correct"] = route_correct
        if route_correct:
            score += 1
    else:
        flags["route_correct"] = True  # N/A, give benefit of doubt

    # ── asset_correct (1 pt) — only when expected_asset is specified ─────────
    if expected_asset is not None:
        asset_correct = (detected_asset == expected_asset)
        flags["asset_correct"] = asset_correct
        if asset_correct:
            score += 1
    else:
        flags["asset_correct"] = True  # N/A

    # ── hallucination risk check (−2 if flagged) ─────────────────────────────
    hallucination_risk = _check_hallucination_risk(answer, has_mdata)
    # Emotional route is exempt — short supportive answers aren't fabricating data
    if category == "emotional":
        hallucination_risk = False
    # Short answers that are purely factual confirmations are usually OK
    if len(answer) < 15:
        hallucination_risk = False
    # If the answer explicitly states evidence is insufficient, not a hallucination
    if any(p in answer for p in _INSUFFICIENT_EVIDENCE_PHRASES):
        hallucination_risk = False
    flags["hallucination_risk"] = hallucination_risk
    if hallucination_risk:
        score = max(0, score - 2)

    result_out = dict(result)
    result_out.update({
        "has_clear_conclusion": flags["has_clear_conclusion"],
        "has_market_data": flags["has_market_data"],
        "has_risk_warning": flags["has_risk_warning"],
        "has_action_suggestion": flags["has_action_suggestion"],
        "has_confidence_level": flags["has_confidence_level"],
        "direct_answer_used": flags["direct_answer_used"],
        "route_correct": flags["route_correct"],
        "asset_correct": flags["asset_correct"],
        "hallucination_risk": flags["hallucination_risk"],
        "total_score": score,
    })
    return result_out
