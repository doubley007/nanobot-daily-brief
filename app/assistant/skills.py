"""
Skills —— 实际可运行的技能模块（不是只起名字）。

每个 skill 是一个可独立调用的函数，返回结构化结果。
技能之间可以组合；pipeline 按需调用。

技能列表：
  1. detect_user_intent_and_emotion   意图 + 情绪联合检测
  2. retrieve_market_context          拉取市场 RAG 证据
  3. aggregate_asset_sentiment        聚合资产情绪
  4. assess_entry_quality             入场质量评估
  5. compose_in_house_style_reply     生成公司风格回答
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ─── Skill 1: detect_user_intent_and_emotion ────────────────────────────────

@dataclass
class IntentEmotionResult:
    route: str             # "emotional_chat" | "market_decision" | "market_summary"
    asset: str | None
    confidence: float
    primary_emotion: str
    emotion_intensity: float
    risk_of_impulsive: bool
    rationale: str = ""


def detect_user_intent_and_emotion(
    text: str,
    llm_callable: Callable[[str], str] | None = None,
) -> IntentEmotionResult:
    """
    联合路由 + 情绪分析。
    复用 query_router 和 user_emotion 模块，但在这里提供统一接口。
    """
    from assistant.query_router import route_query
    from assistant.user_emotion import analyze_user_emotion

    route_result = route_query(text, llm_callable=llm_callable)
    emotion_result = analyze_user_emotion(text, llm_callable=llm_callable)

    return IntentEmotionResult(
        route=route_result.route,
        asset=route_result.asset,
        confidence=route_result.confidence,
        primary_emotion=emotion_result.primary_emotion,
        emotion_intensity=emotion_result.emotion_intensity,
        risk_of_impulsive=emotion_result.risk_of_impulsive_action,
        rationale=route_result.rationale,
    )


# ─── Skill 2: retrieve_market_context ───────────────────────────────────────

@dataclass
class MarketContextResult:
    asset: str | None
    news_count: int
    community_count: int
    top_news_titles: list[str]
    community_summary: str
    window_hours: int


def retrieve_market_context(
    asset: str | None,
    window_hours: int = 72,
    top_k_news: int = 8,
    top_k_community: int = 100,
    rag_store: Any = None,
) -> MarketContextResult:
    """
    从 RAG 检索市场证据，返回结构化摘要。
    实际 news/community 对象传给 aggregate / decision 用；
    这里返回可读摘要供调试和 trace。
    """
    from assistant.rag.retriever import Retriever
    from assistant.rag.store import default_store

    store = rag_store or default_store()
    retriever = Retriever(store=store)
    evidence = retriever.retrieve(
        asset=asset,
        window_hours=window_hours,
        top_k_news=top_k_news,
        top_k_community=top_k_community,
    )

    top_titles = [n.title for n in evidence.news[:5]]
    community_summary = f"{len(evidence.community)} community posts"

    return MarketContextResult(
        asset=asset,
        news_count=len(evidence.news),
        community_count=len(evidence.community),
        top_news_titles=top_titles,
        community_summary=community_summary,
        window_hours=window_hours,
    )


# ─── Skill 3: aggregate_asset_sentiment ──────────────────────────────────────

@dataclass
class AssetSentimentResult:
    asset: str | None
    overall_bias: str          # "bullish" | "bearish" | "neutral" | "mixed"
    bullish_ratio: float
    bearish_ratio: float
    fomo_ratio: float
    crowding_risk: str
    narrative_keywords: list[str]
    post_count: int
    summary: str


def aggregate_asset_sentiment(
    asset: str | None,
    window_hours: int = 72,
    top_k: int = 200,
    rag_store: Any = None,
) -> AssetSentimentResult:
    """
    一步到位：检索社区 → 聚合情绪 → 返回结构化读数。
    """
    from assistant.rag.retriever import Retriever
    from assistant.rag.store import default_store
    from assistant.sentiment_aggregator import aggregate

    store = rag_store or default_store()
    retriever = Retriever(store=store)
    community_docs = retriever.retrieve_community(
        asset=asset, window_hours=window_hours, top_k=top_k,
    )
    agg = aggregate(community_docs, asset=asset, window_hours=window_hours)

    return AssetSentimentResult(
        asset=asset,
        overall_bias=agg.overall_bias,
        bullish_ratio=agg.bullish_ratio,
        bearish_ratio=agg.bearish_ratio,
        fomo_ratio=agg.fomo_ratio,
        crowding_risk=agg.crowded_trade_risk,
        narrative_keywords=agg.narrative_keywords,
        post_count=agg.post_count,
        summary=agg.summary,
    )


# ─── Skill 4: assess_entry_quality ───────────────────────────────────────────

@dataclass
class EntryAssessmentResult:
    asset: str | None
    entry_quality: str        # "good" | "medium" | "poor"
    chasing_risk: str         # "low" | "medium" | "high"
    direction_score: float    # -1..+1
    crowding_score: float     # 0..1
    action_suggestion: str    # buy | buy_small | hold_wait | avoid_chasing | avoid
    confidence: str           # high | medium | low
    one_line: str


def assess_entry_quality(
    asset: str | None,
    window_hours: int = 72,
    rag_store: Any = None,
) -> EntryAssessmentResult:
    """
    完整跑一遍决策引擎，返回聚焦在"入场质量"的结构化评估。
    不依赖 LLM（只用规则引擎）。
    """
    from assistant.rag.retriever import Retriever
    from assistant.rag.store import default_store
    from assistant.sentiment_aggregator import aggregate
    from assistant.decision_engine import (
        assess_news, _decide_action,
        community_direction_score, trend_direction_score,
        _calc_crowding_score, _calc_entry_quality,
    )
    from assistant.trend_signals import fetch_trend_signal

    store = rag_store or default_store()
    retriever = Retriever(store=store)
    evidence = retriever.retrieve(
        asset=asset, window_hours=window_hours,
        top_k_news=8, top_k_community=100,
    )
    agg = aggregate(evidence.community, asset=asset, window_hours=window_hours)
    news_assess = assess_news(asset, evidence.news)
    trend = fetch_trend_signal(asset)

    action, confidence, _, scores = _decide_action(news_assess, agg, trend)

    one_lines = {
        "buy": "Signal skews bullish and entry quality is good — participate.",
        "buy_small": "Signal skews bullish but crowding is elevated — small position, scale in.",
        "hold_wait": "Signal isn't consistent enough — wait.",
        "avoid_chasing": "Signal doesn't support chasing here — wait for a pullback.",
        "avoid": "Signal skews bearish — step aside.",
    }

    return EntryAssessmentResult(
        asset=asset,
        entry_quality=scores.entry_quality,
        chasing_risk=scores.chasing_risk,
        direction_score=scores.direction_score,
        crowding_score=scores.crowding_score,
        action_suggestion=action,
        confidence=confidence,
        one_line=one_lines.get(action, ""),
    )


# ─── Skill 5: compose_in_house_style_reply ───────────────────────────────────

def compose_in_house_style_reply(
    decision: Any,   # Decision
    route: Any,      # RouterResult
    emotion: Any,    # UserEmotionProfile
    profile: Any,    # UserProfile
    company: Any,    # CompanyContext
    llm_callable: Callable[[str], str] | None = None,
) -> str:
    """
    用公司风格渲染最终回复。
    1. 调 reply_composer 生成基础文本
    2. 注入 few-shot 风格示例（如果有 LLM）
    3. 用 reply_policy 做最终审查/修正
    4. apply_style_constraints 收尾
    """
    from assistant.reply_composer import compose_decision_reply
    from assistant.reply_policy import (
        check_reply_policy,
        apply_style_constraints,
        get_style_examples_for_prompt,
    )

    base_reply = compose_decision_reply(decision, route, emotion)

    # Policy 审查
    policy = check_reply_policy(
        reply=base_reply,
        decision=decision,
        emotion=emotion,
        profile=profile,
        company=company,
    )

    if not policy.is_valid:
        logger.debug("Reply policy violations: %s", policy.violations)
        # 违规时用 LLM 重写（如果可用）或直接修正
        if llm_callable is not None:
            examples = get_style_examples_for_prompt(
                emotion=emotion.primary_emotion,
                decision_action=decision.action if decision else None,
                n=2,
            )
            revised = _llm_rewrite_with_policy(
                base_reply, decision, emotion, profile, company,
                llm_callable, examples,
            )
            if revised:
                base_reply = revised
        # 无论 LLM 有没有，都清掉禁用短语
        base_reply = apply_style_constraints(base_reply, profile, emotion, company)
    else:
        base_reply = apply_style_constraints(base_reply, profile, emotion, company)

    return base_reply


def _llm_rewrite_with_policy(
    draft: str,
    decision: Any,
    emotion: Any,
    profile: Any,
    company: Any,
    llm_callable: Callable[[str], str],
    style_examples: str,
) -> str | None:
    """让 LLM 按 policy 重写 draft，返回 None 表示失败。"""
    from assistant.decision_engine import Decision
    import json

    action = decision.action if decision else "unknown"
    confidence = decision.confidence if decision else "low"

    prompt = f"""You are a professional market analyst. Rewrite the reply below in our in-house style.

House style rules:
  - Bottom line first, evidence second.
  - No hedge-speak ("depends on your risk appetite", "consult a professional", "this is not financial advice", etc.).
  - If the user is FOMO and the decision is buy, you MUST include a scale-in / small-position caution.
  - The decision skeleton (action={action}, confidence={confidence}) must NOT be overridden — only rephrase.
  - Internal users: keep it concise. Retail users: explain more.
  - Respond in English.

User emotion: {emotion.primary_emotion} (intensity {emotion.emotion_intensity:.2f})
User role: {profile.role} (internal={profile.is_internal})

{style_examples}

Original reply (may violate policy — fix it):
{draft}

Output only the corrected reply body. No JSON, no preamble, no explanation."""

    try:
        from llm_adapter import local_llm_plain
        result = local_llm_plain(prompt)
        return result.strip() if result else None
    except Exception as e:
        logger.warning("_llm_rewrite_with_policy failed: %s", e)
        return None
