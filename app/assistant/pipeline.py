"""
对外统一入口。telegram_bot / tests / CLI 都通过这个函数调 bot 能力。

升级后的流水线：
    text + user_id
     ├─► context_builder.build_context()
     │     ├─ company_context (auto-inject)
     │     ├─ user_profile (auto-inject by user_id)
     │     ├─ route_query + analyze_user_emotion
     │     ├─ RAG retrieve (news + community)
     │     ├─ derived_signal cache (optional)
     │     └─ trend_signal
     │
     ├─► [emotional_chat branch]
     │     └─► compose_emotional_reply
     │
     ├─► [market_summary branch]
     │     ├─ aggregate
     │     └─► compose_summary_reply
     │
     └─► [market_decision branch]
           ├─ aggregate
           ├─ fetch_trend_signal (already in context)
           ├─ make_decision
           └─► compose_in_house_style_reply (skills layer)

PipelineTrace 现在包含 company + profile + policy violations 供调试。
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from assistant.company_context import CompanyContext, get_company_context
from assistant.user_profile import (
    UserProfile, get_user_profile,
    update_profile_from_interaction, flush_profile_store,
)
from assistant.context_builder import ContextPackage, build_context
from assistant.decision_engine import Decision, make_decision
from assistant.query_router import RouterResult, route_query
from assistant.rag.retriever import Retriever
from assistant.reply_composer import (
    compose_decision_reply,
    compose_emotional_reply,
    compose_summary_reply,
)
from assistant.reply_policy import check_reply_policy, apply_style_constraints
from assistant.sentiment_aggregator import AggregatedSentiment, aggregate
from assistant.trend_signals import TrendSignal, fetch_trend_signal, get_current_price
from assistant.user_emotion import UserEmotionProfile, analyze_user_emotion
from assistant.session_memory import (
    SessionContext, record_turn, resolve_session_context,
)

logger = logging.getLogger(__name__)


def _debug_mode() -> bool:
    return os.getenv("ASSISTANT_DEBUG", "").strip() in ("1", "true", "yes")


@dataclass
class PipelineTrace:
    """开发/测试可见。主 Telegram 回复只返回 str，调试时用 answer_question_traced。"""
    route: RouterResult
    emotion: UserEmotionProfile
    aggregate: AggregatedSentiment | None = None
    trend: TrendSignal | None = None
    decision: Decision | None = None
    reply: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    # 新增：company + profile + policy
    company: CompanyContext | None = None
    profile: UserProfile | None = None
    policy_violations: list[str] = field(default_factory=list)
    context_pkg: ContextPackage | None = None
    # v5: session context
    session_context: SessionContext | None = None


def _llm_callable() -> Callable[[str], str] | None:
    """尝试拿到可用的 LLM；拿不到就返回 None，让下游走规则兜底。"""
    try:
        from llm_adapter import check_llm_available, local_llm_callable
        if not check_llm_available():
            return None
        return local_llm_callable
    except Exception as e:
        logger.info("LLM unavailable: %s", e)
        return None


def _window_hours() -> int:
    try:
        return int(os.getenv("ASSISTANT_WINDOW_HOURS", "72"))
    except ValueError:
        return 72


def answer_question_traced(
    text: str,
    user_id: str | int | None = None,
    llm_callable: Callable[[str], str] | None = None,
) -> PipelineTrace:
    llm = llm_callable if llm_callable is not None else _llm_callable()
    window_hours = _window_hours()

    # ── 0. Session context: resolve follow-up asset before building context ──
    from assistant.asset_taxonomy import detect_asset
    detected_asset = detect_asset(text)
    session_ctx = resolve_session_context(user_id, text, detected_asset)
    # If session resolved a different asset (follow-up), inject it back into text
    # by appending the asset name so route_query can detect it
    effective_text = text
    if session_ctx.is_followup and session_ctx.resolved_asset and not detected_asset:
        effective_text = f"{text} {session_ctx.resolved_asset}"

    # ── 1. 构建完整 context（company + profile + RAG + emotion + route）──────
    ctx = build_context(
        effective_text,
        user_id=user_id,
        llm_callable=llm,
        window_hours=window_hours,
        top_k_news=8,
        top_k_community=100,
        use_derived_cache=True,
    )

    company = ctx.company
    profile = ctx.profile
    router = ctx.route
    emotion = ctx.user_emotion

    trace = PipelineTrace(
        route=router,
        emotion=emotion,
        company=company,
        profile=profile,
        context_pkg=ctx,
        session_context=session_ctx,
    )
    trace.meta["user_id"] = user_id
    trace.meta["window_hours"] = window_hours
    trace.meta["context_build_ms"] = round(ctx.build_time_ms, 1)
    trace.meta["is_followup"] = session_ctx.is_followup
    trace.meta["session_asset"] = session_ctx.resolved_asset
    trace.meta["resolved_from_session"] = session_ctx.resolved_from_session
    trace.meta["resolver_confidence"] = session_ctx.resolver_confidence
    # Derived signal cache status: "hit:<summary>" or "miss"
    if ctx.derived_signal is not None:
        trace.meta["derived_signal_status"] = f"hit:{ctx.derived_signal.summary[:80]}"
    else:
        trace.meta["derived_signal_status"] = "miss"
    # v5: vector index status
    if ctx.evidence is not None:
        trace.meta["index_status"] = ctx.evidence.index_status
    else:
        trace.meta["index_status"] = "none"

    # ── emotional_chat ────────────────────────────────────────────────────
    if router.route == "emotional_chat":
        reply = compose_emotional_reply(emotion)
        reply = apply_style_constraints(reply, profile, emotion, company)
        trace.reply = reply
        record_turn(user_id=user_id, asset=router.asset, intent=router.route,
                    emotion=emotion.primary_emotion, action="unknown",
                    topic=f"emotional_chat:{emotion.primary_emotion}")
        return trace

    # ── market_summary ────────────────────────────────────────────────────
    if router.route == "market_summary":
        agg = aggregate(ctx.community, asset=router.asset, window_hours=window_hours)
        trace.aggregate = agg
        reply = compose_summary_reply(agg, ctx.news, router.asset)
        reply = apply_style_constraints(reply, profile, emotion, company)
        trace.reply = reply
        record_turn(user_id=user_id, asset=router.asset, intent=router.route,
                    emotion=emotion.primary_emotion, action="unknown",
                    topic=f"summary:{router.asset or 'market'}")
        return trace

    # ── market_decision ───────────────────────────────────────────────────
    agg = aggregate(ctx.community, asset=router.asset, window_hours=window_hours)
    trend = ctx.trend or fetch_trend_signal(router.asset)

    decision = make_decision(
        asset=router.asset,
        news=ctx.news,
        community_docs=ctx.community,
        trend=trend,
        sentiment_aggregate=agg,
        llm_callable=llm,
    )

    trace.aggregate = agg
    trace.trend = trend
    trace.decision = decision

    # ── 应用 skills 层 compose（带 policy 检查）────────────────────────────
    from assistant.skills import compose_in_house_style_reply
    from assistant.holdings import holdings_reply_addendum
    reply = compose_in_house_style_reply(
        decision=decision,
        route=router,
        emotion=emotion,
        profile=profile,
        company=company,
        llm_callable=llm,
    )

    # Append holdings-aware addendum with live price for P&L context
    _price: float | None = None
    _price_status: str = "missing"
    if router.asset:
        try:
            _price, _price_status = get_current_price(router.asset)
        except Exception:
            _price_status = "fallback"
    trace.meta["holdings_price_status"] = _price_status

    holding_note = holdings_reply_addendum(user_id, router.asset, current_price=_price)
    if holding_note:
        reply = reply + "\n\n" + holding_note

    # Record pnl_status in trace for debug visibility
    from assistant.holdings import default_holdings_store
    _h = default_holdings_store().get(user_id, router.asset) if router.asset else None
    trace.meta["pnl_status"] = _h.pnl_status(_price) if _h else "no_holding"

    # 记录 policy 检查结果（已经在 compose_in_house_style_reply 内部处理过了，这里补存 trace）
    policy_result = check_reply_policy(reply, decision, emotion, profile, company)
    trace.policy_violations = policy_result.violations

    trace.reply = reply

    # Update user profile from interaction (conservative, threshold-gated)
    if user_id is not None:
        if emotion.primary_emotion == "fomo":
            update_profile_from_interaction(user_id, "fomo")
        if router.asset:
            update_profile_from_interaction(user_id, "asset_mention", router.asset)
        # Flush dirty profiles to disk (non-blocking; skips if nothing changed)
        try:
            flush_profile_store()
        except Exception as _flush_err:
            logger.debug("profile flush failed (non-fatal): %s", _flush_err)

    # v5: record this turn into session memory
    record_turn(
        user_id=user_id,
        asset=router.asset,
        intent=router.route,
        emotion=emotion.primary_emotion,
        action=decision.action if decision else "unknown",
        topic=_build_topic_summary(router, decision),
    )

    if _debug_mode():
        _log_debug_trace(trace)

    return trace


def _build_topic_summary(router: RouterResult, decision: "Decision | None") -> str:
    """One-line summary of this turn for session memory."""
    asset = router.asset or "market"
    if decision is not None:
        return f"{asset}:{decision.action}:{router.route}"
    return f"{asset}:{router.route}"


def _log_debug_trace(trace: PipelineTrace) -> None:
    """
    ASSISTANT_DEBUG=1 时把关键决策路径打印到日志。
    升级版：包含 company context + profile + policy violations。
    """
    r = trace.route
    e = trace.emotion
    logger.info(
        "[DEBUG] route=%s  asset=%s  user_emotion=%s(%.2f)  conf=%.2f",
        r.route, r.asset, e.primary_emotion, e.emotion_intensity, r.confidence,
    )

    if trace.company:
        c = trace.company
        logger.info(
            "[DEBUG] company_context: name=%s  type=%s  risk=%s  style=%s",
            c.company_name, c.business_type, c.risk_appetite, c.preferred_output_style,
        )

    if trace.profile:
        p = trace.profile
        logger.info(
            "[DEBUG] user_profile: role=%s  style=%s  is_internal=%s",
            p.role, p.preferred_style, p.is_internal,
        )

    if trace.context_pkg:
        pkg = trace.context_pkg
        logger.info(
            "[DEBUG] context: news=%d  community=%d  derived=%s  build_ms=%.1f",
            len(pkg.news), len(pkg.community),
            "yes" if pkg.derived_signal else "no",
            pkg.build_time_ms,
        )

    if trace.aggregate is not None:
        agg = trace.aggregate
        logger.info(
            "[DEBUG] aggregate: posts=%d  bias=%s  bullish=%.0f%%  bearish=%.0f%%  "
            "fomo=%.0f%%  conviction=%.0f%%  crowded=%s",
            agg.post_count, agg.overall_bias,
            agg.bullish_ratio * 100, agg.bearish_ratio * 100,
            agg.fomo_ratio * 100, agg.conviction_ratio * 100,
            agg.crowded_trade_risk,
        )
    if trace.trend is not None:
        t = trace.trend
        logger.info(
            "[DEBUG] trend: momentum=%s  7d=%s  30d=%s  overheating=%s",
            t.momentum_label,
            f"{t.recent_return_7d:+.1%}" if t.recent_return_7d is not None else "n/a",
            f"{t.recent_return_30d:+.1%}" if t.recent_return_30d is not None else "n/a",
            t.overheating_risk,
        )
    if trace.decision is not None:
        d = trace.decision
        sc = d.scores
        logger.info(
            "[DEBUG] decision: action=%s  confidence=%s  "
            "direction=%.3f  crowding=%.3f  entry=%s  chasing=%s",
            d.action, d.confidence,
            sc.direction_score if sc else 0.0,
            sc.crowding_score if sc else 0.0,
            sc.entry_quality if sc else "?",
            sc.chasing_risk if sc else "?",
        )
        news_ev = d.evidence.get("news", [])
        logger.info(
            "[DEBUG] retrieved: news=%d  community=%d",
            len(news_ev),
            len(d.evidence.get("community_samples", [])),
        )
        for reason in d.evidence.get("engine_trace", []):
            logger.info("[DEBUG] engine_trace: %s", reason)

    if trace.policy_violations:
        logger.info("[DEBUG] policy_violations: %s", trace.policy_violations)
    else:
        logger.info("[DEBUG] policy: OK")


def answer_question(
    text: str,
    user_id: str | int | None = None,
    llm_callable: Callable[[str], str] | None = None,
) -> str:
    """生产路径：只返回最终要发出去的字符串。"""
    trace = answer_question_traced(text, user_id=user_id, llm_callable=llm_callable)
    if _debug_mode():
        _log_debug_trace(trace)
    return trace.reply
