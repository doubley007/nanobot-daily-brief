"""
把 Decision / AggregatedSentiment / RouterResult 渲染成最终 Telegram 文本。

风格契约（严格遵守，不走普通 ChatGPT 金融免责声明那套）：
  1. 先结论，再依据。
  2. 不用"这取决于你的风险偏好/建议咨询专业人士"之类的对冲话术。
  3. 根据用户情绪调整语气：
        fomo         -> 先降温，再给分批建议，不完全否决
        frustrated   -> 先共情，再给下一步动作
        anxious      -> 更稳定的节奏、更明确的指引
        uncertain    -> 直给立场，不反问
  4. 每段都有实际信息密度，不写空话。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from assistant.asset_taxonomy import asset_display
from assistant.decision_engine import Decision
from assistant.query_router import RouterResult
from assistant.sentiment_aggregator import AggregatedSentiment
from assistant.user_emotion import UserEmotionProfile


# ─── 情绪化开头（可选，只在用户情绪较强时出现） ───────────────────────────

_EMPATHY_OPENERS = {
    "frustrated": "Loss already happened — what matters now is the next move, not replaying the past.",
    "anxious": "Take a breath — I'll lay out what's actually happening.",
    "fomo": "Slow down for a sec — let me walk you through where things actually stand.",
    "seeking_confirmation": "You want a direct call — I'll give you one.",
}


def _empathy_opener(profile: UserEmotionProfile) -> str:
    if profile.emotion_intensity < 0.3:
        return ""
    return _EMPATHY_OPENERS.get(profile.primary_emotion, "")


# ─── market_decision 回复渲染 ────────────────────────────────────────────────

_ACTION_HEADLINES = {
    "buy": "✅ Bottom line: participate",
    "buy_small": "🟡 Bottom line: small, staggered participation",
    "hold_wait": "⏸ Bottom line: hold off — wait for a clearer signal",
    "avoid_chasing": "⚠️ Bottom line: don't chase here",
    "avoid": "🛑 Bottom line: step aside for now",
}

_CONFIDENCE_LINE = {
    "high": "Confidence: high",
    "medium": "Confidence: medium",
    "low": "Confidence: low (evidence not yet consistent)",
}

_LOW_CONF_CAVEAT = (
    "⚠️ Evidence is limited — treat the above as directional only. "
    "Not enough to justify a firm call or a sized-up position."
)


def _format_news_lines(decision: Decision) -> str:
    news_items = decision.evidence.get("news", [])
    assessment = decision.evidence.get("news_assessment", {})
    if not news_items:
        return "No strongly relevant news in the recent window."
    direction = assessment.get("direction", "neutral")
    dir_text = {"bullish": "skew bullish", "bearish": "skew bearish",
                "neutral": "broadly neutral"}.get(direction, "broadly neutral")
    lines = [f"[Recent news] {dir_text}:"]
    for item in news_items[:3]:
        title = item.get("title", "")
        source = item.get("source", "")
        snippet = item.get("snippet", "")
        if not title:
            continue
        line = f"  • {title}"
        if source:
            line += f" ({source})"
        # Show a short snippet if it adds value beyond the title
        if snippet and snippet.strip() != title.strip() and len(snippet) > 20:
            short = snippet[:100].rstrip()
            if not short.endswith("…"):
                short += "…"
            line += f"\n    {short}"
        lines.append(line)
    return "\n".join(lines)


def _format_community_lines(decision: Decision) -> str:
    agg: dict[str, Any] = decision.evidence.get("community_aggregate", {}) or {}
    if not agg or agg.get("post_count", 0) == 0:
        return "Not enough community discussion picked up to read sentiment."
    line1 = agg.get("summary") or ""
    extras: list[str] = []
    if agg.get("fomo_ratio", 0) >= 0.25:
        extras.append(f"FOMO ratio {agg['fomo_ratio']:.0%}")
    if agg.get("crowded_trade_risk") == "high":
        extras.append("crowding=high")
    elif agg.get("crowded_trade_risk") == "medium":
        extras.append("crowding=medium")
    if agg.get("narrative_keywords"):
        nks = ", ".join(agg["narrative_keywords"][:3])
        extras.append(f"dominant narrative: {nks}")
    if extras:
        line1 = line1 + " (" + "; ".join(extras) + ")"

    # Surface entry quality from decision scores when informative
    dscores: dict[str, Any] = decision.evidence.get("decision_scores", {}) or {}
    entry = dscores.get("entry_quality", "")
    chasing = dscores.get("chasing_risk", "")
    entry_note = ""
    if entry == "poor" or chasing == "high":
        entry_note = "\n  ⚠️ Poor entry window right now (crowding / overheating both elevated)"
    elif entry == "medium" and chasing == "medium":
        entry_note = "\n  Note: some crowding signs already — scale in rather than chase in one clip"

    return "[Community sentiment] " + line1 + entry_note


def _format_risks(decision: Decision) -> str:
    if not decision.risks:
        return ""
    lines = ["Risks to watch:"] + [f"  • {r}" for r in decision.risks]
    return "\n".join(lines)


def _fomo_addendum(profile: UserEmotionProfile, decision: Decision) -> str:
    """用户 FOMO 时，即使 action 是 buy，也要补一句冷静建议。"""
    if profile.primary_emotion != "fomo":
        return ""
    if decision.action in ("buy", "buy_small"):
        return (
            "\n\nOne more thing — the signal I'm reading from you is 'fear of missing out'. "
            "That doesn't rule out buying, but don't go all-in in one clip. "
            "Scale in over 2-3 tranches so you leave yourself room to react."
        )
    if decision.action == "avoid_chasing":
        return (
            "\n\nI get the FOMO — but 'others made money' isn't a reason you have to enter now. "
            "Wait for a pullback or consolidation; there'll still be a trade."
        )
    return ""


def _frustrated_addendum(profile: UserEmotionProfile) -> str:
    if profile.primary_emotion == "frustrated" and profile.emotion_intensity >= 0.4:
        return (
            "\n\nThe loss you're sitting on isn't the question right now — "
            "the only question is the next move: hold, cut, or add. "
            "Don't let sunk cost trap you into inaction."
        )
    return ""


def _format_evidence_sources(decision: Decision) -> str:
    """Build a compact '主要依据' line from the decision's evidence dict."""
    pieces: list[str] = []

    # News sources
    news_items = decision.evidence.get("news", [])
    if news_items:
        sources = list(dict.fromkeys(
            item.get("source", "") for item in news_items[:3] if item.get("source")
        ))
        if sources:
            pieces.append("News: " + ", ".join(sources[:3]))

    # Community data
    agg: dict = decision.evidence.get("community_aggregate", {}) or {}
    comm_count = agg.get("post_count", 0)
    if comm_count:
        pieces.append(f"{comm_count} community posts")

    # Trend data
    trend_ev: dict = decision.evidence.get("trend", {}) or {}
    ds = trend_ev.get("data_source", "") or ""
    if ds and ds != "stub":
        pieces.append(f"price trend ({ds})")

    if not pieces:
        return ""
    return "Key evidence: " + "; ".join(pieces)


def compose_decision_reply(
    decision: Decision,
    router: RouterResult,
    emotion: UserEmotionProfile,
) -> str:
    asset_name = asset_display(decision.asset)
    headline = _ACTION_HEADLINES[decision.action]
    conf_line = _CONFIDENCE_LINE[decision.confidence]

    parts: list[str] = []

    opener = _empathy_opener(emotion)
    if opener:
        parts.append(opener)

    # 1. 结论
    parts.append(f"{headline} ({asset_name})")
    # 2. 主要依据 (thesis)
    parts.append(decision.thesis)
    # 3. 信心等级
    parts.append(conf_line)
    # 4. 主要依据来源
    evidence_line = _format_evidence_sources(decision)
    if evidence_line:
        parts.append(evidence_line)
    # 5. 新闻面
    parts.append(_format_news_lines(decision))
    # 6. 社区情绪
    parts.append(_format_community_lines(decision))
    # 7. 主要风险
    risks = _format_risks(decision)
    if risks:
        parts.append(risks)

    if decision.suitable_for:
        parts.append(f"Suits: {decision.suitable_for}")
    # 8. 行动建议
    if decision.one_line_advice:
        parts.append(f"👉 {decision.one_line_advice}")

    # Low-confidence caveat: explicit "当前证据不足" notice
    if decision.confidence == "low":
        parts.append(_LOW_CONF_CAVEAT)

    text = "\n\n".join([p for p in parts if p])
    text += _fomo_addendum(emotion, decision)
    text += _frustrated_addendum(emotion)
    return text


# ─── market_summary 回复渲染 ─────────────────────────────────────────────────

def compose_summary_reply(
    agg: AggregatedSentiment | None,
    news_docs: list,
    asset: str | None,
) -> str:
    parts: list[str] = []
    name = asset_display(asset) if asset else "the market"
    parts.append(f"📊 Recent read on {name}:")

    if news_docs:
        parts.append("Top stories:")
        for d in news_docs[:3]:
            parts.append(f"  • {d.title}")
    else:
        parts.append("News: no material catalysts picked up in the recent window.")

    if agg and agg.post_count:
        parts.append("Community: " + agg.summary)
        if agg.narrative_keywords:
            nks = ", ".join(agg.narrative_keywords[:5])
            parts.append(f"Dominant narratives: {nks}")
    else:
        parts.append("Community: discussion thin or data unavailable for now.")

    return "\n\n".join(parts)


# ─── emotional_chat 回复渲染 ─────────────────────────────────────────────────

_EMOTIONAL_REPLIES = {
    "frustrated": (
        "Loss already happened — what matters now is the next move, not replaying the past.\n\n"
        "Tell me which position you're sitting on and I'll walk you through the news + community read, "
        "and frame the next step concretely (hold / cut / scale down in tranches)."
    ),
    "anxious": (
        "Take a breath — I'll lay out what's actually happening. "
        "Markets are uncomfortable, but panic is the worst state to make decisions from.\n\n"
        "Tell me which asset or situation is worrying you and I'll pull the facts and community read "
        "together before you touch the position."
    ),
    "fomo": (
        "Slow down for a sec. Fear of missing out is normal, but it's not a signal.\n\n"
        "Tell me the ticker and I'll tell you — based on news and community flow — "
        "whether this is a real setup or just 'everyone's buying'."
    ),
    "uncertain": (
        "When it's unclear, don't guess.\n\n"
        "Tell me the asset you're weighing and I'll frame buy / wait / avoid "
        "with the current news + community evidence."
    ),
    "seeking_confirmation": (
        "I'll give you a direct view — but I need to know the specific asset.\n\n"
        "The more concrete the question, the sharper the answer (no hedging)."
    ),
}


def compose_emotional_reply(emotion: UserEmotionProfile) -> str:
    return _EMOTIONAL_REPLIES.get(
        emotion.primary_emotion,
        "I'm here. Tell me the specific asset or situation you're weighing and I'll help you judge.",
    )
