"""
Verbalization layer for community analysis.

Turns structured signals (sentiment profile, trend profile, insurance
framework, cross-platform spread) into natural Chinese business-language
phrases that can be shown to non-technical readers.

The brief formatter consumes these functions. Raw numeric scores are
still accessible on the underlying dataclasses — this layer just picks
the sentence, nothing more.

Design notes:
  - Keep each function pure: no I/O, no LLM call.
  - Prefer short phrases. The brief is dense; a sentence is more useful
    than a paragraph.
  - When inputs are weak or contradictory, fall back to neutral phrasing
    rather than overclaim.
"""
from __future__ import annotations

from community.schema import (
    InsuranceFramework,
    SentimentProfile,
    TopicCluster,
    TrendProfile,
)


# ─── Sentiment → business phrase ─────────────────────────────────────────────

_LABEL_EN = {
    "bullish": "bullish lean",
    "bearish": "bearish lean",
    "neutral": "neutral",
    "mixed": "mixed / split",
}

_DIMENSION_EN = {
    "optimism": "optimism",
    "fear": "fear",
    "uncertainty": "uncertainty",
    "skepticism": "skepticism",
    "hype": "hype",
}


def verbalize_sentiment(profile: SentimentProfile) -> str:
    """
    Produce a short business-language read of a sentiment profile.
    """
    label = profile.label or "neutral"
    dom = profile.dominant_dimension
    intensity = profile.intensity

    if dom == "hype" and intensity >= 0.55:
        return "high chatter, direction is split"

    if dom == "fear" and intensity >= 0.5:
        if label == "mixed":
            return "split, with fear dominating"
        if label == "bearish":
            return "risk-off tone building"
        return "fear-led"

    if dom == "uncertainty" and intensity >= 0.5:
        if label == "mixed":
            return "wait-and-see, high uncertainty"
        if label == "bullish":
            return "bullish lean, but consensus is thin"
        if label == "bearish":
            return "cautious, direction still unclear"
        return "wait-and-see"

    if dom == "skepticism" and intensity >= 0.5:
        if label == "bullish":
            return "rally narrative being questioned"
        return "skeptical of the upside"

    if dom == "optimism" and intensity >= 0.5:
        if label == "mixed":
            return "bullish lean, but split remains"
        return "bullish lean"

    return {
        "bullish": "bullish lean",
        "bearish": "cautious lean",
        "mixed": "split, direction unclear",
        "neutral": "steady sentiment",
    }.get(label, "steady sentiment")


def sentiment_score_tail(profile: SentimentProfile) -> str:
    """
    Optional supplemental score tail, e.g. ' (mixed · fear 0.62)'.
    Only rendered when the dominant dimension is meaningfully strong.
    """
    if profile.intensity < 0.55 or not profile.dominant_dimension:
        label_en = _LABEL_EN.get(profile.label, "")
        return f" ({label_en})" if label_en else ""
    label_en = _LABEL_EN.get(profile.label, "")
    dim_en = _DIMENSION_EN.get(profile.dominant_dimension, profile.dominant_dimension)
    parts = [p for p in [label_en, f"{dim_en} {profile.intensity:.2f}"] if p]
    return f" ({' · '.join(parts)})"


# ─── Trend → business phrase ─────────────────────────────────────────────────

_TREND_DIRECTION_EN = {
    "rising": "heating up",
    "stable": "steady discussion",
    "fading": "cooling off",
    "new": "emerging",
}

_PERSISTENCE_EN = {
    "new": "just starting",
    "continuing": "ongoing",
    "short-lived": "brief focus",
}

_PLATFORM_SPREAD_EN = {
    "reddit-led": "Reddit-led",
    "discord-led": "Discord-led",
    "x-led": "X-led",
    "cross-platform": "cross-platform",
    "single-platform": "single-platform",
}

_BREADTH_EN = {
    "narrow": "narrow",
    "moderate": "moderate",
    "broad": "broad",
}


def verbalize_trend(trend: TrendProfile) -> str:
    """
    Render a trend profile as a short chip:
      'heating up · cross-platform · moderate'
    Omits fields that are at their neutral default to avoid noise.
    """
    parts: list[str] = []

    direction = _TREND_DIRECTION_EN.get(trend.trend_direction)
    if direction:
        parts.append(direction)

    spread = _PLATFORM_SPREAD_EN.get(trend.platform_spread)
    if spread and trend.platform_spread != "single-platform":
        parts.append(spread)
    elif trend.platform_spread == "single-platform":
        # Only mention single-platform when direction is "rising" or "new",
        # otherwise it's noise.
        if trend.trend_direction in ("rising", "new"):
            parts.append(_PLATFORM_SPREAD_EN["single-platform"])

    breadth = _BREADTH_EN.get(trend.discussion_breadth)
    if breadth and trend.discussion_breadth != "narrow":
        parts.append(breadth)
    elif trend.discussion_breadth == "narrow" and trend.trend_direction == "rising":
        parts.append(breadth)  # rising-but-narrow is a useful caveat

    return " · ".join(parts)


# ─── Insurance framework (fallback derivation) ───────────────────────────────

def derive_insurance_framework(cluster: TopicCluster) -> InsuranceFramework:
    """
    When the LLM didn't return a structured two-layer framework, derive a
    conservative one from free-text `insurance_angle` + the cluster's rule
    label. The goal is not to invent positions — it's to express the same
    thought as an observation framework rather than a direct instruction.
    """
    angle = (cluster.insurance_angle or "").strip()
    rule = cluster.rule_label or ""

    if not angle and not rule:
        return InsuranceFramework()

    # Heuristic implications based on the coarse topic bucket.
    # Keys remain in Chinese because rule_label values come from the LLM in Chinese
    # (cluster taxonomy); the English text below is what the user sees.
    implications_by_rule = {
        "美联储与利率政策": "Bears on duration and credit spreads; read together with the forward rate path to gauge reinvestment yields.",
        "通胀与物价": "Affects real yields and pricing of inflation-protected assets.",
        "美债与收益率": "Direct impact on fixed-income duration and reinvestment yields.",
        "企业财报与盈利": "Marginal effect on credit spreads and equity-risk premia.",
        "衰退与宏观经济": "Shapes the read on credit quality and rate-sensitive assets.",
        "科技与半导体": "Affects concentration and volatility in equity-linked holdings.",
        "关税与贸易": "Potential transmission to FX liquidity and credit of import-sensitive sectors.",
        "原油与能源": "Affects energy-sector exposure and the inflation path.",
        "中国经济": "Indirect effect on regional / Asia market exposure and FX positioning.",
        "房地产与信贷": "Affects property-linked credit exposure and mortgage-related valuations.",
        "加密货币": "Limited direct impact for an insurance book; more of an overall risk-appetite read.",
    }

    implications = implications_by_rule.get(rule, "")
    if angle and not implications:
        # fallback: keep the angle but reframe as an observation
        implications = angle

    triggers = "Keep watching discussion breadth and cross-platform spread; only consider further action when confirmed by actual data or a rate move."

    return InsuranceFramework(
        implications=implications,
        triggers=triggers,
    )


# ─── Market implication — confidence-aware rewording ────────────────────────

# Strong terms the LLM sometimes emits on weak evidence. When the
# credibility is not high enough to back them up, we replace the clause
# with a hedged observation-style phrase.
_ASSERTIVE_DIRECTIONAL_PATTERNS = {
    # Longer phrases FIRST so they match before the shorter substring variants.
    # Keys stay in Chinese because they match LLM-emitted text; replacements
    # are the hedged English phrasing users see.
    "信用利差进一步收窄": "watch whether credit spreads widen back out",
    "信用利差进一步走阔": "watch whether credit spreads widen further",
    "信用利差明显收窄": "direction of credit spreads still needs more confirmation",
    "利差显著收窄": "direction of spreads still needs further confirmation",
    "利差进一步收窄": "watch whether credit spreads widen back out",
    "利差进一步走阔": "watch whether credit spreads widen further",
    "进一步收窄的方向压力": "direction still needs more confirmation",
    "进一步走阔的方向压力": "direction still needs more confirmation",
    "信用利差收窄": "watch whether credit spreads widen back out",
    "利差收窄": "watch whether credit spreads widen back out",
    "利差走阔": "watch whether credit spreads widen further",
    "久期延长": "the duration-extension window still needs more confirmation",
    "延长久期": "the duration-extension window still needs more confirmation",
    "缩短久期": "any duration-shortening move still needs more confirmation",
    "利率下行": "the room for rates to fall still needs more confirmation",
    "利率上行": "the pace of any rate rise still needs more confirmation",
    "银行净息差改善": "pressure on bank NIM and growth expectations",
    "净息差改善": "pressure on bank NIM and growth expectations",
    "净息差扩大": "pressure on bank NIM and growth expectations",
    # English-side patterns for cases where LLM already emits English
    "credit spreads tighten further": "watch whether credit spreads widen back out",
    "spreads tighten further": "watch whether credit spreads widen back out",
    "credit spreads tighten": "watch whether credit spreads widen back out",
    "extend duration": "duration-extension window still needs more confirmation",
    "shorten duration": "any duration-shortening move still needs more confirmation",
}

# Topics where we prefer hedged language unless evidence is overwhelming.
_HEDGE_FAVORED_RULE_LABELS = {
    "衰退与宏观经济",
    "美联储与利率政策",
    "美债与收益率",
    "通胀与物价",
}


def soften_market_implication(cluster: TopicCluster) -> str:
    """
    Return a version of `cluster.market_relevance` that doesn't overclaim
    when the cluster's credibility is modest. Keeps LLM wording intact
    when credibility is high.

    Rules:
      - credibility.overall >= 0.70 → keep the raw line unchanged.
      - 0.50 ≤ credibility < 0.70 → if the line contains aggressive
        phrases (credit spread tightening, duration extension, etc.),
        swap them for hedged observation phrasing.
      - credibility < 0.50, or the topic is in _HEDGE_FAVORED_RULE_LABELS
        with weak evidence → always hedge, and add a caveat tail.
    """
    raw = (cluster.market_relevance or "").strip()
    if not raw:
        return raw

    credibility = cluster.credibility.overall
    rule = cluster.rule_label or ""

    # Strong evidence: trust the LLM text as-is.
    if credibility >= 0.70:
        return raw

    # Mid-evidence: patch any overclaim phrases.
    softened = raw
    patched = False
    # Strip leading "建议" / "应" verbs in front of known directional words so
    # the hedged rewrite doesn't yield "建议对X的判断仍需更多确认".
    import re
    softened = re.sub(
        r"(建议|应当|应)(延长久期|缩短久期|延长|缩短|加仓|减仓|增持|减持)",
        r"\2",
        softened,
    )
    for strong, soft in _ASSERTIVE_DIRECTIONAL_PATTERNS.items():
        if strong in softened:
            softened = softened.replace(strong, soft)
            patched = True

    # Weak-evidence hedge or macro-sensitive topics: append caveat if the
    # sentence reads like a decisive call and we haven't already softened.
    needs_caveat = (
        credibility < 0.50
        or (rule in _HEDGE_FAVORED_RULE_LABELS and credibility < 0.65)
    )
    if needs_caveat and not patched:
        if not any(hint in softened for hint in (
            "仍需", "需关注", "关注", "尚未", "确认",
            "needs more", "watch whether", "still needs", "confirm", "confirmation",
        )):
            tail = "; confirm against forthcoming data and the rate path"
            if softened.endswith(("。", ".", "!", "?")):
                softened = softened[:-1] + tail + "."
            else:
                softened = softened.rstrip("；;，,。 .") + tail
    return softened


def align_insurance_framework(
    framework: InsuranceFramework,
    cluster: TopicCluster,
) -> InsuranceFramework:
    """
    Ensure the insurance framework's `implications` don't read more
    aggressively than the (already-softened) market implication.

    If credibility < 0.50 and the implication line contains a specific
    instruction-style phrase ("延长久期 0.3 年"), rewrite it to its
    observation-style variant. This guards against the "前面偏谨慎，
    后面给激进动作" pattern the user flagged.
    """
    credibility = cluster.credibility.overall
    implications = (framework.implications or "").strip()
    triggers = (framework.triggers or "").strip()

    if not implications:
        return framework

    if credibility >= 0.70:
        return framework

    rewrote = False
    new_impl = implications

    # Strip specific bps / year-duration / percent instructions FIRST so the
    # directional-pattern swap doesn't leave dangling numbers behind.
    import re
    if credibility < 0.60:
        patterns = [
            # "延长久期0.3年" / "延长久期 0.3 年"
            r"(延长|缩短|增加|减少)[^\s，,；;。]{0,4}\s*\d+(\.\d+)?\s*(年|个百分点|bps|bp|%)",
            # "减持信用利差敏感资产5%"
            r"(减持|增持|加仓|减仓)[^\s，,；;。]{0,10}\s*\d+(\.\d+)?\s*%?",
            # English analogues: "extend duration by 0.3 years", "trim 5% of credit-sensitive exposure"
            r"\b(extend|shorten|increase|decrease|trim|add|cut)\b[^.,;]{0,30}\b\d+(\.\d+)?\s*(years?|yrs?|bps?|bp|%|percent)",
        ]
        replacement = "the size of any position adjustment still needs more confirmation"
        for pat in patterns:
            new_impl2 = re.sub(pat, replacement, new_impl, flags=re.IGNORECASE)
            if new_impl2 != new_impl:
                new_impl = new_impl2
                rewrote = True
        # Collapse any duplicate replacement clauses produced by the sub.
        new_impl = re.sub(
            re.escape(replacement) + r"(\s*[，,；;])?\s*" + re.escape(replacement),
            replacement,
            new_impl,
        )

    for strong, soft in _ASSERTIVE_DIRECTIONAL_PATTERNS.items():
        if strong in new_impl:
            new_impl = new_impl.replace(strong, soft)
            rewrote = True

    if rewrote:
        return InsuranceFramework(implications=new_impl, triggers=triggers)
    return framework


def render_insurance_framework(framework: InsuranceFramework) -> list[str]:
    """
    Render a two-layer insurance block into formatter-ready lines.
    Returns [] when both halves are empty.
    """
    lines: list[str] = []
    if framework.implications:
        lines.append(f"   Allocation read: {framework.implications}")
    if framework.triggers:
        lines.append(f"   Watch / triggers: {framework.triggers}")
    return lines


# ─── Overall sentiment structure (clean, non-stacked) ───────────────────────


def compose_sentiment_structure(
    clusters: list[TopicCluster],
    llm_text: str = "",
) -> str:
    """
    Produce a manager-readable one-liner for "整体情绪".

    Style goals:
      - One main judgement sentence, optional short complement
      - No model-label stacking like "分歧·担忧(0.70) + 不确定性高 + …"
      - Keep the conclusion grounded in the cluster sentiments, not in
        whatever phrasing the LLM chose this run.

    The LLM's own sentiment_structure is passed in as `llm_text`. We use
    it only when it's already tight (short, no obvious stacking); otherwise
    we replace it with a deterministic one-liner derived from the cluster
    profiles, so the user-visible text always reads like a conclusion
    rather than a stitched list of labels.
    """
    cleaned = (llm_text or "").strip()
    if cleaned and _looks_concise(cleaned):
        return cleaned

    non_noise = [c for c in clusters if not c.credibility.is_noise]
    if not non_noise:
        return "Community chatter is thin today — no clear dominant sentiment."

    label_counts: dict[str, int] = {}
    fear_sum = uncertainty_sum = hype_sum = skepticism_sum = optimism_sum = 0.0
    for c in non_noise:
        label_counts[c.sentiment.label] = label_counts.get(c.sentiment.label, 0) + 1
        fear_sum += c.sentiment.fear
        uncertainty_sum += c.sentiment.uncertainty
        hype_sum += c.sentiment.hype
        skepticism_sum += c.sentiment.skepticism
        optimism_sum += c.sentiment.optimism

    n = len(non_noise)
    fear_avg = fear_sum / n
    uncertainty_avg = uncertainty_sum / n
    hype_avg = hype_sum / n
    skepticism_avg = skepticism_sum / n
    optimism_avg = optimism_sum / n

    # Dominant coarse stance (by count, with mixed taking precedence when
    # close). Keeps it a judgement, not a vote tally.
    top_label = max(label_counts.items(), key=lambda kv: kv[1])[0]
    is_split = len({l for l, v in label_counts.items() if v >= 1}) >= 3

    if top_label == "bearish" or fear_avg >= 0.55:
        main = "Overall sentiment leans cautious; risk-off tone is building but not yet one-sided."
    elif is_split or uncertainty_avg >= 0.55:
        main = "Overall sentiment leans cautious — the key feature is high uncertainty and a clear directional split."
    elif top_label == "bullish" and optimism_avg >= 0.5:
        main = "Overall sentiment leans bullish, but a durable long-consensus isn't yet in place."
    elif hype_avg >= 0.55:
        main = "Discussion is heating up, but the market hasn't formed a clear consensus."
    elif skepticism_avg >= 0.5:
        main = "The market is skeptical of the rally narrative; bullish evidence still needs confirmation."
    else:
        main = "Overall sentiment is steady with no meaningful one-way bias."

    return main


def _looks_concise(text: str) -> bool:
    """
    Gate for surfacing the LLM's own sentiment_structure.

    Reject anything that stacks labels, contains inline numeric scores, or
    chains multiple clauses — even when separated by a single "；". The
    goal is to always reduce to ONE main judgement sentence (plus at most
    one complement), so we err on the side of rejecting and regenerating.
    """
    if not text:
        return False
    if len(text) > 120:
        return False
    if any(marker in text for marker in ("0.", "(", "（", " · ", "·", "+ ")):
        return False
    # Even a single "；" usually signals two stacked clauses of equal weight.
    if "；" in text or ";" in text:
        return False
    if text.count(",") >= 3 or text.count("，") >= 3:
        return False
    # Common stacked connector phrases — reject outright.
    stacked_markers = (
        "分歧明显", "担忧主导", "避险升温", "观望为主",
        "clear split", "fear dominating", "wait-and-see", "risk-off tone building",
    )
    hits = sum(1 for m in stacked_markers if m in text)
    if hits >= 2:
        return False
    return True


# ─── News ↔ Social bridge summary (deterministic) ────────────────────────────

def build_news_social_bridge(
    news_items: list,
    headline_topics: list[TopicCluster],
    linked_count: int,
) -> str:
    """
    Short (1-3 sentence) bridge comparing the news narrative and the
    community narrative. Intentionally deterministic — no extra LLM call.

    Returns empty string when evidence is too thin to say anything useful.
    """
    if not news_items or not headline_topics:
        return ""

    # Build crude token sets for lightweight overlap scoring.
    def _tokens(text: str) -> set[str]:
        return {w for w in (text or "").lower().split() if len(w) > 3}

    news_tokens: set[str] = set()
    for n in news_items[:5]:
        title = getattr(n, "title", "") or ""
        summary = getattr(n, "summary", "") or ""
        news_tokens |= _tokens(f"{title} {summary}")

    # Which headline topics overlap with the news set?
    overlapping: list[TopicCluster] = []
    community_only: list[TopicCluster] = []
    for t in headline_topics:
        topic_text = " ".join(
            [t.headline or "", t.rule_label or "", t.discussion_focus or ""]
        ).lower()
        topic_tokens = _tokens(topic_text) | {
            w for p in t.posts[:3] for w in _tokens(p.title)
        }
        if news_tokens & topic_tokens:
            overlapping.append(t)
        else:
            community_only.append(t)

    rising_only = [t for t in community_only if t.trend.trend_direction == "rising"]

    sentences: list[str] = []

    if overlapping and linked_count > 0:
        sample = overlapping[0].headline or overlapping[0].rule_label
        sentences.append(
            f"Community discussion broadly tracks the news narrative — {len(overlapping)} topic(s) overlap with today's news (e.g. {sample})"
        )
    elif overlapping:
        sentences.append(
            f"Community discussion partially overlaps with the news narrative, but the focus is on the risk side of {overlapping[0].headline or overlapping[0].rule_label}"
        )
    elif headline_topics:
        sentences.append("Community discussion has low overlap with today's news — it's following a separate thread of attention")

    if rising_only:
        sample = rising_only[0].headline or rising_only[0].rule_label
        sentences.append(
            f"Notably, community chatter on '{sample}' is heating up while news coverage has yet to catch up"
        )

    if len(sentences) < 2 and len(community_only) >= 1 and not rising_only:
        sample = community_only[0].headline or community_only[0].rule_label
        sentences.append(f"Separately, community attention on '{sample}' hasn't surfaced in today's news — worth tracking")

    return "; ".join(sentences[:3]).strip()
