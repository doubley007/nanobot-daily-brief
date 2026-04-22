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

_LABEL_ZH = {
    "bullish": "偏乐观",
    "bearish": "偏悲观",
    "neutral": "中性",
    "mixed": "分歧",
}

_DIMENSION_ZH = {
    "optimism": "乐观",
    "fear": "担忧",
    "uncertainty": "不确定",
    "skepticism": "质疑",
    "hype": "情绪化炒作",
}


def verbalize_sentiment(profile: SentimentProfile) -> str:
    """
    Produce a short business-language read of a sentiment profile.

    Shape: "<main stance>，<texture>"
      - 主立场 based on coarse label
      - 纹理 based on dominant dimension + intensity

    Examples (illustrative):
      - "观望为主，不确定性较高"
      - "分歧明显，担忧主导"
      - "情绪偏乐观，但共识不足"
      - "避险情绪升温"
      - "讨论热度高，方向分歧明显"
    """
    label = profile.label or "neutral"
    dom = profile.dominant_dimension
    intensity = profile.intensity

    # Highly hyped without clear direction → "热讨论方向分歧"
    if dom == "hype" and intensity >= 0.55:
        return "讨论热度高，方向分歧明显"

    # Fear-led states
    if dom == "fear" and intensity >= 0.5:
        if label == "mixed":
            return "分歧明显，担忧主导"
        if label == "bearish":
            return "避险情绪升温"
        return "担忧情绪占优"

    # Uncertainty-led
    if dom == "uncertainty" and intensity >= 0.5:
        if label == "mixed":
            return "观望为主，不确定性较高"
        if label == "bullish":
            return "情绪偏乐观，但共识不足"
        if label == "bearish":
            return "偏谨慎，方向尚不明朗"
        return "观望情绪占优"

    # Skepticism-led
    if dom == "skepticism" and intensity >= 0.5:
        if label == "bullish":
            return "反弹叙事被质疑"
        return "对利多存疑"

    # Optimism-led
    if dom == "optimism" and intensity >= 0.5:
        if label == "mixed":
            return "偏乐观，但分歧仍在"
        return "情绪偏乐观"

    # Fallback by coarse label when no dimension is strong enough
    return {
        "bullish": "情绪偏乐观",
        "bearish": "情绪偏谨慎",
        "mixed": "分歧为主，方向不明",
        "neutral": "情绪平稳",
    }.get(label, "情绪平稳")


def sentiment_score_tail(profile: SentimentProfile) -> str:
    """
    Optional supplemental score tail, e.g. '（分歧 · 担忧 0.62）'.
    Only rendered when the dominant dimension is meaningfully strong.
    Kept separate so the formatter can choose whether to show it.
    """
    if profile.intensity < 0.55 or not profile.dominant_dimension:
        label_zh = _LABEL_ZH.get(profile.label, "")
        return f"（{label_zh}）" if label_zh else ""
    label_zh = _LABEL_ZH.get(profile.label, "")
    dim_zh = _DIMENSION_ZH.get(profile.dominant_dimension, profile.dominant_dimension)
    parts = [p for p in [label_zh, f"{dim_zh} {profile.intensity:.2f}"] if p]
    return f"（{' · '.join(parts)}）"


# ─── Trend → business phrase ─────────────────────────────────────────────────

_TREND_DIRECTION_ZH = {
    "rising": "明显升温",
    "stable": "持续讨论",
    "fading": "热度回落",
    "new": "新出现",
}

_PERSISTENCE_ZH = {
    "new": "刚开始",
    "continuing": "持续中",
    "short-lived": "短暂聚焦",
}

_PLATFORM_SPREAD_ZH = {
    "reddit-led": "Reddit主导",
    "discord-led": "Discord主导",
    "x-led": "X主导",
    "cross-platform": "双平台共振",
    "single-platform": "单平台讨论",
}

_BREADTH_ZH = {
    "narrow": "小范围",
    "moderate": "中等范围",
    "broad": "较广泛",
}


def verbalize_trend(trend: TrendProfile) -> str:
    """
    Render a trend profile as a short chip:
      '明显升温 · 双平台共振 · 中等范围'
    Omits fields that are at their neutral default to avoid noise.
    """
    parts: list[str] = []

    direction = _TREND_DIRECTION_ZH.get(trend.trend_direction)
    if direction:
        parts.append(direction)

    spread = _PLATFORM_SPREAD_ZH.get(trend.platform_spread)
    if spread and trend.platform_spread != "single-platform":
        parts.append(spread)
    elif trend.platform_spread == "single-platform":
        # Only mention single-platform when direction is "rising" or "new",
        # otherwise it's noise.
        if trend.trend_direction in ("rising", "new"):
            parts.append(_PLATFORM_SPREAD_ZH["single-platform"])

    breadth = _BREADTH_ZH.get(trend.discussion_breadth)
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

    # Heuristic implications based on the coarse topic bucket
    implications_by_rule = {
        "美联储与利率政策": "对久期管理与信用利差均有影响，需结合后续利率路径判断再投资收益率走向",
        "通胀与物价": "对实际收益率与通胀保护型资产定价存在影响",
        "美债与收益率": "对固收久期与再投资收益率构成直接影响",
        "企业财报与盈利": "对信用利差与股权相关资产的风险溢价存在边际影响",
        "衰退与宏观经济": "对信用质量与利率敏感资产的方向判断存在影响",
        "科技与半导体": "对股权相关资产集中度与波动性存在影响",
        "关税与贸易": "对外汇流动性、进口敏感行业信用存在传导可能",
        "原油与能源": "对能源板块敞口与通胀路径存在影响",
        "中国经济": "对区域/亚洲市场敞口与外汇敞口有间接影响",
        "房地产与信贷": "对房地产相关信用敞口与抵押类资产估值存在影响",
        "加密货币": "保险组合直接敞口有限，更多反映整体风险偏好",
    }

    implications = implications_by_rule.get(rule, "")
    if angle and not implications:
        # fallback: keep the angle but reframe as an observation
        implications = angle

    triggers = "继续观察讨论广度与跨平台扩散度，只有在伴随实际数据或利率变动确认时再考虑进一步动作"

    return InsuranceFramework(
        implications=implications,
        triggers=triggers,
    )


def render_insurance_framework(framework: InsuranceFramework) -> list[str]:
    """
    Render a two-layer insurance block into formatter-ready lines.
    Returns [] when both halves are empty.
    """
    lines: list[str] = []
    if framework.implications:
        lines.append(f"   配置含义：{framework.implications}")
    if framework.triggers:
        lines.append(f"   观察/触发条件：{framework.triggers}")
    return lines


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
            f"社区讨论整体与新闻主线一致，{len(overlapping)} 个主题与今日新闻重合（如：{sample}）"
        )
    elif overlapping:
        sentences.append(
            f"社区讨论与新闻主线部分重合，但讨论重点放在 {overlapping[0].headline or overlapping[0].rule_label} 的风险面"
        )
    elif headline_topics:
        sentences.append("社区讨论与今日新闻主线相关度较低，呈现另一条关注轨迹")

    if rising_only:
        sample = rising_only[0].headline or rising_only[0].rule_label
        sentences.append(
            f"值得注意的是，社区在『{sample}』上的讨论升温明显，新闻层面尚未同步突出"
        )

    if len(sentences) < 2 and len(community_only) >= 1 and not rising_only:
        sample = community_only[0].headline or community_only[0].rule_label
        sentences.append(f"另有社区关注点『{sample}』未进入今日新闻焦点，可作为延伸观察")

    return "；".join(sentences[:3]).strip()
