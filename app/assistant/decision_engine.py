"""
Decision Engine —— 最核心模块。

设计思路：
  LLM 不负责"该不该买"这个判断本身，它只负责措辞和具象化解释。
  程序先基于四类结构化信号算出一个决策骨架：

    1. News 事实信号      —— 利多/利空计分 + 关键事件抽取
    2. Community 情绪信号 —— 多空分布 + 拥挤风险（来自 sentiment_aggregator）
    3. Trend 趋势信号     —— 7d/30d 收益 + 过热判断（来自 trend_signals）
    4. Risk 风险修正      —— 情绪过热时降档、情绪与事实背离时降 confidence、
                             证据不足时直接返回 hold_wait

  然后 LLM 只做一件事：把骨架翻译成人话（thesis 一两句、risks 2-3 条）。
  即使 LLM 不可用，骨架本身就能生成可读的最终答案。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Literal

from assistant.rag.store import CommunityDoc, NewsDoc
from assistant.sentiment_aggregator import AggregatedSentiment
from assistant.trend_signals import TrendSignal
from assistant.asset_taxonomy import asset_display

logger = logging.getLogger(__name__)


Action = Literal["buy", "buy_small", "hold_wait", "avoid_chasing", "avoid"]
Confidence = Literal["high", "medium", "low"]


EntryQuality = Literal["good", "medium", "poor"]


@dataclass
class DecisionScores:
    """Intermediate scores exposed in evidence for transparency / debugging."""
    direction_score: float      # -1..+1  (net directional signal)
    crowding_score: float       # 0..1    (how crowded/consensus the trade is)
    entry_quality: EntryQuality # good | medium | poor
    chasing_risk: str           # low | medium | high


@dataclass
class Decision:
    asset: str | None
    action: Action
    confidence: Confidence
    thesis: str
    evidence: dict[str, Any] = field(default_factory=dict)
    risks: list[str] = field(default_factory=list)
    suitable_for: str = ""
    one_line_advice: str = ""
    scores: DecisionScores | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


# ─── 1. News 信号打分 ────────────────────────────────────────────────────────

_BULLISH_HEADLINE_TRIGGERS = [
    ("rate cut", 0.4), ("降息", 0.4),
    ("geopolitical tension", 0.3), ("war", 0.3), ("sanction", 0.25),
    ("inflation hedge", 0.3), ("safe haven", 0.25),
    ("record high", 0.2), ("avaro", 0.15),
    ("利好", 0.25), ("大涨", 0.15), ("避险", 0.25),
]
_BEARISH_HEADLINE_TRIGGERS = [
    ("rate hike", 0.4), ("加息", 0.4),
    ("stronger dollar", 0.3), ("dollar rally", 0.3), ("usd strength", 0.25),
    ("risk on", 0.2), ("de-escalation", 0.2),
    ("利空", 0.25), ("下跌", 0.15),
]


@dataclass
class NewsAssessment:
    bullish_score: float
    bearish_score: float
    net_score: float
    key_bullets: list[str]

    @property
    def direction(self) -> str:
        if self.net_score > 0.2:
            return "bullish"
        if self.net_score < -0.2:
            return "bearish"
        return "neutral"


def _asset_specific_adjust(asset: str | None, text: str) -> float:
    """某些关键词对特定资产意义相反。例：美元走强 对黄金是 bearish，对美元自己是 bullish。"""
    if asset == "gold" and ("stronger dollar" in text or "usd strength" in text):
        return -0.2
    if asset == "usd" and ("rate cut" in text or "降息" in text):
        return -0.2
    return 0.0


def assess_news(asset: str | None, news: list[NewsDoc]) -> NewsAssessment:
    bullish = 0.0
    bearish = 0.0
    bullets: list[str] = []

    for d in news:
        text = f"{d.title} {d.raw_text}".lower()
        b_hits = [(kw, w) for kw, w in _BULLISH_HEADLINE_TRIGGERS if kw in text]
        s_hits = [(kw, w) for kw, w in _BEARISH_HEADLINE_TRIGGERS if kw in text]
        bullish += sum(w for _, w in b_hits)
        bearish += sum(w for _, w in s_hits)

        # 粗的 per-doc sentiment 也加权（importance 重的文章占比更大）
        weight = 0.2 + d.importance_score
        if d.sentiment == "bullish":
            bullish += 0.15 * weight
        elif d.sentiment == "bearish":
            bearish += 0.15 * weight

        adj = _asset_specific_adjust(asset, text)
        if adj > 0:
            bullish += adj
        else:
            bearish += -adj

        if b_hits or s_hits or d.sentiment in ("bullish", "bearish"):
            bullet = d.title.strip()
            if bullet and bullet not in bullets:
                bullets.append(bullet)

    net = bullish - bearish
    return NewsAssessment(
        bullish_score=round(bullish, 2),
        bearish_score=round(bearish, 2),
        net_score=round(net, 2),
        key_bullets=bullets[:5],
    )


# ─── 2. 社区情绪 → 方向分（直接用聚合器输出） ────────────────────────────────

def community_direction_score(agg: AggregatedSentiment) -> float:
    """返回 -1..+1。"""
    if agg.post_count == 0:
        return 0.0
    base = agg.bullish_ratio - agg.bearish_ratio
    # 拥挤高时降权（不是反方向，而是更不置信）
    if agg.crowded_trade_risk == "high":
        base *= 0.6
    elif agg.crowded_trade_risk == "medium":
        base *= 0.85
    return round(base, 2)


# ─── 3. 趋势分 ───────────────────────────────────────────────────────────────

def trend_direction_score(trend: TrendSignal) -> float:
    if trend.momentum_label == "up":
        return 0.5
    if trend.momentum_label == "down":
        return -0.5
    return 0.0


# ─── 4. 核心决策逻辑 ─────────────────────────────────────────────────────────

def _evidence_strength(news: NewsAssessment, agg: AggregatedSentiment,
                       trend: TrendSignal) -> float:
    """0-1，证据越多越强。用于判断是否要直接 hold_wait。"""
    s = 0.0
    if news.bullish_score + news.bearish_score >= 0.3:
        s += 0.4
    if agg.post_count >= 10:
        s += 0.3
    if trend.momentum_label in ("up", "down"):
        s += 0.3
    return min(1.0, s)


def _calc_crowding_score(agg: AggregatedSentiment, trend: TrendSignal) -> float:
    """
    0..1  —— 综合拥挤程度。
    来源：社区拥挤风险 + FOMO 比例 + conviction 比例 + 价格过热。
    纯粹描述"市场有多拥挤"，与方向无关。
    """
    base = {"high": 0.75, "medium": 0.45, "low": 0.15}.get(
        agg.crowded_trade_risk, 0.1
    )
    # FOMO 和 strong_conviction 都放大拥挤感
    fomo_adj = agg.fomo_ratio * 0.3
    conv_adj = agg.conviction_ratio * 0.2
    # 价格过热也加拥挤分
    heat_adj = {"high": 0.2, "medium": 0.1, "low": 0.0, "unknown": 0.0}.get(
        trend.overheating_risk, 0.0
    )
    return min(1.0, base + fomo_adj + conv_adj + heat_adj)


def _calc_entry_quality(
    direction_score: float,
    crowding_score: float,
    trend: TrendSignal,
) -> tuple[EntryQuality, str]:
    """
    把方向分和拥挤分组合成入场质量。
    返回 (entry_quality, chasing_risk)。

    设计原则：方向是否偏多（值得参与）和现在是否适合追入（时机）要拆开。
    一个资产可以方向很好（看多）但时机很差（刚暴涨一截、仓拥挤）。
    """
    # 方向强度（取绝对值，因为方向已在 direction_score 里带正负）
    dir_strength = abs(direction_score)

    # 价格过热单独映射到追高风险
    price_chasing = {"high": "high", "medium": "medium",
                     "low": "low", "unknown": "low"}.get(
        trend.overheating_risk, "low"
    )

    # 综合拥挤 + 价格过热 -> chasing_risk
    if crowding_score >= 0.65 or price_chasing == "high":
        chasing_risk = "high"
    elif crowding_score >= 0.35 or price_chasing == "medium":
        chasing_risk = "medium"
    else:
        chasing_risk = "low"

    # 入场质量：方向够强 + 追高风险低 -> good
    if dir_strength >= 0.4 and chasing_risk == "low":
        entry = "good"
    elif dir_strength >= 0.25 and chasing_risk != "high":
        entry = "medium"
    elif dir_strength >= 0.4 and chasing_risk == "medium":
        entry = "medium"
    else:
        entry = "poor"

    return entry, chasing_risk


def _trend_only_decision(
    trend: TrendSignal,
    reasons: list[str],
) -> tuple[Action, Confidence, list[str], DecisionScores]:
    """
    Fallback when evidence_strength < 0.3 but trend has real price data.
    Returns a trend-only decision with medium/low confidence.
    """
    r7 = trend.recent_return_7d
    mom = trend.momentum_label
    heat = trend.overheating_risk

    trend_dir = trend_direction_score(trend)
    crowding_score = {"high": 0.75, "medium": 0.45, "low": 0.15, "unknown": 0.1}.get(heat, 0.1)
    entry_quality: EntryQuality = "poor"
    chasing_risk = "low"

    if mom == "up":
        if heat == "high":
            action: Action = "avoid_chasing"
            chasing_risk = "high"
        elif heat == "medium":
            action = "buy_small"
            chasing_risk = "medium"
            entry_quality = "medium"
        else:
            action = "buy_small"
            entry_quality = "medium"
    elif mom == "down":
        action = "hold_wait" if (r7 is not None and r7 > -0.08) else "avoid"
    else:
        action = "hold_wait"

    scores = DecisionScores(
        direction_score=round(trend_dir, 3),
        crowding_score=round(crowding_score, 3),
        entry_quality=entry_quality,
        chasing_risk=chasing_risk,
    )
    reasons.append(
        f"trend-only fallback: momentum={mom}, r7={r7}, overheating={heat} -> {action}"
    )
    conf: Confidence = "medium" if mom in ("up", "down") else "low"
    return action, conf, reasons, scores


def _decide_action(
    news: NewsAssessment,
    agg: AggregatedSentiment,
    trend: TrendSignal,
) -> tuple[Action, Confidence, list[str], DecisionScores]:
    """
    返回 (action, confidence, internal_reasons, scores)。

    决策矩阵：
      direction strong  + entry good              -> buy
      direction strong  + entry medium            -> buy_small
      direction strong  + entry poor / chasing high -> avoid_chasing
      direction weak    + evidence thin           -> hold_wait
      direction bearish                           -> avoid / avoid_chasing
    """
    reasons: list[str] = []

    has_news = news.bullish_score + news.bearish_score >= 0.3
    has_community = agg.post_count >= 10
    has_real_trend = trend.data_source != "stub" and trend.recent_return_7d is not None

    strength = _evidence_strength(news, agg, trend)

    # When neither news nor community has data, full matrix can't give a useful signal
    # (trend weight is only 20% — direction_score stays near 0). Route to trend-only path.
    if not has_news and not has_community:
        reasons.append(f"evidence_strength={strength:.2f}, news+community absent")
        if has_real_trend:
            return _trend_only_decision(trend, reasons)
        scores = DecisionScores(
            direction_score=0.0, crowding_score=0.0,
            entry_quality="poor", chasing_risk="low",
        )
        return "hold_wait", "low", reasons, scores

    if strength < 0.3:
        reasons.append(f"evidence_strength={strength:.2f}, too thin — news/community data sparse")
        if has_real_trend:
            return _trend_only_decision(trend, reasons)
        scores = DecisionScores(
            direction_score=0.0, crowding_score=0.0,
            entry_quality="poor", chasing_risk="low",
        )
        return "hold_wait", "low", reasons, scores

    news_dir = 1 if news.direction == "bullish" else (-1 if news.direction == "bearish" else 0)
    com_dir = community_direction_score(agg)
    trend_dir = trend_direction_score(trend)

    # 加权合成方向分（-1..+1）
    direction_score = news_dir * 0.45 + com_dir * 0.35 + trend_dir * 0.2
    reasons.append(
        f"news={news.direction}({news.net_score:+.2f}), "
        f"community_bias={agg.overall_bias}({com_dir:+.2f}), "
        f"trend={trend.momentum_label}({trend_dir:+.2f}) "
        f"-> direction_score={direction_score:+.2f}"
    )

    # 情绪与事实背离（新闻明显利空但社区疯狂看多，或反之）
    divergence = (news.direction == "bearish" and com_dir > 0.3) \
                 or (news.direction == "bullish" and com_dir < -0.3)
    if divergence:
        reasons.append("fact/sentiment divergence detected")

    crowding_score = _calc_crowding_score(agg, trend)
    entry_quality, chasing_risk = _calc_entry_quality(direction_score, crowding_score, trend)
    reasons.append(
        f"crowding={crowding_score:.2f}, entry_quality={entry_quality}, "
        f"chasing_risk={chasing_risk}"
    )

    scores = DecisionScores(
        direction_score=round(direction_score, 3),
        crowding_score=round(crowding_score, 3),
        entry_quality=entry_quality,
        chasing_risk=chasing_risk,
    )

    # ── action 映射 ──────────────────────────────────────────────────────────
    action: Action
    if direction_score <= -0.4:
        action = "avoid"
    elif direction_score <= -0.15:
        action = "avoid_chasing"
    elif direction_score >= 0.45:
        # strong bullish direction — entry quality determines aggression
        if entry_quality == "good":
            action = "buy"
        elif entry_quality == "medium":
            action = "buy_small"
        else:  # poor entry (crowded / overheated)
            action = "avoid_chasing"
    elif direction_score >= 0.2:
        # moderate bullish
        if entry_quality == "good":
            action = "buy_small"
        elif chasing_risk == "high":
            action = "avoid_chasing"
        else:
            action = "buy_small"
    else:
        action = "hold_wait"

    # ── confidence ──────────────────────────────────────────────────────────
    if divergence or strength < 0.5:
        conf: Confidence = "low"
    elif strength >= 0.7 and abs(direction_score) >= 0.35 and chasing_risk != "high":
        conf = "high"
    else:
        conf = "medium"

    return action, conf, reasons, scores


# ─── 5. 基础 thesis / risks 生成（LLM 不可用时的兜底） ────────────────────────

_ACTION_TEMPLATES: dict[Action, str] = {
    "buy": "News and community sentiment on {name} both skew bullish, short-term trend confirms up — this is a spot to participate.",
    "buy_small": "{name} still skews bullish, but sentiment or price is showing signs of heat — better to scale in with a small position than go all-in.",
    "hold_wait": "Evidence on {name} isn't consistent enough and the signal itself isn't strong enough — better to stay flat and keep watching.",
    "avoid_chasing": "People are still buying {name}, but the short-term move is already large and sentiment is one-sided — the chase payoff is worse than the risk here.",
    "avoid": "News flow and sentiment on {name} both skew negative — this isn't a good entry window; step aside.",
}


def _fallback_thesis(
    action: Action,
    asset: str | None,
    trend: TrendSignal | None = None,
    agg: AggregatedSentiment | None = None,
    news_assess: NewsAssessment | None = None,
) -> str:
    name = asset_display(asset)

    # Build data fragments to inject into the template
    trend_fragment = ""
    if trend and trend.recent_return_7d is not None:
        r7 = trend.recent_return_7d * 100
        r7_str = f"{r7:+.1f}%"
        if trend.momentum_label == "up":
            trend_fragment = f"up {r7_str} over the last 7 days"
        elif trend.momentum_label == "down":
            trend_fragment = f"down {abs(r7):.1f}% over the last 7 days"
        else:
            trend_fragment = f"{r7_str} over the last 7 days, trend is choppy"

    community_fragment = ""
    if agg and agg.post_count > 0:
        bull_pct = int(agg.bullish_ratio * 100)
        bear_pct = int(agg.bearish_ratio * 100)
        community_fragment = f"across {agg.post_count} community posts, bulls {bull_pct}% / bears {bear_pct}%"

    news_fragment = ""
    if news_assess and (news_assess.bullish_score + news_assess.bearish_score) > 0.1:
        news_fragment = f"news flow {news_assess.direction} (net {news_assess.net_score:+.2f})"

    # Assemble contextual sentence
    data_parts = [p for p in [trend_fragment, community_fragment, news_fragment] if p]
    context = "; ".join(data_parts) + ". " if data_parts else ""

    base = _ACTION_TEMPLATES[action].format(name=name)
    if context:
        return context + base
    return base


def _fallback_risks(action: Action, agg: AggregatedSentiment,
                    trend: TrendSignal) -> list[str]:
    risks: list[str] = []
    if agg.crowded_trade_risk == "high":
        risks.append("Community sentiment is very one-sided — any counter-catalyst can trigger a fast reversal.")
    elif agg.crowded_trade_risk == "medium":
        risks.append("Sentiment is already leaning one way — keep a stop in mind if you chase.")
    if trend.overheating_risk == "high":
        risks.append("Short-term run has been large — drawdown risk on a chase is meaningful.")
    if agg.fomo_ratio >= 0.3:
        risks.append("Heavy FOMO in the community — win rate on emotional entries is lower than it looks.")
    if action in ("hold_wait", "avoid"):
        risks.append("Current fact + sentiment evidence isn't enough to justify sizing up.")
    if not risks:
        risks.append("Always keep a stop and room to scale in — don't size up on a single signal.")
    return risks[:3]


def _fallback_suitable_for(action: Action) -> str:
    mapping = {
        "buy": "holders with a medium-term view on the asset who can tolerate normal volatility",
        "buy_small": "investors who want to participate but are worried about chasing — scale-in setups",
        "hold_wait": "investors without pressing position needs who can wait for one or two confirming signals",
        "avoid_chasing": "existing holders can sit tight; no reason for new entries to chase here",
        "avoid": "not for short-term traders; longer-term allocators should wait for a better level too",
    }
    return mapping[action]


# ─── 6. LLM 改写层（只负责把骨架变成人话） ──────────────────────────────────

_LLM_THESIS_PROMPT = """You are an analyst with a view, but not a hype artist. Below is the decision skeleton the system has already computed, plus raw news details. Your job: rewrite the skeleton into a natural English explanation. Rules:

  1. Do NOT override the skeleton's action / confidence — only polish the language.
  2. Produce a thesis (2-3 sentences) that cites specific news events and sources from news_details. Do not write generic lines like "news flow is bullish"; reference what actually happened.
  3. Produce risks (up to 3) — one sentence each, concrete.
  4. Produce one_line_advice — a single, direct, actionable line.
  5. No hedge-speak ("depends on your risk appetite", "consult a professional", "not financial advice", etc.).
  6. The thesis should tell the user "because THIS specific thing happened", not just cite numbers and ratios.
  7. Respond in English.

Output strict JSON:
{{
  "thesis": "...",
  "risks": ["...", "..."],
  "one_line_advice": "..."
}}

Skeleton:
{skeleton}
"""


def _skeleton_payload(
    asset: str | None,
    action: Action,
    confidence: Confidence,
    news: NewsAssessment,
    agg: AggregatedSentiment,
    trend: TrendSignal,
    scores: DecisionScores,
) -> dict[str, Any]:
    return {
        "asset": asset_display(asset),
        "action": action,
        "confidence": confidence,
        "direction_score": scores.direction_score,
        "crowding_score": scores.crowding_score,
        "entry_quality": scores.entry_quality,
        "chasing_risk": scores.chasing_risk,
        "news_direction": news.direction,
        "news_net_score": news.net_score,
        "news_headlines": news.key_bullets,
        "community_overall_bias": agg.overall_bias,
        "community_bullish_ratio": agg.bullish_ratio,
        "community_bearish_ratio": agg.bearish_ratio,
        "community_fomo_ratio": agg.fomo_ratio,
        "community_crowded_trade_risk": agg.crowded_trade_risk,
        "community_summary": agg.summary,
        "trend_7d": trend.recent_return_7d,
        "trend_30d": trend.recent_return_30d,
        "trend_momentum": trend.momentum_label,
        "trend_overheating": trend.overheating_risk,
    }


def _llm_refine(
    skeleton: dict[str, Any],
    llm_callable: Callable[[str], str],
) -> dict[str, Any] | None:
    try:
        payload = json.dumps(skeleton, ensure_ascii=False, indent=2)
        raw = llm_callable(_LLM_THESIS_PROMPT.format(skeleton=payload))
        data = json.loads(raw)
    except Exception as e:
        logger.warning("decision_engine LLM refine failed: %s", e)
        return None

    thesis = str(data.get("thesis", "")).strip()
    risks = data.get("risks", [])
    if not isinstance(risks, list):
        risks = [str(risks)]
    risks = [str(r).strip() for r in risks if str(r).strip()][:3]
    advice = str(data.get("one_line_advice", "")).strip()

    if not thesis:
        return None
    return {"thesis": thesis, "risks": risks, "one_line_advice": advice}


# ─── 7. 对外入口 ─────────────────────────────────────────────────────────────

def make_decision(
    asset: str | None,
    news: list[NewsDoc],
    community_docs: list[CommunityDoc],
    trend: TrendSignal,
    sentiment_aggregate: AggregatedSentiment | None = None,
    llm_callable: Callable[[str], str] | None = None,
) -> Decision:
    """
    所有信号都要传进来；community_docs 用来生成 evidence 里的原话引用，
    sentiment_aggregate 用来走决策。没传 aggregate 就现场算。

    NOTE — emotion firewall:
    This function intentionally does NOT accept a UserEmotionProfile.
    User emotion (FOMO, anxious, …) is the caller's psychological state and
    must never influence the market decision.  It belongs only in
    reply_composer, which adjusts tone AFTER the decision is made.
    Market-side emotion signals come exclusively from community_docs /
    sentiment_aggregate (CommunityDoc.emotion_label = labels on third-party
    social posts, not the user's own message).
    """
    from assistant.sentiment_aggregator import aggregate as _aggregate
    if sentiment_aggregate is None:
        sentiment_aggregate = _aggregate(community_docs, asset=asset)

    news_assess = assess_news(asset, news)
    action, confidence, internal_reasons, scores = _decide_action(
        news_assess, sentiment_aggregate, trend,
    )

    # 默认兜底：非 LLM 也能产出可读回答
    thesis = _fallback_thesis(action, asset, trend=trend, agg=sentiment_aggregate, news_assess=news_assess)
    risks = _fallback_risks(action, sentiment_aggregate, trend)
    advice = {
        "buy": "Fine to participate, but keep a stop — don't chase after a big up-day.",
        "buy_small": "If you're going, take a small position and scale in; don't chase emotionally.",
        "hold_wait": "Stay flat — wait for a clearer catalyst or a pullback.",
        "avoid_chasing": "Don't chase here; existing positions can sit tight.",
        "avoid": "Step aside for now — let sentiment and price both cool off.",
    }[action]

    if llm_callable is not None:
        skeleton = _skeleton_payload(asset, action, confidence,
                                     news_assess, sentiment_aggregate, trend, scores)
        skeleton["news_details"] = [
            {
                "title": d.title,
                "source": d.source,
                "content": (d.summary or d.raw_text)[:280],
                "sentiment": d.sentiment,
            }
            for d in news[:5]
        ]
        refined = _llm_refine(skeleton, llm_callable)
        if refined:
            thesis = refined["thesis"] or thesis
            if refined["risks"]:
                risks = refined["risks"]
            if refined["one_line_advice"]:
                advice = refined["one_line_advice"]

    # 在 evidence 里附几条代表性社区原话（最多 3 条，按 confidence 排）
    sample_posts = sorted(
        community_docs, key=lambda d: (d.confidence, d.engagement_score),
        reverse=True,
    )[:3]
    community_samples = [
        {
            "platform": p.platform,
            "channel": p.channel_or_group,
            "text": p.raw_text[:180],
            "bull_bear": p.bullish_bearish_label,
            "emotion": p.emotion_label,
            "url": p.url,
        }
        for p in sample_posts
    ]

    evidence = {
        "news": [
            {
                "title": d.title,
                "source": d.source,
                "sentiment": d.sentiment,
                "url": d.url,
                "published_at": d.published_at,
                "snippet": (d.summary or d.raw_text)[:160],
                "retrieval_reason": (
                    f"asset_tag={asset}" if asset and asset in d.asset_tags
                    else "keyword_match"
                ),
            }
            for d in news[:5]
        ],
        "news_assessment": {
            "bullish_score": news_assess.bullish_score,
            "bearish_score": news_assess.bearish_score,
            "direction": news_assess.direction,
            "key_bullets": news_assess.key_bullets,
        },
        "community_aggregate": sentiment_aggregate.to_dict(),
        "community_samples": community_samples,
        "trend": trend.to_dict(),
        "decision_scores": {
            "direction_score": scores.direction_score,
            "crowding_score": scores.crowding_score,
            "entry_quality": scores.entry_quality,
            "chasing_risk": scores.chasing_risk,
        },
        "engine_trace": internal_reasons,
    }

    return Decision(
        asset=asset,
        action=action,
        confidence=confidence,
        thesis=thesis,
        evidence=evidence,
        risks=risks,
        suitable_for=_fallback_suitable_for(action),
        one_line_advice=advice,
        scores=scores,
    )
