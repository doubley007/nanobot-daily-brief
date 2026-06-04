"""
LLM-driven community interpretation.

Three stages, each one LLM call:

  analyze_topic_cluster()        — per cluster. Produces multi-dim sentiment,
                                    credibility judgment, discussion focus,
                                    reasons, market relevance, insurance angle.

  score_cluster_credibility()    — deterministic signal-vs-noise helper
                                    (not an LLM call). Combines with LLM
                                    judgment inside analyze_topic_cluster.

  community_analyst_report()     — ingests all analyzed clusters and
                                    synthesizes a structured analyst report.

The existing llm_adapter.local_llm_callable(prompt) -> str (returning a
JSON string) is reused as-is.
"""
from __future__ import annotations

import json
import logging
from typing import Callable

from community.schema import (
    CommunityAnalystReport,
    CredibilityProfile,
    InsuranceFramework,
    SentimentProfile,
    TopicCluster,
    UnifiedPost,
)

logger = logging.getLogger(__name__)

MAX_TITLES_PER_PROMPT = 6
_VALID_SENTIMENTS = {"bullish", "bearish", "neutral", "mixed"}
_DIMENSIONS = ("optimism", "fear", "uncertainty", "skepticism", "hype")


# ─── Shared utilities ────────────────────────────────────────────────────────

def dedupe_posts(posts: list[UnifiedPost]) -> list[UnifiedPost]:
    """Drop posts with near-identical normalized titles, keep highest-engagement."""
    seen: dict[str, UnifiedPost] = {}
    for p in posts:
        key = " ".join(p.title.lower().split())[:120]
        if not key:
            continue
        prev = seen.get(key)
        if prev is None or p.engagement_raw > prev.engagement_raw:
            seen[key] = p
    return list(seen.values())


def _top_titles(posts: list[UnifiedPost], n: int = MAX_TITLES_PER_PROMPT) -> list[str]:
    ranked = sorted(posts, key=lambda p: p.engagement_raw, reverse=True)
    return [p.title.strip() for p in ranked[:n] if p.title.strip()]


def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


# ─── Deterministic credibility signals ───────────────────────────────────────

_SPECIFICITY_MARKERS = (
    "%", "bps", "yield", "cpi", "fomc", "earnings", "guidance",
    "q1", "q2", "q3", "q4", "billion", "million", "$", "rate",
    "fed", "treasury", "cpi", "pce", "jobs", "unemployment",
    "default", "downgrade", "upgrade", "ipo", "buyback", "merger",
)


def score_cluster_credibility(cluster: TopicCluster) -> CredibilityProfile:
    """
    Deterministic portion of signal-vs-noise scoring. Three 0-1 signals that
    the LLM cannot reliably compute: specificity markers, channel/author
    diversity, engagement-quality shape.
    """
    posts = cluster.posts
    if not posts:
        return CredibilityProfile()

    # 1. Specificity — concrete numbers / event vocabulary
    hits = 0
    total_tokens = 0
    for p in posts:
        text = p.text.lower()
        total_tokens += max(len(text.split()), 1)
        hits += sum(1 for m in _SPECIFICITY_MARKERS if m in text)
    specificity = min(hits / max(len(posts), 1) / 2.0, 1.0)

    # 2. Source diversity — distinct channels + authors
    channels = {p.channel for p in posts if p.channel}
    authors = {p.author for p in posts if p.author}
    diversity_raw = len(channels) + len(authors) * 0.5
    source_diversity = min(diversity_raw / 5.0, 1.0)

    # 3. Engagement quality — concentrated-on-one-viral-post is lower quality
    engagements = sorted((p.engagement_raw for p in posts), reverse=True)
    total_engagement = sum(engagements) or 1
    top_share = engagements[0] / total_engagement
    engagement_quality = 1.0 - (top_share - 0.4) if top_share > 0.4 else 1.0
    engagement_quality = _clamp01(engagement_quality)

    return CredibilityProfile(
        specificity=specificity,
        source_diversity=source_diversity,
        engagement_quality=engagement_quality,
    )


# ─── Prompt 1: per-cluster interpretation ────────────────────────────────────

_CLUSTER_PROMPT_EXAMPLE = (
    'Example (format reference only — do NOT copy content):\n'
    'Input titles include "Fed minutes dovish, Powell hints at Q2 cut" etc.\n'
    'A qualifying output:\n'
    '{\n'
    '  "real_topic": "FOMC minutes turn dovish, lifting Q2 rate-cut bets",\n'
    '  "discussion_focus": "Bulls argue softening CPI has reached the cut threshold; bears worry the labor market is still tight and any cut will slip",\n'
    '  "market_relevance": "Bullish for duration (room for 10Y to fall); bullish for gold miners and rate-sensitive equities; puts pressure on bank NIM",\n'
    '  "insurance_implications": "Potentially opens a duration-extension window; modest pressure toward tighter credit spreads; reinvestment-yield direction still needs confirmation from the forward rate path",\n'
    '  "insurance_triggers": "Keep watching whether the long end breaks cleanly below 4.10%; only consider actually extending duration on confirmation from softer inflation or labor data"\n'
    '}\n'
    'Counter-examples (forbidden):\n'
    '- "Effect on commodity-market risk" (direction unclear, duplicates real_topic)\n'
    '- "Extend fixed-income duration 0.3-0.5 years" (concrete number instruction not justified by the evidence)\n'
    '- "Watch credit spreads and FX moves" (no direction, no observation conditions)\n'
    '- "Market watching Fed policy" (generic summary)\n\n'
)


def _build_cluster_prompt(cluster: TopicCluster) -> str:
    titles = _top_titles(cluster.posts)
    titles_block = "\n".join(f"- {t}" for t in titles)
    platforms = ", ".join(cluster.platforms) or cluster.posts[0].platform
    rule_hint = cluster.rule_label or "uncategorized"

    return (
        "You are a community analyst for a Singapore insurance investment team. "
        "Your task is NOT to paraphrase titles — it is to: "
        "(1) name the specific catalyst; "
        "(2) give a directionally clear cross-asset read; "
        "(3) give actionable insurance-book observations. "
        "Respond in English.\n\n"
        f"{_CLUSTER_PROMPT_EXAMPLE}"
        f"Platforms covered: {platforms}\n"
        f"Coarse keyword bucket: {rule_hint}\n"
        f"Post count: {cluster.post_count}\n"
        f"Is chatter rising: {'yes' if cluster.is_rising else 'no'}"
        f" (relative heat {cluster.rise_ratio:.2f}x)\n"
        f"Top titles (by engagement):\n{titles_block}\n\n"
        "Strict JSON output (no markdown, no explanation). Hard rules:\n"
        "- market_relevance must include a direction word (bullish/bearish/pressures/supportive/tighten/widen/up/down) "
        "and name at least one asset class (duration / equity index / credit spread / commodity / FX).\n"
        "- insurance_implications must read like an observation ('may affect', 'pressure on X', 'needs to be read with Y'), "
        "covering 1-2 of: fixed-income duration, credit allocation, reinvestment yield, rate-sensitive assets. "
        "Do NOT give concrete size instructions (e.g. 'extend 0.3 years', 'cut 5%') unless credibility_judgment >= 0.8 AND evidence is very direct.\n"
        "- insurance_triggers must name 'what to keep watching' AND 'under what condition further action would be considered'.\n"
        "- Forbidden phrasing: 'the market is watching', 'investors are discussing', 'worth monitoring', 'has an impact' — anything without direction.\n"
        "- Do NOT write real_topic and market_relevance as the same sentence.\n"
        "- For sentiment_dimensions: only score >= 0.6 when the dimension genuinely dominates. "
        "If sentiment is actually flat or split, scores should sit in 0.3-0.5 — don't inflate.\n\n"
        "Fields (respond in English):\n"
        "{\n"
        '  "real_topic": "~10-20 words. Specific event or catalyst; correct the coarse bucket if it disagrees.",\n'
        '  "discussion_focus": "~25-50 words. The point of debate + the case from each side.",\n'
        '  "reasons": "~40-80 words. 2-3 reasons supporting your sentiment call, as one paragraph.",\n'
        '  "sentiment_label": "bullish / bearish / neutral / mixed",\n'
        '  "sentiment_dimensions": {\n'
        '    "optimism": 0-1, "fear": 0-1, "uncertainty": 0-1,\n'
        '    "skepticism": 0-1, "hype": 0-1\n'
        "  },\n"
        '  "credibility_judgment": 0-1. 1.0 = clear event + data + professional language; 0.0 = pure emotion / meme,\n'
        '  "is_noise": true/false,\n'
        '  "should_include_in_brief": true/false. True only if it adds value for macro / cross-asset / risk reads,\n'
        '  "market_relevance": "~30-50 words. Directionally clear + asset class named. Return empty string when should_include=false.",\n'
        '  "insurance_implications": "~40-70 words. Insurance / allocation read, observation language, no concrete-size instructions.",\n'
        '  "insurance_triggers": "~30-60 words. What to keep watching + the condition that would justify further action."\n'
        "}"
    )


def analyze_topic_cluster(
    cluster: TopicCluster,
    llm_callable: Callable[[str], str],
) -> TopicCluster | None:
    """
    One LLM call. Writes interpretation + multi-dim sentiment + LLM
    credibility judgment onto the cluster. Combines LLM judgment with
    deterministic credibility signals into a composite `credibility.overall`.
    """
    prompt = _build_cluster_prompt(cluster)
    try:
        raw = llm_callable(prompt)
        data = json.loads(raw)
    except Exception as e:
        logger.warning("Cluster LLM analysis failed (%s): %s", cluster.rule_label, e)
        return None

    cluster.headline = (data.get("real_topic") or "").strip()
    cluster.discussion_focus = (data.get("discussion_focus") or "").strip()
    cluster.reasons = (data.get("reasons") or "").strip()
    cluster.market_relevance = (data.get("market_relevance") or "").strip()
    # Two-layer insurance framework (preferred). Keep legacy `insurance_angle`
    # populated by concatenation so older formatters still work.
    implications = (data.get("insurance_implications") or "").strip()
    triggers = (data.get("insurance_triggers") or "").strip()
    # Backward compat: some prompts may still return a single field.
    legacy_angle = (data.get("insurance_angle") or "").strip()
    if not implications and legacy_angle:
        implications = legacy_angle
    cluster.insurance_framework = InsuranceFramework(
        implications=implications,
        triggers=triggers,
    )
    cluster.insurance_angle = " ".join(p for p in [implications, triggers] if p).strip() or legacy_angle
    cluster.should_include_in_brief = bool(data.get("should_include_in_brief", False))

    # Multi-dim sentiment
    dims = data.get("sentiment_dimensions", {}) or {}
    profile = SentimentProfile(
        label=data.get("sentiment_label", "neutral") if data.get("sentiment_label") in _VALID_SENTIMENTS else "neutral",
        optimism=_clamp01(dims.get("optimism", 0)),
        fear=_clamp01(dims.get("fear", 0)),
        uncertainty=_clamp01(dims.get("uncertainty", 0)),
        skepticism=_clamp01(dims.get("skepticism", 0)),
        hype=_clamp01(dims.get("hype", 0)),
    )
    dim_values = {d: getattr(profile, d) for d in _DIMENSIONS}
    top_dim = max(dim_values.items(), key=lambda kv: kv[1])
    if top_dim[1] > 0:
        profile.dominant_dimension = top_dim[0]
        profile.intensity = top_dim[1]
    cluster.sentiment = profile

    # Credibility: merge LLM judgment with deterministic signals
    det = score_cluster_credibility(cluster)
    llm_judgment = _clamp01(data.get("credibility_judgment", 0))
    overall = (
        det.specificity * 0.25
        + det.source_diversity * 0.15
        + det.engagement_quality * 0.15
        + llm_judgment * 0.45
    )
    cluster.credibility = CredibilityProfile(
        specificity=det.specificity,
        source_diversity=det.source_diversity,
        engagement_quality=det.engagement_quality,
        llm_judgment=llm_judgment,
        overall=overall,
        is_noise=bool(data.get("is_noise", False)) or overall < 0.3,
    )

    if not cluster.headline:
        logger.info("LLM returned no headline for cluster — dropping")
        return None
    return cluster


# ─── Prompt 2: Community Analyst synthesis ───────────────────────────────────

def _format_cluster_for_analyst(c: TopicCluster, idx: int) -> str:
    dims = c.sentiment
    top_dim = dims.dominant_dimension or "neutral"
    return (
        f"{idx}. [{', '.join(c.platforms)}] {c.headline or c.rule_label}\n"
        f"   Heat: {c.heat_score:.0f} ({c.post_count} posts) "
        f"{'(rising)' if c.is_rising else ''}\n"
        f"   Sentiment: {dims.label}, dominant={top_dim}({dims.intensity:.2f})"
        f" [opt={dims.optimism:.2f} fear={dims.fear:.2f}"
        f" unc={dims.uncertainty:.2f} skep={dims.skepticism:.2f} hype={dims.hype:.2f}]\n"
        f"   Credibility: {c.credibility.overall:.2f}"
        f"{' (noise)' if c.credibility.is_noise else ''}\n"
        f"   Debate: {c.discussion_focus}\n"
        f"   Insurance angle: {c.insurance_angle or '—'}"
    )


def _build_analyst_prompt(
    clusters: list[TopicCluster],
    platforms: list[str],
    total_posts: int,
) -> str:
    cluster_block = "\n".join(_format_cluster_for_analyst(c, i + 1) for i, c in enumerate(clusters))
    single_platform = len(platforms) <= 1
    cross_platform_rule = (
        '  "cross_platform_signal": "Today we cover only '
        f'{platforms[0] if platforms else "a single"}'
        ' — no cross-platform data; return empty string."'
        if single_platform
        else
        '  "cross_platform_signal": "~40-80 words. Only write when >=2 platforms point to the same topic. '
        'Must name the specific topic and the consistent direction; if platforms actually diverge, state that. '
        'Return empty string if no cross-platform read can be drawn."'
    )

    return (
        "You are the Community Analyst serving a Singapore insurance investment team. "
        "Below are structured results for today's Reddit/X/Discord discussion after clustering, "
        "multi-dimensional sentiment scoring, and credibility scoring. Respond in English.\n\n"
        "Your job is to:\n"
        "  - pick the topics that genuinely belong in the daily brief (headline_topics);\n"
        "  - flag the noise (noise_topics);\n"
        "  - summarize today's discussion with a 'sentiment structure' rather than a single label;\n"
        "  - articulate insurance-book implications in direction-clear, actionable language.\n"
        "Do NOT invent topics that are not in the structured input.\n\n"
        f"Platforms covered: {', '.join(platforms) if platforms else 'none'}\n"
        f"Total posts: {total_posts}\n"
        f"Clusters: {len(clusters)}\n\n"
        "Structured input:\n"
        f"{cluster_block}\n\n"
        "Hard rules:\n"
        "- headline_topics must include at least one cluster with credibility >= 0.5; "
        "if every cluster is noise or below threshold, return an empty array.\n"
        "- sentiment_structure must use business language (e.g. 'wait-and-see with high uncertainty', "
        "'split with fear dominating', 'risk-off tone building'). Do NOT use model-label syntax like 'mixed·fear(0.70)'.\n"
        "- insurance_implications should read as observation-language covering the likely effect on "
        "duration / credit / reinvestment yields / rate-sensitive assets. "
        "Avoid concrete size instructions (e.g. 'extend 0.3 years') unless multiple clusters score credibility >= 0.7.\n"
        "- insurance_triggers must name 'what variable(s) to keep watching' AND 'under what condition further action would be considered'.\n"
        "- Forbidden: 'the market is watching', 'worth monitoring', 'has an impact' — anything without direction.\n\n"
        "Strict JSON output (respond in English):\n"
        "{\n"
        '  "headline_topics": [up to 3 cluster index numbers],\n'
        '  "noise_topics": [up to 3 cluster index numbers],\n'
        '  "sentiment_structure": "~60-110 words. Business-language description of the overall sentiment mix '
        '(e.g. wait-and-see with high uncertainty; split with fear dominating; risk-off tone building). '
        'Do NOT use model-label syntax.",\n'
        f"{cross_platform_rule},\n"
        '  "insurance_implications": "~50-100 words. Likely directional effect on 1-2 of '
        '(duration / credit / reinvestment yield / rate-sensitive assets). Observation language, no concrete-size instructions.",\n'
        '  "insurance_triggers": "~40-80 words. 1-2 key variables to keep watching + the condition that would '
        'trigger further action (e.g. whether the long end breaks through a specific band; whether sentiment spreads to X / mainstream media).",\n'
        '  "brief_recommendation": "~40-80 words. Tell the editor which community topic to lead with today, '
        'which to de-emphasize, and why."\n'
        "}"
    )


def community_analyst_report(
    clusters: list[TopicCluster],
    platforms: list[str],
    total_posts: int,
    llm_callable: Callable[[str], str],
) -> CommunityAnalystReport:
    """
    Second-pass LLM call. Returns a populated CommunityAnalystReport.
    On failure, returns an empty report so the daily brief can hide
    the community section without crashing.
    """
    report = CommunityAnalystReport(
        platforms_covered=platforms,
        total_posts=total_posts,
    )
    if not clusters:
        return report

    prompt = _build_analyst_prompt(clusters, platforms, total_posts)
    try:
        raw = llm_callable(prompt)
        data = json.loads(raw)
    except Exception as e:
        logger.warning("Community Analyst LLM call failed: %s", e)
        return report

    def _pick(indices: list) -> list[TopicCluster]:
        out = []
        for i in indices or []:
            try:
                idx = int(i) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(clusters):
                out.append(clusters[idx])
        return out

    report.headline_topics = _pick(data.get("headline_topics", []))
    report.noise_topics = _pick(data.get("noise_topics", []))
    report.sentiment_structure = (data.get("sentiment_structure") or "").strip()
    report.cross_platform_signal = (data.get("cross_platform_signal") or "").strip()
    implications = (data.get("insurance_implications") or "").strip()
    triggers = (data.get("insurance_triggers") or "").strip()
    legacy_angle = (data.get("insurance_angle") or "").strip()
    if not implications and legacy_angle:
        implications = legacy_angle
    report.insurance_framework = InsuranceFramework(
        implications=implications,
        triggers=triggers,
    )
    report.insurance_angle = " ".join(p for p in [implications, triggers] if p).strip() or legacy_angle
    report.brief_recommendation = (data.get("brief_recommendation") or "").strip()
    return report


# ─── End-to-end pipeline entry (unified schema) ──────────────────────────────

def run_llm_pipeline(
    platform: str,
    posts: list[UnifiedPost],
    clusters: list[TopicCluster],
    top_n_topics: int,
    llm_callable: Callable[[str], str] | None,
) -> tuple[list[TopicCluster], str]:
    """
    Shared per-platform LLM pass. Takes already-clustered UnifiedPosts,
    LLM-analyzes each cluster (multi-dim sentiment + credibility), drops
    noise + low-signal clusters, returns kept clusters + legacy overall
    sentiment label.
    """
    if llm_callable is None or not clusters:
        return [], "neutral"

    kept: list[TopicCluster] = []
    for cluster in clusters[: max(top_n_topics * 2, top_n_topics)]:
        analyzed = analyze_topic_cluster(cluster, llm_callable)
        if analyzed is None:
            continue
        if not analyzed.should_include_in_brief and analyzed.credibility.is_noise:
            continue
        kept.append(analyzed)
        if len([k for k in kept if k.should_include_in_brief]) >= top_n_topics:
            break

    if not kept:
        return [], "neutral"

    # Legacy overall sentiment label (weighted by heat, non-noise only)
    labels: dict[str, float] = {"bullish": 0.0, "bearish": 0.0, "neutral": 0.0, "mixed": 0.0}
    for c in kept:
        if c.credibility.is_noise:
            continue
        labels[c.sentiment.label] = labels.get(c.sentiment.label, 0.0) + c.heat_score
    overall = max(labels.items(), key=lambda kv: kv[1])[0] if any(labels.values()) else "neutral"

    return kept, overall
