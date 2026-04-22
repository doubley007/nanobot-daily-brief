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
    '示例（仅供格式参考，不要照抄内容）：\n'
    '输入标题包含"Fed minutes dovish, Powell hints at Q2 cut"等\n'
    '合格输出：{\n'
    '  "real_topic": "FOMC 纪要转鸽推升二季度降息押注",\n'
    '  "discussion_focus": "多头认为CPI趋缓已到降息临界点；空头担心劳动市场仍紧、降息被推迟",\n'
    '  "market_relevance": "利好久期（10Y 利率下行空间），利好金矿与利率敏感股；对银行净息差形成压力",\n'
    '  "insurance_implications": "对久期管理可能打开延长窗口；对信用利差存在进一步收窄的方向压力；再投资收益率的走向需结合后续利率路径确认",\n'
    '  "insurance_triggers": "继续观察长端利率是否有效跌破4.10%；只有在伴随更软的通胀或劳动力数据时，才考虑真正延长久期"\n'
    '}\n'
    '反面示例（禁止）：\n'
    '- "对商品市场风险影响"（方向不明、和 real_topic 重复）\n'
    '- "延长固收久期0.3-0.5年"（证据链不够强时不应给出这种具体数字指令）\n'
    '- "关注信用利差和汇率变动"（没说方向、没说观察条件）\n'
    '- "市场关注美联储政策"（泛化总结）\n\n'
)


def _build_cluster_prompt(cluster: TopicCluster) -> str:
    titles = _top_titles(cluster.posts)
    titles_block = "\n".join(f"- {t}" for t in titles)
    platforms = ", ".join(cluster.platforms) or cluster.posts[0].platform
    rule_hint = cluster.rule_label or "未分类"

    return (
        "你是新加坡保险投资团队的社区分析师。任务不是复述标题，而是：\n"
        "(1) 指明具体催化剂；(2) 给出方向明确的大类资产含义；"
        "(3) 给出可执行的保险组合调整提示。\n\n"
        f"{_CLUSTER_PROMPT_EXAMPLE}"
        f"覆盖平台：{platforms}\n"
        f"关键词粗分类：{rule_hint}\n"
        f"帖子数量：{cluster.post_count}\n"
        f"是否热度上升：{'是' if cluster.is_rising else '否'}"
        f"（相对热度 {cluster.rise_ratio:.2f}x）\n"
        f"热门标题（按互动排序）：\n{titles_block}\n\n"
        "严格输出一个 JSON（不要 markdown、不要解释）。硬规则：\n"
        "- market_relevance 必须包含『利好/利空/施压/承压/收窄/走阔/上行/下行』等方向词，"
        "且必须点名至少一类资产（久期 / 股指 / 信用利差 / 商品 / 外汇）。\n"
        "- insurance_implications 写成『可能影响 / 对X存在压力 / 需结合Y确认』这类观察性语言，"
        "涵盖 固收久期 / 信用配置 / 再投资收益率 / 利率敏感资产 中的 1-2 项。"
        "不要给出具体的加减仓数字（例如『延长0.3年』『减5%』），除非 credibility_judgment >= 0.8 且证据非常直接。\n"
        "- insurance_triggers 必须指明『继续观察什么』以及『只有在什么条件成立时才考虑进一步动作』。\n"
        "- 禁止使用：『市场关注』『投资者讨论』『需留意』『构成影响』这类无方向表述。\n"
        "- 禁止把 real_topic 和 market_relevance 写成同一句话。\n"
        "- sentiment_dimensions 中，只有确实主导情绪时才给 >= 0.6 的分。"
        "若情绪确实平淡/分歧，各维度应在 0.3-0.5 之间，不要强行拔高。\n\n"
        "字段：\n"
        "{\n"
        '  "real_topic": "10-22字。具体事件或催化剂；若与粗分类不符请直接修正",\n'
        '  "discussion_focus": "25-50字。争论点 + 多空双方的分歧理由",\n'
        '  "reasons": "40-80字。支持情绪判断的 2-3 条依据，连写一段",\n'
        '  "sentiment_label": "bullish / bearish / neutral / mixed",\n'
        '  "sentiment_dimensions": {\n'
        '    "optimism": 0-1, "fear": 0-1, "uncertainty": 0-1,\n'
        '    "skepticism": 0-1, "hype": 0-1\n'
        "  },\n"
        '  "credibility_judgment": 0-1。1.0 = 有明确事件+数据+专业表述；0.0 = 纯情绪/meme,\n'
        '  "is_noise": true/false,\n'
        '  "should_include_in_brief": true/false。仅当对宏观/大类配置/风险判断有价值时 true,\n'
        '  "market_relevance": "30-50字。方向明确 + 点名资产类别。若 should_include=false 返回空字符串",\n'
        '  "insurance_implications": "40-70字。保险/配置含义，观察性语言，不给具体数字指令",\n'
        '  "insurance_triggers": "30-60字。继续观察的变量 + 触发进一步动作的条件"\n'
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
        f"   热度：{c.heat_score:.0f} ({c.post_count}贴) "
        f"{'（升温）' if c.is_rising else ''}\n"
        f"   情绪：{dims.label}，主导维度={top_dim}({dims.intensity:.2f})"
        f" [opt={dims.optimism:.2f} fear={dims.fear:.2f}"
        f" unc={dims.uncertainty:.2f} skep={dims.skepticism:.2f} hype={dims.hype:.2f}]\n"
        f"   可信度：{c.credibility.overall:.2f}"
        f"{'（噪音）' if c.credibility.is_noise else ''}\n"
        f"   争论点：{c.discussion_focus}\n"
        f"   保险角度：{c.insurance_angle or '—'}"
    )


def _build_analyst_prompt(
    clusters: list[TopicCluster],
    platforms: list[str],
    total_posts: int,
) -> str:
    cluster_block = "\n".join(_format_cluster_for_analyst(c, i + 1) for i, c in enumerate(clusters))
    single_platform = len(platforms) <= 1
    cross_platform_rule = (
        '  "cross_platform_signal": "今日仅覆盖 '
        f'{platforms[0] if platforms else "单一"}'
        '，无跨平台数据，请直接返回空字符串"'
        if single_platform
        else
        '  "cross_platform_signal": "40-80字。只有当≥2个平台指向同一话题时才写，'
        '必须点名具体话题和一致方向；若平台间实际分化也写明。若无法得出跨平台结论返回空字符串"'
    )

    return (
        "你是服务新加坡保险投资团队的社区分析师（Community Analyst）。"
        "以下是今天 Reddit/X/Discord 社区讨论经过聚类+多维情绪+可信度打分后的结构化结果。"
        "你的工作是：\n"
        "  - 挑出真正值得进日报的主题（headline_topics）\n"
        "  - 指出哪些是噪音（noise_topics）\n"
        "  - 用『情绪结构』而不是单一标签概括今天的讨论气氛\n"
        "  - 用方向明确、可执行的语言给出保险组合含义\n"
        "绝对不要编造未在结构化输入中出现的主题。\n\n"
        f"覆盖平台：{', '.join(platforms) if platforms else '无'}\n"
        f"总帖子数：{total_posts}\n"
        f"聚类数：{len(clusters)}\n\n"
        "结构化输入：\n"
        f"{cluster_block}\n\n"
        "硬规则：\n"
        "- headline_topics 必须至少有一个 cluster 的 credibility >= 0.5；"
        "若所有 cluster 都是噪音或可信度不足，返回空数组。\n"
        "- sentiment_structure 必须使用业务语言（如：观望为主+不确定性较高 / 分歧明显+担忧主导 / "
        "避险情绪升温），不要使用『分歧·主导X(0.70)』这种模型标签形式。\n"
        "- insurance_implications 用观察性语言描述『对久期/信用/再投资收益率/利率敏感资产的可能影响』，"
        "避免直接给出具体加减仓数字（如『延长0.3年』），除非多条 cluster 的 credibility 都 >= 0.7。\n"
        "- insurance_triggers 必须说明『继续观察什么变量』+『只有在什么条件成立时才考虑进一步动作』。\n"
        "- 禁止『市场关注』『值得留意』『构成影响』等无方向表述。\n\n"
        "严格输出 JSON：\n"
        "{\n"
        '  "headline_topics": [最多 3 个 cluster 序号的数字数组],\n'
        '  "noise_topics": [最多 3 个 cluster 序号的数字数组],\n'
        '  "sentiment_structure": "60-110字。业务语言描述整体情绪组合（如：观望为主，不确定性较高；'
        '分歧明显，担忧主导；避险情绪升温）。不要使用模型标签形式",\n'
        f"{cross_platform_rule},\n"
        '  "insurance_implications": "50-100字。保险组合各维度可能受影响的方向（久期/信用/再投资收益率/'
        '利率敏感资产中的 1-2 项），采用观察性语言，不给具体数字指令",\n'
        '  "insurance_triggers": "40-80字。继续观察的 1-2 个关键变量 + 触发进一步动作的条件（例如长端利率'
        '是否有效跌破/突破某一区间、情绪是否向X/主流媒体扩散）",\n'
        '  "brief_recommendation": "40-80字。告诉日报编辑今天社区部分应突出哪个主题、'
        '弱化哪个主题，说明理由"\n'
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
