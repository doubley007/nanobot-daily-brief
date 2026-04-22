from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from community.base import CommunitySentiment as _CS
    from community.schema import CommunityAnalystReport as _CAR


# =========================
# Data Models
# =========================

@dataclass
class MarketSnapshot:
    us_equities: Optional[str] = None
    rates: Optional[str] = None
    asia_sg: Optional[str] = None


@dataclass
class NewsItem:
    title: str
    summary: str
    source: str
    category: str = "general"
    importance_score: int = 0
    why_it_matters: str = ""
    url: Optional[str] = None
    published_at: Optional[str] = None


@dataclass
class BriefingInput:
    date_str: str
    market_snapshot: MarketSnapshot
    news_items: List[NewsItem] = field(default_factory=list)
    watchlist: List[str] = field(default_factory=list)
    community_sentiments: list = field(default_factory=list)  # list[CommunitySentiment]
    community_report: Optional["_CAR"] = None  # CommunityAnalystReport


# =========================
# Importance Scoring
# =========================

HIGH_PRIORITY_KEYWORDS = {
    "fed": 3,
    "federal reserve": 3,
    "cpi": 3,
    "inflation": 3,
    "payrolls": 3,
    "payroll": 3,
    "treasury": 2,
    "yield": 2,
    "yields": 2,
    "downgrade": 3,
    "default": 3,
    "credit": 2,
    "bank": 2,
    "banks": 2,
    "insurer": 2,
    "insurance": 2,
    "earnings": 2,
    "guidance": 2,
    "m&a": 2,
    "merger": 2,
    "acquisition": 2,
    "mas": 2,
    "singapore": 2,
    "tariff": 2,
    "sanction": 2,
    "oil": 1,
    "war": 2,
    "recession": 3,
    "ceasefire": 1,
}

CATEGORY_BONUS = {
    "macro": 3,
    "rates": 3,
    "credit": 3,
    "equity": 2,
    "singapore": 2,
    "asia": 1,
    "general": 0,
}

NEGATIVE_CONTEXT_KEYWORDS = {
    "probe": -2,
    "appeal": -2,
    "lawsuit": -2,
    "nominee": -1,
    "celebrity": -3,
    "gossip": -3,
    "ai model": -1,
    "spending plans": -1,
    "meta": -1,
    "peace": -1,
    "relevant parties": -2,
    "hopes": -1,
    "iran war": -1,
    "peace talks": -2,
    "diplomatic": -2,
    "statement": -1,
}

REGION_PENALTY_KEYWORDS = {
    "india": -2,
    "new zealand": -2,
    "australia": -1,
}


def score_news_item(item: NewsItem) -> int:
    text = f"{item.title} {item.summary}".lower()
    score = 0

    for keyword, weight in HIGH_PRIORITY_KEYWORDS.items():
        if keyword in text:
            score += weight

    for keyword, penalty in NEGATIVE_CONTEXT_KEYWORDS.items():
        if keyword in text:
            score += penalty

    for keyword, penalty in REGION_PENALTY_KEYWORDS.items():
        if keyword in text:
            score += penalty

    score += CATEGORY_BONUS.get(item.category.lower(), 0)
    return score


def assign_importance_label(score: int) -> str:
    if score >= 8:
        return "high"
    if score >= 4:
        return "medium"
    return "low"


# =========================
# News Processing
# =========================

def deduplicate_news_items(items: List[NewsItem]) -> List[NewsItem]:
    seen = set()
    unique_items = []

    for item in items:
        normalized_title = item.title.strip().lower()
        if normalized_title in seen:
            continue
        seen.add(normalized_title)
        unique_items.append(item)

    return unique_items


def detect_topic_for_selection(item: NewsItem) -> str:
    text = f"{item.title} {item.summary}".lower()

    if any(k in text for k in ["fed", "federal reserve", "rate cut", "rate cuts"]):
        return "fed_rates"

    if any(k in text for k in ["treasury", "yield", "yields", "bond"]):
        return "treasury_yields"

    if any(k in text for k in ["default", "downgrade", "credit", "liquidity", "spread"]):
        return "credit_risk"

    if any(k in text for k in ["bank", "banks", "lender", "lenders"]):
        return "banking"

    if any(k in text for k in ["earnings", "guidance", "revenue", "profit", "forecast"]):
        return "earnings"

    if any(k in text for k in ["mas", "singapore"]):
        return "singapore"

    if any(k in text for k in ["india", "new zealand", "australia"]):
        return "other_central_banks"

    return "general"


def select_diverse_top_items(items: List[NewsItem], top_n: int = 3) -> List[NewsItem]:
    """
    Pick top items with topic diversity.
    Prefer not to include multiple items from the same theme.
    """
    selected: List[NewsItem] = []
    used_topics = set()

    # First pass: keep only one per topic
    for item in items:
        topic = detect_topic_for_selection(item)
        if topic in used_topics:
            continue
        selected.append(item)
        used_topics.add(topic)
        if len(selected) >= top_n:
            return selected

    # Second pass: fill remaining slots if needed
    for item in items:
        if item in selected:
            continue
        selected.append(item)
        if len(selected) >= top_n:
            return selected

    return selected


def prepare_news_items(
    items: List[NewsItem],
    top_n: int = 3,
    llm_callable: Optional[Callable[[str], str]] = None,
) -> List[NewsItem]:
    """
    1. exact dedup
    2. score
    3. rank
    4. select diverse top items
    5. enrich selected items with fallback or LLM
    """
    from news_enricher import enrich_news_items

    items = deduplicate_news_items(items)

    for item in items:
        item.importance_score = score_news_item(item)

    items.sort(key=lambda x: x.importance_score, reverse=True)

    selected = select_diverse_top_items(items, top_n=top_n)

    if not selected:
        selected = items[:top_n]

    enriched = enrich_news_items(selected, llm_callable=llm_callable)

    return enriched


# =========================
# Market Takeaway Generator
# =========================

def generate_market_takeaway(
    market_snapshot: MarketSnapshot,
    top_news: List[NewsItem],
) -> str:
    themes = []

    if market_snapshot.rates and "暂时无法获取" not in market_snapshot.rates:
        themes.append("利率预期")
    if market_snapshot.us_equities and "暂时无法获取" not in market_snapshot.us_equities:
        themes.append("美股表现")
    if market_snapshot.asia_sg and "暂时无法获取" not in market_snapshot.asia_sg:
        themes.append("亚洲/新加坡动态")

    if top_news:
        top_title = top_news[0].title.lower()
        if any(k in top_title for k in ["fed", "cpi", "inflation", "payroll"]):
            themes.insert(0, "宏观数据")
        elif any(k in top_title for k in ["yield", "treasury", "rates"]):
            themes.insert(0, "债券收益率")
        elif any(k in top_title for k in ["earnings", "guidance", "profit"]):
            themes.insert(0, "公司业绩")
        elif any(k in top_title for k in ["singapore", "mas"]):
            themes.insert(0, "新加坡/区域信号")

    if not themes:
        return "今天市场整体较平静，重点更多在存量趋势延续。"

    unique_themes = []
    for t in themes:
        if t not in unique_themes:
            unique_themes.append(t)

    return f"今天市场关注点集中在“{' + '.join(unique_themes[:3])}”。"


# =========================
# Formatter
# =========================

def format_market_snapshot(snapshot: MarketSnapshot) -> str:
    us = snapshot.us_equities or "暂无明显异常波动"
    rates = snapshot.rates or "暂无明显异常变化"
    asia = snapshot.asia_sg or "暂无突出区域信号"

    return (
        f"📊 市场概览\n"
        f"- 美股：{us}\n"
        f"- 利率：{rates}\n"
        f"- 亚洲/新加坡：{asia}"
    )


def format_news_section(news_items: List[NewsItem], linked_reactions: Optional[dict] = None) -> str:
    if not news_items:
        return (
            "📰 今日重点\n"
            "1. 今日暂无特别突出的重大事件\n"
            "   - 影响：市场较平静时，观察存量趋势和后续数据窗口更重要。"
        )

    if linked_reactions is None:
        linked_reactions = {}

    lines = ["📰 今日重点"]
    for idx, item in enumerate(news_items, start=1):
      lines.append(f"{idx}. {item.title}")

      if item.summary:
        lines.append(f"   - 发生了什么：{item.summary}")

      if item.why_it_matters:
        impact_text = item.why_it_matters.strip()
        impact_text = impact_text.removeprefix("对组合影响：").strip()
        impact_text = impact_text.removeprefix("影响：").strip()
        lines.append(f"   - 对组合影响：{impact_text}")
      else:
        lines.append("   - 对组合影响：这条信息有助于判断当天组合风险偏好和市场主线变化。")

      # Linked community reaction
      reaction = linked_reactions.get(idx - 1)  # idx is 1-based, dict is 0-based
      if reaction:
        sent_cn = _SENTIMENT_CN.get(reaction.sentiment, "中性")
        angle = reaction.topic_label or reaction.discussion_focus
        line = f"   - 社区反应：{reaction.post_count} 条讨论 · 情绪{sent_cn}"
        if angle:
            line += f" · {angle}"
        lines.append(line)

    return "\n".join(lines)


_SENTIMENT_CN = {
    "bullish": "偏多",
    "bearish": "偏空",
    "neutral": "中性",
    "mixed": "分歧",
}


def _format_topic_lines(topic, idx: int, compact: bool = False) -> list[str]:
    """
    Render a single TopicCluster. `compact` trims reasoning lines when
    we only have one platform and the section would otherwise be
    dominated by a single noisy topic.

    Compared to the old renderer: this version leads with business-language
    trend + sentiment phrases (via community.verbalize) and only shows the
    numeric score tail as supplemental. It also renders the two-layer
    insurance framework (配置含义 + 观察/触发条件) instead of a single
    free-text 保险角度 line.
    """
    from community.verbalize import (
        derive_insurance_framework,
        render_insurance_framework,
        sentiment_score_tail,
        verbalize_sentiment,
        verbalize_trend,
    )

    headline = (getattr(topic, "headline", "") or "").strip()
    primary = headline or topic.rule_label or "（未命名主题）"
    platforms = "/".join(topic.platforms) if topic.platforms else ""

    # Meta chip: [reddit | 7贴 | 可信度 0.48]
    meta_parts = []
    if platforms:
        meta_parts.append(platforms)
    meta_parts.append(f"{topic.post_count}贴")
    meta_parts.append(f"可信度{topic.credibility.overall:.2f}")
    if topic.credibility.is_noise:
        meta_parts.append("噪音")
    meta = f"[{' | '.join(meta_parts)}]"

    header = f"{idx}. 📍 {primary}  {meta}"
    lines = [header]

    # Trend line — always show when we have a trend profile
    trend = getattr(topic, "trend", None)
    if trend is not None:
        trend_phrase = verbalize_trend(trend)
        if trend_phrase:
            lines.append(f"   趋势：{trend_phrase}")

    # Sentiment line — business phrase first, numeric tail in parens
    if not topic.credibility.is_noise:
        sent_phrase = verbalize_sentiment(topic.sentiment)
        tail = sentiment_score_tail(topic.sentiment)
        if sent_phrase:
            lines.append(f"   情绪：{sent_phrase}{tail}")

    if topic.discussion_focus:
        lines.append(f"   争论点：{topic.discussion_focus}")
    if topic.market_relevance:
        lines.append(f"   市场含义：{topic.market_relevance}")

    # Insurance framework (two-layer). Prefer LLM-provided; fall back to
    # derived observation framework so we never show a bare instruction.
    framework = getattr(topic, "insurance_framework", None)
    if framework is None or (not framework.implications and not framework.triggers):
        framework = derive_insurance_framework(topic)
    lines.extend(render_insurance_framework(framework))

    # Reasons line adds length without adding much signal when compact
    if not compact:
        reasons = (getattr(topic, "reasons", "") or "").strip()
        if reasons:
            lines.append(f"   理由：{reasons}")

    return lines


def format_community_section(
    sentiments: list,
    unlinked_topics: Optional[list] = None,
    analyst_report=None,
    news_social_bridge: Optional[str] = None,
) -> str:
    """
    Render the community section, driven by the Community Analyst report.

    Output shape:
      - Platform + post-count header
      - Sentiment structure paragraph (LLM cross-cluster read)
      - Cross-platform signal (when multiple platforms agree)
      - Top topics (analyst-selected headlines + any still unlinked)
      - Insurance angle
      - Editor recommendation
    """
    if not sentiments:
        return ""

    total_posts = sum(s.post_count for s in sentiments)
    platforms = [s.platform.capitalize() for s in sentiments if s.post_count > 0]
    platform_label = " + ".join(platforms) if platforms else "社区"

    # Pick topics to render: prefer analyst-selected headlines, but fall back
    # to unlinked or all when report is absent.
    if analyst_report is not None and analyst_report.headline_topics:
        primary_topics = analyst_report.headline_topics
    elif unlinked_topics is not None:
        primary_topics = unlinked_topics
    else:
        primary_topics = []
        for s in sentiments:
            primary_topics.extend(s.trending_topics)

    # When both the analyst picked headlines AND the news pipeline linked some
    # topics inline, keep the analyst's picks that are still unlinked.
    if unlinked_topics is not None and analyst_report is not None and analyst_report.headline_topics:
        unlinked_ids = {getattr(t, "cluster_id", id(t)) for t in unlinked_topics}
        primary_topics = [
            t for t in primary_topics
            if getattr(t, "cluster_id", id(t)) in unlinked_ids
        ]

    primary_topics = [
        t for t in primary_topics
        if getattr(t, "should_include_in_brief", True) and not t.credibility.is_noise
    ]

    if not primary_topics and not analyst_report:
        return ""

    # Signal summary: N 贴 · M 聚类 · K 入选 · L 噪音
    if analyst_report is not None:
        summary_parts = [f"{total_posts} 贴"]
        if analyst_report.total_clusters:
            summary_parts.append(f"{analyst_report.total_clusters} 聚类")
        summary_parts.append(f"{len(analyst_report.headline_topics)} 入选")
        if analyst_report.noise_topics:
            summary_parts.append(f"{len(analyst_report.noise_topics)} 噪音")
        signal_summary = " · ".join(summary_parts)
    else:
        signal_summary = f"{total_posts} 贴"

    lines = [
        "💬 社区情绪解读",
        f"{platform_label} · {signal_summary}",
    ]

    if analyst_report is not None:
        # Platform coverage disclosure — surfaces 402 / not-configured states
        coverage = [
            s for s in analyst_report.platform_status
            if s and not s.startswith(tuple(p.lower() + "=ok" for p in platforms))
        ]
        if coverage:
            lines.append(f"数据覆盖：{' | '.join(coverage)}")

        if analyst_report.sentiment_structure:
            lines.append(f"整体情绪：{analyst_report.sentiment_structure}")
        if analyst_report.cross_platform_signal:
            lines.append(f"跨平台信号：{analyst_report.cross_platform_signal}")

    # News ↔ Social bridge — bridges the news section and community topics.
    # Rendered after sentiment structure so reader sees the comparison
    # before diving into individual topics.
    bridge_text = news_social_bridge
    if not bridge_text and analyst_report is not None:
        bridge_text = analyst_report.news_social_bridge
    if bridge_text:
        lines.append(f"新闻 ↔ 社区：{bridge_text}")

    if primary_topics:
        compact = len(platforms) <= 1
        lines.append("")
        for idx, topic in enumerate(primary_topics, start=1):
            lines.extend(_format_topic_lines(topic, idx, compact=compact))

    if analyst_report is not None:
        framework = getattr(analyst_report, "insurance_framework", None)
        has_framework = framework is not None and (framework.implications or framework.triggers)
        if has_framework:
            lines.append("")
            lines.append("🛡 保险组合视角（观察框架）")
            if framework.implications:
                lines.append(f"   配置含义：{framework.implications}")
            if framework.triggers:
                lines.append(f"   观察/触发条件：{framework.triggers}")
        elif analyst_report.insurance_angle:
            # Legacy fallback — still render as observation, not instruction
            lines.append("")
            lines.append(f"🛡 保险组合视角：{analyst_report.insurance_angle}")
        if analyst_report.brief_recommendation:
            lines.append(f"📝 编辑建议：{analyst_report.brief_recommendation}")

    return "\n".join(lines).rstrip()


def format_watchlist(watchlist: List[str]) -> str:
    if not watchlist:
        watchlist = [
            "关注今晚是否有新的关键宏观数据",
            "关注美联储官员表态",
            "关注长端利率是否继续波动",
        ]

    lines = ["👀 后续关注"]
    for item in watchlist[:3]:
        lines.append(f"- {item}")
    return "\n".join(lines)


def format_daily_brief(
    briefing_input: BriefingInput,
    llm_callable: Optional[Callable[[str], str]] = None,
) -> str:
    prepared_news = prepare_news_items(
        briefing_input.news_items,
        top_n=3,
        llm_callable=llm_callable,
    )
    takeaway = generate_market_takeaway(briefing_input.market_snapshot, prepared_news)

    # ── News-to-community linking ──────────────────────────────────────
    linked_reactions: dict = {}
    unlinked_topics: Optional[list] = None
    news_social_bridge: str = ""

    all_community_topics = []
    for s in briefing_input.community_sentiments:
        all_community_topics.extend(s.trending_topics)

    if prepared_news and all_community_topics:
        from community.linker import link_news_to_community, get_unlinked_topics
        linked_reactions = link_news_to_community(prepared_news, all_community_topics)
        unlinked_topics = get_unlinked_topics(all_community_topics, linked_reactions)

    # ── News ↔ Social narrative bridge ────────────────────────────────
    # Short deterministic comparison between today's news and community
    # narratives. Produced here so both the community formatter and any
    # future consumer can reuse the same string.
    headline_topics = (
        briefing_input.community_report.headline_topics
        if briefing_input.community_report is not None
        else all_community_topics
    )
    if prepared_news and headline_topics:
        from community.verbalize import build_news_social_bridge
        news_social_bridge = build_news_social_bridge(
            prepared_news, headline_topics, len(linked_reactions)
        )
        if briefing_input.community_report is not None and not briefing_input.community_report.news_social_bridge:
            briefing_input.community_report.news_social_bridge = news_social_bridge

    # ── Assemble brief ────────────────────────────────────────────────
    parts = [
        f"📌 每日金融简报 | {briefing_input.date_str}",
        "",
        "一句话总结：",
        takeaway,
        "",
        format_market_snapshot(briefing_input.market_snapshot),
        "",
        format_news_section(prepared_news, linked_reactions),
    ]

    community_text = format_community_section(
        briefing_input.community_sentiments,
        unlinked_topics=unlinked_topics,
        analyst_report=briefing_input.community_report,
        news_social_bridge=news_social_bridge,
    )
    if community_text:
        parts.append("")
        parts.append(community_text)

    parts.extend([
        "",
        format_watchlist(briefing_input.watchlist),
        "",
        "#DailyBrief #Finance",
    ])

    return "\n".join(parts)


# =========================
# Example / Local Test
# =========================

if __name__ == "__main__":
    sample_input = BriefingInput(
        date_str=datetime.now().strftime("%Y-%m-%d"),
        market_snapshot=MarketSnapshot(
            us_equities="S&P 500 小幅上涨，Nasdaq 表现相对更强",
            rates="10Y Treasury yield 报 4.29%，较前一交易日下行 0.05 个百分点",
            asia_sg="STI 走势平稳",
        ),
        news_items=[
            NewsItem(
                title="Fed rate cut bets revived, a bit, by Iran war ceasefire - Reuters",
                summary="市场对年内降息的押注有所回升。",
                source="Reuters",
                category="macro",
            ),
            NewsItem(
                title="US bank profits to rise on deals, but Iran war fuels outlook uncertainty - Reuters",
                summary="银行盈利前景获得部分支撑，但地缘风险仍让市场保持谨慎。",
                source="Reuters",
                category="equity",
            ),
            NewsItem(
                title="Singapore financial sector sees stable outlook amid regional uncertainty",
                summary="在区域不确定性背景下，新加坡金融板块整体维持相对稳健预期。",
                source="Straits Times",
                category="singapore",
            ),
        ],
        watchlist=[
            "今晚是否有新的美国通胀或就业相关数据",
            "美债收益率是否继续波动",
            "亚洲市场对美国利率预期的跟随反应",
        ],
    )

    brief_text = format_daily_brief(sample_input)
    print(brief_text)