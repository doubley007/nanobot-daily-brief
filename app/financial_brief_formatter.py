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
    singapore_extended: Optional[str] = None  # SGD/USD, SG bond yield, DBS/OCBC/UOB


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
    # portfolio_relevance controls how the "影响" line is rendered.
    # "macro"   — rates / credit / real estate / insurance allocation signal
    # "sector"  — sector-level event, some spillover
    # "low"     — single-stock or low-relevance item, don't force a
    #             portfolio-impact framing
    portfolio_relevance: str = "macro"


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


_MACRO_RELEVANCE_KEYWORDS = {
    # macro / rates
    "fed", "federal reserve", "fomc", "rate cut", "rate cuts", "rate hike",
    "interest rate", "interest rates", "central bank",
    "treasury", "yield", "yields", "bond", "bonds", "10y", "10-year",
    "cpi", "inflation", "pce", "payroll", "payrolls", "unemployment",
    "recession", "gdp",
    # credit / funding
    "credit", "spread", "spreads", "liquidity", "default", "downgrade",
    "refinancing", "bank failure", "bank stress",
    # real estate / loans / insurance allocation
    "real estate", "commercial real estate", "property",
    "corporate loans", "insurance regulation", "capital requirement",
    "solvency", "reinvestment",
    # sg / fx / macro cross
    "mas", "singapore", "sgd", "fx", "foreign exchange",
    # broad market / allocation
    "broad market", "index rebalance", "systemic",
    # geopolitics / trade / sanctions (macro risk channel)
    "sanctions", "tariff", "tariffs", "trade war", "trade deal",
}

_SECTOR_RELEVANCE_KEYWORDS = {
    "bank", "banks", "lender", "lenders", "insurer", "insurance",
    "sector", "industry",
    "oil", "crude", "opec", "commodity", "commodities", "energy",
    "semiconductor", "chip", "chips",
}


def _word_set(text: str) -> set[str]:
    """Lowercased bag of word tokens — for word-boundary keyword matching."""
    cleaned = []
    for ch in text.lower():
        cleaned.append(ch if ch.isalnum() else " ")
    return set("".join(cleaned).split())


_SINGLE_STOCK_PATTERN_HINTS = (
    # Common single-stock noise patterns — if any of these appear in the
    # title, we treat the item as "low" regardless of upstream category.
    "stock surging", "stock surge", "stock jumps", "stock soars",
    "stock plunges", "stock drops", "stock falls", "stock rises",
    "ai downgrades", "ai upgrade", "ai upgrades",
    "why is", "why are", "why does",
    "what investors", "price target", "top stock",
    "moved up by", "moved down by",
    "could soar", "could jump", "could rally", "could surge",
    "undervalued", "overvalued", "bull case", "bear case",
)


def classify_portfolio_relevance(item: NewsItem) -> str:
    """
    Decide whether a news item carries macro / sector / low relevance for
    an insurance-investment book. Called once, before rendering.

    "macro"  → deserves a portfolio-impact sentence.
    "sector" → deserves a short sector-level read.
    "low"    → single-stock or low-relevance; the formatter will NOT
               force a "对组合影响" line.

    Precedence:
      1. Title matches a clear single-stock noise pattern → "low"
         (even when the upstream source tagged it as "macro")
      2. Macro keyword in title/summary → "macro"
      3. Sector keyword in title/summary → "sector"
      4. Category hint (macro/credit/singapore/regulation, etc.)
      5. Otherwise → "low"

    Rule #1 exists because sources like Alpha Vantage tag news with
    broad topic labels ("Financial Markets" → category="macro") even for
    pure single-stock headlines. Without this gate, items like
    "Why Is Micron Stock Surging" would inherit a macro-bucket treatment.
    """
    raw = f"{item.title} {item.summary}".lower()
    title_lower = item.title.lower()
    tokens = _word_set(raw)

    def _hits(keywords: set[str]) -> bool:
        for kw in keywords:
            if " " in kw or "-" in kw:
                if kw in raw:
                    return True
            elif kw in tokens:
                return True
        return False

    # Rule 1: obvious single-stock noise patterns veto any macro promotion
    if any(p in title_lower for p in _SINGLE_STOCK_PATTERN_HINTS):
        # Still allow sector promotion when the title carries an explicit
        # sector cue — "Tesla (TSLA) stock surges on oil prices" is
        # primarily a sector story. Pure single-stock → low.
        if _hits(_SECTOR_RELEVANCE_KEYWORDS):
            return "sector"
        return "low"

    # Rule 2: macro keyword in text
    if _hits(_MACRO_RELEVANCE_KEYWORDS):
        return "macro"

    # Rule 3: sector keyword in text
    if _hits(_SECTOR_RELEVANCE_KEYWORDS):
        return "sector"

    # Rule 4: category hint (treat as a weak default when text has no
    # explicit keyword match). Note "equity" is NOT a sector hint — see
    # the rule 1 rationale above.
    macro_categories = {"rates", "credit", "singapore", "regulation"}
    sector_only_categories = {"banking", "insurance", "commodities"}
    cat = (item.category or "").lower()
    if cat in macro_categories:
        return "macro"
    if cat in sector_only_categories:
        return "sector"

    # Rule 5: default — low relevance
    return "low"


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

    # Classify relevance BEFORE selection so we can prefer macro/sector
    # items and let "low" items only act as fillers. This matches the
    # Great Eastern SG insurance use-case: rates / credit / SG / macro
    # trumps single-stock noise.
    for it in items:
        it.portfolio_relevance = classify_portfolio_relevance(it)

    macro_pool = [it for it in items if it.portfolio_relevance == "macro"]
    sector_pool = [it for it in items if it.portfolio_relevance == "sector"]
    low_pool = [it for it in items if it.portfolio_relevance == "low"]

    # Run the topic-diversity selector over the ranked macro+sector pool
    # first. Only fall back to low-relevance items if we can't fill top_n.
    preferred_pool = macro_pool + sector_pool
    selected = select_diverse_top_items(preferred_pool, top_n=top_n)

    if len(selected) < top_n:
        # Top-up from low pool as last resort — don't let a quiet day
        # force the brief into single-stock noise.
        for it in low_pool:
            if it in selected:
                continue
            selected.append(it)
            if len(selected) >= top_n:
                break

    if not selected:
        selected = items[:top_n]

    # Keep pre-enrichment relevance since enricher rewrites summary.
    pre_relevance = {id(it): it.portfolio_relevance for it in selected}
    enriched = enrich_news_items(selected, llm_callable=llm_callable)
    for item in enriched:
        item.portfolio_relevance = pre_relevance.get(
            id(item), classify_portfolio_relevance(item)
        )

    return enriched


# =========================
# Market Takeaway Generator
# =========================

def generate_market_takeaway(
    market_snapshot: MarketSnapshot,
    top_news: List[NewsItem],
) -> str:
    themes = []

    if market_snapshot.rates and "data unavailable" not in market_snapshot.rates:
        themes.append("rate expectations")
    if market_snapshot.us_equities and "data unavailable" not in market_snapshot.us_equities:
        themes.append("US equities")
    if market_snapshot.asia_sg and "data unavailable" not in market_snapshot.asia_sg:
        themes.append("Asia / Singapore")

    if top_news:
        top_title = top_news[0].title.lower()
        if any(k in top_title for k in ["fed", "cpi", "inflation", "payroll"]):
            themes.insert(0, "macro data")
        elif any(k in top_title for k in ["yield", "treasury", "rates"]):
            themes.insert(0, "bond yields")
        elif any(k in top_title for k in ["earnings", "guidance", "profit"]):
            themes.insert(0, "corporate earnings")
        elif any(k in top_title for k in ["singapore", "mas"]):
            themes.insert(0, "Singapore / regional signal")

    if not themes:
        return "Market tone is subdued today; focus stays on the continuation of existing trends."

    unique_themes = []
    for t in themes:
        if t not in unique_themes:
            unique_themes.append(t)

    return f"Today's market focus is on {' + '.join(unique_themes[:3])}."


# =========================
# Formatter
# =========================

def format_market_snapshot(snapshot: MarketSnapshot) -> str:
    us = snapshot.us_equities or "no notable moves"
    rates = snapshot.rates or "no notable changes"
    asia = snapshot.asia_sg or "no standout regional signal"

    lines = [
        "📊 Market Snapshot",
        f"- US equities: {us}",
        f"- Rates: {rates}",
        f"- Asia / Singapore: {asia}",
    ]

    sg_ext = getattr(snapshot, "singapore_extended", "") or ""
    if sg_ext:
        lines.append(f"- Singapore (extended): {sg_ext}")

    return "\n".join(lines)


def format_news_section(news_items: List[NewsItem], linked_reactions: Optional[dict] = None) -> str:
    if not news_items:
        return (
            "📰 Top stories\n"
            "1. No standout events today\n"
            "   - Read: quiet sessions call for a closer watch on existing trends and upcoming data."
        )

    if linked_reactions is None:
        linked_reactions = {}

    lines = ["📰 Top stories"]
    for idx, item in enumerate(news_items, start=1):
      lines.append(f"{idx}. {item.title}")

      if item.summary:
        lines.append(f"   - What happened: {item.summary}")

      relevance = getattr(item, "portfolio_relevance", "macro")

      if relevance == "low":
        lines.append("   - Portfolio relevance: single-stock or low-relevance item — limited bearing on the insurance book, not flagged for core tracking.")
      elif item.why_it_matters:
        impact_text = item.why_it_matters.strip()
        impact_text = impact_text.removeprefix("Portfolio impact:").strip()
        impact_text = impact_text.removeprefix("Impact:").strip()
        label = "Portfolio impact" if relevance == "macro" else "Sector read"
        lines.append(f"   - {label}: {impact_text}")
      else:
        if relevance == "macro":
          lines.append("   - Portfolio impact: relevant for today's risk-appetite read and the dominant market theme.")
        else:
          lines.append("   - Sector read: sector-level signal with limited spillover to the overall book.")

      # Linked community reaction
      reaction = linked_reactions.get(idx - 1)  # idx is 1-based, dict is 0-based
      if reaction:
        sent_en = _SENTIMENT_EN.get(reaction.sentiment, "neutral")
        angle = reaction.topic_label or reaction.discussion_focus
        line = f"   - Community reaction: {reaction.post_count} posts · sentiment {sent_en}"
        if angle:
            line += f" · {angle}"
        lines.append(line)

    return "\n".join(lines)


_SENTIMENT_EN = {
    "bullish": "bullish",
    "bearish": "bearish",
    "neutral": "neutral",
    "mixed": "mixed",
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
        align_insurance_framework,
        derive_insurance_framework,
        render_insurance_framework,
        sentiment_score_tail,
        soften_market_implication,
        verbalize_sentiment,
        verbalize_trend,
    )

    headline = (getattr(topic, "headline", "") or "").strip()
    primary = headline or topic.rule_label or "(unnamed topic)"
    platforms = "/".join(topic.platforms) if topic.platforms else ""

    # Meta chip: [reddit | 7 posts | cred 0.48]
    meta_parts = []
    if platforms:
        meta_parts.append(platforms)
    meta_parts.append(f"{topic.post_count} posts")
    meta_parts.append(f"cred {topic.credibility.overall:.2f}")
    if topic.credibility.is_noise:
        meta_parts.append("noise")
    meta = f"[{' | '.join(meta_parts)}]"

    header = f"{idx}. 📍 {primary}  {meta}"
    lines = [header]

    # Trend line — always show when we have a trend profile
    trend = getattr(topic, "trend", None)
    if trend is not None:
        trend_phrase = verbalize_trend(trend)
        if trend_phrase:
            lines.append(f"   Trend: {trend_phrase}")

    # Sentiment line — business phrase first, numeric tail in parens
    if not topic.credibility.is_noise:
        sent_phrase = verbalize_sentiment(topic.sentiment)
        tail = sentiment_score_tail(topic.sentiment)
        if sent_phrase:
            lines.append(f"   Sentiment: {sent_phrase}{tail}")

    if topic.discussion_focus:
        lines.append(f"   Debate: {topic.discussion_focus}")
    if topic.market_relevance:
        softened = soften_market_implication(topic)
        if softened:
            lines.append(f"   Market read: {softened}")

    # Insurance framework (two-layer). Prefer LLM-provided; fall back to
    # derived observation framework so we never show a bare instruction.
    framework = getattr(topic, "insurance_framework", None)
    if framework is None or (not framework.implications and not framework.triggers):
        framework = derive_insurance_framework(topic)
    # Align the framework with the softened market implication so we
    # don't get "cautious read → aggressive allocation call".
    framework = align_insurance_framework(framework, topic)
    lines.extend(render_insurance_framework(framework))

    # Reasons line adds length without adding much signal when compact
    if not compact:
        reasons = (getattr(topic, "reasons", "") or "").strip()
        if reasons:
            lines.append(f"   Rationale: {reasons}")

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
    import logging as _logging
    _log = _logging.getLogger(__name__)

    if not sentiments:
        return ""

    total_posts = sum(s.post_count for s in sentiments)
    platforms = [s.platform.capitalize() for s in sentiments if s.post_count > 0]
    platform_label = " + ".join(platforms) if platforms else "Community"

    # Pick topics to render: prefer analyst-selected headlines, but fall back
    # to unlinked or all when report is absent.
    if analyst_report is not None and analyst_report.headline_topics:
        primary_topics = list(analyst_report.headline_topics)
    elif unlinked_topics is not None:
        primary_topics = list(unlinked_topics)
    else:
        primary_topics = []
        for s in sentiments:
            primary_topics.extend(s.trending_topics)

    analyst_selected_count = (
        len(analyst_report.headline_topics)
        if analyst_report is not None else 0
    )
    count_before_should_include = len(primary_topics)

    # Analyst picks override per-cluster `should_include_in_brief`. The
    # analyst ran a second LLM pass with full cross-cluster context, so
    # if it explicitly picked a topic we trust that judgement even if the
    # per-cluster LLM had marked should_include=False. Without this
    # override, analyst=3 collapses to rendered=1 whenever two LLM passes
    # disagree, which is exactly the "前后数量不一致" bug.
    #
    # We still drop clusters the pipeline considers outright noise, since
    # that's a stronger signal than a soft include/exclude vote.
    analyst_picked_ids = set()
    if analyst_report is not None:
        analyst_picked_ids = {
            getattr(t, "cluster_id", id(t))
            for t in analyst_report.headline_topics
        }

    def _keep(t) -> bool:
        if t.credibility.is_noise:
            return False
        if getattr(t, "cluster_id", id(t)) in analyst_picked_ids:
            return True
        return getattr(t, "should_include_in_brief", True)

    dropped_by_noise = [t for t in primary_topics if t.credibility.is_noise]
    dropped_by_should_include = [
        t for t in primary_topics
        if not getattr(t, "should_include_in_brief", True)
        and getattr(t, "cluster_id", id(t)) not in analyst_picked_ids
        and t not in dropped_by_noise
    ]
    primary_topics = [t for t in primary_topics if _keep(t)]

    rendered_count = len(primary_topics)
    if analyst_report is not None:
        # Stash the headline counts on the report so daily_job (which owns
        # the on-disk logger) can surface the pre-filter vs post-filter
        # totals in its pipeline summary.
        analyst_report.render_stats = {
            "analyst_selected": analyst_selected_count,
            "formatter_rendered": rendered_count,
            "dropped_by_noise": len(dropped_by_noise),
            "dropped_by_should_include": len(dropped_by_should_include),
        }
    if analyst_selected_count and rendered_count != analyst_selected_count:
        reasons = []
        if dropped_by_noise:
            reasons.append(f"noise={len(dropped_by_noise)}")
        if dropped_by_should_include:
            reasons.append(f"should_include=false×{len(dropped_by_should_include)}")
        reason_str = ", ".join(reasons) or "unknown"
        _log.info(
            "community: analyst selected %d headline(s), formatter rendered %d (dropped: %s)",
            analyst_selected_count,
            rendered_count,
            reason_str,
        )
    elif analyst_selected_count:
        _log.info(
            "community: analyst selected %d headline(s), formatter rendered %d (no drop)",
            analyst_selected_count,
            rendered_count,
        )

    if not primary_topics and not analyst_report:
        return ""

    # Signal summary: N posts · M clusters · K selected · L noise
    if analyst_report is not None:
        summary_parts = [f"{total_posts} posts"]
        if analyst_report.total_clusters:
            summary_parts.append(f"{analyst_report.total_clusters} clusters")
        summary_parts.append(f"{len(analyst_report.headline_topics)} selected")
        if analyst_report.noise_topics:
            summary_parts.append(f"{len(analyst_report.noise_topics)} noise")
        signal_summary = " · ".join(summary_parts)
    else:
        signal_summary = f"{total_posts} posts"

    lines = [
        "💬 Community Sentiment",
        f"{platform_label} · {signal_summary}",
    ]

    if analyst_report is not None:
        # Platform coverage disclosure — surfaces 402 / not-configured states
        coverage = [
            s for s in analyst_report.platform_status
            if s and not s.startswith(tuple(p.lower() + "=ok" for p in platforms))
        ]
        if coverage:
            lines.append(f"Data coverage: {' | '.join(coverage)}")

        from community.verbalize import compose_sentiment_structure
        # Always route sentiment structure through the composer so stacked
        # model-label phrasing never reaches the user. Raw llm text is kept
        # on the report for internal use.
        all_clusters_for_struct = list(analyst_report.headline_topics)
        all_clusters_for_struct.extend(analyst_report.noise_topics)
        sentiment_line = compose_sentiment_structure(
            all_clusters_for_struct,
            llm_text=analyst_report.sentiment_structure,
        )
        if sentiment_line:
            lines.append(f"Overall sentiment: {sentiment_line}")
        if analyst_report.cross_platform_signal:
            lines.append(f"Cross-platform signal: {analyst_report.cross_platform_signal}")

    # News ↔ Social bridge — bridges the news section and community topics.
    # Rendered after sentiment structure so reader sees the comparison
    # before diving into individual topics.
    bridge_text = news_social_bridge
    if not bridge_text and analyst_report is not None:
        bridge_text = analyst_report.news_social_bridge
    if bridge_text:
        lines.append(f"News ↔ Social: {bridge_text}")

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
            lines.append("🛡 Insurance-book view (watch framework)")
            if framework.implications:
                lines.append(f"   Allocation read: {framework.implications}")
            if framework.triggers:
                lines.append(f"   Watch / triggers: {framework.triggers}")
        elif analyst_report.insurance_angle:
            # Legacy fallback — still render as observation, not instruction
            lines.append("")
            lines.append(f"🛡 Insurance-book view: {analyst_report.insurance_angle}")
        if analyst_report.brief_recommendation:
            lines.append(f"📝 Editor's note: {analyst_report.brief_recommendation}")

    return "\n".join(lines).rstrip()


def format_watchlist(watchlist: List[str]) -> str:
    if not watchlist:
        watchlist = [
            "Watch for any key US macro prints later tonight",
            "Watch Fed officials' commentary",
            "Watch whether long-end yields keep moving",
        ]

    lines = ["👀 Watchlist"]
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
        f"📌 Daily Financial Brief | {briefing_input.date_str}",
        "",
        "Bottom line:",
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
            us_equities="S&P 500 modestly higher, Nasdaq outperforms",
            rates="10Y Treasury yield at 4.29%, down 5 bps from the prior session",
            asia_sg="STI trades steady",
        ),
        news_items=[
            NewsItem(
                title="Fed rate cut bets revived, a bit, by Iran war ceasefire - Reuters",
                summary="Market pricing for rate cuts this year has firmed.",
                source="Reuters",
                category="macro",
            ),
            NewsItem(
                title="US bank profits to rise on deals, but Iran war fuels outlook uncertainty - Reuters",
                summary="Bank earnings find partial support, though geopolitical risk keeps the market cautious.",
                source="Reuters",
                category="equity",
            ),
            NewsItem(
                title="Singapore financial sector sees stable outlook amid regional uncertainty",
                summary="Against a backdrop of regional uncertainty, Singapore's financial sector holds a relatively steady outlook.",
                source="Straits Times",
                category="singapore",
            ),
        ],
        watchlist=[
            "Whether new US inflation or payrolls data prints tonight",
            "Whether Treasury yields keep moving",
            "How Asian markets track the US rate-path reset",
        ],
    )

    brief_text = format_daily_brief(sample_input)
    print(brief_text)