"""
Demo that exercises the 5 credibility fixes end-to-end.

Prints a short BEFORE/AFTER for each fix so reviewers can eyeball
the shift without needing to run the real pipeline.
"""
from __future__ import annotations

import uuid

from community.schema import (
    CommunityAnalystReport,
    CommunitySentiment,
    CredibilityProfile,
    InsuranceFramework,
    SentimentProfile,
    TopicCluster,
    TrendProfile,
    UnifiedPost,
)
from community.verbalize import (
    align_insurance_framework,
    compose_sentiment_structure,
    soften_market_implication,
)
from financial_brief_formatter import (
    NewsItem,
    classify_portfolio_relevance,
    format_community_section,
    format_news_section,
)
from news_fetcher import RawNewsItem, _has_headline_body_conflict


def _section(title: str):
    bar = "─" * 70
    print(f"\n{bar}\n{title}\n{bar}")


# ─── Fix 1 ───────────────────────────────────────────────────────────────────

def demo_headline_body_consistency():
    _section("FIX 1 — Headline/body directional consistency filter")
    cases = [
        RawNewsItem(
            title="Apple shares surge on earnings beat",
            summary="Apple stock plunged after the report disappointed investors.",
            source="Test", category="equity",
        ),
        RawNewsItem(
            title="Tesla drops on weak guidance",
            summary="Tesla shares rallied to record highs.",
            source="Test", category="equity",
        ),
        RawNewsItem(
            title="S&P 500 rises on dovish Fed minutes",
            summary="Stocks gained as investors cheered the shift.",
            source="Test", category="macro",
        ),
    ]
    for c in cases:
        flag = "DROP" if _has_headline_body_conflict(c) else "keep"
        print(f"  [{flag}] {c.title}")
        print(f"         body: {c.summary}")


# ─── Fix 2 ───────────────────────────────────────────────────────────────────

def demo_portfolio_relevance():
    _section("FIX 2 — Portfolio relevance classifier gating 对组合影响")

    items = [
        NewsItem(
            title="Fed rate cut bets rise after dovish FOMC minutes",
            summary="Treasury yields dropped.",
            source="Reuters", category="macro",
            why_it_matters="直接影响固定收益估值与再投资收益率。",
        ),
        NewsItem(
            title="Apple AI workflow launch for federal agencies",
            summary="Beta only; no pricing.",
            source="PR", category="general",
            why_it_matters="（模板占位：这条信息有助于判断当天组合风险偏好和市场主线变化）",
        ),
    ]
    for it in items:
        it.portfolio_relevance = classify_portfolio_relevance(it)
        print(f"  relevance={it.portfolio_relevance!r}")
        print(format_news_section([it]))
        print()


# ─── Fix 3 ───────────────────────────────────────────────────────────────────

def demo_sentiment_structure():
    _section("FIX 3 — Overall sentiment: stacked labels → clean conclusion")

    def _cluster(label, fear=0.0, uncertainty=0.0):
        c = TopicCluster(cluster_id=uuid.uuid4().hex[:12])
        c.sentiment = SentimentProfile(label=label, fear=fear, uncertainty=uncertainty)
        c.credibility = CredibilityProfile(overall=0.5, is_noise=False)
        return c

    clusters = [
        _cluster("bearish", fear=0.7),
        _cluster("mixed", uncertainty=0.6),
    ]

    stacked_llm = "混合情绪主导 + 不确定性高 + 分歧·担忧(0.70) + 避险升温 + 观望为主"
    print(f"  BEFORE (raw LLM): {stacked_llm}")
    out = compose_sentiment_structure(clusters, llm_text=stacked_llm)
    print(f"  AFTER  (composed): {out}")


# ─── Fix 4 ───────────────────────────────────────────────────────────────────

def demo_confidence_aware_implications():
    _section("FIX 4 — Confidence-aware market implication / insurance framework")

    c_weak = TopicCluster(
        cluster_id="w",
        rule_label="衰退与宏观经济",
        market_relevance="利空风险偏好，信用利差收窄；建议延长久期",
    )
    c_weak.credibility = CredibilityProfile(overall=0.45, is_noise=False)
    print(f"  [weak cred=0.45] BEFORE: {c_weak.market_relevance}")
    print(f"  [weak cred=0.45] AFTER : {soften_market_implication(c_weak)}")

    fw = InsuranceFramework(
        implications="延长久期0.3年，减持信用利差敏感资产5%",
        triggers="继续观察长端利率",
    )
    aligned = align_insurance_framework(fw, c_weak)
    print(f"  [weak cred=0.45] FRAMEWORK BEFORE: {fw.implications}")
    print(f"  [weak cred=0.45] FRAMEWORK AFTER : {aligned.implications}")

    c_strong = TopicCluster(
        cluster_id="s",
        rule_label="美债与收益率",
        market_relevance="利空风险偏好，信用利差收窄",
    )
    c_strong.credibility = CredibilityProfile(overall=0.78, is_noise=False)
    print(f"  [strong cred=0.78] keeps wording: {soften_market_implication(c_strong)}")


# ─── Fix 5 ───────────────────────────────────────────────────────────────────

def demo_render_count_diagnostics():
    _section("FIX 5 — Analyst-selected vs formatter-rendered accounting")

    def _topic(label, include=True, is_noise=False):
        return TopicCluster(
            cluster_id=uuid.uuid4().hex[:12],
            rule_label=label,
            headline=label,
            platforms=["reddit"],
            heat_score=50.0,
            posts=[UnifiedPost(platform="reddit", post_id="p", channel="r/t", title="t")],
            sentiment=SentimentProfile(label="neutral"),
            credibility=CredibilityProfile(overall=0.6, is_noise=is_noise),
            should_include_in_brief=include,
        )

    t1 = _topic("Fed")
    t2 = _topic("Yields")
    t3 = _topic("Recession")

    report = CommunityAnalystReport(headline_topics=[t1, t2, t3], total_clusters=3)
    sentiments = [CommunitySentiment(
        platform="reddit", trending_topics=[t1, t2, t3], post_count=60,
    )]

    # Simulate the old situation: t1 was linked inline to news, so
    # unlinked_topics only had t2 and t3. Previously the formatter would
    # drop t1 from the community section → analyst=3 but rendered=2.
    format_community_section(
        sentiments,
        unlinked_topics=[t2, t3],
        analyst_report=report,
    )
    print(f"  render_stats after fix: {report.render_stats}")
    # expected: analyst_selected=3, formatter_rendered=3, drops=0


def main():
    demo_headline_body_consistency()
    demo_portfolio_relevance()
    demo_sentiment_structure()
    demo_confidence_aware_implications()
    demo_render_count_diagnostics()


if __name__ == "__main__":
    main()
