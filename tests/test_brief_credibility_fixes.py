"""Tests for the five targeted fixes landed for daily-brief credibility:

1. Headline/body directional consistency filter in news_fetcher.
2. Portfolio-relevance classifier + gated "对组合影响" rendering.
3. compose_sentiment_structure collapses stacked model labels into a
   manager-readable single sentence.
4. soften_market_implication / align_insurance_framework hedge strong
   directional claims when credibility is low.
5. format_community_section populates analyst_report.render_stats so
   the caller can detect analyst-vs-formatter headline count drift.
"""
import uuid

import pytest

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
from news_fetcher import (
    RawNewsItem,
    _has_headline_body_conflict,
)


# ─── Fix 1: headline/body consistency filter ────────────────────────────────

class TestHeadlineBodyConsistency:
    def test_surge_title_with_plunge_body_flagged(self):
        item = RawNewsItem(
            title="Apple shares surge on earnings beat",
            summary="Apple stock plunged after the earnings report missed.",
            source="Test",
            category="equity",
        )
        assert _has_headline_body_conflict(item) is True

    def test_drop_title_with_rally_body_flagged(self):
        item = RawNewsItem(
            title="Tesla stock drops on weak guidance",
            summary="Tesla shares rallied and climbed to record highs.",
            source="Test",
            category="equity",
        )
        assert _has_headline_body_conflict(item) is True

    def test_consistent_directions_pass(self):
        item = RawNewsItem(
            title="S&P 500 rises on dovish Fed minutes",
            summary="Stocks gained as investors cheered the dovish shift.",
            source="Test",
            category="macro",
        )
        assert _has_headline_body_conflict(item) is False

    def test_missing_summary_does_not_trigger(self):
        item = RawNewsItem(
            title="Market surges on Fed cut",
            summary="暂无摘要",
            source="Test",
            category="macro",
        )
        assert _has_headline_body_conflict(item) is False

    def test_mixed_direction_body_does_not_trigger(self):
        # A body that mentions both up and down words is ambiguous, not
        # a conflict — we only flag clean up-vs-down.
        item = RawNewsItem(
            title="Treasury yields surge on hot CPI",
            summary="Yields rose then fell later in the session.",
            source="Test",
            category="macro",
        )
        assert _has_headline_body_conflict(item) is False


# ─── Fix 2: portfolio relevance classifier ──────────────────────────────────

class TestPortfolioRelevance:
    def test_fed_cpi_is_macro(self):
        item = NewsItem(
            title="Fed signals pause after CPI softens",
            summary="Treasury yields dropped.",
            source="Reuters",
        )
        assert classify_portfolio_relevance(item) == "macro"

    def test_bank_story_is_sector(self):
        item = NewsItem(
            title="Regional bank launches new product",
            summary="The lender expands its retail footprint.",
            source="Reuters",
        )
        assert classify_portfolio_relevance(item) == "sector"

    def test_single_stock_is_low(self):
        item = NewsItem(
            title="CompanyX announces AI workflow launch",
            summary="The company targets federal agencies.",
            source="PR",
        )
        assert classify_portfolio_relevance(item) == "low"

    def test_low_relevance_renders_honest_phrase(self):
        item = NewsItem(
            title="CompanyX launches AI workflow",
            summary="Feature now in beta.",
            source="PR",
            why_it_matters="这条信息有助于判断当天组合风险偏好和市场主线变化。",
        )
        item.portfolio_relevance = "low"
        rendered = format_news_section([item])
        assert "对保险配置影响有限" in rendered
        assert "对组合影响" not in rendered

    def test_single_stock_vetoes_macro_category(self):
        """Alpha Vantage sometimes tags single-stock noise as category='macro'
        via its broad topic taxonomy. Single-stock noise patterns in the
        title should veto that promotion.
        """
        axos = NewsItem(
            title="Axos Financial Sees AI Downgrades as Technical Pressure",
            summary="",
            source="TipRanks",
            category="macro",
        )
        micron = NewsItem(
            title="Why Is Micron Tech Stock Surging On Tuesday?",
            summary="",
            source="Zacks",
            category="macro",
        )
        assert classify_portfolio_relevance(axos) == "low"
        assert classify_portfolio_relevance(micron) == "low"

    def test_sanctions_is_macro(self):
        item = NewsItem(
            title="US imposes new sanctions against suppliers of weapons to Iran",
            summary="",
            source="Reuters",
            category="general",
        )
        assert classify_portfolio_relevance(item) == "macro"

    def test_macro_relevance_keeps_impact_line(self):
        item = NewsItem(
            title="Fed rate cut bets rise",
            summary="Futures price in more easing.",
            source="Reuters",
            why_it_matters="影响久期和再投资收益率。",
        )
        item.portfolio_relevance = "macro"
        rendered = format_news_section([item])
        assert "对组合影响" in rendered
        assert "影响久期和再投资收益率" in rendered


# ─── Fix 3: compose_sentiment_structure ─────────────────────────────────────

def _cluster(label="neutral", fear=0.0, uncertainty=0.0, optimism=0.0,
             hype=0.0, skepticism=0.0, is_noise=False, credibility=0.5):
    c = TopicCluster(cluster_id=uuid.uuid4().hex[:12])
    c.sentiment = SentimentProfile(
        label=label,
        fear=fear, uncertainty=uncertainty, optimism=optimism,
        hype=hype, skepticism=skepticism,
    )
    c.credibility = CredibilityProfile(overall=credibility, is_noise=is_noise)
    return c


class TestComposeSentimentStructure:
    def test_stacked_llm_text_gets_replaced(self):
        stacked = "分歧·担忧(0.70) + 不确定性高 + 方向分歧 + 担忧为主 + 避险升温"
        clusters = [
            _cluster(label="bearish", fear=0.7),
            _cluster(label="mixed", uncertainty=0.6),
        ]
        out = compose_sentiment_structure(clusters, llm_text=stacked)
        assert "0.70" not in out
        assert "·" not in out
        # Should read like a conclusion
        assert out.endswith("。")

    def test_concise_llm_text_is_preserved(self):
        clean = "整体情绪偏谨慎，避险情绪升温。"
        clusters = [_cluster(label="bearish", fear=0.6)]
        assert compose_sentiment_structure(clusters, llm_text=clean) == clean

    def test_empty_non_noise_clusters_returns_fallback(self):
        noise = [_cluster(is_noise=True)]
        out = compose_sentiment_structure(noise)
        assert out  # non-empty
        assert "未形成" in out or "信号偏弱" in out

    def test_bearish_cluster_produces_risk_off_conclusion(self):
        clusters = [_cluster(label="bearish", fear=0.7)]
        out = compose_sentiment_structure(clusters, llm_text="")
        assert "避险" in out or "谨慎" in out


# ─── Fix 4: soften_market_implication / align_insurance_framework ───────────

class TestSoftenMarketImplication:
    def test_low_credibility_hedges_spread_tightening(self):
        c = _cluster(credibility=0.45)
        c.market_relevance = "利空风险偏好，信用利差收窄"
        out = soften_market_implication(c)
        assert "信用利差收窄" not in out
        assert "信用利差是否重新走阔" in out

    def test_high_credibility_keeps_direct_wording(self):
        c = _cluster(credibility=0.75)
        c.market_relevance = "利空风险偏好，信用利差收窄"
        out = soften_market_implication(c)
        assert out == "利空风险偏好，信用利差收窄"

    def test_weak_macro_topic_gets_caveat_tail(self):
        c = _cluster(credibility=0.4)
        c.rule_label = "衰退与宏观经济"
        c.market_relevance = "利好久期"
        out = soften_market_implication(c)
        assert "需结合" in out


class TestAlignInsuranceFramework:
    def test_removes_specific_duration_instruction_on_weak_evidence(self):
        c = _cluster(credibility=0.45)
        c.rule_label = "美债与收益率"
        fw = InsuranceFramework(
            implications="建议延长0.3年久期，配置更多长端",
            triggers="继续观察长端利率",
        )
        out = align_insurance_framework(fw, c)
        assert "0.3年" not in out.implications
        assert "仍需更多确认" in out.implications or "调整幅度" in out.implications

    def test_keeps_framework_when_credibility_high(self):
        c = _cluster(credibility=0.8)
        c.rule_label = "美债与收益率"
        fw = InsuranceFramework(implications="延长久期0.3年", triggers="数据确认")
        out = align_insurance_framework(fw, c)
        assert out.implications == "延长久期0.3年"


# ─── Fix 5: analyst-vs-formatter render count diagnostics ───────────────────

def _make_sentiment(platform="reddit", topics=None, count=20):
    return CommunitySentiment(
        platform=platform,
        trending_topics=topics or [],
        overall_sentiment="neutral",
        post_count=count,
    )


def _renderable_topic(label, include=True, is_noise=False):
    c = TopicCluster(
        cluster_id=uuid.uuid4().hex[:12],
        rule_label=label,
        headline=label,
        platforms=["reddit"],
        heat_score=50.0,
        posts=[UnifiedPost(platform="reddit", post_id="p1", channel="r/test", title="t")],
        sentiment=SentimentProfile(label="neutral"),
        credibility=CredibilityProfile(overall=0.6, is_noise=is_noise),
    )
    c.should_include_in_brief = include
    return c


class TestRenderStats:
    def test_all_headlines_render_and_stats_reflect_no_drop(self):
        topics = [_renderable_topic("T1"), _renderable_topic("T2"), _renderable_topic("T3")]
        report = CommunityAnalystReport(headline_topics=topics, total_clusters=3)
        sentiments = [_make_sentiment(topics=topics, count=30)]
        out = format_community_section(sentiments, analyst_report=report)
        assert "T1" in out and "T2" in out and "T3" in out
        assert report.render_stats["analyst_selected"] == 3
        assert report.render_stats["formatter_rendered"] == 3
        assert report.render_stats["dropped_by_noise"] == 0

    def test_analyst_pick_overrides_should_include_false(self):
        """When the analyst explicitly chose a topic, it renders even if
        the per-cluster LLM marked should_include_in_brief=False. The
        analyst's cross-cluster view wins.
        """
        good = _renderable_topic("Keep")
        analyst_override = _renderable_topic("AnalystPick", include=False)
        topics = [good, analyst_override]
        report = CommunityAnalystReport(headline_topics=topics, total_clusters=2)
        sentiments = [_make_sentiment(topics=topics, count=10)]
        out = format_community_section(sentiments, analyst_report=report)
        assert "Keep" in out
        assert "AnalystPick" in out
        assert report.render_stats["formatter_rendered"] == 2
        assert report.render_stats["dropped_by_should_include"] == 0

    def test_should_include_false_still_drops_when_not_analyst_pick(self):
        """A topic NOT selected by the analyst but with should_include=False
        still gets dropped (unchanged behavior for the non-override path).
        """
        picked = _renderable_topic("AnalystPicked")
        unlinked_extra = _renderable_topic("ExtraNotPicked", include=False)
        report = CommunityAnalystReport(headline_topics=[picked], total_clusters=2)
        sentiments = [_make_sentiment(topics=[picked, unlinked_extra], count=10)]
        # With headline_topics non-empty, primary_topics starts from the
        # analyst's picks; ExtraNotPicked is not there, so we don't test
        # the drop path that way. Simulate it via unlinked_topics fallback
        # when report.headline_topics is empty.
        empty_report = CommunityAnalystReport(headline_topics=[], total_clusters=2)
        out = format_community_section(
            sentiments,
            unlinked_topics=[picked, unlinked_extra],
            analyst_report=empty_report,
        )
        assert "AnalystPicked" in out
        assert "ExtraNotPicked" not in out

    def test_linked_analyst_topic_still_renders(self):
        """The old bug: analyst picks a topic, news pipeline also links it
        inline, and then the formatter's unlinked-only filter drops it from
        the community section — so analyst=3 but rendered=2.

        New behavior: analyst picks are kept regardless of inline linkage.
        """
        t1 = _renderable_topic("Fed")
        t2 = _renderable_topic("Yields")
        t3 = _renderable_topic("Recession")
        # Simulate that t1 was linked inline (not in unlinked_topics)
        report = CommunityAnalystReport(headline_topics=[t1, t2, t3], total_clusters=3)
        sentiments = [_make_sentiment(topics=[t1, t2, t3], count=40)]
        out = format_community_section(
            sentiments,
            unlinked_topics=[t2, t3],
            analyst_report=report,
        )
        assert "Fed" in out
        assert "Yields" in out
        assert "Recession" in out
        assert report.render_stats["formatter_rendered"] == 3
