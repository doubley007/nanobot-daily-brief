"""Tests for community-related formatting in financial_brief_formatter."""
import uuid

from community.base import CommunitySentiment, TopicCluster
from community.linker import LinkedReaction
from community.schema import SentimentProfile, UnifiedPost
from financial_brief_formatter import (
    format_news_section,
    format_community_section,
    NewsItem,
)


def _make_topic(label, sentiment="neutral", post_count=5, focus="焦点", relevance="影响"):
    posts = [
        UnifiedPost(
            platform="reddit",
            post_id=f"p{i}",
            channel="r/test",
            title=f"test post {i}",
        )
        for i in range(post_count)
    ]
    return TopicCluster(
        cluster_id=uuid.uuid4().hex[:12],
        posts=posts,
        platforms=["reddit"],
        rule_label=label,
        heat_score=100.0,
        discussion_focus=focus,
        market_relevance=relevance,
        sentiment=SentimentProfile(label=sentiment),
    )


def _make_sentiment(platform="reddit", topics=None, overall="neutral", count=10):
    return CommunitySentiment(
        platform=platform,
        trending_topics=topics or [],
        overall_sentiment=overall,
        post_count=count,
    )


# ─── format_news_section with linked reactions ──────────────────────────────

class TestFormatNewsSection:
    def test_no_linked_reactions(self):
        news = [NewsItem(title="Test", summary="Summary", source="T", why_it_matters="Impact")]
        result = format_news_section(news)
        assert "社区反应" not in result

    def test_with_linked_reaction(self):
        news = [NewsItem(title="Fed rate cut", summary="Summary", source="T", why_it_matters="Impact")]
        linked = {0: LinkedReaction(
            topic_label="美联储与利率政策",
            sentiment="bearish",
            discussion_focus="利率决策",
            post_count=8,
            confidence=0.9,
        )}
        result = format_news_section(news, linked)
        assert "社区反应" in result
        assert "8 条讨论" in result
        assert "偏空" in result
        # Formatter prefers the cluster's topic_label over discussion_focus
        # — see financial_brief_formatter._format_topic_lines.
        assert "美联储与利率政策" in result

    def test_empty_news(self):
        result = format_news_section([])
        assert "今日暂无" in result

    def test_linked_reaction_only_for_matched_items(self):
        news = [
            NewsItem(title="Fed rate cut", summary="s1", source="T"),
            NewsItem(title="Oil prices", summary="s2", source="T"),
        ]
        linked = {0: LinkedReaction(
            topic_label="美联储与利率政策", sentiment="bullish",
            discussion_focus="", post_count=5, confidence=0.8,
        )}
        result = format_news_section(news, linked)
        lines = result.split("\n")
        # Only the first news item should have a community reaction
        reaction_lines = [l for l in lines if "社区反应" in l]
        assert len(reaction_lines) == 1


# ─── format_community_section ────────────────────────────────────────────────

class TestFormatCommunitySection:
    def test_empty_sentiments(self):
        assert format_community_section([]) == ""

    def test_single_platform(self):
        s = _make_sentiment("reddit", [_make_topic("美联储与利率政策")], "bearish", 45)
        result = format_community_section([s])
        assert "Reddit" in result
        assert "45 贴" in result
        assert "美联储与利率政策" in result

    def test_multi_platform(self):
        reddit = _make_sentiment("reddit", [_make_topic("美联储与利率政策")], "bearish", 45)
        x = _make_sentiment("x", [_make_topic("关税与贸易")], "mixed", 30)
        result = format_community_section([reddit, x])
        assert "Reddit + X" in result
        assert "75 贴" in result

    def test_unlinked_topics_shown(self):
        s = _make_sentiment("reddit", [_make_topic("加密货币")], "neutral", 20)
        topic = _make_topic("加密货币", "bullish", 10, "加密走势", "风险偏好")
        result = format_community_section([s], unlinked_topics=[topic])
        assert "加密货币" in result
        assert "加密走势" in result
        assert "风险偏好" in result

    def test_topic_details_rendered(self):
        topic = _make_topic("原油与能源", "bearish", 8, "油价下跌", "通胀预期")
        s = _make_sentiment("reddit", [topic], "bearish", 30)
        result = format_community_section([s])
        assert "原油与能源" in result
        assert "争论点" in result
        assert "油价下跌" in result
        assert "市场含义" in result
        assert "通胀预期" in result


# ─── format_community_section edge cases ─────────────────────────────────────

class TestFormatCommunityEdgeCases:
    def test_platform_label_excludes_zero_post_sources(self):
        """When X has 0 posts, only Reddit should appear in the label."""
        reddit = _make_sentiment("reddit", [_make_topic("美联储与利率政策")], "bearish", 45)
        x_empty = _make_sentiment("x", [], "neutral", 0)
        result = format_community_section([reddit, x_empty])
        assert "Reddit" in result
        # The header line (before topics) should not list X — but topics
        # may still legitimately mention e.g. "Reddit主导" so we only
        # check the first line.
        header = result.split("\n", 2)[1]
        assert "X" not in header

    def test_three_platforms(self):
        reddit = _make_sentiment("reddit", [_make_topic("美联储与利率政策")], "bearish", 30)
        x = _make_sentiment("x", [_make_topic("关税与贸易")], "mixed", 20)
        discord = _make_sentiment("discord", [_make_topic("加密货币")], "bullish", 15)
        result = format_community_section([reddit, x, discord])
        assert "Reddit + X + Discord" in result
        assert "65 贴" in result
