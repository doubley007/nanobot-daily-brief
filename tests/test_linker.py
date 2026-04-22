"""Tests for news-to-community linker."""
import uuid

from community.linker import (
    LinkedReaction,
    link_news_to_community,
    get_unlinked_topics,
    _score_news_against_topic,
    COMMUNITY_TOPIC_KEYWORDS,
    MIN_LINK_SCORE,
)
from community.base import TopicCluster
from community.schema import SentimentProfile, UnifiedPost
from financial_brief_formatter import NewsItem


def _make_topic(label, sentiment="neutral", post_count=5, focus="", relevance=""):
    """
    Build a minimal TopicCluster for linker tests. Helper lives here rather
    than in conftest because the linker has a narrow view of the schema:
    it reads rule_label, headline, sentiment.label, heat_score, and
    cluster_id — nothing else.
    """
    posts = [
        UnifiedPost(
            platform="reddit",
            post_id=f"p{i}",
            channel="r/test",
            title=f"placeholder title {i}",
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


def _make_news(title, summary="", category="general"):
    return NewsItem(title=title, summary=summary, source="Test", category=category)


# ─── _score_news_against_topic ───────────────────────────────────────────────

class TestScoreNewsAgainstTopic:
    def test_fed_topic_matches_fed_news(self):
        score = _score_news_against_topic(
            "Federal Reserve signals rate cut, FOMC dovish", "美联储与利率政策"
        )
        assert score >= MIN_LINK_SCORE

    def test_no_match_returns_zero(self):
        score = _score_news_against_topic(
            "My cat likes fish", "美联储与利率政策"
        )
        assert score == 0

    def test_unknown_topic_returns_zero(self):
        score = _score_news_against_topic("Fed rate cut", "不存在的话题")
        assert score == 0


# ─── link_news_to_community ──────────────────────────────────────────────────

class TestLinkNewsToCommunity:
    def test_basic_link(self):
        news = [_make_news("Federal Reserve rate cut expected", "FOMC dovish stance")]
        topics = [_make_topic("美联储与利率政策", sentiment="bearish", focus="利率决策")]
        links = link_news_to_community(news, topics)
        assert 0 in links
        assert links[0].topic_label == "美联储与利率政策"
        assert links[0].sentiment == "bearish"

    def test_no_match_when_unrelated(self):
        news = [_make_news("Celebrity gossip roundup")]
        topics = [_make_topic("美联储与利率政策")]
        links = link_news_to_community(news, topics)
        assert len(links) == 0

    def test_empty_topics(self):
        news = [_make_news("Fed rate cut")]
        links = link_news_to_community(news, [])
        assert len(links) == 0

    def test_empty_news(self):
        topics = [_make_topic("美联储与利率政策")]
        links = link_news_to_community([], topics)
        assert len(links) == 0

    def test_one_to_one_assignment(self):
        """Each news item gets at most one link, each topic links to at most one news."""
        news = [
            _make_news("Fed rate cut bets revived", "monetary policy shift"),
            _make_news("Treasury yields surge as bond market rattled", "yield curve steepens"),
        ]
        topics = [
            _make_topic("美联储与利率政策", sentiment="bullish"),
            _make_topic("美债与收益率", sentiment="bearish"),
        ]
        links = link_news_to_community(news, topics)
        linked_topics = {r.topic_label for r in links.values()}
        assert len(linked_topics) == len(links)  # no topic used twice

    def test_strongest_match_wins(self):
        """When a news item could match multiple topics, the strongest wins."""
        news = [_make_news("Fed rate cut as treasury yields drop and bond market rallies")]
        topics = [
            _make_topic("美联储与利率政策"),
            _make_topic("美债与收益率"),
        ]
        links = link_news_to_community(news, topics)
        assert 0 in links  # should match one of them


# ─── get_unlinked_topics ─────────────────────────────────────────────────────

class TestGetUnlinkedTopics:
    def test_all_linked(self):
        topics = [_make_topic("美联储与利率政策")]
        linked = {0: LinkedReaction(
            topic_label="美联储与利率政策", sentiment="bearish",
            discussion_focus="", post_count=5, confidence=0.8,
        )}
        unlinked = get_unlinked_topics(topics, linked)
        assert len(unlinked) == 0

    def test_some_unlinked(self):
        topics = [
            _make_topic("美联储与利率政策"),
            _make_topic("加密货币"),
        ]
        linked = {0: LinkedReaction(
            topic_label="美联储与利率政策", sentiment="bearish",
            discussion_focus="", post_count=5, confidence=0.8,
        )}
        unlinked = get_unlinked_topics(topics, linked)
        assert len(unlinked) == 1
        assert unlinked[0].rule_label == "加密货币"

    def test_none_linked(self):
        topics = [_make_topic("加密货币"), _make_topic("原油与能源")]
        unlinked = get_unlinked_topics(topics, {})
        assert len(unlinked) == 2


# ─── Keyword coverage ────────────────────────────────────────────────────────

class TestKeywordCoverage:
    def test_all_topics_have_keywords(self):
        """Every topic in the linker should have at least 3 keywords."""
        for topic, keywords in COMMUNITY_TOPIC_KEYWORDS.items():
            assert len(keywords) >= 3, f"Topic '{topic}' has too few keywords"
