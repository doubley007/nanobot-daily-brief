"""
Tests for community.verbalize.

The verbalization layer is where raw model output becomes business-
readable Chinese. Tests cover the mapping rules, threshold behavior
(weak signals should not trigger confident phrases), and the
news↔social bridge.
"""
import uuid

from community.schema import (
    InsuranceFramework,
    SentimentProfile,
    TopicCluster,
    TrendProfile,
    UnifiedPost,
)
from community.verbalize import (
    build_news_social_bridge,
    derive_insurance_framework,
    render_insurance_framework,
    sentiment_score_tail,
    verbalize_sentiment,
    verbalize_trend,
)


def _sent(label="neutral", **dims):
    """Build a SentimentProfile, auto-deriving dominant dimension."""
    p = SentimentProfile(
        label=label,
        optimism=dims.get("optimism", 0.0),
        fear=dims.get("fear", 0.0),
        uncertainty=dims.get("uncertainty", 0.0),
        skepticism=dims.get("skepticism", 0.0),
        hype=dims.get("hype", 0.0),
    )
    values = {k: getattr(p, k) for k in ("optimism", "fear", "uncertainty", "skepticism", "hype")}
    top = max(values.items(), key=lambda kv: kv[1])
    if top[1] > 0:
        p.dominant_dimension = top[0]
        p.intensity = top[1]
    return p


# ─── verbalize_sentiment ─────────────────────────────────────────────────────

class TestVerbalizeSentiment:
    def test_fear_led_bearish_is_risk_off(self):
        s = _sent(label="bearish", fear=0.7)
        assert verbalize_sentiment(s) == "避险情绪升温"

    def test_fear_led_mixed_reports_divergence(self):
        s = _sent(label="mixed", fear=0.6)
        assert "分歧" in verbalize_sentiment(s)

    def test_uncertainty_led_mixed_is_wait_and_see(self):
        s = _sent(label="mixed", uncertainty=0.7)
        assert verbalize_sentiment(s) == "观望为主，不确定性较高"

    def test_uncertainty_led_bullish_notes_low_consensus(self):
        s = _sent(label="bullish", uncertainty=0.7)
        assert "共识不足" in verbalize_sentiment(s)

    def test_hype_with_no_direction_flags_divergence(self):
        s = _sent(label="bullish", hype=0.7)
        assert verbalize_sentiment(s) == "讨论热度高，方向分歧明显"

    def test_weak_signals_fall_back_to_label(self):
        """A sub-0.5 dominant dim shouldn't fabricate a strong phrase."""
        s = _sent(label="neutral", uncertainty=0.3)
        assert verbalize_sentiment(s) == "情绪平稳"

    def test_strong_optimism(self):
        s = _sent(label="bullish", optimism=0.8)
        assert verbalize_sentiment(s) == "情绪偏乐观"


# ─── sentiment_score_tail ────────────────────────────────────────────────────

class TestSentimentScoreTail:
    def test_strong_signal_includes_score(self):
        s = _sent(label="bearish", fear=0.8)
        tail = sentiment_score_tail(s)
        assert "0.80" in tail
        assert "担忧" in tail

    def test_weak_signal_has_no_score(self):
        s = _sent(label="neutral", uncertainty=0.3)
        # Below threshold: tail should be label-only or empty
        tail = sentiment_score_tail(s)
        assert "0.30" not in tail


# ─── verbalize_trend ─────────────────────────────────────────────────────────

class TestVerbalizeTrend:
    def test_rising_cross_platform_broad(self):
        t = TrendProfile(
            trend_direction="rising",
            persistence="continuing",
            platform_spread="cross-platform",
            discussion_breadth="broad",
        )
        phrase = verbalize_trend(t)
        assert "明显升温" in phrase
        assert "双平台共振" in phrase
        assert "较广泛" in phrase

    def test_stable_single_platform_narrow_produces_short_phrase(self):
        """Default/neutral fields are omitted to avoid noise."""
        t = TrendProfile(
            trend_direction="stable",
            persistence="continuing",
            platform_spread="single-platform",
            discussion_breadth="narrow",
        )
        phrase = verbalize_trend(t)
        # Only "持续讨论" should be left; neutral spread/breadth are suppressed.
        assert phrase == "持续讨论"

    def test_rising_but_narrow_flags_caveat(self):
        t = TrendProfile(
            trend_direction="rising",
            persistence="new",
            platform_spread="reddit-led",
            discussion_breadth="narrow",
        )
        phrase = verbalize_trend(t)
        # Rising-but-narrow is a real caveat and must surface.
        assert "小范围" in phrase


# ─── derive_insurance_framework ──────────────────────────────────────────────

class TestDeriveInsuranceFramework:
    def _topic(self, rule_label="", insurance_angle=""):
        return TopicCluster(
            cluster_id=uuid.uuid4().hex[:12],
            rule_label=rule_label,
            insurance_angle=insurance_angle,
        )

    def test_empty_inputs_return_empty_framework(self):
        framework = derive_insurance_framework(self._topic())
        assert framework.implications == ""
        assert framework.triggers == ""

    def test_rule_label_drives_implication(self):
        framework = derive_insurance_framework(self._topic(rule_label="美债与收益率"))
        assert "久期" in framework.implications
        assert framework.triggers  # always has observation text

    def test_free_text_fallback_when_no_rule(self):
        framework = derive_insurance_framework(
            self._topic(insurance_angle="可能对能源板块产生压力")
        )
        assert framework.implications == "可能对能源板块产生压力"


# ─── render_insurance_framework ──────────────────────────────────────────────

class TestRenderInsuranceFramework:
    def test_empty_framework_renders_nothing(self):
        assert render_insurance_framework(InsuranceFramework()) == []

    def test_both_halves_render(self):
        lines = render_insurance_framework(
            InsuranceFramework(implications="A", triggers="B")
        )
        assert any("配置含义：A" in l for l in lines)
        assert any("观察/触发条件：B" in l for l in lines)


# ─── build_news_social_bridge ────────────────────────────────────────────────

class _FakeNews:
    def __init__(self, title, summary=""):
        self.title = title
        self.summary = summary


def _topic_with_text(headline, rule_label="", discussion="", trend_dir="stable"):
    return TopicCluster(
        cluster_id=uuid.uuid4().hex[:12],
        headline=headline,
        rule_label=rule_label,
        discussion_focus=discussion,
        posts=[UnifiedPost(platform="reddit", post_id="1", channel="r/x", title=headline)],
        trend=TrendProfile(trend_direction=trend_dir),
    )


class TestBuildNewsSocialBridge:
    def test_empty_inputs_return_empty(self):
        assert build_news_social_bridge([], [], 0) == ""
        news = [_FakeNews("Fed rate cut expected tomorrow")]
        assert build_news_social_bridge(news, [], 0) == ""

    def test_overlap_reports_alignment(self):
        news = [_FakeNews("Fed rate cut expected after dovish FOMC minutes")]
        topic = _topic_with_text(
            "FOMC minutes shift dovish",
            rule_label="美联储与利率政策",
            discussion="dovish minutes",
        )
        bridge = build_news_social_bridge(news, [topic], linked_count=1)
        assert "一致" in bridge or "重合" in bridge

    def test_rising_community_topic_flagged_when_news_silent(self):
        news = [_FakeNews("Tesla earnings beat estimates")]
        topic = _topic_with_text(
            "Yuan intervention deep dive",
            rule_label="中国经济",
            discussion="yuan pricing",
            trend_dir="rising",
        )
        bridge = build_news_social_bridge(news, [topic], linked_count=0)
        assert "升温" in bridge or "未进入" in bridge or "轨迹" in bridge
