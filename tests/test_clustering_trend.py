"""
Tests for the trend-derivation layer in clustering._derive_trend_profile.

Trend fields are populated deterministically from the current fetch
window — no historical store required. Tests verify the mapping rules
(direction, persistence, spread, breadth) cleanly degrade when signals
are ambiguous.
"""
import time

from community.clustering import _derive_trend_profile, mark_rising_clusters
from community.schema import TopicCluster, UnifiedPost


def _post(platform, channel, engagement=50, age_hours=2.0, author="u1"):
    return UnifiedPost(
        platform=platform,
        post_id=f"{platform}-{author}-{channel}",
        channel=channel,
        title="some finance discussion",
        engagement_raw=engagement,
        author=author,
        created_utc=time.time() - age_hours * 3600,
    )


def _cluster(posts, rise_ratio=1.0, heat=None):
    c = TopicCluster(
        cluster_id="c1",
        posts=posts,
        platforms=sorted({p.platform for p in posts}),
        heat_score=heat if heat is not None else float(sum(p.engagement_raw for p in posts)),
        is_rising=rise_ratio >= 1.8,
        rise_ratio=rise_ratio,
    )
    return c


# ─── Direction ───────────────────────────────────────────────────────────────

class TestDirection:
    def test_high_rise_ratio_is_rising(self):
        posts = [_post("reddit", "r/bonds", engagement=100, age_hours=3)]
        c = _cluster(posts, rise_ratio=2.5)
        tp = _derive_trend_profile(c)
        assert tp.trend_direction == "rising"

    def test_very_fresh_small_cluster_is_new(self):
        posts = [_post("reddit", "r/bonds", engagement=10, age_hours=1)]
        c = _cluster(posts, rise_ratio=1.0)
        tp = _derive_trend_profile(c)
        assert tp.trend_direction == "new"

    def test_old_posts_are_fading(self):
        posts = [_post("reddit", "r/bonds", engagement=10, age_hours=48)]
        c = _cluster(posts, rise_ratio=1.0)
        tp = _derive_trend_profile(c)
        assert tp.trend_direction == "fading"

    def test_middle_ground_is_stable(self):
        posts = [
            _post("reddit", "r/bonds", engagement=30, age_hours=12),
            _post("reddit", "r/bonds", engagement=30, age_hours=18, author="u2"),
        ]
        c = _cluster(posts, rise_ratio=1.0)
        tp = _derive_trend_profile(c)
        assert tp.trend_direction == "stable"


# ─── Platform spread ─────────────────────────────────────────────────────────

class TestPlatformSpread:
    def test_cross_platform(self):
        posts = [
            _post("reddit", "r/bonds"),
            _post("discord", "#macro", author="u2"),
        ]
        tp = _derive_trend_profile(_cluster(posts))
        assert tp.platform_spread == "cross-platform"

    def test_reddit_led(self):
        posts = [_post("reddit", "r/bonds")]
        tp = _derive_trend_profile(_cluster(posts))
        assert tp.platform_spread == "reddit-led"

    def test_discord_led(self):
        posts = [_post("discord", "#macro")]
        tp = _derive_trend_profile(_cluster(posts))
        assert tp.platform_spread == "discord-led"


# ─── Breadth ─────────────────────────────────────────────────────────────────

class TestBreadth:
    def test_narrow_single_author_single_channel(self):
        posts = [_post("reddit", "r/bonds", author="u1")]
        tp = _derive_trend_profile(_cluster(posts))
        assert tp.discussion_breadth == "narrow"

    def test_moderate_several_distinct(self):
        # width = 2 channels + 2 authors = 4 → "moderate" (threshold 3, < 6)
        posts = [
            _post("reddit", "r/bonds", author="u1"),
            _post("reddit", "r/investing", author="u2"),
        ]
        tp = _derive_trend_profile(_cluster(posts))
        assert tp.discussion_breadth == "moderate"

    def test_broad_many_distinct(self):
        # width = 3 channels + 3 authors = 6 → "broad" (threshold 6)
        posts = [
            _post("reddit", f"r/ch{i}", author=f"u{i}")
            for i in range(3)
        ]
        tp = _derive_trend_profile(_cluster(posts))
        assert tp.discussion_breadth == "broad"


# ─── mark_rising_clusters end-to-end ─────────────────────────────────────────

class TestMarkRisingClusters:
    def test_populates_trend_on_every_cluster(self):
        hot = [_post("reddit", "r/bonds", engagement=500, author="u1")]
        cold = [_post("reddit", "r/bonds", engagement=10, author="u2")]
        hot_c = _cluster(hot, heat=500.0)
        cold_c = _cluster(cold, heat=10.0)
        mark_rising_clusters([hot_c, cold_c])
        assert hot_c.trend.trend_direction  # something populated
        assert cold_c.trend.trend_direction

    def test_hottest_cluster_flagged_rising(self):
        """
        With N clusters, the median anchors rise_ratio. Need several small
        clusters alongside the hot one so median is low and the hot cluster
        crosses rise_ratio >= 1.8.
        """
        clusters = []
        for i in range(4):
            small = _cluster(
                [_post("reddit", "r/bonds", engagement=50, author=f"u{i}")],
                heat=50.0,
            )
            clusters.append(small)
        big = _cluster(
            [_post("reddit", "r/bonds", engagement=1000, author="uhot")],
            heat=1000.0,
        )
        clusters.append(big)
        mark_rising_clusters(clusters)
        assert big.is_rising is True
        assert all(not c.is_rising for c in clusters[:4])
