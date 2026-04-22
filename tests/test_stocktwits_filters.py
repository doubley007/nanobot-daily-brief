"""
Tests for StockTwits noise filters.

These filters run offline — no network. We drive them with synthetic
CommunityPost objects shaped like real live samples seen during the
initial probe (see the live-data analysis that drove the filter rules).
"""
from community.base import CommunityPost
from community.stocktwits_source import (
    StockTwitsFilterConfig,
    _apply_author_tag_bias,
    _extract_author,
    _is_pump_signal,
    _is_substantive,
    _passes_quality_threshold,
    filter_posts,
)
from community.schema import SentimentProfile, TopicCluster, UnifiedPost


def _post(username, title, followers=500, likes=0):
    # Mirrors stocktwits_source._parse_messages shape: engagement = min(followers//50, 200) + likes*5
    engagement = min(followers // 50, 200) + likes * 5
    return CommunityPost(
        title=title,
        score=engagement,
        num_comments=likes,
        url=f"https://stocktwits.com/{username}/message/1",
        subreddit="$SPY",
        created_utc=0.0,
    )


# ─── Individual rule tests ───────────────────────────────────────────────────

class TestExtractAuthor:
    def test_basic_url(self):
        p = _post("DragonAlgo", "x")
        assert _extract_author(p) == "dragonalgo"

    def test_missing_url_returns_empty(self):
        p = CommunityPost(title="x", score=0, num_comments=0, url="", subreddit="$SPY")
        assert _extract_author(p) == ""


class TestIsPumpSignal:
    def test_full_template_is_signal(self):
        body = "Entry: 100\nStop: 98\nTP1: 105\nTP2: 110"
        assert _is_pump_signal(body)

    def test_one_marker_alone_is_not_signal(self):
        """A trader saying 'entry around 100' shouldn't be flagged."""
        assert not _is_pump_signal("Good entry: 100 on $SPY")

    def test_two_markers_still_not_signal(self):
        """Raise the bar: need all three markers to avoid false positives."""
        assert not _is_pump_signal("Entry: 100 Stop: 98, watching $SPY")


class TestIsSubstantive:
    def test_multi_cashtag_one_liner_is_not_substantive(self):
        assert not _is_substantive("$SPY $DJT")

    def test_long_macro_post_is_substantive(self):
        text = (
            "$SPY $TLT Fed nominee Warsh testimony flagged "
            "balance-sheet reduction as the next priority."
        )
        assert _is_substantive(text)

    def test_short_body_with_cashtags_only(self):
        # "$SPY havens …" non-cashtag body = "havens …" = 8 chars, below 20
        assert not _is_substantive("$SPY havens …")


# ─── _passes_quality_threshold composition ───────────────────────────────────

class TestQualityThreshold:
    def _cfg(self):
        return StockTwitsFilterConfig(min_followers=50)

    def test_pump_account_rejected(self):
        p = _post("DragonAlgo", "Entry: 1 Stop: 2 TP1: 3", followers=10000)
        assert not _passes_quality_threshold(p, self._cfg())

    def test_pump_content_rejected_even_from_normal_account(self):
        p = _post("NormalTrader", "Entry: 1 Stop: 2 TP1: 3", followers=10000)
        assert not _passes_quality_threshold(p, self._cfg())

    def test_one_liner_rejected(self):
        p = _post("User1", "$SPY $DJT", followers=10000)
        assert not _passes_quality_threshold(p, self._cfg())

    def test_legitimate_macro_post_passes(self):
        p = _post(
            "Ro_Patel",
            "Fed Chair nominee Kevin Warsh emphasized balance-sheet shrinking",
            followers=5000,
        )
        assert _passes_quality_threshold(p, self._cfg())

    def test_low_follower_author_rejected(self):
        p = _post("newbie", "decent post about $SPY Fed policy moves", followers=10)
        assert not _passes_quality_threshold(p, self._cfg())


# ─── End-to-end filter_posts ─────────────────────────────────────────────────

class TestFilterPostsEndToEnd:
    def test_mixed_input_keeps_signal_drops_noise(self):
        cfg = StockTwitsFilterConfig(min_followers=50)
        posts = [
            _post("DragonAlgo", "Entry: 1 Stop: 2 TP1: 3", followers=1000),
            _post("User", "$SPY $DJT", followers=1000),
            _post("Analyst", "Fed policy shift: FOMC minutes turned dovish on rates", followers=2000),
        ]
        kept = filter_posts(posts, cfg)
        kept_titles = [p.title for p in kept]
        assert any("Fed policy" in t for t in kept_titles)
        assert not any("Entry:" in t for t in kept_titles)
        assert not any(t == "$SPY $DJT" for t in kept_titles)


# ─── Author-tag bias ─────────────────────────────────────────────────────────

def _tagged_cluster(bull_n, bear_n, llm_label="neutral", llm_optimism=0.0, llm_fear=0.0):
    posts = [
        UnifiedPost(
            platform="stocktwits", post_id=f"b{i}", channel="$SPY",
            title=f"[ST:Bullish] bullish post {i}",
        ) for i in range(bull_n)
    ] + [
        UnifiedPost(
            platform="stocktwits", post_id=f"bear{i}", channel="$SPY",
            title=f"[ST:Bearish] bearish post {i}",
        ) for i in range(bear_n)
    ]
    return TopicCluster(
        cluster_id="c1", posts=posts, platforms=["stocktwits"], rule_label="x",
        sentiment=SentimentProfile(
            label=llm_label, optimism=llm_optimism, fear=llm_fear,
            dominant_dimension="optimism" if llm_optimism else ("fear" if llm_fear else ""),
            intensity=max(llm_optimism, llm_fear),
        ),
    )


class TestApplyAuthorTagBias:
    def test_majority_overrides_neutral_llm(self):
        c = _tagged_cluster(8, 1, llm_label="neutral")
        _apply_author_tag_bias([c])
        assert c.sentiment.label == "bullish"
        assert c.sentiment.optimism > 0

    def test_genuinely_mixed_leaves_llm_alone(self):
        c = _tagged_cluster(5, 5, llm_label="neutral")
        before_label = c.sentiment.label
        _apply_author_tag_bias([c])
        assert c.sentiment.label == before_label

    def test_agreeing_llm_is_not_downweighted(self):
        """If the LLM already agrees with the tag majority, don't clobber its scores."""
        c = _tagged_cluster(8, 1, llm_label="bullish", llm_optimism=0.85)
        _apply_author_tag_bias([c])
        # Bias only raises optimism; it never lowers it.
        assert c.sentiment.label == "bullish"
        assert c.sentiment.optimism >= 0.85
