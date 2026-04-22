"""
Tests for community.calibration — sentiment-vs-author-tag logger.
"""
import json
import uuid
from pathlib import Path

import pytest

from community.calibration import (
    MIN_TAGGED_POSTS_FOR_ROW,
    _majority_from_tags,
    record_cluster_calibration,
    summarize_log,
)
from community.schema import SentimentProfile, TopicCluster, UnifiedPost


def _cluster_with_tags(bull_n, bear_n, untagged_n=0, llm_label="neutral"):
    """Build a TopicCluster with the required number of [ST:...] tagged posts."""
    posts = []
    for i in range(bull_n):
        posts.append(UnifiedPost(
            platform="stocktwits", post_id=f"b{i}", channel="$SPY",
            title=f"[ST:Bullish] something positive {i}",
        ))
    for i in range(bear_n):
        posts.append(UnifiedPost(
            platform="stocktwits", post_id=f"bear{i}", channel="$SPY",
            title=f"[ST:Bearish] something negative {i}",
        ))
    for i in range(untagged_n):
        posts.append(UnifiedPost(
            platform="stocktwits", post_id=f"u{i}", channel="$SPY",
            title=f"no tag here {i}",
        ))
    return TopicCluster(
        cluster_id=uuid.uuid4().hex[:12],
        posts=posts,
        platforms=["stocktwits"],
        rule_label="test",
        sentiment=SentimentProfile(label=llm_label),
    )


# ─── _majority_from_tags ─────────────────────────────────────────────────────

class TestMajorityFromTags:
    def test_no_tagged_posts_returns_zeros(self):
        c = _cluster_with_tags(0, 0, untagged_n=5)
        label, total, ratio = _majority_from_tags(c)
        assert label == ""
        assert total == 0

    def test_clear_bullish_majority(self):
        c = _cluster_with_tags(7, 1)
        label, total, ratio = _majority_from_tags(c)
        assert label == "bullish"
        assert total == 8
        assert ratio > 0.8

    def test_clear_bearish_majority(self):
        c = _cluster_with_tags(1, 7)
        label, _, _ = _majority_from_tags(c)
        assert label == "bearish"

    def test_near_split_is_mixed(self):
        c = _cluster_with_tags(5, 5)
        label, _, _ = _majority_from_tags(c)
        assert label == "mixed"

    def test_threshold_at_60pct(self):
        """6 bull / 4 bear = 60% — should be bullish (inclusive threshold)."""
        c = _cluster_with_tags(6, 4)
        label, _, _ = _majority_from_tags(c)
        assert label == "bullish"


# ─── record_cluster_calibration ──────────────────────────────────────────────

class TestRecordClusterCalibration:
    def test_skips_when_below_min_tagged(self, tmp_path):
        """Single tagged post is not ground truth — don't record."""
        c = _cluster_with_tags(1, 0, untagged_n=5)
        assert MIN_TAGGED_POSTS_FOR_ROW == 2  # sanity
        log = tmp_path / "calib.jsonl"
        result = record_cluster_calibration(c, log_path=log)
        assert result is None
        assert not log.exists()

    def test_records_when_enough_tags(self, tmp_path):
        c = _cluster_with_tags(3, 0, llm_label="bullish")
        log = tmp_path / "calib.jsonl"
        row = record_cluster_calibration(c, log_path=log)
        assert row is not None
        assert row["author_majority"] == "bullish"
        assert row["llm_label"] == "bullish"
        assert row["agree"] is True

    def test_disagreement_is_recorded(self, tmp_path):
        c = _cluster_with_tags(3, 0, llm_label="bearish")
        log = tmp_path / "calib.jsonl"
        row = record_cluster_calibration(c, log_path=log)
        assert row["agree"] is False

    def test_jsonl_format_on_disk(self, tmp_path):
        c = _cluster_with_tags(2, 0)
        log = tmp_path / "calib.jsonl"
        record_cluster_calibration(c, log_path=log)
        lines = [l for l in log.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert "ts" in parsed
        assert "cluster_id" in parsed


# ─── summarize_log ───────────────────────────────────────────────────────────

class TestSummarizeLog:
    def test_missing_file_returns_zero_total(self, tmp_path):
        stats = summarize_log(log_path=tmp_path / "nope.jsonl")
        assert stats["total"] == 0
        assert stats["agree_rate"] == 0.0

    def test_aggregates_agree_rate(self, tmp_path):
        log = tmp_path / "calib.jsonl"
        record_cluster_calibration(
            _cluster_with_tags(3, 0, llm_label="bullish"), log_path=log
        )
        record_cluster_calibration(
            _cluster_with_tags(0, 3, llm_label="bullish"), log_path=log
        )
        stats = summarize_log(log_path=log)
        assert stats["total"] == 2
        assert stats["agree"] == 1
        assert stats["agree_rate"] == 0.5

    def test_last_n_windowing(self, tmp_path):
        log = tmp_path / "calib.jsonl"
        # 3 disagree + 1 agree → last-1 agree_rate should be 1.0
        for _ in range(3):
            record_cluster_calibration(
                _cluster_with_tags(3, 0, llm_label="bearish"), log_path=log
            )
        record_cluster_calibration(
            _cluster_with_tags(3, 0, llm_label="bullish"), log_path=log
        )
        recent = summarize_log(log_path=log, last_n=1)
        assert recent["total"] == 1
        assert recent["agree_rate"] == 1.0
