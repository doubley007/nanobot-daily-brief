"""
Sentiment calibration using StockTwits author tags as ground truth.

StockTwits is the only platform where users *self-label* each post as
Bullish or Bearish. That's free calibration data for our sentiment
classifier — we can measure how often the LLM agrees with the author.

Nothing here blocks the daily brief. The calibration sample is written
to a JSONL log file; a follow-up tool or manual `tail` of the file
answers the question "is our LLM drifting?".

Schema of each JSONL row:
  {
    "ts": "<ISO-8601>",
    "cluster_id": "...",
    "post_count": <int>,
    "tagged_post_count": <int>,           # posts that had Bullish/Bearish
    "llm_label": "bullish|bearish|neutral|mixed",
    "author_majority": "bullish|bearish|mixed",
    "agree": bool,                         # llm_label == author_majority
    "bull_ratio": <float>,                 # bullish tags / tagged posts
    "headline": "<LLM topic>",
  }
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from community.schema import TopicCluster

logger = logging.getLogger(__name__)

_DEFAULT_LOG = (
    Path(__file__).resolve().parent.parent.parent
    / "logs"
    / "sentiment_calibration.jsonl"
)


def _majority_from_tags(cluster: TopicCluster) -> tuple[str, int, float]:
    """
    Return (majority_label, tagged_post_count, bull_ratio).
    majority_label is "bullish", "bearish", or "mixed" when neither side
    reaches a 60% threshold (same threshold the sentiment bias uses).
    """
    bull = bear = 0
    for p in cluster.posts:
        title = p.title or ""
        if title.startswith("[ST:Bullish]"):
            bull += 1
        elif title.startswith("[ST:Bearish]"):
            bear += 1
    total = bull + bear
    if total == 0:
        return "", 0, 0.0
    bull_ratio = bull / total
    if bull_ratio >= 0.6:
        return "bullish", total, bull_ratio
    if bull_ratio <= 0.4:
        return "bearish", total, bull_ratio
    return "mixed", total, bull_ratio


MIN_TAGGED_POSTS_FOR_ROW = 2  # single-tag clusters are noise, not signal


def record_cluster_calibration(
    cluster: TopicCluster,
    log_path: Path | None = None,
) -> dict | None:
    """
    Append one calibration row to the JSONL log. Silently no-ops if the
    cluster has fewer than MIN_TAGGED_POSTS_FOR_ROW tagged posts —
    a single self-tag is not enough to claim ground truth. Returns
    the row written, or None.
    """
    majority, tagged_total, bull_ratio = _majority_from_tags(cluster)
    if tagged_total < MIN_TAGGED_POSTS_FOR_ROW:
        return None

    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cluster_id": cluster.cluster_id,
        "post_count": cluster.post_count,
        "tagged_post_count": tagged_total,
        "llm_label": cluster.sentiment.label,
        "author_majority": majority,
        "agree": cluster.sentiment.label == majority,
        "bull_ratio": round(bull_ratio, 3),
        "headline": cluster.headline or cluster.rule_label or "",
    }

    path = log_path or _DEFAULT_LOG
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.warning("Failed to write sentiment calibration row: %s", e)
        return None
    return row


def record_batch(
    clusters: list[TopicCluster],
    log_path: Path | None = None,
) -> int:
    """Record calibration for every cluster. Returns rows written."""
    written = 0
    for c in clusters:
        if record_cluster_calibration(c, log_path) is not None:
            written += 1
    return written


# ─── Read-side helpers (for a future dashboard or quick manual check) ────────

def summarize_log(
    log_path: Path | None = None,
    last_n: int | None = None,
) -> dict:
    """
    Compute agreement statistics from the JSONL log.
    Returns {total, agree, agree_rate, bull_precision, bear_precision, ...}.
    An empty log returns zeros.
    """
    path = log_path or _DEFAULT_LOG
    if not path.exists():
        return {"total": 0, "agree": 0, "agree_rate": 0.0}

    rows: list[dict] = []
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return {"total": 0, "agree": 0, "agree_rate": 0.0}

    if last_n is not None:
        rows = rows[-last_n:]

    total = len(rows)
    if total == 0:
        return {"total": 0, "agree": 0, "agree_rate": 0.0}

    agree = sum(1 for r in rows if r.get("agree"))
    # Per-direction precision: of clusters the LLM called X, how many matched?
    llm_bull = [r for r in rows if r.get("llm_label") == "bullish"]
    llm_bear = [r for r in rows if r.get("llm_label") == "bearish"]
    bull_prec = (
        sum(1 for r in llm_bull if r.get("author_majority") == "bullish") / len(llm_bull)
        if llm_bull else 0.0
    )
    bear_prec = (
        sum(1 for r in llm_bear if r.get("author_majority") == "bearish") / len(llm_bear)
        if llm_bear else 0.0
    )

    return {
        "total": total,
        "agree": agree,
        "agree_rate": round(agree / total, 3),
        "llm_bullish_count": len(llm_bull),
        "llm_bearish_count": len(llm_bear),
        "bull_precision": round(bull_prec, 3),
        "bear_precision": round(bear_prec, 3),
    }


if __name__ == "__main__":
    import sys
    # Pretty-print current stats
    stats = summarize_log()
    print(json.dumps(stats, indent=2))
    sys.exit(0)
