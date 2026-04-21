"""
Cross-platform community orchestration.

Fetches per-platform sentiment via the existing source modules, then runs
the Community Analyst LLM pass across the combined cluster set to produce
a single structured report for the daily brief.

Design goal: per-platform modules stay focused on fetch+filter+cluster+
per-cluster LLM analysis; this module owns the cross-platform synthesis.
"""
from __future__ import annotations

import logging
from typing import Callable

from community.discord_source import fetch_discord_sentiment
from community.llm_analyst import community_analyst_report
from community.reddit_source import fetch_reddit_sentiment
from community.schema import (
    CommunityAnalystReport,
    CommunitySentiment,
    TopicCluster,
)
from community.x_source import XPlanUnavailableError, fetch_x_sentiment

logger = logging.getLogger(__name__)


def _disclose_x_failure(exc: Exception) -> str:
    """Human-readable platform status line for a failed X fetch."""
    if isinstance(exc, XPlanUnavailableError):
        return "x=unavailable (plan 402)"
    return f"x=error ({type(exc).__name__})"


def run_community_analyst(
    llm_callable: Callable[[str], str] | None,
) -> tuple[list[CommunitySentiment], CommunityAnalystReport]:
    """
    Fetch every configured platform, combine clusters, run the analyst pass.

    Always returns a report (possibly empty) with `platform_status`
    populated so the formatter can disclose what ran, what skipped,
    and why — demo consumers should see the pipeline's real coverage.
    """
    sentiments: list[CommunitySentiment] = []
    status_lines: list[str] = []

    report_stub = lambda: CommunityAnalystReport(platform_status=list(status_lines))

    if llm_callable is None:
        logger.info("Community analyst skipped — LLM unavailable")
        status_lines.append("all platforms=skipped (LLM unavailable)")
        return sentiments, report_stub()

    for name, fetcher in (
        ("reddit", fetch_reddit_sentiment),
        ("x", fetch_x_sentiment),
        ("discord", fetch_discord_sentiment),
    ):
        try:
            result = fetcher(llm_callable=llm_callable)
        except XPlanUnavailableError as e:
            status_lines.append(_disclose_x_failure(e))
            continue
        except Exception as e:
            logger.warning("%s sentiment fetch raised: %s", name, e)
            status_lines.append(f"{name}=error ({type(e).__name__})")
            continue

        if result.trending_topics:
            sentiments.append(result)
            status_lines.append(f"{name}=ok ({result.post_count}贴, {len(result.trending_topics)}簇)")
        elif result.post_count == 0:
            if name == "x":
                status_lines.append("x=unavailable or not configured")
            elif name == "discord":
                status_lines.append("discord=not configured")
            else:
                status_lines.append(f"{name}=no data")
        else:
            status_lines.append(f"{name}=filtered out ({result.post_count}贴, 0簇)")

    if not sentiments:
        return sentiments, report_stub()

    all_clusters: list[TopicCluster] = []
    for s in sentiments:
        all_clusters.extend(s.trending_topics)
    all_clusters.sort(key=lambda c: c.heat_score, reverse=True)

    platforms = [s.platform for s in sentiments]
    total_posts = sum(s.post_count for s in sentiments)

    report = community_analyst_report(
        clusters=all_clusters,
        platforms=platforms,
        total_posts=total_posts,
        llm_callable=llm_callable,
    )
    report.total_clusters = len(all_clusters)
    report.platform_status = list(status_lines)

    # Attach report to each per-platform sentiment for downstream convenience
    for s in sentiments:
        s.analyst_report = report

    return sentiments, report
