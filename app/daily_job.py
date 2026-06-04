from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from financial_brief_formatter import (
    BriefingInput,
    MarketSnapshot,
    NewsItem,
    format_daily_brief,
)
from community.orchestrator import run_community_analyst
from community.schema import CommunityAnalystReport
from market_data import get_market_summary
from news_fetcher import fetch_financial_news
from telegram_sender import send_to_telegram


LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "daily_job.log"
MAX_LOG_SIZE = 1 * 1024 * 1024  # 1 MB
LOG_BACKUP_COUNT = 3            # keep daily_job.log.1 … .3

# ─── Logging ──────────────────────────────────────────────────────────────────

def _rotate_log_if_needed() -> None:
    """Simple size-based log rotation (no extra dependencies)."""
    if not LOG_FILE.exists() or LOG_FILE.stat().st_size < MAX_LOG_SIZE:
        return
    for i in range(LOG_BACKUP_COUNT, 0, -1):
        src = LOG_FILE.with_suffix(f".log.{i}") if i > 0 else LOG_FILE
        dst = LOG_FILE.with_suffix(f".log.{i + 1}") if i < LOG_BACKUP_COUNT else None
        if i == LOG_BACKUP_COUNT:
            src = LOG_FILE.with_suffix(f".log.{i}")
            if src.exists():
                src.unlink()
        else:
            src = LOG_FILE.with_suffix(f".log.{i}")
            dst = LOG_FILE.with_suffix(f".log.{i + 1}")
            if src.exists():
                src.rename(dst)
    LOG_FILE.rename(LOG_FILE.with_suffix(".log.1"))


def log_message(message: str, level: str = "INFO") -> None:
    _rotate_log_if_needed()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] [{level}] {message}"
    print(full_message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(full_message + "\n")


# ─── Pre-flight health checks ────────────────────────────────────────────────

REQUIRED_ENV_VARS = [
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "FINNHUB_API_KEY",
]

OPTIONAL_ENV_VARS = [
    "OLLAMA_API_BASE",
    "OLLAMA_MODEL",
    "FMP_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
]


def check_env_vars() -> list[str]:
    """Validate required environment variables. Return list of warnings."""
    warnings = []
    for var in REQUIRED_ENV_VARS:
        if not os.getenv(var, "").strip():
            warnings.append(f"Missing required env var: {var}")
    for var in OPTIONAL_ENV_VARS:
        if not os.getenv(var, "").strip():
            log_message(f"Optional env var not set: {var}", level="WARN")
    return warnings


def check_llm() -> bool:
    """Check if the configured LLM backend is usable (Ollama or remote API)."""
    from llm_adapter import check_llm_available
    return check_llm_available()


def run_preflight() -> tuple[bool, list[str]]:
    """
    Run all pre-flight checks.
    Returns (can_proceed, list_of_warnings).
    """
    warnings = check_env_vars()

    # Fatal: cannot send without Telegram credentials
    fatal = [w for w in warnings if "TELEGRAM" in w]
    if fatal:
        log_message(f"FATAL: {fatal}", level="ERROR")
        return False, warnings

    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if not check_llm():
        warnings.append(
            f"LLM backend ({provider}) unreachable — enrichment will use fallback rules"
        )
        log_message(f"LLM backend ({provider}) not reachable at startup", level="WARN")

    return True, warnings


# ─── Pipeline steps ───────────────────────────────────────────────────────────

PIPELINE_STEPS = [
    "market data",
    "news sources",
    "community",
    "community analyst",
    "derived signals",
    "brief generation",
    "telegram",
]

_DERIVED_SIGNAL_ASSETS = [
    "gold", "bitcoin", "nvidia", "sp500", "oil",
    "sti", "dbs", "ocbc", "uob", "cict", "nasdaq", "sgd", "silver", "copper",
]


def _new_status() -> dict[str, str]:
    return {step: "pending" for step in PIPELINE_STEPS}


def log_status_summary(status: dict[str, str]) -> None:
    """Print a concise end-of-pipeline source status summary."""
    log_message("-" * 40)
    log_message("Pipeline source status summary:")
    width = max(len(step) for step in PIPELINE_STEPS)
    for step in PIPELINE_STEPS:
        log_message(f"  - {step.ljust(width)} : {status[step]}")
    log_message("-" * 40)


def safe_get_market_snapshot(status: dict[str, str]) -> MarketSnapshot:
    try:
        market = get_market_summary()
        log_message("Market data fetched successfully.")
        sg_ext = getattr(market, "singapore_extended", "") or ""
        if sg_ext:
            log_message(f"Singapore extended data: {sg_ext[:80]}")
        status["market data"] = "success"
        return MarketSnapshot(
            us_equities=market.us_equities,
            rates=market.rates,
            asia_sg=market.asia_sg,
            singapore_extended=sg_ext or None,
        )
    except Exception as e:
        log_message(f"Market data fetch failed: {e}", level="ERROR")
        status["market data"] = "failed"
        return MarketSnapshot(
            us_equities="data unavailable for major US indices",
            rates="data unavailable for rates",
            asia_sg="data unavailable for Asia / Singapore markets",
        )


def safe_get_news_items(status: dict[str, str]) -> list[NewsItem]:
    try:
        # Full fetch: topic dedup keeps up to 8 per topic, giving RAG broad coverage.
        rag_news = fetch_financial_news(limit=50)

        # Index everything into RAG store — the bot's Q&A benefits from the full set.
        try:
            from assistant.rag.news_indexer import index_news
            n_indexed = index_news(rag_news)
            log_message(f"RAG: indexed {n_indexed} news items into knowledge store")
        except Exception as e:
            log_message(f"RAG news index failed (non-fatal): {e}", level="WARN")

        # For the daily brief, re-apply bucket diversity selection to get 8 varied items.
        from news_fetcher import select_diverse_by_bucket
        brief_news = select_diverse_by_bucket(rag_news, top_n=8)

        news_items = [
            NewsItem(
                title=item.title,
                summary=item.summary,
                source=item.source,
                category=item.category,
                url=item.url,
                published_at=item.published_at,
            )
            for item in brief_news
        ]
        log_message(
            f"News fetched: {len(rag_news)} for RAG, {len(news_items)} for brief."
        )
        if not news_items:
            status["news sources"] = "failed (no items)"
        elif len(news_items) < 5:
            status["news sources"] = f"partial success ({len(news_items)} items)"
        else:
            status["news sources"] = f"success ({len(news_items)} items, {len(rag_news)} indexed)"
        return news_items
    except Exception as e:
        log_message(f"News fetch failed: {e}", level="ERROR")
        status["news sources"] = "failed"
        return []


def safe_index_community_raw(status: dict[str, str]) -> int:
    """
    Fetch raw community posts and index them into the RAG store — no LLM required.

    This runs unconditionally before safe_get_community_sentiment so the RAG
    knowledge base always has fresh posts even when the LLM is offline.
    Only fetch+filter+normalize steps are used here; LLM cluster analysis
    happens separately in safe_get_community_sentiment.
    """
    total_indexed = 0
    try:
        from community.normalize import normalize_posts
        from community.llm_analyst import dedupe_posts
        from assistant.rag.community_indexer import index_community

        # ── Reddit ────────────────────────────────────────────────────────────
        try:
            from community.reddit_source import (
                fetch_reddit_posts, filter_posts as reddit_filter, load_filter_config as reddit_cfg,
            )
            cfg = reddit_cfg()
            raw = fetch_reddit_posts(cfg.subreddits)
            filtered = reddit_filter(raw, cfg)
            unified = normalize_posts("reddit", filtered)
            unified = dedupe_posts(unified)
            n = index_community(unified)
            total_indexed += n
            log_message(f"RAG raw-index: reddit {n} posts")
        except Exception as e:
            log_message(f"RAG raw-index reddit failed (non-fatal): {e}", level="WARN")

        # ── Discord ───────────────────────────────────────────────────────────
        try:
            from community.discord_source import (
                fetch_discord_posts, filter_posts as discord_filter,
                load_filter_config as discord_cfg,
            )
            cfg = discord_cfg()
            if cfg.bot_token and cfg.channel_ids:
                raw = fetch_discord_posts(cfg)
                filtered = discord_filter(raw, cfg)
                unified = normalize_posts("discord", filtered)
                unified = dedupe_posts(unified)
                n = index_community(unified)
                total_indexed += n
                log_message(f"RAG raw-index: discord {n} posts")
        except Exception as e:
            log_message(f"RAG raw-index discord failed (non-fatal): {e}", level="WARN")

        log_message(f"RAG raw community index: {total_indexed} posts total")
        status["community"] = f"raw-indexed {total_indexed} posts (LLM analysis pending)"
    except Exception as e:
        log_message(f"RAG raw community index failed: {e}", level="WARN")

    return total_indexed


def safe_get_community_sentiment(
    status: dict[str, str],
    llm_callable=None,
) -> tuple[list, CommunityAnalystReport]:
    """
    Fetch per-platform sentiments and run the cross-platform analyst pass.

    Returns (sentiments, analyst_report). Both may be empty if LLM is
    unavailable or no platform yields data.
    """
    if llm_callable is None:
        status["community"] = "skipped (LLM unavailable)"
        status["community analyst"] = "skipped (LLM unavailable)"
        return [], CommunityAnalystReport()

    try:
        sentiments, report = run_community_analyst(llm_callable=llm_callable)
    except Exception as e:
        log_message(f"Community orchestrator failed: {e}", level="ERROR")
        status["community"] = "failed"
        status["community analyst"] = "failed"
        return [], CommunityAnalystReport()

    if not sentiments:
        log_message("Community: no platform returned trending topics")
        status["community"] = "no data"
        status["community analyst"] = "no data"
        return [], report

    platform_counts = ", ".join(f"{s.platform}={s.post_count}" for s in sentiments)
    log_message(f"Community sentiments fetched: {platform_counts}")
    status["community"] = f"success ({len(sentiments)} platforms)"

    # 非破坏性：把聚合好的社区 UnifiedPost 索引进 RAG 知识库。
    # 失败不影响日报主流程。
    try:
        from assistant.rag.community_indexer import index_community
        indexed = 0
        for s in sentiments:
            posts: list = []
            for cluster in s.trending_topics:
                posts.extend(cluster.posts)
            indexed += index_community(posts)
        log_message(f"RAG: indexed {indexed} community posts into knowledge store")
    except Exception as e:
        log_message(f"RAG community index failed (non-fatal): {e}", level="WARN")

    if report.headline_topics or report.sentiment_structure:
        log_message(
            f"Community analyst: {len(report.headline_topics)} headline, "
            f"{len(report.noise_topics)} noise"
        )
        status["community analyst"] = "success"
    else:
        status["community analyst"] = "no synthesis"

    return sentiments, report


def safe_refresh_derived_signals(status: dict[str, str]) -> None:
    """
    Compute and persist DerivedSignals for all tracked assets.
    Non-fatal: a failure here must not block the daily brief.
    """
    try:
        from assistant.rag.derived_signals import refresh_derived_signals_batch
        results = refresh_derived_signals_batch(_DERIVED_SIGNAL_ASSETS, window_hours=72)
        n_ok = sum(1 for v in results.values() if v == "ok")
        n_sparse = sum(1 for v in results.values() if v == "sparse")
        n_err = sum(1 for v in results.values() if v.startswith("error:"))
        if n_err == len(results):
            status["derived signals"] = "failed (all assets errored)"
        elif n_err > 0:
            status["derived signals"] = f"partial ({n_ok} ok, {n_sparse} sparse, {n_err} error)"
        else:
            status["derived signals"] = f"success ({n_ok} ok, {n_sparse} sparse)"
        log_message(f"Derived signals refreshed: {results}")
    except Exception as e:
        log_message(f"Derived signal refresh failed (non-fatal): {e}", level="WARN")
        status["derived signals"] = f"failed: {e}"


def run_source_health_check() -> str:
    """
    Run source health check and return a compact footer line.
    Non-fatal: any exception returns an empty string so the brief is unaffected.
    """
    try:
        from source_health import check_all_sources, render_status_footer
        statuses = check_all_sources(write_report=True)
        footer = render_status_footer(statuses)
        n_down = sum(1 for s in statuses if s.status == "down")
        n_degraded = sum(1 for s in statuses if s.status == "degraded")
        if n_down or n_degraded:
            log_message(
                f"Source health: {n_down} down, {n_degraded} degraded — see reports/source_status.json",
                level="WARN",
            )
        else:
            log_message("Source health: all sources OK")
        return footer
    except Exception as e:
        log_message(f"Source health check failed (non-fatal): {e}", level="WARN")
        return ""


def _build_risk_review(alerts: list, llm_callable=None) -> str:
    """
    Build a '昨日风险回顾' section for the morning brief.
    Uses LLM to generate a short narrative; falls back to a bullet list.
    """
    if not alerts:
        return ""

    high   = [a for a in alerts if a.get("severity") == "HIGH"]
    medium = [a for a in alerts if a.get("severity") == "MEDIUM"]
    total  = len(alerts)

    # Try LLM narrative
    if llm_callable:
        try:
            bullets = "\n".join(
                f"- [{a.get('severity','?')}][{a.get('bucket','')}] "
                f"{a['title']} (trigger: {a.get('keyword','')})"
                for a in alerts[-15:]
            )
            prompt = (
                "You are a Singapore insurance-investment risk analyst. Below are the "
                "risk alerts fired in the past 24 hours. In 2-4 concise sentences "
                "(coherent paragraph, NOT a list) write a [Risk Review — last 24h] "
                "covering: the dominant risk themes, the overall read on impact for "
                "the insurance book's fixed-income / credit / SG-local exposures, and "
                "what deserves focus today. Respond in English.\n\n"
                f"{bullets}"
            )
            from llm_adapter import local_llm_plain
            narrative = local_llm_plain(prompt, timeout=60)
            header = f"━━━ Risk Review — last 24h ({total} alerts) ━━━"
            if high:
                header += f"\n🔴 HIGH: {len(high)}  🟠 MEDIUM: {len(medium)}"
            return f"{header}\n\n{narrative}"
        except Exception:
            pass

    # Fallback: bullet list
    lines = [f"━━━ Risk Review — last 24h ({total} alerts) ━━━"]
    if high:
        lines.append(f"🔴 HIGH ({len(high)})")
        for a in high[-3:]:
            lines.append(f"  • {a['title'][:55]}")
    if medium:
        lines.append(f"🟠 MEDIUM ({len(medium)})")
        for a in medium[-3:]:
            lines.append(f"  • {a['title'][:55]}")
    return "\n".join(lines)


def build_daily_brief(status: dict[str, str]) -> str:
    from llm_adapter import local_llm_callable

    # Run source health check first — non-fatal, captures footer for brief
    source_footer = run_source_health_check()

    llm_for_community = local_llm_callable if check_llm() else None
    if llm_for_community is None:
        log_message("LLM backend unreachable — community section will be hidden", level="WARN")

    market_snapshot = safe_get_market_snapshot(status)
    news_items = safe_get_news_items(status)

    # Always index raw community posts into RAG regardless of LLM availability.
    # safe_get_community_sentiment may then add LLM-analyzed clusters on top.
    safe_index_community_raw(status)

    community_sentiments, community_report = safe_get_community_sentiment(
        status, llm_callable=llm_for_community
    )

    # Refresh derived signals now that news + community are indexed
    safe_refresh_derived_signals(status)

    # ── Risk detection ────────────────────────────────────────────────────────
    try:
        from risk_detector import (
            detect_risk_deduped, save_alert, has_risk_keyword,
            load_seen, save_seen, load_cooldown, save_cooldown,
            format_alert_msg,
        )
        from risk_monitor import _llm_analyze_risk

        _seen = load_seen()
        _cooldown = load_cooldown()
        _fired_titles: list[str] = []
        for item in news_items:
            if not has_risk_keyword(item.title, item.summary or ""):
                title = (item.title or "").strip()
                if title and title not in _seen:
                    _seen.append(title)
                continue
            article = {
                "title":   item.title,
                "summary": item.summary or "",
                "source":  item.source,
                "url":     item.url or "",
                "bucket":  getattr(item, "bucket", ""),
            }
            llm_result = _llm_analyze_risk(
                title    = item.title,
                summary  = item.summary or "",
                source   = item.source,
                keywords = item.title,
                bucket   = getattr(item, "bucket", ""),
            )
            if not llm_result.get("confirmed", False):
                _seen.append(item.title.strip())
                log_message(f"LLM rejected risk: {item.title[:60]}")
                continue
            article["llm_analysis"] = llm_result
            alert = detect_risk_deduped(article, llm_result["sentiment"], _seen, _cooldown, _fired_titles)
            if alert:
                save_alert(alert)
                log_message(f"[INFO] Risk detected: {alert['keyword']} ({alert.get('severity','?')}) — {alert['title'][:60]}")
                try:
                    send_to_telegram(format_alert_msg(alert))
                    log_message("[INFO] Alert sent to Telegram")
                except Exception as tg_err:
                    log_message(f"Alert Telegram send failed (non-fatal): {tg_err}", level="WARN")
        save_seen(_seen)
        save_cooldown(_cooldown)
    except Exception as e:
        log_message(f"Risk detection failed (non-fatal): {e}", level="WARN")

    log_message(f"Raw news count: {len(news_items)}")

    briefing_input = BriefingInput(
        date_str=datetime.now().strftime("%Y-%m-%d"),
        market_snapshot=market_snapshot,
        news_items=news_items,
        watchlist=[],
        community_sentiments=community_sentiments,
        community_report=community_report,
    )

    brief_text = format_daily_brief(
        briefing_input,
        llm_callable=llm_for_community,
    )

    render_stats = getattr(community_report, "render_stats", None)
    if render_stats:
        selected = render_stats.get("analyst_selected", 0)
        rendered = render_stats.get("formatter_rendered", 0)
        if selected and rendered != selected:
            log_message(
                f"Community headline count mismatch: analyst={selected}, "
                f"rendered={rendered} "
                f"(noise_dropped={render_stats.get('dropped_by_noise', 0)}, "
                f"should_include_false={render_stats.get('dropped_by_should_include', 0)})",
                level="WARN",
            )
        elif selected:
            log_message(
                f"Community headlines: analyst selected {selected}, "
                f"rendered {rendered} (no drop)"
            )

    log_message("Brief generated successfully.")

    # ── 昨日风险回顾板块 ──────────────────────────────────────────────────────
    try:
        from risk_detector import load_yesterday_alerts
        yesterday_alerts = load_yesterday_alerts()
        if yesterday_alerts:
            risk_section = _build_risk_review(yesterday_alerts, llm_callable=llm_for_community)
            if risk_section:
                brief_text = brief_text + "\n\n" + risk_section
    except Exception as e:
        log_message(f"Risk review section failed (non-fatal): {e}", level="WARN")

    if source_footer:
        brief_text = brief_text + "\n\n" + source_footer

    return brief_text


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    log_message("=" * 40)
    log_message("Daily brief pipeline started.")

    status = _new_status()

    # Pre-flight checks
    can_proceed, warnings = run_preflight()
    if not can_proceed:
        log_message("Aborting: pre-flight checks failed.", level="ERROR")
        log_status_summary(status)
        sys.exit(1)

    # Build brief
    try:
        text = build_daily_brief(status)
        status["brief generation"] = "success"
    except Exception as e:
        fallback_text = (
            f"📌 Daily Financial Brief | {datetime.now().strftime('%Y-%m-%d')}\n\n"
            f"The brief failed to generate today: {e}\n"
            f"Please check data sources and logs manually."
        )
        log_message(f"Brief generation failed: {e}", level="ERROR")
        status["brief generation"] = "failed (fallback used)"
        text = fallback_text

    # Prepend degraded-mode warnings if any
    if warnings:
        header = "⚠️ System notice:\n" + "\n".join(f"• {w}" for w in warnings) + "\n\n"
        text = header + text

    # Send to Telegram
    try:
        send_to_telegram(text)
        log_message("Telegram message sent successfully.")
        status["telegram"] = "success"
    except Exception as e:
        log_message(f"Telegram send failed after retries: {e}", level="ERROR")
        status["telegram"] = "failed"

    log_message("Daily brief pipeline finished.")
    log_status_summary(status)


if __name__ == "__main__":
    main()
