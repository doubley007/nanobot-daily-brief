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


def check_ollama() -> bool:
    """Check if the local Ollama service is reachable."""
    import requests
    base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1").rstrip("/")
    try:
        # Ollama exposes a simple health endpoint at the root
        health_url = base.replace("/v1", "")
        resp = requests.get(health_url, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


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

    if not check_ollama():
        warnings.append("Ollama service unreachable — LLM enrichment will use fallback rules")
        log_message("Ollama not reachable at startup", level="WARN")

    return True, warnings


# ─── Pipeline steps ───────────────────────────────────────────────────────────

PIPELINE_STEPS = [
    "market data",
    "news sources",
    "community",
    "community analyst",
    "brief generation",
    "telegram",
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
        status["market data"] = "success"
        return MarketSnapshot(
            us_equities=market.us_equities,
            rates=market.rates,
            asia_sg=market.asia_sg,
        )
    except Exception as e:
        log_message(f"Market data fetch failed: {e}", level="ERROR")
        status["market data"] = "failed"
        return MarketSnapshot(
            us_equities="暂时无法获取美股主要指数数据",
            rates="暂时无法获取利率数据",
            asia_sg="暂时无法获取亚洲/新加坡市场数据",
        )


def safe_get_news_items(status: dict[str, str]) -> list[NewsItem]:
    try:
        raw_news = fetch_financial_news(limit=10)
        news_items = [
            NewsItem(
                title=item.title,
                summary=item.summary,
                source=item.source,
                category=item.category,
                url=item.url,
                published_at=item.published_at,
            )
            for item in raw_news
        ]
        log_message(f"News fetched successfully: {len(news_items)} items.")
        if not news_items:
            status["news sources"] = "failed (no items)"
        elif len(news_items) < 5:
            status["news sources"] = f"partial success ({len(news_items)} items)"
        else:
            status["news sources"] = f"success ({len(news_items)} items)"
        return news_items
    except Exception as e:
        log_message(f"News fetch failed: {e}", level="ERROR")
        status["news sources"] = "failed"
        return []


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

    if report.headline_topics or report.sentiment_structure:
        log_message(
            f"Community analyst: {len(report.headline_topics)} headline, "
            f"{len(report.noise_topics)} noise"
        )
        status["community analyst"] = "success"
    else:
        status["community analyst"] = "no synthesis"

    return sentiments, report


def build_daily_brief(status: dict[str, str]) -> str:
    from llm_adapter import local_llm_callable

    llm_for_community = local_llm_callable if check_ollama() else None
    if llm_for_community is None:
        log_message("Ollama unreachable — community section will be hidden", level="WARN")

    market_snapshot = safe_get_market_snapshot(status)
    news_items = safe_get_news_items(status)
    community_sentiments, community_report = safe_get_community_sentiment(
        status, llm_callable=llm_for_community
    )

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
        llm_callable=local_llm_callable,
    )
    log_message("Brief generated successfully.")
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
            f"📌 每日金融简报 | {datetime.now().strftime('%Y-%m-%d')}\n\n"
            f"系统今天生成简报时出现异常：{e}\n"
            f"请稍后手动检查数据源或日志。"
        )
        log_message(f"Brief generation failed: {e}", level="ERROR")
        status["brief generation"] = "failed (fallback used)"
        text = fallback_text

    # Prepend degraded-mode warnings if any
    if warnings:
        header = "⚠️ 系统提示：\n" + "\n".join(f"• {w}" for w in warnings) + "\n\n"
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
