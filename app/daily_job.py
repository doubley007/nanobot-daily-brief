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

def safe_get_market_snapshot() -> MarketSnapshot:
    try:
        market = get_market_summary()
        log_message("Market data fetched successfully.")
        return MarketSnapshot(
            us_equities=market.us_equities,
            rates=market.rates,
            asia_sg=market.asia_sg,
        )
    except Exception as e:
        log_message(f"Market data fetch failed: {e}", level="ERROR")
        return MarketSnapshot(
            us_equities="暂时无法获取美股主要指数数据",
            rates="暂时无法获取利率数据",
            asia_sg="暂时无法获取亚洲/新加坡市场数据",
        )


def safe_get_news_items() -> list[NewsItem]:
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
        return news_items
    except Exception as e:
        log_message(f"News fetch failed: {e}", level="ERROR")
        return []


def build_daily_brief() -> str:
    market_snapshot = safe_get_market_snapshot()
    news_items = safe_get_news_items()

    log_message(f"Raw news count: {len(news_items)}")

    briefing_input = BriefingInput(
        date_str=datetime.now().strftime("%Y-%m-%d"),
        market_snapshot=market_snapshot,
        news_items=news_items,
        watchlist=[],
    )

    from llm_adapter import local_llm_callable
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

    # Pre-flight checks
    can_proceed, warnings = run_preflight()
    if not can_proceed:
        log_message("Aborting: pre-flight checks failed.", level="ERROR")
        sys.exit(1)

    # Build brief
    try:
        text = build_daily_brief()
    except Exception as e:
        fallback_text = (
            f"📌 每日金融简报 | {datetime.now().strftime('%Y-%m-%d')}\n\n"
            f"系统今天生成简报时出现异常：{e}\n"
            f"请稍后手动检查数据源或日志。"
        )
        log_message(f"Brief generation failed: {e}", level="ERROR")
        text = fallback_text

    # Prepend degraded-mode warnings if any
    if warnings:
        header = "⚠️ 系统提示：\n" + "\n".join(f"• {w}" for w in warnings) + "\n\n"
        text = header + text

    # Send to Telegram
    try:
        send_to_telegram(text)
        log_message("Telegram message sent successfully.")
    except Exception as e:
        log_message(f"Telegram send failed after retries: {e}", level="ERROR")

    log_message("Daily brief pipeline finished.")


if __name__ == "__main__":
    main()
