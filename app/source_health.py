"""
Source Health Check — lightweight connectivity probe for all data sources.

Each check is independent and wrapped in try/except; a single source failure
never raises. Returns a list of SourceStatus and writes reports/source_status.json.

Usage:
    from source_health import check_all_sources, render_status_footer
    statuses = check_all_sources()
    footer = render_status_footer(statuses)

Or as CLI:
    python -m app.source_health
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

Status = Literal["ok", "degraded", "down", "unconfigured"]

_PROJ = Path(__file__).resolve().parent.parent
_REPORTS_DIR = _PROJ / "reports"


@dataclass
class SourceStatus:
    name: str
    status: Status
    latency_ms: float | None = None
    detail: str = ""


def _timed(fn) -> tuple[any, float]:
    t0 = time.monotonic()
    result = fn()
    return result, round((time.monotonic() - t0) * 1000, 1)


# ─── Individual probes ────────────────────────────────────────────────────────

def _check_yfinance_ticker(symbol: str) -> SourceStatus:
    name = f"yfinance({symbol})"
    try:
        import yfinance as yf
        def _fetch():
            t = yf.Ticker(symbol)
            hist = t.history(period="2d")
            return hist

        hist, ms = _timed(_fetch)
        if hist is None or len(hist) == 0:
            return SourceStatus(name=name, status="degraded", latency_ms=ms,
                                detail="no price data returned")
        return SourceStatus(name=name, status="ok", latency_ms=ms)
    except Exception as e:
        return SourceStatus(name=name, status="down", detail=str(e)[:120])


def _check_yfinance() -> list[SourceStatus]:
    symbols = {
        "yfinance/STI": "^STI",
        "yfinance/SP500": "^GSPC",
        "yfinance/SGDUSD": "SGDUSD=X",
    }
    results = []
    for name, sym in symbols.items():
        try:
            import yfinance as yf
            def _fetch(s=sym):
                t = yf.Ticker(s)
                return t.history(period="2d")
            hist, ms = _timed(_fetch)
            if hist is None or len(hist) == 0:
                results.append(SourceStatus(name=name, status="degraded", latency_ms=ms,
                                            detail="no price data"))
            else:
                results.append(SourceStatus(name=name, status="ok", latency_ms=ms))
        except Exception as e:
            results.append(SourceStatus(name=name, status="down", detail=str(e)[:120]))
    return results


def _check_sg_market_data() -> list[SourceStatus]:
    results = []
    sg_symbols = {
        "yfinance/DBS": "D05.SI",
        "yfinance/OCBC": "O39.SI",
        "yfinance/UOB": "U11.SI",
        "yfinance/SG10Y": "SG10Y=X",
    }
    for name, sym in sg_symbols.items():
        try:
            import yfinance as yf
            def _fetch(s=sym):
                t = yf.Ticker(s)
                return t.history(period="2d")
            hist, ms = _timed(_fetch)
            if hist is None or len(hist) == 0:
                results.append(SourceStatus(name=name, status="degraded", latency_ms=ms,
                                            detail="no data"))
            else:
                results.append(SourceStatus(name=name, status="ok", latency_ms=ms))
        except Exception as e:
            results.append(SourceStatus(name=name, status="down", detail=str(e)[:120]))
    return results


def _check_rss() -> SourceStatus:
    try:
        import feedparser
        # Try one representative feed that should always be available
        test_feed = "https://feeds.bbci.co.uk/news/business/rss.xml"
        def _fetch():
            return feedparser.parse(test_feed)
        feed, ms = _timed(_fetch)
        if feed.get("bozo") and not feed.get("entries"):
            return SourceStatus(name="RSS", status="degraded", latency_ms=ms,
                                detail="feed parse error (bozo)")
        if not feed.get("entries"):
            return SourceStatus(name="RSS", status="degraded", latency_ms=ms,
                                detail="no entries")
        return SourceStatus(name="RSS", status="ok", latency_ms=ms,
                            detail=f"{len(feed.entries)} entries")
    except Exception as e:
        return SourceStatus(name="RSS", status="down", detail=str(e)[:120])


def _check_finnhub() -> SourceStatus:
    api_key = os.getenv("FINNHUB_API_KEY", "").strip()
    if not api_key:
        return SourceStatus(name="Finnhub", status="unconfigured",
                            detail="FINNHUB_API_KEY not set")
    try:
        import urllib.request
        url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={api_key}"
        def _fetch():
            with urllib.request.urlopen(url, timeout=8) as resp:
                return json.loads(resp.read())
        data, ms = _timed(_fetch)
        if "c" in data and data["c"]:
            return SourceStatus(name="Finnhub", status="ok", latency_ms=ms)
        return SourceStatus(name="Finnhub", status="degraded", latency_ms=ms,
                            detail=f"unexpected response: {str(data)[:80]}")
    except Exception as e:
        err = str(e)
        if "403" in err or "401" in err:
            return SourceStatus(name="Finnhub", status="down",
                                detail=f"auth error: {err[:80]}")
        return SourceStatus(name="Finnhub", status="down", detail=err[:120])


def _check_fmp() -> SourceStatus:
    api_key = os.getenv("FMP_API_KEY", "").strip()
    if not api_key:
        return SourceStatus(name="FMP", status="unconfigured",
                            detail="FMP_API_KEY not set")
    try:
        import urllib.request
        url = f"https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={api_key}"
        def _fetch():
            with urllib.request.urlopen(url, timeout=8) as resp:
                return resp.status, json.loads(resp.read())
        (status_code, data), ms = _timed(_fetch)
        if status_code == 402:
            return SourceStatus(name="FMP", status="down", latency_ms=ms,
                                detail="HTTP 402 — subscription required")
        if isinstance(data, list) and data:
            return SourceStatus(name="FMP", status="ok", latency_ms=ms)
        return SourceStatus(name="FMP", status="degraded", latency_ms=ms,
                            detail=f"empty response: {str(data)[:60]}")
    except Exception as e:
        err = str(e)
        if "402" in err:
            return SourceStatus(name="FMP", status="down",
                                detail="HTTP 402 — subscription required (known)")
        return SourceStatus(name="FMP", status="down", detail=err[:120])


def _check_alpha_vantage() -> SourceStatus:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
    if not api_key:
        return SourceStatus(name="AlphaVantage", status="unconfigured",
                            detail="ALPHA_VANTAGE_API_KEY not set")
    try:
        import urllib.request
        url = (
            f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE"
            f"&symbol=IBM&apikey={api_key}"
        )
        def _fetch():
            with urllib.request.urlopen(url, timeout=10) as resp:
                return json.loads(resp.read())
        data, ms = _timed(_fetch)
        if "Global Quote" in data and data["Global Quote"]:
            return SourceStatus(name="AlphaVantage", status="ok", latency_ms=ms)
        if "Note" in data:
            return SourceStatus(name="AlphaVantage", status="degraded", latency_ms=ms,
                                detail="rate-limited")
        return SourceStatus(name="AlphaVantage", status="degraded", latency_ms=ms,
                            detail=f"unexpected: {str(data)[:60]}")
    except Exception as e:
        return SourceStatus(name="AlphaVantage", status="down", detail=str(e)[:120])


def _check_reddit() -> SourceStatus:
    subs = os.getenv("REDDIT_SUBREDDITS", "").strip()
    if not subs:
        return SourceStatus(name="Reddit", status="unconfigured",
                            detail="REDDIT_SUBREDDITS not set")
    try:
        import urllib.request
        first_sub = subs.split(",")[0].strip()
        url = f"https://www.reddit.com/r/{first_sub}/hot.json?limit=1"
        req = urllib.request.Request(url, headers={"User-Agent": "nanobot-health/1.0"})
        def _fetch():
            with urllib.request.urlopen(req, timeout=8) as resp:
                return json.loads(resp.read())
        data, ms = _timed(_fetch)
        if data.get("data", {}).get("children"):
            return SourceStatus(name="Reddit", status="ok", latency_ms=ms,
                                detail=f"r/{first_sub} reachable")
        return SourceStatus(name="Reddit", status="degraded", latency_ms=ms,
                            detail="no posts returned")
    except Exception as e:
        err = str(e)
        if "429" in err:
            return SourceStatus(name="Reddit", status="degraded",
                                detail="rate-limited (429)")
        return SourceStatus(name="Reddit", status="down", detail=err[:120])


def _check_discord() -> SourceStatus:
    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        return SourceStatus(name="Discord", status="unconfigured",
                            detail="DISCORD_BOT_TOKEN not set")
    channel_ids = os.getenv("DISCORD_CHANNEL_IDS", "").strip()
    if not channel_ids:
        return SourceStatus(name="Discord", status="unconfigured",
                            detail="DISCORD_CHANNEL_IDS not set")
    try:
        import urllib.request
        url = "https://discord.com/api/v10/users/@me"
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bot {token}"}
        )
        def _fetch():
            with urllib.request.urlopen(req, timeout=8) as resp:
                return json.loads(resp.read())
        data, ms = _timed(_fetch)
        if data.get("id"):
            return SourceStatus(name="Discord", status="ok", latency_ms=ms,
                                detail=f"bot: {data.get('username', 'unknown')}")
        return SourceStatus(name="Discord", status="degraded", latency_ms=ms,
                            detail=f"unexpected: {str(data)[:60]}")
    except Exception as e:
        err = str(e)
        if "401" in err:
            return SourceStatus(name="Discord", status="down",
                                detail="invalid bot token (401)")
        return SourceStatus(name="Discord", status="down", detail=err[:120])


def _check_polymarket() -> SourceStatus:
    try:
        import urllib.request
        url = "https://gamma-api.polymarket.com/markets?limit=1"
        def _fetch():
            with urllib.request.urlopen(url, timeout=8) as resp:
                return json.loads(resp.read())
        data, ms = _timed(_fetch)
        if isinstance(data, list) and data:
            return SourceStatus(name="Polymarket", status="ok", latency_ms=ms)
        return SourceStatus(name="Polymarket", status="degraded", latency_ms=ms,
                            detail="empty response")
    except Exception as e:
        err = str(e)
        if "404" in err or "403" in err:
            return SourceStatus(name="Polymarket", status="degraded",
                                detail=f"API endpoint error: {err[:80]}")
        return SourceStatus(name="Polymarket", status="down", detail=err[:120])


# ─── Aggregate check ─────────────────────────────────────────────────────────

def check_all_sources(write_report: bool = True) -> list[SourceStatus]:
    """
    Run all source probes and return results. Each probe is independent.
    If write_report=True, writes reports/source_status.json.
    """
    results: list[SourceStatus] = []

    probes = [
        ("yfinance", _check_yfinance),
        ("sg_market", _check_sg_market_data),
        ("RSS", _check_rss),
        ("Finnhub", _check_finnhub),
        ("FMP", _check_fmp),
        ("AlphaVantage", _check_alpha_vantage),
        ("Reddit", _check_reddit),
        ("Discord", _check_discord),
        ("Polymarket", _check_polymarket),
    ]

    for label, probe_fn in probes:
        try:
            probe_result = probe_fn()
            if isinstance(probe_result, list):
                results.extend(probe_result)
            else:
                results.append(probe_result)
        except Exception as e:
            results.append(SourceStatus(
                name=label, status="down",
                detail=f"probe crashed: {e}"
            ))

    if write_report:
        try:
            _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            report = {
                "checked_at": datetime.now().isoformat(),
                "sources": [asdict(s) for s in results],
                "summary": {
                    "ok": sum(1 for s in results if s.status == "ok"),
                    "degraded": sum(1 for s in results if s.status == "degraded"),
                    "down": sum(1 for s in results if s.status == "down"),
                    "unconfigured": sum(1 for s in results if s.status == "unconfigured"),
                },
            }
            out = _REPORTS_DIR / "source_status.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info("Source health report written: %s", out)
        except Exception as e:
            logger.warning("Failed to write source_status.json: %s", e)

    return results


# ─── Footer renderer (for daily brief) ───────────────────────────────────────

_STATUS_ICONS = {"ok": "✓", "degraded": "~", "down": "✗", "unconfigured": "-"}

# Group names for compact footer display
_FOOTER_GROUPS = {
    "yfinance": ["yfinance/STI", "yfinance/SP500", "yfinance/SGDUSD"],
    "SG market": ["yfinance/DBS", "yfinance/OCBC", "yfinance/UOB", "yfinance/SG10Y"],
}


def render_status_footer(statuses: list[SourceStatus]) -> str:
    """
    Render a compact one-line status summary for daily brief footer.
    Groups yfinance tickers into one entry (worst status wins).
    """
    # Aggregate grouped items
    grouped: dict[str, Status] = {}

    for s in statuses:
        name = s.name
        # Map to group
        group = None
        for grp, members in _FOOTER_GROUPS.items():
            if name in members:
                group = grp
                break
        key = group if group else name

        current = grouped.get(key)
        if current is None:
            grouped[key] = s.status
        else:
            # Worst status wins: down > degraded > unconfigured > ok
            rank = {"down": 3, "degraded": 2, "unconfigured": 1, "ok": 0}
            if rank.get(s.status, 0) > rank.get(current, 0):
                grouped[key] = s.status

    parts = []
    for name, status in grouped.items():
        icon = _STATUS_ICONS.get(status, "?")
        parts.append(f"{name}{icon}")

    return "Data sources: " + " ".join(parts) if parts else ""


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    _app = _Path(__file__).resolve().parent
    if str(_app) not in sys.path:
        sys.path.insert(0, str(_app))

    from dotenv import load_dotenv
    load_dotenv(_Path(__file__).resolve().parent.parent / ".env")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    results = check_all_sources(write_report=True)

    print("\n=== Source Health Check ===")
    for s in results:
        icon = _STATUS_ICONS.get(s.status, "?")
        ms_str = f" ({s.latency_ms:.0f}ms)" if s.latency_ms else ""
        detail = f" — {s.detail}" if s.detail else ""
        print(f"  [{icon}] {s.name:<25} {s.status}{ms_str}{detail}")

    print()
    print(render_status_footer(results))
