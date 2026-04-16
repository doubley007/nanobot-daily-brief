from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import yfinance as yf


@dataclass
class MarketSummary:
    us_equities: str
    rates: str
    asia_sg: str


def _format_pct_change(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def _safe_history(symbol: str, period: str = "5d"):
    """
    Fetch recent history safely.
    Returns a pandas DataFrame or None if unavailable.
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, auto_adjust=False)
        if hist is None or hist.empty:
            return None
        return hist
    except Exception:
        return None


def _get_last_two_closes(symbol: str) -> Optional[tuple[float, float]]:
    """
    Return (previous_close, latest_close) from recent history.
    """
    hist = _safe_history(symbol, period="5d")
    if hist is None or len(hist) < 2:
        return None

    closes = hist["Close"].dropna()
    if len(closes) < 2:
        return None

    previous_close = float(closes.iloc[-2])
    latest_close = float(closes.iloc[-1])
    return previous_close, latest_close


def _build_index_line(symbol: str, label: str) -> Optional[str]:
    values = _get_last_two_closes(symbol)
    if values is None:
        return None

    previous_close, latest_close = values
    if previous_close == 0:
        return None

    pct_change = (latest_close - previous_close) / previous_close * 100
    return f"{label} {_format_pct_change(pct_change)}"


def _build_us_equities_summary() -> str:
    """
    Build a simple summary for major US indices.
    """
    items = []

    spx = _build_index_line("^GSPC", "S&P 500")
    ndx = _build_index_line("^IXIC", "Nasdaq")
    dji = _build_index_line("^DJI", "Dow")

    if spx:
        items.append(spx)
    if ndx:
        items.append(ndx)
    if dji:
        items.append(dji)

    if not items:
        return "暂时无法获取美股主要指数数据"

    return "，".join(items)


def _build_rates_summary() -> str:
    """
    Use 10Y Treasury yield index on Yahoo Finance: ^TNX
    Yahoo Finance currently returns ^TNX directly in percentage terms,
    e.g. 4.29 means 4.29%.
    """
    values = _get_last_two_closes("^TNX")
    if values is None:
        return "暂时无法获取10Y Treasury yield数据"

    previous_close, latest_close = values
    pct_point_change = latest_close - previous_close
    latest_yield = latest_close

    if pct_point_change > 0:
        direction = "上行"
    elif pct_point_change < 0:
        direction = "下行"
    else:
        direction = "持平"

    return (
        f"10Y Treasury yield 报 {latest_yield:.2f}%，"
        f"较前一交易日{direction} {abs(pct_point_change):.2f} 个百分点"
    )


def _build_asia_sg_summary() -> str:
    """
    Use STI as Singapore market proxy.
    Yahoo symbol may vary by region/account. ^STI works in many cases.
    If unavailable, return a graceful fallback.
    """
    sti = _build_index_line("^STI", "STI")
    if sti:
        return sti

    return "暂时无法获取STI数据"


def get_market_summary() -> MarketSummary:
    """
    Main entry used by daily_job.py
    """
    return MarketSummary(
        us_equities=_build_us_equities_summary(),
        rates=_build_rates_summary(),
        asia_sg=_build_asia_sg_summary(),
    )


if __name__ == "__main__":
    summary = get_market_summary()
    print("US Equities:", summary.us_equities)
    print("Rates:", summary.rates)
    print("Asia/SG:", summary.asia_sg)