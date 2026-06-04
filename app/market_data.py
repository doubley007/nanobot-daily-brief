from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import yfinance as yf


@dataclass
class MarketSummary:
    us_equities: str
    rates: str
    asia_sg: str
    singapore_extended: str = ""   # SGD/USD, SG bonds, banks


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
        return "data unavailable for major US indices"

    return ", ".join(items)


def _build_rates_summary() -> str:
    """
    Use 10Y Treasury yield index on Yahoo Finance: ^TNX
    Yahoo Finance currently returns ^TNX directly in percentage terms,
    e.g. 4.29 means 4.29%.
    """
    values = _get_last_two_closes("^TNX")
    if values is None:
        return "data unavailable for 10Y Treasury yield"

    previous_close, latest_close = values
    pct_point_change = latest_close - previous_close
    latest_yield = latest_close

    if pct_point_change > 0:
        direction = "up"
    elif pct_point_change < 0:
        direction = "down"
    else:
        direction = "flat"

    return (
        f"10Y Treasury yield at {latest_yield:.2f}%, "
        f"{direction} {abs(pct_point_change):.2f} pp from the prior session"
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

    return "data unavailable for STI"


def _build_sgdusd_line() -> Optional[str]:
    """SGD/USD exchange rate via Yahoo Finance symbol SGDUSD=X."""
    values = _get_last_two_closes("SGDUSD=X")
    if values is None:
        return None
    prev, latest = values
    if prev == 0:
        return None
    pct = (latest - prev) / prev * 100
    sign = "+" if pct >= 0 else ""
    return f"SGD/USD {latest:.4f} ({sign}{pct:.3f}%)"


def _build_sg_bond_yield_line() -> Optional[str]:
    """
    Singapore 10Y government bond yield.
    Yahoo Finance symbol: SG10Y=X or ^SBND10Y — try both.
    If neither works, skip gracefully.
    """
    for symbol in ("SG10Y=X", "^SBND10Y"):
        values = _get_last_two_closes(symbol)
        if values is not None:
            prev, latest = values
            change = latest - prev
            direction = "↑" if change > 0 else ("↓" if change < 0 else "—")
            return f"SG 10Y Bond {latest:.2f}% ({direction}{abs(change):.2f}bp)"
    return None


def _build_sg_banks_summary() -> Optional[str]:
    """DBS, OCBC, UOB — Singapore banking trio (SGX symbols via Yahoo: D05.SI, O39.SI, U11.SI)."""
    bank_symbols = [
        ("D05.SI", "DBS"),
        ("O39.SI", "OCBC"),
        ("U11.SI", "UOB"),
    ]
    items = []
    for symbol, label in bank_symbols:
        line = _build_index_line(symbol, label)
        if line:
            items.append(line)
    if not items:
        return None
    return "SG Banks: " + ", ".join(items)


def _build_singapore_extended_summary() -> str:
    """
    Extended Singapore data block: SGD/USD + SG10Y bond + DBS/OCBC/UOB.
    Returns a formatted multi-line string or empty string if all sources fail.
    """
    parts: list[str] = []

    sgd = _build_sgdusd_line()
    if sgd:
        parts.append(sgd)

    sg_bond = _build_sg_bond_yield_line()
    if sg_bond:
        parts.append(sg_bond)

    banks = _build_sg_banks_summary()
    if banks:
        parts.append(banks)

    if not parts:
        return ""
    return "; ".join(parts)


def get_market_summary() -> MarketSummary:
    """
    Main entry used by daily_job.py
    """
    return MarketSummary(
        us_equities=_build_us_equities_summary(),
        rates=_build_rates_summary(),
        asia_sg=_build_asia_sg_summary(),
        singapore_extended=_build_singapore_extended_summary(),
    )


if __name__ == "__main__":
    summary = get_market_summary()
    print("US Equities:", summary.us_equities)
    print("Rates:", summary.rates)
    print("Asia/SG:", summary.asia_sg)