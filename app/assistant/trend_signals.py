"""
价格趋势信号 —— 决策引擎的第 3 类信号源。

优先用 yfinance 取真实价格；取不到就返回占位值（不会让决策流程崩），
后续若有更好的数据源（FMP、AV），在 fetch_trend_signal() 里 fork 即可。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

from assistant.asset_taxonomy import get_asset

logger = logging.getLogger(__name__)


@dataclass
class TrendSignal:
    asset: str | None
    recent_return_7d: float | None
    recent_return_30d: float | None
    momentum_label: str           # "up" | "down" | "flat" | "unknown"
    overheating_risk: str         # "low" | "medium" | "high" | "unknown"
    data_source: str              # "yfinance" | "stub" | "manual"
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _classify_momentum(r7: float | None, r30: float | None) -> str:
    if r7 is None and r30 is None:
        return "unknown"
    r7_v = r7 or 0.0
    r30_v = r30 or 0.0
    if r7_v > 0.02 and r30_v > 0.0:
        return "up"
    if r7_v < -0.02 and r30_v < 0.0:
        return "down"
    return "flat"


def _classify_overheating(r7: float | None, r30: float | None) -> str:
    """
    粗粒度过热判断：短期涨幅显著强于中期 -> 偏热；且累计涨得多 -> 高风险。
    没数据就 unknown。
    """
    if r7 is None and r30 is None:
        return "unknown"
    r7_v = r7 or 0.0
    r30_v = r30 or 0.0
    if r7_v >= 0.05 and r30_v >= 0.12:
        return "high"
    if r7_v >= 0.03 or r30_v >= 0.08:
        return "medium"
    return "low"


# ─── yfinance 实现（可选） ───────────────────────────────────────────────────

def _fetch_via_yfinance(ticker: str) -> tuple[float | None, float | None]:
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return None, None

    try:
        hist = yf.Ticker(ticker).history(period="35d", auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None, None
        closes = hist["Close"].dropna().tolist()
        if len(closes) < 8:
            return None, None
        last = closes[-1]
        r7 = (last - closes[-8]) / closes[-8] if last else None
        r30 = (last - closes[0]) / closes[0] if len(closes) >= 30 else None
        return r7, r30
    except Exception as e:
        logger.info("trend_signals: yfinance fetch failed for %s: %s", ticker, e)
        return None, None


# ─── 对外入口 ────────────────────────────────────────────────────────────────

def fetch_trend_signal(
    asset: str | None,
    prefer_source: str = "yfinance",
) -> TrendSignal:
    """
    返回资产的短中期趋势信号。prefer_source 保留扩展位；当前只实现 yfinance
    和 stub。stub 用于测试和纯离线环境。
    """
    if not asset:
        return TrendSignal(
            asset=None, recent_return_7d=None, recent_return_30d=None,
            momentum_label="unknown", overheating_risk="unknown",
            data_source="stub", note="no asset",
        )

    spec = get_asset(asset)
    if not spec or not spec.tickers:
        return TrendSignal(
            asset=asset, recent_return_7d=None, recent_return_30d=None,
            momentum_label="unknown", overheating_risk="unknown",
            data_source="stub", note="no ticker mapped",
        )

    if prefer_source == "yfinance":
        for ticker in spec.tickers:
            r7, r30 = _fetch_via_yfinance(ticker)
            if r7 is not None or r30 is not None:
                return TrendSignal(
                    asset=asset,
                    recent_return_7d=round(r7, 4) if r7 is not None else None,
                    recent_return_30d=round(r30, 4) if r30 is not None else None,
                    momentum_label=_classify_momentum(r7, r30),
                    overheating_risk=_classify_overheating(r7, r30),
                    data_source="yfinance",
                    note=f"ticker={ticker}",
                )

    # 所有源都失败
    return TrendSignal(
        asset=asset, recent_return_7d=None, recent_return_30d=None,
        momentum_label="unknown", overheating_risk="unknown",
        data_source="stub", note="all sources unavailable",
    )


def get_current_price(
    asset: str | None,
) -> tuple[float | None, str]:
    """
    Return (current_price, status) for an asset.
    status: "live" | "fallback" | "missing"

    Reuses the yfinance last-close price already fetched for trend signals.
    Falls back gracefully — never raises.
    """
    if not asset:
        return None, "missing"
    spec = get_asset(asset)
    if not spec or not spec.tickers:
        return None, "missing"

    for ticker in spec.tickers:
        try:
            import yfinance as yf  # type: ignore
            hist = yf.Ticker(ticker).history(period="2d", auto_adjust=False)
            if hist is None or hist.empty or "Close" not in hist.columns:
                continue
            closes = hist["Close"].dropna().tolist()
            if closes:
                return float(closes[-1]), "live"
        except Exception as e:
            logger.debug("get_current_price: yfinance failed for %s: %s", ticker, e)

    return None, "fallback"


def trend_from_values(
    asset: str | None,
    r7: float | None = None,
    r30: float | None = None,
    note: str = "manual",
) -> TrendSignal:
    """测试/fixture 用：直接从给定数值构造 TrendSignal。"""
    return TrendSignal(
        asset=asset,
        recent_return_7d=r7,
        recent_return_30d=r30,
        momentum_label=_classify_momentum(r7, r30),
        overheating_risk=_classify_overheating(r7, r30),
        data_source="manual",
        note=note,
    )
