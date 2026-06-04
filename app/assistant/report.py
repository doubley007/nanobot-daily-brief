"""
Report / Snapshot Mode — 市场快照生成器（v6 增强：executive style）。

支持两种风格：
  analyst   — 详细分析版（默认），包含所有信号字段
  executive — 简洁汇报版，一句话判断 + 关键数据

入口：
  Telegram:  /report gold           /snapshot gold executive
  CLI:       demo.py --report       demo.py --report --report-style executive
  Python:    from assistant.report import generate_report
             generate_report("gold", style="executive")
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def generate_report(
    asset: str,
    window_hours: int = 72,
    rag_store: Any = None,
    ds_store: Any = None,
    fetch_trend=None,
    style: str = "analyst",
) -> str:
    """
    Generate a market snapshot report for `asset`.

    style:
      "analyst"   — detailed (default): all signal fields, trend numbers
      "executive" — concise: one-liner judgment + key data, no raw numbers

    1. Try to load derived signal from cache (DerivedSignalStore, 3h TTL).
    2. If cache miss, compute fresh from RAG.
    3. Format per requested style.
    """
    from assistant.rag.derived_signals import (
        DerivedSignal, DerivedSignalStore,
        default_derived_store, compute_derived_signal,
    )
    from assistant.rag.store import default_store
    from assistant.rag.retriever import Retriever
    from assistant.sentiment_aggregator import aggregate
    from assistant.trend_signals import TrendSignal

    store = rag_store or default_store()
    ds = ds_store or default_derived_store()
    window_label = f"{window_hours // 24}d" if window_hours >= 24 else f"{window_hours}h"

    # Try cache first (extended TTL for report = 3h)
    sig: DerivedSignal | None = ds.get(asset, window=window_label, max_age_seconds=10800)
    if sig is None:
        # Compute fresh — may fail gracefully
        try:
            sig = compute_derived_signal(
                asset, window_hours=window_hours,
                store=store, ds_store=ds,
            )
        except Exception as e:
            logger.warning("report: compute_derived_signal failed for %s: %s", asset, e)

    # Fetch trend
    trend: TrendSignal | None = None
    _fetch_fn = fetch_trend
    if _fetch_fn is None:
        try:
            from assistant.trend_signals import fetch_trend_signal as _ft
            _fetch_fn = _ft
        except Exception:
            pass
    if _fetch_fn is not None:
        try:
            trend = _fetch_fn(asset)
        except Exception as e:
            logger.debug("report: trend unavailable: %s", e)

    # Fall back to community-only if no derived signal
    if sig is None:
        return _fallback_report(asset, store, window_hours, trend, style=style)

    if style == "executive":
        return _format_executive_report(asset, sig, trend, window_label)
    return _format_report(asset, sig, trend, window_label)


def _format_report(
    asset: str,
    sig: Any,
    trend: Any,
    window_label: str,
) -> str:
    """Format a derived signal into a readable snapshot."""
    from assistant.asset_taxonomy import get_asset
    spec = get_asset(asset)
    display = spec.display_name if spec else asset.upper()

    lines: list[str] = []
    ts_str = _format_ts(sig.computed_at)
    lines.append(f"📊 {display} ({asset.upper()}) market snapshot")
    lines.append(f"   Window: {window_label}   Updated: {ts_str}")
    lines.append("─" * 36)

    # Signal direction + strength
    direction_emoji = {"bullish": "📈", "bearish": "📉", "neutral": "➡️"}.get(
        sig.news_direction, "❓")
    lines.append(
        f"{direction_emoji} News signal: {sig.news_direction}"
        f" (strength {sig.news_strength:.2f})"
    )

    # Community bias
    bias_emoji = {"bullish": "🐂", "bearish": "🐻", "neutral": "😐", "mixed": "🔀"}.get(
        sig.community_bias, "❓")
    lines.append(
        f"{bias_emoji} Community bias: {sig.community_bias}"
        f"  bull {sig.bullish_ratio:.0%} / bear {sig.bearish_ratio:.0%}"
    )

    # FOMO + crowding
    fomo_flag = "⚠️ " if sig.fomo_ratio > 0.3 else ""
    crowding_emoji = {"low": "✅", "medium": "⚠️", "high": "🚨"}.get(
        sig.crowding_risk, "")
    lines.append(
        f"{crowding_emoji} Crowding risk: {sig.crowding_risk}"
        f"  {fomo_flag}FOMO {sig.fomo_ratio:.0%}"
    )

    # Entry quality + trend momentum
    quality_emoji = {"good": "✅", "medium": "🟡", "poor": "❌"}.get(
        sig.entry_quality, "")
    momentum_label = sig.trend_momentum
    if trend is not None:
        momentum_label = trend.momentum_label
    lines.append(
        f"{quality_emoji} Entry quality: {sig.entry_quality}"
        f"  Trend momentum: {momentum_label}"
    )

    # Trend numbers
    if trend is not None and (trend.recent_return_7d is not None or
                               trend.recent_return_30d is not None):
        r7 = f"{trend.recent_return_7d:+.1%}" if trend.recent_return_7d is not None else "n/a"
        r30 = f"{trend.recent_return_30d:+.1%}" if trend.recent_return_30d is not None else "n/a"
        lines.append(f"   7d={r7}  30d={r30}  overheating={trend.overheating_risk}")

    # Key narratives
    kws = sig.narrative_keywords[:6] if sig.narrative_keywords else []
    if kws:
        lines.append(f"🔑 Dominant narratives: {' · '.join(kws)}")

    # Summary line
    if sig.summary:
        lines.append(f"\n💬 {sig.summary}")

    # Data coverage
    lines.append(f"\n   Data coverage: {sig.news_count} news  /  {sig.post_count} community posts")

    return "\n".join(lines)


def _format_executive_report(
    asset: str,
    sig: Any,
    trend: Any,
    window_label: str,
) -> str:
    """
    Executive style: one-screen briefing, suitable for presenting to stakeholders.
    Structure:
      1. One-line verdict
      2. Community bias + crowding
      3. Key narratives (max 4)
      4. Suggested action tendency
      5. Why it matters
    """
    from assistant.asset_taxonomy import get_asset
    spec = get_asset(asset)
    display = spec.display_name if spec else asset.upper()
    ts_str = _format_ts(sig.computed_at)

    # 1. Verdict
    direction_map = {
        "bullish": "positive", "bearish": "negative", "neutral": "neutral",
    }
    direction_en = direction_map.get(sig.news_direction, sig.news_direction)
    quality_map = {"good": "good", "medium": "moderate", "poor": "poor"}
    quality_en = quality_map.get(sig.entry_quality, sig.entry_quality)
    verdict = (f"{display} news flow is {direction_en}, community sentiment {sig.community_bias}, "
               f"current entry quality {quality_en}.")

    # 2. Action tendency
    action_map = {
        ("bullish", "good"): "consider buying on dips",
        ("bullish", "medium"): "participate cautiously, control sizing",
        ("bullish", "poor"): "stay cautious, wait for a better entry",
        ("bearish", "poor"): "avoid or stay on the sidelines",
        ("bearish", "medium"): "lean cautious, keep sizing small",
        ("neutral", "medium"): "no clear direction — stay neutral",
    }
    key = (sig.news_direction, sig.entry_quality)
    action_tendency = action_map.get(key, "stay watchful; decide per your own risk tolerance")

    # 3. Why it matters
    crowding_note = {
        "high": "Crowding is elevated — watch for reversal risk",
        "medium": "Crowding is moderate — resist the chase impulse",
        "low": "Crowding is low — sentiment is relatively rational",
    }.get(sig.crowding_risk, "")
    fomo_note = f"FOMO is elevated ({sig.fomo_ratio:.0%}) — chase risk is real." if sig.fomo_ratio > 0.3 else ""

    # 4. Key narratives
    kws = sig.narrative_keywords[:4] if sig.narrative_keywords else []

    lines = [
        f"[{display} Executive Summary]  {ts_str}",
        "",
        f"▶ {verdict}",
        f"📌 Action tendency: {action_tendency}",
        "",
        f"Community: bull {sig.bullish_ratio:.0%} / bear {sig.bearish_ratio:.0%}  Crowding: {sig.crowding_risk}",
    ]
    if kws:
        lines.append(f"Dominant narratives: {' · '.join(kws)}")
    if crowding_note:
        lines.append(f"⚠ {crowding_note}")
    if fomo_note:
        lines.append(f"⚠ {fomo_note}")
    if trend is not None and trend.momentum_label not in ("unknown", None):
        lines.append(f"Trend momentum: {trend.momentum_label}  Overheating: {trend.overheating_risk}")
    lines.append("")
    lines.append(f"Data: {window_label} window  {sig.news_count} news  /  {sig.post_count} community posts")

    return "\n".join(lines)


def _fallback_report(
    asset: str,
    store: Any,
    window_hours: int,
    trend: Any,
    style: str = "analyst",
) -> str:
    """Minimal report from raw community data when derived signal unavailable."""
    from assistant.rag.retriever import Retriever
    from assistant.sentiment_aggregator import aggregate
    from assistant.asset_taxonomy import get_asset

    spec = get_asset(asset)
    display = spec.display_name if spec else asset.upper()

    retriever = Retriever(store=store)
    evidence = retriever.retrieve(asset=asset, window_hours=window_hours,
                                  top_k_news=8, top_k_community=200)
    agg = aggregate(evidence.community, asset=asset, window_hours=window_hours)
    kws = agg.narrative_keywords[:4] if agg.narrative_keywords else []
    window_label = f"{window_hours // 24}d" if window_hours >= 24 else f"{window_hours}h"

    if style == "executive":
        lines = [
            f"[{display} Executive Summary] (limited data)",
            "",
            f"▶ Community sentiment {agg.overall_bias}, bull {agg.bullish_ratio:.0%} / bear {agg.bearish_ratio:.0%}.",
            "📌 Action tendency: data is thin — stay on the sidelines",
            "",
            f"Crowding: {agg.crowded_trade_risk}  FOMO {agg.fomo_ratio:.0%}",
        ]
        if kws:
            lines.append(f"Dominant narratives: {' · '.join(kws)}")
        if trend is not None and trend.momentum_label not in ("unknown", None):
            lines.append(f"Trend momentum: {trend.momentum_label}  Overheating: {trend.overheating_risk}")
        lines.append("")
        lines.append(f"Data: {window_label} window  {len(evidence.news)} news  /  {len(evidence.community)} community posts")
        lines.append("(derived signal cache miss)")
        return "\n".join(lines)

    lines = [
        f"📊 {display} ({asset.upper()}) market snapshot (lite)",
        f"   Window: {window_hours}h",
        "─" * 36,
        f"Community bias: {agg.overall_bias}  bull {agg.bullish_ratio:.0%} / bear {agg.bearish_ratio:.0%}",
        f"Crowding: {agg.crowded_trade_risk}  FOMO {agg.fomo_ratio:.0%}",
    ]
    if trend is not None and trend.momentum_label != "unknown":
        lines.append(f"Trend momentum: {trend.momentum_label}  overheating={trend.overheating_risk}")
    if kws:
        lines.append(f"Dominant narratives: {' · '.join(kws)}")
    lines.append(f"\nData coverage: {len(evidence.news)} news  /  {len(evidence.community)} community posts")
    lines.append("(derived signal cache miss — showing raw community data only)")
    return "\n".join(lines)


def _format_ts(ts: float) -> str:
    import datetime
    try:
        dt = datetime.datetime.fromtimestamp(ts)
        return dt.strftime("%m-%d %H:%M")
    except Exception:
        return "unknown"


# ─── CLI entry for report ─────────────────────────────────────────────────────

def report_cli(asset: str, window_hours: int = 72) -> str:
    """Standalone CLI call — installs fixture if store is empty."""
    from assistant.rag.store import default_store
    store = default_store()
    counts = store.count()
    if counts["news"] == 0 and counts["community"] == 0:
        # Install fixture for demo
        if asset == "bitcoin":
            from assistant.fixtures import install_bitcoin_fixture
            install_bitcoin_fixture()
        else:
            from assistant.fixtures import install_gold_fixture
            install_gold_fixture()
    return generate_report(asset, window_hours=window_hours)
