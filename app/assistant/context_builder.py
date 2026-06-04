"""
Context Builder —— 统一 context 组装层。

在用户消息进入 LLM 之前，把所有上下文拼成一个结构化对象：

  ContextPackage:
    company_context     公司语境
    user_profile        用户画像
    route               路由结果（asset, route, confidence）
    user_emotion        用户情绪
    retrieved_news      检索到的新闻
    retrieved_community 检索到的社区内容
    derived_signal      派生信号（可选，从缓存取）
    trend_signal        趋势信号

外部调用只需要：
    pkg = build_context(text, user_id=..., llm_callable=...)
    # 然后把 pkg 传给 reply 链路
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from assistant.company_context import CompanyContext, get_company_context
from assistant.user_profile import UserProfile, get_user_profile
from assistant.query_router import RouterResult, route_query
from assistant.user_emotion import UserEmotionProfile, analyze_user_emotion
from assistant.rag.retriever import RetrievedEvidence, Retriever
from assistant.rag.store import NewsDoc, CommunityDoc
from assistant.trend_signals import TrendSignal, fetch_trend_signal
from assistant.holdings import Holding, build_holdings_context_block, default_holdings_store
from assistant.rag.retriever import RetrievedEvidence

logger = logging.getLogger(__name__)


# ─── 数据类 ──────────────────────────────────────────────────────────────────

@dataclass
class ContextPackage:
    """所有信号的聚合快照，传给 decision / compose 层使用。"""

    # ── 身份层 ───────────────────────────────────────────────────────────────
    company: CompanyContext
    profile: UserProfile

    # ── 意图层 ───────────────────────────────────────────────────────────────
    route: RouterResult
    user_emotion: UserEmotionProfile

    # ── 证据层 ───────────────────────────────────────────────────────────────
    news: list[NewsDoc] = field(default_factory=list)
    community: list[CommunityDoc] = field(default_factory=list)
    trend: TrendSignal | None = None

    # ── 派生层 ───────────────────────────────────────────────────────────────
    derived_signal: Any | None = None   # DerivedSignal | None

    # ── 持仓层 ───────────────────────────────────────────────────────────────
    holding: Any | None = None          # Holding | None (for the queried asset)

    # ── 检索证据（v5：含 index_status） ──────────────────────────────────────
    evidence: Any | None = None         # RetrievedEvidence | None

    # ── 调试元信息 ────────────────────────────────────────────────────────────
    build_time_ms: float = 0.0
    window_hours: int = 72

    @property
    def asset(self) -> str | None:
        return self.route.asset

    @property
    def route_name(self) -> str:
        return self.route.route

    def to_debug_dict(self) -> dict[str, Any]:
        """debug / trace 输出格式。"""
        return {
            "route": self.route.route,
            "asset": self.route.asset,
            "confidence": self.route.confidence,
            "user_emotion": self.user_emotion.primary_emotion,
            "emotion_intensity": self.user_emotion.emotion_intensity,
            "profile": {
                "role": self.profile.role,
                "style": self.profile.preferred_style,
                "is_internal": self.profile.is_internal,
            },
            "company_context_injected": True,
            "retrieved_news_count": len(self.news),
            "retrieved_community_count": len(self.community),
            "derived_signal": (
                self.derived_signal.to_dict() if self.derived_signal else None
            ),
            "trend": self.trend.to_dict() if self.trend else None,
            "holding": self.holding.to_dict() if self.holding else None,
            "build_time_ms": round(self.build_time_ms, 1),
        }

    def to_prompt_block(self) -> str:
        """
        把整个 ContextPackage 序列化成可注入 LLM prompt 的文本块。
        只包含需要模型知道的内容，不暴露敏感用户信息。
        """
        parts: list[str] = []

        parts.append(self.company.to_system_block())
        parts.append(self.profile.to_context_block())

        if self.holding is not None:
            parts.append(self.holding.to_context_block())

        if self.derived_signal:
            parts.append(self.derived_signal.to_context_block())
        elif self.news or self.community:
            # 没有派生信号缓存，用原始证据摘要
            parts.append(self._build_evidence_block())

        if self.trend:
            parts.append(self._build_trend_block())

        return "\n\n".join(p for p in parts if p)

    def _build_evidence_block(self) -> str:
        lines = []
        if self.news:
            asset_label = self.asset or "market"
            lines.append(f"[{asset_label} recent news — {len(self.news)} items]")
            for n in self.news[:4]:
                lines.append(f"  • {n.title} ({n.source}, {n.sentiment})")
        if self.community:
            bullish = sum(1 for c in self.community if c.bullish_bearish_label == "bullish")
            bearish = sum(1 for c in self.community if c.bullish_bearish_label == "bearish")
            total = len(self.community)
            lines.append(
                f"[Community sentiment {total} posts] bull {bullish/total:.0%} / bear {bearish/total:.0%}"
            )
        return "\n".join(lines)

    def _build_trend_block(self) -> str:
        t = self.trend
        if t is None:
            return ""
        r7 = f"{t.recent_return_7d:+.1%}" if t.recent_return_7d is not None else "n/a"
        r30 = f"{t.recent_return_30d:+.1%}" if t.recent_return_30d is not None else "n/a"
        base = (
            f"[Trend signal] momentum={t.momentum_label}, overheating={t.overheating_risk}, "
            f"7d={r7}, 30d={r30}"
        )
        if t.note and ("earnings" in t.note.lower() or "财报" in t.note):
            base += f"\n{t.note}"
        return base


# ─── Earnings calendar helper ────────────────────────────────────────────────

def _inject_earnings_note(asset: str, trend: "TrendSignal | None") -> None:
    """
    Fetch next earnings date via yfinance and append to trend.note if within 7 days.
    Only runs for equity assets; silently skips commodities/crypto/fx/indices.
    Mutates trend.note in-place (non-critical — never raises to caller).
    """
    if trend is None:
        return
    from assistant.asset_taxonomy import get_asset
    spec = get_asset(asset)
    if not spec or spec.category not in ("equity",):
        return
    if not spec.tickers:
        return

    import datetime
    try:
        import yfinance as yf
        ticker = yf.Ticker(spec.tickers[0])
        cal = ticker.calendar  # dict with 'Earnings Date' key
        if cal is None:
            return
        # calendar can be a dict or DataFrame depending on yfinance version
        if hasattr(cal, "to_dict"):
            cal = cal.to_dict()
        earnings_dates = cal.get("Earnings Date") or cal.get("earningsDate")
        if not earnings_dates:
            return
        if not isinstance(earnings_dates, (list, tuple)):
            earnings_dates = [earnings_dates]
        today = datetime.date.today()
        for ed in earnings_dates:
            # yfinance returns datetime.date or datetime.datetime depending on version
            if hasattr(ed, "to_pydatetime"):
                ed = ed.to_pydatetime().date()
            if isinstance(ed, datetime.datetime):
                ed = ed.date()
            if not isinstance(ed, datetime.date):
                continue
            days_away = (ed - today).days
            if 0 <= days_away <= 7:
                note = trend.note or ""
                earnings_note = f"⚠️ Earnings in {days_away} days ({ed.strftime('%Y-%m-%d')})"
                trend.note = (note + "; " + earnings_note).lstrip("; ")
                break
    except Exception as e:
        logger.debug("Earnings fetch failed for %s: %s", asset, e)


# ─── Builder ──────────────────────────────────────────────────────────────────

def build_context(
    text: str,
    user_id: str | int | None = None,
    llm_callable: Callable[[str], str] | None = None,
    window_hours: int = 72,
    top_k_news: int = 8,
    top_k_community: int = 100,
    use_derived_cache: bool = True,
    rag_store: Any = None,
    holdings_store: Any = None,
) -> ContextPackage:
    """
    一步到位：拿公司语境 + 用户画像 + 路由 + 情绪 + RAG + 趋势。
    返回 ContextPackage，给 pipeline 用。
    """
    t0 = time.time()

    company = get_company_context()
    profile = get_user_profile(user_id)

    route = route_query(text, llm_callable=llm_callable)
    emotion = analyze_user_emotion(text, llm_callable=llm_callable)

    asset = route.asset
    news: list[NewsDoc] = []
    community: list[CommunityDoc] = []
    trend: TrendSignal | None = None
    derived = None
    holding = None
    ev: Any = None

    if route.route in ("market_decision", "market_summary") and asset:
        # 尝试从缓存取 derived signal（1 小时 TTL）
        if use_derived_cache:
            try:
                from assistant.rag.derived_signals import default_derived_store
                ds_store = default_derived_store()
                window_label = f"{window_hours // 24}d" if window_hours >= 24 else f"{window_hours}h"
                derived = ds_store.get(asset, window=window_label, max_age_seconds=3600)
            except Exception as e:
                logger.debug("Derived signal cache miss: %s", e)

        # 原始层 RAG（无论有没有 derived，都要拉证据给 decision_engine 用）
        from assistant.rag.store import default_store
        store = rag_store or default_store()
        retriever = Retriever(store=store)
        ev: RetrievedEvidence = retriever.retrieve(
            asset=asset,
            window_hours=window_hours,
            top_k_news=top_k_news,
            top_k_community=top_k_community,
            query_text=text,  # enables TF-IDF vector scoring
        )
        news = ev.news
        community = ev.community

        # 趋势
        try:
            trend = fetch_trend_signal(asset)
        except Exception as e:
            logger.debug("Trend signal unavailable: %s", e)

        # 持仓
        if user_id is not None:
            try:
                hs = holdings_store or default_holdings_store()
                holding = hs.get(user_id, asset)
            except Exception as e:
                logger.debug("Holdings lookup failed: %s", e)

        # 财报日历（非阻塞，只对股票类资产尝试）
        try:
            _inject_earnings_note(asset, trend)
        except Exception as e:
            logger.debug("Earnings calendar unavailable: %s", e)

    build_ms = (time.time() - t0) * 1000
    _evidence = ev
    return ContextPackage(
        company=company,
        profile=profile,
        route=route,
        user_emotion=emotion,
        news=news,
        community=community,
        trend=trend,
        derived_signal=derived,
        holding=holding,
        evidence=_evidence,
        build_time_ms=build_ms,
        window_hours=window_hours,
    )
