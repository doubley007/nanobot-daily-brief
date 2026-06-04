"""
MCP stdio server: 把 assistant.pipeline 包装成 nanobot 可调的工具 `market_rag`。

启动方式（由 nanobot 自动通过 config.json 拉起）：
    PYTHONPATH=/Users/yangyang/nanobot_project/app \
    /Users/yangyang/nanobot_project/.venv/bin/python -m assistant.rag_mcp_server

返回格式（全字段）：
    {
      "question": str,
      "asset": str | null,
      "route": str | null,
      "router_confidence": float | null,
      "direct_answer": str | null,   # 事实型问题的一句直接答案
      "final_answer": str,           # 完整 in-house 风格答案（thesis+risks）
      "citations": [{title, source, url, published}],
      "market_data_used": {
        "current_price": float | null,
        "price_status": str,
        "return_7d": float | null,
        "return_30d": float | null,
        "momentum_label": str | null,
        "overheating_risk": str | null,
      },
      "news_used": [{title, source, url, published, snippet}],
      "community_signals": {
        "overall_bias": str | null,
        "bullish_ratio": float,
        "bearish_ratio": float,
        "fomo_ratio": float,
        "crowded_trade_risk": str | null,
        "sample_size": int | null,
      },
      "decision": {
        "action": str,
        "confidence": str,
        "thesis": str,
        "risks": [str],
        "one_line_advice": str,
        "scores": {...} | null,
      } | null,
      "confidence": str,             # overall answer confidence: "high"|"medium"|"low"|"insufficient"
      "risks": [str],
      "singapore_lens": str | null,  # Singapore/insurance specific note if applicable
      "data_quality": {
        "news_count": int,
        "community_count": int,
        "derived_signal_status": str,
        "index_status": str,
        "warning": str | null,       # "当前证据不足" when data is sparse
      }
    }
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types


logger = logging.getLogger("assistant.rag_mcp")

_SG_KEYWORDS = frozenset({
    "mas", "singapore", "sgd", "sti", "dbs", "ocbc", "uob",
    "great eastern", "insurance", "insurer", "asean", "sg bond",
})


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _summarize_news(news_docs: list[Any], limit: int = 5) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for doc in (news_docs or [])[:limit]:
        if not hasattr(doc, "title"):
            continue
        out.append({
            "title": getattr(doc, "title", "") or "",
            "source": getattr(doc, "source", "") or "",
            "url": getattr(doc, "url", "") or "",
            "published": str(getattr(doc, "published_at", "") or ""),
            "snippet": (getattr(doc, "text", "") or "")[:300],
        })
    return out


def _summarize_community(community_docs: list[Any], limit: int = 5) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for doc in (community_docs or [])[:limit]:
        if not hasattr(doc, "text"):
            continue
        out.append({
            "source": getattr(doc, "source", "") or "",
            "author": getattr(doc, "author", "") or "",
            "text": (getattr(doc, "text", "") or "")[:240],
            "sentiment": getattr(doc, "sentiment", "") or "",
        })
    return out


_FACT_KEYWORDS = (
    "涨", "跌", "价格", "多少钱", "报价", "现在", "最近", "今天",
    "rising", "falling", "price", "quote", "current", "now", "today", "lately",
    "recent", "how much", "how is",
)


def _asset_display_name(asset: str | None) -> str:
    if not asset:
        return "the asset"
    try:
        from assistant.asset_taxonomy import asset_display
        return asset_display(asset) or asset
    except Exception:
        return asset


def _build_direct_answer(
    question: str,
    asset: str | None,
    trend_block: dict[str, Any] | None,
) -> str | None:
    """针对"涨没涨/多少钱"这类事实型问题，基于 7d 收益拼一句直接答案。"""
    if not trend_block:
        return None
    q_lower = question.lower()
    is_fact_question = any(kw in question or kw in q_lower for kw in _FACT_KEYWORDS)
    if not is_fact_question:
        return None

    r7 = trend_block.get("return_7d")
    r30 = trend_block.get("return_30d")
    price = trend_block.get("current_price")
    display = _asset_display_name(asset)

    if r7 is None and r30 is None and price is None:
        return None

    pieces: list[str] = []
    if r7 is not None:
        pct7 = r7 * 100
        if r7 > 0.01:
            verdict = "is up"
        elif r7 < -0.01:
            verdict = "is down"
        else:
            verdict = "is roughly flat"
        pieces.append(f"{display} {verdict} over the last 7 days ({pct7:+.2f}%)")
        if r30 is not None:
            pieces.append(f"last 30 days {r30 * 100:+.2f}%")
    elif r30 is not None:
        pct30 = r30 * 100
        verdict = "up" if r30 > 0.01 else ("down" if r30 < -0.01 else "roughly flat")
        pieces.append(f"{display} {verdict} over the last 30 days ({pct30:+.2f}%)")

    if price is not None:
        pieces.append(f"last price {price:,.2f}")

    return "; ".join(pieces) + "." if pieces else None


def _detect_singapore_lens(question: str, asset: str | None) -> str | None:
    """
    If the question or asset is SG-related, return a one-line Singapore/insurance lens note.
    Returns None when not applicable.
    """
    combined = f"{question} {asset or ''}".lower()
    if not any(kw in combined for kw in _SG_KEYWORDS):
        return None

    notes: list[str] = []
    if any(k in combined for k in ("dbs", "ocbc", "uob")):
        notes.append("Touches SG local banks — watch NIM trend, spread compression, and MAS prudential cues")
    if any(k in combined for k in ("sti",)):
        notes.append("STI directly affects SG insurers' equity book")
    if any(k in combined for k in ("sgd", "mas", "singapore dollar")):
        notes.append("SGD moves feed through to cross-border reinvestment yields and FX exposure")
    if any(k in combined for k in ("insurance", "insurer", "great eastern")):
        notes.append("Insurance lens: solvency ratio, rate sensitivity, and liability-duration matching")
    if not notes:
        notes.append("SG / regional lens: watch MAS policy direction and ASEAN macro transmission")
    return "; ".join(notes)


def _overall_confidence(
    decision_conf: str | None,
    news_count: int,
    community_count: int,
    has_trend: bool,
) -> str:
    """Derive an overall answer confidence level from available evidence."""
    if news_count == 0 and community_count == 0 and not has_trend:
        return "insufficient"
    if decision_conf == "high" and (news_count >= 3 or has_trend):
        return "high"
    if decision_conf in ("medium", "high") and (news_count >= 1 or has_trend):
        return "medium"
    if news_count == 0 and not has_trend:
        return "low"
    return "medium"


def _build_payload(trace: Any, question: str) -> dict[str, Any]:
    """把 PipelineTrace 压成一份 LLM 能直接用的结构化证据包。"""
    route = trace.route
    pkg = trace.context_pkg

    # ── basic routing info ──────────────────────────────────────────────────
    asset = getattr(route, "asset", None)
    payload: dict[str, Any] = {
        "question": question,
        "asset": asset,
        "route": getattr(route, "route", None),
        "router_confidence": getattr(route, "confidence", None),
        "final_answer": trace.reply,
    }

    # ── community sentiment ─────────────────────────────────────────────────
    community_signals: dict[str, Any] = {
        "overall_bias": None,
        "bullish_ratio": 0.0,
        "bearish_ratio": 0.0,
        "fomo_ratio": 0.0,
        "crowded_trade_risk": None,
        "sample_size": None,
    }
    if trace.aggregate is not None:
        agg = trace.aggregate
        community_signals = {
            "overall_bias": getattr(agg, "overall_bias", None),
            "bullish_ratio": round(getattr(agg, "bullish_ratio", 0.0) or 0.0, 3),
            "bearish_ratio": round(getattr(agg, "bearish_ratio", 0.0) or 0.0, 3),
            "fomo_ratio": round(getattr(agg, "fomo_ratio", 0.0) or 0.0, 3),
            "crowded_trade_risk": getattr(agg, "crowded_trade_risk", None),
            "sample_size": getattr(agg, "sample_size", None),
        }
    payload["community_signals"] = community_signals

    # ── trend / market data ──────────────────────────────────────────────────
    market_data: dict[str, Any] = {
        "current_price": None,
        "price_status": "missing",
        "return_7d": None,
        "return_30d": None,
        "momentum_label": None,
        "overheating_risk": None,
        "data_source": None,
        "note": None,
        "earnings_alert": None,
    }
    trend_block: dict[str, Any] | None = None
    if trace.trend is not None:
        tr = trace.trend
        note = getattr(tr, "note", "") or ""
        trend_block = {
            "return_7d": getattr(tr, "recent_return_7d", None),
            "return_30d": getattr(tr, "recent_return_30d", None),
            "momentum_label": getattr(tr, "momentum_label", None),
            "overheating_risk": getattr(tr, "overheating_risk", None),
            "data_source": getattr(tr, "data_source", None),
            "note": note,
            "earnings_alert": note if "earnings" in note.lower() else None,
        }
        # Live spot price
        try:
            from assistant.trend_signals import get_current_price
            price, price_status = get_current_price(asset)
            trend_block["current_price"] = price
            trend_block["price_status"] = price_status
        except Exception:
            trend_block["price_status"] = "error"
        market_data = trend_block
    payload["market_data_used"] = market_data

    # ── decision ────────────────────────────────────────────────────────────
    decision_conf: str | None = None
    risks: list[str] = []
    if trace.decision is not None:
        d = trace.decision
        decision_conf = d.confidence
        risks = list(d.risks or [])
        dd: dict[str, Any] = {
            "action": d.action,
            "confidence": d.confidence,
            "thesis": d.thesis,
            "risks": risks,
            "one_line_advice": d.one_line_advice,
        }
        if d.scores is not None:
            dd["scores"] = {
                "direction_score": round(d.scores.direction_score, 3),
                "crowding_score": round(d.scores.crowding_score, 3),
                "entry_quality": d.scores.entry_quality,
                "chasing_risk": d.scores.chasing_risk,
            }
        payload["decision"] = dd
    else:
        payload["decision"] = None

    # ── news / citations ─────────────────────────────────────────────────────
    news_list: list[Any] = []
    community_list: list[Any] = []
    if pkg is not None:
        news_list = pkg.news or []
        community_list = pkg.community or []

        # Holdings context
        if pkg.holding is not None:
            h = pkg.holding
            price = market_data.get("current_price")
            payload["holding"] = {
                "asset": h.asset,
                "position_size": h.position_size,
                "avg_cost": h.avg_cost,
                "horizon": h.horizon,
                "pnl_status": h.pnl_status(price),
                "context_block": h.to_context_block(price),
            }

    news_used = _summarize_news(news_list, limit=5)
    payload["news_used"] = news_used
    payload["community_top"] = _summarize_community(community_list, limit=5)

    # Citations: deduplicate news by url
    citations: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for item in news_used:
        url = item.get("url") or ""
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        citations.append({
            "title": item.get("title", ""),
            "source": item.get("source", ""),
            "url": url,
            "published": item.get("published", ""),
        })
    payload["citations"] = citations

    # ── overall confidence & data quality warning ────────────────────────────
    news_count = len(news_list)
    community_count = len(community_list)
    has_trend = trace.trend is not None

    overall_conf = _overall_confidence(decision_conf, news_count, community_count, has_trend)
    payload["confidence"] = overall_conf

    # When evidence is truly insufficient, override final_answer with explicit notice
    if overall_conf == "insufficient":
        payload["final_answer"] = (
            "Evidence is insufficient — not enough to give a firm call. "
            "No relevant news, community discussion, or price-trend data was found. "
            "Wait for more data before judging."
        )

    warning: str | None = None
    if overall_conf == "insufficient":
        warning = "Evidence insufficient: no news, no community data, no trend signal — wait for more data."
    elif news_count == 0 and not has_trend:
        warning = "News and price data both missing — the call rests on community sentiment alone; confidence is low."
    elif news_count < 2:
        warning = "News sources are sparse — reference value of the call is limited."
    payload["data_quality"] = {
        "news_count": news_count,
        "community_count": community_count,
        "derived_signal_status": trace.meta.get("derived_signal_status", "unknown") if trace.meta else "unknown",
        "index_status": trace.meta.get("index_status", "none") if trace.meta else "none",
        "warning": warning,
    }

    payload["risks"] = risks

    # ── evidence_sources (real sources only, no fabrication) ─────────────────
    evidence_sources: list[dict[str, str]] = []
    for item in news_used:
        if item.get("title"):
            evidence_sources.append({
                "type": "news",
                "name": item.get("title", "")[:80],
                "source": item.get("source", ""),
                "url": item.get("url", ""),
            })
    if community_count > 0:
        evidence_sources.append({
            "type": "community",
            "name": f"{community_count} community posts",
            "source": "reddit/discord",
            "url": "",
        })
    if has_trend and trend_block:
        ds = trend_block.get("data_source") or "yfinance"
        evidence_sources.append({
            "type": "trend",
            "name": f"price trend ({ds})",
            "source": ds,
            "url": "",
        })
    payload["evidence_sources"] = evidence_sources

    # ── Singapore / insurance lens ──────────────────────────────────────────
    payload["singapore_lens"] = _detect_singapore_lens(question, asset)

    # ── direct factual answer (if question asks for price/change fact) ───────
    direct = _build_direct_answer(question, asset, trend_block)
    payload["direct_answer"] = direct

    # ── meta ──────────────────────────────────────────────────────────────────
    payload["meta"] = {
        "window_hours": trace.meta.get("window_hours") if trace.meta else None,
        "derived_signal_status": trace.meta.get("derived_signal_status") if trace.meta else None,
        "index_status": trace.meta.get("index_status") if trace.meta else None,
    }

    return _to_jsonable(payload)


# ─── MCP server setup ────────────────────────────────────────────────────────

server = Server("market-rag")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="market_rag",
            description=(
                "In-house market RAG and sentiment pipeline for insurance investment context "
                "(Great Eastern SG / Singapore focus). "
                "Given a user question about an asset, market, or macro topic, returns a "
                "structured JSON evidence package including: "
                "`direct_answer` (a factual one-liner for price/change questions), "
                "`final_answer` (full in-house reply with thesis + risks), "
                "`citations` (news sources used), "
                "`market_data_used` (price, 7d/30d return, momentum), "
                "`news_used` (top relevant news), "
                "`community_signals` (bullish/bearish/FOMO ratios, crowded-trade risk), "
                "`decision` (action + confidence + thesis + risks + scores), "
                "`confidence` (overall evidence quality: high/medium/low/insufficient), "
                "`risks` (key risk factors), "
                "`singapore_lens` (Singapore/insurance specific note when applicable), "
                "`data_quality.warning` (set to 'Evidence insufficient' when data is sparse), "
                "`evidence_sources` (list of {type, name, source, url} for actual sources used — "
                "news/community/trend only from real retrieved data). "
                "USAGE RULES: "
                "(1) For factual questions (price/change): lead reply with `direct_answer` if set. "
                "(2) For advisory questions: output conclusion + thesis + risks from `decision`. "
                "(3) For report-style questions: use `final_answer` as executive summary. "
                "(4) NEVER fabricate prices, returns, or news. If `confidence` == 'insufficient', "
                "explicitly tell the user data is unavailable. "
                "(5) Always call this tool FIRST before web_search for market/asset questions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user's original market question, verbatim.",
                    },
                    "user_id": {
                        "type": ["integer", "string", "null"],
                        "description": "Telegram user id if available, for holdings/profile lookup.",
                    },
                },
                "required": ["question"],
            },
        )
    ]


_SLIM_NEWS_LIMIT = 3
_SLIM_SNIPPET_CHARS = 140
_SLIM_THESIS_CHARS = 400


def _slim_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Trim the full payload down to the fields a small-context LLM actually needs.

    Why: the full payload routinely exceeds 10k tokens, which on an 8k-context model
    causes history snipping to drop the user's original question and produce empty
    or off-topic replies. The slim view keeps the load-bearing answer fields and
    drops large structures (full news/community text, raw scores, meta blocks).
    """
    if not isinstance(payload, dict):
        return payload

    slim: dict[str, Any] = {
        "question": payload.get("question"),
        "asset": payload.get("asset"),
        "direct_answer": payload.get("direct_answer"),
        "confidence": payload.get("confidence"),
    }

    final_answer = payload.get("final_answer")
    if isinstance(final_answer, str) and len(final_answer) > _SLIM_THESIS_CHARS * 2:
        slim["final_answer"] = final_answer[: _SLIM_THESIS_CHARS * 2] + "…"
    else:
        slim["final_answer"] = final_answer

    decision = payload.get("decision")
    if isinstance(decision, dict):
        thesis = decision.get("thesis") or ""
        if isinstance(thesis, str) and len(thesis) > _SLIM_THESIS_CHARS:
            thesis = thesis[:_SLIM_THESIS_CHARS] + "…"
        slim["decision"] = {
            "action": decision.get("action"),
            "confidence": decision.get("confidence"),
            "thesis": thesis,
            "risks": (decision.get("risks") or [])[:3],
            "one_line_advice": decision.get("one_line_advice"),
        }
    else:
        slim["decision"] = None

    market = payload.get("market_data_used") or {}
    if isinstance(market, dict):
        slim["market_data_used"] = {
            "current_price": market.get("current_price"),
            "price_status": market.get("price_status"),
            "return_7d": market.get("return_7d"),
            "return_30d": market.get("return_30d"),
            "momentum_label": market.get("momentum_label"),
            "overheating_risk": market.get("overheating_risk"),
        }

    community = payload.get("community_signals") or {}
    if isinstance(community, dict):
        slim["community_signals"] = {
            "overall_bias": community.get("overall_bias"),
            "bullish_ratio": community.get("bullish_ratio"),
            "bearish_ratio": community.get("bearish_ratio"),
            "sample_size": community.get("sample_size"),
            "crowded_trade_risk": community.get("crowded_trade_risk"),
        }

    news_used = payload.get("news_used") or []
    slim_news: list[dict[str, Any]] = []
    for item in news_used[:_SLIM_NEWS_LIMIT]:
        if not isinstance(item, dict):
            continue
        snippet = item.get("snippet") or ""
        if isinstance(snippet, str) and len(snippet) > _SLIM_SNIPPET_CHARS:
            snippet = snippet[:_SLIM_SNIPPET_CHARS] + "…"
        slim_news.append({
            "title": item.get("title", ""),
            "source": item.get("source", ""),
            "snippet": snippet,
        })
    slim["news_used"] = slim_news

    holding = payload.get("holding")
    if isinstance(holding, dict):
        slim["holding"] = {
            "asset": holding.get("asset"),
            "position_size": holding.get("position_size"),
            "pnl_status": holding.get("pnl_status"),
        }

    sg_lens = payload.get("singapore_lens")
    if sg_lens:
        slim["singapore_lens"] = sg_lens

    risks = payload.get("risks")
    if isinstance(risks, list) and risks:
        slim["risks"] = risks[:3]

    dq = payload.get("data_quality") or {}
    if isinstance(dq, dict):
        warning = dq.get("warning")
        if warning:
            slim["data_quality_warning"] = warning

    if payload.get("error"):
        slim["error"] = payload["error"]

    return slim


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    if name != "market_rag":
        return [types.TextContent(type="text", text=json.dumps({"error": f"unknown tool {name}"}))]

    question = (arguments or {}).get("question") or ""
    user_id = (arguments or {}).get("user_id")
    if not question.strip():
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": "question is required"}, ensure_ascii=False),
        )]

    def _run() -> dict[str, Any]:
        from assistant.pipeline import answer_question_traced
        trace = answer_question_traced(str(question), user_id=user_id)
        return _build_payload(trace, question)

    try:
        payload = await asyncio.to_thread(_run)
    except Exception as e:
        logger.exception("market_rag failed")
        payload = {
            "error": f"{type(e).__name__}: {e}",
            "question": question,
            "direct_answer": None,
            "final_answer": "System can't process this question right now — please retry shortly.",
            "confidence": "insufficient",
            "data_quality": {"warning": "Evidence insufficient: pipeline error."},
        }

    slim = _slim_payload(payload)
    return [types.TextContent(
        type="text",
        text=json.dumps(slim, ensure_ascii=False, default=str),
    )]


async def _amain() -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    logging.basicConfig(
        level=os.getenv("ASSISTANT_MCP_LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,  # stdout belongs to MCP protocol
    )
    app_dir = Path(__file__).resolve().parent.parent
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
