"""
Telegram long-poll 入口。

升级版：
  - 传入 user_id 到 pipeline（用于 user profile 查询）
  - ASSISTANT_DEBUG=1 时在回复后追加 debug 摘要（仅限内部用户）
  - 支持 /debug 命令临时开启 trace 输出
  - 支持 /profile 命令查看当前用户画像

怎么用：
    cd /Users/yangyang/nanobot_project
    PYTHONPATH=app python -m assistant.telegram_bot
"""
from __future__ import annotations

import logging
import os
import signal
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from assistant.pipeline import answer_question, answer_question_traced
from telegram_sender import _split_message, send_to_telegram

logger = logging.getLogger("assistant.telegram_bot")


POLL_TIMEOUT = 25
POLL_INTERVAL_ON_ERROR = 10

# 运行时临时 debug 开关（/debug 命令切换）
_DEBUG_USERS: set[int] = set()


# ─── Telegram API helpers ────────────────────────────────────────────────────

def _api_url(method: str) -> str:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in .env")
    return f"https://api.telegram.org/bot{token}/{method}"


def _get_updates(offset: int | None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"timeout": POLL_TIMEOUT}
    if offset is not None:
        params["offset"] = offset
    resp = requests.get(_api_url("getUpdates"), params=params,
                        timeout=POLL_TIMEOUT + 5)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"getUpdates failed: {data}")
    return data.get("result", [])


def _send_reply(chat_id: int | str, text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
    url = _api_url("sendMessage")
    for chunk in _split_message(text, max_len=4000):
        payload = {"chat_id": chat_id, "text": chunk}
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                r = requests.post(url, json=payload, timeout=20)
                if 400 <= r.status_code < 500:
                    r.raise_for_status()
                if r.status_code >= 500:
                    raise RuntimeError(f"server {r.status_code}")
                break
            except Exception as e:
                last_exc = e
                if attempt < 3:
                    time.sleep(5 * (2 ** (attempt - 1)))
        else:
            if last_exc:
                raise last_exc


# ─── Debug trace formatter ────────────────────────────────────────────────────

def _format_debug_block(trace) -> str:
    """把 PipelineTrace 格式化成 Telegram 可读的 debug 摘要。"""
    if trace.context_pkg is None:
        return ""
    pkg = trace.context_pkg
    lines = [
        "─── debug trace ───",
        f"route={trace.route.route}  asset={trace.route.asset}  conf={trace.route.confidence:.2f}",
        f"user_emotion={trace.emotion.primary_emotion}({trace.emotion.emotion_intensity:.2f})",
    ]
    if trace.profile:
        lines.append(f"profile: role={trace.profile.role}  internal={trace.profile.is_internal}")
    derived_status = trace.meta.get("derived_signal_status", "unknown") if trace.meta else "unknown"
    price_status = trace.meta.get("holdings_price_status", "") if trace.meta else ""
    pnl_status = trace.meta.get("pnl_status", "") if trace.meta else ""
    lines.append(
        f"retrieved: news={len(pkg.news)}  community={len(pkg.community)}  "
        f"derived_cache={derived_status}"
    )
    if price_status:
        lines.append(f"price_status={price_status}  pnl={pnl_status}")
    if trace.aggregate:
        agg = trace.aggregate
        lines.append(
            f"aggregate: bias={agg.overall_bias}  bullish={agg.bullish_ratio:.0%}  "
            f"fomo={agg.fomo_ratio:.0%}  crowded={agg.crowded_trade_risk}"
        )
    if trace.decision:
        d = trace.decision
        sc = d.scores
        lines.append(
            f"decision: {d.action}({d.confidence})  "
            f"dir={sc.direction_score:+.2f}  entry={sc.entry_quality}"
            if sc else f"decision: {d.action}({d.confidence})"
        )
    if trace.policy_violations:
        lines.append(f"policy_violations: {trace.policy_violations}")
    lines.append(f"build_ms={pkg.build_time_ms:.0f}")
    return "\n".join(lines)


def _format_why_block(trace) -> str:
    """
    /why 输出：决策骨架推理链，一步一步展示打分过程。
    比 debug 更面向"理解决策"，而不是系统状态。
    """
    if trace.decision is None:
        route = trace.route.route
        return f"No decision path was triggered this turn (route={route}) — nothing to walk through."

    d = trace.decision
    sc = d.scores
    ev = d.evidence

    lines = ["─── Decision reasoning /why ───"]

    # 1. 资产 + 决策结论
    lines.append(f"Asset: {d.asset}  →  Call: {d.action} ({d.confidence})")

    # 2. 新闻面
    news_a = ev.get("news_assessment", {})
    lines.append(
        f"\n[News]  direction={news_a.get('direction','?')}  "
        f"bullish={news_a.get('bullish_score', 0):.2f}  bearish={news_a.get('bearish_score', 0):.2f}"
    )
    for bullet in (news_a.get("key_bullets") or [])[:3]:
        lines.append(f"  • {bullet}")

    # 3. 社区情绪
    ca = ev.get("community_aggregate", {})
    lines.append(
        f"\n[Community]  bias={ca.get('overall_bias','?')}  "
        f"bull={ca.get('bullish_ratio', 0):.0%}  bear={ca.get('bearish_ratio', 0):.0%}  "
        f"FOMO={ca.get('fomo_ratio', 0):.0%}  crowding={ca.get('crowded_trade_risk','?')}"
    )

    # 4. 趋势 + 打分
    tr = ev.get("trend", {})
    lines.append(
        f"\n[Trend]  momentum={tr.get('momentum_label','?')}  "
        f"overheating={tr.get('overheating_risk','?')}"
    )

    if sc:
        lines.append(
            f"\n[Composite]  direction={sc.direction_score:+.3f}  "
            f"crowding={sc.crowding_score:.3f}  entry_quality={sc.entry_quality}  chasing_risk={sc.chasing_risk}"
        )

    # 5. Engine trace (the step-by-step reasoning)
    engine_trace = ev.get("engine_trace", [])
    if engine_trace:
        lines.append("\n[Reasoning steps]")
        for step in engine_trace:
            lines.append(f"  → {step}")

    return "\n".join(lines)


# ─── 消息处理 ────────────────────────────────────────────────────────────────

_HELP_TEXT = (
    "Hi, I'm your market sentiment assistant.\n\n"
    "You can ask me things like:\n"
    "  • Should I buy gold here?\n"
    "  • Can I still chase Nvidia?\n"
    "  • What's everyone talking about lately?\n\n"
    "I combine recent news with community sentiment and give you a direct call with reasoning — "
    "no \"depends on your risk appetite\" hedging.\n\n"
    "Commands:\n"
    "  /debug             — toggle debug trace info\n"
    "  /why               — show the decision reasoning for the last question\n"
    "  /profile           — view your current user profile\n"
    "  /risk [keyword|summary] [count] — view risk alerts or a rolled-up summary\n"
    "  /mute <keyword> [1h|2h|6h|24h] — temporarily silence a trigger\n"
    "  /alert add <keyword> — add a trigger on the fly\n"
    "  /alert list — list all current triggers\n"
    "  /holdings          — view your holdings\n"
    "  /setholding <asset> [size] [avg_cost] [horizon]  — record a holding\n"
    "    e.g. /setholding gold medium 3320 long\n"
    "  /clearholding <asset>  — clear a holding\n"
    "  /report <asset> [executive]   — generate an asset snapshot report\n"
    "  /snapshot <asset> [executive] — same as /report"
)

# per-user last trace cache for /why
_LAST_TRACE: dict[int, object] = {}


def _handle_update(update: dict[str, Any]) -> None:
    message = update.get("message") or update.get("edited_message")
    if not message:
        return
    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    text = (message.get("text") or "").strip()
    user = message.get("from") or {}
    user_id = user.get("id")

    if not chat_id or not text:
        return

    logger.info("[bot] msg from chat=%s user=%s: %s", chat_id, user_id, text[:80])

    # ── 特殊命令 ─────────────────────────────────────────────────────────────
    if text.startswith("/start") or text.startswith("/help"):
        _send_reply(chat_id, _HELP_TEXT)
        return

    if text.startswith("/debug"):
        if user_id in _DEBUG_USERS:
            _DEBUG_USERS.discard(user_id)
            _send_reply(chat_id, "Debug mode off.")
        else:
            _DEBUG_USERS.add(user_id)
            _send_reply(chat_id, "Debug mode on. Trace info will be appended to your next reply.")
        return

    if text.startswith("/holdings"):
        from assistant.holdings import default_holdings_store
        hs = default_holdings_store()
        holdings = hs.get_all(user_id)
        if not holdings:
            _send_reply(chat_id,
                        "You haven't recorded any holdings yet.\n"
                        "Use /setholding <asset> [small|medium|large] to record, e.g.:\n"
                        "  /setholding gold small")
        else:
            lines = ["Your current holdings:"]
            for h in holdings:
                cost_note = f"  avg cost: {h.avg_cost:,.2f}" if h.avg_cost else ""
                notes_note = f"  notes: {h.notes}" if h.notes else ""
                lines.append(
                    f"  • {h.asset}: {h.position_size} ({h.horizon})"
                    + cost_note + notes_note
                )
            _send_reply(chat_id, "\n".join(lines))
        return

    if text.startswith("/setholding"):
        from assistant.holdings import default_holdings_store, parse_setholding_args
        asset, size, avg_cost, horizon = parse_setholding_args(text)
        if not asset:
            _send_reply(chat_id,
                        "Usage: /setholding <asset> [small|medium|large] [avg_cost] [short|mid|long]\n"
                        "e.g.  /setholding gold medium 3320 long")
            return
        hs = default_holdings_store()
        hs.set(user_id, asset, position_size=size, avg_cost=avg_cost, horizon=horizon)
        cost_note = f", avg cost {avg_cost:,.2f}" if avg_cost else ""
        horizon_note = f", horizon {horizon}" if horizon != "unknown" else ""
        _send_reply(chat_id,
                    f"Recorded: {asset} = {size} position{cost_note}{horizon_note}.\n"
                    "Next time you ask about this asset I'll factor in your position and P&L.")
        return

    if text.startswith("/clearholding"):
        from assistant.holdings import default_holdings_store, parse_clearholding_args
        asset = parse_clearholding_args(text)
        if not asset:
            _send_reply(chat_id, "Usage: /clearholding <asset>  e.g. /clearholding gold")
            return
        hs = default_holdings_store()
        removed = hs.clear(user_id, asset)
        if removed:
            _send_reply(chat_id, f"Cleared holding record for {asset}.")
        else:
            _send_reply(chat_id, f"No holding record found for {asset} — nothing to clear.")
        return

    if text.startswith("/report") or text.startswith("/snapshot"):
        parts = text.strip().split()
        asset = parts[1].lower() if len(parts) > 1 else None
        if not asset:
            _send_reply(chat_id, "Usage: /report <asset> [executive]  e.g. /report gold executive")
            return
        style = "executive" if len(parts) > 2 and parts[2].lower() == "executive" else "analyst"
        try:
            from assistant.report import generate_report
            report_text = generate_report(asset, style=style)
            _send_reply(chat_id, report_text)
        except Exception as e:
            logger.exception("report failed: %s", e)
            _send_reply(chat_id, f"Error generating report: {type(e).__name__}. Please try again later.")
        return

    if text.startswith("/risk"):
        parts = text.strip().split()[1:]
        n = 5
        keyword: str | None = None
        want_summary = False
        for part in parts:
            if part.isdigit():
                n = min(int(part), 20)
            elif part.lower() == "summary":
                want_summary = True
            else:
                keyword = part
        try:
            from risk_detector import load_recent_alerts, format_alert_msg
            alerts = load_recent_alerts(max(n, 20) if want_summary else n, keyword=keyword)
        except Exception as e:
            _send_reply(chat_id, f"Failed to load risk alerts: {e}")
            return
        if not alerts:
            hint = f" for '{keyword}'" if keyword else ""
            _send_reply(chat_id, f"No risk alerts{hint} on record.")
            return

        if want_summary:
            try:
                from llm_adapter import local_llm_callable, check_llm_available
                if check_llm_available():
                    import json as _j
                    bullets = "\n".join(
                        f"- [{a.get('severity','?')}] {a['title']} ({a.get('keyword','')})"
                        for a in alerts[-20:]
                    )
                    prompt = (
                        "You are a Singapore insurance-asset-management risk analyst. "
                        "Below is the list of recently fired risk alerts. "
                        "Output a concise roll-up summary (3-5 sentences) covering: "
                        "the dominant risk theme, which signals matter most, "
                        "and the net direction of impact on an insurer's portfolio. "
                        "Respond in English.\n\n"
                        f"{bullets}"
                    )
                    from llm_adapter import local_llm_plain
                    summary_text = local_llm_plain(prompt, timeout=60)
                    _send_reply(chat_id, f"🗂 Risk summary (last {len(alerts)} alerts)\n\n{summary_text}")
                    return
            except Exception as e:
                logger.warning("risk summary LLM failed: %s", e)
            # fallback: severity 分组
            high   = [a for a in alerts if a.get("severity") == "HIGH"]
            medium = [a for a in alerts if a.get("severity") == "MEDIUM"]
            lines  = [f"🗂 Risk overview (last {len(alerts)} alerts)"]
            if high:
                lines.append(f"\n🔴 HIGH ({len(high)})")
                lines += [f"  • {a['title'][:50]}" for a in high[-3:]]
            if medium:
                lines.append(f"\n🟠 MEDIUM ({len(medium)})")
                lines += [f"  • {a['title'][:50]}" for a in medium[-3:]]
            _send_reply(chat_id, "\n".join(lines))
            return

        # 按 severity 排序展示
        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        alerts_sorted = sorted(alerts, key=lambda a: severity_order.get(a.get("severity", "LOW"), 2))
        kw_hint = f"'{keyword}'" if keyword else f"last {n}"
        blocks = [f"🚨 Risk alerts ({kw_hint}):"]
        for a in alerts_sorted:
            blocks.append(format_alert_msg(a))
        _send_reply(chat_id, "\n\n─────\n\n".join(blocks))
        return

    if text.startswith("/mute"):
        # /mute <keyword> [1h|2h|6h|24h]
        parts = text.strip().split()[1:]
        if not parts:
            _send_reply(chat_id, "Usage: /mute <keyword> [1h|2h|6h|24h]")
            return
        import re as _re
        duration_map = {"1h": 1, "2h": 2, "6h": 6, "24h": 24}
        hours = 2
        kw_parts = []
        for p in parts:
            if p.lower() in duration_map:
                hours = duration_map[p.lower()]
            else:
                kw_parts.append(p)
        mute_kw = " ".join(kw_parts).lower()
        if not mute_kw:
            _send_reply(chat_id, "Please specify a keyword to mute.")
            return
        try:
            from risk_detector import load_cooldown, save_cooldown
            from datetime import datetime as _dt, timedelta as _td
            cd = load_cooldown()
            cd[mute_kw] = (_dt.now() + _td(hours=hours - 1)).isoformat()
            save_cooldown(cd)
            _send_reply(chat_id, f"🔕 Muted '{mute_kw}' for {hours}h — no alerts for this trigger during that window.")
        except Exception as e:
            _send_reply(chat_id, f"Mute failed: {e}")
        return

    if text.startswith("/alert"):
        parts = text.strip().split()[1:]
        if not parts:
            _send_reply(chat_id, "Usage: /alert add <keyword>  or  /alert list")
            return
        subcmd = parts[0].lower()
        if subcmd == "list":
            from risk_detector import TRIGGERS_PRECISION, TRIGGERS_BROAD
            p_list = "\n".join(f"  • {t}" for t in TRIGGERS_PRECISION)
            b_list = "\n".join(f"  • {t}" for t in TRIGGERS_BROAD)
            _send_reply(chat_id,
                f"🎯 Precision triggers (fire on hit):\n{p_list}\n\n"
                f"📡 Broad triggers (require negative sentiment):\n{b_list}"
            )
            return
        if subcmd == "add" and len(parts) >= 2:
            new_kw = " ".join(parts[1:]).lower().strip()
            try:
                import risk_detector as _rd
                if new_kw not in _rd.TRIGGERS_PRECISION and new_kw not in _rd.TRIGGERS_BROAD:
                    _rd.TRIGGERS_PRECISION.append(new_kw)
                    _rd.TRIGGERS.append(new_kw)
                    _send_reply(chat_id, f"✅ Added trigger '{new_kw}' (in-process only; lost on restart).")
                else:
                    _send_reply(chat_id, f"'{new_kw}' is already in the trigger list.")
            except Exception as e:
                _send_reply(chat_id, f"Add failed: {e}")
            return
        _send_reply(chat_id, "Usage: /alert add <keyword>  or  /alert list")
        return

    if text.startswith("/why"):
        last = _LAST_TRACE.get(user_id)
        if last is None:
            _send_reply(chat_id, "Ask me a market question first, then use /why to see the reasoning.")
        else:
            _send_reply(chat_id, _format_why_block(last))
        return

    if text.startswith("/profile"):
        from assistant.user_profile import get_user_profile
        profile = get_user_profile(user_id)
        _send_reply(
            chat_id,
            f"Your profile:\n"
            f"  Role: {profile.role}\n"
            f"  Risk preference: {profile.risk_preference}\n"
            f"  Reply style: {profile.preferred_style}\n"
            f"  Interests: {', '.join(profile.interests) or 'none set'}\n"
            f"  Internal user: {profile.is_internal}",
        )
        return

    # ── 正常问答流程 ─────────────────────────────────────────────────────────
    show_debug = (
        user_id in _DEBUG_USERS
        or os.getenv("ASSISTANT_DEBUG", "").strip() in ("1", "true", "yes")
    )

    try:
        trace = answer_question_traced(text, user_id=user_id)
        # cache trace for /why
        if user_id is not None:
            _LAST_TRACE[user_id] = trace
        reply = trace.reply
        if show_debug:
            debug_block = _format_debug_block(trace)
            if debug_block:
                reply = reply + "\n\n" + debug_block
    except Exception as e:
        logger.exception("pipeline failed")
        reply = f"System hiccup: {type(e).__name__}. Send another message in a moment."

    _send_reply(chat_id, reply)


# ─── 主循环 ─────────────────────────────────────────────────────────────────

_RUNNING = True


def _install_signal_handlers() -> None:
    def _stop(signum, frame):  # noqa: ARG001
        global _RUNNING
        _RUNNING = False
        logger.info("signal %s received, shutting down after current poll", signum)
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)


def run_forever() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _install_signal_handlers()
    logger.info("Telegram market assistant started. Polling...")

    offset: int | None = None
    while _RUNNING:
        try:
            updates = _get_updates(offset)
        except Exception as e:
            logger.warning("getUpdates failed: %s (retry in %ds)",
                           e, POLL_INTERVAL_ON_ERROR)
            time.sleep(POLL_INTERVAL_ON_ERROR)
            continue

        for update in updates:
            try:
                _handle_update(update)
            except Exception as e:
                logger.exception("handler error: %s", e)
            finally:
                offset = int(update["update_id"]) + 1

    logger.info("Telegram market assistant stopped.")


if __name__ == "__main__":
    run_forever()
