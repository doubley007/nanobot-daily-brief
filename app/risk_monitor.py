"""
Real-time risk monitor — runs every 30 minutes via launchd.

Pipeline:
  1. Fetch articles (RSS + Finnhub + Alpha Vantage) in parallel.
  2. Bucket pre-filter: drop anything not in the 7 risk-relevant buckets.
  3. Keyword pre-filter: skip articles with no risk trigger word (fast, no LLM).
  4. LLM analysis: for every article that passes steps 2-3, call the LLM once.
     The LLM acts as final judge — it confirms whether the story is a real risk
     signal and produces a structured analysis (impact on SG insurance portfolio,
     recommended action, severity).  confirmed=false suppresses the alert entirely.
  5. Dedup: seen-cache + per-keyword 1-hour cooldown + Jaccard event dedup.
  6. Send Telegram alert with LLM-generated text.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from risk_detector import (
    detect_risk_deduped,
    has_risk_keyword,
    match_tier1,
    save_alert,
    load_seen,
    save_seen,
    load_seen_recent,
    save_seen_recent,
    load_cooldown,
    save_cooldown,
    format_digest_msg,
    check_trend_surge,
)

_LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "risk_monitor.log"
_LOG_FILE.parent.mkdir(exist_ok=True)

_handler = logging.handlers.RotatingFileHandler(
    _LOG_FILE, maxBytes=1 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[_handler, logging.StreamHandler()])
logger = logging.getLogger("risk_monitor")


# Buckets worth scanning for risk signals — general equity noise is excluded.
# singapore_insurer is the TIER-1 bucket for GE / OCBC group / SG competitors.
_RISK_BUCKETS = {
    "singapore_insurer",
    "fixed_income_macro",
    "credit",
    "fx_liquidity",
    "singapore_local",
    "regulation",
    "real_estate_loans",
    "private_equity",
}

# Irrelevant single-country stories — these regions have negligible direct
# impact on a SG insurance portfolio and generate persistent false positives.
_REGION_NOISE_TERMS = {
    "south korea", "korea inflation", "korea rate",
    "australia rate", "australia inflation", "australia's central bank",
    "reserve bank of australia", "rba rate", "rba hike", "rba cut",
    "brazil", "argentina", "peru", "colombia", "mexico rate", "turkey rate",
    "pakistan", "bangladesh", "sri lanka",
}


def _is_region_noise(title: str, summary: str) -> bool:
    text = (title + " " + summary).lower()
    return any(term in text for term in _REGION_NOISE_TERMS)


_WATCHLIST_PREFIXES = (
    "stocks to watch",
    "stocks in focus",
    "stocks to buy",
    "stocks in play",
    "stocks that moved",
    "shares to watch",
)


def _is_watchlist_roundup(title: str) -> bool:
    """
    Watchlist roundup titles ("Stocks to watch: X, Y, Z") name GE or a competitor
    alongside 3+ unrelated companies — they're not actual news events. Drop them.
    """
    t = title.strip().lower()
    if not any(t.startswith(p) for p in _WATCHLIST_PREFIXES):
        return False
    # Extract the list portion after the colon
    if ":" not in t:
        return False
    after_colon = t.split(":", 1)[1]
    # Strip trailing " - Source" suffix
    after_colon = after_colon.split(" - ")[0]
    # Count comma-separated names; 3+ commas → listicle
    return after_colon.count(",") >= 2


def _fetch_rss_articles() -> list[dict]:
    """
    Fetch RSS articles and pre-filter to risk-relevant buckets only.
    Uses news_fetcher pipeline (low-value filter + bucket detection) so
    single-stock noise and general fluff never reach the risk detector.
    """
    try:
        from news_sources.rss_source import fetch_from_rss
        from news_fetcher import (
            _normalize_item,
            _is_excluded,
            _is_low_value_single_stock_story,
            detect_portfolio_impact_bucket,
        )

        raw_items = fetch_from_rss(limit=200)
        articles: list[dict] = []
        skipped_bucket = 0

        for item in raw_items:
            if not item.get("title"):
                continue
            normalized = _normalize_item(item)
            if normalized is None:
                continue
            if _is_excluded(normalized):
                continue
            if _is_low_value_single_stock_story(normalized):
                continue
            if _is_region_noise(normalized.title, normalized.summary):
                skipped_bucket += 1
                continue
            if _is_watchlist_roundup(normalized.title):
                skipped_bucket += 1
                continue
            bucket = detect_portfolio_impact_bucket(normalized)
            if bucket not in _RISK_BUCKETS:
                skipped_bucket += 1
                continue
            articles.append({
                "title":   normalized.title,
                "summary": normalized.summary,
                "source":  normalized.source,
                "url":     item.get("url", ""),
                "bucket":  bucket,
            })

        logger.info(
            "RSS pre-filter: %d kept, %d dropped (non-risk bucket) from %d raw",
            len(articles), skipped_bucket, len(raw_items),
        )
        return articles
    except Exception as e:
        logger.warning("RSS fetch failed: %s", e)
        return []


_LLM_ANALYSIS_PROMPT = """\
You are a risk analyst at Great Eastern Holdings (Singapore). Respond in English.

GE's core business:
  - Life & Health (Integrated Shield, critical illness, hospitalization, long-tail life, par products)
  - Investment portfolio (fixed income + equity + alternatives, backing long-tail life liabilities)
  - Solvency / Capital (MAS RBC 2 solvency ratio)
  - Product / distribution / competitors (bancassurance via OCBC, tied agents, ILP)
  - Parent OCBC Group strategy
Direct SG competitors: Prudential Singapore, AIA Singapore, Income Insurance, Singlife, Manulife, Tokio Marine, FWD Singapore.

Analyze the news below and decide whether it should be surfaced to the GE risk desk. Return strict JSON.

News title: {title}
News summary: {summary}
News source: {source}
Trigger keywords: {keywords}
Bucket: {bucket}

Return strict JSON of the form:
{{
  "confirmed": true/false,
  "sentiment": "NEGATIVE"/"NEUTRAL"/"POSITIVE",
  "severity": "HIGH"/"MEDIUM"/"LOW",
  "business_line": "life_health"/"investment_portfolio"/"solvency_capital"/"product_distribution"/"macro_context",
  "impact": "Specific impact on GE (1-2 sentences, English, name the affected business line).",
  "action": "Suggested focus or positioning adjustment (1 sentence, English).",
  "reason": "Why this is / isn't a real risk signal (1 sentence, English)."
}}

Four-tier priority framework:

[T1 must push — confirmed=true] — the story directly concerns:
  - Great Eastern / Great Eastern Holdings / Great Eastern General / Great Eastern Life
  - Parent OCBC Group (group-level events) or the bancassurance channel
  - MAS regulatory action on the insurance industry (circular, notice, RBC 2, capital requirement, Integrated Shield rules)
  - Direct SG competitors (Prudential SG, AIA SG, Income, Singlife, Manulife SG, Tokio Marine SG, FWD SG) — product, distribution, capital, or regulatory events
  These fire confirmed=true unconditionally, even if tone is neutral / slightly positive (severity scaled by importance).

[T2 likely push — confirmed=true] — content clearly moves SG insurance:
  - Substantive changes in Singapore economy / rates / SGD FX (not routine market commentary)
  - Direct Life & Health terms: medical inflation, longevity, par fund, ILP, annuity, claims inflation, lapse rate
  - Major moves in global reinsurance (Munich Re / Swiss Re level)
  - Substantive Asian rate-policy decisions with a clear read-through to the SG fixed-income book

[T3 conditional push — confirmed=true only if ALL hold] — global macro events:
  - Actual policy decision or realized market shock (not prediction/debate)
  - Clear, quantifiable channel to fixed-income duration / credit spread / solvency ratio / capital requirements
  - Transmission path explainable in 1-2 steps (e.g., Fed +50bps -> SG long-duration bond exposure)
  If the piece is commentary / forecasting / retrospective, set confirmed=false even if macro keywords hit.

[T4 reject — confirmed=false]:
  - Single-stock explainers (steel, cement, tech, consumer, non-bank / non-insurance names)
  - Mortgage / high-yield deposit / personal finance coverage
  - Single-country EM stories (Australia, Korea, Peru, Brazil, etc.) with negligible SG-insurer impact
  - Retrospective / predictive pieces with no substantive policy decision
  - Risk keyword mentioned only in passing; article body unrelated to insurance

business_line mapping:
  - T1 GE / OCBC / MAS / competitor events -> product_distribution / solvency_capital / investment_portfolio / life_health (based on content)
  - T2 rates / FX / medical inflation etc. -> life_health or investment_portfolio
  - T3 global macro -> macro_context

severity:
  - HIGH: T1 on GE itself, or MAS-on-insurance + NEGATIVE; or anything that can directly trigger portfolio rebalancing / compliance changes
  - MEDIUM: T1 competitor events; T2 with clear business impact
  - LOW: T3 background macro; neutral T1 events

Positive-event handling:
  - Positive T1 stories (e.g. "GE Q1 profit +16%", "OCBC acquisition closes") are NOT risk signals.
  - Set confirmed=false; if forced to confirmed=true, hold severity=LOW.
  - Upgrade only if a positive event triggers a secondary regulatory / compliance risk (e.g. regulator probes outsize results).\
"""


def _llm_analyze_risk(
    title: str,
    summary: str,
    source: str,
    keywords: str,
    bucket: str,
    av_sentiment: str = "",
) -> dict:
    """
    Call the LLM for a full risk analysis of one article.
    Returns a dict with keys: confirmed, sentiment, severity, impact, action, reason.
    Falls back to a keyword-heuristic result when LLM is unavailable.
    """
    try:
        from llm_adapter import check_llm_available, local_llm_callable
        if check_llm_available():
            prompt = _LLM_ANALYSIS_PROMPT.format(
                title=title,
                summary=summary or "（无摘要）",
                source=source or "Unknown",
                keywords=keywords,
                bucket=bucket or "unknown",
            )
            raw = local_llm_callable(prompt, timeout=45)
            result = json.loads(raw)
            # Normalise fields in case LLM returns unexpected casing
            result["confirmed"] = bool(result.get("confirmed", False))
            result["sentiment"] = result.get("sentiment", "NEUTRAL").upper()
            result["severity"]  = result.get("severity", "MEDIUM").upper()
            return result
    except Exception as e:
        logger.warning("LLM analysis failed, using fallback: %s", e)

    # Keyword heuristic fallback (LLM unavailable or parse error)
    _NEG = {"crash", "crisis", "default", "collapse", "recession", "downgrade",
            "selloff", "sell-off", "tumble", "plunge", "slump", "spike", "stress",
            "bank run", "bank failure", "solvency", "capital shortfall"}
    text = (title + " " + (summary or "")).lower()
    is_neg = any(w in text for w in _NEG)
    # Use AV sentiment as a secondary signal when available
    if av_sentiment == "NEGATIVE":
        is_neg = True
    return {
        "confirmed": is_neg,
        "sentiment": "NEGATIVE" if is_neg else "NEUTRAL",
        "severity":  "MEDIUM",
        "impact":    "(LLM unavailable — impact analysis not produced)",
        "action":    "Please assess this risk signal manually.",
        "reason":    "Keyword match; LLM unavailable, fallback heuristic applied.",
    }


def _fetch_api_articles() -> list[dict]:
    """
    Fetch from Finnhub and Alpha Vantage, apply the same bucket pre-filter
    as RSS. Alpha Vantage items carry av_sentiment so the LLM is skipped.
    """
    from news_fetcher import (
        _normalize_item,
        _is_excluded,
        _is_low_value_single_stock_story,
        detect_portfolio_impact_bucket,
    )

    raw: list[dict] = []

    try:
        from news_sources.finnhub_source import fetch_from_finnhub
        raw.extend(fetch_from_finnhub(limit=40))
    except Exception as e:
        logger.warning("Finnhub fetch failed: %s", e)

    try:
        from news_sources.alpha_vantage_source import fetch_from_alpha_vantage
        raw.extend(fetch_from_alpha_vantage(limit=40))
    except Exception as e:
        logger.warning("Alpha Vantage fetch failed: %s", e)

    articles: list[dict] = []
    skipped = 0
    for item in raw:
        if not item.get("title"):
            continue
        normalized = _normalize_item(item)
        if normalized is None:
            continue
        if _is_excluded(normalized):
            continue
        if _is_low_value_single_stock_story(normalized):
            continue
        if _is_region_noise(normalized.title, normalized.summary):
            skipped += 1
            continue
        if _is_watchlist_roundup(normalized.title):
            skipped += 1
            continue
        bucket = detect_portfolio_impact_bucket(normalized)
        if bucket not in _RISK_BUCKETS:
            skipped += 1
            continue
        articles.append({
            "title":        normalized.title,
            "summary":      normalized.summary,
            "source":       normalized.source,
            "url":          item.get("url", ""),
            "bucket":       bucket,
            "av_sentiment": item.get("av_sentiment", ""),
        })

    logger.info(
        "API pre-filter: %d kept, %d dropped (non-risk bucket) from %d raw",
        len(articles), skipped, len(raw),
    )
    return articles


def _send_digest(alerts: list[dict]) -> None:
    from telegram_sender import send_to_telegram
    msg = format_digest_msg(alerts)
    if msg:
        send_to_telegram(msg)


def run() -> None:
    logger.info("Risk monitor started")
    seen        = load_seen()
    seen_recent = load_seen_recent()
    cooldown    = load_cooldown()
    seen_set    = set(seen)

    # Fetch all sources in parallel
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_rss = pool.submit(_fetch_rss_articles)
        f_api = pool.submit(_fetch_api_articles)
        rss_articles = f_rss.result()
        api_articles = f_api.result()

    # Merge and deduplicate by title before processing
    seen_titles_this_run: set[str] = set()
    articles: list[dict] = []
    for art in rss_articles + api_articles:
        t = art.get("title", "").strip()
        if t and t not in seen_titles_this_run:
            seen_titles_this_run.add(t)
            articles.append(art)

    logger.info(
        "Fetched %d total articles (%d RSS, %d API) after merge-dedup",
        len(articles), len(rss_articles), len(api_articles),
    )

    to_llm_count = sum(
        1 for a in articles
        if a["title"].strip()
        and a["title"].strip() not in seen_set
        and has_risk_keyword(a["title"], a.get("summary", ""))
    )
    logger.info(
        "Articles to LLM (risk keywords, unseen): %d / %d with keywords / %d total — seen-cache=%d",
        to_llm_count,
        sum(1 for a in articles if has_risk_keyword(a["title"], a.get("summary", ""))),
        len(articles),
        len(seen_set),
    )

    fired_titles: list[str] = []
    pending_alerts: list[dict] = []   # collected for digest at run end
    surge_msgs:    list[str]   = []
    llm_confirmed = 0
    llm_rejected  = 0

    for art in articles:
        title = art["title"].strip()
        if not title or title in seen_set:
            continue
        seen_set.add(title)

        if not has_risk_keyword(title, art.get("summary", "")):
            seen.append(title)
            continue

        tier1_hits = match_tier1(title, art.get("summary", ""))

        # Full LLM analysis — sentiment + risk confirmation + portfolio impact
        llm_result = _llm_analyze_risk(
            title    = title,
            summary  = art.get("summary", ""),
            source   = art.get("source", ""),
            keywords = ", ".join(kw for kw in
                       (art.get("matched_keywords") or [title])
                       if kw),
            bucket   = art.get("bucket", ""),
            av_sentiment = art.get("av_sentiment", ""),
        )

        if tier1_hits:
            sentiment = (llm_result.get("sentiment") or "").upper()
            if sentiment == "POSITIVE":
                # Positive T1 stories (e.g. "GE Q1 profit +16%") are not risks.
                # Let the LLM's verdict stand; if it said confirmed, downgrade
                # severity to LOW so it surfaces quietly in the digest tail.
                if llm_result.get("confirmed"):
                    llm_result["severity"] = "LOW"
                logger.info(
                    "TIER-1 positive: no force-confirm, severity=LOW — hits=%s — %s",
                    tier1_hits, title[:60],
                )
            elif not llm_result.get("confirmed", False):
                # TIER-1 non-positive: force confirm (LLM can still shape text
                # and severity, but can't veto).
                logger.info(
                    "TIER-1 override: LLM said false but forcing confirm — hits=%s — %s",
                    tier1_hits, title[:60],
                )
                llm_result["confirmed"] = True
                if not llm_result.get("severity"):
                    llm_result["severity"] = "MEDIUM"

        if not llm_result.get("confirmed", False):
            llm_rejected += 1
            logger.info(
                "LLM rejected (not a real risk): %s — %s",
                llm_result.get("reason", ""), title[:60],
            )
            seen.append(title)
            continue

        llm_confirmed += 1
        # Inject LLM fields into the article dict so detect_risk can use them
        art["llm_analysis"] = llm_result

        alert = detect_risk_deduped(
            art, llm_result["sentiment"], seen, cooldown, fired_titles,
            seen_recent=seen_recent,
        )
        if alert:
            save_alert(alert)
            logger.info(
                "[INFO] Risk detected: %s (%s) [%s] — %s",
                alert["keyword"], alert.get("severity", "?"),
                alert.get("bucket", "?"), title[:60],
            )
            pending_alerts.append(alert)
            surge = check_trend_surge(alert)
            if surge:
                surge_msgs.append(surge)

    save_seen(seen)
    save_seen_recent(seen_recent)
    save_cooldown(cooldown)

    # Single digest push for all alerts this run
    if pending_alerts:
        try:
            _send_digest(pending_alerts)
            logger.info("[INFO] Digest sent (%d alerts)", len(pending_alerts))
        except Exception as e:
            logger.warning("Digest send failed (non-fatal): %s", e)
    else:
        # Heartbeat: send an empty-digest notice so subscribers can distinguish
        # "quiet run" from "monitor crashed".
        try:
            from telegram_sender import send_to_telegram
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            send_to_telegram(
                f"📭 Risk Digest  {now}\nNo new alerts (scanned {len(articles)} items)"
            )
            logger.info("[INFO] Empty-digest heartbeat sent")
        except Exception as e:
            logger.warning("Empty digest send failed (non-fatal): %s", e)

    # Trend surge messages still go separately — they signal accumulation.
    for surge in surge_msgs:
        try:
            from telegram_sender import send_to_telegram
            send_to_telegram(surge)
            logger.info("[INFO] Trend surge alert sent")
        except Exception as e:
            logger.warning("Trend surge send failed: %s", e)

    logger.info(
        "Risk monitor finished — %d alert(s) in digest | LLM: %d confirmed, %d rejected | %d articles total",
        len(pending_alerts), llm_confirmed, llm_rejected, len(articles),
    )


if __name__ == "__main__":
    run()
