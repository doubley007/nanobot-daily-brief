"""
Risk alert detector for daily news pipeline.

Scans each news article for risk keywords and combines with sentiment
to produce a scored alert. Alerts are persisted to data/risk_alerts.json
so both the daily job and the Telegram bot process can access them.

Dedup logic:
- seen-titles cache (data/risk_monitor_seen.json): ordered list, preserves
  insertion order so the _MAX_SEEN cap reliably drops the oldest entries.
  Shared by risk_monitor and daily_job — no cross-process re-alerts.
- keyword cooldown (1 hour): same keyword will not fire more than once per
  hour even if multiple headlines match it in the same run.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── TIER-1 triggers ──────────────────────────────────────────────────────────
# Keyword hit alone is sufficient AND bypasses LLM veto — these are assumed
# relevant by definition (GE / parent company OCBC / MAS acting on insurance /
# direct SG insurance competitors). Used by risk_monitor to force-confirm.
TRIGGERS_TIER1 = [
    # GE + parent
    "great eastern",
    "great eastern holdings",
    "great eastern general",
    "great eastern life",
    # Parent OCBC group (bancassurance / wealth impacts GE directly)
    "ocbc bank",
    "ocbc group",
    "ocbc wealth",
    "ocbc insurance",
    "oversea-chinese banking",
    # SG regulator actions on insurance
    "mas insurance",
    "mas circular",
    "mas notice",
    "mas regulatory",
    "monetary authority of singapore",
    "rbc 2",
    "rbc2",
    "risk-based capital",
    "life insurance association",
    "integrated shield",
    # Direct SG insurance competitors — only SG entity hits are T1.
    # Global parent groups (AIA Group, Prudential plc) create too much noise from
    # buybacks/share price/fund holdings; they are NOT T1.
    "prudential singapore",
    "prudential assurance",
    "aia singapore",
    "income insurance",
    "ntuc income",
    "singlife",
    "aviva singapore",
    "manulife singapore",
    "tokio marine singapore",
    "fwd singapore",
    "sunday insurance",
    "etiqa singapore",
]

# Precision triggers: keyword hit alone is sufficient (specific enough phrases).
TRIGGERS_PRECISION = [
    "bank crisis",
    "banking crisis",
    "bank run",
    "rate hike",
    "rate hikes",
    "recession",
    "geopolitical tension",
    "credit crunch",
    "sovereign default",
    "debt crisis",
    "financial crisis",
    "solvency crisis",
    "capital shortfall",
    "systemic risk",
    # Singapore / insurance specific — always relevant
    "capital requirement",
    "solvency ratio",
    "sgd depreciation",
    "sgd devaluation",
    "singapore recession",
    "singapore bank",
    # Insurance business lines (新增 — 寿险/健康险/资本/产品)
    "medical inflation",
    "longevity risk",
    "par fund",
    "participating fund",
    "bancassurance",
    "ilp",
    "investment-linked",
    "annuity crisis",
    "lapse rate",
    "claims inflation",
]

# Broad triggers: keyword hit requires NEGATIVE sentiment to fire.
TRIGGERS_BROAD = [
    "inflation",
    "liquidity",
    "default",
    "credit spread",
    "yield spike",
    "yield surge",
    "bond selloff",
    "bond sell-off",
    "rate cut",
    "interest rate",
    "fed pivot",
    "bank failure",
    "bank stress",
    "insurance regulation",
    "reinsurance",
    "credit downgrade",
    "rating downgrade",
    "loan default",
    "property crisis",
    "commercial real estate",
]

# All triggers combined for the fast pre-check in has_risk_keyword().
TRIGGERS = TRIGGERS_TIER1 + TRIGGERS_PRECISION + TRIGGERS_BROAD


def match_tier1(title: str, summary: str = "") -> list[str]:
    """Return the list of TIER-1 keywords that hit in title/summary."""
    text = (title + " " + summary).lower()
    return [t for t in TRIGGERS_TIER1 if t in text]

_DATA_DIR        = Path(__file__).resolve().parent.parent / "data"
_ALERTS_FILE     = _DATA_DIR / "risk_alerts.json"
_SEEN_FILE       = _DATA_DIR / "risk_monitor_seen.json"
_SEEN_RECENT_FILE = _DATA_DIR / "risk_monitor_seen_recent.json"
_COOLDOWN_FILE   = _DATA_DIR / "risk_keyword_cooldown.json"

_MAX_STORED         = 50
_MAX_SEEN           = 500
_COOLDOWN_HOURS     = 1
_CROSS_RUN_HOURS    = 24    # Jaccard dedup window across runs
# Titles share the entity (e.g. "Great Eastern") plus stop words, so similarity
# scores are low. 0.4 catches headlines that share a distinctive event noun
# (e.g. both mention "profit" alongside the entity) without collapsing
# genuinely different stories about the same entity.
_CROSS_RUN_JACCARD  = 0.4


# ── Seen-titles cache ────────────────────────────────────────────────────────
# Stored as an ordered JSON array so _MAX_SEEN cap always drops the oldest.

def load_seen() -> list[str]:
    if not _SEEN_FILE.exists():
        return []
    try:
        data = json.loads(_SEEN_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else list(data)
    except Exception:
        return []


def save_seen(seen: list[str]) -> None:
    _DATA_DIR.mkdir(exist_ok=True)
    _SEEN_FILE.write_text(
        json.dumps(seen[-_MAX_SEEN:], ensure_ascii=False),
        encoding="utf-8",
    )


# ── Keyword cooldown ─────────────────────────────────────────────────────────

def load_cooldown() -> dict[str, str]:
    if not _COOLDOWN_FILE.exists():
        return {}
    try:
        return json.loads(_COOLDOWN_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cooldown(cooldown: dict[str, str]) -> None:
    _DATA_DIR.mkdir(exist_ok=True)
    _COOLDOWN_FILE.write_text(json.dumps(cooldown, ensure_ascii=False), encoding="utf-8")


# ── Cross-run semantic dedup ─────────────────────────────────────────────────
# Stored as list of {title, ts} — only fired-alert titles, not every seen title.
# Purged to _CROSS_RUN_HOURS on every load/save.

def load_seen_recent() -> list[dict]:
    if not _SEEN_RECENT_FILE.exists():
        return []
    try:
        data = json.loads(_SEEN_RECENT_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _prune_seen_recent(entries: list[dict]) -> list[dict]:
    cutoff = datetime.now() - timedelta(hours=_CROSS_RUN_HOURS)
    result: list[dict] = []
    for e in entries:
        ts = e.get("ts", "")
        try:
            if datetime.fromisoformat(ts) >= cutoff:
                result.append(e)
        except Exception:
            continue
    return result


def save_seen_recent(entries: list[dict]) -> None:
    _DATA_DIR.mkdir(exist_ok=True)
    pruned = _prune_seen_recent(entries)
    _SEEN_RECENT_FILE.write_text(
        json.dumps(pruned, ensure_ascii=False),
        encoding="utf-8",
    )


def _is_semantic_dupe(title: str, seen_recent: list[dict]) -> bool:
    """
    True if any title in seen_recent (within _CROSS_RUN_HOURS) shares ≥
    _CROSS_RUN_JACCARD token overlap with the new title.
    """
    if not title or not seen_recent:
        return False
    new_tokens = _title_tokens(title)
    if not new_tokens:
        return False
    for entry in seen_recent:
        prev = entry.get("title", "")
        if not prev:
            continue
        if _jaccard(new_tokens, _title_tokens(prev)) >= _CROSS_RUN_JACCARD:
            return True
    return False


# Private aliases kept for any legacy callers
_load_cooldown = load_cooldown
_save_cooldown = save_cooldown


def _is_on_cooldown(keyword: str, cooldown: dict[str, str]) -> bool:
    last_str = cooldown.get(keyword)
    if not last_str:
        return False
    try:
        return datetime.now() - datetime.fromisoformat(last_str) < timedelta(hours=_COOLDOWN_HOURS)
    except Exception:
        return False


def _mark_cooldown(keyword: str, cooldown: dict[str, str]) -> None:
    cooldown[keyword] = datetime.now().isoformat()


# ── Keyword pre-check (used to skip LLM when no keyword present) ─────────────

def has_risk_keyword(title: str, summary: str = "") -> bool:
    """Fast check — returns True if any trigger keyword appears in title/summary."""
    text = (title + " " + summary).lower()
    return any(t in text for t in TRIGGERS)


# ── Core detection ────────────────────────────────────────────────────────────

def _clean_title(title: str, source: str) -> str:
    """Strip trailing ' - Source Name' suffix that RSS feeds append to titles."""
    t = title.strip()
    if source:
        suffix = f" - {source}"
        if t.endswith(suffix):
            return t[: -len(suffix)].strip()
    # Generic pattern: strip anything after the last ' - ' that looks like a source
    import re
    t = re.sub(r"\s+-\s+[A-Z][A-Za-z0-9 .]+$", "", t).strip()
    return t


def _why_for_keyword(keyword: str) -> str:
    """Return a one-line portfolio impact note for a risk keyword."""
    _MAP = {
        # Broad triggers
        "inflation":              "Rising inflation expectations weigh on fixed-income valuations and increase reinvestment-yield uncertainty.",
        "rate hike":              "Hike expectations pressure long-duration bond prices and lift fixed-income portfolio volatility.",
        "rate hikes":             "Hike expectations pressure long-duration bond prices and lift fixed-income portfolio volatility.",
        "rate cut":               "Cut signals move reinvestment yields and asset-liability matching for the insurance book.",
        "interest rate":          "Rate moves feed directly into fixed-income duration risk and the liability discount rate.",
        "liquidity":              "Tighter liquidity widens credit spreads and tightens funding conditions / short-asset pricing.",
        "default":                "Default-risk signals push credit spreads wider; watch credit exposure in the book.",
        "credit spread":          "Wider credit spreads mean the market is repricing default risk; corporate-bond valuations come under pressure.",
        "yield spike":            "A yield spike enlarges bond unrealized losses and pressures the solvency ratio.",
        "yield surge":            "A yield surge enlarges bond unrealized losses and pressures the solvency ratio.",
        "bond selloff":           "A bond sell-off lifts yields and pressures the insurance book's fixed-income carrying values.",
        "bond sell-off":          "A bond sell-off lifts yields and pressures the insurance book's fixed-income carrying values.",
        "fed pivot":              "A Fed policy pivot reshapes the yield curve and affects duration positioning / liability discounting.",
        "bank failure":           "Bank failure raises systemic concern and can transmit through credit channels to the broader market.",
        "bank stress":            "Bank stress tests or liquidity strains can hit financials and insurer counterparty exposure.",
        "insurance regulation":   "Tighter insurance regulation affects capital requirements and product pricing — watch compliance cost.",
        "reinsurance":            "Reinsurance-market shifts affect cedant costs and underwriting capacity.",
        "credit downgrade":       "A credit downgrade lifts funding costs and may trigger insurer investment-limit constraints.",
        "rating downgrade":       "A rating downgrade lifts funding costs and may trigger insurer investment-limit constraints.",
        "loan default":           "Rising loan defaults signal a deteriorating credit cycle; affects fixed-income and private-credit allocation.",
        "property crisis":        "A property crisis hits commercial real-estate loans and related fixed-income exposure.",
        "commercial real estate": "CRE stress pressures carrying values for insurers holding related assets.",
        # Precision triggers
        "bank crisis":            "A banking crisis hits financials and can transmit through credit channels to the broader market.",
        "banking crisis":         "A banking crisis hits financials and can transmit through credit channels to the broader market.",
        "bank run":               "Bank runs trigger systemic liquidity concerns and historically come with sharp credit-spread widening.",
        "geopolitical tension":   "Geopolitical tension lifts safe-haven demand and can drive FX volatility and capital-flow shifts.",
        "recession":              "Rising recession risk pressures risk assets and favors high-quality bonds / defensive positioning.",
        "credit crunch":          "A credit crunch sharply tightens funding conditions and raises expected corporate defaults.",
        "sovereign default":      "Sovereign default risk hits insurer solvency through sovereign-bond holdings.",
        "debt crisis":            "A debt crisis drives credit-spread blow-outs and asset repricing, hitting overall portfolio valuations.",
        "financial crisis":       "In a financial crisis cross-asset correlations rise and diversification benefits collapse.",
        "solvency crisis":        "A solvency crisis directly threatens the insurer's operating license and client confidence.",
        "capital shortfall":      "Capital-shortfall signals raise regulatory-intervention risk; watch industry contagion.",
        "systemic risk":          "Systemic-risk signals mean one event can trigger chain reactions; raise liquidity buffers.",
        "mas regulatory":         "MAS regulatory action directly affects SG insurer compliance cost and product strategy.",
        "mas notice":             "An MAS notice can adjust capital requirements or operating constraints — track immediately.",
        "mas circular":           "An MAS circular means a change in regulatory guidance — affects compliance and product design.",
        "great eastern":          "Great Eastern direct event — top priority; may involve regulatory, results, or operating impact.",
        "great eastern holdings":  "GE Holdings parent-level event — potentially strategic, capital structure, or regulatory.",
        "great eastern general":   "GE general-insurance event — affects non-life premiums / claims / distribution.",
        "great eastern life":      "GE Life event — directly impacts the core long-tail life franchise.",
        "ocbc bank":              "OCBC parent-bank event — can transmit to GE via bancassurance distribution and group capital.",
        "ocbc group":             "OCBC Group-level move — direct bearing on GE's strategic position and resource allocation.",
        "ocbc wealth":            "Change in OCBC Wealth affects intra-group resources and GE client-referral channels.",
        "ocbc insurance":         "OCBC insurance strategy shift is directly tied to GE's positioning.",
        "oversea-chinese banking": "OCBC's formal name appearing signals a group-level event.",
        "mas insurance":          "MAS action on insurance directly affects SG insurer capital requirements and compliance cost.",
        "monetary authority of singapore": "MAS policy action affects the local financial / insurance regulatory environment.",
        "rbc 2":                  "RBC 2 framework changes directly affect how insurer solvency is calculated.",
        "rbc2":                   "RBC 2 framework changes directly affect how insurer solvency is calculated.",
        "risk-based capital":     "Risk-based capital framework changes affect insurer capital requirements and portfolio strategy.",
        "life insurance association": "LIA industry-level action or data — reflects overall SG life direction.",
        "integrated shield":      "Integrated Shield scheme changes directly affect the SG health-insurance product line.",
        "prudential singapore":   "Pru SG is a primary GE competitor — moves reflect industry competitive dynamics.",
        "prudential assurance":   "Prudential SG event — direct competitor activity.",
        "aia singapore":          "AIA SG is a primary GE competitor — moves reflect industry competitive dynamics.",
        "aia group":              "AIA Group moves can spill over into SG subsidiaries and the regional market.",
        "income insurance":       "Income Insurance is one of GE's strongest local competitors — watch pricing wars and distribution shifts.",
        "ntuc income":            "NTUC Income moves affect SG insurance competitive structure and price bands.",
        "singlife":               "Singlife (ex-Aviva SG) is a key digital-channel competitor — watch product and distribution moves.",
        "aviva singapore":        "Aviva SG / Singlife event — affects SG mid-to-high-end market competition.",
        "manulife singapore":     "Manulife SG competitor event — watch channel and product strategy.",
        "tokio marine singapore": "Tokio Marine SG move — watch general insurance / composite competition.",
        "fwd singapore":          "FWD SG move — watch digital channel and product innovation.",
        "medical inflation":      "Medical inflation lifts health-insurance claims and compresses underwriting margin.",
        "longevity risk":         "Rising longevity risk directly affects life liabilities and annuity business.",
        "par fund":               "Par fund dynamics affect competitiveness of par products and client retention.",
        "participating fund":     "Par fund dynamics affect competitiveness of par products and client retention.",
        "bancassurance":          "Bancassurance-channel shifts directly affect GE's sales via OCBC distribution.",
        "ilp":                    "ILP-related events affect product mix and capital-markets linkage.",
        "investment-linked":      "ILP-related events affect product mix and capital-markets linkage.",
        "annuity crisis":         "An annuity crisis affects the stability of long-tail life liabilities.",
        "lapse rate":              "Lapse-rate moves affect life cash flows and product-pricing assumptions.",
        "claims inflation":       "Claims inflation compresses insurer underwriting profit.",
        "capital requirement":    "Capital-requirement changes directly affect insurer solvency ratios and portfolio allocation strategy.",
        "solvency ratio":         "Solvency-ratio moves are the core indicator of insurer financial health.",
        "sgd depreciation":       "SGD depreciation affects SGD-denominated liabilities and FX translation of foreign-currency assets.",
        "sgd devaluation":        "SGD devaluation affects SGD-denominated liabilities and FX translation of foreign-currency assets.",
        "singapore recession":    "A Singapore recession pressures local asset valuations and affects STI-related holdings.",
        "singapore bank":         "Major SG banking events affect DBS/OCBC/UOB holdings and the local credit environment.",
    }
    return _MAP.get(keyword, "Short-term risk signal that may pressure market sentiment and asset pricing.")


def _keyword_score(matched_count: int) -> float:
    """
    Score contribution from keyword hits (capped at 0.6).
    1 keyword → 0.4,  2 → 0.5,  3+ → 0.6
    Keeps score in [0, 1.0] after adding 0.4 for NEGATIVE sentiment.
    """
    return min(0.4 + (matched_count - 1) * 0.1, 0.6)


def _title_tokens(title: str) -> set[str]:
    """Lowercase word tokens, stripping short stop-words."""
    _STOP = {"the", "a", "an", "in", "of", "on", "at", "to", "is", "are",
             "was", "as", "by", "for", "its", "be", "with", "and", "or"}
    import re
    return {w for w in re.sub(r"[^a-z0-9 ]", " ", title.lower()).split()
            if len(w) > 2 and w not in _STOP}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def detect_risk(article: dict, sentiment_label: str) -> Optional[dict]:
    """
    Returns an alert dict if risk score >= 0.7, else None.

    Scoring:
      TIER-1 keyword hit:     base score 0.9 (GE / OCBC / MAS insurance / SG competitor)
      precision keyword hit:  +0.5 each (capped at 0.6 for 2+)
      broad keyword hit:      only counted when sentiment == NEGATIVE
      NEGATIVE sentiment:     +0.4
      Maximum possible score: 1.0

    article should have "title"; "summary", "source", "url" are optional.
    sentiment_label: "NEGATIVE", "POSITIVE", or "NEUTRAL".
    Does NOT check seen-cache or cooldown — callers manage those.
    """
    raw_title = article.get("title", "")
    source    = article.get("source", "")
    text      = f"{raw_title} {article.get('summary', '')}".lower()

    tier1_matched     = [t for t in TRIGGERS_TIER1 if t in text]
    precision_matched = [t for t in TRIGGERS_PRECISION if t in text]
    broad_matched     = [t for t in TRIGGERS_BROAD if t in text] if sentiment_label == "NEGATIVE" else []
    matched = tier1_matched + precision_matched + broad_matched

    if not matched:
        return None

    if tier1_matched:
        # TIER-1 bypasses the usual threshold — GE/OCBC/MAS/SG competitor
        # stories always fire, sentiment merely nudges severity.
        score = 0.9
        if sentiment_label == "NEGATIVE":
            score = 1.0
    else:
        score = _keyword_score(len(matched))
        if sentiment_label == "NEGATIVE":
            score += 0.4
    score = round(min(score, 1.0), 2)

    if score < 0.7:
        return None

    # Prefer LLM-generated analysis when available
    llm = article.get("llm_analysis") or {}
    if llm.get("impact"):
        why = llm["impact"]
        if llm.get("action"):
            why = why.rstrip("。") + "。" + llm["action"]
    else:
        why_parts = [_why_for_keyword(kw) for kw in matched]
        why = " ".join(why_parts) if len(why_parts) > 1 else why_parts[0]

    severity = llm.get("severity") or ("HIGH" if score >= 0.9 else "MEDIUM" if score >= 0.7 else "LOW")

    return {
        "title":         _clean_title(raw_title, source),
        "source":        source,
        "url":           article.get("url", ""),
        "keyword":       ", ".join(matched),
        "sentiment":     sentiment_label,
        "score":         score,
        "severity":      severity,
        "why":           why,
        "llm_reason":    llm.get("reason", ""),
        "bucket":        article.get("bucket", ""),
        "business_line": llm.get("business_line", ""),
        "detected_at":   datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


_JACCARD_THRESHOLD = 0.5   # titles sharing ≥50% tokens = same event


def detect_risk_deduped(
    article: dict,
    sentiment_label: str,
    seen: list[str],
    cooldown: dict[str, str],
    fired_titles: list[str] | None = None,
    seen_recent: list[dict] | None = None,
) -> Optional[dict]:
    """
    Like detect_risk() but also checks:
      1. seen-cache        — exact-match title skip (across processes)
      2. cross-run dedup   — Jaccard ≥ 0.6 vs alerts fired in the last 24h
      3. keyword cooldown  — same keyword fires at most once per hour
      4. this-run dedup    — Jaccard ≥ 0.5 vs alerts already fired this run

    fired_titles: in-memory list of alert titles fired so far in this run.
                  Pass the same list object across calls. If None, step 4 skipped.
    seen_recent:  list of {title, ts} from load_seen_recent(). Mutated in-place
                  when an alert fires. If None, step 2 skipped.
    Mutates seen, cooldown, fired_titles, seen_recent in-place.
    Callers must persist seen/cooldown/seen_recent afterwards.
    """
    title = article.get("title", "").strip()
    if not title or title in seen:
        return None
    seen.append(title)

    # Cross-run semantic dedup — catches the same event with different wording
    # across the 09:30 and 15:30 runs.
    if seen_recent is not None and _is_semantic_dupe(title, seen_recent):
        logger.debug("Cross-run semantic dupe, skipping: %s", title[:60])
        return None

    alert = detect_risk(article, sentiment_label)
    if alert is None:
        return None

    if _is_on_cooldown(alert["keyword"], cooldown):
        logger.debug("Cooldown active for '%s', skipping: %s", alert["keyword"], title[:60])
        return None

    # Event dedup within this run: suppress near-duplicate headlines from different sources
    if fired_titles is not None:
        tokens = _title_tokens(alert["title"])
        for prev in fired_titles:
            if _jaccard(tokens, _title_tokens(prev)) >= _JACCARD_THRESHOLD:
                logger.debug(
                    "Event dedup: similar to '%s', skipping: %s", prev[:50], title[:60]
                )
                return None

    _mark_cooldown(alert["keyword"], cooldown)
    if fired_titles is not None:
        fired_titles.append(alert["title"])
    if seen_recent is not None:
        seen_recent.append({
            "title": alert["title"],
            "ts":    datetime.now().isoformat(timespec="seconds"),
        })
    return alert


# ── Alert message formatter (shared by all callers) ──────────────────────────

def _relative_time(detected_at: str) -> str:
    """Convert 'YYYY-MM-DD HH:MM' to a human-readable relative string."""
    try:
        dt = datetime.strptime(detected_at, "%Y-%m-%d %H:%M")
        diff = datetime.now() - dt
        mins  = int(diff.total_seconds() / 60)
        if mins < 1:
            return "just now"
        if mins < 60:
            return f"{mins} min ago"
        hours = mins // 60
        if hours < 24:
            return f"{hours} h ago"
        days = hours // 24
        if days == 1:
            return f"yesterday {dt.strftime('%H:%M')}"
        return f"{days} d ago {dt.strftime('%H:%M')}"
    except Exception:
        return detected_at


def _severity_label(severity: str) -> str:
    return {"HIGH": "🔴 HIGH", "MEDIUM": "🟠 MEDIUM", "LOW": "🟡 LOW"}.get(
        (severity or "").upper(), "🟠 MEDIUM"
    )


_BUSINESS_LINE_LABEL = {
    "life_health":          "Life & Health",
    "investment_portfolio": "Investment Portfolio",
    "solvency_capital":     "Solvency & Capital",
    "product_distribution": "Product & Distribution",
    "macro_context":        "Macro",
}


def _business_line_label(value: str) -> str:
    return _BUSINESS_LINE_LABEL.get((value or "").lower(), value or "")


def format_alert_msg(alert: dict) -> str:
    """Return the canonical Telegram message string for a single alert."""
    source_line = f"Source: {alert['source']}\n"  if alert.get("source")     else ""
    url_line    = f"Link: {alert['url']}\n"       if alert.get("url")        else ""
    bucket_line = f"Bucket: {alert['bucket']}\n"  if alert.get("bucket")     else ""
    bl_label    = _business_line_label(alert.get("business_line", ""))
    bl_line     = f"Business line: {bl_label}\n" if bl_label                else ""
    why_line    = f"\n📌 Impact\n{alert['why']}\n" if alert.get("why")       else ""
    reason_line = f"({alert['llm_reason']})\n"    if alert.get("llm_reason") else ""
    rel_time    = _relative_time(alert.get("detected_at", ""))
    severity    = _severity_label(alert.get("severity", ""))
    keyword     = alert.get("keyword", "")
    return (
        f"🚨 Risk Alert  {rel_time}\n"
        f"{severity}\n"
        f"\nTitle: {alert['title']}\n"
        f"{source_line}"
        f"{bucket_line}"
        f"{bl_line}"
        f"Trigger: {keyword}\n"
        f"{why_line}"
        f"{reason_line}"
        f"{url_line}"
    ).rstrip()


# ── Digest formatter ──────────────────────────────────────────────────────────

def format_digest_msg(alerts: list[dict], run_time: str = "") -> str:
    """
    Build a single digest message grouping alerts by severity (HIGH → LOW).
    Used when risk monitor batches alerts instead of sending per-alert.
    """
    if not alerts:
        return ""

    buckets: dict[str, list[dict]] = {"HIGH": [], "MEDIUM": [], "LOW": []}
    for a in alerts:
        sev = (a.get("severity") or "MEDIUM").upper()
        if sev not in buckets:
            sev = "MEDIUM"
        buckets[sev].append(a)

    header_time = run_time or datetime.now().strftime("%Y-%m-%d %H:%M")
    total = sum(len(v) for v in buckets.values())
    lines: list[str] = [
        f"🚨 Risk Digest  {header_time}",
        f"{total} alerts total  "
        f"🔴{len(buckets['HIGH'])}  🟠{len(buckets['MEDIUM'])}  🟡{len(buckets['LOW'])}",
    ]

    for sev in ("HIGH", "MEDIUM", "LOW"):
        if not buckets[sev]:
            continue
        lines.append("")
        lines.append(f"━━━━━ {_severity_label(sev)} ━━━━━")
        for a in buckets[sev]:
            bl_label = _business_line_label(a.get("business_line", ""))
            bl_chip  = f"[{bl_label}] " if bl_label else ""
            kw       = a.get("keyword", "")
            src      = a.get("source", "")
            lines.append("")
            lines.append(f"• {bl_chip}{a.get('title', '')}")
            meta_bits: list[str] = []
            if src:
                meta_bits.append(src)
            if kw:
                meta_bits.append(f"Trigger: {kw}")
            if meta_bits:
                lines.append(f"  {'  |  '.join(meta_bits)}")
            if a.get("why"):
                lines.append(f"  📌 {a['why']}")
            if a.get("url"):
                lines.append(f"  🔗 {a['url']}")

    return "\n".join(lines)


# ── Persistence helpers ───────────────────────────────────────────────────────

def save_alert(alert: dict) -> None:
    """Append alert to the persistent JSON file (capped at _MAX_STORED)."""
    _DATA_DIR.mkdir(exist_ok=True)
    alerts: list[dict] = []
    if _ALERTS_FILE.exists():
        try:
            alerts = json.loads(_ALERTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            alerts = []
    alerts.append(alert)
    _ALERTS_FILE.write_text(
        json.dumps(alerts[-_MAX_STORED:], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_recent_alerts(n: int = 5, keyword: str | None = None) -> list[dict]:
    """
    Return the n most recent alerts from disk.
    If keyword is given, filter to alerts whose keyword contains that string
    (case-insensitive). n applies after filtering.
    """
    if not _ALERTS_FILE.exists():
        return []
    try:
        alerts = json.loads(_ALERTS_FILE.read_text(encoding="utf-8"))
        if keyword:
            kw = keyword.lower()
            alerts = [a for a in alerts if kw in a.get("keyword", "").lower()
                                        or kw in a.get("title", "").lower()]
        return alerts[-n:]
    except Exception:
        return []


_TREND_FILE         = _DATA_DIR / "risk_trend_state.json"
_TREND_WINDOW_HOURS = 24
_TREND_THRESHOLD    = 3   # same primary keyword N times → surge alert


def _primary_keyword(keyword_str: str) -> str:
    """Extract the first keyword from a comma-separated keyword string."""
    return keyword_str.split(",")[0].strip().lower()


def check_trend_surge(alert: dict) -> str | None:
    """
    Track how many times each primary keyword has fired in the last 24h.
    Returns a formatted surge-alert string when the threshold is crossed,
    None otherwise.  Mutates and persists the trend state file.
    """
    _DATA_DIR.mkdir(exist_ok=True)
    state: dict = {}
    if _TREND_FILE.exists():
        try:
            state = json.loads(_TREND_FILE.read_text(encoding="utf-8"))
        except Exception:
            state = {}

    now = datetime.now()
    cutoff = (now - timedelta(hours=_TREND_WINDOW_HOURS)).isoformat()
    primary = _primary_keyword(alert.get("keyword", "unknown"))

    # Prune old entries
    entries: list[str] = [t for t in state.get(primary, []) if t >= cutoff]
    entries.append(now.isoformat())
    state[primary] = entries
    _TREND_FILE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    count = len(entries)
    if count == _TREND_THRESHOLD:
        return (
            f"⚠️ Risk accumulation\n"
            f"Trigger '{primary}' has fired {count} times in the last {_TREND_WINDOW_HOURS}h — "
            f"signal is clustering.\n"
            f"Review exposure and hedges on this theme."
        )
    if count > _TREND_THRESHOLD and count % 5 == 0:
        # Re-alert every 5 subsequent hits to avoid spam
        return (
            f"⚠️ Risk build-up continues\n"
            f"Trigger '{primary}' has now fired {count} times in the last {_TREND_WINDOW_HOURS}h."
        )
    return None


def load_yesterday_alerts() -> list[dict]:
    """Return alerts from the last 24h, for use in the morning brief."""
    if not _ALERTS_FILE.exists():
        return []
    try:
        alerts = json.loads(_ALERTS_FILE.read_text(encoding="utf-8"))
        cutoff = (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M")
        return [a for a in alerts if a.get("detected_at", "") >= cutoff]
    except Exception:
        return []
