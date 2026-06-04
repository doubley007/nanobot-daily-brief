"""
RSS/Atom news source — free, no API keys required.

Fetches from a curated list of public financial news feeds:
Reuters, Yahoo Finance, MarketWatch, Investing.com, CNBC, FT, Bloomberg.

Returns items in the same dict format as other news sources so
news_fetcher._normalize_item() can handle them uniformly.
"""
from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from typing import Any

logger = logging.getLogger(__name__)

# Each entry: (source_label, feed_url, category)
# Category aligns with CORE_BUCKETS in news_fetcher.
RSS_FEEDS: list[tuple[str, str, str]] = [
    # ── TIER-1: Great Eastern / OCBC group / SG insurance regulator & industry ──
    # Highest-signal channel. Google News queries use when:Nd to get fresh items;
    # per-category age window below (singapore_insurer = 7 days) keeps the slow
    # competitor beat alive without pulling in multi-month old stories.
    ("GN GE", "https://news.google.com/rss/search?q=%22Great+Eastern%22+insurance+OR+life+OR+MAS+OR+OCBC&when:14d&hl=en-SG&gl=SG&ceid=SG:en", "singapore_insurer"),
    ("GN GE Holdings", "https://news.google.com/rss/search?q=%22Great+Eastern+Holdings%22+OR+%22Great+Eastern+General%22&when:30d&hl=en-SG&gl=SG&ceid=SG:en", "singapore_insurer"),
    ("GN GE SGX", "https://news.google.com/rss/search?q=%22Great+Eastern+Holdings%22+SGX+OR+announcement+OR+dividend+OR+results&when:30d&hl=en-SG&gl=SG&ceid=SG:en", "singapore_insurer"),
    ("GN OCBC Group", "https://news.google.com/rss/search?q=OCBC+%22Great+Eastern%22+OR+bancassurance+OR+wealth+OR+insurance&when:7d&hl=en-SG&gl=SG&ceid=SG:en", "singapore_insurer"),
    # SG insurance competitors — low-volume beat uses 14-30d windows
    ("GN Prudential SG", "https://news.google.com/rss/search?q=%22Prudential+Singapore%22+OR+%22Prudential+Assurance%22&when:30d&hl=en-SG&gl=SG&ceid=SG:en", "singapore_insurer"),
    ("GN AIA SG", "https://news.google.com/rss/search?q=%22AIA+Singapore%22&when:14d&hl=en-SG&gl=SG&ceid=SG:en", "singapore_insurer"),
    ("GN Income SG", "https://news.google.com/rss/search?q=%22Income+Insurance%22+OR+%22NTUC+Income%22+singapore&when:14d&hl=en-SG&gl=SG&ceid=SG:en", "singapore_insurer"),
    ("GN Singlife", "https://news.google.com/rss/search?q=%22Singlife%22+OR+%22Aviva+Singapore%22&when:14d&hl=en-SG&gl=SG&ceid=SG:en", "singapore_insurer"),
    ("GN Manulife Tokio SG", "https://news.google.com/rss/search?q=%22Manulife+Singapore%22+OR+%22Tokio+Marine%22+singapore+OR+%22FWD%22+singapore&when:14d&hl=en-SG&gl=SG&ceid=SG:en", "singapore_insurer"),
    # Vertical insurance media + SG majors' insurance sections
    ("GN Insurance Asia", "https://news.google.com/rss/search?q=site:insuranceasia.com+singapore+OR+%22great+eastern%22+OR+prudential+OR+aia&when:14d&hl=en-SG&gl=SG&ceid=SG:en", "singapore_insurer"),
    ("GN Asia Insurance Review", "https://news.google.com/rss/search?q=site:asiainsurancereview.com+singapore&when:14d&hl=en-SG&gl=SG&ceid=SG:en", "singapore_insurer"),
    ("GN Straits Times Insurance", "https://news.google.com/rss/search?q=site:straitstimes.com+insurance+OR+%22great+eastern%22+OR+prudential+OR+aia&when:7d&hl=en-SG&gl=SG&ceid=SG:en", "singapore_insurer"),
    ("GN BT Insurance", "https://news.google.com/rss/search?q=site:businesstimes.com.sg+insurance+OR+insurer+OR+%22great+eastern%22&when:7d&hl=en-SG&gl=SG&ceid=SG:en", "singapore_insurer"),
    # SG insurance regulator & industry body
    ("GN MAS Insurance", "https://news.google.com/rss/search?q=MAS+insurance+OR+%22Integrated+Shield%22+OR+RBC2+OR+%22risk-based+capital%22&when:14d&hl=en-SG&gl=SG&ceid=SG:en", "regulation"),
    ("GN LIA Singapore", "https://news.google.com/rss/search?q=%22Life+Insurance+Association%22+singapore+OR+%22LIA+Singapore%22&when:30d&hl=en-SG&gl=SG&ceid=SG:en", "regulation"),
    ("GN SG Insurance Industry", "https://news.google.com/rss/search?q=singapore+insurance+premium+OR+claims+OR+%22medical+inflation%22+OR+%22par+fund%22&when:7d&hl=en-SG&gl=SG&ceid=SG:en", "regulation"),
    # ── Macro / rates (trimmed — kept the feeds that actually yield items) ─────
    ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_bulletins", "macro"),
    ("MarketWatch Top Stories", "https://feeds.content.dowjones.io/public/rss/mw_topstories", "macro"),
    ("CNBC Finance", "https://www.cnbc.com/id/10000664/device/rss/rss.html", "macro"),
    ("CNBC Rates", "https://www.cnbc.com/id/20910258/device/rss/rss.html", "macro"),
    ("Yahoo Finance Markets", "https://finance.yahoo.com/rss/topfinstories", "macro"),
    ("GN FOMC", "https://news.google.com/rss/search?q=FOMC+fed+rate+decision+monetary+policy&when:7d&hl=en-US&gl=US&ceid=US:en", "macro"),
    ("GN Reuters Finance", "https://news.google.com/rss/search?q=site:reuters.com+finance+OR+economy+OR+markets&when:3d&hl=en-US&gl=US&ceid=US:en", "macro"),
    # ── Credit (dropped HY/Sovereign/Private Credit empty searches) ─────────────
    ("GN Credit Ratings", "https://news.google.com/rss/search?q=Moodys+OR+Fitch+OR+%22S%26P%22+downgrade+OR+upgrade+credit+rating&when:7d&hl=en-US&gl=US&ceid=US:en", "credit"),
    # ── Singapore / MAS general ────────────────────────────────────────────────
    ("Business Times SG", "https://www.businesstimes.com.sg/rss/companies-markets", "singapore_local"),
    ("Business Times Banking", "https://www.businesstimes.com.sg/rss/banking-finance", "singapore_local"),
    ("Business Times Economy", "https://www.businesstimes.com.sg/rss/economy-policy", "singapore_local"),
    ("CNA Business", "https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml&category=6511", "singapore_local"),
    ("Straits Times Business", "https://www.straitstimes.com/news/business/rss.xml", "singapore_local"),
    ("GN SG Banks", "https://news.google.com/rss/search?q=DBS+OR+OCBC+OR+UOB+singapore+bank&when:3d&hl=en-SG&gl=SG&ceid=SG:en", "singapore_local"),
    ("GN SG Market", "https://news.google.com/rss/search?q=singapore+economy+OR+MAS+OR+SGD+interest+rate&when:3d&hl=en-SG&gl=SG&ceid=SG:en", "singapore_local"),
    # ── Real estate (SG-focused) + reinsurance Asia ────────────────────────────
    ("GN SG CRE", "https://news.google.com/rss/search?q=singapore+%22commercial+real+estate%22+OR+%22office+market%22+OR+CapitaLand+OR+REIT&when:7d&hl=en-SG&gl=SG&ceid=SG:en", "real_estate_loans"),
    ("GN Reinsurance Asia", "https://news.google.com/rss/search?q=reinsurance+asia+OR+Munich+Re+OR+Swiss+Re+OR+catastrophe+bond&when:7d&hl=en-US&gl=US&ceid=US:en", "regulation"),
]

# Per-bucket max age (hours). singapore_insurer + regulation get a 7-day window
# because SG insurance news is sparse; macro/credit keep 48h to stay current.
_MAX_AGE_BY_CATEGORY: dict[str, int] = {
    "singapore_insurer": 168,
    "regulation":        168,
    "singapore_local":    72,
    "real_estate_loans":  72,
    "macro":              48,
    "credit":              48,
}
_DEFAULT_MAX_AGE_HOURS = 48

_FETCH_TIMEOUT = 12    # seconds per feed
_MAX_WORKERS   = 6     # parallel feed fetches
_MAX_AGE_HOURS = _DEFAULT_MAX_AGE_HOURS    # back-compat; real filter uses per-category


def _parse_published(entry: Any) -> str | None:
    """Best-effort ISO timestamp from a feedparser entry."""
    # feedparser populates published_parsed (time.struct_time UTC) when possible
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            ts = time.mktime(entry.published_parsed)
            return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
        except Exception:
            pass
    if hasattr(entry, "published") and entry.published:
        try:
            dt = parsedate_to_datetime(entry.published)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return entry.published
    return None


def _entry_url(entry: Any) -> str | None:
    link = getattr(entry, "link", None)
    if link:
        return link
    for alt in getattr(entry, "links", []):
        href = alt.get("href") or alt.get("url")
        if href:
            return href
    return None


def _entry_summary(entry: Any) -> str:
    for attr in ("summary", "description", "content"):
        val = getattr(entry, attr, None)
        if not val:
            continue
        if isinstance(val, list):
            val = val[0].get("value", "") if val else ""
        # strip basic html tags inline
        import re
        val = re.sub(r"<[^>]+>", " ", str(val))
        val = " ".join(val.split())
        if len(val) > 20:
            return val[:800]
    return ""


def _is_too_old(entry: Any, max_age_hours: int = _DEFAULT_MAX_AGE_HOURS) -> bool:
    if not hasattr(entry, "published_parsed") or not entry.published_parsed:
        return False  # can't tell, keep it
    try:
        ts = time.mktime(entry.published_parsed)
        return (time.time() - ts) > max_age_hours * 3600
    except Exception:
        return False


def _fetch_one_feed(source: str, url: str, category: str) -> list[dict[str, Any]]:
    max_age = _MAX_AGE_BY_CATEGORY.get(category, _DEFAULT_MAX_AGE_HOURS)
    try:
        import feedparser
        feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})
        items: list[dict[str, Any]] = []
        for entry in feed.entries:
            if _is_too_old(entry, max_age):
                continue
            title = (getattr(entry, "title", "") or "").strip()
            if not title:
                continue
            summary = _entry_summary(entry)
            pub = _parse_published(entry)
            link = _entry_url(entry)
            # stable id for dedup
            uid = hashlib.sha1(f"{source}::{title}".encode()).hexdigest()[:16]
            items.append({
                "id": uid,
                "title": title,
                "summary": summary or title,
                "source": source,
                "category": category,
                "url": link,
                "published_at": pub,
            })
        logger.debug("RSS %s (%s): %d items", source, url, len(items))
        return items
    except Exception as e:
        logger.warning("RSS fetch failed — source=%s url=%s error=%s", source, url, e)
        return []


def _normalize_title_for_dedup(title: str) -> str:
    """
    Strip common Google News source suffixes like " - Reuters" for dedup purposes,
    so the same story fetched from multiple feeds is deduplicated correctly.
    """
    import re
    # Remove trailing " - Source Name" (Google News appends this)
    title = re.sub(r"\s+-\s+[A-Za-z0-9 &\.]+$", "", title.strip())
    # Lowercase + collapse whitespace
    return " ".join(title.lower().split())


def fetch_from_rss(limit: int = 60, timeout: int = _FETCH_TIMEOUT) -> list[dict[str, Any]]:
    """
    Fetch all configured RSS feeds in parallel.
    Returns raw dicts compatible with news_fetcher._normalize_item().
    Uses two-pass deduplication:
      1. By stable id (sha1 of source::title) — catches exact duplicates per source.
      2. By normalized title (strips " - Source" suffix) — catches cross-feed duplicates.
    """
    all_items: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_one_feed, src, url, cat): (src, url)
            for src, url, cat in RSS_FEEDS
        }
        for fut in as_completed(futures, timeout=timeout + 5):
            try:
                all_items.extend(fut.result())
            except Exception as e:
                src, url = futures[fut]
                logger.warning("RSS future error — source=%s: %s", src, e)

    # Pass 1: dedup by item id
    seen_ids: set[str] = set()
    pass1: list[dict[str, Any]] = []
    for item in all_items:
        key = item["id"]
        if key not in seen_ids:
            seen_ids.add(key)
            pass1.append(item)

    # Pass 2: cross-feed dedup by normalized title (keeps first occurrence, which is
    # typically from the more specific / higher-priority feed since RSS_FEEDS is ordered)
    seen_titles: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in pass1:
        norm = _normalize_title_for_dedup(item["title"])
        if norm not in seen_titles:
            seen_titles.add(norm)
            unique.append(item)

    logger.info(
        "RSS: fetched %d unique items (from %d raw, %d feeds)",
        len(unique), len(all_items), len(RSS_FEEDS),
    )
    return unique[:limit]
