"""
StockTwits community sentiment source.

Why StockTwits:
  - Public JSON endpoints, no auth required, no monthly fee.
  - Finance-native platform: every message is scoped to cashtags ($SPY,
    $TLT, $VIX, ...). No off-topic noise to filter out.
  - Authors self-tag each post as Bullish or Bearish. That tag is
    ground-truth sentiment and bypasses the LLM classifier — we treat
    it as the primary sentiment signal and use follower weighting to
    avoid one-user drowning a stream.

Shape of the pipeline (parallels reddit_source / discord_source):
  fetch per symbol → dedupe → finance filter (light — platform is already
  finance-only) → normalize → cluster → LLM-analyze → return.

Env vars (all optional):
  STOCKTWITS_SYMBOLS          comma-separated cashtags without $
                              (default: macro ETFs + rate proxies)
  STOCKTWITS_MIN_FOLLOWERS    ignore authors with fewer followers (default 50)
  STOCKTWITS_LIMIT_PER_SYMBOL max messages per symbol (default 30, API max)
  STOCKTWITS_TOP_N_TOPICS     max topics to return (default 4)
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from community.analysis import (
    DEFAULT_FINANCE_KEYWORDS,
    SENTIMENT_LABELS_CN,
    classify_post,
)
from community.base import CommunityPost, CommunitySentiment, TopicCluster
from community.calibration import record_batch as record_calibration_batch
from community.clustering import cluster_posts, mark_rising_clusters
from community.llm_analyst import dedupe_posts, run_llm_pipeline
from community.normalize import normalize_posts

logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────

REQUEST_TIMEOUT = 15
STREAM_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"

# StockTwits rejects requests without a browser-ish UA (returns 403).
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120 Safari/537.36"
)
_HEADERS = {"User-Agent": _UA, "Accept": "application/json"}

# Symbols chosen to mirror the angles the daily brief already tracks:
# broad equity tape, rates, vol, gold, and macro-relevant sectors.
DEFAULT_SYMBOLS = [
    "SPY",    # broad equity
    "QQQ",    # tech
    "TLT",    # long-end rates
    "TNX",    # 10Y proxy
    "VIX",    # vol
    "GLD",    # gold
    "XLF",    # financials / banks
]


@dataclass
class StockTwitsFilterConfig:
    """StockTwits pipeline config. All loaded from env vars with defaults."""
    symbols: list[str] = field(default_factory=lambda: list(DEFAULT_SYMBOLS))
    min_followers: int = 50
    limit_per_symbol: int = 30
    finance_keywords: list[str] = field(default_factory=lambda: list(DEFAULT_FINANCE_KEYWORDS))
    top_n_topics: int = 4


def load_filter_config() -> StockTwitsFilterConfig:
    config = StockTwitsFilterConfig()

    env_syms = os.getenv("STOCKTWITS_SYMBOLS", "").strip()
    if env_syms:
        config.symbols = [
            s.strip().lstrip("$").upper()
            for s in env_syms.split(",")
            if s.strip()
        ]

    env_min = os.getenv("STOCKTWITS_MIN_FOLLOWERS", "").strip()
    if env_min:
        try:
            config.min_followers = int(env_min)
        except ValueError:
            pass

    env_limit = os.getenv("STOCKTWITS_LIMIT_PER_SYMBOL", "").strip()
    if env_limit:
        try:
            config.limit_per_symbol = max(5, min(int(env_limit), 30))
        except ValueError:
            pass

    env_topn = os.getenv("STOCKTWITS_TOP_N_TOPICS", "").strip()
    if env_topn:
        try:
            config.top_n_topics = int(env_topn)
        except ValueError:
            pass

    return config


# ─── Fetch ───────────────────────────────────────────────────────────────────

def _fetch_symbol_stream(symbol: str, limit: int = 30) -> list[dict[str, Any]]:
    """
    Fetch the recent message stream for one cashtag. Returns raw message
    dicts on success, empty list on rate-limit or error. We never raise
    from inside a single-symbol fetch because one symbol failing should
    not kill the entire batch.
    """
    url = STREAM_URL.format(symbol=symbol)
    try:
        r = requests.get(url, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        logger.warning("StockTwits fetch for $%s failed: %s", symbol, e)
        return []

    if r.status_code == 429:
        logger.info("StockTwits rate limited (429) on $%s — skipping", symbol)
        return []
    if r.status_code != 200:
        logger.warning("StockTwits returned %d for $%s", r.status_code, symbol)
        return []

    try:
        data = r.json()
    except ValueError as e:
        logger.warning("StockTwits JSON decode failed for $%s: %s", symbol, e)
        return []

    messages = data.get("messages", []) or []
    for m in messages:
        # Tag each message with the cashtag it came from so we can surface
        # it as the cluster channel.
        m["_symbol"] = symbol
    return messages[:limit]


def _parse_messages(raw: list[dict[str, Any]]) -> list[CommunityPost]:
    """
    Convert raw StockTwits messages into CommunityPost objects.

    We pack the author-declared sentiment ('Bullish' / 'Bearish') into the
    body via a special marker so downstream LLM classification can see it.
    This is the only place that marker is produced.
    """
    posts: list[CommunityPost] = []
    for m in raw:
        body = (m.get("body") or "").strip()
        if not body:
            continue

        user = m.get("user", {}) or {}
        followers = user.get("followers") or 0
        username = user.get("username") or ""

        likes_obj = m.get("likes") or {}
        like_total = likes_obj.get("total", 0) if isinstance(likes_obj, dict) else 0

        sent = (m.get("entities") or {}).get("sentiment")
        sent_basic = sent.get("basic") if isinstance(sent, dict) else None
        # Prepend an in-band sentiment marker if the author tagged the post.
        # This lets rule-based and LLM paths both see it without schema churn.
        if sent_basic:
            body = f"[ST:{sent_basic}] {body}"

        symbol = m.get("_symbol", "")
        created = m.get("created_at", "")
        created_utc = 0.0
        if created:
            try:
                import datetime as _dt
                created_utc = _dt.datetime.fromisoformat(
                    created.replace("Z", "+00:00")
                ).timestamp()
            except (TypeError, ValueError):
                created_utc = 0.0

        # Engagement model: followers give weight to the voice, explicit
        # likes give weight to the post itself. Both matter.
        engagement = min(followers // 50, 200) + like_total * 5

        posts.append(CommunityPost(
            title=body,
            score=engagement,
            num_comments=like_total,
            url=f"https://stocktwits.com/{username}/message/{m.get('id','')}"
                if username else "",
            # channel = cashtag; author kept in `subreddit` for the normalizer's
            # pre-existing author extraction path.
            subreddit=f"${symbol}" if symbol else "stocktwits",
            created_utc=created_utc,
        ))
    return posts


def fetch_stocktwits_posts(config: StockTwitsFilterConfig) -> list[CommunityPost]:
    """Walk every configured cashtag, merge unique messages."""
    all_posts: list[CommunityPost] = []
    seen_ids: set[str] = set()

    for symbol in config.symbols:
        raw = _fetch_symbol_stream(symbol, limit=config.limit_per_symbol)
        new = 0
        for m in raw:
            mid = str(m.get("id", ""))
            if not mid or mid in seen_ids:
                continue
            seen_ids.add(mid)
            parsed = _parse_messages([m])
            all_posts.extend(parsed)
            new += len(parsed)
        logger.info("StockTwits $%s: %d messages", symbol, new)
        # Polite gap — public endpoint, be a good citizen.
        time.sleep(0.3)

    logger.info(
        "StockTwits: fetched %d unique messages across %d symbols",
        len(all_posts), len(config.symbols),
    )
    return all_posts


# ─── Filtering ───────────────────────────────────────────────────────────────

# Authors to suppress outright — bot / paid-signal accounts seen in live
# sampling. Matched case-insensitively against the username in the post URL.
# Keep this list small and empirical; broad name matching produces false
# positives for legitimate accounts like "FedWatch" or "AlgoCrypto_Research".
_BLOCKED_AUTHORS = {
    "dragonalgo",
    "signalfeedbot",
    "optionsbot",
}

# Structured pump-signal shape: posts that read like machine-generated option
# plans. All three phrases have to appear — occasional human traders mention
# one or two ("Entry 86.50, stop 85") without being pump accounts.
_PUMP_SIGNAL_MARKERS = ("entry:", "stop:", "tp1")


def _extract_author(post: CommunityPost) -> str:
    url = post.url or ""
    if "stocktwits.com/" not in url:
        return ""
    try:
        return url.split("stocktwits.com/", 1)[1].split("/", 1)[0].lower()
    except IndexError:
        return ""


def _is_pump_signal(title: str) -> bool:
    """Detect structured option pump signals (Entry/Stop/TP1 template)."""
    lower = title.lower()
    hits = sum(1 for m in _PUMP_SIGNAL_MARKERS if m in lower)
    return hits >= 3


def _is_substantive(title: str) -> bool:
    """
    A very short post like "$SPY $DJT" or "$SPY havens …" is not a discussion,
    it's a hot-take fragment. Without enough body there's nothing for the
    topic classifier or LLM to reason about.

    The threshold is measured against *non-cashtag* content length so that
    "$SPY $QQQ $TLT" doesn't sneak through on pure cashtag count.
    """
    non_cashtag = " ".join(tok for tok in title.split() if not tok.startswith("$"))
    return len(non_cashtag.strip()) >= 20


def _passes_quality_threshold(
    post: CommunityPost, config: StockTwitsFilterConfig
) -> bool:
    """
    Multi-layer quality floor:
      1. Author-level: blocklist of known bot/paid-signal accounts
      2. Content-level: reject machine-generated option-plan templates
      3. Content-level: reject trivially short posts that only contain cashtags
      4. Engagement: derived from follower count, configurable floor

    Each layer is narrow so false positives stay low. Broader filters
    (multi-cashtag count, link density) were tested but flagged legitimate
    macro posts, so they're deliberately not used.
    """
    if _extract_author(post) in _BLOCKED_AUTHORS:
        return False
    if _is_pump_signal(post.title):
        return False
    if not _is_substantive(post.title):
        return False

    # Engagement floor, approximate. Score = min(followers//50, 200) + like*5.
    approx_followers_floor = config.min_followers // 50
    return post.score >= approx_followers_floor


def _is_finance_related(post: CommunityPost, config: StockTwitsFilterConfig) -> bool:
    # Platform is already cashtag-scoped, but we keep the filter so a
    # stream isn't polluted by spam like "$SPY check out my crypto course".
    text = post.title.lower()
    return any(kw in text for kw in config.finance_keywords) or "$" in post.title


def filter_posts(
    posts: list[CommunityPost], config: StockTwitsFilterConfig
) -> list[CommunityPost]:
    before = len(posts)

    posts = [p for p in posts if _passes_quality_threshold(p, config)]
    after_quality = len(posts)

    kept: list[CommunityPost] = []
    for p in posts:
        if _is_finance_related(p, config):
            kept.append(p)
            continue
        topic, _ = classify_post(p.title)
        if topic:
            kept.append(p)
    posts = kept
    after_finance = len(posts)

    logger.info(
        "StockTwits filtered: %d fetched → %d after quality (min_followers=%d) "
        "→ %d after finance relevance",
        before, after_quality, config.min_followers, after_finance,
    )
    return posts


# ─── Author-tag → sentiment bias ─────────────────────────────────────────────

def _apply_author_tag_bias(clusters: list[TopicCluster]) -> list[TopicCluster]:
    """
    StockTwits posts carry an author-declared Bullish/Bearish tag.
    After LLM analysis, blend that tag into the cluster's sentiment so we
    don't throw away free ground truth.

    Rule: if ≥60% of tagged posts lean one direction, and the LLM's
    label was 'neutral' or disagreed with the majority tag, override to
    the majority tag and raise the optimism/fear dimension accordingly.
    Otherwise leave LLM output untouched.

    The tag is the `[ST:Bullish]` / `[ST:Bearish]` prefix injected in
    _parse_messages. We read it back off the post titles here.
    """
    for c in clusters:
        bull = bear = 0
        for p in c.posts:
            title = p.title or ""
            if title.startswith("[ST:Bullish]"):
                bull += 1
            elif title.startswith("[ST:Bearish]"):
                bear += 1
        total_tagged = bull + bear
        if total_tagged == 0:
            continue

        bull_ratio = bull / total_tagged
        bear_ratio = bear / total_tagged

        if bull_ratio >= 0.6:
            majority = "bullish"
            strength = bull_ratio
        elif bear_ratio >= 0.6:
            majority = "bearish"
            strength = bear_ratio
        else:
            continue  # genuinely mixed — leave LLM call intact

        sent = c.sentiment
        if sent.label in ("neutral", "mixed") or sent.label != majority:
            sent.label = majority
            if majority == "bullish":
                sent.optimism = max(sent.optimism, round(0.5 + strength * 0.4, 2))
            else:
                sent.fear = max(sent.fear, round(0.5 + strength * 0.4, 2))
            # Recompute dominant dimension
            dims = {
                "optimism": sent.optimism, "fear": sent.fear,
                "uncertainty": sent.uncertainty, "skepticism": sent.skepticism,
                "hype": sent.hype,
            }
            top = max(dims.items(), key=lambda kv: kv[1])
            if top[1] > 0:
                sent.dominant_dimension = top[0]
                sent.intensity = top[1]
    return clusters


# ─── Main entry point ────────────────────────────────────────────────────────

def fetch_stocktwits_sentiment(
    config: StockTwitsFilterConfig | None = None,
    llm_callable: "Callable[[str], str] | None" = None,
) -> CommunitySentiment:
    """StockTwits sentiment pipeline — fully LLM-interpreted + author-tag biased."""
    if config is None:
        config = load_filter_config()

    if llm_callable is None:
        logger.info("StockTwits sentiment skipped — LLM not available")
        return CommunitySentiment(platform="stocktwits")

    raw_posts = fetch_stocktwits_posts(config)
    if not raw_posts:
        logger.warning("No messages fetched from StockTwits")
        return CommunitySentiment(platform="stocktwits")

    legacy_posts = filter_posts(raw_posts, config)
    unified = normalize_posts("stocktwits", legacy_posts)
    unified = dedupe_posts(unified)
    if not unified:
        logger.warning("All StockTwits messages filtered out")
        return CommunitySentiment(platform="stocktwits")

    clusters = cluster_posts(unified)
    clusters = mark_rising_clusters(clusters)

    kept, overall = run_llm_pipeline(
        platform="stocktwits",
        posts=unified,
        clusters=clusters,
        top_n_topics=config.top_n_topics,
        llm_callable=llm_callable,
    )

    # Calibration: log LLM label vs author-tag majority *before* the bias
    # overrides the LLM output. We want to measure the raw LLM, not the
    # already-biased sentiment, otherwise agreement is guaranteed-ish.
    try:
        wrote = record_calibration_batch(kept)
        if wrote:
            logger.info("StockTwits calibration: logged %d cluster rows", wrote)
    except Exception as e:
        logger.warning("Calibration logging failed (non-fatal): %s", e)

    kept = _apply_author_tag_bias(kept)

    return CommunitySentiment(
        platform="stocktwits",
        trending_topics=kept,
        overall_sentiment=overall,
        post_count=len(unified),
    )


# ─── Standalone test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = load_filter_config()
    print(f"Config: symbols={config.symbols}")
    print(f"  min_followers={config.min_followers}, "
          f"limit_per_symbol={config.limit_per_symbol}, "
          f"top_n={config.top_n_topics}")
    print()

    from llm_adapter import local_llm_callable
    result = fetch_stocktwits_sentiment(config, llm_callable=local_llm_callable)
    print(f"Platform: {result.platform}")
    print(f"Posts after filtering: {result.post_count}")
    print(f"Overall sentiment: {result.overall_sentiment}")
    print(f"Topics: {len(result.trending_topics)}")
    print()
    for c in result.trending_topics:
        sent_cn = SENTIMENT_LABELS_CN.get(c.sentiment.label, "中性")
        title = c.headline or c.rule_label
        print(f"【{title}】 {c.post_count} 条讨论 | 情绪{sent_cn}")
        print(f"  争论点：{c.discussion_focus}")
        print(f"  策略含义：{c.market_relevance}")
        print()
