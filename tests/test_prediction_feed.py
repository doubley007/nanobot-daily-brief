"""
Tests for prediction_markets_feed Polymarket integration.

We don't hit the live Gamma API in tests — we stub _fetch_polymarket_markets
with fixtures so the filter/categorize/format logic is verified
deterministically. Coverage:

- keyword filter accepts macro/geo/SG/oil markets
- reject-list filters out celebrity / novelty markets
- uncertainty gate drops markets priced at 0.99/0.01
- YES-price + volume + end-date render into the summary line
- category assignment prefers SG → macro → event
"""
from unittest.mock import patch

import prediction_markets_feed as pmf


def _raw_market(
    question: str,
    yes: float = 0.5,
    volume: float = 10_000_000,
    end_date: str = "2026-12-31T00:00:00Z",
    slug: str = "stub-slug",
):
    # Polymarket returns outcomePrices as a JSON-encoded string.
    import json
    return {
        "question": question,
        "outcomePrices": json.dumps([f"{yes}", f"{1 - yes}"]),
        "volume": volume,
        "endDate": end_date,
        "slug": slug,
    }


class TestPolymarketFilter:
    def test_macro_market_kept(self):
        markets = [
            _raw_market("Will the Fed cut interest rates in June?", yes=0.45),
        ]
        with patch.object(pmf, "_fetch_polymarket_markets", return_value=markets):
            items = pmf._polymarket_to_feed_items()
        assert len(items) == 1
        assert items[0].category == "rates_macro"
        assert "YES 隐含概率 45%" in items[0].summary

    def test_geo_market_kept_as_event(self):
        markets = [
            _raw_market("Will the U.S. invade Iran before 2027?", yes=0.30),
        ]
        with patch.object(pmf, "_fetch_polymarket_markets", return_value=markets):
            items = pmf._polymarket_to_feed_items()
        assert len(items) == 1
        assert items[0].category == "event"

    def test_singapore_market_categorized_as_sg(self):
        markets = [
            _raw_market("Will MAS tighten policy in April?", yes=0.25),
        ]
        with patch.object(pmf, "_fetch_polymarket_markets", return_value=markets):
            items = pmf._polymarket_to_feed_items()
        assert len(items) == 1
        assert items[0].category == "sg_property"

    def test_celebrity_noise_rejected(self):
        markets = [
            _raw_market("Will LeBron James win the 2028 election?", yes=0.30),
            _raw_market("Will Kim Kardashian win the 2028 primary?", yes=0.40),
        ]
        with patch.object(pmf, "_fetch_polymarket_markets", return_value=markets):
            items = pmf._polymarket_to_feed_items()
        assert items == []

    def test_certain_markets_filtered_out(self):
        # YES at 0.99 or 0.01 is a certainty — no trading information.
        markets = [
            _raw_market("Will the Fed raise rates by 500bps in April?", yes=0.005),
            _raw_market("Will CPI be positive in 2026?", yes=0.995),
        ]
        with patch.object(pmf, "_fetch_polymarket_markets", return_value=markets):
            items = pmf._polymarket_to_feed_items()
        assert items == []

    def test_non_priority_market_filtered(self):
        markets = [
            _raw_market("Will Taylor Swift go on tour in Asia?", yes=0.55),
        ]
        with patch.object(pmf, "_fetch_polymarket_markets", return_value=markets):
            items = pmf._polymarket_to_feed_items()
        assert items == []

    def test_item_count_capped_at_max(self):
        # Build more than _PM_MAX_ITEMS valid markets.
        markets = [
            _raw_market(f"Will the Fed cut rates in month {i}?", yes=0.4 + 0.01 * i)
            for i in range(10)
        ]
        with patch.object(pmf, "_fetch_polymarket_markets", return_value=markets):
            items = pmf._polymarket_to_feed_items()
        assert len(items) <= pmf._PM_MAX_ITEMS

    def test_item_has_polymarket_url(self):
        markets = [_raw_market(
            "Will Fed cut rates?", yes=0.5, slug="will-fed-cut-rates",
        )]
        with patch.object(pmf, "_fetch_polymarket_markets", return_value=markets):
            items = pmf._polymarket_to_feed_items()
        assert items[0].url == "https://polymarket.com/market/will-fed-cut-rates"
        assert items[0].source == "Polymarket"

    def test_fetch_failure_returns_empty(self):
        with patch.object(pmf, "_fetch_polymarket_markets", return_value=[]):
            items = pmf._polymarket_to_feed_items()
        assert items == []


class TestFormatSummary:
    def test_summary_carries_probability_volume_date(self):
        m = _raw_market("x", yes=0.42, volume=5_000_000, end_date="2026-07-15T00:00:00Z")
        out = pmf._pm_format_summary(m)
        assert "42%" in out
        assert "$5.0M" in out
        assert "2026-07-15" in out


class TestUncertainGate:
    def test_mid_range_is_uncertain(self):
        assert pmf._pm_uncertain('["0.3", "0.7"]') is True

    def test_extreme_low_filtered(self):
        assert pmf._pm_uncertain('["0.01", "0.99"]') is False

    def test_extreme_high_filtered(self):
        assert pmf._pm_uncertain('["0.99", "0.01"]') is False

    def test_malformed_returns_false(self):
        assert pmf._pm_uncertain("not-json") is False
        assert pmf._pm_uncertain(None) is False
