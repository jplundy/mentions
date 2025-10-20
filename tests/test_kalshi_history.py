from __future__ import annotations

import json
import pytest

from modeling.kalshi_history import (
    MarketReference,
    _serialize_payload,
    _validate_period,
    fetch_market_candlesticks,
    parse_timestamp,
)


class DummyClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get_market_candlesticks(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload


def test_market_reference_parses_url():
    reference = MarketReference.parse(
        "https://kalshi.com/markets/kxsnfmention/sunday-night-football-mention/kxsnfmention-25oct20"
    )
    assert reference.series_ticker == "kxsnfmention"
    assert reference.market_ticker == "kxsnfmention-25oct20"


def test_market_reference_falls_back_to_prefix_when_series_missing():
    reference = MarketReference.parse("kxpowellmention-2023", series=None)
    assert reference.series_ticker == "kxpowellmention"
    assert reference.market_ticker == "kxpowellmention-2023"


def test_market_reference_requires_series_if_prefix_missing():
    with pytest.raises(ValueError):
        MarketReference.parse("SINGLEMARKET", series=None)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1603584000", 1603584000),
        ("1603584000.0", 1603584000),
        ("2020-10-25T00:00:00Z", 1603584000),
        ("2020-10-25", 1603584000),
    ],
)
def test_parse_timestamp_accepts_unix_and_iso_strings(value, expected):
    assert parse_timestamp(value) == expected


def test_parse_timestamp_rejects_invalid_values():
    with pytest.raises(ValueError):
        parse_timestamp("not-a-timestamp")


def test_validate_period_enforces_allowed_intervals():
    assert _validate_period(60) == 60
    with pytest.raises(ValueError):
        _validate_period(30)


def test_fetch_market_candlesticks_wraps_client_response():
    client = DummyClient(payload=[{"end_period_ts": 1}])
    reference = MarketReference(series_ticker="series", market_ticker="market-1")

    payload = fetch_market_candlesticks(
        client=client,
        reference=reference,
        start_ts=0,
        end_ts=10,
        period_minutes=60,
    )

    assert payload["candlesticks"] == [{"end_period_ts": 1}]
    assert client.calls == [
        {
            "series_ticker": "series",
            "market_ticker": "market-1",
            "start_ts": 0,
            "end_ts": 10,
            "period_minutes": 60,
        }
    ]


def test_serialize_payload_produces_sorted_json():
    serialized = _serialize_payload({"b": 2, "a": 1})
    assert json.loads(serialized) == {"a": 1, "b": 2}
