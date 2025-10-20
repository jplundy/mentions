from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from modeling.markets import (
    KalshiAPIError,
    KalshiClient,
    MarketComparison,
    compare_model_to_market_odds,
)


class DummyResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class DummySession:
    def __init__(self):
        self.headers = {}
        self.requests = []

    def request(self, method, url, timeout=10.0, **kwargs):
        self.requests.append(SimpleNamespace(method=method, url=url, kwargs=kwargs))
        if not hasattr(self, "response"):
            raise AssertionError("DummySession.response must be set before use")
        return self.response


def test_market_probability_converts_cents_to_probability():
    session = DummySession()
    session.response = DummyResponse(200, {"market": {"last_price": 42}})
    client = KalshiClient(session=session)

    probability = client.get_market_probability("TEST-1")

    assert probability == pytest.approx(0.42)
    assert session.requests[0].url.endswith("/markets/TEST-1")


def test_market_probability_supports_fractional_price():
    session = DummySession()
    session.response = DummyResponse(200, {"market": {"last_trade_price": 0.63}})
    client = KalshiClient(session=session)

    probability = client.get_market_probability("TEST-2")

    assert probability == pytest.approx(0.63)


def test_missing_price_fields_raise_error():
    session = DummySession()
    session.response = DummyResponse(200, {"market": {"description": "no prices"}})
    client = KalshiClient(session=session)

    with pytest.raises(KalshiAPIError):
        client.get_market_probability("TEST-3")


def test_compare_model_to_market_builds_summary_dataclass():
    session = DummySession()
    session.response = DummyResponse(200, {"market": {"yes_bid": 55}})
    client = KalshiClient(session=session)

    comparison = compare_model_to_market_odds(
        ticker="TEST-4", model_probability=0.62, client=client
    )

    assert isinstance(comparison, MarketComparison)
    assert comparison.market_probability == pytest.approx(0.55)
    assert comparison.model_probability == pytest.approx(0.62)
    assert comparison.edge == pytest.approx(0.07)


def test_authentication_sets_authorization_header():
    session = DummySession()
    session.response = DummyResponse(200, {"token": "abc123"})
    client = KalshiClient(session=session)

    # Trigger authentication manually to inspect header mutation.
    session.response = DummyResponse(200, {"token": "xyz"})
    client.authenticate(email="user@example.com", password="secret")

    assert session.headers["Authorization"] == "Bearer xyz"


def test_get_market_candlesticks_returns_list_of_mappings():
    session = DummySession()
    session.response = DummyResponse(
        200,
        {
            "ticker": "SERIES-1",
            "candlesticks": [
                {"end_period_ts": 100, "volume": 5},
                {"end_period_ts": 200, "volume": 7},
            ],
        },
    )
    client = KalshiClient(session=session)

    candlesticks = client.get_market_candlesticks(
        series_ticker="SERIES",
        market_ticker="SERIES-1",
        start_ts=0,
        end_ts=3600,
        period_minutes=60,
    )

    assert candlesticks == [
        {"end_period_ts": 100, "volume": 5},
        {"end_period_ts": 200, "volume": 7},
    ]
    request = session.requests[-1]
    assert request.url.endswith("/series/SERIES/markets/SERIES-1/candlesticks")
    assert request.kwargs["params"] == {
        "start_ts": 0,
        "end_ts": 3600,
        "period_interval": 60,
    }


def test_get_market_candlesticks_validates_period_interval():
    session = DummySession()
    session.response = DummyResponse(200, {"candlesticks": []})
    client = KalshiClient(session=session)

    with pytest.raises(KalshiAPIError):
        client.get_market_candlesticks(
            series_ticker="SERIES",
            market_ticker="SERIES-1",
            start_ts=0,
            end_ts=3600,
            period_minutes=5,
        )
