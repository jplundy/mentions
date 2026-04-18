"""Utilities for working with Kalshi market odds."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency is environment specific
    import requests as _requests  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _requests = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from requests import Session as SessionType
else:  # pragma: no cover - runtime alias when requests is absent
    SessionType = Any

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://trading-api.kalshi.com/trade-api/v2"
CANDLESTICK_INTERVALS: Sequence[int] = (1, 60, 1440)


class KalshiAPIError(RuntimeError):
    """Raised when the Kalshi API returns an unexpected response."""


@dataclass(frozen=True)
class MarketComparison:
    """Summary of the difference between model and market probabilities."""

    ticker: str
    model_probability: float
    market_probability: float

    @property
    def edge(self) -> float:
        """Return the model's probability edge over the market."""

        return self.model_probability - self.market_probability

    def as_dict(self) -> Mapping[str, float | str]:
        """Serialize the comparison to a dictionary for reporting."""

        return {
            "ticker": self.ticker,
            "model_probability": self.model_probability,
            "market_probability": self.market_probability,
            "edge": self.edge,
        }


class KalshiClient:
    """Lightweight Kalshi API client focused on public market data."""

    _DEFAULT_CACHE_TTL: float = 60.0

    def __init__(
        self,
        *,
        email: Optional[str] = None,
        password: Optional[str] = None,
        auth_token: Optional[str] = None,
        session: Optional[SessionType] = None,
        base_url: str = DEFAULT_API_BASE,
        timeout: float = 10.0,
        cache_ttl: float = _DEFAULT_CACHE_TTL,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or self._build_default_session()
        if not hasattr(self.session, "headers"):
            raise TypeError("HTTP session must expose a 'headers' attribute")
        self.session.headers.setdefault("Accept", "application/json")
        self.timeout = timeout
        self._cache_ttl = cache_ttl
        self._market_cache: Dict[str, Tuple[float, Mapping[str, Any]]] = {}

        if auth_token is not None:
            self._set_auth_header(auth_token)
        elif email is not None and password is not None:
            self.authenticate(email=email, password=password)

    def _build_default_session(self):
        if _requests is None:
            raise KalshiAPIError(
                "The 'requests' package is required unless a custom session is provided"
            )
        return _requests.Session()

    def _request(self, method: str, path: str, **kwargs: Any):
        url = f"{self.base_url}/{path.lstrip('/')}"
        response = self.session.request(method, url, timeout=self.timeout, **kwargs)
        if response.status_code >= 400:
            message = getattr(response, "text", "")
            raise KalshiAPIError(
                f"Kalshi API request to {url} failed with status "
                f"{response.status_code}: {message}"
            )
        return response

    def authenticate(self, *, email: str, password: str) -> str:
        """Authenticate with email/password credentials and persist the token."""

        payload = {"email": email, "password": password}
        response = self._request("POST", "/session/login", json=payload)
        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise KalshiAPIError("Kalshi login response was not valid JSON") from exc

        token = (
            data.get("token")
            or data.get("access_token")
            or data.get("session_token")
        )
        if not token:
            raise KalshiAPIError("Kalshi login response did not include a token")
        self._set_auth_header(token)
        return token

    def _set_auth_header(self, token: str) -> None:
        self.session.headers["Authorization"] = f"Bearer {token}"

    def get_market(self, ticker: str) -> Mapping[str, Any]:
        """Return raw market information for the given ticker."""

        if self._cache_ttl > 0:
            cached = self._market_cache.get(ticker)
            if cached is not None:
                ts, data = cached
                if monotonic() - ts < self._cache_ttl:
                    return data

        response = self._request("GET", f"/markets/{ticker}")
        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise KalshiAPIError(
                f"Kalshi market response for {ticker!r} was not valid JSON"
            ) from exc

        market = data.get("market") if isinstance(data, Mapping) else None
        if market is None:
            if isinstance(data, Mapping):
                market = data
            else:
                raise KalshiAPIError(
                    f"Kalshi market response for {ticker!r} was not a mapping"
                )
        if not isinstance(market, Mapping):
            raise KalshiAPIError(
                f"Kalshi market payload for {ticker!r} was not a mapping"
            )

        if self._cache_ttl > 0:
            self._market_cache[ticker] = (monotonic(), market)
        return market

    def get_market_probability(self, ticker: str) -> float:
        """Fetch the market probability implied by the current Kalshi price."""

        market = self.get_market(ticker)
        price = self._extract_price(market, ticker)
        return self._price_to_probability(price)

    PRICE_FIELDS = ("last_price", "last_trade_price", "yes_bid", "yes_ask")

    def _extract_price(self, market: Mapping[str, Any], ticker: str) -> float:
        for field in self.PRICE_FIELDS:
            value = market.get(field)
            if value in (None, ""):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                raise KalshiAPIError(
                    f"Kalshi market field {field!r} for {ticker!r} was not numeric"
                )
        raise KalshiAPIError(
            f"Kalshi market data for {ticker!r} did not include a price field"
        )

    @staticmethod
    def _price_to_probability(value: float) -> float:
        if value < 0:
            raise KalshiAPIError("Kalshi prices must be non-negative")
        probability = value
        if probability > 1:
            probability = probability / 100.0
        if probability > 1:
            raise KalshiAPIError(
                "Normalized Kalshi probability exceeded 1.0; check price format"
            )
        return probability

    def get_market_candlesticks(
        self,
        *,
        series_ticker: str,
        market_ticker: str,
        start_ts: int,
        end_ts: int,
        period_minutes: int,
    ) -> list[Mapping[str, Any]]:
        """Return historical candlesticks for a market within a time window.

        Parameters
        ----------
        series_ticker:
            The Kalshi series ticker that contains the requested market.
        market_ticker:
            The market ticker (for example ``"kxsnfmention-25oct20"``).
        start_ts:
            Inclusive Unix timestamp delimiting the earliest candlestick to
            return.
        end_ts:
            Inclusive Unix timestamp delimiting the latest candlestick to
            return.
        period_minutes:
            Size of each candlestick bucket in minutes. Kalshi currently
            supports ``1`` (minute), ``60`` (hour) and ``1440`` (day).

        Returns
        -------
        list[Mapping[str, Any]]
            A list of candlestick payloads exactly as returned by the Kalshi
            API. Each mapping contains OHLC information for YES prices along
            with volume and open interest statistics.
        """

        if period_minutes not in CANDLESTICK_INTERVALS:
            raise KalshiAPIError(
                "period_minutes must be one of {intervals}".format(
                    intervals=", ".join(str(i) for i in CANDLESTICK_INTERVALS)
                )
            )

        params = {
            "start_ts": int(start_ts),
            "end_ts": int(end_ts),
            "period_interval": int(period_minutes),
        }
        path = f"/series/{series_ticker}/markets/{market_ticker}/candlesticks"
        response = self._request("GET", path, params=params)
        try:
            payload = response.json()
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise KalshiAPIError(
                f"Kalshi candlestick response for {market_ticker!r} was not valid JSON"
            ) from exc

        if not isinstance(payload, Mapping):
            raise KalshiAPIError(
                f"Kalshi candlestick payload for {market_ticker!r} was not a mapping"
            )

        candlesticks: Iterable[Any] = payload.get("candlesticks", [])
        if candlesticks in (None, ""):
            return []

        normalized: list[Mapping[str, Any]] = []
        for index, entry in enumerate(candlesticks):
            if not isinstance(entry, Mapping):
                raise KalshiAPIError(
                    "Kalshi candlestick entry #{idx} for {ticker!r} was not a mapping"
                    .format(idx=index, ticker=market_ticker)
                )
            normalized.append(dict(entry))
        return normalized


    def get_market_probabilities_batch(
        self,
        tickers: Iterable[str],
        *,
        max_workers: int = 4,
    ) -> Dict[str, float]:
        """Fetch probabilities for multiple tickers concurrently.

        Failures for individual tickers are logged as warnings and omitted from
        the returned dict rather than aborting the whole batch.
        """

        tickers_list = list(tickers)
        results: Dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.get_market_probability, t): t for t in tickers_list}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    results[ticker] = future.result()
                except KalshiAPIError as exc:
                    logger.warning("Failed to fetch market probability for %s: %s", ticker, exc)
        return results


def compare_model_to_market_odds(
    *,
    ticker: str,
    model_probability: float,
    client: KalshiClient,
) -> MarketComparison:
    """Compare the model probability against the current Kalshi market odds."""

    market_probability = client.get_market_probability(ticker)
    return MarketComparison(
        ticker=ticker,
        model_probability=float(model_probability),
        market_probability=market_probability,
    )
