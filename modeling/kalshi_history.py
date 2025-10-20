"""Utilities for downloading historical Kalshi mentions market data."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlparse

from .markets import CANDLESTICK_INTERVALS, KalshiClient


@dataclass(frozen=True)
class MarketReference:
    """Reference to a Kalshi market within a series."""

    series_ticker: str
    market_ticker: str

    @classmethod
    def parse(cls, value: str, *, series: str | None = None) -> "MarketReference":
        """Parse a market ticker or URL into a :class:`MarketReference`.

        ``value`` may be a bare ticker (``"kxsnfmention-25oct20"``) or a full
        Kalshi market URL such as
        ``"https://kalshi.com/markets/kxsnfmention/.../kxsnfmention-25oct20"``.
        When ``value`` is not a URL, the ``series`` argument is used to infer the
        series ticker. If ``series`` is omitted, the prefix before the first dash
        in the market ticker is used as a best-effort fallback.
        """

        parsed = urlparse(value)
        if parsed.scheme and parsed.netloc:
            parts = [segment for segment in parsed.path.split("/") if segment]
            if len(parts) >= 4 and parts[0] == "markets":
                return cls(series_ticker=parts[1], market_ticker=parts[-1])
            raise ValueError(
                "Could not extract tickers from Kalshi market URL: {value}".format(
                    value=value
                )
            )

        market_ticker = value.strip()
        if not market_ticker:
            raise ValueError("Market ticker must be a non-empty string")

        series_ticker = series.strip() if series else ""
        if not series_ticker:
            if "-" in market_ticker:
                series_ticker = market_ticker.split("-", 1)[0]
            else:
                raise ValueError(
                    "Series ticker must be provided when market reference is not a URL"
                )

        return cls(series_ticker=series_ticker, market_ticker=market_ticker)


def parse_timestamp(value: str) -> int:
    """Parse a Unix timestamp or ISO-8601 string into integer seconds."""

    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Timestamp must be a non-empty string")

    try:
        return int(cleaned)
    except ValueError:
        try:
            return int(float(cleaned))
        except ValueError:
            normalized = cleaned.upper().replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError as exc:
                raise ValueError(
                    "Invalid timestamp {value!r}; expected Unix seconds or ISO-8601 string"
                    .format(value=value)
                ) from exc
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return int(parsed.timestamp())


def fetch_market_candlesticks(
    *,
    client: KalshiClient,
    reference: MarketReference,
    start_ts: int,
    end_ts: int,
    period_minutes: int,
) -> Mapping[str, object]:
    """Return candlestick data for ``reference`` within the requested window."""

    candlesticks = client.get_market_candlesticks(
        series_ticker=reference.series_ticker,
        market_ticker=reference.market_ticker,
        start_ts=start_ts,
        end_ts=end_ts,
        period_minutes=period_minutes,
    )
    return {
        "series_ticker": reference.series_ticker,
        "market_ticker": reference.market_ticker,
        "start_ts": int(start_ts),
        "end_ts": int(end_ts),
        "period_minutes": int(period_minutes),
        "candlesticks": candlesticks,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "market",
        help=(
            "Kalshi market ticker or URL. If a bare ticker is supplied the series "
            "is inferred from the prefix before the first dash unless --series is "
            "provided."
        ),
    )
    parser.add_argument(
        "--series",
        help="Series ticker containing the market (used when MARKET is not a URL).",
    )
    parser.add_argument("--start", required=True, help="Start timestamp (Unix seconds or ISO-8601).")
    parser.add_argument("--end", required=True, help="End timestamp (Unix seconds or ISO-8601).")
    parser.add_argument(
        "--period",
        type=int,
        default=1440,
        help="Candlestick size in minutes (default: 1440, valid: 1, 60, 1440).",
    )
    parser.add_argument(
        "--auth-token",
        help="Optional Kalshi API bearer token. Public markets do not require authentication.",
    )
    parser.add_argument("--email", help="Kalshi account email (used with --password to authenticate).")
    parser.add_argument("--password", help="Kalshi account password (used with --email to authenticate).")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override the Kalshi API base URL. Defaults to the production endpoint.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds for API requests (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="If provided, write the JSON payload to this path instead of stdout.",
    )
    return parser


def _build_client_from_args(args: argparse.Namespace) -> KalshiClient:
    client_kwargs = {"timeout": args.timeout}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url

    if args.auth_token:
        client_kwargs["auth_token"] = args.auth_token
    elif args.email and args.password:
        client_kwargs["email"] = args.email
        client_kwargs["password"] = args.password

    return KalshiClient(**client_kwargs)


def _validate_period(period: int) -> int:
    if period not in CANDLESTICK_INTERVALS:
        raise ValueError(
            "Invalid candlestick period {period}; expected one of {allowed}".format(
                period=period, allowed=", ".join(str(i) for i in CANDLESTICK_INTERVALS)
            )
        )
    return period


def _serialize_payload(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        reference = MarketReference.parse(args.market, series=args.series)
        start_ts = parse_timestamp(args.start)
        end_ts = parse_timestamp(args.end)
        period = _validate_period(int(args.period))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    client = _build_client_from_args(args)
    payload = fetch_market_candlesticks(
        client=client,
        reference=reference,
        start_ts=start_ts,
        end_ts=end_ts,
        period_minutes=period,
    )

    serialized = _serialize_payload(payload)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
