from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json
import random
import ssl
import time

try:
    import certifi
except ImportError:  # pragma: no cover - optional runtime dependency
    certifi = None


KALSHI_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"


@dataclass(frozen=True)
class KalshiBookTop:
    ticker: str
    yes_bid: float | None
    yes_bid_qty: float | None
    yes_ask: float | None
    yes_ask_qty: float | None


class KalshiClient:
    def __init__(self, base_url: str = KALSHI_BASE_URL, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_market(self, ticker: str) -> dict[str, Any]:
        return self._get(f"/markets/{ticker}").get("market", {})

    def list_markets(
        self,
        *,
        status: str = "open",
        series_ticker: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
        mve_filter: str | None = "exclude",
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"status": status, "limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        if mve_filter:
            params["mve_filter"] = mve_filter
        return self._get("/markets", params)

    def get_orderbook(self, ticker: str, depth: int = 1) -> dict[str, Any]:
        return self._get(f"/markets/{ticker}/orderbook", {"depth": depth})

    def get_book_top(self, ticker: str) -> KalshiBookTop:
        data = self.get_orderbook(ticker, depth=1)
        book = data.get("orderbook_fp") or data.get("orderbook") or {}
        yes_levels = book.get("yes_dollars") or book.get("yes") or []
        no_levels = book.get("no_dollars") or book.get("no") or []

        yes_bid, yes_bid_qty = _best_level(yes_levels)
        no_bid, no_bid_qty = _best_level(no_levels)
        yes_ask = None if no_bid is None else 1.0 - no_bid
        yes_ask_qty = no_bid_qty

        if yes_bid is None or yes_ask is None:
            market = self.get_market(ticker)
            yes_bid = _parse_optional_price(market.get("yes_bid_dollars", market.get("yes_bid")))
            yes_ask = _parse_optional_price(market.get("yes_ask_dollars", market.get("yes_ask")))
            yes_bid_qty = _parse_optional_float(market.get("yes_bid_size_fp", market.get("yes_bid_size")))
            yes_ask_qty = _parse_optional_float(market.get("yes_ask_size_fp", market.get("yes_ask_size")))

        return KalshiBookTop(
            ticker=ticker,
            yes_bid=yes_bid,
            yes_bid_qty=yes_bid_qty,
            yes_ask=yes_ask,
            yes_ask_qty=yes_ask_qty,
        )

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urlencode(params, doseq=True)}"
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "rv-heston-hmm/0.1"})
        context = ssl.create_default_context(cafile=certifi.where()) if certifi else None
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urlopen(request, timeout=self.timeout, context=context) as response:
                    payload = response.read().decode("utf-8")
                return json.loads(payload)
            except Exception as exc:
                last_error = exc
                if attempt == 2:
                    break
                time.sleep(0.25 * (2**attempt) + random.random() * 0.15)
        raise last_error


def _best_level(levels: list[Any]) -> tuple[float | None, float | None]:
    if not levels:
        return None, None
    price_raw, qty_raw = levels[0][0], levels[0][1]
    price = _parse_price(price_raw)
    qty = float(qty_raw)
    return price, qty


def _parse_price(value: Any) -> float:
    price = float(value)
    if price > 1.0:
        return price / 100.0
    return price


def _parse_optional_price(value: Any) -> float | None:
    if value is None:
        return None
    return _parse_price(value)


def _parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
