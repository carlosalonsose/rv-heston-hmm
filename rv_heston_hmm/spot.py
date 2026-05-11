from __future__ import annotations

from dataclasses import dataclass
from urllib.request import Request, urlopen
import json
import logging
import re
import ssl

try:
    import certifi
except ImportError:  # pragma: no cover - optional runtime dependency
    certifi = None


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpotQuote:
    source: str
    price: float
    raw_source: str


def fetch_spot(source: str) -> SpotQuote:
    if source == "cf_brti_page":
        return fetch_cf_brti_page()
    if source == "cf_brti_with_coinbase_fallback":
        try:
            return fetch_cf_brti_page()
        except Exception as exc:
            logger.warning("BRTI public page fetch failed; falling back to Coinbase: %s", exc)
            return fetch_coinbase_btc_usd()
    if source == "coinbase_btc_usd":
        return fetch_coinbase_btc_usd()
    raise ValueError(f"unsupported spot_source: {source}")


def fetch_coinbase_btc_usd() -> SpotQuote:
    data = _get_json("https://api.coinbase.com/v2/prices/BTC-USD/spot")
    return SpotQuote(source="coinbase_btc_usd", price=float(data["data"]["amount"]), raw_source="Coinbase BTC-USD spot")


def fetch_cf_brti_page() -> SpotQuote:
    html = _get_text("https://www.cfbenchmarks.com/data/indices/BRTI")
    patterns = [
        r'"value"\s*:\s*"?([0-9][0-9,]*\.?[0-9]*)"?',
        r'BRTI[^$]{0,300}\$\s*([0-9][0-9,]*\.?[0-9]*)',
        r'\$\s*([0-9][0-9,]*\.[0-9]{2})',
    ]
    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            price = float(match.group(1).replace(",", ""))
            if price > 1_000:
                return SpotQuote(source="cf_brti_page", price=price, raw_source="CF Benchmarks BRTI public page")
    raise ValueError("could not parse BRTI price from CF Benchmarks page")


def _get_json(url: str) -> dict:
    return json.loads(_get_text(url))


def _get_text(url: str) -> str:
    request = Request(url, headers={"Accept": "application/json,text/html", "User-Agent": "rv-heston-hmm/0.1"})
    context = ssl.create_default_context(cafile=certifi.where()) if certifi else None
    with urlopen(request, timeout=10.0, context=context) as response:
        return response.read().decode("utf-8")
