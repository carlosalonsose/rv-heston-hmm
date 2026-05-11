from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import argparse
import json
import logging

from .config_loader import load_model_config, regime_model_from_config, signal_config_from_model, simulation_defaults
from .kalshi import KalshiClient
from .pricing import BinaryEvent, MarketQuote, SignalConfig, price_binary_event
from .simulator import SimulationConfig
from .spot import fetch_spot


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WatchMarket:
    ticker: str
    spot: float | None = None
    barrier: float | None = None
    horizon_days: float | None = None
    event_kind: str = "terminal_above"
    initial_variance: float = 0.50**2
    auto_from_kalshi: bool = True
    spot_source: str | None = None
    title: str | None = None
    yes_bid: float | None = None
    yes_bid_qty: float | None = None
    yes_ask: float | None = None
    yes_ask_qty: float | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Kalshi edge analysis pass.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one configured edge analysis pass.")
    run_parser.add_argument("--config", default="config/kalshi_watchlist.json")
    run_parser.add_argument("--calibration", default="config/model_calibration.json")
    run_parser.add_argument("--paths", type=int, default=None)
    run_parser.add_argument("--fee-rate", type=float, default=None)
    run_parser.add_argument("--model-buffer", type=float, default=None)
    run_parser.add_argument("--slippage", type=float, default=None)
    run_parser.add_argument("--min-edge", type=float, default=None)
    run_parser.add_argument("--min-liquidity", type=float, default=None)

    discover_parser = subparsers.add_parser("discover", help="List open Kalshi markets.")
    discover_parser.add_argument("--series", default=None)
    discover_parser.add_argument("--limit", type=int, default=20)

    args = parser.parse_args()
    if args.command == "discover":
        print(json.dumps(discover_markets(series=args.series, limit=args.limit), indent=2, sort_keys=True))
        return

    calibration = load_model_config(args.calibration)
    signal_config = signal_config_from_model(
        calibration,
        taker_fee_rate=args.fee_rate,
        model_buffer=args.model_buffer,
        slippage=args.slippage,
        min_edge=args.min_edge,
        min_liquidity=args.min_liquidity,
    )
    sim_defaults = simulation_defaults(calibration)
    results = run_once(
        config_path=Path(args.config),
        calibration=calibration,
        paths=args.paths or sim_defaults["paths"],
        signal_config=signal_config,
    )
    print(json.dumps(results, indent=2, sort_keys=True))


def discover_markets(series: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    client = KalshiClient()
    data = client.list_markets(status="open", series_ticker=series, limit=limit)
    markets = data.get("markets", [])
    return [
        {
            "ticker": market.get("ticker"),
            "event_ticker": market.get("event_ticker"),
            "title": market.get("title"),
            "yes_bid": _price_to_prob(market.get("yes_bid_dollars", market.get("yes_bid"))),
            "yes_ask": _price_to_prob(market.get("yes_ask_dollars", market.get("yes_ask"))),
            "last_price": _price_to_prob(market.get("last_price_dollars", market.get("last_price"))),
            "volume": market.get("volume_fp", market.get("volume")),
            "close_time": market.get("close_time"),
            "expiration_time": market.get("expiration_time"),
        }
        for market in markets
    ]


def run_once(
    *,
    config_path: Path,
    calibration: dict[str, Any],
    paths: int,
    signal_config: SignalConfig,
) -> list[dict[str, Any]]:
    client = KalshiClient()
    watchlist = _load_watchlist(config_path, client)
    regimes = regime_model_from_config(calibration)
    sim_defaults = simulation_defaults(calibration)
    timestamp = datetime.now(timezone.utc).isoformat()
    output: list[dict[str, Any]] = []

    for market in watchlist:
        book = client.get_book_top(market.ticker)
        if book.yes_bid is None or book.yes_ask is None:
            output.append(
                {
                    "timestamp": timestamp,
                    "ticker": market.ticker,
                    "status": "NO_BOOK",
                    "reason": "missing yes bid or yes ask",
                }
            )
            continue

        if market.barrier is None or market.horizon_days is None:
            output.append(
                {
                    "timestamp": timestamp,
                    "ticker": market.ticker,
                    "status": "BAD_CONFIG",
                    "reason": "missing barrier or horizon_days",
                }
            )
            continue
        if market.horizon_days < 5.0 / 1440.0:
            output.append(
                {
                    "timestamp": timestamp,
                    "ticker": market.ticker,
                    "status": "EXPIRED",
                    "reason": "horizon is below 5 minutes",
                    "horizon_days": market.horizon_days,
                }
            )
            continue

        spot = market.spot
        if spot is None and market.spot_source:
            spot = _fetch_spot(market.spot_source)
        if spot is None:
            output.append(
                {
                    "timestamp": timestamp,
                    "ticker": market.ticker,
                    "status": "BAD_CONFIG",
                    "reason": "missing spot or spot_source",
                }
            )
            continue

        quote = MarketQuote(
            bid_yes=book.yes_bid,
            ask_yes=book.yes_ask,
            yes_bid_qty=book.yes_bid_qty,
            yes_ask_qty=book.yes_ask_qty,
        )
        event = BinaryEvent(kind=market.event_kind, barrier=market.barrier)
        simulation = SimulationConfig(
            spot=spot,
            initial_variance=market.initial_variance,
            horizon_days=market.horizon_days,
            paths=paths,
            steps_per_day=sim_defaults["steps_per_day"],
            annualization_days=sim_defaults["annualization_days"],
            random_seed=None,
            store_paths=False,
        )
        priced = price_binary_event(event, quote, signal_config, simulation, regimes)
        if priced.status != "OK":
            output.append(
                {
                    "timestamp": timestamp,
                    "ticker": market.ticker,
                    "status": priced.status,
                    "yes_bid": book.yes_bid,
                    "yes_ask": book.yes_ask,
                }
            )
            continue
        output.append(
            {
                "timestamp": timestamp,
                "ticker": market.ticker,
                "status": "OK",
                "spot": spot,
                "barrier": market.barrier,
                "event_kind": market.event_kind,
                "horizon_days": market.horizon_days,
                "yes_bid": book.yes_bid,
                "yes_bid_qty": book.yes_bid_qty,
                "yes_ask": book.yes_ask,
                "yes_ask_qty": book.yes_ask_qty,
                "model_probability": priced.model_probability,
                "monte_carlo_se": priced.monte_carlo_se,
                "fair_bid": priced.fair_bid,
                "fair_ask": priced.fair_ask,
                "statistical_fair_bid": priced.statistical_fair_bid,
                "statistical_fair_ask": priced.statistical_fair_ask,
                "buy_yes_edge": priced.buy_yes_edge,
                "sell_yes_edge": priced.sell_yes_edge,
                "signal": priced.signal,
            }
        )

    return output


def _load_watchlist(config_path: Path, client: KalshiClient) -> list[WatchMarket]:
    data = json.loads(config_path.read_text())
    markets = data.get("markets", [])
    if not markets:
        raise ValueError(f"No markets configured in {config_path}")
    loaded = []
    for item in markets:
        ticker = str(item["ticker"])
        auto_from_kalshi = bool(item.get("auto_from_kalshi", True)) and not ticker.startswith("REPLACE")
        kalshi_market = client.get_market(ticker) if auto_from_kalshi else {}
        barrier = item.get("barrier")
        if barrier is None:
            barrier = kalshi_market.get("floor_strike") or kalshi_market.get("cap_strike")
        horizon_days = item.get("horizon_days")
        if horizon_days is None:
            close_time = kalshi_market.get("close_time")
            if close_time:
                horizon_days = _days_until(close_time)
        try:
            event_kind = str(item.get("event_kind") or _event_kind_from_kalshi(kalshi_market))
        except ValueError as exc:
            logger.warning("Skipping unsupported Kalshi market %s: %s", ticker, exc)
            continue

        loaded.append(
            WatchMarket(
                ticker=ticker,
                spot=None if item.get("spot") is None else float(item["spot"]),
                barrier=None if barrier is None else float(barrier),
                horizon_days=None if horizon_days is None else float(horizon_days),
                event_kind=event_kind,
                initial_variance=float(item.get("initial_variance", 0.50**2)),
                auto_from_kalshi=auto_from_kalshi,
                spot_source=item.get("spot_source"),
                title=item.get("title"),
            )
        )
    return loaded


def _price_to_prob(value: Any) -> float | None:
    if value is None:
        return None
    price = float(value)
    if price > 1.0:
        return price / 100.0
    return price


def _days_until(iso_time: str) -> float:
    close_dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    return max((close_dt - now).total_seconds() / 86400.0, 0.0)


def _event_kind_from_kalshi(market: dict[str, Any]) -> str:
    strike_type = market.get("strike_type")
    if strike_type == "greater":
        return "terminal_above"
    if strike_type == "less":
        return "terminal_below"
    if strike_type in {"between", "range"}:
        raise ValueError(f"unsupported Kalshi range strike_type: {strike_type}")
    logger.warning("Unknown Kalshi strike_type=%s for ticker=%s", strike_type, market.get("ticker"))
    raise ValueError(f"unknown Kalshi strike_type: {strike_type}")


def _fetch_spot(source: str) -> float:
    return fetch_spot(source).price


if __name__ == "__main__":
    main()
