from __future__ import annotations

from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
import argparse
import json
import logging
import mimetypes
import time

import numpy as np

from .calibration import calibrate_from_history, calibrate_short_dte, merge_into_model_config
from .config_loader import regime_model_from_config, signal_config_from_model, simulation_defaults
from .kalshi import KalshiClient
from .monitor import WatchMarket, _event_kind_from_kalshi, _load_watchlist
from .pricing import BinaryEvent, MarketQuote, SignalConfig, _binary_fee
from .simulator import RegimeModel, SimulationConfig, simulate_paths
from .spot import SpotQuote, fetch_spot


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web"
DEFAULT_WATCHLIST = PROJECT_ROOT / "config" / "kalshi_watchlist.json"
DEFAULT_CALIBRATION = PROJECT_ROOT / "config" / "model_calibration.json"
LAST_SNAPSHOT_STATS: dict[str, Any] = {
    "status": "STARTING",
    "latency_ms": None,
    "markets": 0,
    "actionable": 0,
    "alerts": 0,
    "error": None,
    "timestamp_utc": None,
}
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Kalshi edge dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--watchlist", default=str(DEFAULT_WATCHLIST))
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    handler = _make_handler(Path(args.watchlist), Path(args.calibration))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    logger.info("Dashboard running at http://%s:%s", args.host, args.port)
    server.serve_forever()


def _make_handler(watchlist_path: Path, calibration_path: Path):
    class DashboardHandler(SimpleHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/snapshot":
                self._send_json(_snapshot(watchlist_path, calibration_path))
                return
            if parsed.path == "/api/health":
                self._send_json(_health(calibration_path))
                return
            if parsed.path == "/api/distribution":
                params = parse_qs(parsed.query)
                ticker = params.get("ticker", [""])[0]
                self._send_json(_distribution(ticker, watchlist_path, calibration_path))
                return
            if parsed.path == "/api/calibration":
                self._send_json(_read_json(calibration_path))
                return
            if parsed.path == "/api/watchlist":
                self._send_json(_read_json(watchlist_path))
                return
            self._serve_static(parsed.path)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if parsed.path == "/api/calibration":
                _write_json(calibration_path, payload)
                self._send_json({"ok": True})
                return
            if parsed.path == "/api/watchlist":
                _write_json(watchlist_path, payload)
                self._send_json({"ok": True})
                return
            if parsed.path == "/api/recalibrate":
                mode = str(payload.get("mode", "short_dte"))
                ticker = str(payload.get("ticker", "BTC-USD"))
                if mode == "historical":
                    period = str(payload.get("period", "45d"))
                    interval = str(payload.get("interval", "1h"))
                    historical = calibrate_from_history(ticker=ticker, period=period, interval=interval)
                else:
                    historical = calibrate_short_dte(
                        ticker=ticker,
                        short_period=str(payload.get("short_period", "7d")),
                        short_interval=str(payload.get("short_interval", "5m")),
                        long_period=str(payload.get("long_period", "45d")),
                        long_interval=str(payload.get("long_interval", "1h")),
                    )
                merged = merge_into_model_config(calibration_path, historical)
                _write_json(calibration_path, merged)
                self._send_json({"ok": True, "metadata": merged.get("metadata", {}).get("historical_calibration", {})})
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _serve_static(self, request_path: str) -> None:
            rel = "index.html" if request_path in ("", "/") else request_path.lstrip("/")
            target = (WEB_ROOT / rel).resolve()
            if not str(target).startswith(str(WEB_ROOT.resolve())) or not target.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return
            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            body = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: Any) -> None:
            body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return DashboardHandler


def _snapshot(watchlist_path: Path, calibration_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        calibration = _read_json(calibration_path)
        client = KalshiClient()
        markets = _combined_markets(client, watchlist_path, calibration)
        signal = signal_config_from_model(calibration)
        regimes = regime_model_from_config(calibration)
        sim_defaults = simulation_defaults(calibration)
        rows = _price_markets(client, markets, signal, regimes, sim_defaults)

        actionable = [row for row in rows if row.get("signal") in ("BUY_YES", "SELL_YES_OR_BUY_NO")]
        scanner_config = calibration.get("scanner", {})
        min_alert_edge = float(scanner_config.get("min_edge_alert", signal.min_edge))
        min_liquidity = float(scanner_config.get("min_liquidity_contracts", 0.0))
        alerts = [row for row in actionable if _is_alert(row, min_alert_edge, min_liquidity)]
        payload = {
            "calibration": calibration,
            "markets": rows,
            "actionable": sorted(actionable, key=lambda row: max(row.get("buy_yes_edge", 0), row.get("sell_yes_edge", 0)), reverse=True),
            "alerts": sorted(alerts, key=lambda row: max(row.get("buy_yes_edge", 0), row.get("sell_yes_edge", 0)), reverse=True),
        }
        _record_snapshot_stats("OK", started, len(rows), len(actionable), len(alerts), None)
        return payload
    except Exception as exc:
        _record_snapshot_stats("ERROR", started, 0, 0, 0, str(exc))
        raise


def _health(calibration_path: Path) -> dict[str, Any]:
    calibration = _read_json(calibration_path)
    metadata = calibration.get("metadata", {}).get("historical_calibration", {})
    return {
        "status": LAST_SNAPSHOT_STATS["status"],
        "snapshot": LAST_SNAPSHOT_STATS,
        "calibration_mode": calibration.get("calibration_mode"),
        "calibration_timestamp_utc": metadata.get("timestamp_utc"),
        "calibration_warnings": metadata.get("warnings", []),
    }


def _record_snapshot_stats(
    status: str,
    started: float,
    markets: int,
    actionable: int,
    alerts: int,
    error: str | None,
) -> None:
    LAST_SNAPSHOT_STATS.update(
        {
            "status": status,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "markets": markets,
            "actionable": actionable,
            "alerts": alerts,
            "error": error,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
    )


def _distribution(ticker: str, watchlist_path: Path, calibration_path: Path) -> dict[str, Any]:
    if not ticker:
        return {"status": "ERROR", "reason": "missing ticker"}

    calibration = _read_json(calibration_path)
    client = KalshiClient()
    markets = _combined_markets(client, watchlist_path, calibration)
    market = next((item for item in markets if item.ticker == ticker), None)
    if market is None:
        return {"status": "ERROR", "reason": f"{ticker} is not in watchlist"}

    spot_quote = _market_spot_quote(market)
    spot = None if spot_quote is None else spot_quote.price
    if spot is None or market.barrier is None or market.horizon_days is None:
        return {"status": "ERROR", "reason": "market is missing spot, barrier, or horizon"}

    sim_defaults = calibration.get("simulation", {})
    simulation = SimulationConfig(
        spot=spot,
        initial_variance=market.initial_variance,
        horizon_days=market.horizon_days,
        paths=int(min(sim_defaults.get("paths", 100_000), 150_000)),
        steps_per_day=int(sim_defaults.get("steps_per_day", 24)),
        annualization_days=int(sim_defaults.get("annualization_days", 365)),
        random_seed=None,
        store_paths=False,
    )
    sim = simulate_paths(simulation, regime_model_from_config(calibration))
    event = BinaryEvent(kind=market.event_kind, barrier=market.barrier)
    hits = event.evaluate(sim.terminal, sim.path_max, sim.path_min)
    hist_counts, hist_edges = np.histogram(sim.terminal, bins=48)
    percentiles = np.percentile(sim.terminal, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    return {
        "status": "OK",
        "ticker": ticker,
        "spot": spot,
        "barrier": market.barrier,
        "event_kind": market.event_kind,
        "horizon_days": market.horizon_days,
        "model_probability": float(np.mean(hits)),
        "histogram": {
            "counts": hist_counts.astype(int).tolist(),
            "edges": hist_edges.astype(float).tolist(),
        },
        "percentiles": {
            "p01": float(percentiles[0]),
            "p05": float(percentiles[1]),
            "p10": float(percentiles[2]),
            "p25": float(percentiles[3]),
            "p50": float(percentiles[4]),
            "p75": float(percentiles[5]),
            "p90": float(percentiles[6]),
            "p95": float(percentiles[7]),
            "p99": float(percentiles[8]),
        },
    }


def _combined_markets(client: KalshiClient, watchlist_path: Path, calibration: dict[str, Any]) -> list[WatchMarket]:
    markets = []
    if calibration.get("scanner", {}).get("enabled", False):
        markets.extend(_scan_series(client, calibration["scanner"]))
    markets.extend(_load_watchlist(watchlist_path, client))
    deduped: dict[str, WatchMarket] = {}
    for market in markets:
        deduped[market.ticker] = market
    return list(deduped.values())


def _scan_series(client: KalshiClient, scanner: dict[str, Any]) -> list[WatchMarket]:
    series = scanner.get("series", "KXBTCD")
    max_markets = int(scanner.get("max_markets", 80))
    max_spread = float(scanner.get("max_spread", 1.0))
    spot_source = scanner.get("spot_source", "cf_brti_with_coinbase_fallback")
    initial_variance = float(scanner.get("initial_variance", 0.50**2))
    scanned: list[WatchMarket] = []
    cursor = None
    now = datetime.now(timezone.utc)

    while len(scanned) < max_markets:
        data = client.list_markets(status="open", series_ticker=series, limit=min(100, max_markets), cursor=cursor)
        for item in data.get("markets", []):
            yes_bid = _price_to_prob(item.get("yes_bid_dollars", item.get("yes_bid")))
            yes_ask = _price_to_prob(item.get("yes_ask_dollars", item.get("yes_ask")))
            if yes_bid is None or yes_ask is None:
                continue
            if yes_ask - yes_bid > max_spread:
                continue
            barrier = item.get("floor_strike") or item.get("cap_strike")
            close_time = item.get("close_time")
            if barrier is None or not close_time:
                continue
            close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            horizon_days = (close_dt - now).total_seconds() / 86400.0
            if horizon_days < 5.0 / 1440.0:
                continue
            try:
                event_kind = _event_kind_from_kalshi(item)
            except ValueError:
                continue
            scanned.append(
                WatchMarket(
                    ticker=str(item["ticker"]),
                    spot_source=spot_source,
                    barrier=float(barrier),
                    horizon_days=horizon_days,
                    event_kind=event_kind,
                    initial_variance=initial_variance,
                    auto_from_kalshi=False,
                    title=item.get("title"),
                    yes_bid=yes_bid,
                    yes_bid_qty=_optional_float(item.get("yes_bid_size_fp", item.get("yes_bid_size"))),
                    yes_ask=yes_ask,
                    yes_ask_qty=_optional_float(item.get("yes_ask_size_fp", item.get("yes_ask_size"))),
                )
            )
            if len(scanned) >= max_markets:
                break
        cursor = data.get("cursor")
        if not cursor:
            break
    return scanned


def _price_markets(
    client: KalshiClient,
    markets: list[WatchMarket],
    signal: SignalConfig,
    regimes: RegimeModel,
    sim_defaults: dict[str, Any],
) -> list[dict[str, Any]]:
    prepared = []
    rows: list[dict[str, Any]] = []
    spot_cache: dict[str, Any] = {}
    for market in markets:
        try:
            quote = _market_quote(client, market)
            if quote is None:
                rows.append({"ticker": market.ticker, "status": "NO_BOOK", "reason": "missing yes bid or yes ask"})
                continue
            spot_quote = _market_spot_quote(market, spot_cache)
            if spot_quote is None or market.barrier is None or market.horizon_days is None:
                rows.append({"ticker": market.ticker, "status": "BAD_CONFIG", "reason": "missing spot, barrier, or horizon_days"})
                continue
            if market.horizon_days < 5.0 / 1440.0:
                rows.append(
                    {
                        "ticker": market.ticker,
                        "status": "EXPIRED",
                        "reason": "horizon is below 5 minutes",
                        "spot": spot_quote.price,
                        "barrier": market.barrier,
                        "horizon_days": market.horizon_days,
                        "signal": "EXPIRED",
                    }
                )
                continue
            prepared.append((market, quote, spot_quote))
        except Exception as exc:
            rows.append({"ticker": market.ticker, "status": "ERROR", "reason": str(exc)})

    groups: dict[tuple[Any, ...], list[tuple[WatchMarket, Any, Any]]] = {}
    for market, quote, spot_quote in prepared:
        key = (
            round(spot_quote.price, 2),
            round(float(market.horizon_days), 8),
            float(market.initial_variance),
            market.event_kind,
            spot_quote.source,
        )
        groups.setdefault(key, []).append((market, quote, spot_quote))

    for group in groups.values():
        sample_market, _, sample_spot = group[0]
        simulation = SimulationConfig(
            spot=sample_spot.price,
            initial_variance=sample_market.initial_variance,
            horizon_days=float(sample_market.horizon_days),
            paths=int(sim_defaults.get("paths", 100_000)),
            steps_per_day=int(sim_defaults.get("steps_per_day", 24)),
            annualization_days=int(sim_defaults.get("annualization_days", 365)),
            random_seed=None,
            store_paths=False,
        )
        sim = simulate_paths(simulation, regimes)
        for market, quote, spot_quote in group:
            rows.append(_priced_row(market, quote, spot_quote, sim, signal))

    return sorted(rows, key=lambda row: (row.get("status") != "OK", row.get("barrier") or 0.0))


def _priced_row(market: WatchMarket, quote: Any, spot_quote: Any, sim: Any, signal: SignalConfig) -> dict[str, Any]:
    quote_status = quote.validate()
    if quote_status != "OK":
        return {
            "ticker": market.ticker,
            "title": market.title,
            "status": quote_status,
            "spot": spot_quote.price,
            "spot_source": spot_quote.source,
            "barrier": market.barrier,
            "yes_bid": quote.bid_yes,
            "yes_ask": quote.ask_yes,
            "signal": quote_status,
        }
    event = BinaryEvent(kind=market.event_kind, barrier=float(market.barrier))
    hits = event.evaluate(sim.terminal, sim.path_max, sim.path_min)
    p_model = float(np.mean(hits))
    n = int(hits.size)
    mc_se = float(np.sqrt(max(p_model * (1.0 - p_model), 0.0) / n))
    buy_fee = _binary_fee(quote.ask_yes, signal.taker_fee_rate)
    sell_fee = _binary_fee(quote.bid_yes, signal.taker_fee_rate)
    statistical_band = 2.0 * mc_se
    economic_haircut = signal.model_buffer + statistical_band
    buy_yes_edge = p_model - quote.ask_yes - buy_fee - signal.slippage - economic_haircut
    sell_yes_edge = quote.bid_yes - p_model - sell_fee - signal.slippage - economic_haircut
    has_buy_liquidity = quote.yes_ask_qty is not None and quote.yes_ask_qty >= signal.min_liquidity
    has_sell_liquidity = quote.yes_bid_qty is not None and quote.yes_bid_qty >= signal.min_liquidity
    if has_buy_liquidity and buy_yes_edge >= signal.min_edge and buy_yes_edge > sell_yes_edge:
        signal_name = "BUY_YES"
    elif has_sell_liquidity and sell_yes_edge >= signal.min_edge:
        signal_name = "SELL_YES_OR_BUY_NO"
    else:
        signal_name = "NO_TRADE"
    return {
        "ticker": market.ticker,
        "title": market.title,
        "status": "OK",
        "spot": spot_quote.price,
        "spot_source": spot_quote.source,
        "spot_raw_source": spot_quote.raw_source,
        "barrier": market.barrier,
        "event_kind": market.event_kind,
        "horizon_days": market.horizon_days,
        "initial_variance": market.initial_variance,
        "yes_bid": quote.bid_yes,
        "yes_bid_qty": quote.yes_bid_qty,
        "yes_ask": quote.ask_yes,
        "yes_ask_qty": quote.yes_ask_qty,
        "spread": quote.ask_yes - quote.bid_yes,
        "buy_fee": buy_fee,
        "sell_fee": sell_fee,
        "slippage": signal.slippage,
        "model_buffer": signal.model_buffer,
        "model_probability": p_model,
        "monte_carlo_se": mc_se,
        "fair_bid": max(0.0, p_model - economic_haircut),
        "fair_ask": min(1.0, p_model + economic_haircut),
        "statistical_fair_bid": max(0.0, p_model - statistical_band),
        "statistical_fair_ask": min(1.0, p_model + statistical_band),
        "buy_yes_edge": float(buy_yes_edge),
        "sell_yes_edge": float(sell_yes_edge),
        "edge_net": float(max(buy_yes_edge, sell_yes_edge)),
        "signal": signal_name,
    }


def _market_quote(client: KalshiClient, market: WatchMarket) -> Any | None:
    if market.yes_bid is not None and market.yes_ask is not None:
        return MarketQuote(
            bid_yes=market.yes_bid,
            ask_yes=market.yes_ask,
            yes_bid_qty=market.yes_bid_qty,
            yes_ask_qty=market.yes_ask_qty,
        )
    book = client.get_book_top(market.ticker)
    if book.yes_bid is None or book.yes_ask is None:
        return None
    return MarketQuote(
        bid_yes=book.yes_bid,
        ask_yes=book.yes_ask,
        yes_bid_qty=book.yes_bid_qty,
        yes_ask_qty=book.yes_ask_qty,
    )


def _market_spot_quote(market: Any, spot_cache: dict[str, Any] | None = None) -> Any | None:
    if market.spot is not None:
        return SpotQuote(source="manual", price=float(market.spot), raw_source="manual watchlist spot")
    if market.spot_source:
        if spot_cache is not None and market.spot_source in spot_cache:
            return spot_cache[market.spot_source]
        quote = fetch_spot(market.spot_source)
        if spot_cache is not None:
            spot_cache[market.spot_source] = quote
        return quote
    return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _price_to_prob(value: Any) -> float | None:
    if value is None:
        return None
    price = float(value)
    if price > 1.0:
        return price / 100.0
    return price


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _is_alert(row: dict[str, Any], min_edge: float, min_liquidity: float) -> bool:
    if row.get("signal") == "BUY_YES":
        return row.get("buy_yes_edge", 0.0) >= min_edge and (row.get("yes_ask_qty") or 0.0) >= min_liquidity
    if row.get("signal") == "SELL_YES_OR_BUY_NO":
        return row.get("sell_yes_edge", 0.0) >= min_edge and (row.get("yes_bid_qty") or 0.0) >= min_liquidity
    return False


if __name__ == "__main__":
    main()
