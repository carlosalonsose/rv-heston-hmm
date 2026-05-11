# Relative Value Binary Pricing Engine

Prototype for pricing binary prediction-market contracts with a regime-filtered
Heston jump-diffusion Monte Carlo model.

The core workflow is:

```text
market history -> short-DTE regime calibration
current BRTI/spot + regime posterior -> Heston jump-diffusion paths
contract rule -> model probability
market bid/ask + fees/slippage/model buffer -> trade signal
```

This is research infrastructure, not a production trading bot.

## Quick Start

```bash
python -m rv_heston_hmm.cli synthetic-demo
```

Example custom run:

```bash
python -m rv_heston_hmm.cli price \
  --spot 100000 \
  --barrier 105000 \
  --days 3 \
  --ask 0.42 \
  --bid 0.39 \
  --paths 100000
```

Run one Kalshi monitoring pass:

```bash
python -m rv_heston_hmm.monitor run \
  --config config/kalshi_watchlist.json \
  --calibration config/model_calibration.json
```

Discover open Kalshi markets:

```bash
python -m rv_heston_hmm.monitor discover --limit 20
```

Run the local dashboard:

```bash
python -m rv_heston_hmm.dashboard
```

Then open `http://127.0.0.1:8765`.

Calibrate the model for short-dated BTC contracts:

```bash
python -m rv_heston_hmm.calibration \
  --mode short_dte \
  --ticker BTC-USD \
  --short-period 7d \
  --short-interval 5m \
  --period 45d \
  --interval 1h \
  --output config/model_calibration.json
```

`short_dte` calibrates the starting variance from 5-minute EWMA realized
variance, estimates intraday jumps from recent 5-minute 6-sigma tails, uses a
5-minute two-regime Markov model for current state and transitions, estimates
Heston `kappa` from AR(1) persistence of realized variance, estimates
`vol_of_vol` from variance-shock moments, and estimates `rho` from the
correlation of returns with realized-variance changes.

The dashboard can scan the full open `KXBTCD` series automatically. Scanner
settings live in `config/model_calibration.json`:

- `scanner.enabled`: turn automatic series scanning on/off.
- `scanner.series`: Kalshi series ticker, default `KXBTCD`.
- `scanner.max_markets`: maximum strikes to scan per refresh.
- `scanner.spot_source`: `cf_brti_with_coinbase_fallback`, `cf_brti_page`, or
  `coinbase_btc_usd`.
- `scanner.min_edge_alert`: minimum net edge for alert cards.
- `scanner.min_liquidity_contracts`: minimum available size on the actionable
  side of the book.
- `scanner.max_spread`: skip very wide markets.

The monitor reads public Kalshi orderbooks, converts the book to YES bid/ask,
simulates the configured event, and prints JSON with model probability, fair
band, edge, and signal.

## What It Implements

- Short-DTE Markov-regime calibration from BTC-USD history.
- Optional synthetic Gaussian HMM demo.
- Heston variance process with correlated spot/variance shocks.
- Poisson lognormal jumps.
- Regime-dependent parameters.
- Terminal and path barrier binary events.
- Monte Carlo standard error.
- Net edge calculation against bid/ask.

## Measure

The simulator currently runs under real-world measure `P`, because the target
use case is held-to-resolution prediction-market EV. Historical drift is kept
in the path dynamics and jump returns are not risk-neutral compensated. If you
want relative value against listed options, add a separate risk-neutral `Q`
mode with an explicit risk-free rate and option-implied calibration.

## Model Notes

The simulator supports two use cases:

- `terminal_above`: event resolves from terminal price, for example
  `BTC > 105000 at expiry`.
- `touches_above`: event resolves if the path ever crosses a barrier before
  expiry.

For live use, the event implementation must match the exchange resolution rule
exactly: timestamp, source exchange, candle convention, closing auction, oracle,
rounding, and fallback rules.

## Limitations

- Heston parameters are short-DTE historical estimates, not an option IV surface
  fit. Extreme `kappa`, `vol_of_vol`, jump intensity, or regime drift values are
  surfaced as calibration warnings instead of being silently clipped.
- Path-touch events do not yet use Brownian-bridge crossing correction.
- The variance scheme uses a simple positivity truncation, not Andersen QE.
- BRTI spot uses a public CF Benchmarks page parser with Coinbase fallback, not
  a licensed real-time CF feed.
- No sizing, position limits, realized PnL attribution, or execution logic is
  included.

## Files

- `hmm.py`: small self-contained Gaussian HMM (used by `synthetic-demo`).
- `calibration.py`: historical and short-DTE calibration routines.
- `model_guardrails.py`: post-calibration bounds enforcement (Feller, kappa, vol\_of\_vol, mu caps).
- `config_loader.py`: shared model/signal/simulation config loader.
- `simulator.py`: regime-switching Heston jump-diffusion Monte Carlo.
- `pricing.py`: event probability and trade edge.
- `kalshi.py`: public Kalshi REST client for market/orderbook data.
- `monitor.py`: one-pass Kalshi edge monitor for scheduled runs.
- `dashboard.py`: local web UI and API server.
- `web/`: dashboard frontend.
- `cli.py`: runnable demo and simple pricing command.
- `tests/`: unit tests (simulator accuracy, calibration estimators, Kalshi parsing, pricing logic).
