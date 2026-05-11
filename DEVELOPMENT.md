# Development

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Common Commands

```bash
python3 -m rv_heston_hmm.calibration --mode short_dte --output config/model_calibration.json
python3 -m rv_heston_hmm.dashboard
python3 -m rv_heston_hmm.monitor run --config config/kalshi_watchlist.json --calibration config/model_calibration.json
python3 -m unittest discover -s tests
```

If installed as a package, the equivalent scripts are:

```bash
rv-calibrate --mode short_dte --output config/model_calibration.json
rv-dashboard
rv-monitor run --config config/kalshi_watchlist.json --calibration config/model_calibration.json
rv price --spot 81400 --barrier 82000 --days 0.05 --bid 0.1 --ask 0.12
```

## Health

The dashboard exposes:

```text
/api/health
```

It reports the last snapshot status, latency, market counts, actionable signal
counts, alert counts, calibration mode, calibration timestamp, and calibration
warnings.

## Calibration Notes

`short_dte` is intended for Kalshi BTC contracts with hours to one day of
horizon. It is a historical real-world measure model, not an option-implied
risk-neutral surface fit.

Extreme estimates are surfaced in `metadata.historical_calibration.warnings`.
Do not silently clip them without documenting the economic reason.

