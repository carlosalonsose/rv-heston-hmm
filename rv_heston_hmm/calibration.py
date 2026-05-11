from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import argparse
import json

import numpy as np

from .model_guardrails import apply_regime_guardrails

DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "config" / "model_calibration.json"


@dataclass(frozen=True)
class HistoricalCalibration:
    mode: str
    ticker: str
    period: str
    interval: str
    observations: int
    timestamp_utc: str
    mu_hist: float
    sigma_hist: float
    jump_intensity: float
    jump_mean: float
    jump_std: float
    transition: list[list[float]]
    initial_prob: list[float]
    regime_params: list[dict[str, Any]]
    initial_variance: float
    realized_vol_1h: float | None = None
    realized_vol_4h: float | None = None
    realized_vol_24h: float | None = None
    kappa_estimate: float | None = None
    vol_of_vol_estimate: float | None = None
    rho_estimate: float | None = None
    warnings: list[str] | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Heston jump-regime parameters from BTC history.")
    parser.add_argument("--mode", choices=["historical", "short_dte"], default="short_dte")
    parser.add_argument("--ticker", default="BTC-USD")
    parser.add_argument("--period", default="45d")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--short-period", default="7d")
    parser.add_argument("--short-interval", default="5m")
    parser.add_argument("--output", default=str(DEFAULT_CALIBRATION_PATH))
    args = parser.parse_args()

    if args.mode == "short_dte":
        calibration = calibrate_short_dte(
            ticker=args.ticker,
            short_period=args.short_period,
            short_interval=args.short_interval,
            long_period=args.period,
            long_interval=args.interval,
        )
    else:
        calibration = calibrate_from_history(ticker=args.ticker, period=args.period, interval=args.interval)
    output_path = Path(args.output)
    merged = merge_into_model_config(output_path, calibration)
    output_path.write_text(json.dumps(merged, indent=2) + "\n")
    print(json.dumps({"ok": True, "metadata": merged["metadata"]["historical_calibration"]}, indent=2))


def calibrate_from_history(
    *,
    ticker: str = "BTC-USD",
    period: str = "45d",
    interval: str = "1h",
) -> HistoricalCalibration:
    yf, sm = _import_dependencies()
    data = yf.Ticker(ticker).history(period=period, interval=interval)
    if data.empty or "Close" not in data:
        raise ValueError(f"No historical close data returned for {ticker}.")

    log_ret = np.log(data["Close"] / data["Close"].shift(1)).dropna()
    if len(log_ret) < 100:
        raise ValueError("Need at least 100 return observations for calibration.")

    periods_per_year = _periods_per_year(interval)
    mu_hist = float(log_ret.mean() * periods_per_year)
    sigma_hist = float(log_ret.std() * np.sqrt(periods_per_year))

    threshold = 3.0 * float(log_ret.std())
    jumps = log_ret[np.abs(log_ret) > threshold]
    jump_intensity = float((len(jumps) / len(log_ret)) * periods_per_year)
    jump_mean = float(jumps.mean()) if len(jumps) else 0.0
    jump_std = float(jumps.std()) if len(jumps) > 1 else max(float(log_ret.std()), 1e-6)

    regime_fit = _fit_markov_regimes(log_ret, periods_per_year, sm)
    transition = regime_fit["transition"]
    initial_prob = regime_fit["initial_prob"]
    mu_low = regime_fit["mu_low"]
    mu_high = regime_fit["mu_high"]
    sigma_low = regime_fit["sigma_low"]
    sigma_high = regime_fit["sigma_high"]

    p_bull_to_bear = max(float(transition[0, 1]), 1e-10)
    p_bear_to_bull = max(float(transition[1, 0]), 1e-10)
    dt_years = 1.0 / periods_per_year
    intensity_bull_to_bear = -np.log(max(float(transition[0, 0]), 1e-10)) / dt_years
    intensity_bear_to_bull = -np.log(max(float(transition[1, 1]), 1e-10)) / dt_years
    kappa = float(intensity_bull_to_bear + intensity_bear_to_bull)

    denom = p_bull_to_bear + p_bear_to_bull
    stationary_low = p_bear_to_bull / denom
    stationary_high = 1.0 - stationary_low
    theta_global = float(stationary_low * sigma_low**2 + stationary_high * sigma_high**2)
    vol_of_vol = float(np.clip(abs(sigma_high - sigma_low) * np.sqrt(max(kappa, 1e-12)), 0.1, 1.0))
    rho = -0.5

    regime_params = [
        {
            "name": "bull_low_vol",
            "mu": float(mu_low),
            "kappa": kappa,
            "theta": float(sigma_low**2),
            "vol_of_vol": vol_of_vol,
            "rho": rho,
            "jump_intensity": jump_intensity,
            "jump_mean": jump_mean,
            "jump_std": jump_std,
        },
        {
            "name": "bear_high_vol",
            "mu": float(mu_high),
            "kappa": kappa,
            "theta": float(sigma_high**2),
            "vol_of_vol": vol_of_vol,
            "rho": rho,
            "jump_intensity": jump_intensity,
            "jump_mean": jump_mean,
            "jump_std": jump_std,
        },
    ]
    regime_params, guardrail_warnings = apply_regime_guardrails(regime_params, fallback_mu=mu_hist)

    return HistoricalCalibration(
        mode="historical",
        ticker=ticker,
        period=period,
        interval=interval,
        observations=int(len(log_ret)),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        mu_hist=mu_hist,
        sigma_hist=sigma_hist,
        jump_intensity=float(regime_params[0]["jump_intensity"]),
        jump_mean=jump_mean,
        jump_std=jump_std,
        transition=transition.tolist(),
        initial_prob=initial_prob.tolist(),
        regime_params=regime_params,
        initial_variance=float(sigma_hist**2),
        kappa_estimate=float(regime_params[0]["kappa"]),
        vol_of_vol_estimate=float(regime_params[0]["vol_of_vol"]),
        warnings=guardrail_warnings,
    )


def calibrate_short_dte(
    *,
    ticker: str = "BTC-USD",
    short_period: str = "7d",
    short_interval: str = "5m",
    long_period: str = "45d",
    long_interval: str = "1h",
    ewma_lambda: float = 0.94,
) -> HistoricalCalibration:
    yf, sm = _import_dependencies()
    short_data = yf.Ticker(ticker).history(period=short_period, interval=short_interval)
    long_data = yf.Ticker(ticker).history(period=long_period, interval=long_interval)
    if short_data.empty or long_data.empty:
        raise ValueError(f"No historical data returned for {ticker}.")

    short_ret = np.log(short_data["Close"] / short_data["Close"].shift(1)).dropna()
    long_ret = np.log(long_data["Close"] / long_data["Close"].shift(1)).dropna()
    if len(short_ret) < 200:
        raise ValueError("Need at least 200 short-interval observations for short_dte calibration.")
    if len(long_ret) < 100:
        raise ValueError("Need at least 100 long-interval observations for short_dte calibration.")

    short_ppy = _periods_per_year(short_interval)
    long_ppy = _periods_per_year(long_interval)
    warnings: list[str] = []
    long_mu = float(long_ret.mean() * long_ppy)
    long_sigma = float(long_ret.std() * np.sqrt(long_ppy))
    initial_variance = _ewma_variance(short_ret.to_numpy(dtype=float), ewma_lambda) * short_ppy
    initial_variance = float(np.clip(initial_variance, 0.10**2, 1.75**2))

    threshold = 6.0 * float(short_ret.std())
    jumps = short_ret[np.abs(short_ret) >= threshold]
    jump_intensity = float((len(jumps) / len(short_ret)) * short_ppy)
    if jump_intensity > 50.0:
        warnings.append(f"jump_intensity is high ({jump_intensity:.2f}/year) with 6-sigma threshold")
    jump_mean = float(jumps.mean()) if len(jumps) else 0.0
    jump_std = float(jumps.std()) if len(jumps) > 1 else max(float(short_ret.std()), 1e-6)

    regime_fit = _fit_markov_regimes(short_ret, short_ppy, sm)
    transition = regime_fit["transition"]
    initial_prob = regime_fit["initial_prob"]
    mu_low = regime_fit["mu_low"]
    mu_high = regime_fit["mu_high"]
    sigma_low = regime_fit["sigma_low"]
    sigma_high = regime_fit["sigma_high"]

    ret_array = short_ret.to_numpy(dtype=float)
    rv_window = max(3, _steps_for_hours(short_interval, 1))
    variance_series = _rolling_realized_variance(ret_array, rv_window, short_ppy)
    theta_global = float(np.mean(variance_series)) if len(variance_series) else long_sigma**2
    variance_ppy = max(1, int(round(short_ppy / rv_window)))
    kappa, kappa_warnings = _estimate_kappa_from_variance(variance_series, variance_ppy)
    warnings.extend(kappa_warnings)
    vol_of_vol, vov_warnings = _estimate_vol_of_vol(variance_series, kappa, theta_global, variance_ppy)
    warnings.extend(vov_warnings)
    rho, rho_warnings = _estimate_rho(ret_array, variance_series, rv_window)
    warnings.extend(rho_warnings)
    theta_low = float(max(sigma_low**2, 1e-8))
    theta_high = float(max(sigma_high**2, 1e-8))
    if abs(mu_low) > 5.0 or abs(mu_high) > 5.0:
        warnings.append(f"regime drift is extreme: mu_low={mu_low:.3f}, mu_high={mu_high:.3f}")

    regime_params = [
        {
            "name": "short_dte_low_vol",
            "mu": float(mu_low),
            "kappa": kappa,
            "theta": theta_low,
            "vol_of_vol": vol_of_vol,
            "rho": rho,
            "jump_intensity": jump_intensity,
            "jump_mean": jump_mean,
            "jump_std": jump_std,
        },
        {
            "name": "short_dte_high_vol",
            "mu": float(mu_high),
            "kappa": kappa,
            "theta": theta_high,
            "vol_of_vol": vol_of_vol,
            "rho": rho,
            "jump_intensity": jump_intensity,
            "jump_mean": jump_mean,
            "jump_std": jump_std,
        },
    ]
    regime_params, guardrail_warnings = apply_regime_guardrails(
        regime_params,
        fallback_mu=float(np.clip(long_mu, -0.5, 0.5)),
    )
    warnings.extend(guardrail_warnings)
    kappa = float(regime_params[0]["kappa"])
    vol_of_vol = float(regime_params[0]["vol_of_vol"])
    jump_intensity = float(regime_params[0]["jump_intensity"])

    return HistoricalCalibration(
        mode="short_dte",
        ticker=ticker,
        period=f"{short_period}/{long_period}",
        interval=f"{short_interval}/{long_interval}",
        observations=int(len(short_ret)),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        mu_hist=long_mu,
        sigma_hist=long_sigma,
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_std=jump_std,
        transition=transition.tolist(),
        initial_prob=initial_prob.tolist(),
        regime_params=regime_params,
        initial_variance=initial_variance,
        realized_vol_1h=_realized_vol(short_ret.to_numpy(dtype=float), short_ppy, _steps_for_hours(short_interval, 1)),
        realized_vol_4h=_realized_vol(short_ret.to_numpy(dtype=float), short_ppy, _steps_for_hours(short_interval, 4)),
        realized_vol_24h=_realized_vol(short_ret.to_numpy(dtype=float), short_ppy, _steps_for_hours(short_interval, 24)),
        kappa_estimate=kappa,
        vol_of_vol_estimate=vol_of_vol,
        rho_estimate=rho,
        warnings=warnings,
    )


def merge_into_model_config(path: Path, historical: HistoricalCalibration) -> dict[str, Any]:
    config = json.loads(path.read_text()) if path.exists() else {}
    config.setdefault("signal", {"taker_fee_rate": 0.035, "slippage": 0.002, "model_buffer": 0.025, "min_edge": 0.01})
    config.setdefault("simulation", {"paths": 100_000, "steps_per_day": 24, "annualization_days": 365})
    config.setdefault("scanner", {})

    sim_interval = historical.interval.split("/")[0]
    config["simulation"]["steps_per_day"] = int(_steps_per_day(sim_interval))
    config["simulation"]["annualization_days"] = 365
    config["simulation"].pop("trading_days", None)
    config["scanner"]["initial_variance"] = float(historical.initial_variance)
    config["calibration_mode"] = historical.mode

    config["regime"] = {
        "initial_prob": historical.initial_prob,
        "transition": historical.transition,
        "params": historical.regime_params,
    }
    config.setdefault("metadata", {})
    config["metadata"]["historical_calibration"] = {
        "ticker": historical.ticker,
        "mode": historical.mode,
        "period": historical.period,
        "interval": historical.interval,
        "observations": historical.observations,
        "timestamp_utc": historical.timestamp_utc,
        "mu_hist": historical.mu_hist,
        "sigma_hist": historical.sigma_hist,
        "jump_intensity": historical.jump_intensity,
        "jump_mean": historical.jump_mean,
        "jump_std": historical.jump_std,
        "initial_variance": historical.initial_variance,
        "initial_vol": float(np.sqrt(historical.initial_variance)),
        "realized_vol_1h": historical.realized_vol_1h,
        "realized_vol_4h": historical.realized_vol_4h,
        "realized_vol_24h": historical.realized_vol_24h,
        "kappa_estimate": historical.kappa_estimate,
        "vol_of_vol_estimate": historical.vol_of_vol_estimate,
        "rho_estimate": historical.rho_estimate,
        "warnings": historical.warnings or [],
    }
    return config


def _fit_markov_regimes(log_ret: Any, periods_per_year: int, sm: Any) -> dict[str, Any]:
    log_ret_pct = log_ret * 100.0
    log_ret_pct.name = "log_returns_pct"
    model = sm.tsa.MarkovRegression(log_ret_pct, k_regimes=2, switching_variance=True)
    result = model.fit(search_reps=10, disp=False)

    sigma2_0 = float(result.params["sigma2[0]"])
    sigma2_1 = float(result.params["sigma2[1]"])
    low_vol_regime, high_vol_regime = (0, 1) if sigma2_0 < sigma2_1 else (1, 0)

    mu_low_period = float(result.params[f"const[{low_vol_regime}]"]) / 100.0
    mu_high_period = float(result.params[f"const[{high_vol_regime}]"]) / 100.0
    sigma_low = np.sqrt((float(result.params[f"sigma2[{low_vol_regime}]"]) / 100.0**2) * periods_per_year)
    sigma_high = np.sqrt((float(result.params[f"sigma2[{high_vol_regime}]"]) / 100.0**2) * periods_per_year)

    p00 = float(result.params["p[0->0]"])
    p10 = float(result.params["p[1->0]"])
    raw_transition = np.array([[p00, 1.0 - p00], [p10, 1.0 - p10]], dtype=float)
    order = [low_vol_regime, high_vol_regime]
    transition = raw_transition[np.ix_(order, order)]
    transition = transition / transition.sum(axis=1, keepdims=True)

    current_probs = result.filtered_marginal_probabilities.iloc[-1].to_numpy(dtype=float)
    initial_prob = current_probs[order]
    initial_prob = initial_prob / initial_prob.sum()
    return {
        "transition": transition,
        "initial_prob": initial_prob,
        "mu_low": float(mu_low_period * periods_per_year),
        "mu_high": float(mu_high_period * periods_per_year),
        "sigma_low": float(sigma_low),
        "sigma_high": float(sigma_high),
    }


def _ewma_variance(returns: np.ndarray, decay: float) -> float:
    variance = float(np.var(returns))
    for ret in returns:
        variance = decay * variance + (1.0 - decay) * float(ret * ret)
    return max(variance, 1e-12)


def _rolling_realized_variance(returns: np.ndarray, window: int, periods_per_year: int) -> np.ndarray:
    if len(returns) < window:
        return np.asarray([], dtype=float)
    n_blocks = len(returns) // window
    if n_blocks < 1:
        return np.asarray([], dtype=float)
    trimmed = returns[: n_blocks * window]
    blocks = trimmed.reshape(n_blocks, window)
    return np.mean(blocks * blocks, axis=1) * periods_per_year


def _estimate_kappa_from_variance(variance: np.ndarray, periods_per_year: int) -> tuple[float, list[str]]:
    warnings: list[str] = []
    if len(variance) < 20 or float(np.var(variance[:-1])) <= 1e-16:
        warnings.append("kappa fallback used because realized variance series is too short or flat")
        return 10.0, warnings
    x = variance[:-1]
    y = variance[1:]
    phi = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    if not np.isfinite(phi):
        warnings.append("kappa fallback used because AR(1) phi is non-finite")
        return 10.0, warnings
    if phi <= 0.0:
        warnings.append(f"kappa AR(1) phi <= 0 ({phi:.4f}); using fast mean-reversion floor")
        phi = 1e-6
    if phi >= 1.0:
        warnings.append(f"kappa AR(1) phi >= 1 ({phi:.4f}); using near-unit-root cap")
        phi = 0.999999
    dt = 1.0 / periods_per_year
    return float(-np.log(phi) / dt), warnings


def _estimate_vol_of_vol(
    variance: np.ndarray,
    kappa: float,
    theta: float,
    periods_per_year: int,
) -> tuple[float, list[str]]:
    warnings: list[str] = []
    if len(variance) < 20:
        warnings.append("vol_of_vol fallback used because realized variance series is too short")
        return 0.3, warnings
    raw_v_t = np.asarray(variance[:-1], dtype=float)
    positive_v = raw_v_t[np.isfinite(raw_v_t) & (raw_v_t > 0.0)]
    if len(positive_v) < 10:
        warnings.append("vol_of_vol fallback used because realized variance is near zero")
        return 0.3, warnings
    variance_floor = max(float(np.quantile(positive_v, 0.10)), float(np.median(positive_v)) * 0.05, 0.02**2)
    raw_v_next = np.asarray(variance[1:], dtype=float)
    stable_mask = raw_v_t >= variance_floor
    stable_mask &= raw_v_next >= variance_floor
    if int(np.sum(stable_mask)) >= 10:
        raw_v_t = raw_v_t[stable_mask]
        raw_v_next = raw_v_next[stable_mask]
    v_t = np.maximum(raw_v_t, variance_floor)
    dv = raw_v_next - raw_v_t
    dt = 1.0 / periods_per_year
    residual = dv - kappa * (theta - v_t) * dt
    scaled = residual / (np.sqrt(v_t) * np.sqrt(dt))
    scaled = scaled[np.isfinite(scaled)]
    if len(scaled) < 10:
        warnings.append("vol_of_vol fallback used because scaled variance shocks are invalid")
        return 0.3, warnings
    lower, upper = np.quantile(scaled, [0.01, 0.99])
    scaled = np.clip(scaled, lower, upper)
    estimate = float(np.std(scaled, ddof=1))
    if estimate <= 0.0 or not np.isfinite(estimate):
        warnings.append("vol_of_vol fallback used because estimate is non-positive")
        return 0.3, warnings
    return estimate, warnings


def _estimate_rho(returns: np.ndarray, variance: np.ndarray, rv_window: int) -> tuple[float, list[str]]:
    warnings: list[str] = []
    if len(variance) < 20:
        warnings.append("rho fallback used because realized variance series is too short")
        return -0.5, warnings
    dv = variance[1:] - variance[:-1]
    block_returns = _non_overlapping_block_returns(returns, rv_window)
    aligned_returns = block_returns[:-1]
    n = min(len(aligned_returns), len(dv))
    if n < 10:
        warnings.append("rho fallback used because return/variance alignment is too short")
        return -0.5, warnings
    corr = float(np.corrcoef(aligned_returns[:n], dv[:n])[0, 1])
    if not np.isfinite(corr):
        warnings.append("rho fallback used because correlation is non-finite")
        return -0.5, warnings
    return float(np.clip(corr, -0.95, 0.95)), warnings


def _non_overlapping_block_returns(returns: np.ndarray, window: int) -> np.ndarray:
    if window <= 0 or len(returns) < window:
        return np.asarray([], dtype=float)
    n_blocks = len(returns) // window
    trimmed = returns[: n_blocks * window]
    return np.sum(trimmed.reshape(n_blocks, window), axis=1)


def _realized_vol(returns: np.ndarray, periods_per_year: int, window: int) -> float | None:
    if window <= 1 or len(returns) < window:
        return None
    return float(np.std(returns[-window:]) * np.sqrt(periods_per_year))


def _steps_for_hours(interval: str, hours: float) -> int:
    if interval.endswith("m"):
        minutes = float(interval[:-1])
        return int(round(hours * 60 / minutes))
    if interval.endswith("h"):
        return int(round(hours / float(interval[:-1])))
    return 0


def _import_dependencies():
    try:
        import yfinance as yf
        import statsmodels.api as sm
    except ImportError as exc:
        raise ImportError(
            "Calibration requires yfinance and statsmodels. Install with: "
            "python3 -m pip install -r requirements.txt"
        ) from exc
    return yf, sm


def _periods_per_year(interval: str) -> int:
    if interval.endswith("h"):
        hours = float(interval[:-1])
        return int(round(24 * 365 / hours))
    if interval.endswith("m"):
        minutes = float(interval[:-1])
        return int(round(60 * 24 * 365 / minutes))
    if interval.endswith("d"):
        days = float(interval[:-1])
        return int(round(365 / days))
    raise ValueError(f"Unsupported interval: {interval}")


def _steps_per_day(interval: str) -> int:
    return max(1, int(round(_periods_per_year(interval) / 365)))


if __name__ == "__main__":
    main()
