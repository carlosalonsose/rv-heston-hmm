from __future__ import annotations

import argparse

import numpy as np

from .config_loader import (
    initial_variance_from_model,
    load_model_config,
    regime_model_from_config,
    signal_config_from_model,
    simulation_defaults,
)
from .hmm import fit_gaussian_hmm, standardize_features
from .pricing import BinaryEvent, MarketQuote, SignalConfig, price_binary_event
from .simulator import HestonJumpParams, RegimeModel, SimulationConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Relative value binary pricing prototype.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("synthetic-demo", help="Run a synthetic HMM + Monte Carlo demo.")
    demo_parser.add_argument("--paths", type=int, default=100_000)

    price_parser = subparsers.add_parser("price", help="Price a simple terminal-above event.")
    price_parser.add_argument("--spot", type=float, required=True)
    price_parser.add_argument("--barrier", type=float, required=True)
    price_parser.add_argument("--days", type=float, required=True)
    price_parser.add_argument("--bid", type=float, required=True)
    price_parser.add_argument("--ask", type=float, required=True)
    price_parser.add_argument("--paths", type=int, default=100_000)
    price_parser.add_argument("--calibration", default="config/model_calibration.json")
    price_parser.add_argument("--fee-rate", type=float, default=None)
    price_parser.add_argument("--model-buffer", type=float, default=None)
    price_parser.add_argument(
        "--event",
        choices=["terminal_above", "terminal_below", "touches_above", "touches_below"],
        default="terminal_above",
    )

    args = parser.parse_args()
    if args.command == "synthetic-demo":
        run_demo(paths=args.paths)
    elif args.command == "price":
        run_price(args)


def run_demo(paths: int = 100_000) -> None:
    returns = _synthetic_returns()
    features = _make_features(returns)
    standardized, _, _ = standardize_features(features)
    hmm = fit_gaussian_hmm(standardized, n_states=3, n_iter=80)
    current_regime_prob = hmm.current_regime_prob(standardized)

    regimes = _default_regimes(current_regime_prob, hmm.transmat)
    event = BinaryEvent(kind="terminal_above", barrier=105_000.0)
    quote = MarketQuote(bid_yes=0.39, ask_yes=0.42)
    signal_config = SignalConfig(taker_fee_rate=0.04, slippage=0.002, model_buffer=0.025)
    simulation_config = SimulationConfig(
        spot=100_000.0,
        initial_variance=0.55**2,
        horizon_days=3.0,
        paths=paths,
        steps_per_day=24,
        random_seed=11,
        store_paths=False,
    )
    result = price_binary_event(event, quote, signal_config, simulation_config, regimes)
    _print_result(result)
    print("current_regime_prob:", np.round(current_regime_prob, 4))


def run_price(args: argparse.Namespace) -> None:
    calibration = load_model_config(args.calibration)
    regimes = regime_model_from_config(calibration)
    sim_defaults = simulation_defaults(calibration)
    event = BinaryEvent(kind=args.event, barrier=args.barrier)
    quote = MarketQuote(bid_yes=args.bid, ask_yes=args.ask)
    signal_config = signal_config_from_model(calibration, taker_fee_rate=args.fee_rate, model_buffer=args.model_buffer)
    simulation_config = SimulationConfig(
        spot=args.spot,
        initial_variance=initial_variance_from_model(calibration),
        horizon_days=args.days,
        paths=args.paths,
        steps_per_day=sim_defaults["steps_per_day"],
        annualization_days=sim_defaults["annualization_days"],
        random_seed=42,
        store_paths=False,
    )
    result = price_binary_event(event, quote, signal_config, simulation_config, regimes)
    _print_result(result)


def _default_regimes(initial_prob: np.ndarray, transition: np.ndarray) -> RegimeModel:
    params = (
        HestonJumpParams(
            mu=0.05,
            kappa=5.0,
            theta=0.22**2,
            vol_of_vol=0.45,
            rho=-0.35,
            jump_intensity=4.0,
            jump_mean=-0.015,
            jump_std=0.030,
        ),
        HestonJumpParams(
            mu=0.02,
            kappa=3.0,
            theta=0.50**2,
            vol_of_vol=0.85,
            rho=-0.50,
            jump_intensity=12.0,
            jump_mean=-0.020,
            jump_std=0.055,
        ),
        HestonJumpParams(
            mu=-0.10,
            kappa=2.0,
            theta=0.90**2,
            vol_of_vol=1.25,
            rho=-0.65,
            jump_intensity=30.0,
            jump_mean=-0.035,
            jump_std=0.100,
        ),
    )
    return RegimeModel(transition=transition, initial_prob=initial_prob, params=params)


def _synthetic_returns(n: int = 1500, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    transition = np.array([[0.97, 0.025, 0.005], [0.05, 0.90, 0.05], [0.02, 0.10, 0.88]])
    vols = np.array([0.012, 0.026, 0.055])
    drifts = np.array([0.0005, 0.0000, -0.0010])
    states = np.zeros(n, dtype=int)
    for i in range(1, n):
        states[i] = np.searchsorted(np.cumsum(transition[states[i - 1]]), rng.random(), side="right")
    return drifts[states] + vols[states] * rng.standard_normal(n)


def _make_features(returns: np.ndarray, vol_window: int = 20) -> np.ndarray:
    returns = np.asarray(returns, dtype=float)
    realized = np.empty_like(returns)
    for i in range(len(returns)):
        start = max(0, i - vol_window + 1)
        realized[i] = np.std(returns[start : i + 1])
    abs_ret = np.abs(returns)
    return np.column_stack([realized, abs_ret, returns])


def _print_result(result) -> None:
    print(f"model_probability: {result.model_probability:.4f}")
    print(f"monte_carlo_se:    {result.monte_carlo_se:.4f}")
    print(f"fair_band:         [{result.fair_bid:.4f}, {result.fair_ask:.4f}]")
    print(f"buy_yes_edge:      {result.buy_yes_edge:.4f}")
    print(f"sell_yes_edge:     {result.sell_yes_edge:.4f}")
    print(f"signal:            {result.signal}")


if __name__ == "__main__":
    main()
