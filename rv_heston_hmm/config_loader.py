from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np

from .model_guardrails import apply_regime_guardrails
from .pricing import SignalConfig
from .simulator import HestonJumpParams, RegimeModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_PATH = PROJECT_ROOT / "config" / "model_calibration.json"


def load_model_config(path: Path | str = DEFAULT_CALIBRATION_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def signal_config_from_model(
    calibration: dict[str, Any],
    *,
    taker_fee_rate: float | None = None,
    slippage: float | None = None,
    model_buffer: float | None = None,
    min_edge: float | None = None,
    min_liquidity: float | None = None,
) -> SignalConfig:
    signal = calibration.get("signal", {})
    return SignalConfig(
        taker_fee_rate=_override(signal, "taker_fee_rate", 0.035, taker_fee_rate),
        slippage=_override(signal, "slippage", 0.002, slippage),
        model_buffer=_override(signal, "model_buffer", 0.025, model_buffer),
        min_edge=_override(signal, "min_edge", 0.01, min_edge),
        min_liquidity=_override(signal, "min_liquidity", 0.0, min_liquidity),
    )


def regime_model_from_config(calibration: dict[str, Any]) -> RegimeModel:
    regime = calibration.get("regime", {})
    metadata = calibration.get("metadata", {}).get("historical_calibration", {})
    raw_params = [dict(item) for item in regime.get("params", [])]
    guarded_params, _ = apply_regime_guardrails(raw_params, fallback_mu=float(metadata.get("mu_hist", 0.0)))
    params = tuple(
        HestonJumpParams(
            mu=float(item["mu"]),
            kappa=float(item["kappa"]),
            theta=float(item["theta"]),
            vol_of_vol=float(item["vol_of_vol"]),
            rho=float(item["rho"]),
            jump_intensity=float(item["jump_intensity"]),
            jump_mean=float(item["jump_mean"]),
            jump_std=float(item["jump_std"]),
        )
        for item in guarded_params
    )
    return RegimeModel(
        transition=np.asarray(regime["transition"], dtype=float),
        initial_prob=np.asarray(regime["initial_prob"], dtype=float),
        params=params,
    )


def simulation_defaults(calibration: dict[str, Any]) -> dict[str, Any]:
    simulation = calibration.get("simulation", {})
    annualization_days = simulation.get("annualization_days", simulation.get("trading_days", 365))
    return {
        "paths": int(simulation.get("paths", 100_000)),
        "steps_per_day": int(simulation.get("steps_per_day", 24)),
        "annualization_days": int(annualization_days),
    }


def initial_variance_from_model(calibration: dict[str, Any], default: float = 0.50**2) -> float:
    metadata = calibration.get("metadata", {}).get("historical_calibration", {})
    scanner = calibration.get("scanner", {})
    value = metadata.get("initial_variance", scanner.get("initial_variance", default))
    return float(value)


def _override(source: dict[str, Any], key: str, default: float, value: float | None) -> float:
    return float(source.get(key, default) if value is None else value)
