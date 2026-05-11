from __future__ import annotations

from typing import Any
import math


MAX_ABS_MU = 3.0
MAX_KAPPA = 500.0
MAX_VOL_OF_VOL = 5.0
MAX_JUMP_INTENSITY = 50.0
FALLBACK_KAPPA = 50.0
FALLBACK_VOL_OF_VOL = 1.0
MIN_THETA = 0.02**2
MAX_THETA = 2.0**2
MAX_ABS_RHO = 0.95


def apply_regime_guardrails(
    params: list[dict[str, Any]],
    *,
    fallback_mu: float = 0.0,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    reasons_by_regime = [_regime_breakers(item) for item in params]
    hard_break = any(reasons for reasons in reasons_by_regime)
    safe_mu = _clip(_finite(fallback_mu, 0.0), -0.5, 0.5)

    guarded: list[dict[str, Any]] = []
    for item, reasons in zip(params, reasons_by_regime, strict=True):
        cleaned = dict(item)
        theta = _clip(_finite(item.get("theta"), MIN_THETA), MIN_THETA, MAX_THETA)
        jump_intensity = _clip(_finite(item.get("jump_intensity"), 0.0), 0.0, MAX_JUMP_INTENSITY)

        if hard_break:
            kappa = FALLBACK_KAPPA
            vol_of_vol = min(FALLBACK_VOL_OF_VOL, 0.95 * math.sqrt(max(2.0 * kappa * theta, 1e-12)))
            cleaned.update(
                {
                    "mu": safe_mu,
                    "kappa": kappa,
                    "theta": theta,
                    "vol_of_vol": vol_of_vol,
                    "rho": _clip(_finite(item.get("rho"), -0.5), -MAX_ABS_RHO, MAX_ABS_RHO),
                    "jump_intensity": jump_intensity,
                    "jump_mean": _finite(item.get("jump_mean"), 0.0),
                    "jump_std": max(_finite(item.get("jump_std"), 0.01), 1e-6),
                    "calibration_status": "FALLBACK",
                    "guardrail_reason": "; ".join(reasons or ["another regime failed guardrails"]),
                }
            )
        else:
            kappa = _clip(_finite(item.get("kappa"), FALLBACK_KAPPA), 1e-6, MAX_KAPPA)
            max_feller_vov = 0.95 * math.sqrt(max(2.0 * kappa * theta, 1e-12))
            cleaned.update(
                {
                    "mu": _clip(_finite(item.get("mu"), 0.0), -MAX_ABS_MU, MAX_ABS_MU),
                    "kappa": kappa,
                    "theta": theta,
                    "vol_of_vol": min(
                        _clip(_finite(item.get("vol_of_vol"), FALLBACK_VOL_OF_VOL), 1e-6, MAX_VOL_OF_VOL),
                        max_feller_vov,
                    ),
                    "rho": _clip(_finite(item.get("rho"), -0.5), -MAX_ABS_RHO, MAX_ABS_RHO),
                    "jump_intensity": jump_intensity,
                    "jump_mean": _finite(item.get("jump_mean"), 0.0),
                    "jump_std": max(_finite(item.get("jump_std"), 0.01), 1e-6),
                    "calibration_status": "OK",
                }
            )

        guarded.append(cleaned)

    if hard_break:
        unique_reasons = sorted({reason for reasons in reasons_by_regime for reason in reasons})
        warnings.append("guardrail fallback used: " + "; ".join(unique_reasons))
    for item in params:
        raw_jump_intensity = _finite(item.get("jump_intensity"), 0.0)
        if raw_jump_intensity > MAX_JUMP_INTENSITY:
            warnings.append(
                f"jump_intensity capped from {raw_jump_intensity:.2f}/year to {MAX_JUMP_INTENSITY:.2f}/year"
            )
            break
    return guarded, warnings


def regime_feller_margin(param: dict[str, Any]) -> float:
    kappa = _finite(param.get("kappa"), float("nan"))
    theta = _finite(param.get("theta"), float("nan"))
    vol_of_vol = _finite(param.get("vol_of_vol"), float("nan"))
    return 2.0 * kappa * theta - vol_of_vol**2


def _regime_breakers(param: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    mu = _finite(param.get("mu"), float("nan"))
    kappa = _finite(param.get("kappa"), float("nan"))
    theta = _finite(param.get("theta"), float("nan"))
    vol_of_vol = _finite(param.get("vol_of_vol"), float("nan"))

    if not math.isfinite(mu) or abs(mu) > MAX_ABS_MU:
        reasons.append(f"|mu|>{MAX_ABS_MU:g}")
    if not math.isfinite(kappa) or kappa <= 0.0 or kappa > MAX_KAPPA:
        reasons.append(f"kappa>{MAX_KAPPA:g}")
    if not math.isfinite(theta) or theta <= 0.0:
        reasons.append("theta<=0")
    if not math.isfinite(vol_of_vol) or vol_of_vol <= 0.0 or vol_of_vol > MAX_VOL_OF_VOL:
        reasons.append(f"vol_of_vol>{MAX_VOL_OF_VOL:g}")
    if math.isfinite(kappa) and math.isfinite(theta) and math.isfinite(vol_of_vol):
        if 2.0 * kappa * theta < vol_of_vol**2:
            reasons.append("Feller violated")
    return reasons


def _finite(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
