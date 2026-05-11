from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HestonJumpParams:
    """Annualized Heston jump-diffusion parameters for one regime."""

    mu: float
    kappa: float
    theta: float
    vol_of_vol: float
    rho: float
    jump_intensity: float
    jump_mean: float
    jump_std: float


@dataclass(frozen=True)
class RegimeModel:
    """Regime transition matrix and parameters."""

    transition: np.ndarray
    initial_prob: np.ndarray
    params: tuple[HestonJumpParams, ...]

    def validate(self) -> None:
        n = len(self.params)
        if self.transition.shape != (n, n):
            raise ValueError("transition shape must match number of regimes")
        if self.initial_prob.shape != (n,):
            raise ValueError("initial_prob shape must match number of regimes")
        if not np.allclose(self.transition.sum(axis=1), 1.0):
            raise ValueError("transition rows must sum to 1")
        if not np.isclose(self.initial_prob.sum(), 1.0):
            raise ValueError("initial_prob must sum to 1")


@dataclass(frozen=True)
class SimulationConfig:
    spot: float
    initial_variance: float
    horizon_days: float
    paths: int = 100_000
    steps_per_day: int = 24
    annualization_days: int = 365
    random_seed: int | None = 42
    store_paths: bool = True


@dataclass(frozen=True)
class SimulationResult:
    terminal: np.ndarray
    path_max: np.ndarray
    path_min: np.ndarray
    paths: np.ndarray | None


def simulate_paths(config: SimulationConfig, regimes: RegimeModel) -> SimulationResult:
    """Simulate spot paths under regime-switching Heston jump-diffusion."""

    regimes.validate()
    if config.spot <= 0:
        raise ValueError("spot must be positive")
    if config.initial_variance < 0:
        raise ValueError("initial_variance must be non-negative")
    if config.horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    if config.paths <= 0:
        raise ValueError("paths must be positive")

    rng = np.random.default_rng(config.random_seed)
    n_steps = max(1, int(round(config.horizon_days * config.steps_per_day)))
    dt = 1.0 / (config.annualization_days * config.steps_per_day)
    sqrt_dt = np.sqrt(dt)
    n_paths = config.paths
    n_regimes = len(regimes.params)

    transition = np.asarray(regimes.transition, dtype=float)
    transition_cdf = np.cumsum(transition, axis=1)
    initial_cdf = np.cumsum(np.asarray(regimes.initial_prob, dtype=float))

    states = np.searchsorted(initial_cdf, rng.random(n_paths), side="right")
    states = np.minimum(states, n_regimes - 1)

    log_s = np.full(n_paths, np.log(config.spot), dtype=float)
    variance = np.full(n_paths, config.initial_variance, dtype=float)
    path_max = np.full(n_paths, config.spot, dtype=float)
    path_min = np.full(n_paths, config.spot, dtype=float)
    stored = np.empty((n_steps + 1, n_paths), dtype=float) if config.store_paths else None
    if stored is not None:
        stored[0] = config.spot

    for step in range(1, n_steps + 1):
        next_states = np.empty_like(states)
        uniforms = rng.random(n_paths)
        for regime_idx in range(n_regimes):
            mask = states == regime_idx
            if np.any(mask):
                next_states[mask] = np.searchsorted(transition_cdf[regime_idx], uniforms[mask], side="right")
        states = np.minimum(next_states, n_regimes - 1)

        z_s = rng.standard_normal(n_paths)
        z_v_independent = rng.standard_normal(n_paths)

        for regime_idx, params in enumerate(regimes.params):
            mask = states == regime_idx
            if not np.any(mask):
                continue

            idx = np.where(mask)[0]
            v = np.maximum(variance[idx], 0.0)
            rho = float(np.clip(params.rho, -0.999, 0.999))
            z_v = rho * z_s[idx] + np.sqrt(1.0 - rho * rho) * z_v_independent[idx]

            dv = (
                params.kappa * (params.theta - v) * dt
                + params.vol_of_vol * np.sqrt(v) * sqrt_dt * z_v
            )
            v_next = np.maximum(v + dv, 0.0)
            variance[idx] = v_next

            n_jumps = rng.poisson(params.jump_intensity * dt, size=idx.size)
            jump_log_return = np.zeros(idx.size)
            jump_mask = n_jumps > 0
            if np.any(jump_mask):
                jump_log_return[jump_mask] = (
                    n_jumps[jump_mask] * params.jump_mean
                    + np.sqrt(n_jumps[jump_mask]) * params.jump_std * rng.standard_normal(np.sum(jump_mask))
                )

            diffusion = np.sqrt(v) * sqrt_dt * z_s[idx]
            drift = (params.mu - 0.5 * v) * dt
            log_s[idx] = log_s[idx] + drift + diffusion + jump_log_return

        spot_t = np.exp(log_s)
        path_max = np.maximum(path_max, spot_t)
        path_min = np.minimum(path_min, spot_t)
        if stored is not None:
            stored[step] = spot_t

    return SimulationResult(terminal=np.exp(log_s), path_max=path_max, path_min=path_min, paths=stored)
