from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianHMM:
    """Gaussian HMM with diagonal covariance matrices.

    This is intentionally compact: enough for regime filtering research without
    taking a dependency on hmmlearn. Features should be standardized before
    fitting.
    """

    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    variances: np.ndarray

    @property
    def n_states(self) -> int:
        return int(self.startprob.shape[0])

    def posterior(self, observations: np.ndarray) -> np.ndarray:
        """Return P(z_t | observations up to t) for every t."""

        observations = _as_2d(observations)
        log_emission = _log_diag_gaussian_pdf(observations, self.means, self.variances)
        _, alpha = _forward_log(self.startprob, self.transmat, log_emission)
        return np.exp(alpha - _logsumexp(alpha, axis=1)[:, None])

    def current_regime_prob(self, observations: np.ndarray) -> np.ndarray:
        return self.posterior(observations)[-1]


def fit_gaussian_hmm(
    observations: np.ndarray,
    n_states: int = 3,
    n_iter: int = 100,
    tol: float = 1e-5,
    random_seed: int | None = 7,
) -> GaussianHMM:
    """Fit a diagonal Gaussian HMM with Baum-Welch EM."""

    x = _as_2d(observations)
    if len(x) < n_states * 5:
        raise ValueError("Need more observations for stable HMM fitting.")

    rng = np.random.default_rng(random_seed)
    n_obs, n_features = x.shape

    means = _init_means_by_quantile(x, n_states)
    global_var = np.var(x, axis=0) + 1e-6
    variances = np.tile(global_var, (n_states, 1))
    startprob = np.full(n_states, 1.0 / n_states)
    transmat = np.full((n_states, n_states), 0.05 / max(n_states - 1, 1))
    np.fill_diagonal(transmat, 0.95)
    transmat = _normalize_rows(transmat)

    previous_ll = -np.inf
    for _ in range(n_iter):
        log_emission = _log_diag_gaussian_pdf(x, means, variances)
        log_likelihood, alpha = _forward_log(startprob, transmat, log_emission)
        beta = _backward_log(transmat, log_emission)

        log_gamma = alpha + beta - log_likelihood
        gamma = np.exp(log_gamma)
        gamma = gamma / gamma.sum(axis=1, keepdims=True)

        xi_sum = np.zeros((n_states, n_states))
        log_trans = np.log(np.clip(transmat, 1e-300, 1.0))
        for t in range(n_obs - 1):
            log_xi = (
                alpha[t, :, None]
                + log_trans
                + log_emission[t + 1, None, :]
                + beta[t + 1, None, :]
                - log_likelihood
            )
            xi = np.exp(log_xi)
            xi_sum += xi / max(xi.sum(), 1e-300)

        startprob = _normalize_vector(gamma[0] + 1e-8)
        transmat = _normalize_rows(xi_sum + 1e-8)

        weights = gamma.sum(axis=0) + 1e-8
        means = (gamma.T @ x) / weights[:, None]
        diffs = x[:, None, :] - means[None, :, :]
        variances = (gamma[:, :, None] * diffs * diffs).sum(axis=0) / weights[:, None]
        variances = np.clip(variances, 1e-6, None)

        if abs(log_likelihood - previous_ll) < tol:
            break
        previous_ll = log_likelihood

        if not np.isfinite(log_likelihood):
            means += rng.normal(scale=0.01, size=means.shape)
            variances = np.tile(global_var, (n_states, 1))

    order = np.argsort(means[:, 0])
    return GaussianHMM(
        startprob=startprob[order],
        transmat=transmat[np.ix_(order, order)],
        means=means[order],
        variances=variances[order],
    )


def standardize_features(observations: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _as_2d(observations)
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (x - mu) / sigma, mu, sigma


def _as_2d(observations: np.ndarray) -> np.ndarray:
    x = np.asarray(observations, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("observations must be a 1D or 2D array")
    if not np.all(np.isfinite(x)):
        raise ValueError("observations contain non-finite values")
    return x


def _init_means_by_quantile(x: np.ndarray, n_states: int) -> np.ndarray:
    score = x[:, 0]
    qs = np.linspace(0.1, 0.9, n_states)
    anchors = np.quantile(score, qs)
    means = []
    for anchor in anchors:
        idx = int(np.argmin(np.abs(score - anchor)))
        means.append(x[idx])
    return np.asarray(means, dtype=float)


def _log_diag_gaussian_pdf(x: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    var = np.clip(variances, 1e-12, None)
    diff = x[:, None, :] - means[None, :, :]
    log_det = np.sum(np.log(var), axis=1)
    maha = np.sum((diff * diff) / var[None, :, :], axis=2)
    n_features = x.shape[1]
    return -0.5 * (n_features * np.log(2.0 * np.pi) + log_det[None, :] + maha)


def _forward_log(
    startprob: np.ndarray,
    transmat: np.ndarray,
    log_emission: np.ndarray,
) -> tuple[float, np.ndarray]:
    n_obs, n_states = log_emission.shape
    alpha = np.empty((n_obs, n_states))
    alpha[0] = np.log(np.clip(startprob, 1e-300, 1.0)) + log_emission[0]
    log_trans = np.log(np.clip(transmat, 1e-300, 1.0))
    for t in range(1, n_obs):
        alpha[t] = log_emission[t] + _logsumexp(alpha[t - 1][:, None] + log_trans, axis=0)
    ll = float(_logsumexp(alpha[-1], axis=0))
    return ll, alpha


def _backward_log(transmat: np.ndarray, log_emission: np.ndarray) -> np.ndarray:
    n_obs, n_states = log_emission.shape
    beta = np.zeros((n_obs, n_states))
    log_trans = np.log(np.clip(transmat, 1e-300, 1.0))
    for t in range(n_obs - 2, -1, -1):
        beta[t] = _logsumexp(log_trans + log_emission[t + 1][None, :] + beta[t + 1][None, :], axis=1)
    return beta


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def _normalize_vector(v: np.ndarray) -> np.ndarray:
    v = np.clip(np.asarray(v, dtype=float), 1e-300, None)
    return v / v.sum()


def _normalize_rows(a: np.ndarray) -> np.ndarray:
    a = np.clip(np.asarray(a, dtype=float), 1e-300, None)
    return a / a.sum(axis=1, keepdims=True)
