import math
import unittest

import numpy as np

from rv_heston_hmm.calibration import (
    _estimate_kappa_from_variance,
    _estimate_rho,
    _estimate_vol_of_vol,
    _rolling_realized_variance,
)


class CalibrationEstimatorTests(unittest.TestCase):
    def test_kappa_from_ar1_variance(self):
        phi = 0.92
        periods_per_year = 365 * 24
        x = [0.2]
        for i in range(1, 400):
            x.append(0.04 + phi * (x[-1] - 0.04) + 0.001 * math.sin(i))
        kappa, warnings = _estimate_kappa_from_variance(np.asarray(x), periods_per_year)
        expected = -math.log(phi) * periods_per_year
        self.assertLess(abs(kappa - expected) / expected, 0.15)
        self.assertEqual(warnings, [])

    def test_rho_estimator_sign(self):
        window = 6
        block_returns = np.linspace(-0.01, 0.01, 40)
        returns = np.repeat(block_returns / window, window)
        variance = 0.2 + np.cumsum(np.r_[0.0, block_returns[:-1] * 0.05])
        rho, warnings = _estimate_rho(returns, variance, window)
        self.assertGreater(rho, 0.8)
        self.assertEqual(warnings, [])

    def test_realized_variance_uses_non_overlapping_blocks(self):
        returns = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        variance = _rolling_realized_variance(returns, window=2, periods_per_year=10)
        np.testing.assert_allclose(variance, np.array([25.0, 125.0]))

    def test_vol_of_vol_estimator_ignores_near_zero_denominator_spikes(self):
        rng = np.random.default_rng(7)
        periods_per_year = 365 * 24
        variance = np.full(240, 0.04)
        variance[::17] = 1e-12
        variance += 0.001 * rng.standard_normal(240)
        variance = np.maximum(variance, 1e-12)
        estimate, warnings = _estimate_vol_of_vol(variance, kappa=20.0, theta=0.04, periods_per_year=periods_per_year)
        self.assertLess(estimate, 5.0)
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
