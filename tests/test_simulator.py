import math
import unittest

import numpy as np

from rv_heston_hmm.simulator import HestonJumpParams, RegimeModel, SimulationConfig, simulate_paths


class SimulatorTests(unittest.TestCase):
    def test_constant_vol_terminal_probability_matches_lognormal(self):
        spot = 100.0
        strike = 103.0
        mu = 0.05
        sigma = 0.20
        horizon_days = 30.0
        annualization_days = 365
        t = horizon_days / annualization_days
        regimes = RegimeModel(
            transition=np.array([[1.0]]),
            initial_prob=np.array([1.0]),
            params=(HestonJumpParams(mu, 0.0, sigma**2, 0.0, 0.0, 0.0, 0.0, 0.0),),
        )
        sim = simulate_paths(
            SimulationConfig(
                spot=spot,
                initial_variance=sigma**2,
                horizon_days=horizon_days,
                paths=60_000,
                steps_per_day=1,
                annualization_days=annualization_days,
                random_seed=123,
                store_paths=False,
            ),
            regimes,
        )
        observed = float(np.mean(sim.terminal > strike))
        d = (math.log(spot / strike) + (mu - 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
        expected = _normal_cdf(d)
        self.assertLess(abs(observed - expected), 0.015)


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


if __name__ == "__main__":
    unittest.main()

