import math
import unittest

from rv_heston_hmm.pricing import BinaryEvent, MarketQuote, SignalConfig, price_binary_event
from rv_heston_hmm.simulator import HestonJumpParams, RegimeModel, SimulationConfig

import numpy as np


class PricingTests(unittest.TestCase):
    def test_crossed_book_returns_status(self):
        regimes = RegimeModel(
            transition=np.array([[1.0]]),
            initial_prob=np.array([1.0]),
            params=(HestonJumpParams(0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0),),
        )
        result = price_binary_event(
            BinaryEvent("terminal_above", 100.0),
            MarketQuote(bid_yes=0.60, ask_yes=0.55),
            SignalConfig(),
            SimulationConfig(spot=100.0, initial_variance=0.04, horizon_days=1.0, paths=100, store_paths=False),
            regimes,
        )
        self.assertEqual(result.status, "CROSSED_BOOK")
        self.assertTrue(math.isnan(result.model_probability))

    def test_liquidity_filter_blocks_signal(self):
        regimes = RegimeModel(
            transition=np.array([[1.0]]),
            initial_prob=np.array([1.0]),
            params=(HestonJumpParams(0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0),),
        )
        result = price_binary_event(
            BinaryEvent("terminal_above", 50.0),
            MarketQuote(bid_yes=0.01, ask_yes=0.02, yes_ask_qty=0.0),
            SignalConfig(min_edge=0.01, min_liquidity=1.0),
            SimulationConfig(spot=100.0, initial_variance=0.01, horizon_days=1.0, paths=1000, store_paths=False),
            regimes,
        )
        self.assertEqual(result.signal, "NO_TRADE")

    def test_missing_liquidity_blocks_signal(self):
        regimes = RegimeModel(
            transition=np.array([[1.0]]),
            initial_prob=np.array([1.0]),
            params=(HestonJumpParams(0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0),),
        )
        result = price_binary_event(
            BinaryEvent("terminal_above", 50.0),
            MarketQuote(bid_yes=0.01, ask_yes=0.02),
            SignalConfig(min_edge=0.01, min_liquidity=0.0),
            SimulationConfig(spot=100.0, initial_variance=0.01, horizon_days=1.0, paths=1000, store_paths=False),
            regimes,
        )
        self.assertEqual(result.signal, "NO_TRADE")


if __name__ == "__main__":
    unittest.main()
