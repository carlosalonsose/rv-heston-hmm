from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .simulator import RegimeModel, SimulationConfig, simulate_paths


EventKind = Literal["terminal_above", "terminal_below", "touches_above", "touches_below"]


@dataclass(frozen=True)
class BinaryEvent:
    kind: EventKind
    barrier: float

    def evaluate(self, terminal: np.ndarray, path_max: np.ndarray, path_min: np.ndarray) -> np.ndarray:
        if self.kind == "terminal_above":
            return terminal > self.barrier
        if self.kind == "terminal_below":
            return terminal < self.barrier
        if self.kind == "touches_above":
            return path_max > self.barrier
        if self.kind == "touches_below":
            return path_min < self.barrier
        raise ValueError(f"unsupported event kind: {self.kind}")


@dataclass(frozen=True)
class MarketQuote:
    bid_yes: float
    ask_yes: float
    yes_bid_qty: float | None = None
    yes_ask_qty: float | None = None

    def validate(self) -> str:
        if not 0.0 <= self.bid_yes <= 1.0:
            raise ValueError("bid_yes must be in [0, 1]")
        if not 0.0 <= self.ask_yes <= 1.0:
            raise ValueError("ask_yes must be in [0, 1]")
        if self.bid_yes > self.ask_yes:
            return "CROSSED_BOOK"
        return "OK"


@dataclass(frozen=True)
class SignalConfig:
    taker_fee_rate: float = 0.0
    slippage: float = 0.0
    model_buffer: float = 0.02
    min_edge: float = 0.01
    min_liquidity: float = 0.0


@dataclass(frozen=True)
class PricingResult:
    model_probability: float
    monte_carlo_se: float
    buy_yes_edge: float
    sell_yes_edge: float
    signal: str
    fair_bid: float
    fair_ask: float
    statistical_fair_bid: float
    statistical_fair_ask: float
    status: str = "OK"


def price_binary_event(
    event: BinaryEvent,
    quote: MarketQuote,
    signal_config: SignalConfig,
    simulation_config: SimulationConfig,
    regimes: RegimeModel,
) -> PricingResult:
    quote_status = quote.validate()
    if quote_status != "OK":
        return PricingResult(
            model_probability=float("nan"),
            monte_carlo_se=float("nan"),
            buy_yes_edge=float("nan"),
            sell_yes_edge=float("nan"),
            signal=quote_status,
            fair_bid=float("nan"),
            fair_ask=float("nan"),
            statistical_fair_bid=float("nan"),
            statistical_fair_ask=float("nan"),
            status=quote_status,
        )
    sim = simulate_paths(simulation_config, regimes)
    hits = event.evaluate(sim.terminal, sim.path_max, sim.path_min)
    p_model = float(np.mean(hits))
    n = int(hits.size)
    mc_se = float(np.sqrt(max(p_model * (1.0 - p_model), 0.0) / n))

    buy_cost = _binary_fee(quote.ask_yes, signal_config.taker_fee_rate) + signal_config.slippage
    sell_cost = _binary_fee(quote.bid_yes, signal_config.taker_fee_rate) + signal_config.slippage

    statistical_band = 2.0 * mc_se
    economic_haircut = signal_config.model_buffer + statistical_band
    buy_yes_edge = p_model - quote.ask_yes - buy_cost - economic_haircut
    sell_yes_edge = quote.bid_yes - p_model - sell_cost - economic_haircut

    has_buy_liquidity = quote.yes_ask_qty is not None and quote.yes_ask_qty >= signal_config.min_liquidity
    has_sell_liquidity = quote.yes_bid_qty is not None and quote.yes_bid_qty >= signal_config.min_liquidity

    if has_buy_liquidity and buy_yes_edge >= signal_config.min_edge and buy_yes_edge > sell_yes_edge:
        signal = "BUY_YES"
    elif has_sell_liquidity and sell_yes_edge >= signal_config.min_edge:
        signal = "SELL_YES_OR_BUY_NO"
    else:
        signal = "NO_TRADE"

    statistical_fair_bid = max(0.0, p_model - statistical_band)
    statistical_fair_ask = min(1.0, p_model + statistical_band)
    fair_bid = max(0.0, p_model - economic_haircut)
    fair_ask = min(1.0, p_model + economic_haircut)

    return PricingResult(
        model_probability=p_model,
        monte_carlo_se=mc_se,
        buy_yes_edge=float(buy_yes_edge),
        sell_yes_edge=float(sell_yes_edge),
        signal=signal,
        fair_bid=float(fair_bid),
        fair_ask=float(fair_ask),
        statistical_fair_bid=float(statistical_fair_bid),
        statistical_fair_ask=float(statistical_fair_ask),
    )


def _binary_fee(price: float, fee_rate: float) -> float:
    """Approximate Kalshi fee: rate * price * (1 - price), per $1 contract."""

    return float(fee_rate * price * (1.0 - price))
