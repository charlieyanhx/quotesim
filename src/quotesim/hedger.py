"""Delta hedging rules and the hedge cost model.

Units and conventions (this module works in SECONDS, shares and $):
- h = hedge position in shares of the underlying (+ long); target = -sum_j q_j Delta_j * multiplier
  (`target_shares`); a rule decides the trade dh = target - h (shares, signed) and its cost (>= 0, $).
- cost model: any callable `cost(shares, S) -> $` with shares the SIGNED trade and S the spot mid; the
  default `CostModel(half_spread, Y, sigma_daily, adv)` is
      cost = half_spread * |dh| + Y * sigma_daily * sqrt(|dh| / adv) * S * |dh|
  (tcakit's sqrt-law form: impact as a fraction of price = Y sigma_daily sqrt(qty / ADV)); half_spread in $
  per share, sigma_daily a daily vol fraction, adv in shares per day, Y a LABELLED MODEL CONSTANT (no
  fitted value ships; ~0.3-1 is the usual range). The hedge cost is the one identity term that is a
  model, not a measurement.
- cash convention for the caller (pnl.py): cash -= dh * S + cost; HCOST = sum cost. The hedge fills at the
  mid S; the whole slippage lives in `cost`, so HEDGE = sum h (S_{t+1} - S_t) and HCOST separate exactly.
- BandHedger(band_delta_usd): trade to target when |(h - target) * S| > band_delta_usd ($ delta mismatch);
  band 0 -> every step the mismatch is nonzero. TimeHedger(every_s): trade to target when t - t_last >=
  every_s (and at t = 0 when t_last is None), t_last being the last time the rule was DUE (`HedgeDecision.due`,
  set even when the mismatch was zero and nothing traded), so the grid is anchored at t = 0, not at the first
  fill. NoHedger: never trades (h stays at its start).
- band_ww(S, Gamma, lam, gamma, r, tau) = (3/2 e^{-r tau} lam S Gamma^2 / gamma)^{1/3}: the Whalley-Wilmott
  1997 asymptotic no-transaction half-band in SHARES per option around the Black delta, lam = proportional
  cost (fraction of the traded $), gamma = risk aversion, r rate, tau years to expiry (this one argument is
  in years because Gamma comes from the pricer). Band scales as lam^{1/3} and Gamma^{2/3}.

Invariants kept (tested): cost >= 0 for every trade; a trade always lands exactly on target; no trade
inside the band; `HedgeDecision.shares == 0` implies cost == 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

CostFn = Callable[[float, float], float]


@dataclass(frozen=True)
class CostModel:
    """Default sqrt-law cost; see module docstring. Y is a model constant, not a fitted number."""

    half_spread: float = 0.01
    Y: float = 0.5
    sigma_daily: float = 0.01
    adv: float = 5.0e7

    def __post_init__(self):
        if self.half_spread < 0 or self.Y < 0 or self.sigma_daily < 0 or self.adv <= 0:
            raise ValueError("half_spread, Y, sigma_daily >= 0 and adv > 0")

    def __call__(self, shares: float, S: float) -> float:
        return default_cost_model(shares, S, self.half_spread, self.Y, self.sigma_daily, self.adv)


def default_cost_model(shares: float, S: float, half_spread: float, Y: float, sigma_daily: float, adv: float) -> float:
    """half_spread |dh| + Y sigma_daily sqrt(|dh| / adv) S |dh|; $; >= 0."""
    q = abs(float(shares))
    return half_spread * q + Y * sigma_daily * np.sqrt(q / adv) * S * q


def target_shares(inventory, deltas, multiplier=1.0) -> float:
    """-sum q Delta multiplier: the share position that flattens the option delta."""
    return -float(np.sum(np.asarray(inventory, dtype=float) * np.asarray(deltas, dtype=float) * multiplier))


@dataclass(frozen=True)
class HedgeDecision:
    shares: float  # signed trade; 0.0 = no trade
    cost: float  # $ >= 0
    new_h: float  # position after the trade
    due: bool = False  # the rule fired (a trade to target, possibly of zero size); the sim resets t_last on it

    @property
    def traded(self) -> bool:
        return self.shares != 0.0


class Hedger(Protocol):
    def decide(self, h: float, target: float, S: float, t: float, t_last: float | None) -> HedgeDecision: ...


def _trade_to(h: float, target: float, S: float, cost_model: CostFn) -> HedgeDecision:
    dh = float(target - h)
    if dh == 0.0:
        return HedgeDecision(0.0, 0.0, float(h), due=True)
    cost = float(cost_model(dh, S))
    if cost < 0:
        raise ValueError("cost model returned a negative cost")
    return HedgeDecision(dh, cost, float(target), due=True)


def _hold(h: float) -> HedgeDecision:
    return HedgeDecision(0.0, 0.0, float(h))


@dataclass(frozen=True)
class BandHedger:
    band_delta_usd: float
    cost_model: CostFn = CostModel()

    def __post_init__(self):
        if self.band_delta_usd < 0:
            raise ValueError("band_delta_usd >= 0")

    def decide(self, h: float, target: float, S: float, t: float = 0.0, t_last: float | None = None) -> HedgeDecision:
        if abs((h - target) * S) > self.band_delta_usd:
            return _trade_to(h, target, S, self.cost_model)
        return _hold(h)


@dataclass(frozen=True)
class TimeHedger:
    every_s: float
    cost_model: CostFn = CostModel()

    def __post_init__(self):
        if self.every_s <= 0:
            raise ValueError("every_s > 0")

    def decide(self, h: float, target: float, S: float, t: float = 0.0, t_last: float | None = None) -> HedgeDecision:
        due = t_last is None or (t - t_last) >= self.every_s - 1e-12
        if due:
            return _trade_to(h, target, S, self.cost_model)
        return _hold(h)


@dataclass(frozen=True)
class NoHedger:
    def decide(self, h: float, target: float, S: float, t: float = 0.0, t_last: float | None = None) -> HedgeDecision:
        return _hold(h)


def band_ww(S: float, Gamma: float, lam: float, gamma: float, r: float = 0.0, tau: float = 0.0) -> float:
    """Whalley-Wilmott 1997 half-band H = (3/2 e^{-r tau} lam S Gamma^2 / gamma)^{1/3}, shares per option."""
    if S <= 0 or lam < 0 or gamma <= 0 or tau < 0:
        raise ValueError("S > 0, lam >= 0, gamma > 0, tau >= 0")
    return float((1.5 * np.exp(-r * tau) * lam * S * Gamma**2 / gamma) ** (1.0 / 3.0))
