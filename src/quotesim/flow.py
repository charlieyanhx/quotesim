"""Synthetic order flow: Poisson candidates thinned against the quotes, with an informed fraction.

Units and conventions (this module works in SECONDS and $ per unit of underlying):
- `A`: candidate arrival rate per SIDE per instrument per second; `k`: intensity decay per $ of quote
  distance from fair: lambda(delta) = A exp(-k delta) (Avellaneda-Stoikov 2008).
- sides: index 0 = a customer SELLS to the market maker (hits the bid, MM buys, MM sign +1);
  index 1 = a customer BUYS from the market maker (lifts the ask, MM sells, MM sign -1).
- delta >= 0 is the $ distance of the quote from fair: bid side delta = F - bid, ask side delta = ask - F.
  A quote INSIDE fair (delta < 0) is capped at delta = 0 (accepted with probability 1, never more); a NaN
  quote means "pulled" (delta = inf, probability 0).
- `h_info` in seconds; the informed counterparty reads the fair `h_info` ahead on the pre-drawn path. That
  look-ahead is legitimate for the counterparty in a simulation and is passed in by the sim as
  `fair_ahead`; the quoter never sees it (see fair.PastOnlyFair).
- streams are drawn ONCE per seed in a FIXED order (spot_z, vol_z, jump_u, candidates, accept_u,
  informed_u, direction_u) so that two strategies under the same seed see the same candidates and the same
  uniforms and differ only through thinning (common random numbers by construction).

Thinning law: with c ~ Poisson(A dt) candidates in a cell and acceptance probability p = exp(-k delta),
the number accepted is Binomial(c, p) and therefore Poisson(A p dt) = Poisson(lambda dt), so
P(>= 1 fill in dt) = 1 - exp(-lambda dt) exactly (never lambda dt). One uniform per cell drives the whole
binomial through its quantile: n_accepted = #{n >= 1 : P(Bin(c, p) >= n) > u}; for c = 1 this is exactly
"accept iff u < p", and it is monotone in p (a tighter quote never loses a fill the wider one got under
the same u). The informed flag and direction are per cell (all candidates in a cell share them).

Informed candidate: with probability `p_informed` it trades in the direction of sign(F_{t+h} - F_t)
(customer BUYS when fair will rise, i.e. lifts the ask) and with probability 1 - p_informed AGAINST it, so
p_informed = 0.5 carries no information and the adverse mean is -(2p - 1) E|dF| (the plan's oracle; the
plan's wording "else a coin" would give -p E|dF| and a nonzero mean at p = 0.5, so the oracle wins).
When F_{t+h} == F_t it keeps its nominal side (a fair coin by symmetry of the two Poisson streams).
Uninformed candidate: nominal side,
or with probability `persist` the side of the previous uninformed candidate on that instrument (Markov
sign persistence; needs the previous `StepFills` passed as `prev`).

Oracle (tested): for informed flow on a Brownian fair with vol sigma_F ($/sqrt(s)),
E[s_f (F_{t_f+h} - F_{t_f})] = -(2p - 1) sigma_F sqrt(h) sqrt(2 / pi).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

N_SIDES = 2
MM_SIGN = np.array([1.0, -1.0])  # per side: +1 MM buys, -1 MM sells


@dataclass(frozen=True)
class FlowParams:
    """A per second per side per instrument; k per $; informed_frac in [0, 1]; p_informed in [0.5, 1];
    h_info seconds; persist in [0, 1)."""

    A: float
    k: float
    informed_frac: float = 0.0
    p_informed: float = 1.0
    h_info: float = 60.0
    persist: float = 0.0

    def __post_init__(self):
        if self.A <= 0 or self.k < 0:
            raise ValueError("A > 0 and k >= 0")
        if not 0.0 <= self.informed_frac <= 1.0 or not 0.5 <= self.p_informed <= 1.0:
            raise ValueError("informed_frac in [0, 1], p_informed in [0.5, 1]")
        if self.h_info <= 0 or not 0.0 <= self.persist < 1.0:
            raise ValueError("h_info > 0, persist in [0, 1)")

    def h_steps(self, dt: float) -> int:
        """Look-ahead in steps for a step of dt seconds (at least 1)."""
        return max(1, int(round(self.h_info / dt)))


@dataclass(frozen=True)
class Streams:
    """Every random draw of one seed. Shapes: spot_z, vol_z, jump_u (n_steps,); candidates, accept_u,
    informed_u, direction_u (n_steps, n_instruments, 2)."""

    seed: int
    dt: float
    A: float
    spot_z: np.ndarray
    vol_z: np.ndarray
    jump_u: np.ndarray
    candidates: np.ndarray
    accept_u: np.ndarray
    informed_u: np.ndarray
    direction_u: np.ndarray

    @classmethod
    def draw(cls, seed: int, n_steps: int, n_instruments: int, A: float, dt: float) -> Streams:
        """Draw all streams from one numpy Generator in the fixed order listed in the module docstring."""
        if n_steps <= 0 or n_instruments <= 0 or A <= 0 or dt <= 0:
            raise ValueError("n_steps, n_instruments, A, dt must be > 0")
        gen = np.random.default_rng(seed)
        shape = (n_steps, n_instruments, N_SIDES)
        spot_z = gen.standard_normal(n_steps)
        vol_z = gen.standard_normal(n_steps)
        jump_u = gen.random(n_steps)
        candidates = gen.poisson(A * dt, size=shape).astype(np.int64)
        accept_u = gen.random(shape)
        informed_u = gen.random(shape)
        direction_u = gen.random(shape)
        return cls(int(seed), float(dt), float(A), spot_z, vol_z, jump_u, candidates, accept_u, informed_u, direction_u)

    @property
    def n_steps(self) -> int:
        return int(self.spot_z.shape[0])

    @property
    def n_instruments(self) -> int:
        return int(self.candidates.shape[1])


@dataclass(frozen=True)
class StepFills:
    """Fills of one step. fills[i, side] = lots filled on instrument i (side 0: MM bought at bid[i], side 1:
    MM sold at ask[i]); informed[i, side] = how many of those came from informed candidates;
    last_side[i] = side of the last uninformed candidate seen on instrument i (persistence state)."""

    fills: np.ndarray
    informed: np.ndarray
    last_side: np.ndarray

    @property
    def n_fills(self) -> int:
        return int(self.fills.sum())

    def signed_lots(self) -> np.ndarray:
        """Net MM position change per instrument: bought - sold."""
        return self.fills[:, 0] - self.fills[:, 1]


# ----------------------------------------------------------------------------------------------------------
# thinning
# ----------------------------------------------------------------------------------------------------------


def accept_prob(delta, k: float) -> np.ndarray:
    """exp(-k max(delta, 0)); NaN (pulled quote) -> 0."""
    d = np.asarray(delta, dtype=float)
    p = np.exp(-k * np.maximum(d, 0.0))
    return np.where(np.isnan(d), 0.0, p)


def _binomial_tail_table(counts: np.ndarray, p: np.ndarray) -> np.ndarray:
    """tail[n - 1, ...] = P(Bin(counts, p) >= n) for n = 1..c_max (0 where n > counts)."""
    c_max = int(counts.max()) if counts.size else 0
    if c_max == 0:
        return np.zeros((0,) + counts.shape)
    j = np.arange(c_max + 1).reshape((-1,) + (1,) * counts.ndim)
    comb_table = np.array([[comb(c, jj) for jj in range(c_max + 1)] for c in range(c_max + 1)], dtype=float)
    binom = comb_table[counts[None], j]
    pmf = binom * np.power(p, j) * np.power(1.0 - p, np.maximum(counts - j, 0))
    tail = np.cumsum(pmf[::-1], axis=0)[::-1]  # tail[j] = P(X >= j)
    return tail[1:]


def thin(candidates, accept_u, delta, k: float) -> np.ndarray:
    """Number of candidates accepted per cell: the Binomial(candidates, exp(-k max(delta, 0))) quantile at
    accept_u (see module docstring). All arguments broadcast to one shape; returns int64 of that shape."""
    c, u, d = np.broadcast_arrays(np.asarray(candidates, dtype=np.int64), np.asarray(accept_u, float), np.asarray(delta, float))
    if np.any(c < 0) or np.any((u < 0) | (u >= 1)):
        raise ValueError("candidates >= 0 and accept_u in [0, 1)")
    p = accept_prob(d, k)
    tail = _binomial_tail_table(c, p)
    if tail.shape[0] == 0:
        return np.zeros(c.shape, dtype=np.int64)
    return (tail > u[None]).sum(axis=0).astype(np.int64)


# ----------------------------------------------------------------------------------------------------------
# one step of arrivals
# ----------------------------------------------------------------------------------------------------------


def _side_choice(params, streams, step, s, fair_now, fair_ahead, last_side):
    """Realised side and informed flag for every instrument's cell of nominal side s at this step."""
    inf_flag = streams.informed_u[step, :, s] < params.informed_frac
    dir_u = streams.direction_u[step, :, s]
    move = np.sign(fair_ahead - fair_now)
    with_move = np.where(move > 0, 1, np.where(move < 0, 0, s))
    against = np.where(move > 0, 0, np.where(move < 0, 1, s))
    side_inf = np.where(dir_u < params.p_informed, with_move, against)
    side_uninf = np.where(dir_u < params.persist, last_side, s) if params.persist > 0 else np.full_like(last_side, s)
    side = np.where(inf_flag, side_inf, side_uninf)
    return side.astype(np.int64), inf_flag


def arrivals(params: FlowParams, streams: Streams, step: int, bid, ask, fair_now, fair_ahead, prev: StepFills | None = None) -> StepFills:
    """Fills of one step given the quotes (arrays over instruments; NaN = pulled), the fair now and the fair
    `h_info` ahead (read from the pre-drawn path by the sim; clip to the last state at the end).
    `prev` carries the persistence state; None starts every instrument on nominal sides."""
    bid, ask, fair_now, fair_ahead = (np.asarray(x, dtype=float) for x in (bid, ask, fair_now, fair_ahead))
    n = streams.n_instruments
    if not (bid.shape == ask.shape == fair_now.shape == fair_ahead.shape == (n,)):
        raise ValueError("bid, ask, fair_now, fair_ahead must be (n_instruments,)")
    delta_by_side = np.stack([fair_now - bid, ask - fair_now], axis=1)  # (n, 2)
    last_side = prev.last_side.copy() if prev is not None else np.zeros(n, dtype=np.int64)
    fills = np.zeros((n, N_SIDES), dtype=np.int64)
    informed = np.zeros((n, N_SIDES), dtype=np.int64)
    idx = np.arange(n)
    for s in range(N_SIDES):
        c = streams.candidates[step, :, s]
        side, inf_flag = _side_choice(params, streams, step, s, fair_now, fair_ahead, last_side)
        got = thin(c, streams.accept_u[step, :, s], delta_by_side[idx, side], params.k)
        np.add.at(fills, (idx, side), got)
        np.add.at(informed, (idx, side), np.where(inf_flag, got, 0))
        seen_uninf = (c > 0) & ~inf_flag
        last_side = np.where(seen_uninf, side, last_side)
    return StepFills(fills=fills, informed=informed, last_side=last_side)
