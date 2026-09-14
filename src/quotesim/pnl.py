"""P&L attribution of one simulated run: an exact four-term identity, two exact sub-splits of the
inventory term, and a Greek explanation layer that is the only place with a residual.

Units and conventions ($ per run; time in seconds on the simulator clock, theta and dt in years inside
the Greek terms; MM-side sign: + = the market maker BUYS):
- marks are to FAIR (the surface mid F_t), never to the maker's own quotes; the underlying hedge is marked
  at spot S_t. Every option quantity is multiplied by the instrument's `multiplier` (1.0 in v0.1).
- with q_t the option positions HELD OVER step t (after that step's fills), h_t the hedge shares held over
  step t, fills f = (step t_f, instrument j, signed lots dq_f, price px_f) and hedge trades with cost c:
    SPREAD = sum_f dq_f (F_{t_f} - px_f)            realised half-spread on every fill (>= 0 while the quote sits
                                                     outside fair; an A-S / GLFT per-side offset goes negative at
                                                     |q| > c / w, the quote then sits INSIDE fair, is hit with
                                                     probability 1 and the fill's SPREAD term is negative: the
                                                     maker pays to unwind. `Run.n_inside_fair` counts those lots)
    INV    = sum_t q_t . (F_{t+1} - F_t)             option positions marked step by step
    HEDGE  = sum_t h_t (S_{t+1} - S_t)               hedge shares marked step by step
    HCOST  = sum_hedge trades c                      half-spread + impact of every hedge trade (>= 0)
    R      = cash_T + q_T . F_T + h_T S_T - (q_0 . F_0 + h_0 S_0)      realised P&L, mark-to-fair
  and R = SPREAD + INV + HEDGE - HCOST with NO residual: every cash flow or mark change in the loop is one
  of the four (an option fill moves cash by -dq px and the mark by +dq F; a hedge trade moves cash by
  -dh S - c and the mark by +dh S; holding moves the marks by q dF and h dS). `gap` = R - (the sum), every
  sum a `math.fsum`, so the gap is the rounding of the PRODUCTS the sums are over (qty px mult, q F mult,
  q dF mult, h dS ...), each within 2 ulp of its value: |gap| <= 2 eps * `scale`, where `scale` is the gross
  $ of every such product (all fills, hedge trades and costs, opening and terminal marks, the gross inventory
  and hedge marks). `sim.run` raises if |gap| >= `bar` = max(GAP_BAR = 1e-9, GAP_REL = 1e-12 * scale): the
  relative part is 4,500x the rounding bound (a bookkeeping error is >= one product, 1e12 x larger), the
  absolute floor is where $-scale runs sit (scale ~1e5 $ on the README's strip gives a bar of ~1e-7;
  multiplier 100 on a 500 $ underlying with a 100-lot opening book gives gaps ~1e-9 on a scale ~1e9).
- sub-splits of INV, both exact (the fill decomposition q_t = q_0 + sum_{f: t_f <= t} dq_f):
    INV = OPENING + ADVERSE_h + DRIFT_h,  OPENING = q_0 . (F_T - F_0),
         ADVERSE_h = sum_f dq_f (F_{t_f + h} - F_{t_f}),  DRIFT_h = sum_f dq_f (F_T - F_{t_f + h}),
         t_f + h clipped at T; h = `h` seconds (60 s canonical); ADVERSE_h is the fill-level markout
         summed over fills, a SUBSET of INV, so adverse selection is never a fifth top-level term.
    INV = THETA + INV_exTHETA,  THETA = sum_t q_t . theta_t dt (theta per year at fair, dt in years).
  Both are computed independently of INV and their gaps (`split_gap_fills`, `split_gap_theta`) are
  reported; they are floating-point zeros, not definitions.
- Greek explanation layer (the only layer WITH a residual): INV_greek = sum_t q_t . (Delta_t dS + 1/2 Gamma_t
  dS^2 + Vega_t dsigma_t + theta_t dt) with dsigma_t the change of each instrument's OWN fair vol (sticky
  strike: it carries the smile move from dk = -dS/S); RESID = INV - INV_greek (vanna, volga, third order,
  jumps). `greek_theta` equals `theta` by construction.

Invariants kept (tested): |gap| < bar on every run and |gap| < 1e-11 on the README's $-scale runs;
sub-split gaps < bar; a zero-inventory run has INV = 0 exactly; RESID is small relative to the gross
inventory move at 1-second steps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import fsum

import numpy as np

from quotesim.adverse import horizon_steps
from quotesim.fair import YEAR_SECONDS

GAP_BAR = 1e-9  # absolute floor of the identity bar ($)
GAP_REL = 1e-12  # relative part: bar = max(GAP_BAR, GAP_REL * scale), scale = gross $ of the summed products

TERMS = (
    "realised", "spread", "inventory", "hedge", "hedge_cost", "opening", "adverse_h", "drift_h", "theta",
    "inventory_ex_theta", "greek_delta", "greek_gamma", "greek_vega", "greek_theta", "resid", "gap", "scale",
)


@dataclass(frozen=True)
class Attribution:
    """All terms in $; `h` in seconds. See the module docstring for every definition."""

    spread: float
    inventory: float
    hedge: float
    hedge_cost: float
    realised: float
    gap: float
    opening: float
    adverse_h: float
    drift_h: float
    theta: float
    inventory_ex_theta: float
    greek_delta: float
    greek_gamma: float
    greek_vega: float
    greek_theta: float
    resid: float
    h: float
    scale: float = 0.0

    @property
    def bar(self) -> float:
        """The identity bar of this run: max(GAP_BAR, GAP_REL * scale)."""
        return max(GAP_BAR, GAP_REL * self.scale)

    @property
    def split_gap_fills(self) -> float:
        return self.inventory - fsum([self.opening, self.adverse_h, self.drift_h])

    @property
    def split_gap_theta(self) -> float:
        return self.inventory - fsum([self.theta, self.inventory_ex_theta])

    def as_dict(self) -> dict:
        return asdict(self)

    def waterfall(self) -> str:
        """Fixed-width text: the identity, the two sub-splits, the Greek layer."""
        rows = [
            ("spread", self.spread), ("inventory", self.inventory), ("hedge", self.hedge),
            ("- hedge cost", -self.hedge_cost), ("= realised", self.realised), ("gap", self.gap),
            ("", None),
            ("inventory =", None), ("  opening", self.opening), (f"  adverse ({self.h:g} s)", self.adverse_h),
            (f"  drift ({self.h:g} s)", self.drift_h), ("  split gap", self.split_gap_fills),
            ("inventory =", None), ("  theta", self.theta), ("  ex-theta", self.inventory_ex_theta),
            ("  split gap", self.split_gap_theta),
            ("", None),
            ("greek layer", None), ("  delta dS", self.greek_delta), ("  1/2 gamma dS^2", self.greek_gamma),
            ("  vega dsigma", self.greek_vega), ("  theta dt", self.greek_theta), ("  RESID", self.resid),
        ]
        out = []
        for name, val in rows:
            if val is None:
                out.append(name)
            elif name.endswith("gap") or name.endswith("RESID"):
                out.append(f"{name:<22}{val:>+16.2e}")
            else:
                out.append(f"{name:<22}{val:>+16.6f}")
        return "\n".join(out)


def _fills_columns(fills):
    step = fills["step"].to_numpy(dtype=np.int64)
    inst = fills["instrument"].to_numpy(dtype=np.int64)
    dq = fills["sign"].to_numpy(dtype=float) * fills["qty"].to_numpy(dtype=float)
    px = fills["px"].to_numpy(dtype=float)
    fair = fills["fair_at_fill"].to_numpy(dtype=float)
    return step, inst, dq, px, fair


def attribute(run, h: float = 60.0) -> Attribution:
    """Attribution of a `sim.Run` at markout horizon h seconds (clipped at the end of the run).
    Every sum is a `math.fsum` over the elementwise products, so the terms are exact to ~1e-12 on $-scale
    runs and the identity gap is a rounding residual, not a model one."""
    path = run.path
    F, S = path.prices, path.spot
    mult = run.multiplier
    n = path.n_steps
    pos = run.positions
    hpos = run.hedge_pos
    held = pos[1:]  # (n, m): positions held over step i
    h_held = hpos[1:]
    dF = np.diff(F, axis=0)
    dS = np.diff(S)
    dt_y = run.config.dt / YEAR_SECONDS

    step, inst, dq, px, fair = _fills_columns(run.fills)
    spread = fsum(dq * (fair - px) * mult[inst])
    inventory = fsum((held * dF * mult[None, :]).ravel())
    hedge = fsum(h_held * dS)
    hedge_cost = fsum(run.hedges["cost"].to_numpy(dtype=float))
    marks_T = fsum(np.concatenate([pos[n] * F[n] * mult, [hpos[n] * S[n]]]))
    marks_0 = fsum(np.concatenate([pos[0] * F[0] * mult, [hpos[0] * S[0]]]))
    realised = fsum([run.cash_T, marks_T, -marks_0])
    gap = realised - fsum([spread, inventory, hedge, -hedge_cost])

    h_steps = horizon_steps(h, run.config.dt)
    ahead = np.minimum(step + h_steps, n)
    opening = fsum(pos[0] * (F[n] - F[0]) * mult)
    adverse = fsum(dq * (F[ahead, inst] - F[step, inst]) * mult[inst])
    drift = fsum(dq * (F[n, inst] - F[ahead, inst]) * mult[inst])

    th = path.thetas[:-1] * dt_y
    theta = fsum((held * th * mult[None, :]).ravel())
    inv_ex_theta = fsum((held * (dF - th) * mult[None, :]).ravel())

    g_delta = fsum((held * path.deltas[:-1] * dS[:, None] * mult[None, :]).ravel())
    g_gamma = fsum((held * 0.5 * path.gammas[:-1] * dS[:, None] ** 2 * mult[None, :]).ravel())
    g_vega = fsum((held * path.vegas[:-1] * np.diff(path.vols, axis=0) * mult[None, :]).ravel())
    resid = inventory - fsum([g_delta, g_gamma, g_vega, theta])

    hedge_sh = run.hedges["shares"].to_numpy(dtype=float)
    hedge_S = run.hedges["S"].to_numpy(dtype=float)
    scale = fsum(np.concatenate([
        np.abs(dq * px * mult[inst]), np.abs(dq * (fair - px) * mult[inst]), np.abs(hedge_sh * hedge_S),
        run.hedges["cost"].to_numpy(dtype=float), np.abs(held * dF * mult[None, :]).ravel(), np.abs(h_held * dS),
        np.abs(pos[n] * F[n] * mult), np.abs(pos[0] * F[0] * mult), [abs(hpos[n] * S[n]), abs(hpos[0] * S[0])],
    ]))

    return Attribution(
        spread=spread, inventory=inventory, hedge=hedge, hedge_cost=hedge_cost, realised=realised, gap=gap,
        opening=opening, adverse_h=adverse, drift_h=drift, theta=theta, inventory_ex_theta=inv_ex_theta,
        greek_delta=g_delta, greek_gamma=g_gamma, greek_vega=g_vega, greek_theta=theta, resid=resid, h=float(h),
        scale=scale,
    )


def check_identity(att: Attribution, bar: float | None = None) -> None:
    """Raise AssertionError with the numbers if the top identity or a sub-split is off by >= bar
    (default: the run's own `att.bar` = max(GAP_BAR, GAP_REL * scale))."""
    bar = att.bar if bar is None else bar
    if not abs(att.gap) < bar:
        raise AssertionError(
            f"attribution identity broken: realised {att.realised:.12f} != spread {att.spread:.12f} + inventory "
            f"{att.inventory:.12f} + hedge {att.hedge:.12f} - hedge_cost {att.hedge_cost:.12f} (gap {att.gap:.3e}, bar {bar:.3e})"
        )
    if not abs(att.split_gap_fills) < bar:
        raise AssertionError(f"fill split broken: inventory {att.inventory:.12f} vs opening {att.opening:.12f} + adverse "
                             f"{att.adverse_h:.12f} + drift {att.drift_h:.12f} (gap {att.split_gap_fills:.3e})")
    if not abs(att.split_gap_theta) < bar:
        raise AssertionError(f"theta split broken: inventory {att.inventory:.12f} vs theta {att.theta:.12f} + ex-theta "
                             f"{att.inventory_ex_theta:.12f} (gap {att.split_gap_theta:.3e})")
