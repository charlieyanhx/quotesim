"""The fixed-step event loop, the run record, and the paired-seed harness.

Units and conventions (SECONDS on the clock, $ per unit of underlying times the instrument multiplier, lots
+ = the market maker is long):
- one step of `dt` seconds: quote update (the quoter reads the past-only fair at the clock and the
  current inventory) -> fills (flow.arrivals thins the pre-drawn candidates against the quotes; the
  informed counterparty reads the fair h_info ahead on the pre-drawn path) -> hedge check (the rule sees the
  option delta at the clock; `t_last` is the last time the rule was DUE, whether or not the mismatch was
  zero, so a TimeHedger's grid is 0, every_s, 2 every_s, ... from the start of the run) -> state step (the
  clock advances; positions are marked from F_t to F_{t+1}).
  Fills at step i are priced at the quotes of step i and marked at F_i; the position they create is held
  over step i -> i + 1 (that is why `positions[i + 1]` is "held over step i").
- cash: an option fill moves cash by -sign * lots * px * multiplier; a hedge trade by -dh * S - cost (the
  hedge fills at the mid, the whole slippage is `cost`, see hedger.py). `cash_T` is a `math.fsum` over the
  flows so pnl.py's identity is a rounding residual and not a running-sum artefact; `cash` is the running
  path for diagnostics.
- one `numpy.random.Generator` per seed draws every stream up front in a fixed order (flow.Streams), so
  two quoters under the same seed see the same spot / vol / jump path, the same candidate arrivals, the
  same uniforms, the same informed flags and directions; they differ only through thinning and their own
  hedges (common random numbers by construction; a tighter quote's fills are a superset of a wider one's).
- the identity |gap| < bar = max(1e-9, 1e-12 x the run's gross $) (pnl.check_identity) is asserted inside `run`,
  on every run, not only in tests.
- `paired` keeps only the attribution and a few counts per (seed, quoter), never the runs; `PairedResult.table`
  reports paired differences per identity term: median, IQR, p5 / p95, sign count with its two-sided
  binomial p, and a seeded bootstrap 95 % CI of the mean. Never the best seed; never an unpaired t-test.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import fsum

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from quotesim import flow, pnl
from quotesim.fair import YEAR_SECONDS, FairPath, PastOnlyFair, SyntheticFair
from quotesim.flow import MM_SIGN, FlowParams, StepFills, Streams
from quotesim.hedger import Hedger, target_shares
from quotesim.quoter import Quoter

CLASS_NAMES = ("uninformed", "informed")
FILL_COLUMNS = ("t", "step", "instrument", "side", "sign", "qty", "px", "fair_at_fill", "class")
HEDGE_COLUMNS = ("t", "step", "shares", "cost", "S", "h_after")


@dataclass(frozen=True)
class SimConfig:
    """seconds / dt on the clock; seed for every stream; h_markout = canonical adverse-selection horizon (s);
    q0 = opening option inventory (lots per instrument, None = flat); h0 = opening hedge shares."""

    seconds: float = 3600.0
    dt: float = 1.0
    seed: int = 0
    h_markout: float = 60.0
    q0: tuple[float, ...] | None = None
    h0: float = 0.0

    def __post_init__(self):
        if self.seconds <= 0 or self.dt <= 0 or self.h_markout <= 0:
            raise ValueError("seconds, dt, h_markout must be > 0")
        if self.n_steps < 1:
            raise ValueError("seconds / dt must be >= 1 step")

    @property
    def n_steps(self) -> int:
        return int(round(self.seconds / self.dt))


@dataclass(frozen=True)
class Run:
    """Everything pnl.py and adverse.py need. Arrays: positions (n_steps + 1, m) with row 0 = q0 and row i + 1
    = lots held over step i; hedge_pos (n_steps + 1,) likewise; cash (n_steps + 1,) running; bids / asks
    (n_steps, m) the quotes of each step (NaN = pulled); fills / hedges DataFrames with FILL_COLUMNS /
    HEDGE_COLUMNS; n_candidates = total candidate arrivals of the seed (identical across quoters)."""

    config: SimConfig
    path: FairPath
    multiplier: np.ndarray
    positions: np.ndarray
    hedge_pos: np.ndarray
    cash: np.ndarray
    cash_T: float
    bids: np.ndarray
    asks: np.ndarray
    fills: pd.DataFrame
    hedges: pd.DataFrame
    n_candidates: int
    attribution: pnl.Attribution | None = field(repr=False)

    @property
    def n_fills(self) -> int:
        return int(self.fills["qty"].sum()) if len(self.fills) else 0

    @property
    def n_hedges(self) -> int:
        return int(len(self.hedges))

    @property
    def n_inside_fair(self) -> int:
        """Lots filled at a quote INSIDE fair (sign * (fair - px) < 0): the per-side offset had gone negative."""
        if not len(self.fills):
            return 0
        f = self.fills
        inside = (f["sign"].to_numpy(dtype=float) * (f["fair_at_fill"].to_numpy(dtype=float) - f["px"].to_numpy(dtype=float))) < 0
        return int(f["qty"].to_numpy(dtype=np.int64)[inside].sum())

    @property
    def terminal_inventory(self) -> np.ndarray:
        return self.positions[-1]

    def summary(self) -> dict:
        """Counts and the identity terms in one flat dict (what `paired` keeps)."""
        d = self.attribution.as_dict()
        d.update(
            n_fills=self.n_fills, n_hedges=self.n_hedges, n_candidates=self.n_candidates, n_inside_fair=self.n_inside_fair,
            abs_q_T=float(np.abs(self.terminal_inventory).sum()),
            vega_T=float(np.sum(self.terminal_inventory * self.path.vegas[-1] * self.multiplier)),
        )
        return d


def _fill_rows(i: int, t: float, sf: StepFills, bid, ask, fair_now, out: list) -> None:
    """Append one row per (instrument, side, class) with fills > 0."""
    px_by_side = (bid, ask)
    for side in range(flow.N_SIDES):
        n_f, n_i = sf.fills[:, side], sf.informed[:, side]
        for j in np.flatnonzero(n_f):
            for cls, qty in ((0, int(n_f[j] - n_i[j])), (1, int(n_i[j]))):
                if qty > 0:
                    out.append((t, i, int(j), side, float(MM_SIGN[side]), qty, float(px_by_side[side][j]),
                                float(fair_now[j]), CLASS_NAMES[cls]))


def _empty_fills() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=float if c not in ("step", "instrument", "side", "qty", "class") else
                                      (object if c == "class" else np.int64)) for c in FILL_COLUMNS})


def run(config: SimConfig, quoter: Quoter, hedger: Hedger, fair: SyntheticFair, params: FlowParams) -> Run:
    """Simulate one seed. `fair` is left untouched (the path is built on a copy)."""
    n, dt = config.n_steps, config.dt
    m = fair.n_instruments
    streams = Streams.draw(config.seed, n, m, params.A, dt)
    path = fair.path(streams.spot_z, streams.vol_z, streams.jump_u, dt / YEAR_SECONDS)
    view = PastOnlyFair(path)
    mult = path.instruments["multiplier"].to_numpy(dtype=float)
    h_steps = params.h_steps(dt)

    q = np.zeros(m) if config.q0 is None else np.asarray(config.q0, dtype=float)
    if q.shape != (m,):
        raise ValueError(f"q0 must have {m} entries")
    h = float(config.h0)
    positions = np.empty((n + 1, m))
    hedge_pos = np.empty(n + 1)
    cash = np.empty(n + 1)
    bids = np.empty((n, m))
    asks = np.empty((n, m))
    positions[0], hedge_pos[0], cash[0] = q, h, 0.0
    flows: list[float] = []
    fill_rows: list = []
    hedge_rows: list = []
    prev: StepFills | None = None
    t_last: float | None = None
    running = 0.0
    for i in range(n):
        t = i * dt
        snap = view.current()
        bid, ask = quoter.quotes(snap, q, t)
        bid, ask = np.asarray(bid, dtype=float), np.asarray(ask, dtype=float)
        bids[i], asks[i] = bid, ask
        fair_now = path.prices[i]
        sf = flow.arrivals(params, streams, i, bid, ask, fair_now, path.prices[min(i + h_steps, n)], prev)
        prev = sf
        if sf.n_fills:
            q = q + sf.signed_lots()
            buy_flow = -(sf.fills[:, 0] * bid * mult)[sf.fills[:, 0] > 0]
            sell_flow = (sf.fills[:, 1] * ask * mult)[sf.fills[:, 1] > 0]
            flows.extend(buy_flow.tolist())
            flows.extend(sell_flow.tolist())
            running += float(buy_flow.sum() + sell_flow.sum())
            _fill_rows(i, t, sf, bid, ask, fair_now, fill_rows)
        S = float(path.spot[i])
        target = target_shares(q, path.deltas[i], mult)
        dec = hedger.decide(h, target, S, t, t_last)
        if dec.due:
            t_last = t
        if dec.traded:
            flows.append(-dec.shares * S - dec.cost)
            running += -dec.shares * S - dec.cost
            h = dec.new_h
            hedge_rows.append((t, i, dec.shares, dec.cost, S, h))
        positions[i + 1], hedge_pos[i + 1], cash[i + 1] = q, h, running
        view.advance()
    fills = pd.DataFrame(fill_rows, columns=list(FILL_COLUMNS)) if fill_rows else _empty_fills()
    hedges = pd.DataFrame(hedge_rows, columns=list(HEDGE_COLUMNS)) if hedge_rows else pd.DataFrame(
        {c: pd.Series(dtype=float) for c in HEDGE_COLUMNS})
    partial = Run(
        config=config, path=path, multiplier=mult, positions=positions, hedge_pos=hedge_pos, cash=cash,
        cash_T=fsum(flows), bids=bids, asks=asks, fills=fills, hedges=hedges,
        n_candidates=int(streams.candidates.sum()), attribution=None,
    )
    att = pnl.attribute(partial, h=config.h_markout)
    pnl.check_identity(att)
    return replace(partial, attribution=att)


# ----------------------------------------------------------------------------------------------------------
# paired seeds
# ----------------------------------------------------------------------------------------------------------

PAIRED_TERMS = ("realised", "spread", "inventory", "hedge", "hedge_cost", "opening", "adverse_h", "drift_h",
                "theta", "n_fills", "n_hedges", "abs_q_T", "vega_T")


def paired_stats(d: np.ndarray, n_boot: int = 2000, boot_seed: int = 0) -> dict:
    """Paired-difference summary of d = M_B - M_A over seeds: median, q25/q75, p5/p95, mean, sign count
    (#d > 0 of #d != 0) with a two-sided binomial p, seeded bootstrap 95 % CI of the mean."""
    d = np.asarray(d, dtype=float)
    if d.size == 0:
        raise ValueError("no differences")
    nonzero = d[d != 0]
    n_pos = int((nonzero > 0).sum())
    p_sign = float(binomtest(n_pos, nonzero.size, 0.5).pvalue) if nonzero.size else 1.0
    gen = np.random.default_rng(boot_seed)
    boots = d[gen.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(
        n=int(d.size), median=float(np.median(d)), q25=float(np.percentile(d, 25)), q75=float(np.percentile(d, 75)),
        p5=float(np.percentile(d, 5)), p95=float(np.percentile(d, 95)), mean=float(d.mean()),
        n_pos=n_pos, n_nonzero=int(nonzero.size), p_sign=p_sign, boot_lo=float(lo), boot_hi=float(hi),
    )


@dataclass(frozen=True)
class PairedResult:
    """`values`: index (seed, quoter), columns = Run.summary() keys. Differences are B - A for every pair
    (A, B) of quoter names in order, or against `baseline`."""

    names: tuple[str, ...]
    seeds: tuple[int, ...]
    values: pd.DataFrame

    def diffs(self, a: str, b: str, terms=PAIRED_TERMS) -> pd.DataFrame:
        """Per-seed differences b - a, index = seed, columns = terms."""
        va = self.values.xs(a, level="quoter")[list(terms)]
        vb = self.values.xs(b, level="quoter")[list(terms)]
        return (vb - va).loc[list(self.seeds)]

    def table(self, baseline: str | None = None, terms=PAIRED_TERMS, n_boot: int = 2000, boot_seed: int = 0) -> pd.DataFrame:
        """Rows = (comparison, term); columns = paired_stats keys."""
        base = self.names[0] if baseline is None else baseline
        rows = []
        for name in self.names:
            if name == base:
                continue
            d = self.diffs(base, name, terms)
            for term in terms:
                st = paired_stats(d[term].to_numpy(), n_boot, boot_seed)
                rows.append({"comparison": f"{name} - {base}", "term": term, **st})
        return pd.DataFrame(rows).set_index(["comparison", "term"])

    def levels(self, terms=PAIRED_TERMS) -> pd.DataFrame:
        """Median and IQR of each quoter's own level per term (context for the differences)."""
        g = self.values[list(terms)].groupby(level="quoter")
        out = pd.concat({"median": g.median(), "q25": g.quantile(0.25), "q75": g.quantile(0.75)}, axis=1)
        return out.loc[list(self.names)]


def paired(config: SimConfig, quoters: dict[str, Quoter], seeds, *, hedger: Hedger, fair: SyntheticFair,
           params: FlowParams, hedgers: dict[str, Hedger] | None = None) -> PairedResult:
    """Run every quoter on every seed (same streams per seed) and keep each run's summary. `hedgers`
    overrides the hedger per quoter name (default: the same `hedger` for all)."""
    if len(quoters) < 2:
        raise ValueError("paired needs >= 2 quoters")
    seeds = tuple(int(s) for s in seeds)
    rows = {}
    for seed in seeds:
        cfg = replace(config, seed=seed)
        for name, quoter in quoters.items():
            hd = hedger if hedgers is None else hedgers.get(name, hedger)
            rows[(seed, name)] = run(cfg, quoter, hd, fair, params).summary()
    values = pd.DataFrame.from_dict(rows, orient="index")
    values.index = pd.MultiIndex.from_tuples(values.index, names=["seed", "quoter"])
    return PairedResult(names=tuple(quoters), seeds=seeds, values=values)
