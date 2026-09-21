"""`quotesim report`: the five README tables from fixed seeds, rendered as markdown between markers.

Everything here is synthetic and seeded; the tables regenerate byte-identically on one machine and are
printed at a precision (3-5 significant figures, gaps as a power-of-ten ceiling) chosen so that last-ulp
differences between platforms do not change the text, with one exception: a gap ceiling can print one decade
apart on two CPUs when the gap sits at a decade boundary, so `quotesim report --check` (what CI runs) accepts
a one-decade difference in a ceiling and nothing else. No number in the README comes from anywhere else.

Sections (marker name -> what):
- identity : the attribution identity over N seeds, half price-space and half vol-space, jumps on:
             the count of runs with |gap| < 1e-9 (a measurement: the run-time bar is max(1e-9, 1e-12 x the
             run's gross $), above 1e-9 at this scale), the power-of-ten ceiling of max |gap|, of the two
             sub-split gaps and of the bar, and the Greek-layer residual relative to the gross inventory move.
- as2008   : Avellaneda-Stoikov 2008 Tables 1-3 with THEIR parameters and THEIR fill rule (probability
             lambda(delta) dt per step, one unit per side, arithmetic Brownian mid, P&L marked to mid),
             vectorised over n_sims paths with common random numbers between the inventory and the
             symmetric strategy; the paper's numbers beside ours.
- sweep    : paired differences VolSpace - PriceSpace(GLFT asymptotic) on the synthetic strip over
             gamma x informed fraction x hedge band, per identity term. A v0.1 SENSITIVITY table, not the
             note's result (that needs >= 200 seeds and a pre-registered metric, see docs/PLAN.md).
- markouts : the post-trade review of one run: markout by counterparty class at 1 s / 10 s / 1 min / 5 min.
- ww       : the Whalley-Wilmott band for the strip's ATM option beside a simulated band -> hedge-cost /
             residual-delta frontier.

Units: $ per contract (multiplier 1), vol points, seconds; the A-S table is in the paper's own price units.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from quotesim.adverse import markouts, toxicity
from quotesim.fair import SyntheticFair
from quotesim.flow import FlowParams
from quotesim.hedger import BandHedger, CostModel, band_ww
from quotesim.quoter import AS2008, PriceSpace, VolSpace, match_spreads, per_sqrt_second
from quotesim.sim import SimConfig, paired, paired_stats, run

BEGIN = "<!-- quotesim:begin:{name} -->"
END = "<!-- quotesim:end:{name} -->"
SECTIONS = ("identity", "as2008", "sweep", "markouts", "ww")

# ----------------------------------------------------------------------------------------------------------
# the synthetic strip and the default flow (every table uses these unless it says otherwise)
# ----------------------------------------------------------------------------------------------------------

STRIP = dict(S0=100.0, sigma0=0.2, skew_s=-0.1, curv_c=0.3, alpha=0.6, spot_vol=0.2,
             strikes=(90.0, 95.0, 100.0, 105.0, 110.0), expiries=(30.0 / 365.0,))
FLOW = dict(A=0.5, k=20.0, p_informed=1.0, h_info=60.0)
ALPHA_S, SIGMA_S, TAU_S = per_sqrt_second(0.6), per_sqrt_second(0.2), 600.0
GAMMAS = (1.0, 10.0, 50.0)
INFORMED = (0.0, 0.1, 0.3)
BANDS = (10.0, 100.0)
WW_BANDS = (0.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 1000.0)


@dataclass(frozen=True)
class ReportSizes:
    """Seeds and horizons of each section; `small()` is what the tests use."""

    identity_seeds: int = 100
    identity_seconds: float = 600.0
    as_sims: int = 1000
    sweep_seeds: int = 16
    sweep_seconds: float = 600.0
    markout_seconds: float = 3600.0
    ww_seeds: int = 8
    ww_seconds: float = 600.0

    @classmethod
    def small(cls) -> ReportSizes:
        return cls(identity_seeds=4, identity_seconds=60.0, as_sims=50, sweep_seeds=3, sweep_seconds=60.0,
                   markout_seconds=120.0, ww_seeds=2, ww_seconds=60.0)


def strip(jump=None) -> SyntheticFair:
    return SyntheticFair(**STRIP, jump=jump)


def price_quoter(gamma: float, seconds: float, kind: str = "glft_asym") -> PriceSpace:
    return PriceSpace(kind=kind, gamma=gamma, k=FLOW["k"], A=FLOW["A"], T=seconds, alpha_s=ALPHA_S, sigma_s=SIGMA_S, tau_s=TAU_S)


def vol_quoter(gamma: float, price_q: PriceSpace, fair: SyntheticFair) -> VolSpace:
    """VolSpace matched to `price_q` at q = 0 on the strip's initial snapshot."""
    vs0 = VolSpace(gamma=gamma, alpha_s=ALPHA_S, sigma_s=SIGMA_S, tau_s=TAU_S)
    matched, _ = match_spreads(price_q, vs0, fair.snapshot())
    return matched


# ----------------------------------------------------------------------------------------------------------
# formatting
# ----------------------------------------------------------------------------------------------------------


def pow10_ceiling(x: float) -> str:
    """'< 1e-12' for 3.6e-13: the power-of-ten ceiling; '0' for an exact zero. A last-ulp gap that sits at a decade
    boundary prints one decade apart on two CPUs (9.9e-13 on one summation kernel, 1.01e-12 on another: CI saw
    exactly this between two runs of the same commit), which is why `check_readme` tolerates one decade here."""
    x = abs(float(x))
    if x == 0.0:
        return "0"
    return f"< 1e{math.floor(math.log10(x)) + 1:d}"


CEILING = re.compile(r"^<?\s*1e(-?\d+)$")


def blocks_match(committed: str, fresh: str, decades: int = 1) -> list[str]:
    """Token-by-token comparison of two rendered blocks: every token must be identical, except that a
    power-of-ten ceiling (`1e-12` after a `<`) may differ by at most `decades`. Returns the mismatches."""
    a, b = committed.split(), fresh.split()
    bad = []
    if len(a) != len(b):
        bad.append(f"token count {len(a)} vs {len(b)}")
    for x, y in zip(a, b, strict=False):
        if x == y:
            continue
        mx, my = CEILING.match(x), CEILING.match(y)
        if mx and my and abs(int(mx.group(1)) - int(my.group(1))) <= decades:
            continue
        bad.append(f"{x!r} vs {y!r}")
    return bad


def md_table(df: pd.DataFrame, fmt: dict | None = None, default: str = "{:.3f}", index: bool = False) -> str:
    """Markdown table without tabulate; `fmt` maps column -> format string."""
    fmt = fmt or {}
    cols = list(df.columns)
    header = ([df.index.name or ""] if index else []) + [str(c) for c in cols]
    is_text = [pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]) for c in cols]
    align = (["---"] if index else []) + ["---" if text else "---:" for text in is_text]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    columns = [df[c].tolist() for c in cols]  # per column: dtypes survive (iterrows would upcast ints)
    for idx, *row in zip(df.index, *columns, strict=True):
        cells = [str(idx)] if index else []
        for v in row:
            if isinstance(v, str):
                cells.append(v)
            elif v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("")
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{int(v):d}")
            else:
                cells.append(fmt.get(cols[len(cells) - int(index)], default).format(float(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------
# (1) identity
# ----------------------------------------------------------------------------------------------------------


def identity_check(sizes: ReportSizes | None = None) -> pd.DataFrame:
    """Half the seeds on PriceSpace(glft_asym), half on the matched VolSpace; Merton jumps on."""
    sizes = sizes or ReportSizes()
    fair = strip(jump=(20000.0, -0.01, 0.01))
    params = FlowParams(informed_frac=0.1, **FLOW)
    cfg = SimConfig(seconds=sizes.identity_seconds, h_markout=60.0)
    gamma = 10.0
    ps = price_quoter(gamma, sizes.identity_seconds)
    quoters = {"PriceSpace glft_asym": ps, "VolSpace matched": vol_quoter(gamma, ps, fair)}
    half = max(1, sizes.identity_seeds // 2)
    rows = []
    for i, (name, q) in enumerate(quoters.items()):
        seeds = range(i * half, (i + 1) * half)
        gaps, fills_gap, theta_gap, resid_rel, fills, bars = [], [], [], [], [], []
        for seed in seeds:
            r = run(replace(cfg, seed=seed), q, BandHedger(10.0), fair, params)
            a = r.attribution
            gross = float(np.sum(np.abs(r.positions[1:] * np.diff(r.path.prices, axis=0))))
            gaps.append(abs(a.gap))
            fills_gap.append(abs(a.split_gap_fills))
            theta_gap.append(abs(a.split_gap_theta))
            resid_rel.append(abs(a.resid) / gross if gross > 0 else 0.0)
            fills.append(r.n_fills)
            bars.append(a.bar)
        rows.append({
            "quoter": name, "seeds": f"{seeds.start}-{seeds.stop - 1}", "runs with |gap| < 1e-9": f"{sum(g < 1e-9 for g in gaps)}/{len(gaps)}",
            "max |gap|": pow10_ceiling(max(gaps)), "max fill-split gap": pow10_ceiling(max(fills_gap)),
            "max theta-split gap": pow10_ceiling(max(theta_gap)), "run-time bar": pow10_ceiling(max(bars)),
            "max |RESID| / gross inventory move": pow10_ceiling(max(resid_rel)), "fills per run (mean)": float(np.mean(fills)),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------
# (2) Avellaneda-Stoikov 2008 Tables 1-3
# ----------------------------------------------------------------------------------------------------------

AS_PARAMS = dict(s0=100.0, T=1.0, sigma=2.0, dt=0.005, k=1.5, A=140.0)
AS_PAPER = {  # (avg spread, P&L mean, P&L sd, final q mean, final q sd) from the paper's Tables 1-3
    0.1: {"inventory": (1.49, 65.0, 6.6, 0.08, 2.9), "symmetric": (1.49, 68.4, 12.7, 0.26, 8.4)},
    0.01: {"inventory": (1.35, 68.6, 8.7, 0.12, 5.1), "symmetric": (1.35, 68.8, 12.8, 0.09, 8.7)},
    1.0: {"inventory": (3.02, 31.4, 5.0, 0.02, 1.7), "symmetric": (3.02, 44.0, 11.0, 0.00, 5.1)},
}


def as_benchmark(gamma: float, n_sims: int = 1000, seed: int = 0) -> dict:
    """One A-S 2008 experiment vectorised over n_sims paths: the inventory strategy (AS2008 quotes) and
    the symmetric one (the inventory strategy's time-average spread, centred on the mid), same mid path
    and same fill uniforms for both. Returns per-strategy (avg_spread, pnl mean/sd, q mean/sd)."""
    p = AS_PARAMS
    n = int(round(p["T"] / p["dt"]))
    gen = np.random.default_rng(seed)
    z = gen.standard_normal((n, n_sims))
    u = gen.random((n, n_sims, 2))
    quoter = AS2008(gamma, p["k"], p["sigma"], p["T"])
    times = np.arange(n) * p["dt"]
    avg_spread = float(np.mean([quoter.spread(t) for t in times]))
    out = {}
    for strategy in ("inventory", "symmetric"):
        s = np.full(n_sims, p["s0"])
        q = np.zeros(n_sims)
        cash = np.zeros(n_sims)
        for i in range(n):
            if strategy == "inventory":
                d_b, d_a = quoter.offsets(q, times[i])
            else:
                d_b = d_a = np.full(n_sims, avg_spread / 2.0)
            p_b = np.minimum(1.0, p["A"] * np.exp(-p["k"] * d_b) * p["dt"])
            p_a = np.minimum(1.0, p["A"] * np.exp(-p["k"] * d_a) * p["dt"])
            hit_b, hit_a = u[i, :, 0] < p_b, u[i, :, 1] < p_a
            cash = cash - hit_b * (s - d_b) + hit_a * (s + d_a)
            q = q + hit_b - hit_a
            s = s + p["sigma"] * np.sqrt(p["dt"]) * z[i]
        pnl = cash + q * s
        out[strategy] = (avg_spread, float(pnl.mean()), float(pnl.std(ddof=1)), float(q.mean()), float(q.std(ddof=1)))
    return out


def as_tables(sizes: ReportSizes | None = None) -> pd.DataFrame:
    sizes = sizes or ReportSizes()
    rows = []
    for gamma in (0.01, 0.1, 1.0):
        res = as_benchmark(gamma, sizes.as_sims, seed=0)
        for strategy in ("inventory", "symmetric"):
            ours, paper = res[strategy], AS_PAPER[gamma][strategy]
            rows.append({"gamma": gamma, "strategy": strategy, "avg spread": ours[0], "P&L mean": ours[1], "P&L sd": ours[2],
                         "final q mean": ours[3], "final q sd": ours[4], "paper: spread": paper[0], "paper: P&L mean": paper[1],
                         "paper: P&L sd": paper[2], "paper: q mean": paper[3], "paper: q sd": paper[4]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------
# (3) paired sweep
# ----------------------------------------------------------------------------------------------------------

SWEEP_TERMS = ("realised", "spread", "adverse_h", "drift_h", "hedge", "hedge_cost", "n_fills", "vega_T")


def matched_half_spreads(gamma: float, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """$ half-spread per instrument at q = 0 on the strip's initial snapshot for PriceSpace(glft_asym) and
    the VolSpace matched to it (same sum, different shape across moneyness)."""
    fair = strip()
    ps = price_quoter(gamma, seconds)
    vs = vol_quoter(gamma, ps, fair)
    snap = fair.snapshot()
    zero = np.zeros(snap.n_instruments)
    b, a = ps.quotes(snap, zero, 0.0)
    bv, av = vs.quotes(snap, zero, 0.0)
    return 0.5 * (a - b), 0.5 * (av - bv)


def inside_fair_threshold(gamma: float, seconds: float) -> float:
    """Lots beyond which PriceSpace(glft_asym)'s per-side offset c + (2q + 1) w / 2 is negative on the strip's
    tightest instrument at t = 0: min_j c / w_j (c = ln(1 + gamma / k) / gamma, w_j the skew per lot of
    instrument j); the quote then sits inside fair and its fills carry a negative SPREAD term."""
    fair = strip()
    ps = price_quoter(gamma, seconds)
    snap = fair.snapshot()
    n = snap.n_instruments
    b0, _ = ps.offsets(snap, np.zeros(n), 0.0)
    b1, _ = ps.offsets(snap, np.ones(n), 0.0)
    c = np.log1p(gamma / FLOW["k"]) / gamma
    return float(np.min(c / (b1 - b0)))


def sweep(sizes: ReportSizes | None = None, gammas=GAMMAS, informed=INFORMED, bands=BANDS) -> pd.DataFrame:
    """Paired VolSpace - PriceSpace differences per (gamma, informed_frac, band): realised with its full
    paired summary, the other identity terms as medians."""
    sizes = sizes or ReportSizes()
    fair = strip()
    seeds = range(sizes.sweep_seeds)
    rows = []
    for gamma in gammas:
        ps = price_quoter(gamma, sizes.sweep_seconds)
        vs = vol_quoter(gamma, ps, fair)
        for phi in informed:
            params = FlowParams(informed_frac=phi, **FLOW)
            for band in bands:
                cfg = SimConfig(seconds=sizes.sweep_seconds, h_markout=60.0)
                pr = paired(cfg, {"price": ps, "vol": vs}, seeds, hedger=BandHedger(band), fair=fair, params=params)
                d = pr.diffs("price", "vol", SWEEP_TERMS)
                st = paired_stats(d["realised"].to_numpy())
                row = {"gamma": gamma, "informed": phi, "band $": band, "realised: median": st["median"],
                       "IQR": f"[{st['q25']:.2f}, {st['q75']:.2f}]", "p5 / p95": f"{st['p5']:.2f} / {st['p95']:.2f}",
                       "sign": f"{st['n_pos']}/{st['n_nonzero']}", "boot 95 % CI of mean": f"[{st['boot_lo']:.2f}, {st['boot_hi']:.2f}]"}
                for term in SWEEP_TERMS[1:]:
                    row[f"{term}: median"] = float(d[term].median())
                for name in ("price", "vol"):
                    v = pr.values.xs(name, level="quoter")
                    row[f"inside fair % ({name})"] = 100.0 * float(v["n_inside_fair"].sum()) / float(v["n_fills"].sum())
                rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------
# (4) markouts
# ----------------------------------------------------------------------------------------------------------


def markout_review(sizes: ReportSizes | None = None, seed: int = 3, informed_frac: float = 0.3, gamma: float = 10.0):
    sizes = sizes or ReportSizes()
    fair = strip()
    params = FlowParams(informed_frac=informed_frac, **FLOW)
    ps = price_quoter(gamma, sizes.markout_seconds)
    r = run(SimConfig(seconds=sizes.markout_seconds, seed=seed, h_markout=60.0), ps, BandHedger(10.0), fair, params)
    mk = markouts(r, horizons=(1.0, 10.0, 60.0, 300.0))
    tox = toxicity(r).reset_index()
    return mk, tox, r


# ----------------------------------------------------------------------------------------------------------
# (5) Whalley-Wilmott band vs the simulated hedge-cost frontier
# ----------------------------------------------------------------------------------------------------------


def ww_frontier(sizes: ReportSizes | None = None, gamma: float = 10.0, informed_frac: float = 0.1, bands=WW_BANDS):
    """Rows over the band: hedges per run, hedge cost, RMS residual $ delta, realised mean and sd across
    seeds. Plus the WW half-band for the ATM call at lam = half_spread / S for each sweep gamma."""
    sizes = sizes or ReportSizes()
    fair = strip()
    params = FlowParams(informed_frac=informed_frac, **FLOW)
    ps = price_quoter(gamma, sizes.ww_seconds)
    rows = []
    for band in bands:
        hedges, cost, rms, realised = [], [], [], []
        for seed in range(sizes.ww_seeds):
            r = run(SimConfig(seconds=sizes.ww_seconds, seed=seed), ps, BandHedger(band), fair, params)
            target = -(r.positions[1:] * r.path.deltas[:-1] * r.multiplier[None, :]).sum(axis=1)
            resid = (r.hedge_pos[1:] - target) * r.path.spot[:-1]
            hedges.append(r.n_hedges)
            cost.append(r.attribution.hedge_cost)
            rms.append(float(np.sqrt(np.mean(resid**2))))
            realised.append(r.attribution.realised)
        rows.append({"band $ delta": band, "hedges per run": float(np.mean(hedges)), "hedge cost": float(np.mean(cost)),
                     "RMS residual $ delta": float(np.mean(rms)), "realised mean": float(np.mean(realised)),
                     "realised sd across seeds": float(np.std(realised, ddof=1)) if len(realised) > 1 else float("nan")})
    snap = fair.snapshot()
    atm = int(np.argmin(np.abs(snap.K - snap.spot) + (snap.right != "C") * 1e9))
    lam = CostModel().half_spread / snap.spot
    ww_rows = [{"gamma": g, "lam = half_spread / S": lam, "Gamma (ATM call)": float(snap.gammas[atm]),
                "WW half-band, shares per lot": band_ww(snap.spot, float(snap.gammas[atm]), lam, g, tau=float(snap.T_rem[atm])),
                "WW half-band, $ delta per lot": band_ww(snap.spot, float(snap.gammas[atm]), lam, g, tau=float(snap.T_rem[atm])) * snap.spot}
               for g in GAMMAS]
    return pd.DataFrame(rows), pd.DataFrame(ww_rows)


# ----------------------------------------------------------------------------------------------------------
# build, render, README
# ----------------------------------------------------------------------------------------------------------


def build(sizes: ReportSizes | None = None) -> dict:
    sizes = sizes or ReportSizes()
    mk, tox, mk_run = markout_review(sizes)
    frontier, ww = ww_frontier(sizes)
    return {"sizes": sizes, "identity": identity_check(sizes), "as2008": as_tables(sizes), "sweep": sweep(sizes),
            "markouts": mk, "toxicity": tox, "markout_run": mk_run, "frontier": frontier, "ww": ww}


def _strip_line() -> str:
    return (f"S0 {STRIP['S0']:g}, ATM vol {STRIP['sigma0']:g}, skew {STRIP['skew_s']:g}, curvature {STRIP['curv_c']:g}, "
            f"vol-of-vol {STRIP['alpha']:g} / sqrt(yr) with kappa_vol {STRIP.get('kappa_vol', 0.0):g} (a floored random walk), "
            f"spot vol {STRIP['spot_vol']:g}, strikes {list(STRIP['strikes'])}, "
            f"30-day expiry (10 instruments); flow A {FLOW['A']:g} /s/side/instrument, k {FLOW['k']:g} /$, informed horizon {FLOW['h_info']:g} s, p {FLOW['p_informed']:g}")


def render(res: dict) -> dict[str, str]:
    """Marker name -> markdown block (tables plus their one-line captions)."""
    s: ReportSizes = res["sizes"]
    out = {}
    out["identity"] = (
        f"{md_table(res['identity'], default='{:.1f}')}\n\n"
        f"{s.identity_seeds} seeds ({s.identity_seconds:g} s each, 1 s steps) on the strip ({_strip_line()}), Merton jumps on "
        f"(20,000 / yr, mean -1 %, sd 1 %), informed fraction 0.1, gamma 10, band $10. Gaps are power-of-ten ceilings of the "
        f"largest absolute gap over the seeds; the run-time bar is max(1e-9, 1e-12 x the run's gross $ of summed products), "
        f"its ceiling in the bar column, and `sim.run` raises above it, so the 1e-9 count is a measurement below the bar. "
        f"RESID is the Greek layer's residual (vanna, volga, jumps) over the gross inventory move sum |q dF|."
    )
    out["as2008"] = (
        f"{md_table(res['as2008'], fmt={'gamma': '{:g}', 'avg spread': '{:.2f}', 'P&L mean': '{:.1f}', 'P&L sd': '{:.1f}', 'final q mean': '{:.2f}', 'final q sd': '{:.1f}', 'paper: spread': '{:.2f}', 'paper: P&L mean': '{:.1f}', 'paper: P&L sd': '{:.1f}', 'paper: q mean': '{:.2f}', 'paper: q sd': '{:.1f}'})}\n\n"
        f"Their parameters (s = 100, T = 1, sigma = 2, dt = 0.005, k = 1.5, A = 140, q0 = 0), their fill rule (probability "
        f"lambda(delta) dt per step and side, one unit per fill), their symmetric benchmark (the inventory strategy's "
        f"time-average spread centred on the mid), arithmetic Brownian mid, P&L marked to the mid; {s.as_sims} paths per "
        f"gamma, seed 0, the same mid path and fill uniforms for both strategies. The paper's RNG is not published, so "
        f"only the direction of each effect and the rough magnitude are reproducible."
    )
    hs_p, hs_v = matched_half_spreads(10.0, s.sweep_seconds)
    q_inside = " / ".join(f"{inside_fair_threshold(g, s.sweep_seconds):.1f}" for g in GAMMAS)
    sw = res["sweep"]
    hit = sorted(set(sw.loc[(sw["inside fair % (price)"] > 0) | (sw["inside fair % (vol)"] > 0), "gamma"]))
    hit_line = f"the gamma {' and '.join(f'{g:g}' for g in hit)} cells reach that regime" if hit else "no cell reaches that regime"
    sw_disp = sw.drop(columns=["inside fair % (price)", "inside fair % (vol)"]).assign(**{
        "lots inside fair, % (price / vol)": [f"{a:.2f} / {b:.2f}" for a, b in zip(sw["inside fair % (price)"], sw["inside fair % (vol)"], strict=True)]})
    sw_fmt = {"gamma": "{:g}", "informed": "{:g}", "band $": "{:g}", "realised: median": "{:+.2f}"}
    sw_fmt.update({f"{t}: median": "{:+.2f}" for t in SWEEP_TERMS[1:]})
    out["sweep"] = (
        f"{md_table(sw_disp, fmt=sw_fmt)}\n\n"
        f"v0.1 sensitivity table, not the note's result: {s.sweep_seeds} paired seeds x {s.sweep_seconds:g} s per cell, "
        f"VolSpace matched to PriceSpace(GLFT asymptotic) at q = 0 (same summed $ half-spread and the same summed linearised "
        f"$ skew per lot), differences are vol minus price per seed in $; markout horizon 60 s; the hedge is a $-delta band "
        f"with the default sqrt-law cost model. Sign = seeds with a positive difference / seeds with a nonzero one; the CI "
        f"is a 2,000-resample bootstrap of the mean. n_fills is a count and vega_T the terminal net vega in $ per 1.00 vol; "
        f"neither is a $ P&L term. The matching equates the SUM of "
        f"the $ half-spreads, not their shape: at gamma 10 and q = 0 the price-space half-spread is "
        f"{hs_p.min():.3f}-{hs_p.max():.3f} $ on every strike while the matched vol-space one runs {hs_v.min():.3f} $ at the "
        f"wings to {hs_v.max():.3f} $ at the money, so the vol quoter fills the wings more often at less capture each; the "
        f"spread column, not the inventory skew, carries most of every cell. The last column is the share of lots filled at "
        f"a quote INSIDE fair (the per-side offset c + (2q + 1) w / 2 is negative beyond |q| = c / w, which is "
        f"{q_inside} lots on the tightest instrument at gamma {' / '.join(f'{g:g}' for g in GAMMAS)}); those fills carry a "
        f"negative SPREAD term, the maker paying to unwind; {hit_line}."
    )
    mk_fmt = {"horizon_s": "{:g}", "realised_spread_mean": "{:.4f}", "adverse_mean": "{:+.4f}", "markout_mean": "{:+.4f}",
              "se_clustered": "{:.4f}", "realised_spread_vp": "{:.3f}", "adverse_vp": "{:+.3f}", "markout_vp": "{:+.3f}", "se_clustered_vp": "{:.3f}"}
    mk_df = res["markouts"].drop(columns=["n_vp"])
    tox_fmt = {"share": "{:.3f}", "realised_spread_mean": "{:.4f}", "adverse_mean": "{:+.4f}", "markout_mean": "{:+.4f}", "give_back": "{:+.3f}", "frac_negative": "{:.3f}"}
    r = res["markout_run"]
    out["markouts"] = (
        f"{md_table(mk_df, fmt=mk_fmt)}\n\nToxicity at the 60 s horizon:\n\n{md_table(res['toxicity'], fmt=tox_fmt)}\n\n"
        f"One run, seed {r.config.seed}, {r.config.seconds:g} s, informed fraction 0.3 (p = 1, 60 s look-ahead), gamma 10, "
        f"PriceSpace(GLFT asymptotic), band $10; {r.n_fills} fills. Markout = s_f (F_(t+h) - px_f) against fair, $ per contract "
        f"(`_mean`) and vol points via the fill's vega (`_vp`); SE clustered by time bucket of width h. The horizons are "
        f"synthetic (the seeded fair process, not a recorded tape). give_back = -adverse / realised spread."
    )
    fr_fmt = {"band $ delta": "{:g}", "hedges per run": "{:.1f}", "hedge cost": "{:.2f}", "RMS residual $ delta": "{:.2f}",
              "realised mean": "{:+.2f}", "realised sd across seeds": "{:.2f}"}
    ww_fmt = {"gamma": "{:g}", "lam = half_spread / S": "{:.1e}", "Gamma (ATM call)": "{:.4f}", "WW half-band, shares per lot": "{:.4f}", "WW half-band, $ delta per lot": "{:.2f}"}
    out["ww"] = (
        f"{md_table(res['frontier'], fmt=fr_fmt)}\n\n{md_table(res['ww'], fmt=ww_fmt)}\n\n"
        f"Frontier: PriceSpace(GLFT asymptotic, gamma 10), informed fraction 0.1, {s.ww_seeds} seeds x {s.ww_seconds:g} s per "
        f"band, default cost model (half-spread $0.01 / share, Y 0.5, daily vol 1 %, ADV 5e7). Whalley-Wilmott: H = (3/2 "
        f"lam S Gamma^2 / gamma)^(1/3) shares per option for the ATM call, lam = proportional cost = half-spread / S, r = 0; "
        f"it is a per-lot no-trade half-band, the frontier's band is the book's $-delta mismatch, so they are compared by "
        f"eye, not equated."
    )
    return out


def replace_sections(text: str, blocks: dict[str, str]) -> str:
    """Replace what stands between each pair of markers (the markers stay). Missing markers raise."""
    for name, body in blocks.items():
        b, e = BEGIN.format(name=name), END.format(name=name)
        pattern = re.compile(re.escape(b) + r".*?" + re.escape(e), re.DOTALL)
        if not pattern.search(text):
            raise ValueError(f"README has no markers for section '{name}'")
        text = pattern.sub(lambda _m, rep=f"{b}\n{body}\n{e}": rep, text)
    return text


def read_sections(text: str) -> dict[str, str]:
    """The body between each pair of markers, by section name."""
    out = {}
    for name in SECTIONS:
        b, e = BEGIN.format(name=name), END.format(name=name)
        m = re.search(re.escape(b) + r"\n(.*?)\n" + re.escape(e), text, re.DOTALL)
        if not m:
            raise ValueError(f"README has no markers for section '{name}'")
        out[name] = m.group(1)
    return out


def check_readme(path, sizes: ReportSizes | None = None, decades: int = 1) -> dict[str, list[str]]:
    """Regenerate the blocks and compare them with the README's: {section: mismatches}, empty when they agree.
    Everything must match token for token except a gap ceiling, which may differ by `decades`."""
    with open(path, encoding="utf-8") as f:
        have = read_sections(f.read())
    fresh = render(build(sizes or ReportSizes()))
    return {name: bad for name in SECTIONS if (bad := blocks_match(have[name], fresh[name], decades))}


def update_readme(path, sizes: ReportSizes | None = None) -> dict:
    """Regenerate every marked block of the README in place; returns the rendered blocks."""
    sizes = sizes or ReportSizes()
    res = build(sizes)
    blocks = render(res)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    new = replace_sections(text, blocks)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new)
    return blocks
