"""pnl.py: the four-term identity, its two exact sub-splits, the Greek layer's residual, and the guard.
Oracles: docs/PLAN.md 'Identity' (|gap| < 1e-9; sub-split gaps 0; zero-inventory quoter -> INV = 0;
RESID small at 1 s steps) and research report_a3e5dd section 4 (gaps -3.7e-12 / -7.8e-12 on realised
240.94 with 2,491 fills; split gap 0.0; Greek residual 6.2e-7 on inventory O(1) for one ATM strike)."""

from dataclasses import replace
from math import fsum

import numpy as np
import pytest

from quotesim import pnl
from quotesim.fair import YEAR_SECONDS, SyntheticFair
from quotesim.flow import FlowParams
from quotesim.hedger import BandHedger, NoHedger
from quotesim.quoter import PriceSpace, VolSpace, per_sqrt_second
from quotesim.sim import SimConfig, run

FLOW = FlowParams(A=0.5, k=20.0, informed_frac=0.2, p_informed=1.0, h_info=60.0)


def _fair(**kw):
    base = dict(S0=100.0, sigma0=0.2, skew_s=-0.1, curv_c=0.3, alpha=0.6, spot_vol=0.2,
                strikes=(90.0, 95.0, 100.0, 105.0, 110.0), expiries=(30 / 365,))
    return SyntheticFair(**{**base, **kw})


def _price_quoter(gamma=10.0, T=600.0):
    return PriceSpace(kind="glft_asym", gamma=gamma, k=20.0, A=0.5, T=T, alpha_s=per_sqrt_second(0.6),
                      sigma_s=per_sqrt_second(0.2), tau_s=600.0)


class PulledQuoter:
    """Never quotes: every fill is impossible, inventory stays at q0."""

    def quotes(self, snap, inventory, t):
        n = snap.n_instruments
        return np.full(n, np.nan), np.full(n, np.nan)


def _brute_force(r):
    """Independent re-implementation of the identity from the raw record: walk cash and marks step by step
    with plain floats (no fsum), then compare the four terms and R."""
    F, S, mult = r.path.prices, r.path.spot, r.multiplier
    n = r.path.n_steps
    cash, spread, inv, hedge, hcost = 0.0, 0.0, 0.0, 0.0, 0.0
    fills = r.fills.groupby("step")
    hedges = r.hedges.set_index("step") if len(r.hedges) else None
    for i in range(n):
        if i in fills.groups:
            for _, f in fills.get_group(i).iterrows():
                dq = f["sign"] * f["qty"]
                cash -= dq * f["px"] * mult[int(f["instrument"])]
                spread += dq * (F[i, int(f["instrument"])] - f["px"]) * mult[int(f["instrument"])]
        if hedges is not None and i in hedges.index:
            hrow = hedges.loc[i]
            cash -= hrow["shares"] * S[i] + hrow["cost"]
            hcost += hrow["cost"]
        inv += float(np.sum(r.positions[i + 1] * (F[i + 1] - F[i]) * mult))
        hedge += r.hedge_pos[i + 1] * (S[i + 1] - S[i])
    realised = cash + float(np.sum(r.positions[n] * F[n] * mult)) + r.hedge_pos[n] * S[n] - (
        float(np.sum(r.positions[0] * F[0] * mult)) + r.hedge_pos[0] * S[0])
    return dict(spread=spread, inventory=inv, hedge=hedge, hedge_cost=hcost, realised=realised)


# ----------------------------------------------------------------------------------------------------------
# the identity
# ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("space", ["price", "vol"])
def test_identity_holds_to_1e9_and_matches_a_brute_force_walk(space):
    """Plan oracle |gap| < 1e-9 (research: -3.7e-12 / -7.8e-12 on realised ~241); every term also agrees
    with an independent plain-float walk of the record to 1e-7 (the walk's own rounding)."""
    fair = _fair(jump=(20000.0, -0.01, 0.01))
    q = _price_quoter() if space == "price" else VolSpace(gamma=10.0, alpha_s=per_sqrt_second(0.6), sigma_s=per_sqrt_second(0.2), tau_s=600.0, hs_vol=0.006)
    r = run(SimConfig(seconds=600.0, seed=5, h_markout=60.0), q, BandHedger(10.0), fair, FLOW)
    a = r.attribution
    assert r.n_fills > 500 and r.n_hedges > 50
    assert abs(a.gap) < 1e-9
    assert abs(a.split_gap_fills) < 1e-9 and abs(a.split_gap_theta) < 1e-9
    assert a.realised == pytest.approx(fsum([a.spread, a.inventory, a.hedge, -a.hedge_cost]), abs=1e-9)
    bf = _brute_force(r)
    for k, v in bf.items():
        assert getattr(a, k) == pytest.approx(v, abs=1e-7), k
    assert a.spread > 0 and a.hedge_cost > 0


def test_pulled_quotes_give_zero_inventory_and_zero_realised():
    """Plan oracle: zero-inventory quoter -> INV = 0 (exactly), and with no hedger every term is 0."""
    r = run(SimConfig(seconds=120.0, seed=1), PulledQuoter(), NoHedger(), _fair(), FLOW)
    a = r.attribution
    assert r.n_fills == 0 and len(r.fills) == 0
    assert a.inventory == 0.0 and a.spread == 0.0 and a.hedge == 0.0 and a.hedge_cost == 0.0
    assert a.realised == 0.0 and a.gap == 0.0 and a.resid == 0.0
    assert np.isnan(r.bids).all() and np.isnan(r.asks).all()


def test_opening_inventory_is_its_own_term_and_the_hedge_terms_separate():
    """With q0 set and quotes pulled: INV = OPENING = q0 . (F_T - F_0), ADVERSE = DRIFT = 0, and with a
    band-0 hedger HEDGE = sum h dS, HCOST = sum cost, all of it reconciling to R."""
    fair = _fair()
    q0 = (2.0, -1.0, 0.0, 3.0, 0.0, 0.0, -2.0, 0.0, 1.0, 1.0)
    r = run(SimConfig(seconds=300.0, seed=2, q0=q0), PulledQuoter(), BandHedger(0.0), fair, FLOW)
    a = r.attribution
    F = r.path.prices
    assert a.opening == pytest.approx(fsum(np.asarray(q0) * (F[-1] - F[0])), abs=1e-12)
    assert a.inventory == pytest.approx(a.opening, abs=1e-12)
    assert a.adverse_h == 0.0 and a.drift_h == 0.0 and a.spread == 0.0
    assert r.n_hedges >= 1 and a.hedge_cost > 0
    assert a.hedge == pytest.approx(fsum(r.hedge_pos[1:] * np.diff(r.path.spot)), abs=1e-12)
    assert a.hedge_cost == pytest.approx(r.hedges["cost"].sum(), abs=1e-12)
    assert abs(a.gap) < 1e-9
    # the hedge tracks the option delta at every step (band 0) so the delta P&L is mostly offset
    assert abs(a.greek_delta + a.hedge) < 0.2 * abs(a.greek_delta) + 1e-6


# ----------------------------------------------------------------------------------------------------------
# sub-splits
# ----------------------------------------------------------------------------------------------------------


def test_fill_split_is_exact_at_every_horizon_and_drift_vanishes_beyond_the_run():
    """INV = OPENING + ADVERSE_h + DRIFT_h for h = 1 s, 60 s, 10 min and h >= run length (then DRIFT_h = 0
    exactly because t_f + h clips to T); the sum is h-invariant, the split is not."""
    r = run(SimConfig(seconds=300.0, seed=4), _price_quoter(T=300.0), BandHedger(10.0), _fair(), FLOW)
    parts = {}
    for h in (1.0, 60.0, 600.0, 1e6):
        a = pnl.attribute(r, h=h)
        assert abs(a.split_gap_fills) < 1e-9
        assert a.inventory == pytest.approx(r.attribution.inventory, abs=1e-12)
        parts[h] = (a.adverse_h, a.drift_h)
        assert a.h == h
    assert parts[600.0][1] == 0.0 and parts[1e6][1] == 0.0
    assert parts[1.0][0] != parts[60.0][0]
    assert r.attribution.h == 60.0
    with pytest.raises(ValueError):
        pnl.attribute(r, h=0.0)


def test_theta_split_captures_the_decay_when_nothing_else_moves():
    """spot_vol = alpha = 0: dF = theta dt + O(dt^2), so INV_exTHETA is a second-order remainder of THETA."""
    fair = _fair(spot_vol=0.0, alpha=0.0)
    q0 = (1.0, 1.0, 2.0, 2.0, 5.0, 5.0, 2.0, 2.0, 1.0, 1.0)
    r = run(SimConfig(seconds=3600.0, seed=0, q0=q0), PulledQuoter(), NoHedger(), fair, FLOW)
    a = r.attribution
    assert a.theta < 0  # long options decay
    assert abs(a.inventory_ex_theta) < 1e-4 * abs(a.theta)
    assert abs(a.split_gap_theta) < 1e-9
    assert a.greek_theta == a.theta
    assert abs(a.resid) < 1e-4 * abs(a.theta)  # no spot / vol move: the only residual is theta's own convexity


def test_adverse_term_is_the_fill_level_markout_summed():
    """ADVERSE_h == sum over fills of sign * qty * (F_{t_f + h} - F_{t_f}) recomputed from the record."""
    r = run(SimConfig(seconds=300.0, seed=6), _price_quoter(T=300.0), NoHedger(), _fair(), FLOW)
    f = r.fills
    n = r.path.n_steps
    ahead = np.minimum(f["step"].to_numpy() + 60, n)
    j = f["instrument"].to_numpy()
    expect = fsum(f["sign"].to_numpy() * f["qty"].to_numpy() * (r.path.prices[ahead, j] - r.path.prices[f["step"].to_numpy(), j]))
    assert r.attribution.adverse_h == pytest.approx(expect, abs=1e-12)
    # informed flow at p = 1 is adversely selected on average: the term is negative on a long run
    assert r.attribution.adverse_h < 0


# ----------------------------------------------------------------------------------------------------------
# Greek layer
# ----------------------------------------------------------------------------------------------------------


def test_greek_layer_residual_is_small_at_one_second_steps():
    """Research oracle: RESID 6.2e-7 on inventory O(1) for one ATM strike with volvol 0.16 (relative 1e-6);
    here one ATM strike, alpha 0.16, no jumps: |RESID| < 1e-5 * gross inventory move sum |q dF|. On the
    five-strike strip with alpha 0.6 the wings' volga dominates and the bound is 1e-3."""
    fair1 = _fair(strikes=(100.0,), alpha=0.16, skew_s=0.0, curv_c=0.0)
    r1 = run(SimConfig(seconds=1800.0, seed=7), _price_quoter(T=1800.0), BandHedger(10.0), fair1, FLOW)
    gross1 = float(np.sum(np.abs(r1.positions[1:] * np.diff(r1.path.prices, axis=0))))
    assert gross1 > 0 and abs(r1.attribution.resid) < 1e-5 * gross1
    r5 = run(SimConfig(seconds=600.0, seed=7), _price_quoter(), BandHedger(10.0), _fair(), FLOW)
    gross5 = float(np.sum(np.abs(r5.positions[1:] * np.diff(r5.path.prices, axis=0))))
    assert abs(r5.attribution.resid) < 1e-3 * gross5
    a = r5.attribution
    assert a.inventory == pytest.approx(fsum([a.greek_delta, a.greek_gamma, a.greek_vega, a.greek_theta, a.resid]), abs=1e-12)
    assert a.greek_gamma != 0.0 and a.greek_vega != 0.0


def test_greek_terms_are_the_documented_sums():
    r = run(SimConfig(seconds=200.0, seed=8), _price_quoter(T=200.0), NoHedger(), _fair(), FLOW)
    p = r.path
    held = r.positions[1:]
    dS = np.diff(p.spot)
    dt_y = 1.0 / YEAR_SECONDS
    assert r.attribution.greek_delta == pytest.approx(fsum((held * p.deltas[:-1] * dS[:, None]).ravel()), abs=1e-12)
    assert r.attribution.greek_gamma == pytest.approx(fsum((held * 0.5 * p.gammas[:-1] * dS[:, None] ** 2).ravel()), abs=1e-12)
    assert r.attribution.greek_vega == pytest.approx(fsum((held * p.vegas[:-1] * np.diff(p.vols, axis=0)).ravel()), abs=1e-12)
    assert r.attribution.theta == pytest.approx(fsum((held * p.thetas[:-1] * dt_y).ravel()), abs=1e-15)


# ----------------------------------------------------------------------------------------------------------
# guard and presentation
# ----------------------------------------------------------------------------------------------------------


def test_check_identity_raises_with_the_numbers():
    r = run(SimConfig(seconds=60.0, seed=9), _price_quoter(T=60.0), NoHedger(), _fair(), FLOW)
    a = r.attribution
    pnl.check_identity(a)
    broken = replace(a, gap=2e-9)
    with pytest.raises(AssertionError, match="gap 2.000e-09"):
        pnl.check_identity(broken)
    with pytest.raises(AssertionError, match="fill split"):
        pnl.check_identity(replace(a, drift_h=a.drift_h + 1e-6))
    with pytest.raises(AssertionError, match="theta split"):
        pnl.check_identity(replace(a, theta=a.theta + 1e-6))


def test_waterfall_and_dict_expose_every_term():
    r = run(SimConfig(seconds=60.0, seed=9), _price_quoter(T=60.0), NoHedger(), _fair(), FLOW)
    text = r.attribution.waterfall()
    for token in ("spread", "= realised", "gap", "adverse (60 s)", "drift (60 s)", "theta", "RESID"):
        assert token in text
    d = r.attribution.as_dict()
    assert set(pnl.TERMS) <= set(d) and d["h"] == 60.0
