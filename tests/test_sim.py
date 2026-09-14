"""sim.py: the loop's bookkeeping, determinism and common random numbers, the in-run identity guard, the
past-only fair, and the paired harness. Oracles: docs/PLAN.md 'sim.py' (same seed + identical quoters ->
bit-identical fills and R) and research report_a3e5dd section 8 (CRN: strategies differ only via thinning;
paired report = median, IQR, p5/p95, sign test, bootstrap CI; never the best seed)."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quotesim import pnl, sim
from quotesim.fair import SyntheticFair
from quotesim.flow import FlowParams
from quotesim.hedger import BandHedger, NoHedger, TimeHedger
from quotesim.quoter import PriceSpace, per_sqrt_second
from quotesim.sim import PairedResult, SimConfig, paired, paired_stats, run

FLOW = FlowParams(A=0.5, k=20.0, informed_frac=0.2, p_informed=0.8, h_info=30.0, persist=0.3)


def _fair(**kw):
    base = dict(S0=100.0, sigma0=0.2, skew_s=-0.1, curv_c=0.3, alpha=0.6, spot_vol=0.2,
                strikes=(95.0, 100.0, 105.0), expiries=(30 / 365,))
    return SyntheticFair(**{**base, **kw})


def _quoter(gamma=10.0, k=20.0, T=300.0):
    return PriceSpace(kind="glft_asym", gamma=gamma, k=k, A=0.5, T=T, alpha_s=per_sqrt_second(0.6),
                      sigma_s=per_sqrt_second(0.2), tau_s=600.0)


class Widened:
    """The same quoter with every half-spread widened by `extra` $ (fewer fills under the same uniforms)."""

    def __init__(self, base, extra):
        self.base, self.extra = base, extra

    def quotes(self, snap, inventory, t):
        b, a = self.base.quotes(snap, inventory, t)
        return b - self.extra, a + self.extra


def _fill_counts(r) -> pd.Series:
    return r.fills.groupby(["step", "instrument", "side"])["qty"].sum()


# ----------------------------------------------------------------------------------------------------------
# bookkeeping
# ----------------------------------------------------------------------------------------------------------


def test_positions_and_cash_follow_the_fills_and_hedges():
    r = run(SimConfig(seconds=300.0, seed=1), _quoter(), BandHedger(5.0), _fair(), FLOW)
    n, m = r.path.n_steps, r.path.n_instruments
    assert r.positions.shape == (n + 1, m) and r.hedge_pos.shape == (n + 1,) == r.cash.shape
    assert r.bids.shape == (n, m) == r.asks.shape
    signed = np.zeros((n, m))
    for _, f in r.fills.iterrows():
        signed[int(f["step"]), int(f["instrument"])] += f["sign"] * f["qty"]
    assert np.array_equal(np.diff(r.positions, axis=0), signed)
    assert np.array_equal(r.positions[0], np.zeros(m)) and r.hedge_pos[0] == 0.0 and r.cash[0] == 0.0
    flows = -(r.fills["sign"] * r.fills["qty"] * r.fills["px"]).sum() - (r.hedges["shares"] * r.hedges["S"] + r.hedges["cost"]).sum()
    assert r.cash_T == pytest.approx(flows, abs=1e-8) and r.cash[-1] == pytest.approx(r.cash_T, abs=1e-8)
    # fill prices are the quotes of that step and side; fair_at_fill is the path
    for _, f in r.fills.head(50).iterrows():
        i, j, s = int(f["step"]), int(f["instrument"]), int(f["side"])
        assert f["px"] == (r.bids if s == 0 else r.asks)[i, j] and f["fair_at_fill"] == r.path.prices[i, j]
        assert f["sign"] == (1.0 if s == 0 else -1.0) and f["t"] == i * r.config.dt
    assert set(r.fills["class"]) <= {"informed", "uninformed"} and (r.fills["class"] == "informed").any()
    # hedge lands on target at every step with band 0
    r0 = run(SimConfig(seconds=120.0, seed=1), _quoter(), BandHedger(0.0), _fair(), FLOW)
    target = -(r0.positions[1:] * r0.path.deltas[:-1]).sum(axis=1)
    assert np.allclose(r0.hedge_pos[1:], target, atol=1e-12)
    assert r0.hedges["h_after"].to_numpy()[-1] == r0.hedge_pos[-1]


def test_time_hedger_trades_on_its_grid_and_no_hedger_never():
    """The grid is anchored at t = 0 whether or not the mismatch was zero there: with thin flow (no fill at
    t = 0 for seeds 0, 1, 5) the hedges still land on {60, 120, ...}, never on first-fill + 60 k."""
    r = run(SimConfig(seconds=300.0, seed=2), _quoter(), TimeHedger(60.0), _fair(), FLOW)
    assert set(r.hedges["t"]) <= {0.0, 60.0, 120.0, 180.0, 240.0} and r.n_hedges >= 4
    thin = FlowParams(A=0.02, k=20.0)
    for seed in (0, 1, 5):
        rt = run(SimConfig(seconds=600.0, seed=seed), _quoter(T=600.0), TimeHedger(60.0), _fair(), thin)
        first_fill = float(rt.fills["t"].min())
        assert first_fill > 0.0 and rt.n_hedges >= 2
        assert set(rt.hedges["t"]) <= set(np.arange(0.0, 600.0, 60.0)), (seed, list(rt.hedges["t"]))
        # every grid point after the first fill with a nonzero mismatch traded
        due = [t for t in np.arange(0.0, 600.0, 60.0) if t >= first_fill]
        assert due[0] in set(rt.hedges["t"]) or rt.hedge_pos[int(due[0])] == -float(np.sum(rt.positions[int(due[0]) + 1] * rt.path.deltas[int(due[0])]))
    r2 = run(SimConfig(seconds=300.0, seed=2), _quoter(), NoHedger(), _fair(), FLOW)
    assert r2.n_hedges == 0 and r2.attribution.hedge == 0.0 and r2.attribution.hedge_cost == 0.0
    assert np.all(r2.hedge_pos == 0.0)


def test_summary_and_run_properties():
    r = run(SimConfig(seconds=120.0, seed=3), _quoter(), BandHedger(5.0), _fair(), FLOW)
    s = r.summary()
    assert s["n_fills"] == r.n_fills == int(r.fills["qty"].sum())
    assert s["n_inside_fair"] == r.n_inside_fair == 0  # gamma 10: no per-side offset below zero on this run
    assert s["abs_q_T"] == float(np.abs(r.terminal_inventory).sum())
    assert s["realised"] == r.attribution.realised and s["gap"] == r.attribution.gap
    assert s["n_candidates"] == r.n_candidates > r.n_fills


# ----------------------------------------------------------------------------------------------------------
# determinism and CRN
# ----------------------------------------------------------------------------------------------------------


def test_same_seed_and_identical_quoters_give_bit_identical_fills_and_realised():
    """Plan oracle: same seed + identical quoters -> identical fill sequences and R to the bit."""
    cfg = SimConfig(seconds=300.0, seed=7)
    a = run(cfg, _quoter(), BandHedger(10.0), _fair(), FLOW)
    b = run(cfg, replace(_quoter()), BandHedger(10.0), _fair(), FLOW)
    pd.testing.assert_frame_equal(a.fills, b.fills)
    pd.testing.assert_frame_equal(a.hedges, b.hedges)
    assert a.attribution.realised == b.attribution.realised
    assert a.attribution == b.attribution
    assert np.array_equal(a.path.prices, b.path.prices) and np.array_equal(a.positions, b.positions)
    c = run(replace(cfg, seed=8), _quoter(), BandHedger(10.0), _fair(), FLOW)
    assert not np.array_equal(a.path.spot, c.path.spot) and a.attribution.realised != c.attribution.realised


def test_different_quoters_share_the_candidate_stream_and_the_path_and_the_wide_fills_are_a_subset():
    """CRN: the same seed gives the same path and candidates; a wider quoter's fills are a subset of the
    tighter one's cell by cell (thinning is monotone in the same uniform)."""
    cfg = SimConfig(seconds=300.0, seed=5)
    tight = run(cfg, _quoter(), BandHedger(10.0), _fair(), FLOW)
    wide = run(cfg, Widened(_quoter(), 0.02), BandHedger(10.0), _fair(), FLOW)
    assert tight.n_candidates == wide.n_candidates
    assert np.array_equal(tight.path.prices, wide.path.prices)
    ct, cw = _fill_counts(tight), _fill_counts(wide)
    assert wide.n_fills < tight.n_fills
    joined = pd.concat([ct.rename("t"), cw.rename("w")], axis=1).fillna(0)
    assert (joined["w"] <= joined["t"]).all()
    assert tight.attribution.spread > 0 and wide.attribution.spread > 0


def test_identity_is_asserted_inside_run(monkeypatch):
    """The guard lives in sim.run: a broken attribution stops the run with the numbers in the message."""
    real = pnl.attribute

    def broken(r, h=60.0):
        return replace(real(r, h), gap=5e-9)

    monkeypatch.setattr(pnl, "attribute", broken)
    with pytest.raises(AssertionError, match="gap 5.000e-09"):
        run(SimConfig(seconds=30.0, seed=0), _quoter(), NoHedger(), _fair(), FLOW)


def test_quoter_sees_the_state_at_the_clock_and_nothing_ahead():
    seen = []

    class Spy:
        def __init__(self, base):
            self.base = base

        def quotes(self, snap, inventory, t):
            seen.append((t, snap.spot, snap.t))
            return self.base.quotes(snap, inventory, t)

    r = run(SimConfig(seconds=50.0, seed=0), Spy(_quoter()), NoHedger(), _fair(), FLOW)
    assert len(seen) == 50
    for i, (t, spot, t_y) in enumerate(seen):
        assert t == float(i) and spot == r.path.spot[i] and t_y == r.path.t[i]


def test_config_validation_and_q0_shape():
    with pytest.raises(ValueError):
        SimConfig(seconds=0.0)
    with pytest.raises(ValueError):
        SimConfig(seconds=0.4, dt=1.0)
    with pytest.raises(ValueError):
        SimConfig(h_markout=0.0)
    assert SimConfig(seconds=10.0, dt=0.5).n_steps == 20
    with pytest.raises(ValueError):
        run(SimConfig(seconds=10.0, q0=(1.0, 2.0)), _quoter(), NoHedger(), _fair(), FLOW)


def test_fair_is_left_untouched_by_run():
    f = _fair()
    run(SimConfig(seconds=30.0, seed=0), _quoter(), NoHedger(), f, FLOW)
    assert f.t == 0.0 and f.S == 100.0 and f.sigma_atm == 0.2


# ----------------------------------------------------------------------------------------------------------
# paired harness
# ----------------------------------------------------------------------------------------------------------


def test_paired_stats_on_a_known_sample():
    """Research shape oracle: median, IQR, p5/p95, sign count, bootstrap CI (toy run in report_a3e5dd
    gave median +0.4382, IQR [0.1512, 0.8033], p5 -0.7156, sign 16/20, CI [0.0927, 0.5911]; here the
    same statistics on a known array)."""
    d = np.array([1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 5.0, 6.0])
    st = paired_stats(d, n_boot=500, boot_seed=1)
    assert st["n"] == 8 and st["median"] == 2.5 and st["mean"] == 2.5
    assert st["q25"] == np.percentile(d, 25) and st["q75"] == np.percentile(d, 75)
    assert st["n_pos"] == 6 and st["n_nonzero"] == 7
    assert st["p_sign"] == pytest.approx(2 * 8 / 128)  # two-sided binomial, 6 of 7
    assert st["boot_lo"] < st["mean"] < st["boot_hi"]
    assert paired_stats(d, 500, 1) == st  # seeded
    assert paired_stats(np.zeros(5))["p_sign"] == 1.0 and paired_stats(np.zeros(5))["n_nonzero"] == 0
    with pytest.raises(ValueError):
        paired_stats(np.array([]))


def test_paired_same_quoter_twice_has_zero_differences_and_table_shape():
    cfg = SimConfig(seconds=120.0)
    q = _quoter(T=120.0)
    pr = paired(cfg, {"a": q, "b": replace(q)}, seeds=[0, 1, 2], hedger=BandHedger(10.0), fair=_fair(), params=FLOW)
    assert isinstance(pr, PairedResult) and pr.names == ("a", "b") and pr.seeds == (0, 1, 2)
    assert pr.values.index.names == ["seed", "quoter"] and len(pr.values) == 6
    d = pr.diffs("a", "b")
    assert (d == 0.0).all().all()
    t = pr.table()
    assert list(t.index.get_level_values("comparison").unique()) == ["b - a"]
    assert set(t.index.get_level_values("term")) == set(sim.PAIRED_TERMS)
    assert (t["n_pos"] == 0).all() and (t["p_sign"] == 1.0).all() and (t["median"] == 0.0).all()
    lv = pr.levels()
    assert lv.loc["a"].equals(lv.loc["b"])


def test_paired_wide_vs_tight_reports_the_spread_difference_with_its_sign():
    cfg = SimConfig(seconds=200.0)
    tight = _quoter(T=200.0)
    pr = paired(cfg, {"tight": tight, "wide": Widened(tight, 0.03)}, seeds=range(6), hedger=BandHedger(10.0), fair=_fair(), params=FLOW)
    t = pr.table(baseline="tight")
    assert t.loc[("wide - tight", "n_fills"), "n_pos"] == 0 and t.loc[("wide - tight", "n_fills"), "n_nonzero"] == 6
    d = pr.diffs("tight", "wide")
    assert (d["n_fills"] < 0).all()
    st = t.loc[("wide - tight", "realised")]
    assert st["boot_lo"] <= st["mean"] <= st["boot_hi"] and st["n"] == 6
    with pytest.raises(ValueError):
        paired(cfg, {"only": tight}, seeds=[0], hedger=NoHedger(), fair=_fair(), params=FLOW)


def test_paired_per_quoter_hedgers():
    cfg = SimConfig(seconds=100.0)
    q = _quoter(T=100.0)
    pr = paired(cfg, {"band": q, "none": replace(q)}, seeds=[0, 1], hedger=BandHedger(5.0),
                fair=_fair(), params=FLOW, hedgers={"none": NoHedger()})
    v = pr.values
    assert (v.xs("none", level="quoter")["n_hedges"] == 0).all() and (v.xs("band", level="quoter")["n_hedges"] > 0).all()
    assert (v.xs("none", level="quoter")["n_fills"] == v.xs("band", level="quoter")["n_fills"]).all()
