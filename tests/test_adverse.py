"""adverse.py: markout = realised spread + adverse (exact), the tie to pnl's ADVERSE_h, class tagging,
the informed sign, clustered SEs, vol points, toxicity. Oracles: docs/PLAN.md 'adverse.py' contract and
research report_a3e5dd section 5 (MO_h = s_f (F_{t+h} - px_f); uninformed flow -> E[adverse] = 0;
p = 1 informed -> negative; cluster SE by time bucket because markouts overlap)."""

import numpy as np
import pytest

from quotesim import adverse
from quotesim.adverse import clustered_se, markouts, toxicity
from quotesim.fair import SyntheticFair
from quotesim.flow import FlowParams
from quotesim.hedger import NoHedger
from quotesim.quoter import PriceSpace, VolSpace, per_sqrt_second
from quotesim.sim import SimConfig, run


def _fair(**kw):
    base = dict(S0=100.0, sigma0=0.2, skew_s=-0.1, curv_c=0.3, alpha=0.6, spot_vol=0.2,
                strikes=(90.0, 95.0, 100.0, 105.0, 110.0), expiries=(30 / 365,))
    return SyntheticFair(**{**base, **kw})


def _quoter(T=600.0):
    return PriceSpace(kind="glft_asym", gamma=10.0, k=20.0, A=0.5, T=T, alpha_s=per_sqrt_second(0.6),
                      sigma_s=per_sqrt_second(0.2), tau_s=600.0)


def _run(informed=0.3, seconds=600.0, seed=3, **fair_kw):
    params = FlowParams(A=0.5, k=20.0, informed_frac=informed, p_informed=1.0, h_info=60.0)
    return run(SimConfig(seconds=seconds, seed=seed, h_markout=60.0), _quoter(seconds), NoHedger(), _fair(**fair_kw), params)


def test_markout_is_realised_plus_adverse_and_counts_add_up():
    r = _run()
    mk = markouts(r)
    assert list(mk.columns) == list(adverse.MARKOUT_COLUMNS)
    assert set(mk["horizon_s"]) == {1.0, 10.0, 60.0, 300.0}
    for _, row in mk.iterrows():
        assert row["markout_mean"] == pytest.approx(row["realised_spread_mean"] + row["adverse_mean"], abs=1e-12)
        assert row["markout_vp"] == pytest.approx(row["realised_spread_vp"] + row["adverse_vp"], abs=1e-9)
    for h in (1.0, 10.0, 60.0, 300.0):
        sub = mk[mk["horizon_s"] == h].set_index("class")
        assert sub.loc["all", "n"] == sub.loc["informed", "n"] + sub.loc["uninformed", "n"] == r.n_fills
        assert sub["n_vp"].le(sub["n"]).all()
    # the realised spread does not depend on the horizon
    rs = mk.groupby("class")["realised_spread_mean"].nunique()
    assert (rs == 1).all()


def test_adverse_mean_times_n_ties_to_the_identity_term():
    """Plan: ADVERSE_h in the identity IS the fill-level markout summed at the canonical h."""
    r = _run()
    mk = markouts(r, horizons=(60.0,)).set_index("class")
    assert mk.loc["all", "adverse_mean"] * mk.loc["all", "n"] == pytest.approx(r.attribution.adverse_h, rel=1e-12)
    assert mk.loc["all", "realised_spread_mean"] * mk.loc["all", "n"] == pytest.approx(r.attribution.spread, rel=1e-12)


def test_informed_class_is_adversely_selected_and_uninformed_is_not():
    """p = 1 informed flow: the informed adverse mean is negative by many clustered SEs at 60 s (its horizon)
    and grows from 1 s to 60 s; the uninformed one is within 4 clustered SEs of zero."""
    r = _run(informed=0.3, seconds=1800.0)
    mk = markouts(r, horizons=(1.0, 60.0)).set_index(["horizon_s", "class"])
    inf60 = mk.loc[(60.0, "informed")]
    assert inf60["adverse_mean"] < 0 and inf60["adverse_mean"] < -6 * inf60["se_clustered"]
    assert inf60["adverse_mean"] < mk.loc[(1.0, "informed"), "adverse_mean"]
    un60 = mk.loc[(60.0, "uninformed")]
    assert abs(un60["adverse_mean"]) < 4 * un60["se_clustered"]
    assert inf60["adverse_vp"] < 0 and un60["realised_spread_vp"] > 0


def test_uninformed_only_flow_has_no_informed_rows():
    r = _run(informed=0.0, seconds=300.0)
    mk = markouts(r, horizons=(10.0,)).set_index("class")
    assert mk.loc["informed", "n"] == 0 and np.isnan(mk.loc["informed", "adverse_mean"])
    assert mk.loc["uninformed", "n"] == r.n_fills
    assert (r.fills["class"] == "uninformed").all()


def test_clustered_se_reduces_to_the_iid_se_with_one_fill_per_cluster_and_grows_with_overlap():
    gen = np.random.default_rng(0)
    x = gen.standard_normal(400)
    w = np.ones(400)
    iid = clustered_se(x, w, np.arange(400))
    assert iid == pytest.approx(x.std(ddof=1) / np.sqrt(400), rel=1e-12)
    # every value duplicated inside its cluster: the clustered SE is the iid SE of the 200 distinct values,
    # the naive SE of the 400 is smaller by ~sqrt(2)
    xx = np.repeat(x[:200], 2)
    dup = clustered_se(xx, np.ones(400), np.repeat(np.arange(200), 2))
    assert dup == pytest.approx(x[:200].std(ddof=1) / np.sqrt(200), rel=1e-12)
    assert dup > xx.std(ddof=1) / np.sqrt(400) * 1.3
    assert np.isnan(clustered_se(x[:5], w[:5], np.zeros(5)))  # one cluster
    # weights: doubling a fill's weight equals repeating it
    assert clustered_se(x[:10], np.r_[2.0, np.ones(9)], np.arange(10)) == pytest.approx(
        clustered_se(np.r_[x[0], x[:10]], np.ones(11), np.r_[0, np.arange(10)]), rel=1e-12)


def test_vol_points_are_dollars_over_vega_and_a_vol_half_spread_reads_as_one_vol_point():
    """A VolSpace quoter with hs_vol = 0.01 and no skew captures ~1.00 vol point of realised spread on every
    fill (Black is slightly convex in vol, so within 3 %)."""
    fair = _fair()
    params = FlowParams(A=0.5, k=20.0, informed_frac=0.0)
    vs = VolSpace(gamma=1e-6, alpha_s=per_sqrt_second(0.6), sigma_s=per_sqrt_second(0.2), tau_s=600.0, hs_vol=0.01)
    r = run(SimConfig(seconds=300.0, seed=11), vs, NoHedger(), fair, params)
    mk = markouts(r, horizons=(1.0,)).set_index("class")
    assert mk.loc["all", "realised_spread_vp"] == pytest.approx(1.0, rel=0.03)
    # per-fill definition: $ / vega * 100 with the fill's vega
    f = r.fills
    vega = r.path.vegas[f["step"].to_numpy(), f["instrument"].to_numpy()]
    rs = f["sign"].to_numpy() * (f["fair_at_fill"].to_numpy() - f["px"].to_numpy())
    w = f["qty"].to_numpy()
    assert mk.loc["all", "realised_spread_vp"] == pytest.approx(np.sum(w * rs / vega * 100) / w.sum(), rel=1e-12)


def test_toxicity_summary_shape_and_signs():
    r = _run(informed=0.3, seconds=1800.0)
    t = toxicity(r)
    assert list(t.index) == ["all", "uninformed", "informed"]
    assert t.loc[["uninformed", "informed"], "share"].sum() == pytest.approx(1.0)
    assert t.loc["all", "n"] == r.n_fills
    assert t.loc["informed", "give_back"] > t.loc["uninformed", "give_back"]
    assert t.loc["informed", "give_back"] > 0.1
    assert (t["frac_negative"].between(0, 1)).all()
    assert t.loc["all", "markout_mean"] == pytest.approx(t.loc["all", "realised_spread_mean"] + t.loc["all", "adverse_mean"])
    t2 = toxicity(r, h=1.0)
    assert abs(t2.loc["informed", "adverse_mean"]) < abs(t.loc["informed", "adverse_mean"])


def test_bad_inputs_raise():
    r = _run(seconds=60.0)
    with pytest.raises(ValueError):
        markouts(r, horizons=(0.0,))

    class Pulled:
        def quotes(self, snap, inventory, t):
            return np.full(snap.n_instruments, np.nan), np.full(snap.n_instruments, np.nan)

    empty = run(SimConfig(seconds=30.0, seed=0), Pulled(), NoHedger(), _fair(), FlowParams(A=0.5, k=20.0))
    with pytest.raises(ValueError):
        markouts(empty)
