"""flow.py: thinning law, CRN, informed-markout oracle (research report_a3e5dd section 5/6 and the
plan's Flow oracles), persistence, sign conventions."""

import numpy as np
import pytest

from quotesim import flow
from quotesim.flow import FlowParams, StepFills, Streams, accept_prob, arrivals, thin

SIGMA_F, H = 0.05, 60  # $/sqrt(s) fair vol and markout horizon (s) of the analytic oracle


def _bm_fair(seed, n_steps, n_inst):
    """Independent Brownian fairs, sigma_F $/sqrt(s), dt = 1 s: (n_steps + 1, n_inst)."""
    gen = np.random.default_rng(seed)
    return 100.0 + np.cumsum(np.vstack([np.zeros((1, n_inst)), SIGMA_F * gen.standard_normal((n_steps, n_inst))]), axis=0)


def _run_markouts(params, seed, n_steps, n_inst, hs=0.0):
    """Loop the arrivals with quotes at fair -/+ hs; return the fill-level adverse markouts
    s_f (F_{t+H} - F_t) for every fill, split by informed flag."""
    fair = _bm_fair(seed, n_steps, n_inst)
    streams = Streams.draw(seed, n_steps, n_inst, A=1.0, dt=1.0)
    adverse, informed = [], []
    prev = None
    for i in range(n_steps):
        j = min(i + H, n_steps)
        f_now, f_ahead = fair[i], fair[j]
        prev = arrivals(params, streams, i, f_now - hs, f_now + hs, f_now, f_ahead, prev)
        move = f_ahead - f_now
        for side in range(2):
            n_f = prev.fills[:, side]
            n_i = prev.informed[:, side]
            mo = flow.MM_SIGN[side] * move
            adverse.extend(np.repeat(mo, n_i))
            informed.extend(np.repeat(True, n_i.sum()))
            adverse.extend(np.repeat(mo, n_f - n_i))
            informed.extend(np.repeat(False, (n_f - n_i).sum()))
    return np.array(adverse), np.array(informed)


# ----------------------------------------------------------------------------------------------------------
# streams and thinning
# ----------------------------------------------------------------------------------------------------------


def test_streams_shapes_and_same_seed_gives_identical_draws():
    """CRN: the same seed reproduces every stream bit for bit; a different seed does not."""
    a = Streams.draw(7, 500, 4, A=0.9, dt=1.0)
    b = Streams.draw(7, 500, 4, A=0.9, dt=1.0)
    c = Streams.draw(8, 500, 4, A=0.9, dt=1.0)
    assert a.spot_z.shape == (500,) and a.candidates.shape == (500, 4, 2) == a.accept_u.shape
    for name in ("spot_z", "vol_z", "jump_u", "candidates", "accept_u", "informed_u", "direction_u"):
        assert np.array_equal(getattr(a, name), getattr(b, name))
    assert not np.array_equal(a.candidates, c.candidates)
    assert a.n_steps == 500 and a.n_instruments == 4
    assert a.candidates.dtype == np.int64 and a.candidates.min() >= 0


def test_candidate_counts_are_poisson_at_rate_A_dt():
    s = Streams.draw(3, 20000, 5, A=0.9, dt=1.0)
    mean = s.candidates.mean()
    se = np.sqrt(0.9 / s.candidates.size)
    assert abs(mean - 0.9) < 4 * se


def test_accept_prob_caps_inside_fair_and_pulls_nan():
    k = 0.5
    assert accept_prob(2.0, k) == pytest.approx(np.exp(-1.0))
    assert accept_prob(-3.0, k) == 1.0  # inside fair: capped at delta = 0
    assert accept_prob(np.nan, k) == 0.0
    assert accept_prob(np.inf, k) == 0.0


def test_thin_acceptance_rate_equals_exp_minus_k_delta_within_binomial_ci():
    """Plan oracle: acceptance rate = exp(-k delta) within a binomial CI; and the thinned count is
    Poisson(A e^{-k delta} dt), so P(>= 1 fill) = 1 - exp(-lambda dt) (never lambda dt)."""
    s = Streams.draw(11, 30000, 4, A=0.9, dt=1.0)
    k, delta = 0.5, 1.2
    fills = thin(s.candidates, s.accept_u, delta, k)
    p = np.exp(-k * delta)
    n_cand = s.candidates.sum()
    rate = fills.sum() / n_cand
    assert abs(rate - p) < 4 * np.sqrt(p * (1 - p) / n_cand)
    lam = 0.9 * p
    p_fill = 1 - np.exp(-lam)
    frac = (fills > 0).mean()
    assert abs(frac - p_fill) < 4 * np.sqrt(p_fill * (1 - p_fill) / fills.size)
    assert abs(frac - lam) > 4 * np.sqrt(p_fill * (1 - p_fill) / fills.size)  # lambda dt is the wrong law
    assert np.all(fills <= s.candidates)


def test_thin_single_candidate_is_accept_iff_u_below_p_and_monotone_in_delta():
    u = np.array([0.1, 0.5, 0.9])
    c = np.ones(3, dtype=int)
    k = 1.0
    assert np.array_equal(thin(c, u, 0.0, k), [1, 1, 1])
    assert np.array_equal(thin(c, u, np.log(2.0), k), [1, 0, 0])  # p = 0.5
    assert np.array_equal(thin(c, u, np.nan, k), [0, 0, 0])
    s = Streams.draw(5, 2000, 2, A=1.5, dt=1.0)
    tight = thin(s.candidates, s.accept_u, 0.2, k)
    wide = thin(s.candidates, s.accept_u, 0.8, k)
    assert np.all(tight >= wide)  # same u: a tighter quote never loses a fill the wider one got


def test_thin_multi_candidate_counts_have_the_binomial_law():
    c = np.full(200000, 3)
    u = np.random.default_rng(2).random(c.size)
    p = 0.4
    n = thin(c, u, -np.log(p), 1.0)
    from math import comb

    for j in range(4):
        pj = comb(3, j) * p**j * (1 - p) ** (3 - j)
        frac = (n == j).mean()
        assert abs(frac - pj) < 4 * np.sqrt(pj * (1 - pj) / c.size)


def test_thin_rejects_bad_inputs():
    with pytest.raises(ValueError):
        thin([-1], [0.5], 0.0, 1.0)
    with pytest.raises(ValueError):
        thin([1], [1.0], 0.0, 1.0)


# ----------------------------------------------------------------------------------------------------------
# arrivals
# ----------------------------------------------------------------------------------------------------------


def test_arrivals_sign_convention_and_pulled_quotes():
    """side 0 = MM buys at the bid; side 1 = MM sells at the ask; NaN quote -> no fills on that side."""
    params = FlowParams(A=2.0, k=0.0)  # k = 0: every candidate fills
    s = Streams.draw(1, 10, 3, A=2.0, dt=1.0)
    f = np.array([5.0, 6.0, 7.0])
    step = 0
    fills = arrivals(params, s, step, f - 0.1, f + 0.1, f, f)
    assert np.array_equal(fills.fills.sum(axis=1), s.candidates[step].sum(axis=1))
    assert fills.n_fills == s.candidates[step].sum()
    assert np.array_equal(fills.signed_lots(), fills.fills[:, 0] - fills.fills[:, 1])
    pulled = arrivals(params, s, step, np.full(3, np.nan), f + 0.1, f, f)
    assert pulled.fills[:, 0].sum() == 0 and np.array_equal(pulled.fills[:, 1], fills.fills[:, 1])
    assert isinstance(fills, StepFills) and fills.informed.sum() == 0


def test_uninformed_flow_keeps_nominal_sides_and_two_quoters_share_candidates():
    """CRN across strategies: the same streams thinned by two different quotes give fill sets where the
    tighter quote's fills are a superset per cell."""
    params = FlowParams(A=1.0, k=1.0)
    s = Streams.draw(4, 300, 3, A=1.0, dt=1.0)
    f = np.array([10.0, 11.0, 12.0])
    for i in range(300):
        tight = arrivals(params, s, i, f - 0.1, f + 0.1, f, f)
        wide = arrivals(params, s, i, f - 0.5, f + 0.5, f, f)
        assert np.all(tight.fills >= wide.fills)
        assert np.all(tight.fills <= s.candidates[i])


def test_informed_markout_oracle_p1_and_p06():
    """Research report_a3e5dd section 5: E[s_f (F_{t+h} - F_t)] = -(2p - 1) sigma_F sqrt(h) sqrt(2/pi) for
    p-informed flow on a Brownian fair; sigma_F = 0.05 $/sqrt(s), h = 60 s: E|dF| = 0.3090194;
    p = 1 -> -0.3090194; p = 0.6 -> -0.0618039; p = 0.5 -> 0. Overlapping horizons correlate consecutive
    markouts, so the SE uses the block count N / h."""
    expected_abs = SIGMA_F * np.sqrt(H) * np.sqrt(2 / np.pi)
    assert expected_abs == pytest.approx(0.3090193616, abs=1e-9)
    n_steps, n_inst = 6000, 8
    for p, target in ((1.0, -0.3090194), (0.6, -0.0618039), (0.5, 0.0)):
        params = FlowParams(A=1.0, k=1.0, informed_frac=1.0, p_informed=p, h_info=float(H))
        adverse, informed = _run_markouts(params, seed=21, n_steps=n_steps, n_inst=n_inst)
        assert informed.all()
        n_blocks = adverse.size / H
        se = adverse.std() / np.sqrt(n_blocks)
        assert abs(adverse.mean() - target) < 4 * se, (p, adverse.mean(), se)
        assert abs(target - (-(2 * p - 1) * expected_abs)) < 1e-6


def test_uninformed_flow_has_zero_mean_adverse_within_se():
    params = FlowParams(A=1.0, k=1.0, informed_frac=0.0, h_info=float(H))
    adverse, informed = _run_markouts(params, seed=5, n_steps=6000, n_inst=8)
    assert not informed.any()
    se = adverse.std() / np.sqrt(adverse.size / H)
    assert abs(adverse.mean()) < 4 * se


def test_mixed_flow_tags_informed_fills_and_they_carry_the_adverse_mean():
    params = FlowParams(A=1.0, k=1.0, informed_frac=0.3, p_informed=1.0, h_info=float(H))
    adverse, informed = _run_markouts(params, seed=9, n_steps=6000, n_inst=8)
    share = informed.mean()
    assert abs(share - 0.3) < 4 * np.sqrt(0.3 * 0.7 / informed.size)
    se_i = adverse[informed].std() / np.sqrt(informed.sum() / H)
    assert adverse[informed].mean() < -0.3090194 + 4 * se_i
    se_u = adverse[~informed].std() / np.sqrt((~informed).sum() / H)
    assert abs(adverse[~informed].mean()) < 4 * se_u


def test_persistence_makes_uninformed_sides_autocorrelated():
    """persist = 0.9: the realised side of consecutive uninformed candidates on one instrument repeats
    ~90% of the time (vs ~50% at persist = 0)."""
    n = 4000
    s = Streams.draw(13, n, 1, A=1.0, dt=1.0)
    f = np.array([10.0])

    def sides(persist):
        params = FlowParams(A=1.0, k=0.0, persist=persist)
        out, prev = [], None
        for i in range(n):
            prev = arrivals(params, s, i, f - 0.1, f + 0.1, f, f)
            if prev.n_fills == 1:
                out.append(int(prev.fills[0, 1]))
        return np.array(out)

    for persist, lo, hi in ((0.0, 0.42, 0.58), (0.9, 0.85, 0.97)):
        x = sides(persist)
        repeat = (x[1:] == x[:-1]).mean()
        assert lo < repeat < hi, (persist, repeat)


def test_flow_params_validation_and_h_steps():
    assert FlowParams(A=1.0, k=1.0, h_info=60.0).h_steps(1.0) == 60
    assert FlowParams(A=1.0, k=1.0, h_info=0.4).h_steps(1.0) == 1
    for bad in (dict(A=0.0, k=1.0), dict(A=1.0, k=-1.0), dict(A=1.0, k=1.0, informed_frac=1.5),
                dict(A=1.0, k=1.0, p_informed=0.4), dict(A=1.0, k=1.0, persist=1.0), dict(A=1.0, k=1.0, h_info=0.0)):
        with pytest.raises(ValueError):
            FlowParams(**bad)
    with pytest.raises(ValueError):
        Streams.draw(0, 0, 1, 1.0, 1.0)
