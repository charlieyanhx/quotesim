"""quoter.py oracles: Avellaneda-Stoikov 2008 (eqs 29-31, paper example), GLFT 2013 (exact ODE via the
tridiagonal generator, Prop 3 asymptotics), Stoikov-Saglam 2009 Thm 4 / eq. (17), strip wrappers,
matching, tick rounding and guards. Numbers from docs/PLAN.md, computed independently in research
(scratchpad oracles.py) and re-derived here."""

import numpy as np
import pytest
from scipy.linalg import expm

from quotesim import quoter
from quotesim.fair import YEAR_SECONDS, FutureAccessError, PastOnlyFair, SyntheticFair, black_greeks, black_price
from quotesim.quoter import (
    AS2008,
    GLFT2013,
    GLFTAsymptotic,
    MaxLossGuard,
    PriceSpace,
    VolSpace,
    match_spreads,
    residual_variance_terms,
    round_to_tick,
    vega_cap,
    vol_space_skew,
)

GLFT_EX = dict(gamma=0.01, k=0.3, sigma=0.3, A=0.9)  # GLFT 2013 example: ticks, seconds, T = 600 s
AS_EX = dict(gamma=0.01, k=0.3, sigma=0.3)


# ----------------------------------------------------------------------------------------------------------
# A-S 2008
# ----------------------------------------------------------------------------------------------------------


def test_as_paper_example_spread_and_reservation_shift():
    """A-S 2008 example (s=100, T=1, sigma=2, gamma=0.1, k=1.5): spread 1.690770423 at t=0, 1.290770423 at
    t=T (eq. 31), reservation shift 0.4 per unit q at t=0 (eq. 29)."""
    q = AS2008(gamma=0.1, k=1.5, sigma=2.0, T=1.0)
    assert q.spread(0.0) == pytest.approx(1.690770423, abs=1e-9)
    assert q.spread(1.0) == pytest.approx(1.290770423, abs=1e-9)
    assert q.reservation(100.0, 1, 0.0) == pytest.approx(99.6, abs=1e-12)
    assert q.reservation(100.0, 0, 0.37) == 100.0  # q = 0 -> reservation = mid
    b, a = q.offsets(3, 0.0)
    assert b + a == pytest.approx(q.spread(0.0), abs=1e-12)
    assert (100.0 - b + 100.0 + a) / 2 == pytest.approx(q.reservation(100.0, 3, 0.0), abs=1e-12)


def test_as_glft_example_numbers_and_gamma_to_zero_limit():
    """GLFT example params: A-S spread 7.097964565 at t=0, 6.557964565 at t=T; r(100, q=10, 0) = 94.6;
    (2/gamma) ln(1+gamma/k) -> 2/k = 6.666666667 as gamma -> 0 (1e-2: 6.557965, 1e-4: 6.665556, 1e-6: 6.666656)."""
    q = AS2008(T=600.0, **AS_EX)
    assert q.spread(0.0) == pytest.approx(7.097964565, abs=1e-8)
    assert q.spread(600.0) == pytest.approx(6.557964565, abs=1e-8)
    assert q.reservation(100.0, 10, 0.0) == pytest.approx(94.6, abs=1e-12)
    for g, ref in ((1e-2, 6.557965), (1e-4, 6.665556), (1e-6, 6.666656)):
        assert AS2008(gamma=g, k=0.3, sigma=0.3, T=600.0).spread(600.0) == pytest.approx(ref, abs=1e-6)
    gaps = [2 / 0.3 - AS2008(gamma=g, k=0.3, sigma=0.3, T=600.0).spread(600.0) for g in (1e-3, 1e-4, 1e-5)]
    assert np.allclose(np.array(gaps[:-1]) / np.array(gaps[1:]), 10.0, rtol=1e-2)  # gap = gamma / k^2 + O(gamma^2)


def test_as_first_order_condition_reduces_to_log_term_for_exponential_intensity():
    """A-S eqs 18/19: delta = (1/gamma) ln(1 - gamma lambda/lambda') with lambda = A e^{-k delta} equals
    (1/gamma) ln(1 + gamma/k) for every delta (gap 0)."""
    gamma, k, A = 0.01, 0.3, 0.9
    for d in (0.0, 1.234, 7.0):
        lam = A * np.exp(-k * d)
        dlam = -k * lam
        foc = np.log(1 - gamma * lam / dlam) / gamma
        assert foc == pytest.approx(np.log(1 + gamma / k) / gamma, abs=1e-14)


# ----------------------------------------------------------------------------------------------------------
# GLFT 2013
# ----------------------------------------------------------------------------------------------------------


def test_glft_exact_matches_plan_oracles_at_T600_Q30():
    """GLFT 2013 exact (sigma=0.3, A=0.9, k=0.3, gamma=0.01, T=600, Q=30, t=0): delta_b(0) = 3.313045,
    delta_b(5) = 3.653479, delta_a(5) = 2.972522, delta_b(20) = 4.665542, delta_a(20) = 1.959200."""
    g = GLFT2013(T=600.0, Q=30, **GLFT_EX)
    assert g.offsets(0)[0] == pytest.approx(3.313045, abs=1e-6)
    assert g.offsets(0)[1] == pytest.approx(3.313045, abs=1e-6)
    b5, a5 = g.offsets(5)
    assert (b5, a5) == pytest.approx((3.653479, 2.972522), abs=1e-6)
    b20, a20 = g.offsets(20)
    assert (b20, a20) == pytest.approx((4.665542, 1.959200), abs=1e-6)
    assert g.offsets(0)[0] + g.offsets(0)[1] == pytest.approx(6.626091, abs=1e-6)


def test_glft_uniformization_equals_eigh_and_scipy_expm_of_the_generator():
    """v(t) by uniformization (default) equals both the symmetric eigendecomposition and scipy.linalg.expm
    of the same (shifted) generator to 1e-10 relative where those are resolvable (Q = 30: v_30 / v_0 ~ 1e-2);
    the generator has alpha q^2 on the diagonal and -eta off it."""
    g = GLFT2013(T=600.0, Q=30, **GLFT_EX)
    assert g.alpha == pytest.approx(0.5 * 0.3 * 0.01 * 0.09)
    assert g.eta == pytest.approx(0.9 * (1 + 0.01 / 0.3) ** (-(1 + 0.3 / 0.01)))
    assert g.M[0, 0] == pytest.approx(g.alpha * 900) and g.M[3, 4] == -g.eta and g.M[4, 3] == -g.eta
    for t in (0.0, 123.4, 599.0):
        v_u, v_e, v_x = g.v(t), g.v(t, method="eigh"), g.v(t, method="expm")
        assert np.max(np.abs(v_u / v_x - 1)) < 1e-10
        assert np.max(np.abs(v_e / v_x - 1)) < 1e-10
    with pytest.raises(ValueError):
        g.v(0.0, method="pade")
    shift = np.linalg.eigvalsh(g.M).min()
    raw = expm(-g.M * 600.0) @ np.ones(61)
    assert np.allclose(raw / raw[30], g.v(0.0) / g.v(0.0)[30], rtol=1e-9)
    assert np.isfinite(shift)


# 70-digit mpmath eigendecomposition of the same tridiagonal (scratchpad mp_ref.py, dps = 70; the 40-digit run
# already loses the tail beyond |q| ~ 90): bid offsets at Q = 100, T = 600 for q = -100, -60, 0, 40, 55, 65, 80, 95, 99.
GLFT_Q100_BIDS = {
    0.0: (-2.762203414267, -0.557357008213, 3.313045408625, 5.964227825507, 6.880626291862, 7.459403275992, 8.277408299719, 9.036182745936, 9.320167978865),
    300.0: (-2.760003358452, -0.555232980004, 3.313020732896, 5.962527015553, 6.878569754989, 7.457197664508, 8.275088265986, 9.033847024667, 9.317967923050),
    599.0: (2.393991169583, 3.225435421055, 3.279432255924, 3.315430145864, 3.328929354562, 3.337928827011, 3.351428035656, 3.364980525255, 4.163973395015),
}
GLFT_Q100_QS = (-100, -60, 0, 40, 55, 65, 80, 95, 99)


def test_glft_default_Q100_is_finite_monotone_and_exact_over_the_whole_inventory_range():
    """The default Q = 100 at the paper's parameters: every bid offset finite and strictly increasing in q over
    -Q .. Q - 1 at T = 600 s and 6.5 h, and equal to the 70-digit reference (GLFT_Q100_BIDS) to 1e-11 at
    t = 0, 300, 599 -- including q = 65 (7.459403) and q = 99 (9.320168), where v_q / v_0 is 1e-19 and 1e-41.
    The eigendecomposition (absolute accuracy eps max v) is the wrong tool there: negative entries in its v and
    a bid at q = 65 off by more than 1 tick (it printed -0.172 before the uniformization)."""
    g = GLFT2013(T=600.0, Q=100, **GLFT_EX)
    for t, ref in GLFT_Q100_BIDS.items():
        bid, ask = g.offsets(np.arange(-100, 101), t)
        assert np.all(np.isfinite(bid[:-1])) and np.all(np.isfinite(ask[1:]))
        assert np.all(np.diff(bid[:-1]) > 0) and np.all(np.diff(ask[1:]) < 0)
        assert np.max(np.abs(bid[[q + 100 for q in GLFT_Q100_QS]] - np.array(ref))) < 1e-11
        assert np.allclose(bid[:-1], ask[1:][::-1], atol=1e-11)  # delta_b(q) == delta_a(-q)
    v_eigh = g.v(0.0, method="eigh")
    assert (v_eigh < 0).any() and (g.v(0.0) > 0).all()
    b_eigh = np.log(v_eigh[165] / v_eigh[166]) / g.k + np.log1p(g.gamma / g.k) / g.gamma
    assert not (abs(b_eigh - GLFT_Q100_BIDS[0.0][5]) < 1.0)  # q = 65: eigh is noise, not a quote
    day = GLFT2013(T=6.5 * 3600.0, Q=100, **GLFT_EX)
    bid, _ = day.offsets(np.arange(-100, 101), 0.0)
    assert np.all(np.isfinite(bid[:-1])) and np.all(np.diff(bid[:-1]) > 0)


def test_glft_at_6_5h_equals_the_exact_ground_state_by_the_inward_recurrence():
    """At T = 6.5 h the transients exp(-(w_j - w_0) T) are below 1e-300, so v is the ground state of M, whose
    tail ratios r_q = v_{q-1} / v_q follow the three-term recurrence r_q = (alpha q^2 - w_0) / eta - 1 / r_{q+1}
    from r_Q = (alpha Q^2 - w_0) / eta, stable inward: bid(q) = ln(r_{q+1}) / k + c for q = 0 .. Q - 1 must
    agree with the uniformization to 1e-12 on every q (it does to 6e-14)."""
    Q = 100
    g = GLFT2013(T=6.5 * 3600.0, Q=Q, **GLFT_EX)
    w0 = np.linalg.eigvalsh(g.M)[0]
    r = np.empty(Q + 1)
    r[Q] = (g.alpha * Q**2 - w0) / g.eta
    for q in range(Q - 1, 0, -1):
        r[q] = (g.alpha * q**2 - w0) / g.eta - 1.0 / r[q + 1]
    bid_rec = np.log(r[1:]) / g.k + np.log1p(g.gamma / g.k) / g.gamma
    bid, _ = g.offsets(np.arange(0, Q), 0.0)
    assert np.max(np.abs(bid - bid_rec)) < 1e-12
    with pytest.raises(ValueError):
        GLFT2013(T=600.0, Q=150, **GLFT_EX)  # the ground state spans > 70 decades: not resolvable, refused


def test_glft_exact_vs_asymptotic_within_5e3_for_q_up_to_20():
    """Plan oracle: |exact - asymptotic| <= 5e-3 tick for |q| <= 20; asymptotic values 3.312915 (q=0),
    3.652244 / 2.973586 (q=5), 3.991574 / 2.634257 (q=10), 4.670232 / 1.955599 (q=20)."""
    g = GLFT2013(T=600.0, Q=30, **GLFT_EX)
    asym = GLFTAsymptotic(**GLFT_EX)
    qs = np.arange(-20, 21)
    b_e, a_e = g.offsets(qs)
    b_a, a_a = asym.offsets(qs)
    assert np.max(np.abs(b_e - b_a)) <= 5e-3 and np.max(np.abs(a_e - a_a)) <= 5e-3
    assert asym.offsets(0)[0] == pytest.approx(3.312915, abs=1e-6)
    assert asym.offsets(5) == pytest.approx((3.652244, 2.973586), abs=1e-6)
    assert asym.offsets(10) == pytest.approx((3.991574, 2.634257), abs=1e-6)
    assert asym.offsets(20) == pytest.approx((4.670232, 1.955599), abs=1e-6)
    assert asym.spread(0) == pytest.approx(6.625830, abs=1e-6)


def test_glft_symmetry_terminal_value_and_Q_convergence():
    """delta_b(q) == delta_a(-q) to 1e-12; at t = T every quote is (1/gamma) ln(1+gamma/k) = 3.278982282;
    Q = 30 vs 60 differ by < 1e-8 at q = 0 (1.2e-9); gamma = 1e-6 spread -> 2/k as Q grows (Q=100: 6.666710)."""
    g = GLFT2013(T=600.0, Q=30, **GLFT_EX)
    for q in (0, 1, 5, 17, 29):
        assert g.offsets(q)[0] == pytest.approx(g.offsets(-q)[1], abs=1e-12)
    bT, aT = g.offsets(np.arange(-29, 30), 600.0)
    assert np.allclose(bT, 3.278982282, atol=1e-8) and np.allclose(aT, 3.278982282, atol=1e-8)
    g60 = GLFT2013(T=600.0, Q=60, **GLFT_EX)
    assert abs(g.offsets(0)[0] - g60.offsets(0)[0]) < 1e-8
    small = GLFT2013(gamma=1e-6, k=0.3, sigma=0.3, A=0.9, T=600.0, Q=100)
    assert sum(small.offsets(0)) == pytest.approx(6.666710, abs=1e-5)
    assert sum(GLFT2013(gamma=1e-6, k=0.3, sigma=0.3, A=0.9, T=600.0, Q=30).offsets(0)) == pytest.approx(6.674845, abs=1e-5)


def test_glft_boundary_inventory_has_no_quote_on_the_full_side_and_is_stable_intraday():
    g = GLFT2013(T=600.0, Q=5, **GLFT_EX)
    assert g.offsets(5)[0] == np.inf and np.isfinite(g.offsets(5)[1])
    assert g.offsets(-5)[1] == np.inf and np.isfinite(g.offsets(-5)[0])
    with pytest.raises(ValueError):
        g.offsets(6)
    day = GLFT2013(T=6.5 * 3600, Q=100, **GLFT_EX)
    b, a = day.offsets(5, 0.0)
    assert np.isfinite(b) and np.isfinite(a)
    assert abs(b - GLFTAsymptotic(**GLFT_EX).offsets(5)[0]) < 5e-3


def test_asymptotic_spread_vs_gamma_is_non_monotone_the_dip():
    """GLFT Fig 6/7: (2/gamma) ln(1+gamma/k) falls with gamma while the sqrt term rises, so the spread over
    gamma in {1e-4 .. 10} has an interior minimum."""
    gammas = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    spreads = np.array([GLFTAsymptotic(gamma=g, k=0.3, sigma=0.3, A=0.9).spread(0) for g in gammas])
    i_min = int(np.argmin(spreads))
    assert 0 < i_min < len(gammas) - 1
    assert spreads[0] > spreads[i_min] < spreads[-1]


# ----------------------------------------------------------------------------------------------------------
# Stoikov-Saglam 2009 and the vol-space derivation
# ----------------------------------------------------------------------------------------------------------


def _ss_greeks(t_mat_days):
    """Their units: S=K=100, sigma=0.01/day, Tmat - t_n days."""
    return black_greeks(100.0, 100.0, t_mat_days - 1.0, 0.01, "C")


def test_ss_eq17_gamma_S2_sigma_T_equals_vega():
    """Stoikov-Saglam eq. (17): Gamma S^2 sigma (Tmat - t) == Vega; S=K=100, sigma=0.01/day, 399 d:
    Gamma 0.01987273193, Vega 792.922004, gap 1e-13."""
    _, G, V, _ = _ss_greeks(400.0)
    assert G == pytest.approx(0.01987273193, abs=1e-10)
    assert V == pytest.approx(792.922004, abs=1e-5)
    assert G * 100.0**2 * 0.01 * 399.0 - V == pytest.approx(0.0, abs=1e-9)


def test_ss_theorem4_k_and_vega_share():
    """Thm 4: k = [1/2 sigma^2 tau + alpha^2 (Tmat-t)^2] Gamma^2 S^4 sigma^2 tau, tau=0.5 d, alpha=0.00035/sqrt(d)
    = 1/2 Gamma^2 S^4 sigma^4 tau^2 + alpha^2 tau Vega^2 = 0.03855879058 (gamma part 4.93657e-5, vega
    share 0.99872); Tmat = 2 d: k = 0.0199913505, vega share 0.004876."""
    _, G, V, _ = _ss_greeks(400.0)
    vt, gt = residual_variance_terms(0.00035, 0.5, 0.01, 100.0, G, V)
    assert vt + gt == pytest.approx(0.03855879058, abs=1e-10)
    assert gt == pytest.approx(4.93657e-5, rel=1e-5)
    assert vt / (vt + gt) == pytest.approx(0.99872, abs=1e-5)
    k_paper = (0.5 * 0.01**2 * 0.5 + 0.00035**2 * 399.0**2) * G**2 * 100.0**4 * 0.01**2 * 0.5
    assert vt + gt == pytest.approx(k_paper, rel=1e-12)
    _, G2, V2, _ = _ss_greeks(2.0)
    vt2, gt2 = residual_variance_terms(0.00035, 0.5, 0.01, 100.0, G2, V2)
    assert vt2 + gt2 == pytest.approx(0.0199913505, abs=1e-9)
    assert vt2 / (vt2 + gt2) == pytest.approx(0.004876, abs=1e-5)


def test_ss_premiums_and_theorem5_recursion():
    """C=40, D=200, gamma=0.1: risk-neutral premium C/2D = 0.1, cap C/D = 0.2; ask premium at q=1
    = 0.09807206047, bid 0.1057838186; ask hits 0 at q = 26.43442338; Thm 5 recursion m_i = m_{i+1} +
    dt D m_{i+1}^2 J_i (J=2, n=100, dt=0.01): m_n = -0.003855879058 -> m_0 = -0.001508026137."""
    _, G, V, _ = _ss_greeks(400.0)
    vt, gt = residual_variance_terms(0.00035, 0.5, 0.01, 100.0, G, V)
    k, gam, C, D = vt + gt, 0.1, 40.0, 200.0
    ask = max(0.0, min(C / D, C / (2 * D) - gam * k * (1 - 0.5)))
    bid = max(0.0, min(C / D, C / (2 * D) + gam * k * (1 + 0.5)))
    assert ask == pytest.approx(0.09807206047, abs=1e-10) and bid == pytest.approx(0.1057838186, abs=1e-10)
    assert (C / (2 * D)) / (gam * k) + 0.5 == pytest.approx(26.43442338, abs=1e-6)
    m = -gam * k
    assert m == pytest.approx(-0.003855879058, abs=1e-11)
    for _ in range(100):
        m = m + 0.01 * D * m**2 * 2
    assert m == pytest.approx(-0.001508026137, abs=1e-11)


def test_single_strike_vol_space_tilt_equals_ss_dollar_tilt_over_vega():
    """Vol-space skew per lot = gamma k / Vega = 4.862873068e-6 (vega term alone gamma alpha^2 tau Vega =
    4.856647275e-6) at their params; long vega lowers both quotes."""
    _, G, V, _ = _ss_greeks(400.0)
    skew = vol_space_skew(0.1, 0.00035, 0.5, 0.01, 100.0, [1.0], [G], [V])
    assert skew[0] == pytest.approx(4.862873068e-6, rel=1e-8)
    assert 0.1 * 0.00035**2 * 0.5 * V == pytest.approx(4.856647275e-6, rel=1e-8)
    assert vol_space_skew(0.1, 0.00035, 0.5, 0.01, 100.0, [-2.0], [G], [V])[0] == pytest.approx(-2 * skew[0], rel=1e-12)
    assert vol_space_skew(0.1, 0.00035, 0.5, 0.01, 100.0, [0.0], [G], [V])[0] == 0.0


# ----------------------------------------------------------------------------------------------------------
# strip wrappers
# ----------------------------------------------------------------------------------------------------------


def _strip():
    f = SyntheticFair(S0=100.0, sigma0=0.2, skew_s=-0.1, curv_c=0.3, alpha=0.6, spot_vol=0.2,
                      strikes=(90.0, 95.0, 100.0, 105.0, 110.0), expiries=(30 / 365, 90 / 365))
    return f, f.snapshot()


def _space_params(T=6.5 * 3600):
    return dict(gamma=0.05, k=20.0, A=0.5, T=T, alpha_s=quoter.per_sqrt_second(0.6),
                sigma_s=quoter.per_sqrt_second(0.2), tau_s=600.0)


def test_price_space_glft_refits_sigma_opt_beyond_one_percent_and_rejects_another_strip():
    """kind='glft' builds one GLFT2013 per instrument on the first snapshot's sigma_opt and re-fits it when
    sigma_opt has moved by more than 1 %: an object first used on the sigma 0.2 strip then on a sigma 0.6 one
    (same instruments) quotes exactly what a fresh object quotes (a review found a 0.0005 $ half-spread bias
    when it stayed frozen); a snapshot within the tolerance keeps the fitted object; different strikes raise."""
    f, snap = _strip()
    n = snap.n_instruments
    shared = PriceSpace(kind="glft", **_space_params(T=600.0), Q=20)
    fresh = PriceSpace(kind="glft", **_space_params(T=600.0), Q=20)
    b1, _ = shared.quotes(snap, np.zeros(n), 0.0)
    fitted = dict(shared._exact)
    b2, _ = shared.quotes(snap, np.zeros(n), 0.0)
    assert np.array_equal(b1, b2) and all(shared._exact[j] is fitted[j] for j in range(n))
    hot = SyntheticFair(S0=100.0, sigma0=0.6, skew_s=-0.1, curv_c=0.3, alpha=0.6, spot_vol=0.2,
                        strikes=(90.0, 95.0, 100.0, 105.0, 110.0), expiries=(30 / 365, 90 / 365)).snapshot()
    bs, as_ = shared.quotes(hot, np.ones(n), 0.0)
    bf, af = fresh.quotes(hot, np.ones(n), 0.0)
    refit = np.array([shared._exact[j] is not fitted[j] for j in range(n)])
    sig_hot = shared.sigma_opt(hot)
    assert refit.any() and np.array_equal(bs[refit], bf[refit]) and np.array_equal(as_[refit], af[refit])
    kept = ~refit  # within 1 % of the fitted sigma_opt: the fitted object stays, quotes within 1e-5 $
    assert np.all(np.abs(sig_hot[kept] / np.array([fitted[j].sigma for j in range(n)])[kept] - 1) <= 0.01)
    assert np.max(np.abs(bs - bf)) < 1e-5 and np.max(np.abs(as_ - af)) < 1e-5
    other = SyntheticFair(S0=100.0, sigma0=0.2, strikes=(80.0, 90.0, 100.0, 110.0, 120.0), expiries=(30 / 365,)).snapshot()
    with pytest.raises(ValueError):
        shared.quotes(other, np.zeros(other.n_instruments), 0.0)


def test_price_space_quotes_straddle_fair_and_skew_with_inventory():
    _, snap = _strip()
    for kind in ("as", "glft_asym", "glft"):
        ps = PriceSpace(kind=kind, **_space_params(), Q=20)
        n = snap.n_instruments
        bid0, ask0 = ps.quotes(snap, np.zeros(n), 0.0)
        assert np.all(bid0 < snap.prices) and np.all(ask0 > snap.prices)
        assert np.allclose(0.5 * (bid0 + ask0), snap.prices, atol=1e-9)  # symmetric at q = 0
        wide = PriceSpace(kind=kind, **{**_space_params(), "k": 2.0}, Q=20)
        assert np.isnan(wide.quotes(snap, np.zeros(n), 0.0)[0][1])  # deep OTM bid <= 0 is pulled
        q = np.zeros(n)
        q[4] = 3
        bid3, ask3 = ps.quotes(snap, q, 0.0)
        assert bid3[4] < bid0[4] and ask3[4] < ask0[4]  # long -> both quotes lower
        others = np.arange(n) != 4
        assert np.allclose(bid3[others], bid0[others]) and np.allclose(ask3[others], ask0[others])  # per strike


def test_price_space_sigma_uses_hedged_residual_unless_hedger_off():
    _, snap = _strip()
    p = _space_params()
    hedged = PriceSpace(kind="as", **p).sigma_opt(snap)
    unhedged = PriceSpace(kind="as", hedged=False, **p).sigma_opt(snap)
    vt, gt = residual_variance_terms(p["alpha_s"], p["tau_s"], p["sigma_s"], snap.spot, snap.gammas, snap.vegas)
    assert np.allclose(hedged**2, (vt + gt) / p["tau_s"], rtol=1e-12)
    assert np.allclose(unhedged**2 - hedged**2, (snap.deltas * p["sigma_s"] * snap.spot) ** 2, rtol=1e-9)
    assert np.all(unhedged > hedged)


def test_vol_space_quotes_are_black_at_shifted_vols_and_net_vega_skews_all_strikes():
    _, snap = _strip()
    p = _space_params()
    vs = VolSpace(gamma=p["gamma"], alpha_s=p["alpha_s"], sigma_s=p["sigma_s"], tau_s=p["tau_s"], hs_vol=0.005)
    n = snap.n_instruments
    bid0, ask0 = vs.quotes(snap, np.zeros(n))
    assert np.allclose(bid0, black_price(snap.spot, snap.K, snap.T_rem, snap.vols - 0.005, snap.right))
    assert np.allclose(ask0, black_price(snap.spot, snap.K, snap.T_rem, snap.vols + 0.005, snap.right))
    q = np.zeros(n)
    q[4] = 3
    bid3, ask3 = vs.quotes(snap, q)
    assert np.all(bid3 < bid0) and np.all(ask3 < ask0)  # parallel shift: every strike moves
    bv, av = vs.quote_vols(snap, q)
    assert np.allclose(av - bv, 0.01)
    q2 = np.zeros(n)
    q2[4], q2[5] = 1, -1  # long call / short put same strike: vega nets to ~0, gamma too
    assert np.max(np.abs(vs.skew(snap, q2))) < 1e-15


def test_match_spreads_matches_dollar_half_spread_sum_and_skew_at_q0():
    """sum Vega hs_vol == sum hs_$ at q = 0, and the summed linearised $ skew per lot agrees; with AS2008 at
    tau = T - t and the hedged sigma_opt the skew scale is 1 by algebra."""
    _, snap = _strip()
    p = _space_params(T=600.0)
    n = snap.n_instruments
    ps = PriceSpace(kind="as", **p)
    vs = VolSpace(gamma=p["gamma"], alpha_s=p["alpha_s"], sigma_s=p["sigma_s"], tau_s=p["tau_s"], hs_vol=0.123)
    matched, rep = match_spreads(ps, vs, snap, t=0.0)
    b0, a0 = ps.offsets(snap, np.zeros(n), 0.0)
    assert rep.hs_usd_sum == pytest.approx(np.sum(0.5 * (b0 + a0)))
    assert np.sum(snap.vegas) * matched.hs_vol == pytest.approx(rep.hs_usd_sum, rel=1e-12)
    assert matched.skew_scale == pytest.approx(1.0, rel=1e-9)
    assert rep.vol_skew_sum == pytest.approx(rep.price_skew_sum, rel=1e-9)
    assert vs.hs_vol == 0.123  # the input is untouched; a new object came back
    ps2 = PriceSpace(kind="glft_asym", **p)
    matched2, rep2 = match_spreads(ps2, vs, snap)
    assert rep2.vol_skew_sum == pytest.approx(rep2.price_skew_sum, rel=1e-9)
    assert matched2.skew_scale != pytest.approx(1.0)


def test_round_to_tick_bids_down_asks_up_and_pulls_nonpositive_bids():
    bid, ask = round_to_tick([1.234, 0.004, 2.0], [1.236, 0.011, 2.0], 0.01)
    assert bid[0] == pytest.approx(1.23) and ask[0] == pytest.approx(1.24)
    assert np.isnan(bid[1]) and ask[1] == pytest.approx(0.02)
    assert bid[2] == pytest.approx(2.0) and ask[2] == pytest.approx(2.0)
    b, a = round_to_tick([1.234], [1.236], 0.05)
    assert b[0] == pytest.approx(1.20) and a[0] == pytest.approx(1.25)
    _, snap = _strip()
    ps = PriceSpace(kind="glft_asym", tick=0.05, **_space_params())
    bid, ask = ps.quotes(snap, np.zeros(snap.n_instruments), 0.0)
    ok = ~np.isnan(bid)
    assert np.allclose(np.round(bid[ok] / 0.05) * 0.05, bid[ok], atol=1e-9)
    assert np.allclose(np.round(ask / 0.05) * 0.05, ask, atol=1e-9)
    assert np.all(bid[ok] <= snap.prices[ok]) and np.all(ask >= snap.prices)


def test_guards_pull_quotes():
    _, snap = _strip()
    n = snap.n_instruments
    bid, ask = np.full(n, 1.0), np.full(n, 1.2)
    g = MaxLossGuard(100.0)
    b, a = g.apply(bid, ask, -50.0)
    assert np.array_equal(b, bid) and np.array_equal(a, ask)
    b, a = g.apply(bid, ask, -100.5)
    assert np.all(np.isnan(b)) and np.all(np.isnan(a))
    q = np.zeros(n)
    q[4] = 10
    cap = 0.5 * float(np.sum(q * snap.vegas))
    b, a = vega_cap(bid, ask, q, snap.vegas, cap)
    assert np.all(np.isnan(b)) and np.array_equal(a, ask)
    b, a = vega_cap(bid, ask, -q, snap.vegas, cap)
    assert np.array_equal(b, bid) and np.all(np.isnan(a))
    b, a = vega_cap(bid, ask, q, snap.vegas, 10 * cap)
    assert np.array_equal(b, bid) and np.array_equal(a, ask)
    assert not np.isnan(bid).any()  # inputs untouched


def test_quoter_inputs_are_past_measurable_through_past_only_fair():
    """A quoter fed through PastOnlyFair can quote at the clock and never at clock + 1."""
    f, _ = _strip()
    gen = np.random.default_rng(0)
    path = f.path(gen.standard_normal(20), gen.standard_normal(20), gen.random(20), 1.0 / YEAR_SECONDS)
    view = PastOnlyFair(path)
    ps = PriceSpace(kind="glft_asym", **_space_params())
    q = np.zeros(path.n_instruments)
    for i in range(5):
        bid, ask = ps.quotes(view.current(), q, float(i))
        assert np.all(bid < ask)
        with pytest.raises(FutureAccessError):
            ps.quotes(view.snapshot(i + 1), q, float(i))
        view.advance()


def test_wrapper_validation():
    with pytest.raises(ValueError):
        PriceSpace(kind="bogus", **_space_params())
    with pytest.raises(ValueError):
        VolSpace(gamma=0.0, alpha_s=1e-4, sigma_s=1e-4, tau_s=600.0)
    with pytest.raises(ValueError):
        MaxLossGuard(-1.0)
    with pytest.raises(ValueError):
        vega_cap([1.0], [1.1], [0.0], [1.0], 0.0)
    with pytest.raises(ValueError):
        AS2008(gamma=0.1, k=1.5, sigma=2.0, T=1.0).offsets(0, 2.0)
