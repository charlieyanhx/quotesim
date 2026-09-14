"""fair.py: Black Greeks vs finite differences, parity, dynamics, path == step sequence, past-only guard."""

import numpy as np
import pytest

from quotesim import fair
from quotesim.fair import FutureAccessError, PastOnlyFair, SyntheticFair, black_greeks, black_price


def _fair(**kw):
    base = dict(S0=100.0, sigma0=0.2, skew_s=-0.1, curv_c=0.5, alpha=0.5, kappa_vol=2.0, spot_vol=0.2,
                strikes=(90.0, 95.0, 100.0, 105.0, 110.0), expiries=(30 / 365, 60 / 365))
    base.update(kw)
    return SyntheticFair(**base)


# ----------------------------------------------------------------------------------------------------------
# Black
# ----------------------------------------------------------------------------------------------------------


def test_black_call_matches_textbook_value_and_parity():
    """S=K=100, sigma=0.2, T=1, r=q=0: call = 7.965567455 (Black 1976 with F=S; recomputed with scipy).
    Parity at r=q=0: C - P == S - K to 1e-10."""
    c = black_price(100.0, 100.0, 1.0, 0.2, "C")
    p = black_price(100.0, 100.0, 1.0, 0.2, "P")
    assert c == pytest.approx(7.965567455, abs=1e-8)
    assert c - p == pytest.approx(0.0, abs=1e-10)
    S, K = 100.0, np.array([80.0, 100.0, 125.0])
    assert np.max(np.abs(black_price(S, K, 0.3, 0.25, "C") - black_price(S, K, 0.3, 0.25, "P") - (S - K))) < 1e-10


@pytest.mark.parametrize("right", ["C", "P"])
@pytest.mark.parametrize("K", [85.0, 100.0, 118.0])
def test_greeks_match_central_finite_differences(right, K):
    """Every analytic Greek vs a central finite difference of black_price at fixed sigma: delta, gamma,
    vega, theta (per year, = -dF/dT) to 1e-6 relative-or-absolute."""
    S, T, sig = 100.0, 45 / 365, 0.23
    d, g, v, th = black_greeks(S, K, T, sig, right)
    h = 1e-3
    fd_delta = (black_price(S + h, K, T, sig, right) - black_price(S - h, K, T, sig, right)) / (2 * h)
    fd_gamma = (black_price(S + h, K, T, sig, right) - 2 * black_price(S, K, T, sig, right) + black_price(S - h, K, T, sig, right)) / h**2
    hv = 1e-5
    fd_vega = (black_price(S, K, T, sig + hv, right) - black_price(S, K, T, sig - hv, right)) / (2 * hv)
    ht = 1e-6
    fd_theta = -(black_price(S, K, T + ht, sig, right) - black_price(S, K, T - ht, sig, right)) / (2 * ht)
    assert d == pytest.approx(fd_delta, abs=1e-7)
    assert g == pytest.approx(fd_gamma, rel=1e-5, abs=1e-8)
    assert v == pytest.approx(fd_vega, rel=1e-7, abs=1e-8)
    assert th == pytest.approx(fd_theta, rel=1e-6, abs=1e-6)


def test_expired_option_is_intrinsic_with_zero_second_order_greeks():
    assert black_price(105.0, 100.0, 0.0, 0.2, "C") == 5.0
    assert black_price(105.0, 100.0, -1.0, 0.2, "P") == 0.0
    d, g, v, th = black_greeks(105.0, 100.0, 0.0, 0.2, "C")
    assert (d, g, v, th) == (1.0, 0.0, 0.0, 0.0)


# ----------------------------------------------------------------------------------------------------------
# SyntheticFair
# ----------------------------------------------------------------------------------------------------------


def test_instrument_table_shape_and_columns():
    f = _fair()
    assert list(f.instruments.columns) == ["K", "T_years", "right", "multiplier"]
    assert len(f.instruments) == 5 * 2 * 2 == f.n_instruments
    assert set(f.instruments["right"]) == {"C", "P"}
    assert f.fair_prices().shape == (20,) and f.fair_vols().shape == (20,)


def test_smile_is_quadratic_in_log_moneyness_and_floored():
    f = _fair()
    k = np.log(np.array([90.0, 100.0, 110.0]) / 100.0)
    expected = 0.2 - 0.1 * k + 0.5 * k**2
    assert np.allclose(f.vol(k), expected, atol=1e-15)
    assert f.vol(0.0) == pytest.approx(0.2)
    assert f.vol(0.0, T=1.0) == f.vol(0.0)  # no term structure
    steep = _fair(sigma0=0.05, skew_s=-1.0, curv_c=0.0)
    assert steep.vol(0.5) == fair.VOL_FLOOR


def test_scalar_accessors_agree_with_strip_arrays():
    f = _fair()
    snap = f.snapshot()
    for j in range(f.n_instruments):
        K, T, right = snap.K[j], snap.T_rem[j], snap.right[j]
        assert f.price(K, T, right) == pytest.approx(snap.prices[j], abs=1e-12)
        assert f.delta(K, T, right) == pytest.approx(snap.deltas[j], abs=1e-12)
        assert f.gamma(K, T, right) == pytest.approx(snap.gammas[j], abs=1e-12)
        assert f.vega(K, T, right) == pytest.approx(snap.vegas[j], abs=1e-12)
        assert f.theta(K, T, right) == pytest.approx(snap.thetas[j], abs=1e-12)
    assert isinstance(f, fair.FairSurface)
    assert f.state() == {"t": 0.0, "S": 100.0, "sigma_atm": 0.2}


def test_step_dynamics_reproduce_the_documented_update():
    """One step with known normals: log S moves by -(1/2 sv^2) dt + sv sqrt(dt) z; sigma_ATM by OU + alpha
    sqrt(dt) z_vol; t advances by dt (years)."""
    f = _fair(alpha=0.5, kappa_vol=2.0, spot_vol=0.2)
    dt = 1.0 / fair.YEAR_SECONDS
    f.sigma_atm = 0.25
    f.step(1.5, -0.7, 0.9, dt)
    assert np.log(f.S / 100.0) == pytest.approx(-0.5 * 0.04 * dt + 0.2 * np.sqrt(dt) * 1.5, abs=1e-15)
    assert f.sigma_atm == pytest.approx(0.25 + 2.0 * (0.2 - 0.25) * dt + 0.5 * np.sqrt(dt) * (-0.7), abs=1e-15)
    assert f.t == dt
    assert f.T_rem()[0] == pytest.approx(30 / 365 - dt)


def test_merton_jump_from_a_single_uniform_is_compensated():
    """jump_u < p_J triggers J = mu + sd Phi^{-1}(u / p_J); the drift carries -ln(1 + p_J kappa_J) so that
    the one-step expectation of S is S (checked by a midpoint quadrature over the uniform to 1e-4)."""
    lam, mu, sd = 50.0, -0.05, 0.1
    dt = 0.01
    f = _fair(spot_vol=0.0, jump=(lam, mu, sd))
    p_j = 1 - np.exp(-lam * dt)
    g = f.copy()
    g.step(0.0, 0.0, 0.5 * p_j, dt)
    comp = np.log1p(p_j * (np.exp(mu + 0.5 * sd**2) - 1))
    assert np.log(g.S / 100.0) == pytest.approx(-comp + mu + 0.0, abs=1e-12)  # ppf(0.5) = 0
    us = (np.arange(20000) + 0.5) / 20000
    logs = []
    for u in us:
        h = f.copy()
        h.step(0.0, 0.0, u, dt)
        logs.append(np.log(h.S / 100.0))
    assert np.mean(np.exp(logs)) == pytest.approx(1.0, rel=1e-4)
    h = f.copy()
    h.step(0.0, 0.0, 0.999, dt)  # no jump this step
    assert np.log(h.S / 100.0) == pytest.approx(-comp, abs=1e-12)


def test_path_equals_the_sequence_of_step_snapshots_and_leaves_self_untouched():
    f = _fair(jump=(10.0, 0.0, 0.02))
    gen = np.random.default_rng(0)
    n = 50
    zs, zv, ju = gen.standard_normal(n), gen.standard_normal(n), gen.random(n)
    dt = 1.0 / fair.YEAR_SECONDS
    path = f.path(zs, zv, ju, dt)
    assert f.t == 0.0 and f.S == 100.0
    g = f.copy()
    for i in range(n):
        g.step(zs[i], zv[i], ju[i], dt)
        s = g.snapshot()
        assert path.spot[i + 1] == s.spot and path.sigma_atm[i + 1] == s.sigma_atm
        assert np.array_equal(path.prices[i + 1], s.prices) and np.array_equal(path.vegas[i + 1], s.vegas)
    assert path.n_steps == n and path.prices.shape == (n + 1, f.n_instruments)
    snap = path.snapshot(n)
    assert snap.t == pytest.approx(n * dt) and np.array_equal(snap.thetas, path.thetas[n])


def test_past_only_fair_raises_on_any_read_ahead_of_its_clock():
    f = _fair()
    gen = np.random.default_rng(1)
    path = f.path(gen.standard_normal(10), gen.standard_normal(10), gen.random(10), 1.0 / fair.YEAR_SECONDS)
    view = PastOnlyFair(path)
    assert view.clock == 0 and view.current().t == 0.0
    with pytest.raises(FutureAccessError):
        view.snapshot(1)
    view.advance(3)
    assert view.snapshot(3).spot == path.spot[3] and view.snapshot(2).spot == path.spot[2]
    assert view.spot == path.spot[3]
    with pytest.raises(FutureAccessError):
        view.snapshot(4)
    with pytest.raises(ValueError):
        view.advance(100)


def test_constructor_rejects_bad_parameters():
    with pytest.raises(ValueError):
        SyntheticFair(S0=-1.0, sigma0=0.2)
    with pytest.raises(ValueError):
        SyntheticFair(S0=100.0, sigma0=0.2, jump=(1.0, 0.0))
    with pytest.raises(ValueError):
        SyntheticFair(S0=100.0, sigma0=0.2, expiries=(0.0,))
    with pytest.raises(ValueError):
        _fair().step(0.0, 0.0, 0.5, 0.0)
