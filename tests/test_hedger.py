"""hedger.py: Whalley-Wilmott 1997 band oracle, band / time rules, cost model (tcakit sqrt-law form)."""

import numpy as np
import pytest

from quotesim.fair import black_greeks
from quotesim.hedger import (
    BandHedger,
    CostModel,
    HedgeDecision,
    NoHedger,
    TimeHedger,
    band_ww,
    default_cost_model,
    target_shares,
)


def test_whalley_wilmott_band_oracle_and_lambda_one_third_scaling():
    """Whalley-Wilmott 1997: H = (3/2 e^{-r tau} lam S Gamma^2 / gamma)^{1/3}; S=K=100, sigma=0.2,
    T=0.25, r=0, lam=0.001, gamma=1: Gamma 0.03984439141, H 0.06198337475 shares/option ($619.83 delta per
    100-lot); 8x lam -> 2x H exactly (Zakamouline 2005 confirms the lam^{1/3} law)."""
    _, G, _, _ = black_greeks(100.0, 100.0, 0.25, 0.2, "C")
    assert G == pytest.approx(0.03984439141, abs=1e-10)
    H = band_ww(100.0, G, 0.001, 1.0, r=0.0, tau=0.25)
    assert H == pytest.approx(0.06198337475, abs=1e-10)
    assert H * 100.0 * 100 == pytest.approx(619.8337475, abs=1e-6)
    assert band_ww(100.0, G, 0.008, 1.0, 0.0, 0.25) / H == pytest.approx(2.0, abs=1e-12)
    assert band_ww(100.0, 2 * G, 0.001, 1.0, 0.0, 0.25) / H == pytest.approx(2 ** (2 / 3), abs=1e-12)
    assert band_ww(100.0, G, 0.001, 1.0, r=0.05, tau=0.25) == pytest.approx(H * np.exp(-0.05 * 0.25) ** (1 / 3), abs=1e-12)
    with pytest.raises(ValueError):
        band_ww(100.0, G, 0.001, 0.0)


def test_default_cost_model_formula_and_sign():
    """half_spread |dh| + Y sigma_daily sqrt(|dh|/ADV) S |dh|: 100 shares at S=100, hs=0.01, Y=0.5,
    sigma_daily=0.01, ADV=1e6 -> 1 + 0.5 * 0.01 * 0.01 * 100 * 100 = 1.5; symmetric in the sign of dh."""
    assert default_cost_model(100.0, 100.0, 0.01, 0.5, 0.01, 1e6) == pytest.approx(1.5, abs=1e-12)
    assert default_cost_model(-100.0, 100.0, 0.01, 0.5, 0.01, 1e6) == pytest.approx(1.5, abs=1e-12)
    assert default_cost_model(0.0, 100.0, 0.01, 0.5, 0.01, 1e6) == 0.0
    cm = CostModel(half_spread=0.01, Y=0.5, sigma_daily=0.01, adv=1e6)
    assert cm(100.0, 100.0) == pytest.approx(1.5, abs=1e-12)
    assert cm(400.0, 100.0) == pytest.approx(4.0 + 0.5 * 0.01 * 0.02 * 100 * 400, abs=1e-12)  # sqrt-law: 4x qty -> 8x impact
    assert CostModel(Y=0.0)(1e5, 50.0) == pytest.approx(0.01 * 1e5)
    with pytest.raises(ValueError):
        CostModel(adv=0.0)


def test_target_shares_flattens_delta():
    q = np.array([2.0, -1.0, 0.0])
    d = np.array([0.5, -0.4, 0.3])
    assert target_shares(q, d) == pytest.approx(-(1.0 + 0.4))
    assert target_shares(q, d, multiplier=100.0) == pytest.approx(-140.0)


def test_band_hedger_holds_inside_band_and_lands_on_target_outside():
    cm = CostModel(half_spread=0.02, Y=0.0)
    h = BandHedger(band_delta_usd=50.0, cost_model=cm)
    S = 100.0
    inside = h.decide(h=10.0, target=10.4, S=S, t=0.0)  # $40 mismatch
    assert inside == HedgeDecision(0.0, 0.0, 10.0) and not inside.traded
    outside = h.decide(h=10.0, target=10.6, S=S, t=0.0)  # $60 mismatch
    assert outside.traded and outside.new_h == 10.6
    assert outside.shares == pytest.approx(0.6) and outside.cost == pytest.approx(0.02 * 0.6)
    assert h.decide(h=0.0, target=0.0, S=S).shares == 0.0
    zero = BandHedger(0.0, cm)
    assert zero.decide(1.0, 1.0 + 1e-9, S).traded  # band 0: any nonzero mismatch trades
    assert not zero.decide(1.0, 1.0, S).traded
    with pytest.raises(ValueError):
        BandHedger(-1.0)


def test_band_to_zero_and_cost_to_zero_gives_zero_residual_delta_at_zero_cost():
    """Plan oracle: band -> 0 and cost -> 0 -> residual delta 0 every step, hedge cost >= 0 (here 0)."""
    h = BandHedger(0.0, CostModel(half_spread=0.0, Y=0.0))
    gen = np.random.default_rng(0)
    pos, total_cost = 0.0, 0.0
    for _ in range(200):
        target = float(gen.normal())
        d = h.decide(pos, target, 100.0)
        pos, total_cost = d.new_h, total_cost + d.cost
        assert pos == target and d.cost >= 0.0
    assert total_cost == 0.0


def test_time_hedger_trades_on_schedule_only():
    cm = CostModel(half_spread=0.01, Y=0.0)
    h = TimeHedger(every_s=60.0, cost_model=cm)
    first = h.decide(0.0, 3.0, 100.0, t=0.0, t_last=None)
    assert first.traded and first.new_h == 3.0 and first.cost == pytest.approx(0.03)
    assert not h.decide(3.0, 5.0, 100.0, t=30.0, t_last=0.0).traded
    assert not h.decide(3.0, 5.0, 100.0, t=59.0, t_last=0.0).traded
    due = h.decide(3.0, 5.0, 100.0, t=60.0, t_last=0.0)
    assert due.traded and due.shares == pytest.approx(2.0)
    assert not h.decide(5.0, 5.0, 100.0, t=120.0, t_last=60.0).traded  # on schedule but already on target
    with pytest.raises(ValueError):
        TimeHedger(0.0)


def test_no_hedger_never_trades_and_cost_is_nonnegative_for_every_rule():
    assert NoHedger().decide(1.0, -5.0, 100.0).shares == 0.0
    gen = np.random.default_rng(3)
    for rule in (BandHedger(10.0), TimeHedger(5.0), NoHedger()):
        pos, t_last = 0.0, None
        for i in range(100):
            d = rule.decide(pos, float(gen.normal(scale=3)), 100.0, t=float(i), t_last=t_last)
            assert d.cost >= 0.0 and (d.shares != 0.0 or d.cost == 0.0)
            if d.traded:
                pos, t_last = d.new_h, float(i)


def test_negative_cost_model_is_rejected():
    with pytest.raises(ValueError):
        BandHedger(0.0, cost_model=lambda shares, S: -1.0).decide(0.0, 1.0, 100.0)
