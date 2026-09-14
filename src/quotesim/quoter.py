"""Quoting rules: Avellaneda-Stoikov 2008, Gueant-Lehalle-Fernandez-Tapia 2013 (exact and asymptotic),
and the price-space / vol-space strip wrappers around them.

Units and conventions (this module works in SECONDS; prices in the instrument's own price unit, "ticks"
in the papers' examples, $ per unit of underlying in the strip wrappers):
- scalar quoters return OFFSETS (bid_offset, ask_offset) >= 0 from the fair/mid s: bid = s - bid_offset,
  ask = s + ask_offset. q = inventory in lots (+ long). gamma = risk aversion per price unit (1/$);
  k = intensity decay per price unit (1/$); sigma = fair vol in price units per sqrt(second); A = arrival
  rate per second; T = horizon in seconds; t = seconds elapsed.
- AS2008 (eqs 29-31 of the paper, per side per GLFT 2013 footnote 8):
    r(s, q, t) = s - q gamma sigma^2 (T - t); spread = gamma sigma^2 (T - t) + (2 / gamma) ln(1 + gamma / k)
    bid_offset = (1/gamma) ln(1 + gamma/k) + ((1 + 2q)/2) gamma sigma^2 (T - t);  ask_offset with (1 - 2q).
- GLFT2013 exact (their Prop 2 / Thm 1, finite inventory |q| <= Q): v(t) = exp(-M (T - t)) 1 with M the
  (2Q + 1) tridiagonal matrix diag(alpha q^2), off-diagonal -eta, alpha = k gamma sigma^2 / 2,
  eta = A (1 + gamma/k)^{-(1 + k/gamma)}; bid_offset(q) = (1/k) ln(v_q / v_{q+1}) + (1/gamma) ln(1 + gamma/k),
  ask_offset(q) = (1/k) ln(v_q / v_{q-1}) + ...; no bid at q = Q, no ask at q = -Q (offset = inf). v is
  computed by UNIFORMIZATION (Jensen 1953): with M' = M - w_min I (w_min the smallest eigenvalue, a free
  scale since every quote is a ratio), Lambda = max diag(M') and P = I - M' / Lambda >= 0 entrywise,
  v(t) = sum_n Poisson(n; Lambda (T - t)) P^n 1, every term non-negative, so each v_q carries RELATIVE
  rounding error (~1e-12) however small it is. v_q decays like exp(-1/2 sqrt(alpha/eta) q^2) (the ground
  state of the discrete oscillator), so a method with ABSOLUTE accuracy eps max(v) -- the eigendecomposition
  (kept as `v(t, method="eigh")`) or scipy.linalg.expm (`method="expm"`) -- returns rounding noise for
  |q| beyond ~50 at the paper's parameters (v_q / v_0 < 1e-8): tested against both at Q = 30, against
  the exact long-horizon ground state (inward three-term recurrence) at Q = 100, and for finiteness and
  monotonicity over the whole -Q..Q range. The boundary effect of a finite Q is ~0.008 tick at Q = 30, so
  Q defaults to 100; a Q whose ground state spans more than ~70 decades raises at construction.
- GLFTAsymptotic (their Prop 3, T -> infinity):
    bid_offset = c + ((2q + 1)/2) w, ask_offset = c - ((2q - 1)/2) w, c = (1/gamma) ln(1 + gamma/k),
    w = sqrt(sigma^2 gamma / (2 k A) (1 + gamma/k)^{1 + k/gamma}). Default for intraday horizons (A-S is
    only valid near T).
- price-space strip (PriceSpace): per instrument j the scalar rule on the option price with
    sigma_opt,j^2 = (Vega_j alpha)^2 + 1/2 Gamma_j^2 sigma_S^4 S^4 tau  [+ (Delta_j sigma_S S)^2 if unhedged]
  (the hedged residual variance rate; the delta term only when no hedger runs, otherwise hedged risk would
  be penalised twice). alpha_s = ABSOLUTE vol-of-vol per sqrt(second), sigma_s = spot vol per sqrt(second),
  tau_s = risk horizon in seconds. Convert from the fair's per-year numbers with `per_sqrt_second`.
- vol-space strip (VolSpace), a derivation from Stoikov-Saglam 2009 Thm 4 (their $ tilt / Vega, the
  single-instrument variance generalised to a covariance with the book), NOT their verbatim rule:
    skew_vol(j) = skew_scale [ gamma alpha^2 tau V_net + 1/2 gamma sigma_S^4 S^4 tau^2 Gamma_net (Gamma_j / Vega_j) ]
    bid_vol_j = sigma_fair,j - skew_vol(j) - hs_vol;  ask_vol_j = sigma_fair,j - skew_vol(j) + hs_vol
  with V_net = sum q Vega, Gamma_net = sum q Gamma; long vega LOWERS both vol quotes. Quotes are Black
  prices at those vols (floored at VOL_FLOOR). All time-unit products (alpha^2 tau, sigma^4 tau^2,
  sigma^2 (T - t), sigma^2 / A) are unit-invariant as long as one unit is used throughout a call.
- `match_spreads` returns a VolSpace whose hs_vol satisfies sum_j Vega_j hs_vol = sum_j hs_$,j (the
  price-space half-spreads at q = 0) and whose skew_scale makes the vega-weighted linearised $ skew per
  contract agree at q = 0; with AS2008 at tau = T - t and the hedged sigma_opt the scale is 1 by algebra.
- tick rounding: bids round DOWN, asks round UP (rounding never crosses fair); a bid that rounds to <= 0
  is pulled (NaN). Guards: `MaxLossGuard(max_loss)` pulls every quote (NaN) once running P&L < -max_loss;
  `vega_cap` pulls bids when the net vega is at or above +cap and asks when at or below -cap. One lot per
  fill; no size rules in v0.1.
- every wrapper reads a `fair.FairSnapshot` (past-measurable by construction when it comes from
  `fair.PastOnlyFair`) and the current inventory vector; nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from scipy.linalg import expm
from scipy.special import pdtrc

from quotesim.fair import VOL_FLOOR, YEAR_SECONDS, FairSnapshot, black_price


def per_sqrt_second(x_per_sqrt_year: float) -> float:
    """Convert a vol (per sqrt(year)) to per sqrt(second)."""
    return x_per_sqrt_year / np.sqrt(YEAR_SECONDS)


def _log_term(gamma: float, k: float) -> float:
    return np.log1p(gamma / k) / gamma


# ----------------------------------------------------------------------------------------------------------
# scalar quoters
# ----------------------------------------------------------------------------------------------------------


class ScalarQuoter(Protocol):
    def offsets(self, q, t: float = 0.0) -> tuple[np.ndarray, np.ndarray]: ...


def _check_positive(**kw):
    for name, val in kw.items():
        if val <= 0:
            raise ValueError(f"{name} must be > 0")


def as_offsets(q, t, gamma, k, sigma, T):
    """Vectorised AS2008 per-side offsets (sigma may be an array aligned with q)."""
    q = np.asarray(q, dtype=float)
    c = _log_term(gamma, k)
    var = gamma * np.asarray(sigma, dtype=float) ** 2 * (T - t)
    return c + 0.5 * (1.0 + 2.0 * q) * var, c + 0.5 * (1.0 - 2.0 * q) * var


def glft_asymptotic_offsets(q, gamma, k, sigma, A):
    """Vectorised GLFT 2013 Prop 3 offsets (sigma may be an array aligned with q)."""
    q = np.asarray(q, dtype=float)
    c = _log_term(gamma, k)
    w = np.sqrt(np.asarray(sigma, dtype=float) ** 2 * gamma / (2.0 * k * A) * (1.0 + gamma / k) ** (1.0 + k / gamma))
    return c + 0.5 * (2.0 * q + 1.0) * w, c - 0.5 * (2.0 * q - 1.0) * w


@dataclass(frozen=True)
class AS2008:
    gamma: float
    k: float
    sigma: float
    T: float

    def __post_init__(self):
        _check_positive(gamma=self.gamma, k=self.k, T=self.T)
        if self.sigma < 0:
            raise ValueError("sigma >= 0")

    def reservation(self, s, q, t: float = 0.0):
        return s - np.asarray(q, dtype=float) * self.gamma * self.sigma**2 * (self.T - t)

    def spread(self, t: float = 0.0) -> float:
        return float(self.gamma * self.sigma**2 * (self.T - t) + 2.0 * _log_term(self.gamma, self.k))

    def offsets(self, q, t: float = 0.0):
        if t > self.T:
            raise ValueError("t beyond the horizon T")
        return as_offsets(q, t, self.gamma, self.k, self.sigma, self.T)


@dataclass(frozen=True)
class GLFTAsymptotic:
    gamma: float
    k: float
    sigma: float
    A: float

    def __post_init__(self):
        _check_positive(gamma=self.gamma, k=self.k, A=self.A)
        if self.sigma < 0:
            raise ValueError("sigma >= 0")

    def offsets(self, q, t: float = 0.0):
        return glft_asymptotic_offsets(q, self.gamma, self.k, self.sigma, self.A)

    def spread(self, q=0) -> float:
        b, a = self.offsets(q)
        return float(b + a)


class GLFT2013:
    """Exact finite-inventory GLFT quotes; see module docstring. `offsets(q, t)` vectorises over q."""

    WINDOW = 20.0  # Poisson window half-width in sd units (+ WINDOW^2 terms): dropped mass <= exp(-200)
    TRANSIENT = 1e-12  # relative size of the slowest even mode left in P^n 1 when the powers stop
    MIN_RANGE = 1e-70  # min / max of the ground state that the window still resolves to 1e-12

    def __init__(self, gamma: float, k: float, sigma: float, A: float, T: float, Q: int = 100):
        _check_positive(gamma=gamma, k=k, A=A, T=T)
        if sigma < 0 or Q < 1:
            raise ValueError("sigma >= 0 and Q >= 1")
        self.gamma, self.k, self.sigma, self.A, self.T, self.Q = float(gamma), float(k), float(sigma), float(A), float(T), int(Q)
        self.alpha = 0.5 * k * gamma * sigma**2
        self.eta = A * (1.0 + gamma / k) ** (-(1.0 + k / gamma))
        qs = np.arange(-Q, Q + 1, dtype=float)
        self.M = np.diag(self.alpha * qs**2) - self.eta * (np.eye(2 * Q + 1, k=1) + np.eye(2 * Q + 1, k=-1))
        self._w, self._U = np.linalg.eigh(self.M)
        self._Ut1 = self._U.T @ np.ones(2 * Q + 1)
        self._shift = float(self._w[0])
        self.Lambda = float(self.alpha * Q**2 - self._shift)  # max diag of M - w_min I
        self._p_diag = 1.0 - (self.alpha * qs**2 - self._shift) / self.Lambda  # >= 0, == 0 at |q| = Q
        self._p_off = self.eta / self.Lambda
        self._W = self._powers()
        self.n_cut = self._W.shape[0] - 1
        w_end = self._W[-1]
        if w_end.min() < self.MIN_RANGE * w_end.max():
            raise ValueError(f"Q = {Q} spans more than {-np.log10(self.MIN_RANGE):.0f} decades of v at these parameters; lower Q")

    def _n_max(self, mu: float) -> int:
        return int(np.ceil(mu + self.WINDOW * np.sqrt(mu) + self.WINDOW**2))

    def _powers(self) -> np.ndarray:
        """Rows w_n = P^n 1 (all entries >= 0) up to the first n where the slowest even mode of P is below
        TRANSIENT relative (ratio test scaled by the spectral gap), or the last n with Poisson weight at tau = T."""
        n_max = self._n_max(self.Lambda * self.T)
        gap = (self._w[2] - self._w[0]) / self.Lambda if self._w.shape[0] > 2 else 1.0
        tol = self.TRANSIENT * gap
        rows = [np.ones(2 * self.Q + 1)]
        for _ in range(n_max):
            w = rows[-1]
            nxt = self._p_diag * w
            nxt[1:] += self._p_off * w[:-1]
            nxt[:-1] += self._p_off * w[1:]
            rows.append(nxt)
            if np.max(np.abs(nxt / w - 1.0)) <= tol:
                break
        return np.array(rows)

    def v(self, t: float = 0.0, method: str = "uniform") -> np.ndarray:
        """v(t) = exp(-(M - w_min I) (T - t)) 1, indexed by q + Q: the paper's v scaled by the constant
        exp(w_min (T - t)) (w_min = smallest eigenvalue of M) so entries stay O(1) at any horizon; every
        quote is a ratio v_q / v_{q +- 1}, unchanged by the scaling. method: "uniform" (default, relative
        accuracy in every entry), "eigh" / "expm" (absolute accuracy eps max(v): the tail is noise)."""
        if not 0.0 <= t <= self.T:
            raise ValueError("t must be in [0, T]")
        if method == "uniform":
            return self._v_uniform(self.Lambda * (self.T - t))
        if method == "expm":
            return expm(-(self.M - self._shift * np.eye(2 * self.Q + 1)) * (self.T - t)) @ np.ones(2 * self.Q + 1)
        if method == "eigh":
            return self._U @ (np.exp(-(self._w - self._shift) * (self.T - t)) * self._Ut1)
        raise ValueError("method must be 'uniform', 'eigh' or 'expm'")

    def _v_uniform(self, mu: float) -> np.ndarray:
        """sum_n Poisson(n; mu) w_n with the terms n >= n_cut collapsed onto w_{n_cut} (converged or
        weightless), the Poisson weights from the recursion p_{n+1} / p_n = mu / (n + 1) anchored at the
        mode and normalised to 1 - P(N >= n_cut), so every factor is positive and relative."""
        if mu == 0.0:
            return self._W[0].copy()
        lo = max(0, int(np.floor(mu - self.WINDOW * np.sqrt(mu) - self.WINDOW**2)))
        hi = min(self.n_cut, self._n_max(mu) + 1)
        rest = float(pdtrc(self.n_cut - 1, mu))  # P(N >= n_cut)
        if hi <= lo:
            return rest * self._W[self.n_cut]
        n0 = int(np.clip(np.floor(mu), lo, hi - 1))
        logw = np.zeros(hi - lo)
        up = np.arange(n0 + 1, hi)
        if up.size:
            logw[n0 + 1 - lo:] = np.cumsum(np.log(mu / up))
        dn = np.arange(n0, lo, -1)
        if dn.size:
            logw[: n0 - lo] = np.cumsum(np.log(dn / mu))[::-1]
        om = np.exp(logw)
        return (1.0 - rest) * (om @ self._W[lo:hi]) / om.sum() + rest * self._W[self.n_cut]

    def offsets(self, q, t: float = 0.0):
        q = np.asarray(q)
        if np.any(np.abs(q) > self.Q):
            raise ValueError("|q| > Q")
        v = self.v(t)
        idx = q.astype(int) + self.Q
        c = _log_term(self.gamma, self.k)
        up = np.where(idx + 1 <= 2 * self.Q, v[np.minimum(idx + 1, 2 * self.Q)], np.nan)
        dn = np.where(idx - 1 >= 0, v[np.maximum(idx - 1, 0)], np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            bid = np.where(np.isnan(up), np.inf, np.log(v[idx] / up) / self.k + c)
            ask = np.where(np.isnan(dn), np.inf, np.log(v[idx] / dn) / self.k + c)
        if bid.ndim == 0:
            return float(bid), float(ask)
        return bid, ask


# ----------------------------------------------------------------------------------------------------------
# strip helpers
# ----------------------------------------------------------------------------------------------------------


def residual_variance_terms(alpha, tau, sigma_s, S, gamma_j, vega_j):
    """(vega_term, gamma_term) of the per-contract hedged variance over the risk horizon tau:
    alpha^2 tau Vega^2 and 1/2 Gamma^2 S^4 sigma_s^4 tau^2 (Stoikov-Saglam 2009 Thm 4's k = their sum).
    Any consistent time unit."""
    vega_j = np.asarray(vega_j, dtype=float)
    gamma_j = np.asarray(gamma_j, dtype=float)
    return alpha**2 * tau * vega_j**2, 0.5 * gamma_j**2 * S**4 * sigma_s**4 * tau**2


def price_space_sigma(alpha, tau, sigma_s, S, deltas, gammas, vegas, hedged: bool = True):
    """sigma_opt per instrument (price units per sqrt(time)) from the hedged residual variance RATE
    (variance over tau divided by tau); adds the delta term when `hedged` is False."""
    vt, gt = residual_variance_terms(alpha, tau, sigma_s, S, gammas, vegas)
    var_rate = (vt + gt) / tau
    if not hedged:
        var_rate = var_rate + (np.asarray(deltas, dtype=float) * sigma_s * S) ** 2
    return np.sqrt(var_rate)


def vol_space_skew(gamma, alpha, tau, sigma_s, S, inventory, gammas, vegas, skew_scale: float = 1.0):
    """skew_vol(j) per instrument (vol units) for the inventory vector; see module docstring."""
    inventory = np.asarray(inventory, dtype=float)
    vegas = np.asarray(vegas, dtype=float)
    gammas = np.asarray(gammas, dtype=float)
    v_net = float(np.sum(inventory * vegas))
    g_net = float(np.sum(inventory * gammas))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(vegas > 0, gammas / vegas, 0.0)
    return skew_scale * gamma * (alpha**2 * tau * v_net + 0.5 * sigma_s**4 * S**4 * tau**2 * g_net * ratio)


def round_to_tick(bid, ask, tick: float):
    """Bids down, asks up, on the tick grid; a bid <= 0 after rounding is pulled (NaN). tick <= 0: no rounding."""
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)
    if tick > 0:
        eps = 1e-9 * tick
        bid = np.floor(bid / tick + eps) * tick
        ask = np.ceil(ask / tick - eps) * tick
    bid = np.where(bid <= 0, np.nan, bid)
    return bid, ask


class Quoter(Protocol):
    def quotes(self, snap: FairSnapshot, inventory: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]: ...


_KINDS = ("as", "glft", "glft_asym")


@dataclass(frozen=True)
class PriceSpace:
    """Per-instrument scalar rule on the option price. kind in {'as', 'glft', 'glft_asym'}; gamma per $;
    k per $; A per second; T horizon in seconds (AS and GLFT exact); alpha_s, sigma_s per sqrt(second);
    tau_s seconds; tick in $. For kind='glft' one GLFT2013 per instrument is built at the first call on that
    snapshot's sigma_opt and re-built whenever sigma_opt has moved by more than SIGMA_REFIT_TOL (1 %) from the
    fitted value (a few ms per build), so the exact quotes track the surface to that tolerance -- across a
    run and across different strips of the same instruments alike; Q is its inventory bound. A snapshot
    with different strikes or rights raises ValueError (the instrument index would silently alias)."""

    SIGMA_REFIT_TOL = 0.01

    kind: str
    gamma: float
    k: float
    A: float
    T: float
    alpha_s: float
    sigma_s: float
    tau_s: float
    hedged: bool = True
    tick: float = 0.0
    Q: int = 100

    def __post_init__(self):
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}")
        _check_positive(gamma=self.gamma, k=self.k, A=self.A, T=self.T, tau_s=self.tau_s)
        object.__setattr__(self, "_exact", {})

    def sigma_opt(self, snap: FairSnapshot) -> np.ndarray:
        return price_space_sigma(self.alpha_s, self.tau_s, self.sigma_s, snap.spot, snap.deltas, snap.gammas, snap.vegas, self.hedged)

    def offsets(self, snap: FairSnapshot, inventory, t: float):
        q = np.asarray(inventory, dtype=float)
        sig = self.sigma_opt(snap)
        if self.kind == "as":
            return as_offsets(q, t, self.gamma, self.k, sig, self.T)
        if self.kind == "glft_asym":
            return glft_asymptotic_offsets(q, self.gamma, self.k, sig, self.A)
        bid = np.empty(q.shape[0])
        ask = np.empty(q.shape[0])
        key = (tuple(np.asarray(snap.K, dtype=float).tolist()), tuple(np.asarray(snap.right).tolist()))
        if "strip" not in self._exact:
            self._exact["strip"] = key
        elif self._exact["strip"] != key:
            raise ValueError("PriceSpace(kind='glft') is bound to the strip it first quoted (sigma_opt frozen there); build a new object")
        for j in range(q.shape[0]):
            fitted = self._exact.get(j)
            if fitted is None or abs(float(sig[j]) - fitted.sigma) > self.SIGMA_REFIT_TOL * fitted.sigma:
                fitted = self._exact[j] = GLFT2013(self.gamma, self.k, float(sig[j]), self.A, self.T, self.Q)
            bid[j], ask[j] = fitted.offsets(int(round(q[j])), t)
        return bid, ask

    def quotes(self, snap: FairSnapshot, inventory, t: float):
        b_off, a_off = self.offsets(snap, inventory, t)
        return round_to_tick(snap.prices - b_off, snap.prices + a_off, self.tick)


@dataclass(frozen=True)
class VolSpace:
    """Parallel vol-surface skew from net vega / net gamma, half-spread hs_vol in vol units (scalar or per
    instrument). gamma per $; alpha_s, sigma_s per sqrt(second); tau_s seconds; tick in $."""

    gamma: float
    alpha_s: float
    sigma_s: float
    tau_s: float
    hs_vol: float | np.ndarray = 0.01
    skew_scale: float = 1.0
    tick: float = 0.0

    def __post_init__(self):
        _check_positive(gamma=self.gamma, tau_s=self.tau_s)
        if np.any(np.asarray(self.hs_vol) < 0) or self.skew_scale < 0:
            raise ValueError("hs_vol >= 0 and skew_scale >= 0")

    def skew(self, snap: FairSnapshot, inventory) -> np.ndarray:
        return vol_space_skew(self.gamma, self.alpha_s, self.tau_s, self.sigma_s, snap.spot, inventory, snap.gammas, snap.vegas, self.skew_scale)

    def quote_vols(self, snap: FairSnapshot, inventory):
        mid = snap.vols - self.skew(snap, inventory)
        return np.maximum(mid - self.hs_vol, VOL_FLOOR), np.maximum(mid + self.hs_vol, VOL_FLOOR)

    def quotes(self, snap: FairSnapshot, inventory, t: float = 0.0):
        bv, av = self.quote_vols(snap, inventory)
        bid = black_price(snap.spot, snap.K, snap.T_rem, bv, snap.right)
        ask = black_price(snap.spot, snap.K, snap.T_rem, av, snap.right)
        return round_to_tick(bid, ask, self.tick)

    def dollar_skew_per_lot(self, snap: FairSnapshot) -> np.ndarray:
        """Linearised $ skew per contract at q = 0: Vega_j * d skew_vol(j) / d q_j."""
        vt, gt = residual_variance_terms(self.alpha_s, self.tau_s, self.sigma_s, snap.spot, snap.gammas, snap.vegas)
        return self.skew_scale * self.gamma * (vt + gt)


@dataclass(frozen=True)
class MatchReport:
    hs_vol: float
    skew_scale: float
    hs_usd_sum: float
    vega_sum: float
    price_skew_sum: float
    vol_skew_sum: float


def match_spreads(price_q: PriceSpace, vol_q: VolSpace, snap: FairSnapshot, t: float = 0.0) -> tuple[VolSpace, MatchReport]:
    """New VolSpace (and the report) with hs_vol and skew_scale matched to price_q at q = 0 on `snap`:
    sum Vega hs_vol = sum hs_$ and sum_j (vol-space $ skew per lot) = sum_j (price-space $ skew per lot)."""
    n = snap.n_instruments
    zero = np.zeros(n)
    b0, a0 = price_q.offsets(snap, zero, t)
    hs_usd = 0.5 * (b0 + a0)
    b1, _ = price_q.offsets(snap, np.ones(n), t)
    price_skew = b1 - b0
    vega_sum = float(np.sum(snap.vegas))
    hs_vol = float(np.sum(hs_usd) / vega_sum)
    unit = replace(vol_q, skew_scale=1.0)
    vol_skew_unit = float(np.sum(unit.dollar_skew_per_lot(snap)))
    scale = float(np.sum(price_skew) / vol_skew_unit) if vol_skew_unit > 0 else 1.0
    matched = replace(vol_q, hs_vol=hs_vol, skew_scale=scale)
    rep = MatchReport(hs_vol, scale, float(np.sum(hs_usd)), vega_sum, float(np.sum(price_skew)), float(np.sum(matched.dollar_skew_per_lot(snap))))
    return matched, rep


# ----------------------------------------------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MaxLossGuard:
    """Pull every quote once running P&L (any basis the caller chooses, $) is below -max_loss."""

    max_loss: float

    def __post_init__(self):
        if self.max_loss < 0:
            raise ValueError("max_loss >= 0")

    def apply(self, bid, ask, running_pnl: float):
        bid = np.asarray(bid, dtype=float)
        ask = np.asarray(ask, dtype=float)
        if running_pnl < -self.max_loss:
            return np.full_like(bid, np.nan), np.full_like(ask, np.nan)
        return bid.copy(), ask.copy()


def vega_cap(bid, ask, inventory, vegas, cap: float):
    """Pull bids (no more buying) when net vega >= +cap, asks when net vega <= -cap. Returns new arrays."""
    if cap <= 0:
        raise ValueError("cap > 0")
    bid = np.asarray(bid, dtype=float).copy()
    ask = np.asarray(ask, dtype=float).copy()
    v_net = float(np.sum(np.asarray(inventory, dtype=float) * np.asarray(vegas, dtype=float)))
    if v_net >= cap:
        bid[:] = np.nan
    if v_net <= -cap:
        ask[:] = np.nan
    return bid, ask
