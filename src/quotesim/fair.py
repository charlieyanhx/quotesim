"""Fair-value surface: the mid the quoter quotes around and the mark every position is valued at.

Units and conventions (this module works in YEARS):
- time: t and every T are in calendar years; `YEAR_SECONDS = 365 * 24 * 3600` converts the simulator's
  1-second steps (`dt_years = dt_seconds / YEAR_SECONDS`).
- prices are per unit of underlying (the `multiplier` column of `instruments` is metadata only; nothing
  here multiplies by it); r = q = 0 (Black on the spot, no carry), European exercise.
- vol: sigma(k) = sigma_ATM + skew_s * k + curv_c * k^2 with k = ln(K / S) (log-moneyness against SPOT),
  the same for calls and puts, floored at VOL_FLOOR; no term structure (vol(k, T) ignores T).
- Greeks are Black partials at FIXED vol (sticky-strike): delta = dF/dS|sigma, gamma = d2F/dS2|sigma,
  vega per 1.00 change in vol (i.e. per 100 vol points), theta per YEAR (negative for a long option,
  dF/dt = -dF/dT). The surface move induced by dk = -dS/S is absorbed by the vega * dsigma term of the
  Greek explanation layer in pnl.py, not by delta.
- state dynamics per `step(z_spot, z_vol, jump_u, dt)`:
    log S_{t+dt} = log S_t - 1/2 spot_vol^2 dt - ln(1 + p_J kappa_J) + spot_vol sqrt(dt) z_spot + J
    J = mu_J + sd_J * Phi^{-1}(jump_u / p_J) if jump_u < p_J = 1 - exp(-lam dt) else 0   (one uniform gives
        both occurrence and size, at most one jump per step; kappa_J = exp(mu_J + sd_J^2 / 2) - 1 and the
        ln(1 + p_J kappa_J) term is the exact compensator, so E[S_{t+dt}] = S_t for any dt)
    sigma_ATM,t+dt = max(VOL_FLOOR, sigma_ATM,t + kappa_vol (sigma0 - sigma_ATM,t) dt + alpha sqrt(dt) z_vol)
        (arithmetic OU; alpha is an ABSOLUTE vol-of-vol per sqrt(year), the same alpha the vol-space quoter uses)
  spot_vol per sqrt(year); kappa_vol per year; jump = (lam per year, mu_J, sd_J) in log-return units.
- past-measurability: `step` reads only its arguments and the current state; `FairPath` stores a whole
  pre-computed path (the informed counterparty and the markouts may read it), and `PastOnlyFair` wraps a
  path behind a clock and raises `FutureAccessError` on any read ahead of it; every snapshot is a read-only
  COPY of one row (no view of the path's arrays reaches the quoter, so `.base` cannot expose the future and
  an in-place write cannot corrupt the fair), so a quoter fed through it cannot see the future in any run,
  not only in tests.

Invariants kept (tested): every Greek matches a central finite difference of `black_price`; `path` equals
the sequence of `step` snapshots; call - put == S - K (parity at r = q = 0) to 1e-10.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy.stats import norm

YEAR_SECONDS = 365.0 * 24.0 * 3600.0
VOL_FLOOR = 1e-4
_T_FLOOR = 1e-12


class FutureAccessError(RuntimeError):
    """Raised by `PastOnlyFair` when a caller asks for a state ahead of its clock."""


# ----------------------------------------------------------------------------------------------------------
# Black (r = q = 0), vectorised over broadcastable arrays
# ----------------------------------------------------------------------------------------------------------


def _d1(S, K, T, sigma):
    v = sigma * np.sqrt(T)
    return (np.log(S / K) + 0.5 * sigma**2 * T) / v


def black_price(S, K, T, sigma, right):
    """European price at r = q = 0. `right` is 'C'/'P' (scalar or array). T <= 0 -> intrinsic."""
    S, K, T, sigma = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (S, K, T, sigma)))
    is_call = np.asarray(right) == "C"
    Tp = np.maximum(T, _T_FLOOR)
    d1 = _d1(S, K, Tp, sigma)
    d2 = d1 - sigma * np.sqrt(Tp)
    call = S * norm.cdf(d1) - K * norm.cdf(d2)
    put = call - S + K
    live = T > 0
    out = np.where(is_call, call, put)
    intrinsic = np.where(is_call, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    out = np.where(live, out, intrinsic)
    return float(out) if out.ndim == 0 else out


def black_greeks(S, K, T, sigma, right):
    """(delta, gamma, vega, theta) at fixed sigma; vega per 1.00 vol, theta per year. Expired -> zeros
    except delta = indicator of being in the money."""
    S, K, T, sigma = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (S, K, T, sigma)))
    is_call = np.asarray(right) == "C"
    Tp = np.maximum(T, _T_FLOOR)
    sq = np.sqrt(Tp)
    d1 = _d1(S, K, Tp, sigma)
    pdf = norm.pdf(d1)
    delta = np.where(is_call, norm.cdf(d1), norm.cdf(d1) - 1.0)
    gamma = pdf / (S * sigma * sq)
    vega = S * pdf * sq
    theta = -S * pdf * sigma / (2.0 * sq)
    live = T > 0
    delta_exp = np.where(is_call, (S > K).astype(float), -(S < K).astype(float))
    delta = np.where(live, delta, delta_exp)
    zero = np.zeros_like(gamma)
    gamma, vega, theta = (np.where(live, g, zero) for g in (gamma, vega, theta))
    if delta.ndim == 0:
        return float(delta), float(gamma), float(vega), float(theta)
    return delta, gamma, vega, theta


# ----------------------------------------------------------------------------------------------------------
# Protocol and snapshot
# ----------------------------------------------------------------------------------------------------------


@runtime_checkable
class FairSurface(Protocol):
    """What a quoter may read: the current fair state. T is REMAINING time to expiry in years."""

    @property
    def spot(self) -> float: ...

    def price(self, K: float, T: float, right: str) -> float: ...

    def vol(self, k: float, T: float) -> float: ...

    def delta(self, K: float, T: float, right: str) -> float: ...

    def gamma(self, K: float, T: float, right: str) -> float: ...

    def vega(self, K: float, T: float, right: str) -> float: ...

    def theta(self, K: float, T: float, right: str) -> float: ...

    def state(self) -> dict: ...


@dataclass(frozen=True)
class FairSnapshot:
    """The whole strip at one instant. Arrays are aligned with `instruments` (one row per instrument).
    t in years since start; T_rem in years; theta per year; vega per 1.00 vol."""

    t: float
    spot: float
    sigma_atm: float
    K: np.ndarray
    T_rem: np.ndarray
    right: np.ndarray
    prices: np.ndarray
    vols: np.ndarray
    deltas: np.ndarray
    gammas: np.ndarray
    vegas: np.ndarray
    thetas: np.ndarray

    @property
    def n_instruments(self) -> int:
        return int(self.K.shape[0])


# ----------------------------------------------------------------------------------------------------------
# Synthetic fair
# ----------------------------------------------------------------------------------------------------------


def _instrument_table(strikes, expiries, multiplier) -> pd.DataFrame:
    rows = [
        (float(K), float(T), right, float(multiplier))
        for T in expiries
        for K in strikes
        for right in ("C", "P")
    ]
    return pd.DataFrame(rows, columns=["K", "T_years", "right", "multiplier"])


@dataclass(eq=False)
class SyntheticFair:
    """Black with a quadratic smile in log-moneyness, spot GBM (+ Merton jumps), ATM vol arithmetic OU.
    Mutable: `step` advances the state in place; `copy()` gives an independent instance. See module docstring
    for every unit. `instruments` is a DataFrame (K, T_years, right, multiplier); T_years is the expiry
    measured from t = 0, so remaining time is T_years - t."""

    S0: float
    sigma0: float
    skew_s: float = 0.0
    curv_c: float = 0.0
    alpha: float = 0.0
    kappa_vol: float = 0.0
    spot_vol: float = 0.2
    jump: tuple[float, float, float] | None = None
    strikes: tuple[float, ...] = (100.0,)
    expiries: tuple[float, ...] = (30.0 / 365.0,)
    multiplier: float = 1.0
    instruments: pd.DataFrame = field(init=False, repr=False)
    t: float = field(init=False, default=0.0)
    S: float = field(init=False)
    sigma_atm: float = field(init=False)

    def __post_init__(self):
        if self.S0 <= 0 or self.sigma0 <= 0 or self.spot_vol < 0 or self.alpha < 0 or self.kappa_vol < 0:
            raise ValueError("S0, sigma0 > 0; spot_vol, alpha, kappa_vol >= 0")
        if self.jump is not None and (len(self.jump) != 3 or self.jump[0] < 0 or self.jump[2] < 0):
            raise ValueError("jump = (lam >= 0, mu, sd >= 0)")
        if len(self.strikes) == 0 or len(self.expiries) == 0 or min(self.expiries) <= 0:
            raise ValueError("need >= 1 strike and >= 1 expiry with T > 0")
        self.instruments = _instrument_table(self.strikes, self.expiries, self.multiplier)
        self._K = self.instruments["K"].to_numpy()
        self._T0 = self.instruments["T_years"].to_numpy()
        self._right = self.instruments["right"].to_numpy()
        self.S = float(self.S0)
        self.sigma_atm = float(self.sigma0)

    # -- protocol ------------------------------------------------------------------------------------------
    @property
    def spot(self) -> float:
        return self.S

    def state(self) -> dict:
        return {"t": self.t, "S": self.S, "sigma_atm": self.sigma_atm}

    def vol(self, k, T=None):
        """Fair vol at log-moneyness k (T ignored: no term structure)."""
        v = self.sigma_atm + self.skew_s * np.asarray(k, dtype=float) + self.curv_c * np.asarray(k, dtype=float) ** 2
        v = np.maximum(v, VOL_FLOOR)
        return float(v) if v.ndim == 0 else v

    def vol_at_strike(self, K):
        return self.vol(np.log(np.asarray(K, dtype=float) / self.S))

    def price(self, K, T, right):
        return black_price(self.S, K, T, self.vol_at_strike(K), right)

    def delta(self, K, T, right):
        return black_greeks(self.S, K, T, self.vol_at_strike(K), right)[0]

    def gamma(self, K, T, right):
        return black_greeks(self.S, K, T, self.vol_at_strike(K), right)[1]

    def vega(self, K, T, right):
        return black_greeks(self.S, K, T, self.vol_at_strike(K), right)[2]

    def theta(self, K, T, right):
        return black_greeks(self.S, K, T, self.vol_at_strike(K), right)[3]

    # -- strip accessors -----------------------------------------------------------------------------------
    @property
    def n_instruments(self) -> int:
        return int(self._K.shape[0])

    def T_rem(self) -> np.ndarray:
        return self._T0 - self.t

    def fair_vols(self) -> np.ndarray:
        return self.vol_at_strike(self._K)

    def fair_prices(self) -> np.ndarray:
        return black_price(self.S, self._K, self.T_rem(), self.fair_vols(), self._right)

    def greeks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return black_greeks(self.S, self._K, self.T_rem(), self.fair_vols(), self._right)

    def snapshot(self) -> FairSnapshot:
        d, g, v, th = self.greeks()
        return FairSnapshot(
            t=self.t, spot=self.S, sigma_atm=self.sigma_atm, K=self._K.copy(), T_rem=self.T_rem(),
            right=self._right.copy(), prices=self.fair_prices(), vols=self.fair_vols(),
            deltas=d, gammas=g, vegas=v, thetas=th,
        )

    # -- dynamics ------------------------------------------------------------------------------------------
    def _jump_log_return(self, jump_u: float, dt: float) -> tuple[float, float]:
        """(J, compensator) for one step: J from the single uniform; compensator ln(1 + p_J kappa_J) is the
        exact log-drift that makes E[S_{t+dt}] = S_t with at most one jump per step."""
        if self.jump is None:
            return 0.0, 0.0
        lam, mu, sd = self.jump
        p_jump = -np.expm1(-lam * dt)
        comp = float(np.log1p(p_jump * (np.exp(mu + 0.5 * sd**2) - 1.0)))
        if p_jump <= 0.0 or jump_u >= p_jump:
            return 0.0, comp
        return float(mu + sd * norm.ppf(jump_u / p_jump)), comp

    def step(self, z_spot: float, z_vol: float, jump_u: float, dt: float) -> None:
        """Advance the state by dt YEARS using the pre-drawn normals z_spot, z_vol and uniform jump_u."""
        if dt <= 0:
            raise ValueError("dt must be > 0 (years)")
        J, comp = self._jump_log_return(float(jump_u), dt)
        sq = np.sqrt(dt)
        self.S = float(self.S * np.exp(-0.5 * self.spot_vol**2 * dt - comp + self.spot_vol * sq * z_spot + J))
        sig = self.sigma_atm + self.kappa_vol * (self.sigma0 - self.sigma_atm) * dt + self.alpha * sq * z_vol
        self.sigma_atm = float(max(VOL_FLOOR, sig))
        self.t = float(self.t + dt)

    def copy(self) -> SyntheticFair:
        c = SyntheticFair(
            self.S0, self.sigma0, self.skew_s, self.curv_c, self.alpha, self.kappa_vol, self.spot_vol,
            self.jump, self.strikes, self.expiries, self.multiplier,
        )
        c.S, c.sigma_atm, c.t = self.S, self.sigma_atm, self.t
        return c

    def path(self, spot_z, vol_z, jump_u, dt: float) -> FairPath:
        """Pre-compute n_steps + 1 states (index 0 = the current state) from streams of length n_steps,
        on a COPY (self is left untouched). dt in years. The state recursion runs step by step; the strip
        is then priced on the whole (n_steps + 1, n_instruments) grid with the same Black functions, so the
        result equals the sequence of `step` snapshots."""
        spot_z, vol_z, jump_u = (np.asarray(x, dtype=float) for x in (spot_z, vol_z, jump_u))
        n = spot_z.shape[0]
        if vol_z.shape[0] != n or jump_u.shape[0] != n:
            raise ValueError("spot_z, vol_z, jump_u must have the same length")
        f = self.copy()
        t, spot, sig = np.empty(n + 1), np.empty(n + 1), np.empty(n + 1)
        t[0], spot[0], sig[0] = f.t, f.S, f.sigma_atm
        for i in range(n):
            f.step(spot_z[i], vol_z[i], jump_u[i], dt)
            t[i + 1], spot[i + 1], sig[i + 1] = f.t, f.S, f.sigma_atm
        K, right = self._K[None, :], self._right[None, :]
        S = spot[:, None]
        T_rem = self._T0[None, :] - t[:, None]
        k = np.log(K / S)
        vols = np.maximum(sig[:, None] + self.skew_s * k + self.curv_c * k**2, VOL_FLOOR)
        prices = black_price(S, K, T_rem, vols, right)
        d, g, v, th = black_greeks(S, K, T_rem, vols, right)
        return FairPath(
            instruments=self.instruments.copy(), dt=dt, t=t, spot=spot, sigma_atm=sig, K=self._K.copy(),
            right=self._right.copy(), T_rem=T_rem, prices=prices, vols=vols, deltas=d, gammas=g, vegas=v, thetas=th,
        )


# ----------------------------------------------------------------------------------------------------------
# Pre-computed path and the past-only view
# ----------------------------------------------------------------------------------------------------------


def _frozen(a: np.ndarray) -> np.ndarray:
    out = np.array(a, copy=True)
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class FairPath:
    """n_steps + 1 states of the strip; row i is the state after i steps. 2-D arrays are (n_steps + 1,
    n_instruments), all read-only. The counterparty and the markouts may read any row; the quoter must go
    through `PastOnlyFair`, whose snapshots are copies (see `snapshot`)."""

    instruments: pd.DataFrame
    dt: float
    t: np.ndarray
    spot: np.ndarray
    sigma_atm: np.ndarray
    K: np.ndarray
    right: np.ndarray
    T_rem: np.ndarray
    prices: np.ndarray
    vols: np.ndarray
    deltas: np.ndarray
    gammas: np.ndarray
    vegas: np.ndarray
    thetas: np.ndarray

    @property
    def n_steps(self) -> int:
        return int(self.t.shape[0]) - 1

    @property
    def n_instruments(self) -> int:
        return int(self.K.shape[0])

    def __post_init__(self):
        for name in ("t", "spot", "sigma_atm", "K", "right", "T_rem", "prices", "vols", "deltas", "gammas", "vegas", "thetas"):
            getattr(self, name).flags.writeable = False

    def snapshot(self, i: int) -> FairSnapshot:
        """Row i as read-only COPIES: the snapshot shares no memory with the path (no numpy view whose
        `.base` is the whole pre-drawn future) and an in-place write on it raises rather than corrupting
        the fair every later step is marked against."""
        return FairSnapshot(
            t=float(self.t[i]), spot=float(self.spot[i]), sigma_atm=float(self.sigma_atm[i]), K=_frozen(self.K),
            T_rem=_frozen(self.T_rem[i]), right=_frozen(self.right), prices=_frozen(self.prices[i]), vols=_frozen(self.vols[i]),
            deltas=_frozen(self.deltas[i]), gammas=_frozen(self.gammas[i]), vegas=_frozen(self.vegas[i]), thetas=_frozen(self.thetas[i]),
        )


class PastOnlyFair:
    """A `FairPath` behind a clock. `snapshot(i)` with i > clock raises `FutureAccessError`; `current()` is
    the state at the clock; `advance()` moves the clock one step. Feed quoters through this in every run."""

    def __init__(self, path: FairPath, clock: int = 0):
        if not 0 <= clock <= path.n_steps:
            raise ValueError("clock outside the path")
        self._path = path
        self._clock = int(clock)

    @property
    def clock(self) -> int:
        return self._clock

    @property
    def path_length(self) -> int:
        return self._path.n_steps

    def advance(self, steps: int = 1) -> None:
        if self._clock + steps > self._path.n_steps:
            raise ValueError("cannot advance past the end of the path")
        self._clock += int(steps)

    def snapshot(self, i: int | None = None) -> FairSnapshot:
        i = self._clock if i is None else int(i)
        if i > self._clock:
            raise FutureAccessError(f"asked for step {i} but the clock is at {self._clock}")
        if i < 0:
            raise IndexError("negative step")
        return self._path.snapshot(i)

    def current(self) -> FairSnapshot:
        return self.snapshot(self._clock)

    @property
    def spot(self) -> float:
        return float(self._path.spot[self._clock])
