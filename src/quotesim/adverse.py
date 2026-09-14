"""Post-trade review: fill-level markouts by counterparty class and horizon, and a toxicity summary.

Units and conventions ($ per contract = per unit of underlying times the multiplier; vol points = 0.01 of
vol; horizons in SECONDS on the simulator clock; MM-side sign s_f = +1 when the maker bought):
- markout at horizon h of fill f:  MO_h(f) = s_f (F_{t_f + h} - px_f)   (positive = good for the maker)
                                            = realised spread s_f (F_{t_f} - px_f) + adverse s_f (F_{t_f + h} - F_{t_f})
  against FAIR (the surface mid), never against the maker's own quotes; t_f + h is clipped at the end of
  the run, exactly as pnl.py's ADVERSE_h, so `adverse_mean(all, h_markout) * n == attribution.adverse_h`
  (tested).
- vol points: each $ quantity divided by the instrument's fair vega at the fill (per 1.00 vol) and times
  100; a fill whose vega is below VEGA_FLOOR is left out of the vol-point means (its count is `n_vp`).
- se_clustered: the standard error of the qty-weighted mean markout with fills clustered by time bucket
  of width h (bucket = floor(t_f / h)): fills inside one horizon share the fair path and are not
  independent; se^2 = G / (G - 1) sum_g (sum_{f in g} w_f (x_f - mean))^2 / (sum_f w_f)^2 with G the number
  of non-empty buckets (NaN when G < 2).
- classes: "informed" / "uninformed" as tagged by the synthetic flow (exact in a simulation; a replay would
  say "unknown"), plus "all". These horizons are SYNTHETIC: the fair moves by the seeded process, not by a
  recorded tape.
- `toxicity(run)` per class: fills, share of fills, realised spread, adverse at the run's canonical
  horizon, markout, `give_back` = -adverse / realised spread (the fraction of the captured spread the
  class takes back by the horizon), and the fraction of fills with a negative markout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VEGA_FLOOR = 1e-6
MARKOUT_COLUMNS = ("horizon_s", "class", "n", "realised_spread_mean", "adverse_mean", "markout_mean", "se_clustered",
                   "n_vp", "realised_spread_vp", "adverse_vp", "markout_vp", "se_clustered_vp")


def horizon_steps(h: float, dt: float) -> int:
    """h / dt as an integer; a horizon that is not a positive whole number of steps raises (a shorter or
    fractional h would be silently rounded to whole steps while the row and the cluster width still said h)."""
    steps = h / dt
    if h <= 0 or abs(steps - round(steps)) > 1e-9 * max(1.0, steps) or round(steps) < 1:
        raise ValueError(f"horizon h = {h} s must be a positive whole number of steps of dt = {dt} s")
    return int(round(steps))


def _per_fill(run, h: float):
    """Per-fill arrays at horizon h: weight (qty), realised spread, adverse, markout ($), vega at fill, bucket."""
    f = run.fills
    if len(f) == 0:
        raise ValueError("no fills in the run")
    step = f["step"].to_numpy(dtype=np.int64)
    inst = f["instrument"].to_numpy(dtype=np.int64)
    sign = f["sign"].to_numpy(dtype=float)
    qty = f["qty"].to_numpy(dtype=float)
    px = f["px"].to_numpy(dtype=float)
    mult = run.multiplier[inst]
    F = run.path.prices
    n = run.path.n_steps
    h_steps = horizon_steps(h, run.config.dt)
    ahead = np.minimum(step + h_steps, n)
    realised = sign * (F[step, inst] - px) * mult
    adverse = sign * (F[ahead, inst] - F[step, inst]) * mult
    vega = run.path.vegas[step, inst] * mult
    bucket = np.floor(f["t"].to_numpy(dtype=float) / h).astype(np.int64)
    return qty, realised, adverse, realised + adverse, vega, bucket


def clustered_se(x, w, cluster) -> float:
    """Cluster-robust SE of the w-weighted mean of x (see module docstring); NaN with < 2 clusters."""
    x, w, cluster = np.asarray(x, float), np.asarray(w, float), np.asarray(cluster)
    if x.size == 0 or w.sum() <= 0:
        return float("nan")
    mean = np.sum(w * x) / np.sum(w)
    ids, inv = np.unique(cluster, return_inverse=True)
    g = ids.size
    if g < 2:
        return float("nan")
    score = np.bincount(inv, weights=w * (x - mean), minlength=g)
    return float(np.sqrt(g / (g - 1) * np.sum(score**2)) / np.sum(w))


def _class_masks(run) -> dict[str, np.ndarray]:
    cls = run.fills["class"].to_numpy()
    return {"all": np.ones(cls.size, dtype=bool), "uninformed": cls == "uninformed", "informed": cls == "informed"}


def _wmean(x, w) -> float:
    return float(np.sum(w * x) / np.sum(w))


def _row(h, name, mask, qty, realised, adverse, markout, vega, bucket) -> dict:
    w = qty[mask]
    if w.sum() == 0:
        return {"horizon_s": h, "class": name, "n": 0, "n_vp": 0, **{c: float("nan") for c in MARKOUT_COLUMNS[3:] if c != "n_vp"}}
    ok = mask & (vega > VEGA_FLOOR)
    w_vp = qty[ok]
    row = {
        "horizon_s": h, "class": name, "n": int(w.sum()),
        "realised_spread_mean": _wmean(realised[mask], w), "adverse_mean": _wmean(adverse[mask], w),
        "markout_mean": _wmean(markout[mask], w), "se_clustered": clustered_se(markout[mask], w, bucket[mask]),
        "n_vp": int(w_vp.sum()), "realised_spread_vp": float("nan"), "adverse_vp": float("nan"),
        "markout_vp": float("nan"), "se_clustered_vp": float("nan"),
    }
    if w_vp.sum() > 0:
        to_vp = 100.0 / vega[ok]
        row.update(
            realised_spread_vp=_wmean(realised[ok] * to_vp, w_vp), adverse_vp=_wmean(adverse[ok] * to_vp, w_vp),
            markout_vp=_wmean(markout[ok] * to_vp, w_vp), se_clustered_vp=clustered_se(markout[ok] * to_vp, w_vp, bucket[ok]),
        )
    return row


def markouts(run, horizons=(1.0, 10.0, 60.0, 300.0)) -> pd.DataFrame:
    """One row per (horizon, class) with MARKOUT_COLUMNS; $ per contract and vol points."""
    rows = []
    masks = _class_masks(run)
    for h in horizons:
        if h <= 0:
            raise ValueError("horizons must be > 0 seconds")
        qty, realised, adverse, markout, vega, bucket = _per_fill(run, float(h))
        for name, mask in masks.items():
            rows.append(_row(float(h), name, mask, qty, realised, adverse, markout, vega, bucket))
    return pd.DataFrame(rows, columns=list(MARKOUT_COLUMNS))


def toxicity(run, h: float | None = None) -> pd.DataFrame:
    """Per-class summary at the run's canonical horizon (or h): fills, share, realised spread, adverse,
    markout, give_back = -adverse / realised spread, frac_negative = share of fills with markout < 0."""
    h = float(run.config.h_markout if h is None else h)
    qty, realised, adverse, markout, _, _ = _per_fill(run, h)
    total = qty.sum()
    rows = []
    for name, mask in _class_masks(run).items():
        w = qty[mask]
        if w.sum() == 0:
            rows.append({"class": name, "n": 0, "share": 0.0, "realised_spread_mean": float("nan"), "adverse_mean": float("nan"),
                         "markout_mean": float("nan"), "give_back": float("nan"), "frac_negative": float("nan")})
            continue
        rs = float(np.sum(w * realised[mask]) / w.sum())
        ad = float(np.sum(w * adverse[mask]) / w.sum())
        rows.append({
            "class": name, "n": int(w.sum()), "share": float(w.sum() / total), "realised_spread_mean": rs,
            "adverse_mean": ad, "markout_mean": rs + ad, "give_back": (-ad / rs) if rs > 0 else float("nan"),
            "frac_negative": float(np.sum(w * (markout[mask] < 0)) / w.sum()),
        })
    return pd.DataFrame(rows).set_index("class")
