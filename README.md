# quotesim

[![ci](https://github.com/charlieyanhx/quotesim/actions/workflows/ci.yml/badge.svg)](https://github.com/charlieyanhx/quotesim/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

An options quoting simulator with synthetic flow. It holds Avellaneda-Stoikov (2008) and Guéant-Lehalle-Fernandez-Tapia (2013) quoters behind one interface, in price space and in vol space (a derivation of the Stoikov-Sağlam 2009 tilt divided by vega), a delta-band hedger with a sqrt-law cost model, and a P&L attribution of every run. What is tested: the published closed forms (A-S spreads, GLFT exact-vs-asymptotic offsets, Stoikov-Sağlam Theorem 4, Whalley-Wilmott bands) to their cited digits; the flow model's acceptance law and informed-markout expectation against the analytic value; every Greek against finite differences; a fair object that raises on any future access and hands the quoter read-only copies; and the attribution identity on every run. The one design rule: **the top identity has no residual, and comparisons are paired by seed** — `spread + inventory + hedge - hedge cost = realised` to the rounding of the summed products on every run (the bar is max(1e-9, 1e-12 × the run's gross $); the gaps in the table below are under 1e-11; `sim.run` raises otherwise; only the Greek explanation layer carries a residual), and two quoters are only ever compared on the same pre-drawn streams (common random numbers) with paired differences, never the best seed and never an unpaired t-test.

## Run it

```bash
python -m pip install -e ".[dev]"
pytest -q            # 114 tests, ~25 s on the machine below
quotesim report      # regenerates every table below from fixed seeds (~90 s); --check compares instead (what CI runs)
quotesim run --seed 1   # one run: the attribution waterfall and the toxicity table
quotesim run --seed 1 --space vol --informed 0.3 --band 25
```

Every number in this README is produced by `quotesim report` (src/quotesim/report.py) from fixed seeds and written between `<!-- quotesim:begin:… -->` / `<!-- quotesim:end:… -->` markers; CI re-runs it and fails on any diff. The numbers here were produced on an Apple M1 laptop (macOS 26, Python 3.12, numpy 2.5, scipy 1.18, pandas 3.0).

## What it simulates

A fixed-step loop (1 s default) over a synthetic strip: quote update -> fills -> hedge check -> state step. Say it plainly: the flow is **synthetic** (a Poisson intensity `A e^{-k delta}` per side and instrument, thinned against the quotes), there is **no queue**, **no latency** and **no matching engine**, and the **markout horizons are synthetic** (the fair moves by a seeded process, not by a recorded tape). The simulator answers "how do two quoting rules compare under identical flow", never "what would a desk earn".

| module | what it does | units |
|---|---|---|
| `fair.py` | `SyntheticFair`: Black prices and analytic Greeks on `sigma(k) = sigma_ATM + s k + c k^2`, spot GBM with optional Merton jumps, ATM vol an arithmetic OU (`kappa_vol = 0` in every table here, i.e. a floored random walk over 600-3600 s); `FairPath` (the whole pre-drawn path, read-only) and `PastOnlyFair` (a clock; reading ahead raises `FutureAccessError`; every snapshot a read-only copy of one row) | years, $ per unit of underlying |
| `flow.py` | `FlowParams(A, k, informed_frac, p_informed, h_info, persist)`; `Streams.draw` pre-draws every stream per seed; `arrivals` thins candidates with `U < exp(-k delta)`; informed candidates read the fair `h_info` ahead on the pre-drawn path (documented look-ahead for the counterparty only); `P(fill in dt) = 1 - exp(-lambda dt)`, never `lambda dt` | seconds, $ |
| `quoter.py` | `AS2008`, `GLFT2013` (exact finite-Q by uniformization of the tridiagonal generator: relative accuracy in every entry, so the default Q = 100 is finite and monotone over the whole inventory range at any horizon), `GLFTAsymptotic`; `PriceSpace` (per-instrument rule on the option price with the hedged residual `sigma_opt`) and `VolSpace` (parallel surface skew from net vega and net gamma); `match_spreads` equates the summed $ half-spread and the summed linearised $ skew per lot at q = 0; tick rounding, `MaxLossGuard`, `vega_cap` | seconds, $, vol |
| `hedger.py` | `BandHedger`, `TimeHedger`, `NoHedger`; cost model `half_spread |dh| + Y sigma_daily sqrt(|dh| / ADV) S |dh|` (tcakit's sqrt-law form, `Y` a labelled constant); `band_ww` (Whalley-Wilmott 1997) | seconds, shares, $ |
| `pnl.py` | `attribute(run, h)`: the four-term identity, `INV = OPENING + ADVERSE_h + DRIFT_h` and `INV = THETA + INV_ex_theta` (both exact), the Greek layer with its `RESID`; every sum a `math.fsum` | $ |
| `adverse.py` | `markouts(run, horizons)` by counterparty class in $ per contract and vol points with time-bucket clustered SE; `toxicity(run)` | $, vol points, seconds |
| `sim.py` | `SimConfig`, `Run`, `run`, `paired` / `PairedResult.table()` (median, IQR, p5/p95, sign count, bootstrap CI of paired differences per identity term) | seconds |
| `report.py`, `cli.py` | `quotesim report` (the tables below), `quotesim run --seed N` | |

## Results

### Identity check

<!-- quotesim:begin:identity -->
| quoter | seeds | runs with |gap| < 1e-9 | max |gap| | max fill-split gap | max theta-split gap | run-time bar | max |RESID| / gross inventory move | fills per run (mean) |
|---|---|---|---|---|---|---|---|---:|
| PriceSpace glft_asym | 0-49 | 50/50 | < 1e-12 | < 1e-13 | < 1e-14 | < 1e-7 | < 1e-2 | 2635.8 |
| VolSpace matched | 50-99 | 50/50 | < 1e-11 | < 1e-14 | < 1e-14 | < 1e-6 | < 1e-2 | 2909.2 |

100 seeds (600 s each, 1 s steps) on the strip (S0 100, ATM vol 0.2, skew -0.1, curvature 0.3, vol-of-vol 0.6 / sqrt(yr) with kappa_vol 0 (a floored random walk), spot vol 0.2, strikes [90.0, 95.0, 100.0, 105.0, 110.0], 30-day expiry (10 instruments); flow A 0.5 /s/side/instrument, k 20 /$, informed horizon 60 s, p 1), Merton jumps on (20,000 / yr, mean -1 %, sd 1 %), informed fraction 0.1, gamma 10, band $10. Gaps are power-of-ten ceilings of the largest absolute gap over the seeds; the run-time bar is max(1e-9, 1e-12 x the run's gross $ of summed products), its ceiling in the bar column, and `sim.run` raises above it, so the 1e-9 count is a measurement below the bar. RESID is the Greek layer's residual (vanna, volga, jumps) over the gross inventory move sum |q dF|.
<!-- quotesim:end:identity -->

### Avellaneda-Stoikov 2008, Tables 1-3

<!-- quotesim:begin:as2008 -->
| gamma | strategy | avg spread | P&L mean | P&L sd | final q mean | final q sd | paper: spread | paper: P&L mean | paper: P&L sd | paper: q mean | paper: q sd |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | inventory | 1.35 | 68.6 | 8.9 | 0.16 | 5.2 | 1.35 | 68.6 | 8.7 | 0.12 | 5.1 |
| 0.01 | symmetric | 1.35 | 68.9 | 13.3 | -0.05 | 8.6 | 1.35 | 68.8 | 12.8 | 0.09 | 8.7 |
| 0.1 | inventory | 1.49 | 65.1 | 6.4 | 0.13 | 2.9 | 1.49 | 65.0 | 6.6 | 0.08 | 2.9 |
| 0.1 | symmetric | 1.49 | 68.3 | 13.1 | 0.04 | 8.4 | 1.49 | 68.4 | 12.7 | 0.26 | 8.4 |
| 1 | inventory | 3.03 | 31.2 | 4.8 | 0.08 | 1.7 | 3.02 | 31.4 | 5.0 | 0.02 | 1.7 |
| 1 | symmetric | 3.03 | 43.8 | 10.5 | -0.12 | 5.2 | 3.02 | 44.0 | 11.0 | 0.00 | 5.1 |

Their parameters (s = 100, T = 1, sigma = 2, dt = 0.005, k = 1.5, A = 140, q0 = 0), their fill rule (probability lambda(delta) dt per step and side, one unit per fill), their symmetric benchmark (the inventory strategy's time-average spread centred on the mid), arithmetic Brownian mid, P&L marked to the mid; 1000 paths per gamma, seed 0, the same mid path and fill uniforms for both strategies. The paper's RNG is not published, so only the direction of each effect and the rough magnitude are reproducible.
<!-- quotesim:end:as2008 -->

### Price space vs vol space, paired sweep (v0.1 sensitivity table)

This is a sensitivity table for v0.1, not the result of the note "inventory skew in vol space vs price space". That note needs a pre-registered primary metric, at least 200 paired seeds sized from this table's SE, and a null reported if found (docs/PLAN.md, deferred to v0.2).

<!-- quotesim:begin:sweep -->
| gamma | informed | band $ | realised: median | IQR | p5 / p95 | sign | boot 95 % CI of mean | spread: median | adverse_h: median | drift_h: median | hedge: median | hedge_cost: median | n_fills: median | vega_T: median | lots inside fair, % (price / vol) |
|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0 | 10 | -17.68 | [-19.33, -16.30] | -21.03 / -15.18 | 0/16 | [-18.71, -16.85] | -17.39 | +0.17 | -0.06 | -0.12 | +0.65 | +312.50 | -2.69 | 0.00 / 0.00 |
| 1 | 0 | 100 | -17.80 | [-19.18, -16.40] | -21.21 / -15.20 | 0/16 | [-18.77, -16.88] | -17.39 | +0.17 | -0.06 | -0.11 | +0.65 | +312.50 | -2.69 | 0.00 / 0.00 |
| 1 | 0.1 | 10 | -17.75 | [-18.69, -16.20] | -20.26 / -14.97 | 0/16 | [-18.48, -16.73] | -17.13 | -0.24 | -0.26 | +0.45 | +0.66 | +322.00 | -20.59 | 0.00 / 0.00 |
| 1 | 0.1 | 100 | -17.75 | [-18.78, -16.29] | -20.46 / -14.99 | 0/16 | [-18.52, -16.74] | -17.13 | -0.24 | -0.26 | +0.46 | +0.67 | +322.00 | -20.59 | 0.00 / 0.00 |
| 1 | 0.3 | 10 | -17.65 | [-18.16, -15.90] | -20.65 / -14.75 | 0/16 | [-18.32, -16.47] | -16.78 | -0.53 | -0.43 | +1.24 | +0.82 | +309.50 | -128.87 | 0.00 / 0.00 |
| 1 | 0.3 | 100 | -17.48 | [-18.25, -15.69] | -20.90 / -14.62 | 0/16 | [-18.37, -16.43] | -16.78 | -0.53 | -0.43 | +1.25 | +0.92 | +309.50 | -128.87 | 0.00 / 0.00 |
| 10 | 0 | 10 | -15.77 | [-16.93, -14.69] | -18.19 / -14.05 | 0/16 | [-16.63, -15.18] | -15.35 | +0.47 | +0.13 | -0.52 | +0.48 | +260.00 | -20.15 | 0.00 / 0.00 |
| 10 | 0 | 100 | -15.77 | [-16.90, -14.82] | -18.45 / -13.95 | 0/16 | [-16.69, -15.20] | -15.35 | +0.47 | +0.13 | -0.50 | +0.48 | +260.00 | -20.15 | 0.00 / 0.00 |
| 10 | 0.1 | 10 | -16.03 | [-16.47, -15.01] | -17.78 / -14.11 | 0/16 | [-16.46, -15.24] | -15.73 | +0.07 | -0.30 | +0.46 | +0.58 | +265.50 | -86.20 | 0.00 / 0.00 |
| 10 | 0.1 | 100 | -16.21 | [-16.53, -14.97] | -17.78 / -14.07 | 0/16 | [-16.54, -15.29] | -15.73 | +0.07 | -0.30 | +0.44 | +0.59 | +265.50 | -86.20 | 0.00 / 0.00 |
| 10 | 0.3 | 10 | -15.68 | [-16.25, -14.91] | -16.81 / -14.55 | 0/16 | [-16.06, -15.25] | -15.72 | -0.85 | +0.29 | +1.15 | +0.77 | +252.00 | -86.15 | 0.00 / 0.00 |
| 10 | 0.3 | 100 | -15.67 | [-16.28, -14.85] | -16.81 / -14.48 | 0/16 | [-16.06, -15.23] | -15.72 | -0.85 | +0.29 | +1.15 | +0.90 | +252.00 | -86.15 | 0.00 / 0.00 |
| 50 | 0 | 10 | -10.37 | [-11.50, -9.91] | -12.12 / -9.23 | 0/16 | [-11.03, -10.10] | -10.07 | +0.32 | -0.02 | -0.56 | +0.35 | +117.50 | -15.03 | 0.33 / 0.05 |
| 50 | 0 | 100 | -10.40 | [-11.55, -9.94] | -12.18 / -9.06 | 0/16 | [-11.01, -10.07] | -10.07 | +0.32 | -0.02 | -0.61 | +0.31 | +117.50 | -15.03 | 0.33 / 0.05 |
| 50 | 0.1 | 10 | -10.02 | [-10.76, -9.30] | -11.52 / -8.61 | 0/16 | [-10.54, -9.51] | -9.91 | -1.01 | -0.03 | +0.79 | +0.35 | +136.50 | -38.32 | 0.68 / 0.09 |
| 50 | 0.1 | 100 | -10.06 | [-10.77, -9.33] | -11.57 / -8.59 | 0/16 | [-10.53, -9.51] | -9.91 | -1.01 | -0.03 | +0.82 | +0.25 | +136.50 | -38.32 | 0.68 / 0.09 |
| 50 | 0.3 | 10 | -8.95 | [-10.00, -8.40] | -10.68 / -7.91 | 0/16 | [-9.65, -8.70] | -9.12 | -4.03 | +1.04 | +2.62 | +0.78 | +178.00 | -36.71 | 3.02 / 0.70 |
| 50 | 0.3 | 100 | -9.12 | [-10.11, -8.60] | -10.86 / -8.10 | 0/16 | [-9.77, -8.87] | -9.12 | -4.03 | +1.04 | +2.57 | +0.91 | +178.00 | -36.71 | 3.02 / 0.70 |

v0.1 sensitivity table, not the note's result: 16 paired seeds x 600 s per cell, VolSpace matched to PriceSpace(GLFT asymptotic) at q = 0 (same summed $ half-spread and the same summed linearised $ skew per lot), differences are vol minus price per seed in $; markout horizon 60 s; the hedge is a $-delta band with the default sqrt-law cost model. Sign = seeds with a positive difference / seeds with a nonzero one; the CI is a 2,000-resample bootstrap of the mean. n_fills is a count and vega_T the terminal net vega in $ per 1.00 vol; neither is a $ P&L term. The matching equates the SUM of the $ half-spreads, not their shape: at gamma 10 and q = 0 the price-space half-spread is 0.041-0.041 $ on every strike while the matched vol-space one runs 0.016 $ at the wings to 0.073 $ at the money, so the vol quoter fills the wings more often at less capture each; the spread column, not the inventory skew, carries most of every cell. The last column is the share of lots filled at a quote INSIDE fair (the per-side offset c + (2q + 1) w / 2 is negative beyond |q| = c / w, which is 107.0 / 25.6 / 5.4 lots on the tightest instrument at gamma 1 / 10 / 50); those fills carry a negative SPREAD term, the maker paying to unwind; the gamma 50 cells reach that regime.
<!-- quotesim:end:sweep -->

### Post-trade review: markouts by counterparty class

<!-- quotesim:begin:markouts -->
| horizon_s | class | n | realised_spread_mean | adverse_mean | markout_mean | se_clustered | realised_spread_vp | adverse_vp | markout_vp | se_clustered_vp |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | all | 15988 | 0.0405 | -0.0001 | +0.0404 | 0.0001 | 0.918 | -0.002 | +0.916 | 0.005 |
| 1 | uninformed | 11359 | 0.0400 | -0.0000 | +0.0400 | 0.0001 | 0.903 | -0.000 | +0.903 | 0.006 |
| 1 | informed | 4629 | 0.0417 | -0.0002 | +0.0415 | 0.0001 | 0.953 | -0.005 | +0.948 | 0.009 |
| 10 | all | 15988 | 0.0405 | -0.0003 | +0.0401 | 0.0001 | 0.918 | -0.009 | +0.908 | 0.005 |
| 10 | uninformed | 11359 | 0.0400 | +0.0002 | +0.0402 | 0.0001 | 0.903 | +0.003 | +0.906 | 0.006 |
| 10 | informed | 4629 | 0.0417 | -0.0017 | +0.0399 | 0.0003 | 0.953 | -0.039 | +0.913 | 0.011 |
| 60 | all | 15988 | 0.0405 | -0.0032 | +0.0373 | 0.0002 | 0.918 | -0.076 | +0.842 | 0.008 |
| 60 | uninformed | 11359 | 0.0400 | +0.0003 | +0.0403 | 0.0002 | 0.903 | -0.000 | +0.903 | 0.008 |
| 60 | informed | 4629 | 0.0417 | -0.0117 | +0.0299 | 0.0006 | 0.953 | -0.262 | +0.690 | 0.014 |
| 300 | all | 15988 | 0.0405 | -0.0027 | +0.0377 | 0.0003 | 0.918 | -0.069 | +0.849 | 0.014 |
| 300 | uninformed | 11359 | 0.0400 | -0.0001 | +0.0399 | 0.0005 | 0.903 | -0.014 | +0.889 | 0.017 |
| 300 | informed | 4629 | 0.0417 | -0.0092 | +0.0325 | 0.0011 | 0.953 | -0.203 | +0.750 | 0.027 |

Toxicity at the 60 s horizon:

| class | n | share | realised_spread_mean | adverse_mean | markout_mean | give_back | frac_negative |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 15988 | 1.000 | 0.0405 | -0.0032 | +0.0373 | +0.078 | 0.022 |
| uninformed | 11359 | 0.710 | 0.0400 | +0.0003 | +0.0403 | -0.008 | 0.018 |
| informed | 4629 | 0.290 | 0.0417 | -0.0117 | +0.0299 | +0.281 | 0.032 |

One run, seed 3, 3600 s, informed fraction 0.3 (p = 1, 60 s look-ahead), gamma 10, PriceSpace(GLFT asymptotic), band $10; 15988 fills. Markout = s_f (F_(t+h) - px_f) against fair, $ per contract (`_mean`) and vol points via the fill's vega (`_vp`); SE clustered by time bucket of width h. The horizons are synthetic (the seeded fair process, not a recorded tape). give_back = -adverse / realised spread.
<!-- quotesim:end:markouts -->

### Whalley-Wilmott band vs the hedge-cost frontier

<!-- quotesim:begin:ww -->
| band $ delta | hedges per run | hedge cost | RMS residual $ delta | realised mean | realised sd across seeds |
|---:|---:|---:|---:|---:|---:|
| 0 | 600.0 | 6.22 | 0.00 | +100.69 | 2.24 |
| 5 | 564.0 | 6.21 | 0.67 | +100.70 | 2.24 |
| 10 | 550.0 | 6.20 | 1.40 | +100.71 | 2.24 |
| 25 | 488.9 | 6.10 | 5.98 | +100.81 | 2.25 |
| 50 | 410.1 | 5.83 | 15.60 | +101.08 | 2.23 |
| 100 | 262.8 | 4.94 | 42.56 | +101.95 | 2.23 |
| 250 | 93.8 | 3.16 | 115.95 | +103.70 | 2.15 |
| 1000 | 9.0 | 0.99 | 416.82 | +105.37 | 2.39 |

| gamma | lam = half_spread / S | Gamma (ATM call) | WW half-band, shares per lot | WW half-band, $ delta per lot |
|---:|---:|---:|---:|---:|
| 1 | 1.0e-04 | 0.0695 | 0.0417 | 4.17 |
| 10 | 1.0e-04 | 0.0695 | 0.0194 | 1.94 |
| 50 | 1.0e-04 | 0.0695 | 0.0113 | 1.13 |

Frontier: PriceSpace(GLFT asymptotic, gamma 10), informed fraction 0.1, 8 seeds x 600 s per band, default cost model (half-spread $0.01 / share, Y 0.5, daily vol 1 %, ADV 5e7). Whalley-Wilmott: H = (3/2 lam S Gamma^2 / gamma)^(1/3) shares per option for the ATM call, lam = proportional cost = half-spread / S, r = 0; it is a per-lot no-trade half-band, the frontier's band is the book's $-delta mismatch, so they are compared by eye, not equated.
<!-- quotesim:end:ww -->

## Design rules

Tested:

- The top identity `spread + inventory + hedge - hedge_cost = realised` holds to max(1e-9, 1e-12 × the run's gross $ of summed products) on every run — the rounding of those products is within 2 ulp each, so the bar is 4,500× the rounding bound and 1e12 below one product — and `sim.run` raises `AssertionError` with the numbers otherwise (tests/test_pnl.py at multiplier 1 and at multiplier 100 / S0 500 / ±100-lot opening book, tests/test_sim.py, and the identity table above over 100 seeds with jumps on, gaps under 1e-11 against run-time bars of 1e-7 to 1e-6).
- Both sub-splits of the inventory term are exact (`OPENING + ADVERSE_h + DRIFT_h`, `THETA + INV_ex_theta`); a zero-inventory run has `INV = 0` exactly; the Greek layer's residual is small relative to the gross inventory move at 1 s steps.
- Markout = realised spread + adverse for every fill; the all-class 60 s adverse mean times the fill count equals the attribution's `ADVERSE_h`.
- Same seed and identical quoters give bit-identical fills, hedges and realised; different quoters under the same seed see the same path and candidate stream, and a wider quoter's fills are a cell-by-cell subset of a tighter one's.
- The quoter only ever sees the fair at the clock (`PastOnlyFair` raises on any index ahead, and every snapshot is a read-only copy: no numpy view of the pre-drawn path reaches the quoter, an in-place write on a snapshot raises instead of corrupting the fair); the fill acceptance rate is `exp(-k delta)` within a binomial CI; the informed adverse mean is `-(2p - 1) sigma_F sqrt(h) sqrt(2 / pi)` within block SE.
- The published closed forms: A-S 2008 spread 1.690770 / 1.290770 and reservation shift 0.4 per lot; GLFT exact `delta_b(0) = 3.313045` vs asymptotic 3.312915 and `|exact - asymptotic| <= 5e-3` for `|q| <= 20`, plus the exact quotes at Q = 100 against a 70-digit reference (`delta_b(65) = 7.459403`, `delta_b(99) = 9.320168`) and against the long-horizon ground state on every q; Stoikov-Sağlam eq. 17 and Theorem 4 (`k = 0.03855879`, vega share 99.87 %); Whalley-Wilmott `H = 0.06198337` and `8x lam -> 2x H` exactly.
- Every Greek of `SyntheticFair` against central finite differences; the Merton compensator makes `E[S_{t+dt}] = S_t` for any `dt`.
- `quotesim report` regenerates this README byte-identically on one machine (tests/test_report_cli.py runs it twice at the tests' sizes), and `quotesim report --check` (CI) reproduces every token on another, a gap ceiling within one decade (the gap is a last-ulp quantity and CI's runners do not share a summation kernel: one commit printed `< 1e-12` and `< 1e-11` on two runs); the README contains none of the five banned words listed in tests/test_report_cli.py (no language about what a desk would earn).

By construction (not a test):

- Fills are one lot at the quoted price with no queue position, latency or partial fills; a candidate that arrives inside the spread is a fill, not a price improvement.
- A per-side offset of the A-S / GLFT linear skew goes negative beyond |q| = c / w lots; the quote then sits inside fair, is hit with probability 1 and its spread term is negative (the maker pays to unwind). Nothing floors or pulls it; the sweep table prints the share of such lots per cell (only the gamma 50 cells have any).
- Marks are to fair (the surface mid), never to the maker's own quotes; the hedge fills at the mid with the whole slippage in the cost term, so `HEDGE` and `HCOST` separate exactly.
- The informed counterparty's look-ahead is a property of the synthetic flow, not of the quoter; nothing on the quoting side reads it.
- Vol-space quotes are a derivation (Stoikov-Sağlam tilt over vega, variance generalised to the book's net vega and net gamma), not the paper's price-premium rule.
- No language about what a desk would earn and no deployment language anywhere in the repo.

## What is where

```text
quotesim/
├── src/quotesim/
│   ├── fair.py        # SyntheticFair, FairPath, PastOnlyFair; Black prices and Greeks (years)
│   ├── flow.py        # FlowParams, Streams, arrivals: thinned Poisson flow with informed candidates (seconds)
│   ├── quoter.py      # AS2008, GLFT2013, GLFTAsymptotic; PriceSpace, VolSpace, match_spreads; guards
│   ├── hedger.py      # BandHedger, TimeHedger, NoHedger; CostModel; band_ww
│   ├── pnl.py         # Attribution, attribute, check_identity (the no-residual identity)
│   ├── adverse.py     # markouts by class and horizon, clustered SE, toxicity
│   ├── sim.py         # SimConfig, Run, run, paired, PairedResult, paired_stats
│   ├── report.py      # the five README sections from fixed seeds
│   └── cli.py         # quotesim report / quotesim run
├── tests/             # one file per module (report and cli share one); oracles cited in docstrings
├── docs/PLAN.md       # plan v2: what survived research and why
├── docs/DESIGN.md     # conventions, identities, deviations from the plan, v0.2 semantics
└── CHANGELOG.md
```

## Roadmap

- v0.1 (this): synthetic flow, three quoters in two spaces, band/time hedging, exact attribution, paired seeds, the five report tables.
- v0.2: `book.py` L2/FIFO queue with a latency parameter (semantics recorded in docs/DESIGN.md); EOD-to-EOD fair re-mark from a private chain plus seeded synthetic intraday flow (no replay, see below); OFI / VPIN toxicity on real prints; quote-reactivity to flow tested against a size-matched random-widening null at equal duty cycle; the vol-space vs price-space note with a pre-registered primary metric and at least 200 paired seeds; a volsurf adapter behind `FairSurface`.
- Not planned: any claim about what a desk would earn.

## Data and privacy

Everything in this repository is synthetic: the fair surface, the spot and vol paths, the flow, the counterparty classes, the markout horizons. There is no market data of any kind in the repo, no fixtures derived from market data, and nothing reads a data file at run time (`quotesim report` only rewrites README.md). There is no replay mode because no public, redistributable sub-minute options quote or print sample exists (LOBSTER is an equity limit order book; the delayed Cboe snapshot feeds are a handful of snaps a day and unlicensed; the EOD chain mirrors are one prior-close snapshot per contract per day), and markouts at 1 s to 5 min on end-of-day data would be meaningless. Any private chain used for the v0.2 re-mark stays outside the repo.

## Companion repos

- [tcakit](https://github.com/charlieyanhx/tcakit) — transaction-cost analysis; the sqrt-law cost form used by `hedger.CostModel`
- [deskboard](https://github.com/charlieyanhx/deskboard) — desk risk bus, Greeks, limits
- [pricers](https://github.com/charlieyanhx/pricers) — Black, trees, characteristic-function pricers
- [riskkit](https://github.com/charlieyanhx/riskkit) — SPAN-style margin and scenario ladders
- [volsurf](https://github.com/charlieyanhx/volsurf) — SVI / SSVI surfaces; the v0.2 `FairSurface` adapter
- [quant-research-agent](https://github.com/charlieyanhx/quant-research-agent) — research task runner
- [tickq](https://github.com/charlieyanhx/tickq) — DuckDB market-data SQL; the lake the fills and quotes would live in
- [lobcore](https://github.com/charlieyanhx/lobcore) — bounded-array limit order book in Rust with ITCH 5.0 replay and PyO3 bindings
- [exhibitkit](https://github.com/charlieyanhx/exhibitkit) — sell-side research documents from Markdown
- [claimkeeper](https://github.com/charlieyanhx/claimkeeper) — a ledger that scores a note's falsifiable claims once their dates arrive

MIT © Hanxiong (Charlie) Yan
