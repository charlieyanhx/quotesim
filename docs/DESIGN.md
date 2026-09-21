# quotesim design (v0.1)

This document records the conventions the code keeps, the identities it asserts, where it deviates from
docs/PLAN.md and why, and the semantics reserved for v0.2. Everything numerical is synthetic and seeded;
the README's tables are the only numbers, and `quotesim report` is the only thing that writes them.

## 1. What the simulator is and is not

A fixed-step quoting-strategy simulator: quote update -> fills -> hedge check -> state step, one step per
`dt` seconds (1 s default). Fills are drawn from a parametric intensity `lambda(delta) = A exp(-k delta)`
per side and instrument by thinning pre-drawn candidates against the quotes. There is no queue position,
no latency, no matching engine, no size (one lot per fill), and no market data of any kind. The question it
answers is "how do two quoting rules compare under identical flow"; it never answers "what would a desk
earn", and no language of that kind appears in the repository (a test greps the README).

## 2. Units and sign conventions

| module | clock | money | other |
|---|---|---|---|
| `fair.py` | years (`dt_years`, `T_years`, theta per year, `alpha` and `spot_vol` per sqrt(year); `YEAR_SECONDS = 365 * 24 * 3600`) | $ per unit of underlying | `multiplier` is metadata only |
| `flow.py` | seconds | $ (`delta` is the $ distance of the quote from fair) | `MM_SIGN = [+1, -1]`: side 0 = the maker buys at its bid, side 1 = the maker sells at its ask |
| `quoter.py` | seconds (`T`, `tau_s`); `alpha_s`, `sigma_s` per sqrt(second) via `per_sqrt_second` | $ (`gamma` per $, `k` per $, `A` per second) | `VolSpace` half-spread and skew in vol units (1.00 = 100 vol points) |
| `hedger.py` | seconds | $; hedge shares | `band_ww` takes `tau` in years because `Gamma` comes from the pricer |
| `pnl.py`, `adverse.py` | seconds (`h`) | $ per run / $ per contract; vol points = $ / vega x 100 | `+` = the maker bought |
| `sim.py` | seconds | $ | `positions[i + 1]` = lots held over step i |

Cash convention (the one every identity relies on): an option fill moves cash by `-sign * lots * px *
multiplier`; a hedge trade moves cash by `-dh * S - cost`, i.e. the hedge fills at the mid and the whole
slippage is the cost term, so `HEDGE` and `HCOST` separate exactly. Marks are to fair (the surface mid),
never to the maker's own quotes. Costs are non-negative; a cost model that returns a negative number is
rejected.

## 3. The attribution identity (no residual at the top)

With `F_t` the fair price vector, `S_t` spot, `q_t` the lots held over step t (after that step's fills),
`h_t` the hedge shares held over step t, fills `f = (t_f, j, dq_f, px_f)` and hedge trades with cost `c`:

```
SPREAD = sum_f dq_f (F_{t_f} - px_f)          realised half-spread on every fill (negative for a fill at a quote
                                              inside fair, see section 6)
INV    = sum_t q_t . (F_{t+1} - F_t)          option positions marked step by step
HEDGE  = sum_t h_t (S_{t+1} - S_t)            hedge shares marked step by step
HCOST  = sum_hedge trades c
R      = cash_T + q_T . F_T + h_T S_T - (q_0 . F_0 + h_0 S_0)
R      = SPREAD + INV + HEDGE - HCOST          exactly
```

Why it is exact: every cash flow or mark change in the loop is one of the four. An option fill moves cash
by `-dq px` and the mark by `+dq F` (their sum is the fill's spread term); a hedge trade moves cash by
`-dh S - c` and the mark by `+dh S` (net `-c`); holding moves the marks by `q dF` and `h dS`. Summing the
step-by-step marks telescopes to the terminal marks minus the opening ones. Every sum is a `math.fsum`
over the elementwise products and `cash_T` is a `math.fsum` over the flows, so `gap = R - (sum)` is the
rounding of the products themselves (each within 2 ulp), bounded by `2 eps * scale` with `scale` the gross $
of every summed product (fills, hedge trades and costs, opening and terminal marks, gross inventory and
hedge marks; `Attribution.scale`): of order 1e-12 on $-scale runs and ~1e-9 at multiplier 100 on a 500 $
underlying with a 100-lot opening book (scale ~1e8-1e9 $). `sim.run` calls `pnl.check_identity` on every
run and raises `AssertionError` with the numbers if `|gap| >= bar = max(1e-9, 1e-12 * scale)` (deviation
D11: the relative part is 4,500x the rounding bound and 1e12 below one product, the absolute floor is where
$-scale runs sit; ~1e-7 on the README's strip, where the measured gaps are below 1e-11); the tests do the
same and also re-derive the four terms with plain floats from the raw fill and hedge records, at multiplier
1 and at multiplier 100 / S0 500.

Two sub-splits of `INV`, both exact by the fill decomposition `q_t = q_0 + sum_{f: t_f <= t} dq_f`:

```
INV = OPENING + ADVERSE_h + DRIFT_h,   OPENING   = q_0 . (F_T - F_0)
                                       ADVERSE_h = sum_f dq_f (F_{t_f + h} - F_{t_f})     (t_f + h clipped at T)
                                       DRIFT_h   = sum_f dq_f (F_T - F_{t_f + h})
INV = THETA + INV_ex_theta,            THETA     = sum_t q_t . theta_t dt   (theta per year, dt in years)
```

`ADVERSE_h` is the fill-level markout summed over fills, a subset of `INV`; plan v1's five-way identity
with adverse selection beside inventory double counted, so adverse selection is never a fifth top-level
term. `h = 60 s` is the canonical horizon; the other horizons live only in the markout table.

The Greek explanation layer is the only place with a residual: `INV_greek = sum_t q_t . (Delta dS + 1/2
Gamma dS^2 + Vega dsigma + theta dt)` with `dsigma` each instrument's own fair-vol change (sticky strike,
so the smile move from `dk = -dS / S` sits in the vega term, not in delta), `RESID = INV - INV_greek`
(vanna, volga, third order, jumps). The identity table reports `max |RESID| / sum |q dF|` with Merton jumps
on; it is at the 1e-2 level there because a jump is not a second-order expansion, below 1e-4 on the
five-strike strip without jumps (the wings' volga at vol-of-vol 0.6), and below 1e-5 for one ATM strike at
vol-of-vol 0.16; tests/test_pnl.py asserts 1e-3 and 1e-5 respectively (deviation D12).

## 4. Common random numbers and the paired harness

`flow.Streams.draw(seed, n_steps, n_instruments, A, dt)` draws every stream from one
`numpy.random.Generator` in a fixed order: spot normals, vol normals, jump uniforms, candidate counts
`Poisson(A dt)` per (step, instrument, side), acceptance uniforms, informed uniforms, direction uniforms.
Two quoters under one seed therefore see the same path, the same candidates, the same uniforms and the same
informed flags; they differ only through thinning (`accept iff U < exp(-k delta)`, monotone in `delta`, so
a wider quote's fills are a cell-by-cell subset of a tighter one's) and through their own hedges. The tests
assert bit-identical fills, hedges and attribution for identical quoters, an identical candidate stream and
path for different quoters, and the subset property.

`sim.paired(config, quoters, seeds, hedger=..., fair=..., params=...)` runs every quoter on every seed and
keeps only `Run.summary()` per (seed, quoter). `PairedResult.table()` reports, per identity term, the paired
differences `d_i = M_B(seed i) - M_A(seed i)`: median, q25/q75, p5/p95, mean, sign count with a two-sided
binomial p, and a seeded 2,000-resample bootstrap 95 % CI of the mean. Never the best seed, never an unpaired
t-test.

## 5. Flow

`FlowParams(A, k, informed_frac, p_informed, h_info, persist)`. Candidates arrive at the maximum rate `A`
per second per side per instrument; the thinned count is exactly `Poisson(lambda dt)` because the number
accepted out of `c` candidates is the `Binomial(c, exp(-k delta))` quantile at the cell's uniform (for
`c = 1` this is `accept iff U < p`; see deviation D2). `P(at least one fill in dt) = 1 - exp(-lambda dt)`,
never `lambda dt`.

An informed candidate (probability `informed_frac`) reads the fair `h_info` seconds ahead on the pre-drawn
path (`FairPath.prices[min(i + h_steps, n_steps)]`) and trades WITH the move with probability `p_informed`,
AGAINST it otherwise, so `E[adverse_h] = -(2p - 1) E|F_{t+h} - F_t|` and `p = 0.5` is uninformative (see
deviation D1). This look-ahead is a property of the counterparty in a simulation; nothing on the quoting
side can read it: the quoter is fed `PastOnlyFair.current()`, `PastOnlyFair.snapshot(i)` raises
`FutureAccessError` for any `i` beyond the clock, and every snapshot is a read-only copy of one row, so no
numpy view of the pre-drawn path (whose `.base` would be the whole future) reaches the quoter and an
in-place write in a quoter cannot corrupt the fair the run marks against (tested). Uninformed candidates are 50/50 with optional
Markov persistence (`persist`) using the direction uniforms they would otherwise not consume, so the streams
are unchanged.

## 6. Quoters

- `AS2008(gamma, k, sigma, T)`: reservation `r = s - q gamma sigma^2 (T - t)`, spread `gamma sigma^2 (T - t)
  + (2 / gamma) ln(1 + gamma / k)`. Its linear-in-`(T - t)` term dominates at the open with an intraday `T`,
  so it is not the default.
- `GLFT2013(gamma, k, sigma, A, T, Q=100)`: the exact finite-inventory solution. `v(t) = exp(-M' (T - t)) 1`
  with `M' = M - w_min I` (the spectrum shifted by its minimum eigenvalue: every quote is a ratio
  `v_q / v_{q +- 1}`, so the scale is free) is evaluated by uniformization: `Lambda = max diag(M')`,
  `P = I - M' / Lambda >= 0` entrywise, `v = sum_n Poisson(n; Lambda (T - t)) P^n 1`, every term non-negative,
  so each entry has relative accuracy (~1e-12) however small it is. The powers `P^n 1` are cached per object up
  to the first `n` where the slowest even mode is below 1e-12 relative (ratio test scaled by the spectral gap)
  or the last `n` with Poisson weight at `T`, the Poisson weights come from the ratio recursion anchored at
  the mode, the terms beyond the cache are collapsed onto the last power with the exact tail mass. The
  eigendecomposition (`v(t, method="eigh")`) and raw `scipy.linalg.expm` (`method="expm"`; it overflows
  unshifted, `v ~ 1e170` at `T = 600 s`) are kept for cross-checks: both have ABSOLUTE accuracy `eps max(v)`,
  and since `v_q` decays like `exp(-1/2 sqrt(alpha / eta) q^2)`, their tail is rounding noise once
  `v_q / v_0 < ~1e-8` (|q| > ~50 at the paper's parameters with the default Q = 100: negative entries, a bid
  at q = 65 of -0.17 instead of 7.46, NaN beyond q = 70). Tested: uniformization = eigh = expm to 1e-10
  relative at Q = 30; every offset finite and monotone in q over -Q .. Q at T = 600 s and 6.5 h with Q = 100;
  equality with a 70-digit reference at nine q's and three t's to 1e-11; equality with the exact
  long-horizon ground state (inward three-term recurrence) on every q at 6.5 h to 1e-12. A Q whose ground
  state spans more than 70 decades (beyond the Poisson window's exp(-200) tail bound) raises at construction.
- `GLFTAsymptotic(gamma, k, sigma, A)`: the closed form; the default for intraday horizons. `|exact -
  asymptotic| <= 5e-3` for `|q| <= 20` at the plan's parameters (tested), and the spread-vs-gamma dip is a
  regression test.
- `PriceSpace(kind, ...)`: the scalar rule per instrument on the option price with `sigma_opt^2 = (Vega_j
  alpha)^2 + 1/2 Gamma_j^2 sigma^4 S^4 tau` (hedged residual; the delta term is added only with
  `hedged=False`, otherwise hedged risk is penalised twice). For `kind="glft"` one `GLFT2013` per instrument
  is built at the first `quotes` call on that snapshot's `sigma_opt` and re-built whenever `sigma_opt` has
  moved by more than 1 % from the fitted value (`PriceSpace.SIGMA_REFIT_TOL`), so the exact quotes track the
  surface to that tolerance within a run and across strips of the same instruments (a review found a
  0.0005 $ half-spread bias when the first fit stayed frozen on a much hotter surface); a snapshot with
  different strikes or rights raises. The
  linear skew of `as` and `glft_asym`, `c + (2q + 1) w / 2`, is negative beyond `|q| = c / w` (about 107 /
  26 / 5 lots for the tightest instrument at gamma 1 / 10 / 50 on the report's strip): the quote then sits
  inside fair, `flow.accept_prob` treats a negative distance as zero (accepted with probability 1), and the
  fill's SPREAD term is negative, the maker paying to unwind. Nothing floors or pulls such a quote (it is
  the model's optimal quote); `Run.n_inside_fair` counts those lots and the sweep table prints their share.
- `VolSpace(gamma, alpha_s, sigma_s, tau_s, hs_vol, skew_scale)`: a parallel surface shift `skew_vol(j) =
  gamma alpha^2 tau V_net + 1/2 gamma sigma^4 S^4 tau^2 Gamma_net (Gamma_j / Vega_j)`, `bid_vol = sigma_fair
  - skew - hs_vol`, `ask_vol = sigma_fair - skew + hs_vol`, priced with Black. Long vega lowers both quotes.
  This is a derivation from Stoikov-Sağlam Theorem 4 (their tilt divided by vega, the variance generalised
  to the book's net vega and net gamma), not their price-premium rule with a linear intensity.
- `match_spreads(price_q, vol_q, snap)`: returns a `VolSpace` with `hs_vol = sum hs_$ / sum Vega` and
  `skew_scale = sum(price-space $ skew per lot) / sum(vol-space $ skew per lot)` at `q = 0`, plus a
  `MatchReport`. The matching equates sums, not shapes: constant `hs_vol` is tight in $ at the wings and
  wide at the money, which is what the sweep table measures first (its caption says so with the numbers).
  The research memo matched at the ATM vega instead (`hs_vol = hs_$ / Vega_ATM`); the plan and the code match
  the sum, so the ATM $ half-spread is not equal after matching (0.041 $ price-space vs 0.073 $ vol-space
  at gamma 10, see the sweep caption); see deviation D7.
- Guards: `round_to_tick` (bids down, asks up; a bid `<= 0` is pulled even at `tick = 0`), `MaxLossGuard`,
  `vega_cap`. No size rules in v0.1.

## 7. Hedger

`BandHedger(band_delta_usd, cost_model)` rehedges to `target = -sum_j q_j Delta_j multiplier` when the
$-delta mismatch `|h - target| S` exceeds the band; `TimeHedger(every_s)` on the grid `0, every_s, 2 every_s,
...` from the start of the run (`HedgeDecision.due` marks the rule firing even when the mismatch is zero and
nothing trades, and `sim.run` resets `t_last` on `due`, not on a trade, so the grid is not re-anchored at the
first fill); `NoHedger` never. The
default `CostModel(half_spread=0.01, Y=0.5, sigma_daily=0.01, adv=5e7)` is `half_spread |dh| + Y sigma_daily
sqrt(|dh| / ADV) S |dh|`, tcakit's sqrt-law form with `Y` a labelled constant (tcakit has no impact predictor
to call, and it is not imported). `band_ww(S, Gamma, lam, gamma, r, tau) = (3/2 e^{-r tau} lam S Gamma^2 /
gamma)^(1/3)` shares per option (Whalley-Wilmott 1997), a per-lot no-trade half-band; the report puts it
beside the simulated band frontier for comparison by eye, not as an equation.

## 8. Markouts and toxicity

`MO_h(f) = s_f (F_{t_f + h} - px_f) = realised spread + adverse`, against fair, `t_f + h` clipped at the end
of the run exactly as `ADVERSE_h`, so `adverse_mean(all, h_markout) * n == attribution.adverse_h` (tested).
A horizon that is not a positive whole number of steps raises (`adverse.horizon_steps`, shared with
`pnl.attribute`) rather than being rounded to whole steps under the requested label.
Vol points divide by the fill's fair vega (fills below `VEGA_FLOOR` are left out of the vol-point means and
counted in `n_vp`). The SE is cluster-robust with fills clustered by time bucket of width `h` (fills inside
one horizon share the fair path); it is NaN with fewer than two buckets, which is why the 300 s row of the
tests' short run is blank. Classes are exact in a simulation ("informed" / "uninformed"); a re-mark on real
prints would say "unknown". The horizons are synthetic.

## 9. Report policy

`quotesim report` regenerates the five README sections between `<!-- quotesim:begin:<name> -->` /
`<!-- quotesim:end:<name> -->` markers from fixed seeds (`report.ReportSizes()`): identity over 100 seeds x
600 s (half price-space, half matched vol-space, Merton jumps on), A-S 2008 Tables 1-3 at 1,000 vectorised
paths per gamma, the paired sweep at 16 seeds x 600 s per cell (18 cells), one 3,600 s markout run, and the
band frontier at 8 seeds x 600 s per band; about 90 s of wall time on the machine named in the README.
Numbers are printed at 2-4 significant figures and gaps as power-of-ten ceilings so that last-ulp
differences do not change the text; `tests/test_report_cli.py` regenerates a README skeleton twice at
`ReportSizes.small()` and asserts the bytes are identical. Across machines the ceiling itself is the one
thing that can move: a gap at a decade boundary printed `< 1e-12` and `< 1e-11` on two CI runs of the same
commit (GitHub's runners do not share a CPU, and numpy's pairwise summation blocks differently per SIMD
width), so CI runs `quotesim report --check`, which requires every token to match and lets a ceiling
differ by one decade; a two-decade move, or any other token, fails the check. The A-S table stands beside the paper's numbers;
the paper's RNG is not published, so only directions and rough magnitudes (within 15 %, tested) are
reproducible.

## 10. Deviations from docs/PLAN.md

- D1 (flow, informed direction). The plan's "sign of the move with probability p, else a coin" gives
  `E[adverse] = -p E|dF|` and a nonzero mean at `p = 0.5`, contradicting its own oracle `-(2p - 1) sigma_F
  sqrt(h) sqrt(2 / pi)`. Implemented: with the move with probability `p`, against it otherwise; the oracle
  values (`-0.061804` at `p = 0.6`, `-0.309019` at `p = 1`, 0 at `p = 0.5`) are reproduced.
- D2 (flow, thinning). `accept_u` is one uniform per (step, instrument, side) while candidate counts are
  Poisson and can exceed one; all-or-none acceptance per cell would break `P(fill) = 1 - exp(-lambda dt)`.
  The Binomial quantile at the cell's uniform keeps the thinned count exactly Poisson, reduces to `U < p`
  for one candidate, and is monotone in `p` (the CRN superset property). The informed flag and direction
  are per cell.
- D3 (quoter, GLFT exact). Uniformization of the shifted generator instead of `expm` (overflow at intraday
  horizons) and instead of the eigendecomposition the first implementation used (absolute accuracy only:
  its tail was rounding noise beyond |q| ~ 50 at the default Q = 100, a wrong sign at q = 65 and NaN beyond
  q = 70 at every horizon >= 600 s); `eigh` and `expm` kept as methods and tested equal where resolvable
  (Q = 30). See section 6 for the tests over the whole inventory range.
- D4 (units). `fair.py` in years, everything else in seconds; `per_sqrt_second` converts. `band_ww`'s `tau`
  is in years.
- D5 (fair). ATM vol is an arithmetic OU floored at `VOL_FLOOR`, not lognormal, so `alpha` is the absolute
  vol-of-vol Theorem 4 uses; the skew factor is constant (the plan's `step` has only three shocks). Merton
  jumps: one uniform per step gives occurrence and size (at most one jump per step) and the compensator is
  the exact discrete `ln(1 + p_J kappa_J)`, which keeps `E[S_{t+dt}] = S_t` for any `dt` where the continuous
  `lam kappa_J dt` failed a 1e-3 test at `lam dt = 0.5`. `vol(k, T)` has no term structure; Greeks are
  sticky-strike Black partials; `r = q = 0`.
- D6 (fair, extra objects). `FairPath` (the whole pre-drawn path, vectorised pricing, read-only arrays) and
  `PastOnlyFair` (clock plus `FutureAccessError`) beyond the plan's "a mock that raises", so the leak guard is
  on in every run; `FairPath.snapshot` hands out read-only copies of one row (a view's `.base` would have been
  the whole future, and a writable view an in-place corruption of the fair; both tested). `SyntheticFair.step`
  mutates in place; `run` builds the path on a copy and leaves the caller's object untouched (tested).
- D7 (quoter, matching). `match_spreads` returns `(VolSpace, MatchReport)` and matches the skew as well as
  the spread (`skew_scale` is 1.0 by algebra for AS2008 at `tau = T - t`, tested, and not 1 for the
  asymptotic quoter). The research memo (report_a3e5dd section 3) matched the half-spread at the ATM vega;
  the plan and the code equate the SUM of the $ half-spreads, so after matching the ATM $ half-spread is
  0.041 (price space) vs 0.073 (vol space) at gamma 10, a 75 % difference the sweep caption reports. No size
  rules; a deep-OTM bid below a wide half-spread is pulled rather than quoted negative.
- D8 (hedger). `decide(h, target, S, t, t_last)` returns a `HedgeDecision` rather than mutating; the hedge
  fills at the mid with the slippage in `cost`; `TimeHedger` trades at `t = 0` when `t_last` is None and
  `HedgeDecision.due` (the rule fired, even with a zero mismatch) is what advances `t_last`, so the grid is
  anchored at `t = 0` rather than at the first nonzero mismatch; `NoHedger` and `target_shares` added.
- D9 (sim). `SimConfig` adds `q0` and `h0` (opening inventory and hedge) so `OPENING` is exercised; `Run`
  keeps the full quote arrays (`bids`, `asks`) for diagnostics; `paired` takes `hedger`, `fair`, `params`
  as keyword arguments and an optional per-quoter `hedgers` map.
- D10 (report). The identity table is 50 seeds per quoter (100 in total) at 600 s, not full days, to keep
  `quotesim report` under about 3 minutes on the machine the README names (the per-step costs are not
  printed by any script, so none are quoted here). The sweep is 16 paired seeds per cell and is labelled a
  v0.1 sensitivity table; the note needs >= 200 seeds and a pre-registered metric.
- D11 (pnl, the bar). The plan's `|gap| < 1e-9` is a $-scale statement, not an identity statement: the gap
  is 1-2 ulp of the summed products, so at multiplier 100 on a 500 $ underlying with a 100-lot opening book
  a correct 3600 s run printed gaps of -1.0e-9 to -1.3e-9 and was rejected. The bar is
  `max(1e-9, 1e-12 * scale)` (section 3); the identity table prints the bar's ceiling beside the gaps, and
  its `runs with |gap| < 1e-9` column is a measurement because the run-time bar sits above 1e-9 there.
- D12 (pnl, RESID). The plan's oracle `RESID < 1e-5 |INV|` holds for one ATM strike at vol-of-vol 0.16 and
  is relaxed to `1e-3 * sum |q dF|` on the five-strike strip at vol-of-vol 0.6 (the wings' volga; measured
  below 1e-4), the two bounds tests/test_pnl.py asserts.
- D13 (adverse, horizons). A markout or attribution horizon must be a positive whole number of steps; a
  fractional or sub-step horizon raises instead of being rounded to whole steps under the requested label
  (two rows would otherwise carry the same markout and a different cluster width).

## 11. Reserved for v0.2

- `book.py`: an L2 book per instrument with price-time (FIFO) priority; the maker's resting order carries a
  queue position estimate (`ahead` = displayed size at its price when it joined, decremented by prints and
  cancellations at that price in proportion, a nanolob-style model); a latency parameter is the delay between
  a quote update and its arrival in the book, during which the old quote can still be hit; fills are
  partial and sized. This changes the top identity only through `px_f` and `dq_f` (sized fills), never its
  form.
- Fair re-mark: EOD-to-EOD fair-value re-mark from a private chain behind `FairSurface` (a volsurf adapter),
  plus seeded synthetic intraday flow between the two marks; the horizons stay labelled synthetic because
  no public sub-minute options quote or print sample can be redistributed.
- OFI / VPIN on real prints (formulas recorded in the research notes); quote-reactivity to flow tested
  against a size-matched random-widening null at equal duty cycle; the vol-space vs price-space note with a
  pre-registered primary metric (terminal vega variance at matched spread capture), >= 200 paired seeds sized
  from the v0.1 SE, the gamma frontier and the informed panel, a null reported if found.

## 12. What is not tested

Anything about what a strategy would earn; queue, latency and matching (they do not exist); tick rounding's
effect on the paired comparison (available, off by default in the report); calibration of the strip's
vol-of-vol, skew and flow parameters to any market (they are round numbers chosen for the tables).
