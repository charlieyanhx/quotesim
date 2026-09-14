# Changelog

## 0.1.0 (2026-09-14)

First release. Synthetic-flow options quoting simulator.

- `fair.py`: `SyntheticFair` (Black on a quadratic smile, GBM spot with optional Merton jumps, OU ATM vol), analytic Greeks checked against finite differences, `FairPath` (read-only arrays), `PastOnlyFair` raising `FutureAccessError` on any future access and handing out read-only copies (no view of the path reaches a quoter).
- `flow.py`: pre-drawn streams per seed, thinned Poisson arrivals `A exp(-k delta)`, informed candidates with a documented look-ahead, Markov persistence; the acceptance law and the informed-markout expectation reproduced.
- `quoter.py`: `AS2008`, `GLFT2013` (exact finite-Q by uniformization of the shifted generator, relative accuracy in every entry over the whole inventory range; eigendecomposition and expm kept as cross-checks), `GLFTAsymptotic`; `PriceSpace` (the exact kind re-fits `sigma_opt` beyond 1 % drift) and `VolSpace` behind one `Quoter` interface; `match_spreads`; tick rounding, `MaxLossGuard`, `vega_cap`. Published closed forms reproduced to their cited digits (A-S 2008, GLFT 2013, Stoikov-Sağlam 2009 eq. 17 / Theorem 4 / Theorem 5, Whalley-Wilmott 1997).
- `hedger.py`: `BandHedger`, `TimeHedger` (grid anchored at t = 0 via `HedgeDecision.due`), `NoHedger`, sqrt-law `CostModel`, `band_ww`.
- `pnl.py`: the four-term attribution identity with no residual (`|gap| < max(1e-9, 1e-12 x the run's gross $)`, asserted inside `sim.run`), the two exact inventory sub-splits, the Greek explanation layer with its `RESID`.
- `adverse.py`: markouts by counterparty class at 1 s / 10 s / 1 min / 5 min in $ and vol points with time-bucket clustered SE; `toxicity`; a horizon that is not a whole number of steps raises.
- `sim.py`: the fixed-step loop, `Run`, common random numbers by construction, `paired` with paired-difference tables (median, IQR, p5/p95, sign test, bootstrap CI).
- `report.py`, `cli.py`: `quotesim report` regenerates the five README tables from fixed seeds (byte-identical on one machine); `quotesim run --seed N` prints one attribution waterfall.
- 114 tests; the README contains no language about what a desk would earn (tested).

Not in this release (see docs/DESIGN.md section 11): queue, latency, matching, sized fills, any replay of market data, OFI / VPIN, quote-reactivity, the vol-space vs price-space note.
