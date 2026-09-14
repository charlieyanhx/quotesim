# quotesim — plan v2 (2026-09-12)

Plan v1 called this `mmsim` ("options market-making desk simulator … replays against recorded
quotes … markouts 1s/10s/1m/5m"). What survived research, and why:

1. **Name.** `mmsim` is taken on PyPI (a micromouse maze simulator). This is `quotesim`.
2. **What it is.** A quoting *strategy* simulator with synthetic flow: fills are drawn from a
   parametric intensity λ(δ) = A·e^{−kδ} (Avellaneda–Stoikov) by thinning; no queue position, no
   latency, no matching engine. It answers "how do quoting rules compare under identical flow", never
   "what would this desk earn". No profitability language anywhere (a test greps the README).
3. **No replay.** There is no public, redistributable sub-minute options quote or print sample
   (LOBSTER is equity LOB; etagtick is ~7 Cboe-delayed snaps a day and unlicensed; mztrading is one
   prior-close snap a day). v0.2 is "EOD-to-EOD fair-value re-mark from a private chain + seeded
   synthetic intraday flow"; markout horizons are labelled synthetic.
4. **numpy-only, no volsurf / tcakit imports in v0.1.** `fair.FairSurface` is a protocol with a
   synthetic implementation; the volsurf adapter is v0.2. Hedge cost is a callable with a documented
   default in tcakit's sqrt-law form (tcakit has no impact *predictor* to call).
5. **The attribution identity has no residual at the top level.** With fair F_t (the surface mid),
   spot S_t, option positions q_t, hedge position h_t and MM-side signs (+ = MM buys):
   `SPREAD = Σ_fills Δq_f (F_{t_f} − px_f)`, `INV = Σ_t q_t·(F_{t+1} − F_t)`, `HEDGE = Σ_t h_t (S_{t+1} − S_t)`,
   `HCOST = Σ_hedge fills |Δh|·(half-spread + impact)`; `R = SPREAD + INV + HEDGE − HCOST` exactly
   (every cash or mark change is one of the four; use `math.fsum`). Sub-splits, also exact:
   `INV = OPENING + ADVERSE_h + DRIFT_h` by the fill decomposition of q_t (`ADVERSE_h = Σ_f Δq_f (F_{t_f+h} − F_{t_f})`,
   h = 1 min canonical, clipped at T), and `INV = THETA + INV_exθ`. Only the Greek explanation layer
   (`Σ q (Δ dS + ½Γ dS² + Vega dσ + θ dt)`) carries an explicit `RESID`. Plan v1's five-way top
   identity with adverse selection beside inventory double counts.
6. **GLFT asymptotic is the default quoter for intraday horizons**; A-S 2008's linear-in-(T−t) quotes
   are only valid near T (the γσ²(T−t) term dominates at the open with T = 6.5 h). All three ship
   behind one interface: `AS2008`, `GLFT2013` exact finite-Q (matrix exponential of the tridiagonal
   generator, Q ≥ 100), `GLFT_asymptotic` (closed form).
7. **Vol-space quoting is a derivation, not Stoikov–Sağlam verbatim** (they quote price premiums with a
   linear intensity). Dividing their Theorem-4 tilt by vega and generalising the variance to a
   covariance with the book: `skew_vol(j) = γ α² τ V_net + ½ γ σ⁴ S⁴ τ² Γ_net · (Γ_j / Vega_j)`,
   `bid_vol_j = σ_fair,j − skew_vol(j) − hs_vol`, `ask_vol_j = σ_fair,j − skew_vol(j) + hs_vol`; α = absolute
   vol-of-vol per √time, τ = risk horizon, V_net = Σ q Vega, Γ_net = Σ q Γ. Long vega lowers both quotes.
   Price-space: per-strike A-S/GLFT on the option price with σ_opt² = (Vega_j α)² + ½Γ_j²σ⁴S⁴τ (hedged
   residual; the delta term only when the hedger is off — otherwise hedged risk is penalised twice).
8. **Paired seeds by construction.** One `numpy.random.Generator` per seed draws every stream up front
   in a fixed order (spot z, vol z, candidate arrivals at the max rate A per side and strike, acceptance
   uniforms, informed flags, direction coins); strategies differ only through thinning acceptance and
   their hedges. Reports are paired differences: median, IQR, p5/p95, sign test, bootstrap 95 % CI of the
   mean; never the best seed; never unpaired t-tests.

## v0.1 modules and contracts

| module | contract |
|---|---|
| `fair.py` | `FairSurface` protocol: `price(K, T, right)`, `vol(k, T)`, `delta, gamma, vega, theta(K, T, right)`, `spot`, `state()`; `SyntheticFair(S0, sigma0, skew_s, curv_c, alpha (vol-of-vol), kappa_vol, spot_vol, jump=(lam, mu, sd) or None, strikes, expiries)`: Black with σ(k) = σ_ATM + s·k + c·k², state advanced by `step(z_spot, z_vol, jump_draw, dt)`; re-priced each step; every Greek analytic and checked against finite differences. Inputs are past-measurable: the object never reads a future index (a mock that raises on future access is a test). |
| `flow.py` | `FlowParams(A, k, informed_frac, p_informed, h_info, persist=0.0)`; `Streams.draw(gen, n_steps, n_quotes)` pre-draws all streams; `arrivals(streams, step, quotes, fair)` thins candidates with `U < exp(−k·δ)` where δ is the $ distance of the quote from fair (δ_vol · Vega for vol quotes); informed direction = sign(F_{t+h} − F_t) with prob p, else a coin; uninformed 50/50 with optional Markov persistence; `P(fill in dt) = 1 − exp(−λ dt)`, never λ·dt. |
| `quoter.py` | `Quoter` interface `quotes(state, inventory, t) -> (bid, ask) per instrument`; `AS2008(gamma, k, sigma, T)`: r = s − qγσ²(T−t), spread = γσ²(T−t) + (2/γ)ln(1+γ/k); `GLFT2013(gamma, k, sigma, A, T, Q)` exact via `scipy.linalg.expm` of the (2Q+1) tridiagonal M (diag αq², off-diag −η, α = kγσ²/2, η = A(1+γ/k)^{−(1+k/γ)}), δ_b(q) = (1/k)ln(v_q/v_{q+1}) + (1/γ)ln(1+γ/k), δ_a(q) = (1/k)ln(v_q/v_{q−1}) + …; `GLFTAsymptotic(gamma, k, sigma, A)`: δ_b ≈ (1/γ)ln(1+γ/k) + ((2q+1)/2)·√(σ²γ/(2kA)·(1+γ/k)^{1+k/γ}), δ_a with (2q−1); `Space` = `price` or `vol` wrappers applying the skews of §7; matched initialisation helper `match_spreads(price_q, vol_q, fair)` so Σ vega·hs_vol = Σ hs_$ and the linearised $ skew per contract agrees at q = 0; tick rounding (`tick=0.01`, SPX 0.05), size rules, `max_loss` guard (pull quotes below −L), `vega_cap`. |
| `hedger.py` | `BandHedger(band_delta_usd, cost_model)` and `TimeHedger(every_s, cost_model)`; default `cost_model(shares, S, half_spread, Y, sigma_daily, adv)` = half_spread·|Δh| + Y·σ_daily·√(|Δh|/ADV)·S·|Δh| (tcakit sqrt-law form; Y is a labelled model constant); `band_ww(S, Gamma, lam, gamma, r, tau)` = (3/2·e^{−rτ}·λ·S·Γ²/γ)^{1/3} (Whalley–Wilmott 1997). |
| `pnl.py` | `Attribution` with the four top terms, the two sub-splits, the Greek layer with `RESID`, and `gap` (must be < 1e-9 — asserted in every sim run, not only in tests); `attribute(run, h=60.0)`. |
| `adverse.py` | `markouts(run, horizons=(1, 10, 60, 300)) -> DataFrame`: MO_h = s_f (F_{t_f+h} − px_f) = realised spread + adverse, in $ and vol points, by counterparty class, with time-bucket clustered SE; `toxicity(run)` per class. |
| `sim.py` | `SimConfig(seconds, dt=1.0, seed, ...)`; `run(config, quoter, hedger, fair, flow) -> Run` (fixed-step loop: quote update → fills → hedge check → state step); `paired(config, quoters: dict, seeds) -> PairedResult`; determinism: same seed and identical quoters → bit-identical fill sequences and R. |
| `report.py`, `cli.py` | `quotesim report` regenerates the README tables from fixed seeds: (1) the identity check over 100 seeds (max |gap|), (2) A-S 2008 Tables 1–3 reproduction (their parameters, 1,000 sims; direction of every effect and magnitudes within ~15 %), (3) the paired sweep γ × informed_frac ∈ {0, 0.1, 0.3} × band, decomposed by identity term, (4) the markout table by counterparty class ("post-trade review"), (5) the Whalley–Wilmott band frontier. `quotesim run --seed N` prints one attribution waterfall. |

Oracles (each cited in its test docstring, numbers computed independently during research):
- A-S: q = 0 → reservation = mid; γ→0 → spread → 2/k; A-S params (s=100, T=1, σ=2, γ=0.1, k=1.5): spread 1.690770 at t=0, 1.290770 at T, reservation shift 0.4 per unit q; FOC identity (1/γ)ln(1 − γλ/λ') = (1/γ)ln(1+γ/k) for exponential λ.
- GLFT (σ=0.3, A=0.9, k=0.3, γ=0.01, T=600, Q=30, t=0): exact δ_b(0) = 3.313045, δ_b(5) = 3.653479, δ_a(5) = 2.972522, δ_b(20) = 4.665542; asymptotic 3.312915 / 3.652244 / 2.973586 / 4.670232; |exact − asymptotic| ≤ 5e-3 for |q| ≤ 20; δ_b(q) = δ_a(−q) to 1e-12; terminal δ(T, q) = (1/γ)ln(1+γ/k) = 3.278982 for all q; Q = 30 vs 60 differ < 1e-8; spread-vs-γ is non-monotone (the dip).
- Stoikov–Sağlam eq. (17): Γ S² σ (T_mat − t) = Vega (S=K=100, σ=0.01/day, T_mat−t = 399 d: Γ 0.01987273, Vega 792.922); Thm 4 k = 0.03855879 (vega share 99.87 % at T_mat = 400 d; 0.49 % at 2 d); vol-space tilt per lot = γk/Vega = 4.8629e-6; premiums C/(2D) = 0.1, cap 0.2, ask 0.098072 / bid 0.105784 at q = 1, ask hits 0 at q = 26.434; Thm 5 recursion m_100 = −0.0038559 → m_0 = −0.0015080.
- Whalley–Wilmott: S=K=100, σ=0.2, T=0.25, r=0, λ=0.001, γ=1: Γ 0.03984439, H = 0.06198337; 8× λ → 2× H exactly.
- Identity: |gap| < 1e-9 on 100 seeds; sub-split gaps 0 exactly; zero-inventory quoter → INV = 0; RESID < 1e-5·|INV| at 1 s steps.
- Flow: acceptance rate = exp(−kδ) within a binomial CI; E[adverse_h] = −(2p−1)·σ_F·√h·√(2/π) for informed flow on a BM fair (σ_F = 0.05 $/√s, h = 60 s: E|ΔF| = 0.309019; p = 0.6: −0.061804; p = 1: −0.309019); φ = 0 → mean adverse 0 within SE; CRN: same seed → identical candidate streams.
- A-S 2008 Tables 1–3 (1,000 sims): inventory strategy lower P&L sd (6.6 vs 12.7 at γ=0.1), lower |q| sd (2.9 vs 8.4), lower mean at γ=1 (31.4 vs 44.0); direction asserted, magnitudes within ~15 %.

Deferred (v0.2): `book.py` L2/FIFO with a latency parameter (nanolob semantics recorded in DESIGN.md); OFI / VPIN toxicity on real prints; quote-reactivity to flow tested against a size-matched random-widening null at equal duty cycle; the note "inventory skew in vol space vs price space" with a pre-registered primary metric (terminal vega variance at matched spread capture), ≥ 200 paired seeds sized from the v0.1 SE, the γ frontier and the informed panel, a null reported if found; volsurf adapter.
