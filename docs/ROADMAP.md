# Expansion roadmap: structured products → market surfaces → model risk

The project extends an existing single-underlying Rust/Python Monte Carlo library into a workspace for researching structured notes and comparing numerical methods and model assumptions. The first three milestones deliver products, numerical methods, and stochastic volatility while retaining existing calling conventions. The Rust test baseline before the expansion was 20 tests. Milestone 4 adds optional market validation and Heston calibration outside the core pricing engine.

Milestones 1–3 are implemented. Milestone 4 has a reviewed market-data run and offline regressions; typed supplier-surface verification remains pending account permission. Milestone 5 is implemented in the optional Python market module, with dated market replay, independent numerical tests, and an executed notebook. **Milestones 6–9 below remain planned work, not available features.** Their interface names, CLI examples, and artifact names are proposed contracts to refine during implementation.

The next objective is to connect dated option quotes, a validated volatility surface, outstanding-note valuation, and portfolio risk. The next implementation objective is milestone 6. The current supplier-permission limitation is tracked separately: a surface fitted to eligible option Close prices and dated carry inputs is a locally constructed surface, with its own provenance.

| Milestone | Deliverable | Dependencies and order |
|---|---|---|
| 5 | Quote-derived SSVI surface and calibration diagnostics | Implemented; reuses milestone 4 snapshots, validation, and independent pricing |
| 6 | Curves, actual-date schedules, and outstanding-note state | Follow 5; curve/time-grid infrastructure may proceed alongside surface work |
| 7A | Dupire local-volatility pricing and model comparisons | Requires 5 and the curve/time-grid layer in 6 |
| 7B | Heston QE-M simulation | Independent numerical branch; can proceed alongside 6 without waiting for 7A |
| 8 | Portfolio valuation, scenarios, and explicit risk conventions | Requires 6; add model comparisons as 7 becomes available |
| 9 | Correlated multi-asset GBM and worst-of Phoenix | Requires the dated cashflow layer in 6; integrate with the portfolio layer in 8 |

The completed surface implementation slice is milestone 5: a versioned `VolSurface`, an offline `surface-fit` command, constraint/residual reports, independent regressions, and an executable volatility-surface notebook. Each subsequent milestone has its own acceptance gate; passing tests does not turn an unavailable external entitlement into a completed task.

## Milestone 1: Snowball and Phoenix

### Features and interfaces

- [x] Add `SnowballNote` and `PhoenixNote`, implementing the existing `Payoff` trait.
- [x] Extract shared observation-index validation while preserving the existing `AutocallableNote` convention that permits index 0.
- [x] Add Python `price_snowball`, `price_phoenix`, `greeks_snowball`, and `greeks_phoenix` methods.
- [x] Add the [structured notes notebook](../notebooks/structured_notes.ipynb), comparing the effects of barriers, coupons, and volatility on prices and sensitivities.
- [x] Complete Rust/Python tests, formatting, Clippy, and example acceptance checks.

### Teaching contract conventions

Both products have one underlying and fixed barriers expressed as absolute prices. `notional` is principal; `reference_spot` is the original contractual reference fixing, independent of the model's current `spot`, and stays fixed when computing Greeks. All new contract parameters must be finite; principal, reference fixing, and barriers must be strictly positive, and coupons must be non-negative.

Knock-in monitors every simulation point, including index 0; `S <= knock_in_barrier` counts as a hit. Knock-out observes only its specified dates; `S >= knock_out_barrier` triggers redemption. Coupon eligibility is `S >= coupon_barrier`. Future observation schedules must be non-empty, strictly increasing, and within `1..=steps`; the last observation need not coincide with maturity. Maturity is always the end of the simulated path. Each cashflow is discounted at its actual payment time, and knock-out terminates the contract immediately.

| Snowball outcome | Cashflow |
|---|---|
| First knock-out on a specified date | Pay `N × (1 + coupon_rate × elapsed_years)` that day, even after an earlier knock-in |
| No knock-out and no historical knock-in | Pay `N × (1 + coupon_rate × T)` at maturity |
| No knock-out after a historical knock-in | Pay `N × min(S_T / reference_spot, 1)` at maturity, with no coupon |

`coupon_rate` is an annualised decimal rate: `0.12` means 12%. After a knock-in, recovery above the reference fixing returns only principal in this contract variant.

| Phoenix outcome | Cashflow |
|---|---|
| Eligible coupon observation | Pay `coupon_per_period`; with memory enabled, also pay previously missed coupons |
| Ineligible coupon observation | Pay nothing; with memory enabled, add the missed period to outstanding coupon memory |
| Knock-out on a specified date | Pay any eligible same-day coupon first, then return `N` and terminate |
| No knock-out and no historical knock-in | Return `N` at maturity; any maturity coupon follows its independent eligibility condition |
| No knock-out after a historical knock-in | Maturity principal is `N × min(S_T / reference_spot, 1)`; previously paid and eligible same-day coupons remain independent of the principal loss |
| Outstanding coupon memory at termination | Expire unpaid memory at maturity or early redemption |

`coupon_per_period` is a cash amount per period; `memory=False` is the default. `coupon_indices` and `autocall_indices` are independent schedules, so an autocall date need not be a coupon date.

```python
dates = [63, 126, 189, 252]
engine.price_snowball(model, 100.0, 100.0, 0.12, 70.0, 103.0, dates)
engine.price_phoenix(model, 100.0, 100.0, 2.0, 80.0, 60.0,
                     105.0, dates, dates, memory=True)
# greeks_snowball / greeks_phoenix use the same contract parameters and require GBM.
```

### Acceptance criteria

Explicit paths cover early knock-out, recovery after an interim knock-in, maturity losses, barrier equality, initial knock-in, independent schedules, coupon-memory catch-up and expiry, coupon settlement before same-day knock-out, and cashflow discounting. Fixed-`reference_spot` tests verify that a valuation-spot change does not reset the loss reference. Invalid parameters fail before simulation, and Python raises `ValueError`.

## Milestone 2: Asian and continuous-barrier numerical improvements

### Features and interfaces

- [x] Add `ScheduledAsianOption`: average only explicit observation indices and settle at model maturity.
- [x] Add `geometric_asian_price` and `GeometricAsianControl::{new, try_new}`, providing a closed-form geometric-Asian expectation on the same discrete fixing set.
- [x] Add default `ControlVariate::validate(steps)`; the engine validates the control schedule before simulation.
- [x] Add Python `price_asian_scheduled(..., control="none")`, supporting `none`, `european`, and `geometric`; retain the meaning of the European control in existing `price_asian(..., control=True)` calls.
- [x] Add `McEngine::{price_continuous_barrier, try_price_continuous_barrier}` and matching Python methods, covering up/down × in/out × call/put.
- [x] Add the `try_bump_and_revalue_with` repricing callback, continuous-barrier Greeks, and controlled scheduled-Asian Greeks.
- [x] Add the [numerical methods notebook](../notebooks/numerical_methods.ipynb), recording parameters, seeds, path counts, standard errors, and timings.
- [x] Complete Rust/Python tests, formatting, Clippy, and example acceptance checks.

### Numerical conventions

Scheduled Asians exclude index 0; the arithmetic average and geometric control use the same fixing dates and strike. When refining a grid, callers must scale the observation indices to preserve actual fixing dates. Geometric averages are calculated in log space, and the analytic expectation includes the actual fixing times, dividend yield, and discounting to maturity. Reconstruct a control when changing the model after control construction.

Continuous single barriers use conditional Brownian-bridge crossing probabilities under GBM, integrating out unobserved barrier hits between simulated fixings. Rebates for both in and out variants are paid at maturity. Equality and an initial barrier hit count as hits; zero volatility uses a deterministic path. This entry point supports GBM only; structured-note knock-in keeps its contractual discrete monitoring. [Method reference](https://arxiv.org/abs/0904.1157)

Greeks validate positive, finite bumps and every perturbed model. Downward volatility bumps remain clamped at zero. Every continuous-barrier leg rebuilds its crossing weights, and every controlled leg rebuilds its analytic expectation. Existing `theta` is **`dV/dT`** with fixed step counts and indices, scaling all observation times proportionally with maturity; it is not calendar roll-down theta for a fixed-date contract. Vega and rho are per unit of volatility and interest rate.

```python
engine.price_asian_scheduled(model, "call", 100.0, dates, control="geometric")
engine.greeks_asian_scheduled(model, "call", 100.0, dates, control="geometric")
fe.geometric_asian_price(model, "call", 100.0, dates)
engine.price_continuous_barrier(model, "call", "up_and_out", 100.0, 130.0)
engine.greeks_continuous_barrier(model, "call", "up_and_out", 100.0, 130.0)
```

### Acceptance criteria

Geometric Asians compare Monte Carlo with the analytic expectation and check the single-terminal-fixing European limit, zero volatility, and put-call parity. Observation schedules preserve the contract definition. In a fixed scenario, the geometric control produces a smaller standard error than either no control or a European control, while prices remain statistically consistent.

Continuous barriers check in/out parity for all four barrier kinds, initial hits, zero volatility, rebate discounting, and independent analytic prices. Continuous knock-outs must not exceed discrete knock-outs on the same simulated paths. Greek tests cover invalid bumps, perturbed models outside their valid domains, refreshed control expectations, and the volatility sensitivity of continuous crossing probabilities.

## Milestone 3: Heston stochastic volatility

### Features and interfaces

- [x] Add default `PathGenerator::noise_dim() = steps()` and `validate()` methods; separate random-input dimensions from the number of time steps.
- [x] Add `HestonModel::{new, try_new}`, supporting spot, r, q, initial variance, kappa, long-run variance theta, volatility of variance, and correlation.
- [x] Add Python `HestonModel`; all uncontrolled pricing methods accept GBM or Heston.
- [x] Reject Heston in GBM-specific controls, continuous barriers, and Greeks, raising an explicit `ValueError` in Python.
- [x] Add the [Heston notebook](../notebooks/heston.ipynb), comparing both models and different simulation step counts.
- [x] Complete Rust/Python tests, formatting, Clippy, and example acceptance checks.

### Model conventions

The single-underlying output remains `steps + 1` prices. Heston uses `2 × steps` noise values, interleaving stock noise and independent variance noise; its variance driver is `rho × z_stock + sqrt(1-rho²) × z_independent`. Stock uses log-Euler and variance uses full-truncation Euler: preserve raw variance and use its positive part only in the drift and diffusion terms, rather than projecting the state to zero after each step. [Algorithm reference](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=903116)

Initial variance, long-run variance, and xi may be zero; kappa is positive, rho belongs to `[-1, 1]`, and all numeric parameters must be finite. The Feller condition is not a hard admission constraint. GBM retains bitwise compatibility for its random stream, prices, and standard errors.

```python
heston = fe.HestonModel(spot=100, r=0.03, q=0.0, v0=0.04,
                        kappa=1.5, theta=0.04, xi=0.4,
                        rho=-0.7, t=1.0, steps=252)
engine.price_european(heston, "call", 100.0)
engine.price_snowball(heston, 100.0, 100.0, 0.12, 70.0, 103.0, dates)
```

### Acceptance criteria

Tests cover parameter boundaries, subsequent evolution after negative raw variance, expected discounted stock prices, bitwise serial/parallel agreement, the constant-variance GBM limit, and compatibility with existing GBM results. A test-only Heston characteristic-function integral provides semi-analytic vanilla reference prices, and comparisons at several step counts distinguish Monte Carlo sampling error from time-discretisation differences. [Semi-analytic pricing reference](https://doi.org/10.1093/rfs/6.2.327)

Heston confidence intervals measure sampling error only, excluding time-discretisation bias.

## Milestone 4: Market validation and Heston calibration

### Goal and boundaries

Add an optional Python market-research workflow for market-data access diagnostics, reproducible quote snapshots, independent vanilla validation, and Heston calibration. LSEG/Refinitiv connectivity and instrument entitlements are measured on the actual account. Desktop-session access, the SPX close for 2026-10-06, SPXW Search discovery, native daily option history, and a historical IPA surface with discount/forward inputs have succeeded. Option `TR.PriceClose` was unavailable, so historical option closes use native `historical_pricing.summaries` daily `TRDPRC_1` as a dated fallback. A complete real-data snapshot and eight-start Heston fit have now been reviewed; this dated capture does not establish every instrument's entitlement or real-time data access.

Use QuantLib's analytic/semi-analytic vanilla engines as the independent validation and calibration reference. The existing Rust Monte Carlo engine remains a separate model comparison; its sampling confidence interval is not a calibration tolerance or a measure of Heston discretisation error. Market dependencies remain optional, and saved snapshots support offline replay without a live LSEG session.

### Features and interfaces

- [x] Add `python -m tools.market_validation doctor|fetch|validate|calibrate|report|run`, with `--as-of YYYY-MM-DD`, `--snapshot PATH`, `--curves CSV`, and `--output DIR`.
- [x] Probe desktop-session access using `LSEG_APP_KEY`, recording successful requests, unavailable capabilities, and permission failures without exposing credentials.
- [x] Save raw and normalised SPXW Close quote snapshots with instrument identifiers, observation timestamps, as-of dates, source metadata, and filtering decisions; require the requested/returned date to match and distinguish actual market quotes from synthetic regression fixtures.
- [x] Validate vanilla conventions and inputs against independent QuantLib results; record price differences and core-Monte-Carlo sampling errors separately.
- [x] Add Python `fe.black_scholes_price(model, option_type, strike)`, exposing the existing Rust closed-form formula for independent QuantLib regression checks; it validates the GBM model and strike and rejects Heston.
- [x] Add Heston-only European conditional pricing in Rust and Python (`price_heston_conditional`), with exact mixture likelihoods, default variance shifts `[0, -4, 4]`, and `[0]` for pure conditional Monte Carlo. It shares the existing log-Euler/full-truncation raw-variance scheme.
- [x] Preserve ordinary MC results and automatically recheck zero or imprecise Heston tail estimates with conditional importance sampling, independently tightened QuantLib quadrature, and adaptive grid/path refinements. Automatic grid additions are capped at 16,384 steps.
- [x] Reject legacy SDK unit/convention assumptions; supplier overlays require typed volatility, date, European/Black model, underlying-price provenance, and carry/premium-identity evidence. A separately retrieved typed dataset does not verify the old raw matrix's values.
- [ ] Obtain the required IPA FinancialContracts content permission and verify a dated typed supplier surface on this account. The current request is access-denied; the report excludes all unverified supplier values.
- [x] Calibrate Heston vanilla parameters using eligible quotes and explicit dated discount-factor/forward inputs (CSV: `as_of,expiry,discount_factor,forward,source`), preserving residuals, parameter bounds, convergence status, and excluded-quote reasons. A fetched snapshot without curves records `needs_curves`; validation and calibration require curves. Thin expiries are recorded and excluded; calibration needs at least three qualifying expiries with five training OTM strike groups each.
- [x] Produce reports and deterministic offline snapshot regressions; a failed or unentitled live request is not reported as a successful surface download.
- [x] Translate and refresh all four worked notebooks while preserving their existing demonstrations; execute all 18 code cells, retain nine PNG outputs, and verify the generated charts. Both kernel-free execution and fresh Jupyter kernels pass.
- [x] Complete and review the full real SPXW snapshot/Heston fit, then freeze the curated [LSEG reference fixture](../tests/fixtures/market/lseg_spxw_20261006_reference.json) with the manual regression command; CI reads its expected values without regenerating them.
- [x] Document installation, command usage, curve/snapshot formats, and offline replay in [MARKET_VALIDATION.md](MARKET_VALIDATION.md).

### Acceptance criteria

The doctor records the observed session and permission status; live access is reported as available only for requests that actually succeed. Snapshots preserve the provenance needed to replay validation and calibration without network access. Offline regressions check quote filtering, curve/date handling, independent vanilla prices, calibration outputs, and explicit handling of empty or unusable datasets. Reports identify their source as live, saved market data, or synthetic fixtures.

Validation separates bid/ask discrepancies, model residuals, Monte Carlo sampling uncertainty, and Heston time-discretisation effects. Calibration uses the independent QuantLib pricing path and reports fit quality and convergence rather than claiming a uniquely identified parameter set. Search expiry date values are preserved as source metadata, with `expiry_utc=null` and timestamp status `not_verified` until an actual expiry time is independently established; both pricing paths consistently use date-based ACT/365F. Replay copies and verifies raw-response sidecars; missing evidence is explicitly marked rather than leaving a dangling file reference. All four notebooks pass execution of their 18 code cells with saved outputs; existing Rust/Python checks remain green. Details and exact CLI examples belong to the [market validation guide](MARKET_VALIDATION.md).

### Reviewed market run

The 2026-10-06 SPXW capture contains 74 quotes across six expiries: 72 pass validation and two are rejected by the moneyness window. Five expiries qualify for calibration; 2027-09-30 is recorded as excluded because it has only one training OTM strike group. The selected fit from eight starts converges to `v0=0.00950145813`, `kappa=8.59859584`, `theta=0.03731037892`, `xi=1.64207484`, and `rho=-0.61857426`. Training price RMSE is 3.53048 index points over 48 targets; holdout RMSE is 4.14810 over ten targets, with holdout IV MAE of 239.6 basis points. These are market-fit residuals under the recorded inputs.

The updated independent core/QuantLib comparison contains 150 MC records and 14 summaries, all within the declared `4 × SE + 1e-12 + abs(reference) × 1e-8` sampling tolerance. Conditional importance sampling resolves the two originally missed positive Heston tails at 4,096 steps, 100,000 paths per seed, and seeds 42/43/44; their ordinary zero estimates and insufficient-tail classifications remain in the report. Combined standard error divided by the independent reference price is 0.453% for the 6,300 put and 6.371% for the 8,600 call. Both are below the 10% tail-precision limit. No additional path-budget or 16,384-step refinement was needed. These checks still exclude Heston time-discretisation bias from the sampling interval. Maximum Rust closed-form/QuantLib error is `2.6148e-12`.

All 114 untyped IPA matrix points remain raw evidence and are rejected for overlays. Typed FinancialContracts verification is blocked by this account's content access; alternative numerical IV fields lack explicit unit/model/date evidence. There are zero verified supplier points and no vendor overlay. Strict rejection is implemented, but actual supplier-unit verification remains pending permission. Full artifacts stay local under `artifacts/market_validation/lseg-20261006-validated/`; the replay's recorded load/validation/fit/numerical stages took 32.2 seconds before report generation.

## Milestone 5: Quote-derived volatility surfaces and calibration diagnostics

### Goal and scope

Turn the current total-variance display into a reusable, constrained market-surface module. Reuse the eligible SPXW Close quotes, explicit dated discount factors and forwards, source evidence, and filtering decisions from milestone 4. Keep locally inverted Black IV, fitted surface IV, analytic Heston IV, and any future verified supplier IV separately labelled.

The reviewed Heston holdout price RMSE of 4.14810 points and IV MAE of 239.6 basis points motivate additional diagnostics. Investigate quotation conventions, coverage, weighting, carry assumptions, and model flexibility before attributing all residuals to one cause. Heston Monte Carlo scheme changes affect numerical pricing; the existing calibration uses an analytic engine.

Start with SSVI using shared `rho` and `eta`, fitted ATM total-variance nodes `theta(T)`, and the modified-power skew function `phi(theta) = eta / sqrt(theta × (1 + theta))` (fixed exponent `gamma = 1/2`). Require `abs(rho) < 1`, `eta >= 0`, `eta × (1 + abs(rho)) <= 2 - epsilon`, and positive non-decreasing nodes, with a recorded positive constraint margin `epsilon`. This deliberately restricted first variant uses sufficient static no-arbitrage conditions from [Gatheral and Jacquier, Corollary 4.1 and equation 4.5](https://arxiv.org/abs/1204.0646). Treat `eta = 0` as the flat-smile limit. Dense numerical scans supplement analytic constraints and wing-limit checks.

### Features and interfaces

- [x] Add an optional Python market-tool `VolSurface` object with `total_variance(k, t)`, `implied_vol(strike, expiry)`, `black_price(option_type, strike, expiry)`, `derivatives(k, t)`, and `diagnostics()` methods. Use `k = log(K / F(T))`, `w = IV² × T`, decimal annualised IV, and ACT/365F. Keep ATM total variance distinct from the Heston parameter named `theta`.
- [x] Supply `D(T)` and `F(T)` for every strike/expiry query using exact dated carry nodes or explicitly recorded log-linear interpolation within their supplied range. Reject missing carry coverage, including at interpolated surface maturities. This bounded Python query support precedes the reusable Rust/context curve layer in milestone 6; `total_variance(k, t)` itself needs no carry conversion.
- [x] Fit in price space using fixed pre-fit uncertainty weights and one OTM representative per expiry/strike group. Initially reuse the Heston coverage gate: at least three qualifying expiries with five training OTM strike groups each; record thin expiries and reasons for an insufficient-data result. Retain dated bid/ask evidence when available; clearly identify the existing proxy uncertainty for Close-only observations. Preserve raw quotes when constrained fitting leaves residuals.
- [x] Report residuals by expiry, strike, moneyness, and train/holdout group; include uncertainty assumptions, bound activity, all seeded starts, convergence status, and parameter dispersion. Parameter dispersion across starts is a diagnostic, not a confidence interval.
- [x] Add curve/weight sensitivity runs and a deterministic strike-group holdout. Fit ATM nodes, initial guesses, and tuning choices from training quotes only; held-out ATM quotes cannot set a node. Where coverage permits, also leave out an entire interior expiry and remove its fitted node to assess maturity interpolation. Apply the same eligible targets, weights, and split when comparing SSVI and Heston. Select starts using training results only.
- [x] Linearly interpolate monotone ATM total-variance nodes between fitted expiries, then evaluate SSVI; retain analytic strike derivatives and explicit one-sided time derivatives at knots. Record observed coverage separately from the fitted domain. Default pricing queries require supported maturities and observed moneyness coverage. Explicit `allow_wing=True` permits model wings and marks query/prediction diagnostics as extrapolation; short/long-tenor extension is disabled in this version. A smooth time interpolation suitable for Dupire is a separate milestone 7A task.
- [x] Version `surface.json`: include snapshot/curve hashes, as-of date, units, model/parameterisation, knots and parameters, supported domain, interpolation/extrapolation policy, tolerances, fitting configuration, split, convergence, and dependency versions. Preserve the existing snapshot schema and CLI commands.
- [x] Add `surface-fit` and an opt-in report surface selector. Save `surface.json`, `surface_diagnostics.json`, `surface_residuals.csv`, and plots. Distinguish `converged`, `incomplete`, `failed`, and `insufficient_data`; a requested failed fit returns non-zero while retaining evidence.

Implemented offline command in `tools/market_validation/__main__.py`:

```sh
python -m tools.market_validation surface-fit \
  --snapshot path/to/snapshot.json \
  --surface-model ssvi --starts 8 --seed 42 --max-iter 2000 \
  --output artifacts/ssvi_run
```

The surface calculations live in `tools/market_validation/surface.py`; residuals integrate with the existing calibration/report modules, and optimisation dependencies stay outside the Rust core. A Rust surface evaluator is deferred until milestone 7A needs a validated pricing representation. Per-expiry correlation, a fitted power exponent, and eSSVI are later candidates after the restricted SSVI baseline is accepted.

### Acceptance criteria

- A flat-volatility surface recovers independent Black call/put prices and put-call parity; finite values, invalid dates/strikes, and interpolation boundary behaviour are covered. Check analytic surface derivatives against independent finite differences at well-conditioned points.
- Constrained synthetic cases satisfy the specified analytic inequalities with recorded numerical margins. Independent price/density checks cover monotonicity, convexity, calendar consistency at fixed forward moneyness, and both strike wings. Invalid surface artifacts with crossed slices or negative density are rejected. Arbitrage-inconsistent raw quotes can produce a valid constrained fit with nonzero residuals; preserve those conflicts, or record an explicit failed fit when constraints cannot be satisfied numerically.
- A fixed market replay preserves target eligibility and holdout membership, produces repeatable metrics within declared numerical tolerances, and reports price/IV errors alongside Heston. Sparse or conflicting inputs retain exclusions and fitting residuals. Market-fit quality is reported separately from implementation correctness.
- Freeze independent formula/price references and curated market replay checks. Price → IV → price round trips and synthetic data generated solely by the new implementation are supplementary checks, not independent numerical validation. CI reads reviewed fixtures without regenerating their oracles.
- Add and execute [`notebooks/volatility_surface.ipynb`](../notebooks/volatility_surface.ipynb): observations, fitted smiles, total variance, constraint margins, holdout residuals, and carry sensitivity. Record data date/hash, seeds, fit settings, timings, and coverage. Use saved data in CI.

### First-slice gate

- [x] Establish synthetic constraint/reference cases and deterministic target splits before choosing parameters on the real snapshot.
- [x] Implement the evaluator, constrained fit, versioned serialization, CLI, and report artifacts.
- [x] Review both real snapshot fits. Freeze independent QuantLib SVI/Black references using the specified numerical tolerances; report market residuals independently, without requiring SSVI to beat Heston or choosing fit-quality pass limits retrospectively.
- [x] Pass offline regression and notebook execution, document the APIs and limitations, then begin downstream surface-driven pricing.

### Reviewed implementation and evidence

The implementation branch is `feature/vol-surface`. The evaluator, SLSQP fitter, versioned artifacts, offline CLI, HTML/CSV diagnostics, and optional `report/run --surface-model ssvi` are available. Defaults are eight starts, seed 42, 2,000 iterations and `ftol=1e-10`; the first start is `rho=-0.7, eta=0.5`, with training-only isotonic ATM-node scaling. Every fit retains all starts and independent final constraint checks. Interior-expiry folds and six sensitivity scenarios are separate from primary strike holdout and start selection.

The latest live LSEG capture returned 2026-10-08, with SPX Close 7,765.36, 87 quotes, seven same-date carry expiries, and 84 accepted quotes. The 2026-10-06 capture is retained. Full redacted request/response evidence remains local; both curated quote/carry fixtures carry parent-capture hashes and replay offline. Latest capture and dated carry are complete. Typed supplier-volatility verification remains a separate milestone 4 entitlement item.

| Snapshot | Qualifying expiries | Train / holdout groups | SSVI price RMSE: train / holdout | Holdout IV MAE |
|---|---:|---:|---:|---:|
| 2026-10-06 | 5 | 48 / 10 | 2.61958 / 1.93467 index points | 222.79 bp |
| 2026-10-08 | 6 | 58 / 12 | 2.69105 / 6.14531 index points | 119.87 bp |

Both default fits, all interior-expiry folds, and risk-free/dividend ±1 bp scenarios converge. The 0.5%/2% Close-IV uncertainty scenarios preserve the historical-spread weights present in these two captures, so their parameters are unchanged. These results measure market fit, rather than formula accuracy. See [the market guide](MARKET_VALIDATION.md) for commands, artifacts, and limitations.

The accepted checks are 67 offline market tests (25 new SSVI tests), 79 Rust tests, Python binding/extension/conditional smoke checks, formatting, and default/Python-feature Clippy. Independent QuantLib SVI references use total-variance `atol=1e-12, rtol=1e-10`; Black price references use `atol=1e-9, rtol=1e-10`. Analytic derivative checks use relative tolerances of `1e-6` for first derivatives and `1e-4` for second derivatives at well-conditioned points. All five notebooks run in fresh Jupyter kernels: 23 code cells, 13 saved PNG outputs, with charts inspected. A Rust surface evaluator, smooth Dupire input, Local Vol, and wheel layout changes remain outside this milestone.

## Milestone 6: Curves, calendar schedules, and outstanding notes

### Goal and scope

Price a Snowball or Phoenix after issuance using its original contractual fixings, historical observations, accrued coupon terms, and remaining cashflows. Introduce a dated valuation layer with consistent discount and forward curves. Existing uniform-grid, constant-rate contracts and Python calls retain their current semantics.

This requires explicit interfaces: `PathGenerator` currently exposes a uniform grid and scalar rate, and `Payoff::evaluate(path, dt, r)` returns an already-discounted PV. Truncating a schedule would reset knock-in/memory state and Snowball accrual. Curve-aware pricing must generate dated cashflows and discount each payment once.

### Features and proposed interfaces

- [ ] Add `DiscountCurve` and `ForwardCurve`, with `D(0) = 1` and `F(0) = spot`, positive finite nodes, matching as-of/currency metadata, log-linear interpolation, and explicit coverage. Positive discount factors greater than one remain valid for negative rates. Require curve support for all simulation and payment times.
- [ ] Use `log(F(t_next) / F(t))` for the deterministic log-stock carry increment, with the model's variance correction. Discount a cashflow at `D(payment_time)`. Flat curves reproduce existing flat-rate models; term-structure pricing uses one consistent carry trajectory rather than a different flat rate for each payment.
- [ ] Add `TimeGrid` and `EventSchedule`, with actual dates, ACT/365F times, and separate fixing/payment dates. The grid contains all future contractual observation times and maturity; numerical refinement preserves them. Keep different events on the same date while storing grid times uniquely. Supplied schedules record their calendar source; automatic holiday adjustment is deferred until a calendar and business-day convention are explicitly configured.
- [ ] Introduce additive `ContextPathGenerator`, `CashflowPayoff`, `Cashflow`, and `ValuationContext` abstractions plus a context-pricing entry point. Use explicit cashflow implementations for curve-aware notes; legacy discounted `Payoff` results remain on the existing entry point. Add flat-curve adapters where equivalent semantics can be established.
- [ ] Add `NoteState` and a deterministic history replay function. State records the original reference fixing, issue date, processed observation cutoff, knock-in status, redemption status, Phoenix coupon memory, and known unpaid receivables. Paid cashflows remain audit history and are excluded from current PV. Replay known fixings in order; missing required fixings produce an incomplete-state error. Contract history and model state remain separate: historical stock fixings alone do not specify the current Heston variance.
- [ ] Define an explicit before/after-fixing cutoff for valuation-day events, so history and simulation process each event exactly once. Matured or redeemed trades can still have known unpaid payments; value those deterministically without requesting a positive-maturity simulation.
- [ ] For new dated contracts, configure knock-in observation dates separately from numerical grid points. Preserve existing index-based grid monitoring. Process same-day events as knock-in, coupon eligibility/memory, autocall/principal, then settlement; retain the original equality rules. A valuation spot counts toward knock-in only when it is a contractual observation under the cutoff. Stop economic observations after redemption while retaining unpaid obligations. Snowball coupon accrual starts at the original issue date using the contract's accrual convention.
- [ ] Add Python `price_snowball_dated`, `price_phoenix_dated`, history replay, and `calendar_theta` entry points. Return valuation cutoff, state, future cashflow summary, PV, and sampling error; expose contract/state objects rather than extending positional signatures indefinitely.

### Calendar theta conventions

Keep existing `theta = dV/dT` unchanged. Define calendar theta as the change in current remaining-cashflow PV when the valuation date advances, with fixed contractual dates, and report the roll interval and units. Define the spot, curve rebasing, and surface-roll policy explicitly. Separate PV change from cash paid over the interval so users can reconcile total P&L.

An event-free roll can use a stated frozen-market policy. A roll across an observation needs realised fixings or a documented scenario for advancing state; the tool must not invent realised knock-in, autocall, or coupon outcomes.

### Acceptance criteria and delivery

- Flat curves and a uniform grid reproduce the existing pricing path and teaching contract cashflows. Non-flat-curve manual paths verify each payment's discount factor and forward-driven drift. Refined numerical grids leave contractual observation dates unchanged.
- History replay covers prior knock-in followed by recovery, missed memory coupons and later catch-up, previous redemption, same-day events, valuation-day cutoffs, payments pending after redemption, and fully settled trades. Replaying a full path and splitting that same path at an intermediate valuation date produce identical contractual cashflows and consistent PV after rebasing discount factors.
- Missing historical fixings, stale states, inconsistent references, duplicate history events, and payment dates before their fixings fail explicitly. New monitoring schedules remain independent of grid refinement.
- Calendar theta has deterministic event-free reference cases and explicit before/after-payment cases. Known paid amounts reconcile with the valuation change. Existing `dV/dT` tests continue to pass.
- [ ] Add and execute `notebooks/outstanding_notes.ipynb`; compare issue-date and mid-life valuations and inspect the historical/future cashflow ledger.
- [ ] Complete Rust/Python compatibility checks and document the new cutoff, calendar, accrual, and roll conventions before using dated notes in later milestones.

## Milestone 7: Model dynamics and simulation efficiency

### 7A: Dupire local volatility

Build a single-asset `LocalVolModel` from an accepted surface plus the milestone 6 curves and time grid. The surface describes European option marginals; a path-dependent price also requires a chosen model for dynamics. Model comparisons must record differences in vanilla fit as well as differences in dynamics.

Use the [original Dupire formulation](https://www.risk.net/sites/default/files/import_unmanaged/risk.net/data/Pay_per_view/risk/technical/1994/risk_0194_volatility.pdf). Local variance requires reliable time and strike derivatives and consistent carry. The current piecewise-linear display is unsuitable as the derivative input. Prepare a surface with suitable smoothness, explicit time-knot treatment, and validated short-time and wing extensions before path simulation.

- [ ] Add a versioned surface export and a Rust evaluator with analytic derivatives where available; retain the Python calibration layer. Python/Rust evaluation of the same artifact must agree within fixed numerical tolerances.
- [ ] Build a local-volatility grid or evaluator with explicit supported domain, interpolation rules, derivative tolerances, and short-time treatment. Record path excursions beyond the quote-supported region and the extension used. Materially negative local variance or unstable denominators reject the construction; any round-off handling is bounded and reported.
- [ ] Implement log-stock simulation using the shared forward curve and payment discounting context. Require a complete validated extension over reachable positive spot states for simulation; derivative clipping or a hidden fallback to constant volatility cannot define the pricing model.
- [ ] Add a Python `LocalVolModel` and support it through the new uncontrolled context-pricing interfaces. Existing GBM-specific controls, continuous-barrier weighting, and Greeks retain their explicit model restrictions until independently derived replacements are implemented.
- [ ] Compare GBM, calibrated Heston, and Local Vol for European, barrier, Snowball, and Phoenix examples using the same valuation inputs and contractual monitoring. Report each model's vanilla residuals, sampling errors, grid changes, and structured-note price differences.

Acceptance: recover the constant-volatility GBM limit; verify the forward-stock expectation and non-negative local variance on supported test domains; reproduce independently evaluated surface vanilla prices across strikes/tenors using refined MC and an independent finite-difference/PDE reference. Declare price and grid tolerances per benchmark before evaluating results. A model-comparison notebook must identify extrapolation exposure and distinguish sampling error, discretisation error, calibration residual, and model assumptions.

### 7B: Heston QE-M

Add an optional quadratic-exponential scheme with martingale correction, following [Andersen's simulation paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=946405) and cross-checking the [QuantLib implementation](https://github.com/lballabio/QuantLib/blob/master/ql/processes/hestonprocess.cpp). Measure time to a declared accuracy target against the current full-truncation scheme. Keep full truncation as the default and preserve its random-stream behaviour.

- [ ] Add explicit `HestonScheme::{FullTruncation, QeMartingale}` selection and a Python `scheme` option. Define noise consumption and normal-to-uniform conversion for both variance branches, including antithetic pairing and moment-existence/domain checks.
- [ ] Support parameter limits deliberately: zero initial/long-run variance, zero vol-of-vol, correlations at both endpoints, and parameter sets violating the Feller condition. Handle deterministic limits separately where the general formula degenerates.
- [ ] Retain the current full-truncation conditional estimator and importance-sampling likelihoods on their supported scheme. A QE-M conditional estimator requires a separate derivation and validation; reject unsupported scheme/estimator combinations explicitly.
- [ ] Compare independent analytic vanilla prices, forward-stock expectation, tail accuracy, and bias versus time step. Benchmark multiple seeds and both ordinary and difficult fitted parameter sets at matched error targets, recording wall time and environment.

Acceptance: bitwise serial/parallel reproducibility within each scheme; constant-variance/deterministic limits; valid variance states and both QE branches; no silent substitution on invalid moment conditions; statistically justified agreement with analytic references as grids refine. Preserve old full-truncation fixtures. Publish performance measurements without imposing noisy machine-specific timing thresholds in ordinary CI.

- [ ] Extend the Heston notebook with scheme comparisons and add `notebooks/model_comparison.ipynb` for 7A. Run all existing notebooks after integrating model-selection changes.

## Milestone 8: Portfolio valuation, scenarios, and risk conventions

### Goal and proposed interfaces

Value collections of dated trades against one immutable market snapshot and explain the changes under explicit scenarios. Start with a single currency and equity index. Counterparty/funding adjustments, cross-currency aggregation, and real-world return forecasting remain separate future work.

- [ ] Add `Portfolio`, `Trade`, `Scenario`, and `RiskResult` objects with stable trade IDs, signed quantities, contractual/state references, and market/surface hashes. Expose `price_portfolio`, `scenario_portfolio`, and `risk_portfolio` in Python plus offline report generation.
- [ ] Return trade and aggregate PV, cashflows by payment date, conditional autocall-date probabilities, knock-in probability, and contractual loss summaries. State clearly that simulated event probabilities are risk-neutral model quantities. Scenario P&L is a change in valuation, not a real-world VaR estimate.
- [ ] Support spot, GBM volatility, Heston-parameter, curve, surface-level, and skew scenarios with explicit units and transformation rules. Hold contractual references and realised history fixed under market shocks. When changing spot or rates, specify whether forwards change to preserve proportional carry or are independently shocked. Surface shocks must pass constraints or be rejected; optional constrained refitting records both the requested shock and actual fitted surface.
- [ ] Distinguish frozen-parameter/model sensitivities from market-quote bumps followed by recalibration. Reconstruct volatility-dependent calculations for every leg. Failed recalibration invalidates that risk result rather than silently reusing the base fit. Heston parameter sensitivities and quote sensitivities receive separate names.
- [ ] Use common random numbers for compatible scenario legs. Estimate the SE of differences and portfolio totals from joint path observations, including covariance and antithetic sampling units. Price each trade and aggregate the shared-path cashflows before estimating the total SE.
- [ ] Reuse paths only when models, time grids, market state, and stochastic inputs match; store those identifiers in cache keys. Adopt deterministic path-to-seed assignments and a fixed aggregation order so batch size and worker count do not alter results.

### Acceptance criteria and delivery

- One-trade portfolios reproduce standalone values; signed duplicate positions cancel pathwise; split positions recombine; contractual cashflow totals reconcile with PV. Common-path aggregation correctly accounts for covariance.
- Zero shocks produce zero paired P&L; simple European positions match independent analytic risk/scenario references. Bump-size sweeps expose unstable sensitivities for discontinuous payoffs instead of suggesting vanilla-level gamma accuracy.
- Constraint-breaking shocks, failed recalibration, inconsistent market dates, unsupported model/method combinations, and cache invalidation cases are covered by offline tests. Frozen-parameter and recalibrated results disclose their conventions and differ where expected.
- [ ] Add and execute `notebooks/portfolio_risk.ipynb` with several outstanding notes, scenario tables, cashflow ladders, risk conventions, and estimator uncertainty. Record seeds, paths, model scheme, grid, and market hashes.
- [ ] Document portfolio aggregation and Greek units; complete serial/parallel, Rust/Python, and offline report checks before adding multi-asset portfolios.

## Milestone 9: Correlated multi-asset GBM and worst-of Phoenix

### Goal and proposed interfaces

Extend structured-note research to a small basket of underlyings with an explicit first contract variant. Begin with deterministic carry curves and constant per-asset GBM volatility. Multi-asset Heston, stochastic correlation, quanto/FX effects, and cross-currency baskets are later candidates.

- [ ] Add `MultiAssetPathGenerator` and `MultiAssetPayoff`, separate from the existing scalar traits. Specify a time-major path layout with `assets × (steps + 1)` prices and independent-noise dimensions. Add `CorrelatedGbmModel` and Python model/basket entry points without changing single-asset buffers.
- [ ] Validate positive spots/reference fixings, per-asset inputs, a symmetric unit-diagonal positive-semidefinite correlation matrix, and dimensions. Support singular PSD matrices using an appropriate factorisation; report numerical tolerances and reject materially invalid matrices. Correlation shocks must remain admissible or use an explicitly reported projection.
- [ ] Add `WorstOfPhoenixNote` using `min_i(S_i / reference_i)` on each contractual fixing. Express barriers as dimensionless reference ratios. Coupon eligibility, autocall, and knock-in use that worst-of ratio, with the same equality, memory, and same-day settlement rules as the single-asset teaching variant.
- [ ] Record a historical knock-in if any scheduled worst-of fixing breaches its lower barrier. Without autocall, return principal if there was no knock-in; after knock-in, maturity principal is `N × min(min_i(S_i(T) / reference_i), 1)`. Keep coupon eligibility independent of principal loss and expire unpaid memory at termination.
- [ ] Retain all original reference fixings and historical state. Use one agreed settlement currency and discount curve for the first variant; define a shared observation calendar and complete multi-asset fixing coverage.
- [ ] Add per-asset spot scenarios and correlation sensitivity to portfolio reports. A partial correlation bump is a valid risk calculation only if the resulting full matrix passes validation.

### Acceptance criteria and delivery

- The one-asset case reduces to milestone 6's dated Phoenix with matching knock-in, coupon, autocall, and payment schedules after converting ratio barriers to absolute prices. Comparison with the legacy index-based Phoenix additionally requires its grid-monitoring set to match. Identical assets with perfect correlation reduce to the same dated case. Permuting assets and their manual paths preserves cashflows exactly; independently refactorised MC runs agree within a declared sampling tolerance, with no requirement for identical seeded samples across factorisations.
- Deterministic manual paths cover a change in the worst-performing asset, any-asset knock-in, all-asset autocall qualification, recovery, memory catch-up/expiry, same-day events, and discounting. Test singular matrices and rejected non-PSD examples.
- Correlated driver samples reproduce the intended covariance within a predeclared statistical tolerance, and each asset reproduces its forward expectation. A linear basket has an independent expected discounted payoff; serial/parallel runs remain reproducible.
- [ ] Add and execute `notebooks/worst_of_phoenix.ipynb`, showing asset count, correlation, and volatility scenarios with sampling errors and fixed contract dates.
- [ ] Complete multi-asset Rust/Python validation, document the contract cashflow branches, and integrate regression/report checks before expanding the model family.

## Shared acceptance checks and advancement rules

The recorded acceptance baseline before milestones 5–9 is 79 passing Rust tests, 42 passing offline market tests (including real-data Heston references, rare-tail resolution, strict supplier evidence, duplicate seeds, provenance, and thin-expiry handling), existing/new Python smoke tests, formatting, and default/Python-feature Clippy. The three conditional-pricing Python smoke groups pass on CPython 3.12 and 3.14. Numerical and supplier evidence is reported separately from test counts. The curated fixtures contain six real LSEG cases and twelve synthetic reference cases. Both Rust examples run. The same abi3 extension passed local extension tests on CPython 3.10, 3.12, and 3.14, with CI covering 3.9–3.14; CI additionally runs all `test_market*.py` regressions and notebooks on Python 3.12. All four notebooks have 18 executed code cells and nine saved PNG outputs, passing both kernel-free execution and fresh Jupyter kernels; the conditional-pricing update also passes a fresh kernel-free rerun.

```sh
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features python -- -D warnings
cargo test --release
cargo run --release --example price_options
cargo run --release --example advanced_pricing

# Build the Python extension in the project virtual environment.
maturin develop --release
python tests/test_bindings.py
python tests/test_extensions.py

# Execute all five notebooks and save fresh outputs without changing source cells.
python tools/run_notebooks.py --write
# Restricted environments can use: python tools/run_notebooks.py --kernel-free --write
```

Each milestone delivers code, Python interfaces, numerical/cashflow tests, and executable examples. Notebooks retain fixed seeds and record path counts, parameters, prices, standard errors, and timings. Charts compare error and convergence; they do not promise vanilla-level gamma precision for discontinuous notes.

For milestones 5–9, the following gates apply in addition to their product/model criteria:

- [ ] Keep existing Rust/Python calls and defaults compatible; document proposed APIs as implemented only after their code and examples land.
- [ ] Complete relevant Rust tests, binding smoke tests, default/Python-feature formatting and Clippy checks, and offline market regressions for each implementation slice. A pure Python surface change needs its market checks; it does not create a new Rust API automatically.
- [ ] Execute every checked-in notebook after shared API/tooling changes. Extend `tools/run_notebooks.py` and the Python 3.12 offline CI job when adding the planned notebooks; verify saved chart outputs as well as execution. Notebooks and CI use credential-free saved fixtures.
- [ ] Store dataset/curve/surface hashes, dependency versions, seeds, path counts, grids, scheme, tolerances, and stage status in reports. Keep calibration residuals, sampling errors, discretisation differences, and extrapolation/model assumptions separate.
- [ ] Review independent numerical references and fixture tolerances before freezing them. Optimiser parameter equality alone is not a stable numerical regression; verify prices, constraints, eligibility, and metrics with appropriate tolerances. Live refreshes are manual, dated, and preserve the previous fixture until reviewed.
- [ ] Promote an API only after its benchmark/contract conventions and unsupported combinations are documented. Complete the relevant gate before a dependent milestone consumes its outputs; independent 7B work can proceed separately.

Future candidates remain available after these milestones:

- [ ] Longstaff–Schwartz regression and optimal-exercise engines for American/Bermudan products.
- [ ] Classic exotics such as digital, forward-start, and range-accrual products.
- [ ] Automatic exchange-calendar construction, corporate actions, discrete cash dividends, and settlement conventions beyond the first dated-note variant.
- [ ] Multi-asset stochastic volatility, cross-currency/quanto contracts, and funding/counterparty adjustments.
