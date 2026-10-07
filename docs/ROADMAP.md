# Expansion roadmap: structured products → numerical improvements → Heston

The project extends an existing single-underlying Rust/Python Monte Carlo library into a workspace for researching structured notes and comparing numerical methods and model assumptions. The first three milestones deliver products, numerical methods, and stochastic volatility while retaining existing calling conventions. The Rust test baseline before the expansion was 20 tests. Milestone 4 adds optional market validation and Heston calibration outside the core pricing engine.

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

Independent core/QuantLib checks produce 126 MC records and 14 summaries: 12 fall within sampling error and two are explicitly classified as insufficient tail sampling. This is not an all-passed numerical claim, and the tail outcomes alone do not establish a code defect. Maximum Rust closed-form/QuantLib error is `2.6148e-12`. The 114 IPA vendor points use `SDK_example_convention`, an explicit percent/European-Black assumption with `units_verified_by_response=false` and `independent_reference=false`; they are diagnostic overlays rather than independent references. Full artifacts stay local under `artifacts/market_validation/lseg-20261006-validated/`.

## Shared acceptance checks and future work

All four milestones are implemented and accepted. The latest verification passes 71 Rust tests, 27 offline market unit tests (including real-data Heston references and MC, duplicate seeds, evidence provenance, and thin-expiry handling), existing/new Python smoke tests, formatting, and default/Python-feature Clippy. The curated fixtures contain six real LSEG cases and twelve synthetic reference cases. Both Rust examples run. The same abi3 extension passed local extension tests on CPython 3.10, 3.12, and 3.14, with CI covering 3.9–3.14; CI additionally runs market regressions and notebooks on Python 3.12. All four notebooks have 18 executed code cells and nine saved PNG outputs, passing both kernel-free execution and fresh Jupyter kernels.

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

# Execute all four notebooks and save fresh outputs without changing source cells.
python tools/run_notebooks.py --write
# Restricted environments can use: python tools/run_notebooks.py --kernel-free --write
```

Each milestone delivers code, Python interfaces, numerical/cashflow tests, and executable examples. Notebooks retain fixed seeds and record path counts, parameters, prices, standard errors, and timings. Charts compare error and convergence; they do not promise vanilla-level gamma precision for discontinuous notes.

Future candidates remain available after these milestones:

- [ ] Calendar roll-down theta, historical knock-in state, and paid/outstanding coupons for outstanding notes.
- [ ] Heston-specific Greeks and further calibration extensions beyond milestone 4.
- [ ] Multi-asset path interfaces and basket/worst-of products.
- [ ] Longstaff–Schwartz regression and optimal-exercise engines for American/Bermudan products.
- [ ] Classic exotics such as digital, forward-start, and range-accrual products.
