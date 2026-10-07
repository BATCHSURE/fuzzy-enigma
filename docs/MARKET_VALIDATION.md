# Market validation and Heston calibration

The optional Python tools connect quote snapshots, independent QuantLib vanilla prices, Heston calibration, and comparisons with the Rust Monte Carlo engine. The first release uses a fixed SPXW profile. Market access is an observed capability of your LSEG/Refinitiv account: a desktop session or application key does not establish option-chain, historical-option, or IPA permissions.

Desktop-session access, the SPX close for 2026-10-06, SPXW Search discovery, native daily option history, and historical IPA surface/discount/forward requests have succeeded. The option `TR.PriceClose` field was unavailable; native daily `TRDPRC_1` supplied a dated fallback. A full real-data snapshot and eight-start Heston fit have been reviewed. The tools preserve diagnostics for unavailable or unentitled requests. Saved market snapshots and clearly labelled synthetic fixtures can be replayed offline.

The verified native history probe for `SPXWj132677450.U` returned the 2026-10-06 Close of 94.77 index points, bid 93.4, and ask 98.1, with delayed quote quality of service. This is evidence for that request and date, not a claim of universal option or real-time entitlements.

The reviewed core comparison now resolves both previously missed Heston tails through conditional importance sampling. Supplier volatility remains unavailable: typed IPA FinancialContracts verification is access-denied on this account, and ambiguous legacy values are rejected. The saved report therefore has no vendor overlay.

## Install the optional tools

Use Python 3.12 for the market workflow. The core extension's broader abi3 interpreter support does not imply that every optional market package supports the same range.

From the repository root, create or activate a Python 3.12 market environment:

```sh
python3.12 -m venv .venv-market
source .venv-market/bin/activate
python -m pip install maturin
python -m pip install -e '.[market]'
maturin develop --release
python -m tools.market_validation --help
```

The `market` extra supplies the optional data, independent-pricing, optimisation, and charting dependencies. It is separate from the Rust pricing API. QuantLib and SciPy are used for independent vanilla calibration; the core Monte Carlo engine is used for subsequent numerical comparisons.

Live commands use the desktop LSEG session by default. Provide `LSEG_APP_KEY` through your shell environment. If it is unset, explicitly request a hidden prompt for the doctor command. Do not put keys in notebooks, committed files, snapshots, or command-line arguments.

## Check the actual account first

```sh
python -m tools.market_validation doctor --output artifacts/market_doctor

# Allow hidden application-key input when LSEG_APP_KEY is unset.
python -m tools.market_validation doctor --prompt --output artifacts/market_doctor
```

The doctor reports what it could actually observe: installed packages, session access, underlying data, and the outcome of option-data capability probes. A failure can mean unavailable desktop access, an invalid session/key, missing entitlements, or an unsupported request; use the saved error and request details to distinguish these cases. Do not infer option permissions from the underlying probe. A prompted application key is used only for that invocation and is never persisted.

Without `--prompt`, an unset key is a diagnostic result. Offline validation does not need a running desktop or `LSEG_APP_KEY`.

## Save a dated quote snapshot

```sh
python -m tools.market_validation fetch \
  --as-of 2026-10-06 \
  --curves my_curves.csv \
  --output artifacts/spxw_2026-10-06

# Optionally request typed supplier evidence when the account is entitled.
python -m tools.market_validation fetch \
  --as-of 2026-10-06 --verify-vendor-surface \
  --curves my_curves.csv \
  --output artifacts/spxw_verified_surface
```

`fetch` performs a live SPXW request and writes `snapshot.json`. It does not accept `--snapshot`. Without `--as-of`, it targets the last published complete close. This workflow uses **Close** prices; returned dates must match the requested as-of date. A missing or mismatched historical close is not silently substituted with today's price, the latest available quote, or a current volatility surface.

`--verify-vendor-surface` is an opt-in capability of `fetch` and live `run`. It requests typed supplier evidence and preserves permission/verification failures in capture diagnostics. It is rejected with `--snapshot`, so offline replay never initiates a verification request. An unsuccessful verification does not turn raw volatility values into an accepted overlay or prevent quote/curve evidence from being retained.

Option history first tries the dated `TR.PriceClose` endpoint, then uses native `historical_pricing.summaries` with the daily interval and `TRDPRC_1` for instruments missing there. Each quote records its close source, and raw diagnostics retain primary-endpoint failures and fallback results. Historical bid/ask values remain tied to the same actual observation date.

Keep the snapshot's source metadata, instrument identifiers, dates, raw response evidence, and filtering diagnostics together. Option contracts must be suitable for the European vanilla comparison. Reports preserve distinctions between actual market closes, model prices, and synthetic fixtures.

SPXW discovery uses the current Search index, then requires positive historical Close observations on the requested date. Snapshot metadata explicitly records `historical_survivorship_risk=True`: historical replay is not a reconstruction of the complete option universe that existed on that date. The source also records `delay_status="not_verified"`; a retrieved close does not establish the account's real-time quote entitlement. Contract terms are grounded in the verified SPXW family metadata and linked product specifications.

If quotes can be fetched but appropriate curves are absent, the snapshot is still saved with a `needs_curves` diagnostic. Validation and calibration require supplied discount/forward curves, either already embedded in the snapshot or loaded with `--curves`; fetching alone does not invent interest rates or dividend yields.

### Snapshot and quote fields

Snapshots are versioned JSON: `schema_version` is `1`, `source` records provenance, `spot` is the underlying close, `quotes` is a list of normalised observations, and `curves` contains the supplied dated inputs. `source.as_of` is an ISO date and `source.price_basis` is explicitly `"close"`. Output snapshots additionally record `dataset_sha256` and installed dependency versions so a report can identify its inputs and environment. Provider diagnostics and authentication fields are redacted before JSON is written.

Bulk provider evidence lives in the `raw_responses.json` sidecar, referenced by `source.raw_response_file` and its SHA-256 digest. Offline replay verifies the digest and copies the sidecar into the new output directory. If the original sidecar is absent, replay records `raw_response_status="missing_sidecar"`, preserves its original filename as provenance, and removes the active file reference. It does not create a dangling artifact link or pretend that missing raw evidence was verified.

Each normalised quote includes:

| Field | Meaning |
|---|---|
| `ric` | Unique contract identifier |
| `expiry` | ISO expiry date |
| `close_date` | Actual date of the closing-price observation; must equal `source.as_of` |
| `kind` | `call` or `put` |
| `strike` | Positive finite strike, in index points |
| `close` | Finite non-negative premium, in index points |
| `exercise_style` | `european` |
| `settlement` | `pm` |
| `currency` | `usd` |
| `premium_unit` | `index_points` |

The adapter preserves `expiry_source_value` and `search_metadata`. Search's midnight expiry value is date metadata, not a verified contractual expiry time; the adapter therefore sets `expiry_utc=null` and `expiry_timestamp_status="not_verified"`. Date-based ACT/365F is used consistently for QuantLib and the core Monte Carlo comparisons. A timestamp is treated as exact only after its time provenance has been independently established.

Historical `bid`/`ask` can be used only when their own `bid_date`/`ask_date` (or `bid_ask_date`) matches the snapshot date and the spread is well formed. A stale spread is diagnosed as unavailable instead of replacing the Close reference with a live midpoint.

## Supply discount and forward curves

The CSV schema is:

```csv
as_of,expiry,discount_factor,forward,source
2026-10-06,2026-11-20,0.994,6000.0,illustrative_user_input
```

This row is an illustrative format example, not observed market data. Replace its expiry, discount factor, forward, and source with inputs appropriate to your snapshot. Provide rows for the expiries being analysed. `as_of` and `expiry` use `YYYY-MM-DD`; `discount_factor` and `forward` are positive finite values; `source` identifies the input's provenance. Align curve dates with the snapshot's as-of date and option expiries.

The curves define the pricing assumptions used consistently by independent QuantLib pricing, calibration, and core-Monte-Carlo comparisons. They can come from explicitly dated IPA discount/forward inputs or an overriding CSV. Positive IPA nodes are matched exactly when available, or interpolated log-linearly only inside their date range; interpolation provenance is recorded, and missing coverage remains unavailable. The time convention is date-based ACT/365F, with `T = calendar_days / 365`. Equivalent continuous rates are `r = -log(discount_factor) / T` and `q = r - log(forward / spot) / T`. Duplicate expiries, mismatched as-of dates, non-positive/non-finite numbers, empty files, and missing sources are rejected. A close-only option dataset cannot supply a measured bid/ask spread. Do not fabricate bid/ask quotes, dividend assumptions, or a confidence interval from a closing price.

### Eligibility and consistency checks

The initial defaults retain maturities of 7–365 calendar days and strike/forward ratios of 0.8–1.2. Validation rejects duplicate/missing RICs, mismatched quote dates, incompatible exercise/settlement/currency/premium conventions, missing expiry curves, invalid prices, and prices outside finite implied-volatility bounds. It does not clip bad prices into the admissible range.

QuantLib Black prices and a bracketed implied-volatility inversion provide the independent vanilla reference. Put-call parity, vertical-spread bounds, and strike convexity are reported as Close consistency diagnostics. Close-only inconsistencies are not labelled executable arbitrage. Historical bid/ask feasibility is assessed only when an actual dated spread is usable.

Training and holdout observations are grouped by `(expiry, strike)`, keeping a call and put at the same strike together; every fifth sorted strike group within an expiry is held out. With a usable historical spread, price uncertainty is `max(half_spread, 0.05)` index points. Without one, the explicit Close uncertainty assumption is `max(Black_vega × 0.01, 0.05)`, using a one-volatility-point allowance and the same tick floor. The report identifies the assumed uncertainty separately from the historical-spread case.

An untyped vendor surface requires strict, same-date typed evidence before use. IPA FinancialContracts must return `VolatilityPercent`, `ExerciseStyle="EURO"`, `PricingModelType="BlackScholes"`, and `VolatilityType="SVISurface"`, with matching instrument/expiry/strike, valuation and market-data dates, underlying-price provenance, and discount/dividend carry. Returned premiums must also agree with the Black price implied by the captured spot, forward, discount factor, and typed volatility. A unit identity is accepted only when it is uniquely established by the typed values; unit conventions are never inferred from the size of a number or an SDK example.

If the saved matrix cannot satisfy those checks, its values remain unverified and are rejected for overlays. A separately retrieved typed supplier dataset can use the matrix only for grid locations; its returned IV values and captured spot/carry overrides require their own verification. Supplied dated overrides are labelled as supplied inputs, not newly returned market observations. A separate typed dataset does not retroactively verify the old matrix's values. Requests, responses, coverage, tolerances, and hashes preserve the evidence for each accepted point.

Legacy snapshots with `metadata_basis="SDK_example_convention"`, `unit_assumption`, or `convention_assumption` are rejected as `vendor_surface_unverified`. Missing, ambiguous, stale, or carry-incompatible supplier data remains unavailable. Verified supplier IV is a provider-model diagnostic; independent core pricing validation continues to use QuantLib, and locally inverted Close IV is not relabelled as vendor IV.

## Replay and compare offline

Use a previously saved snapshot with the same curve inputs:

```sh
python -m tools.market_validation validate \
  --snapshot artifacts/spxw_2026-10-06/snapshot.json \
  --curves my_curves.csv \
  --output artifacts/spxw_validation

python -m tools.market_validation calibrate \
  --snapshot artifacts/spxw_2026-10-06/snapshot.json \
  --curves my_curves.csv \
  --starts 8 --max-nfev 500 \
  --output artifacts/spxw_calibration

python -m tools.market_validation report \
  --snapshot artifacts/spxw_2026-10-06/snapshot.json \
  --curves my_curves.csv \
  --paths 100000 --steps 64,256,1024 --seeds 42,43,44 \
  --output artifacts/spxw_report
```

`validate` checks the quote/curve inputs and independent vanilla conventions, and compares GBM Monte Carlo with QuantLib unless MC is skipped. `calibrate` fits Heston parameters through QuantLib's independent analytic pricing path. `report` combines the workflow results and numerical comparisons into HTML and plots; it requires `--snapshot` and never connects to LSEG. Supplying `--snapshot` selects saved data and supports replay without a live LSEG session. If `--as-of` is also supplied, it must match the snapshot date rather than rewriting it.

To execute the complete workflow, use `run` with a saved snapshot, or omit `--snapshot` to start with a live fetch:

```sh
python -m tools.market_validation run \
  --snapshot artifacts/spxw_2026-10-06/snapshot.json \
  --curves my_curves.csv \
  --output artifacts/spxw_run

python -m tools.market_validation run \
  --as-of 2026-10-06 \
  --curves my_curves.csv \
  --output artifacts/spxw_live_run
```

Use `--no-calibration` or `--no-mc` with `run`/`report` to skip the corresponding work. Skipped results remain explicit in the report; a skipped fit is not a successful calibration.

### Calibration conventions

Fit one global five-parameter Heston model across qualifying expiries. Choose one out-of-the-money representative per `(expiry, strike)` group: a call when `strike >= forward`, otherwise a put. Call/put duplicates do not become separate calibration targets. An expiry qualifies only with at least five training OTM strike groups after the holdout split. Thin expiries are recorded in `excluded_expiries` and excluded from training/holdout metrics while their observations remain available as separately labelled diagnostics. Calibration needs at least three qualifying expiries; otherwise it reports `insufficient_data`.

The objective is the model-price minus Close residual divided by the fixed pre-fit price uncertainty. SciPy bounded trust-region least squares uses `soft_l1` loss, with eight seeded starts and 500 objective evaluations per start by default. Start selection uses the training objective, not holdout losses or a vendor surface. Optimisation starts use seed 42; the `--seeds` option controls Monte Carlo runs separately.

| Parameter | Calibration bounds |
|---|---|
| `v0` | `[0.000001, 4]` |
| `kappa` | `[0.001, 20]` |
| `theta` | `[0.000001, 4]` |
| `xi` | `[0.0001, 5]` |
| `rho` | `[-0.999, 0.999]` |

QuantLib's analytic Heston engine prices each target with the expiry-specific equivalent flat carry derived from its supplied discount factor and forward. The report includes the chosen `parameters` (also exposed through the `params` alias), each start's objective/convergence status, training and holdout price/weighted RMSE, implied-volatility errors, and the Feller margin. The Feller condition is reported rather than imposed as an optimisation constraint. `converged`, `incomplete`, `failed`, and `insufficient_data` have distinct meanings; a finite parameter vector is not automatically a converged fit. A non-converged requested calibration gives the CLI a non-zero exit status while preserving its diagnostic report.

Numerical validation samples short/middle/long expiries and low/middle/high strikes. It compares core GBM prices at a fixed 20% volatility with independent Black prices, and calibrated Heston prices with independent analytic Heston prices. It does not use each quote's inverted Close volatility as the GBM verification target. Seeds must be distinct valid u64 integers; duplicate streams cannot supply independent samples. A four-standard-error sampling check and the last-grid change are recorded separately from market-fit residuals.

The additive Python API is `engine.price_heston_conditional(model, option_type, strike, variance_shifts=None)`. It returns the same `PriceResult` as ordinary pricing, requires `HestonModel` and a positive finite strike, and uses `[0.0, -4.0, 4.0]` when shifts are omitted. `[0.0]` selects pure conditional Monte Carlo. Custom shifts must be finite, distinct, include zero, and have finite squares; invalid models/parameters raise `ValueError`. The conditional estimator shares the core log-Euler/full-truncation raw-variance discretisation, so its confidence interval still measures sampling uncertainty rather than grid bias.

Automatic rechecking is Heston-only. It starts when the pooled ordinary estimate is zero for a positive independent reference, or its pooled standard error divided by that reference exceeds 25%. The conditional importance estimator runs the requested grids plus `max(1024, 4 × requested_finest_steps)`, with automatic additions capped at 16,384 steps; an explicitly requested grid above the cap is preserved. One further refinement uses four times the finest-grid path budget when relative error exceeds 10%, or otherwise a fourfold grid refinement when the reference gap exceeds sampling tolerance. Original ordinary estimates and their classification remain in the output.

The selected tail result requires a positive estimate for a positive reference, reference-denominated relative standard error of at most 10%, and a gap no larger than `4 × SE + 1e-12 + abs(reference) × 1e-8`. Tail references record the original 144-node quadrature, adaptive `1e-10` and `1e-12` calculations, and their differences; calibration retains its original pricing path. Records identify the estimator, shifts, reference method, and relative error; summaries retain `crude` and `tail_resolution` details, with top-level `tail_rechecks`. Ordinary and conditional estimates reuse seeds and are not combined as independent observations. A path-budget refinement is not reported as a grid change.

### Rare-event verification

If ordinary Heston sampling misses a positive tail price or has poor relative precision, the validator retains that result and runs the Rust core's conditional importance estimator. This integrates the independent stock noise analytically while retaining the same log-Euler stock update and full-truncation raw variance state. It changes the estimator, not the discretised model.

For variance-driver normals `U_i`, define `A = sum(v_i+ * dt)` and `B = sum(sqrt(v_i+ * dt) * U_i)`. Conditional terminal log-price has mean `log(S0) + (r-q)T - A/2 + rho*B` and variance `(1-rho²)*A`; its vanilla payoff expectation is lognormal and can be calculated directly. Zero conditional variance uses discounted intrinsic value.

The remaining variance-path tail uses a defensive mixture of Gaussian mean shifts `[0, -4, 4]`, spread across the time grid as `shift/sqrt(steps)`. Each shifted path receives the exact target/proposal density ratio. The zero-shift component preserves support and bounds that ratio; weights use the actual allocation among strata. Antithetic pairs count as independent samples, with a conservative pooled standard error. Setting `variance_shifts=[0.0]` selects pure conditional Monte Carlo:

```python
result = engine.price_heston_conditional(heston, "put", strike,
                                         variance_shifts=[0.0, -4.0, 4.0])
```

The method is Heston-only. Reference quadrature is tightened independently for tiny prices; the report stores the quadrature convergence checks, original estimate, selected estimator, mixture shifts, relative standard error, and grid changes. A zero mean/zero standard error cannot verify a positive tail reference. Sampling uncertainty and time-discretisation bias remain separate acceptance criteria.

### Command options

| Option | Meaning/default |
|---|---|
| `--as-of YYYY-MM-DD` | Requested market close date; fetch defaults to the last published complete close |
| `--snapshot PATH` | Saved input snapshot for offline replay; unavailable on `fetch` |
| `--curves CSV` | Dated discount-factor/forward inputs; needed for validation/calibration when the snapshot has no suitable embedded curves |
| `--output DIR` | Artifact destination; defaults to `artifacts/market_validation/<UTCstamp>` for workflow commands; `doctor` saves `doctor.json` only when this is provided |
| `--paths 100000` | Monte Carlo path count |
| `--steps 64,256,1024` | Comma-separated simulation step counts for discretisation comparisons |
| `--seeds 42,43,44` | Comma-separated distinct u64 simulation seeds; duplicate streams are rejected |
| `--starts 8` | Number of calibration starts |
| `--max-nfev 500` | Calibration objective-evaluation limit per start |
| `--no-calibration` | Skip calibration in `run`/`report` |
| `--no-mc` | Skip core-Monte-Carlo comparisons in `run`/`report` |
| `--verify-vendor-surface` | Request typed IPA supplier-volatility evidence on `fetch` or live `run`; requires FinancialContracts content permission and rejects `--snapshot` |
| `--prompt` | Permit hidden-key input for `doctor` if `LSEG_APP_KEY` is unset |

There is no RIC-selection flag in this release: SPXW is the fixed profile. Run a subcommand with `--help` for its supported options.

## Read the results

The output directory records the stages requested and their status. A skipped calibration or numerical stage is represented by `null` JSON and `not_requested` summary status; its CSV has headers without fabricated observations.

| Artifact | Purpose |
|---|---|
| `snapshot.json` | Reproducible quote snapshot and source/permission diagnostics |
| `raw_responses.json` | Redacted bulk provider evidence, preserved and hash-verified during replay when available |
| `validated.json` / `validation.json` | CLI-stage and report copies of quote eligibility, conventions, curve coverage, and independent vanilla validation |
| `summary.json` | Source metadata, accepted/rejected counts, settings, fit/MC status, metrics, and warnings |
| `calibration.json` | Fitted Heston parameters, fitting diagnostics, and residual information |
| `numerical_validation.json` | Core-Monte-Carlo comparisons over the chosen grids and seeds |
| `residuals.csv` | Accepted quote inputs, train/holdout groups, uncertainty assumptions, and fit residuals when calibrated |
| `numerical_validation.csv` | Independent-reference/core prices, standard errors, grid/seed settings, and sampling checks |
| `report.html` | Combined report with source status and links to output artifacts |
| `smiles.png` | Strike/moneyness views of observed and model-implied volatility |
| `surface.png` | Volatility surface view from available eligible observations |
| `term_structure.png` | Expiry-based volatility comparison |
| `residuals.png` | Price/IV residuals by training and holdout group, when calibration predictions exist |
| `error.json` | Redacted failure details when a stage cannot complete |

New CLI runs record stage durations in `timings_seconds`; an earlier captured report without timing evidence retains `null` instead of invented durations.

Calibration reports preserve excluded-quote reasons, optimisation status, and fit residuals. A fit can fail or be weakly identified even when the optimiser returns parameters; inspect the diagnostics and cross-start agreement before using the result. Reports preserve market-snapshot provenance and explicitly label synthetic fixtures; `--snapshot` identifies offline replay for the current invocation. Plot availability depends on the usable dataset; empty or sparse data does not justify invented observations.

The term structure labels nearest-observed ATM Close IV separately from calibrated exact-ATM Heston IV. A 2D surface is drawn only when observations span maturity and moneyness: it linearly interpolates **total variance** inside the observed `(log(strike / forward), maturity)` convex hull, masks the exterior, and does not extrapolate or claim an arbitrage-free projection. Compatible vendor points are labelled separately from locally inverted Close IV.

Compare four different quantities separately:

- **Market/model residuals:** how the independent model fits the eligible observed closes.
- **Pricing conventions and input assumptions:** expiry, settlement, discount factors, forwards, and quote eligibility.
- **Monte Carlo sampling error:** reported standard errors and variation across independent seeds.
- **Heston time-discretisation effects:** changes across the simulation step counts relative to the independent analytic price.

The Rust Monte Carlo engine is not the calibration objective. Its 95% confidence interval measures sampling uncertainty only and does not include Heston discretisation bias, uncertain curve inputs, stale quotes, or parameter-identification uncertainty. Continuous GBM barrier and GBM control/Greek methods retain their existing model restrictions; this market workflow does not make those methods Heston-compatible.

The additive Python helper `fe.black_scholes_price(model, "call" | "put", strike)` exposes the existing Rust closed-form formula for independent QuantLib regression checks. It requires a validated GBM model and positive finite strike; Heston raises `ValueError`. It returns a scalar price and does not change existing pricing methods.

### Reviewed 2026-10-06 market run

The saved local report is `artifacts/market_validation/lseg-20261006-validated/report.html`. Full raw/report artifacts are ignored by Git; the compact [dated LSEG golden reference fixture](../tests/fixtures/market/lseg_spxw_20261006_reference.json) is committed for offline regressions.

| Result | Observed value |
|---|---|
| SPXW capture | 74 quotes across six expiries |
| Input validation | 72 accepted; two moneyness rejections |
| Calibration coverage | Five qualifying expiries; 2027-09-30 excluded with only one training OTM strike group |
| Optimisation | Eight starts; selected fit converged |
| Training fit | 48 targets; price RMSE 3.53048 index points |
| Strike holdout | Ten targets; price RMSE 4.14810 index points; IV MAE 239.6 basis points |
| Independent MC/QuantLib comparison | 150 records, 14 summaries; all satisfy the declared sampling tolerance after two conditional tail rechecks |
| Rust closed-form/QuantLib comparison | Maximum absolute error `2.6148e-12` |
| Supplier volatility | Zero verified points; no vendor overlay; 114 untyped IPA matrix points retained only as raw evidence |
| Recorded replay time | 32.2 seconds for load, validation, calibration, and numerical validation before report generation |

The fitted parameters are `v0=0.00950145813`, `kappa=8.59859584`, `theta=0.03731037892`, `xi=1.64207484`, and `rho=-0.61857426`. The holdout residuals describe market fit under the recorded curve/date assumptions; they are separate from independent pricing verification.

Both originally missed positive tails are now resolved at 4,096 steps with 100,000 paths per seed and independent seeds 42/43/44. No further fourfold path-budget or 16,384-step refinement was needed. The combined standard error is pooled across these independent seeds; relative standard error below uses the independent reference price as its denominator.

| Heston tail | Conditional estimate | Combined SE | Adaptive QuantLib reference | SE / reference |
|---|---|---|---|---|
| Put, strike 6,300 | `2.1104146371e-5` | `9.6263293601e-8` | `2.1262144975e-5` | 0.453% |
| Call, strike 8,600 | `9.1896610019e-6` | `5.1361816279e-7` | `8.0618988256e-6` | 6.371% |

All 14 selected summaries satisfy `abs(estimate - reference) <= 4 × SE + 1e-12 + abs(reference) × 1e-8`; both tail results also meet the 10% relative-precision requirement. The original ordinary zero estimates and insufficient-tail classifications remain available under `crude`. These sampling checks do not establish zero Heston grid bias. The tail references retain the original 144-node and both adaptive quadrature checks.

Actual supplier verification remains blocked by content access. Historical FinancialContracts `ImpliedYield` was unsupported; the alternative typed `HistoricalYield` request with captured spot/discount/forward overrides was access-denied for `/.SPX`. The native historical `IMP_VOLT` value 9.9341 lacks unit/model metadata; `TR.OPWCloseImpliedVolatility` was empty, and `TR.IMPLIEDVOLATILITY=9.9341` lacks unit and dated-response fields. None is accepted as verified supplier volatility. The old untyped matrix is retained as raw provenance, excluded from plots and comparisons, and diagnosed as unavailable. Resolving this remaining supplier limitation requires the account's FinancialContracts content permission and successful typed response checks.

## Regressions and notebooks

All 42 offline market tests pass, covering filtering, date/curve alignment, independent prices, real-data Heston references and MC, conditional rare-tail resolution, strict supplier units/model/date/carry evidence, calibration outputs, thin-expiry handling, duplicate seeds, provenance, and unusable-data diagnostics without contacting LSEG. The Rust suite passes 79 tests, and the three conditional-pricing Python smoke groups pass on CPython 3.12 and 3.14 alongside the existing binding/extension smoke checks. The committed fixtures contain six real LSEG reference cases and twelve synthetic reference cases. Synthetic fixtures are explicitly labelled and do not establish live data permissions. The Python 3.12 CI job runs all `test_market*.py` regressions and notebooks in addition to the existing core checks.

The checked-in [synthetic SPXW regression snapshot](../tests/fixtures/market/synthetic_spxw.json) contains explicit synthetic curves and can demonstrate offline report generation without credentials:

```sh
python -m tools.market_validation report \
  --snapshot tests/fixtures/market/synthetic_spxw.json \
  --no-calibration --no-mc \
  --output artifacts/synthetic_report
```

Remove the skip flags to run calibration and Monte Carlo comparisons. This fixture's prices were generated with an independent Heston model, and its reports must retain the `Synthetic` provider label.

### Freeze a reviewed real-data reference fixture

After a real snapshot and its report have been reviewed, manually freeze a small reference dataset:

```sh
python -m tools.market_validation.regression \
  --snapshot artifacts/spxw_2026-10-06/snapshot.json \
  --output tests/fixtures/market/lseg_reference.json

# Optionally include the reviewed fit's independent analytic Heston prices.
python -m tools.market_validation.regression \
  --snapshot artifacts/spxw_2026-10-06/snapshot.json \
  --heston-parameters artifacts/spxw_run/calibration.json \
  --output tests/fixtures/market/lseg_reference.json
```

This command performs no live requests. It selects low/middle/high strikes from the shortest and longest available expiries, retaining call/put cases where present. It preserves the dated financial inputs and data origin, and stores independent QuantLib Black reference prices at 20% volatility, reference-library/version/convention metadata, numerical tolerances, and optional analytic Heston prices. Market Close values remain diagnostic observations rather than mathematical expected prices.

Commit only the curated source inputs and frozen golden references under `tests/fixtures/market/`; bulk raw responses and full reports remain in ignored `artifacts/`. CI reads the committed expected values and never runs this command to regenerate them. Updating a golden reference is an explicit manual maintenance step after inspecting the source-date/convention and reference-price changes.

Run offline tests with the market environment:

```sh
python -m unittest discover -s tests -p 'test_market*.py' -v
```

The worked notebooks remain [the introductory example](../notebooks/example.ipynb), [structured notes](../notebooks/structured_notes.ipynb), [numerical methods](../notebooks/numerical_methods.ipynb), and [Heston](../notebooks/heston.ipynb). All four have now rerun successfully: 18 code cells and nine PNG outputs are saved, and both the kernel-free runner and fresh Jupyter kernels pass. See the [roadmap](ROADMAP.md) for milestone status and the established pricing conventions.

Install notebook dependencies and execute the portable runner from the repository root:

```sh
python -m pip install -e '.[notebooks]'
python tools/run_notebooks.py --write

# In an environment that cannot open Jupyter kernel ports:
python tools/run_notebooks.py --kernel-free --write
```

The default runner starts a fresh Jupyter kernel per notebook. The kernel-free path uses the Agg plotting backend without kernel ports. `--write` saves fresh outputs while preserving source cells; omit it for a verification run without saving notebook changes.
