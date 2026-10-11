# Dupire local volatility

Milestone 7A adds a native `LocalVolModel` for dated European, discretely
monitored barrier, Snowball and Phoenix pricing. It uses the same spot,
discount curve, forward curve and contractual dates as the GBM/Heston dated
API. A European surface fixes terminal marginals; choosing Local Vol rather
than stochastic volatility is an additional assumption about the joint path.

The optional Python market tools prepare a distinct, versioned **smooth SSVI
export**. The existing milestone 5 `VolSurface`, its fit, its JSON schema and
its piecewise-linear queries remain unchanged. The export and the native
model are available offline using the two committed LSEG fixtures. No new
market capture or verified supplier-volatility dataset is implied.

## Smooth surface and extensions

The evaluator uses `k = log(K/F(T))`, ACT/365F and decimal annualised IV, with
`w = IV² T`. It keeps the fitted shared `rho`, `eta` and ATM total-variance
nodes, adds `(0, 0)`, and uses monotone piecewise-cubic Hermite interpolation
for the ATM curve. Time derivatives agree on both sides of interior knots.
The origin has an explicit right derivative; the last node uses the left
derivative. Queries beyond the last maturity fail.

The short-time policy is declared globally rather than switching formulas
at a small numerical time:

```text
phi(theta) = eta / sqrt((theta + theta_floor) * (1 + theta))
w(k,t) = theta(t)/2 * [1 + rho*phi*k
                      + sqrt((phi*k + rho)^2 + 1-rho^2)]
```

`theta_floor` is a strictly positive **ATM total variance**, not a volatility
floor or a local-variance clipping parameter. The default is 1% of the first
positive fitted ATM node. Its finite origin skew makes the zero-time Local
Vol limit well defined. This changes vanilla smiles from the original fit;
the export reports original-versus-smooth price and IV changes separately
from original quote-fit residuals. The complete analytic SSVI wings cover
every finite log-moneyness. There is no hidden constant-volatility fallback,
strike clamping or arbitrary simulation cutoff in either wing.

The construction retains the legacy sufficient parameter constraint
`eta * (1 + abs(rho)) <= 2 - epsilon`. For `a = theta_floor > 0`, direct
differentiation gives

```text
(d/dtheta)[theta*phi] / phi
    = 1 - theta/(2*(theta+a)) - theta/(2*(1+theta))  in (0,1]
theta*phi <= eta
theta*phi^2 <= eta^2
```

Together with the monotone ATM interpolation, these inequalities satisfy the
SSVI calendar and butterfly conditions over all strikes and supported times.
This is an application of [Gatheral and Jacquier, Theorems 4.1–4.2](https://arxiv.org/html/1204.0646v4#S4),
not a conclusion inferred from a finite strike scan. Dense diagnostics and
runtime denominator checks supplement the analytic coverage argument.

## Dupire evaluator and numerical policy

Strike and time derivatives are analytic. At fixed forward moneyness,
the native evaluator computes

```text
g = (1 - k*w_k/(2*w))^2 - w_k^2/4 * (1/w + 1/4) + w_kk/2
local_variance = w_t / g
```

This is the total-variance form of [Dupire's construction](https://www.risk.net/sites/default/files/import_unmanaged/risk.net/data/Pay_per_view/risk/technical/1994/risk_0194_volatility.pdf);
[QuantLib's LocalVolSurface](https://github.com/lballabio/QuantLib/blob/master/ql/termstructures/volatility/equityfx/localvolsurface.cpp)
implements the same transformation using finite differences. Derivatives
with respect to time must hold `k` fixed; treating strike as fixed would
introduce inconsistent forward carry.

At zero time, write the smile factor as `A(k)`, so `w = theta*A`.
The explicit limit is `theta'(0)*A / (1-k*A_k/(2*A))^2`. It is not an
evaluation at an arbitrary tiny time. To check its stability, set
`x = phi(0)*k` and `q = sqrt(x^2+2*rho*x+1)`. Then
`A-x*A_x = [1+(rho*x+1)/q]/2 >= 0`, because
`q^2-(rho*x+1)^2 = (1-rho^2)*x^2 >= 0`. Consequently the origin
denominator is at least `1/4`. The flat-smile case `eta=0` is treated
directly and recovers `local_variance = theta'(t)`.

The artifact declares a positive minimum denominator. Non-finite evaluations,
any negative time derivative or local variance, and a denominator below the
declared threshold raise an error. Stable algebra avoids cancellation at the
origin and flat time intervals. Reconstruction tolerances check recorded
coefficients without changing derivative values. There is no round-off
clipping or fallback; diagnostics record zero round-off adjustments.

## Artifact and pricing

The smooth artifact has its own schema/version and canonical hash. It stores
the original surface and snapshot identities, same-date context carry,
original and origin-extended ATM nodes, cubic coefficients, parameters,
short-time/wing policies, supported tenor, observed quote coverage,
constraint parameters and numerical thresholds. Constraint margins are
computed from those parameters in the separate diagnostics, rather than
stored as trusted claims in the numeric artifact. Loading rechecks the numeric
representation and derived coefficients rather than trusting a claimed
validation status. ACT/365F and decimal IV/total-variance units are fixed by
the schema. Context date, spot, currency, asset and curve values must agree
with the export for pricing. The export-change report is a separate
diagnostic sidecar, rather than part of the hashed numeric artifact.
Saving a live export writes `<stem>.export_report.json` alongside the numeric
JSON. Loading the numeric artifact revalidates it and does not silently trust
or reconstruct an unhashed report.

```python
import fuzzy_enigma as fe
from tools.market_validation import context_from_snapshot
from tools.market_validation.surface import VolSurface, fit_ssvi
from tools.market_validation.local_vol import export_local_vol
from tools.market_validation.analytics import validate_snapshot
from tools.market_validation.snapshot import read_snapshot

path = "tests/fixtures/market/lseg_spxw_20261008_surface.json"
snapshot = read_snapshot(path)
context, audit = context_from_snapshot(snapshot, currency="USD")
fit = fit_ssvi(validate_snapshot(snapshot), diagnostics=False)
surface = VolSurface.from_dict(fit["surface"])
smooth = export_local_vol(surface, context=context)
smooth.save("artifacts/local_vol_surface.json")
local_vol = fe.LocalVolModel.load("artifacts/local_vol_surface.json")

engine = fe.McEngine(40_000, seed=42, antithetic=True, parallel=True)
result = engine.price_european_context(
    context, local_vol, "call", context.spot, "2026-12-31",
)
print(result.price, result.std_error, result.path_diagnostics)
```

Rust exposes `local_vol::LocalVolModel::{from_json, load, to_json}` and
`McEngine::try_price_local_vol_context`. The prepared generator also implements
the shared `ContextPathGenerator` trait. The existing `Dynamics` enum keeps
its GBM/Heston variants and `Copy` behaviour; the Local Vol entry point is
additive rather than changing downstream exhaustive matches.

With a supplied dated context and expiry, a Rust European valuation is:

```rust
use fuzzy_enigma::{
    dated::{ContextEuropeanOption, TimeGrid},
    local_vol::LocalVolModel,
    OptionType,
};

let model = LocalVolModel::load("artifacts/local_vol_surface.json")?;
let grid = TimeGrid::try_new(context.as_of(), vec![expiry], 1.0)?;
let context = context.with_grid(grid)?;
let payoff = ContextEuropeanOption::try_new(
    OptionType::Call, context.spot, expiry, expiry, context.currency.clone(),
)?;
let result = engine.try_price_local_vol_context(&context, &model, &payoff)?;
```

`price_european_context` and `price_barrier_context` emit nominal dated
cashflows, which the context engine discounts once. Barrier monitoring uses
its explicit supplied dates, including inception where applicable; numerical
refinement adds no contract observations. Rebates settle on the stated
payment date. Existing dated Snowball/Phoenix pricing accepts the native
`LocalVolModel` and retains history, reference fixing and settlement rules.

Log-stock simulation uses the exact curve carry increment
`log(F(t_next)/F(t))`, subtracts half the left-end local variance times the
step length, and adds the local-volatility normal shock. Time discretisation
error remains in addition to Monte Carlo sampling error. Serial and parallel
runs use the same per-path random streams and aggregation order.

The result exposes actual path count in `paths`. Its path diagnostics disclose
total evaluated points, paths using extensions, observed-moneyness wing
evaluations, short-time-extension
evaluations and minimum/maximum evaluated log-moneyness. These measures refer
to left-end grid states where volatility is queried; the terminal fixing is
not an additional volatility evaluation. They do not measure unobserved
continuous path excursions.
An extension is a declared modelling choice; a low Monte Carlo SE does not
validate it against absent market quotes.

GBM controls, continuous-barrier bridge weighting and legacy Greeks retain
their explicit model restrictions. Local Vol is admitted through the dated
context API. `calendar_theta` rejects Local Vol because a surface-roll
convention has not been defined; the existing GBM/Heston frozen roll and
schedule-scaled legacy `dV/dT` remain unchanged.

## Independent validation and model comparisons

The [model comparison notebook](../notebooks/model_comparison.ipynb) uses saved
LSEG quotes/curves with synthetic barrier and note terms. It reports the
snapshot date/hash, fit settings, model parameters, interpolation/extensions,
path counts, seeds, grids, dependencies and elapsed times. GBM, calibrated
Heston and Local Vol share the same valuation context and contractual
monitoring dates. Changing a numerical grid does not change the contract.

Two residuals answer different questions:

- Original quote residuals measure the chosen model's market fit under the
  fixed training/holdout split.
- Native Local Vol MC and the independent finite-difference/PDE price are
  compared with independent Black prices of the **smooth exported surface**.
  This checks implementation and discretisation without treating the original
  piecewise-linear surface as a Dupire input.

The finite-difference reference evolves normalised call prices on a separate
spatial/time mesh. Mesh refinements, declared tolerances and independent
Black prices distinguish PDE truncation/discretisation from native MC error.
Synthetic constant-volatility and curved-smile benchmarks define their
acceptance tolerances before selecting market results. Benchmark failures
remain visible rather than being cured by widening tolerances after a run.

### Reviewed offline example

The 2026-10-08 example fits six qualifying maturities through 2027-03-31 using
58 training and 12 holdout OTM groups. The default smooth export changes the
original fitted call prices by at most 4.23137 index points and IV by at most
23.3550 bp over the declared export probes; these are model-construction
changes, not MC errors. Original-versus-smooth SSVI holdout price RMSE is
6.14531 versus 6.14481 points. GBM and calibrated Heston holdout RMSE is
15.31121 and 10.42921 points in this example. Fit parameters and all starts
remain in local replay artifacts; a converged optimiser does not establish
unique parameter identification.

The independent PDE is run at three forward-moneyness strikes on 2026-11-06
and 2027-03-31. Its maximum absolute error against smooth QuantLib Black
prices decreases from 1.72015 points on the 401×400 mesh to 0.425626 points
on the 801×800 mesh. The native Local Vol example uses 16,000 paths, seed 42,
antithetic sampling and one-day/half-day grids. Maximum absolute vanilla
residual is 4.06324 points, with individual SEs from 0.2551 to 2.3866 points.
Both grids satisfy the predeclared sampling-plus-discretisation budget;
finite-sample residuals need not decrease monotonically with step length.

| Synthetic contract, half-day grid | GBM | Calibrated Heston | Local Vol |
|---|---:|---:|---:|
| Weekly up-and-out call, index points | 111.6387 | 211.2317 | 190.8745 |
| Snowball, USD per 100 principal | 102.4334 | 100.8632 | 100.2872 |
| Memory Phoenix, USD per 100 principal | 103.3969 | 102.1851 | 101.8037 |

On the shared six-month half-day grid, 9.55% of Local Vol evaluations use
analytic wings and 4.02% use the short-time extension. All paths encounter
an extension; evaluated log-moneyness ranges from -1.42467 to 0.54653.
These path-dependent model price differences coexist with different vanilla
fit residuals. They are not a calibrated confidence interval for model risk.
Full fit/export reports, numerical tables and provenance are saved under
`artifacts/local_vol_model_comparison`; saved plot outputs are part of the
worked notebook.

```sh
python tests/test_local_vol.py
python -m unittest discover -s tests -p 'test_market_local_vol*.py' -v
python tools/run_notebooks.py --notebook notebooks/model_comparison.ipynb --write
python tools/run_notebooks.py --output-dir artifacts/local-vol-notebooks
```

The native smoke script joins both endpoint Python builds and the shared
Python 3.9–3.14 abi3 CI wheel checks. Offline market tests and the notebook
runner automatically discover the new market tests and worked example.
Final acceptance is complete: 148 Rust tests, 101 offline market tests
(21 Local Vol tests), formatting, both all-targets Clippy configurations,
both Rust examples and all five native smoke scripts pass. The same final
`cp39-abi3` wheel passes locally on Python 3.12 and 3.14; CI covers its
Python 3.9–3.14 interpreter matrix. All seven notebooks pass fresh Jupyter
execution with 37 code cells and 20 visually inspected PNGs. The new example
also passes kernel-free execution with seven code cells and four charts.
Final gate details and artifact locations are recorded in the
[roadmap](ROADMAP.md).
