# fuzzy-enigma

A Rust Monte Carlo pricing library for **path-dependent exotic derivatives**.
The library focuses on a small, sharp set of pieces that compose cleanly:
a path-generating model, a payoff trait, and a Monte Carlo engine that knows
about neither finance nor probability theory.

The staged expansion plan and exact teaching contract terms live in
[docs/ROADMAP.md](docs/ROADMAP.md). New worked examples cover
[structured notes](notebooks/structured_notes.ipynb),
[numerical methods](notebooks/numerical_methods.ipynb), and
[Heston](notebooks/heston.ipynb). Optional SPXW market validation, offline
snapshot replay, and independent Heston calibration are documented in
[docs/MARKET_VALIDATION.md](docs/MARKET_VALIDATION.md).
The optional market module also provides a quote-derived, constrained SSVI
`VolSurface`, offline fitting, and calibration diagnostics. The worked
[volatility surface notebook](notebooks/volatility_surface.ipynb) uses saved
LSEG data and runs without credentials.

## Features

- **Geometric Brownian Motion** under the risk-neutral measure, simulated
  exactly with the log-Euler scheme.
- **Heston stochastic volatility** with log-Euler stock dynamics and
  full-truncation Euler variance, using two noise factors per step.
- **Path-dependent payoffs** out of the box:
  - European (sanity check vs. Black-Scholes)
  - Arithmetic-average **Asian**
  - Discretely-monitored **barrier** (all eight up/down x in/out x call/put
    flavours, with optional cash rebate)
  - **Lookback** (floating- or fixed-strike, call or put)
  - **Cliquet / ratchet** with local cap & floor and global cap & floor
  - **Autocallable note** with discrete observation dates and a downside
    protection barrier
  - **Snowball** with historical discrete knock-in and scheduled knock-out
  - **Phoenix** with independent coupon/autocall dates and optional coupon memory
  - **Scheduled Asian** with a fixing schedule independent of the simulation grid
- **Variance reduction** via antithetic variates and **control variates**
  (the Black-Scholes European makes a ready-made control for
  single-asset GBM exotics).
- **Geometric Asian control** with an exact discrete-fixing GBM expectation,
  and **continuous GBM single barriers** with Brownian bridge weighting.
- **Heston European conditional Monte Carlo** with analytic integration of
  the independent stock noise and defensive importance sampling of variance tails.
- **Greeks** by bump-and-revalue under common random numbers.
- **Parallel evaluation** via [`rayon`](https://crates.io/crates/rayon).
- **Reproducible** results: path `i` draws from ChaCha20 *stream* `i` under
  the engine's key, so a given `(seed, paths)` pair yields the same price
  independent of `parallel` - and two runs at different seeds are genuinely
  independent rather than shifted copies of each other.
- **Closed-form Black-Scholes** prices and Greeks in
  [`analytic`](src/analytic.rs), used for cross-checks and as the
  expectation behind the control variate.
- Pluggable single-asset models: implement [`PathGenerator`](src/model.rs),
  overriding `noise_dim()` for multiple stochastic factors. Paths remain
  one price per time point; multi-asset products need a separate path interface.

## Quick start

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
fuzzy-enigma = { path = "." }
```

Price an up-and-out barrier call:

```rust
use fuzzy_enigma::{
    BarrierKind, BarrierOption, GbmModel, McEngine, OptionType,
};

fn main() {
    // S0 = 100, r = 3%, q = 0%, sigma = 25%, T = 1y, 252 daily steps
    let model = GbmModel::new(100.0, 0.03, 0.0, 0.25, 1.0, 252);

    let payoff = BarrierOption {
        option_type: OptionType::Call,
        kind: BarrierKind::UpAndOut,
        strike: 100.0,
        barrier: 130.0,
        rebate: 0.0,
    };

    let engine = McEngine::new(200_000)
        .with_seed(42)
        .with_antithetic(true)
        .with_parallel(true);

    let result = engine.price(&model, &payoff);
    let (lo, hi) = result.confidence_95();
    println!("price = {:.4}  (95% CI [{:.4}, {:.4}])", result.price, lo, hi);
}
```

Run the bundled example to price the full menu of payoffs:

```sh
cargo run --release --example price_options
```

```
payoff                                     price      std-err    95% CI
European call K=100                      11.3141       0.0436 [11.23,11.40]
Asian call K=100                          6.4066       0.0232 [ 6.36, 6.45]
Up-and-out call K=100 B=130               2.4067       0.0157 [ 2.38, 2.44]
Down-and-in put  K=100 B=80               7.0254       0.0301 [ 6.97, 7.08]
Floating-strike lookback call            18.9272       0.0350 [18.86,19.00]
Cliquet 5% cap / -3% floor               13.0724       0.0304 [13.01,13.13]
Autocallable note (quarterly)            99.8258       0.0328 [99.76,99.89]
```

## Design

```
                 normals          path             cashflow PV
   McEngine ----noise----> Model ------> Payoff -----------> mean & std-err
   (RNG, MC,                                                   |
    antithetic,                                                v
    parallel)                                              PriceResult
```

Three traits, one job each:

| trait | role | implementations |
|---|---|---|
| `PathGenerator` | turn standard-normal increments into one asset path | `GbmModel`, `HestonModel` |
| `Payoff` | turn a path into a present-value cashflow | European, Asian, scheduled Asian, barrier, lookback, cliquet, autocallable, snowball, Phoenix |
| `McEngine` | draw noise, drive the model, average payoffs | (one struct, configurable) |
| `ControlVariate` | a correlated quantity whose mean is known in closed form | `EuropeanControl`, `GeometricAsianControl` (GBM) |

Because the noise lives outside the model, antithetic variates are a free
feature of the engine: it generates one path, negates the noise vector,
and re-runs the same model to produce the antithetic twin.

## Conventions

**Monitoring and the initial fixing.** `path[0]` is the spot at t = 0 and
`path[i]` is the price at `t = i * dt`. Whether t = 0 counts as an
observation follows each contract's market convention rather than one
blanket rule:

| payoff | window | why |
|---|---|---|
| Barrier | whole path | a contract already through its barrier at inception knocks out (or in) immediately |
| Lookback | whole path | the running extremum starts at the initial fixing, so a floating strike is never worse than the starting spot |
| Asian | `path[1..=N]` | averaging windows conventionally exclude the initial fixing |
| Cliquet | returns from `i = 1` | `path[0]` anchors the first period without being an observation itself |
| Autocallable | exactly the given indices | index 0 is legal and means "observe at inception" |

**Cliquet global floor.** `CliquetOption::global_floor` (and the
`global_floor` argument in Python) defaults to `0.0`, which makes the note
capital-guaranteed: the summed return is floored at zero, so it can never
pay out less than nothing. That is a common structure but it is *not* the
same contract as an uncapped-downside cliquet - pass
`global_floor: f64::NEG_INFINITY` if you want losses to pass through.

## Variance reduction

Two techniques ship, and they can be combined:

```rust
let asian = AsianOption { option_type: OptionType::Call, strike: 100.0 };
let control = EuropeanControl::new(&model, OptionType::Call, 100.0);
let result = engine.price_with_control(&model, &asian, &control);
```

The control's optimal coefficient `beta = Cov(X, Y) / Var(X)` is estimated
from the same sample that produces the price, which is standard practice
and introduces an `O(1/n)` bias - negligible against the `O(1/sqrt(n))`
standard error at any realistic path count.

Note that the two techniques *overlap* rather than compound: both suppress
the linear component of the payoff. On the Asian call above, the European
control cuts the standard error 1.82x on its own but only 1.51x on top of
antithetic variates.

## Greeks

Bump-and-revalue under common random numbers - every bumped run reuses the
engine's seed, so it shares its noise with the base run path-for-path and
the difference reflects the bump rather than sampling error:

```rust
let greeks = bump_and_revalue(&engine, &model, &payoff, BumpSizes::for_model(&model));
println!("delta {:.4}  gamma {:.4}  vega {:.4}", greeks.delta, greeks.gamma, greeks.vega);
```

Delta, gamma, vega and rho use central differences; theta uses a backward
difference in maturity. Keeping step counts and observation indices fixed
rescales **all observation times** with maturity: this is schedule-scaled
`dV/dT`, rather than calendar roll-down theta. Vega and rho are per unit, not per point
or basis point, and `theta` is `dV/dT` (positive for a longer-dated
option). At 400k paths these land within ~0.3% of the Black-Scholes values
for a vanilla call.

## Error handling

Constructors and pricing calls come in both panicking and fallible forms.
`GbmModel::new` and `McEngine::price` panic on an invalid setup;
`GbmModel::try_new` and `McEngine::try_price` return
[`PricingError`](src/error.rs) instead. Validation runs **once**, before
any path is simulated, which matters under `parallel`: a check that fired
per-path would fire inside every rayon worker at once.

## Testing

```sh
cargo test --release
```

The integration tests pin the engine against:

- closed-form **Black-Scholes** for vanilla calls and puts (price must lie
  in the Monte Carlo 95% confidence interval),
- **in/out parity** for barriers (`up-and-in + up-and-out == European`),
- **Asian < European** (averaging dampens volatility),
- **lookback > European** (lookback dominates the vanilla payoff path-wise),
- **bumped Greeks against analytic** delta, gamma and vega,
- **control variate** cuts the standard error without moving the price,
- **seed independence** - runs at neighbouring seeds must differ by
  O(std-err); this guards a fixed bug where path `i` of seed `s` was
  bit-identical to path `i-1` of seed `s+1`,
- **parallel == serial**, bit for bit,
- **invalid input** is rejected up front rather than panicking mid-run.

The Python bindings have their own smoke test, which CI runs against both
ends of the supported interpreter range:

```sh
python tests/test_bindings.py
python tests/test_extensions.py
```

## Python bindings

The same crate exposes a Python extension module via [PyO3](https://pyo3.rs)
and [maturin](https://www.maturin.rs/). The `python` Cargo feature is
opt-in, so the pure-Rust build stays dependency-free.

The extension is built against the **stable ABI** (`abi3-py39`), so one
`fuzzy_enigma.abi3.so` - and one `cp39-abi3` wheel - loads on every CPython
from 3.9 up. Without this a compiled extension is pinned to the exact
interpreter that built it (`fuzzy_enigma.cpython-314-darwin.so` is
unimportable from 3.12, and vice versa), which is a recurring source of
`ModuleNotFoundError` when a venv and a Jupyter kernel drift apart. Note
that this stops at *our* module: `numpy` and friends have no stable ABI and
still need to match the running interpreter.

Build and install into the active virtualenv:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install maturin numpy matplotlib jupyter
maturin develop --release
```

Then from Python:

```python
import fuzzy_enigma as fe

model = fe.GbmModel(spot=100, r=0.03, q=0.0, sigma=0.25, t=1.0, steps=252)
engine = fe.McEngine(paths=200_000, seed=42, antithetic=True, parallel=True)

print(engine.price_european(model, "call", 100.0))
print(engine.price_asian(model, "call", 100.0))
print(engine.price_barrier(model, "call", "up_and_out", 100.0, 130.0))
print(engine.price_lookback(model, "call"))                           # floating strike
print(engine.price_cliquet(model, 100.0, -0.03, 0.05))
print(engine.price_autocallable(model, 100.0, 2.0, 100.0, 70.0,
                                [63, 126, 189, 252]))

# Variance reduction and Greeks
print(engine.price_asian(model, "call", 100.0, control=True))
print(engine.greeks_european(model, "call", 100.0))
# Greeks(delta=0.596661, gamma=0.015494, vega=38.696945, theta=6.292439, rho=48.332904)
```

Invalid input raises a catchable `ValueError` rather than escaping as a
Rust panic:

```python
>>> engine.price_autocallable(model, 100.0, 2.0, 100.0, 70.0, [999])
ValueError: observation index 999 is outside the simulated path
            (model has 252 steps, so the largest legal index is 252)
>>> fe.GbmModel(spot=-100, r=0.03, q=0.0, sigma=0.25, t=1.0)
ValueError: invalid parameter `spot`: must be positive
```

`PriceResult` exposes `.price`, `.std_error`, `.samples` and a
`.confidence_95()` method; `Greeks` exposes `.delta`, `.gamma`, `.vega`,
`.theta` and `.rho`. The Monte Carlo work runs with the GIL
released, so calling from a multi-threaded Python application does not
block other threads.

A worked example with cross-checks against Black-Scholes, a convergence
plot, and a barrier sweep lives at [`notebooks/example.ipynb`](notebooks/example.ipynb).

### New products and models

```python
dates = [63, 126, 189, 252]
print(engine.price_snowball(model, 100.0, 100.0, 0.12, 70.0, 103.0, dates))
print(engine.price_phoenix(model, 100.0, 100.0, 2.0, 80.0, 60.0,
                           105.0, dates, dates, memory=True))
print(engine.price_asian_scheduled(model, "call", 100.0, dates,
                                   control="geometric"))
print(engine.price_continuous_barrier(model, "call", "up_and_out", 100.0, 130.0))

heston = fe.HestonModel(spot=100, r=0.03, q=0.0, v0=0.04, kappa=1.5,
                        theta=0.04, xi=0.4, rho=-0.7, t=1.0, steps=252)
print(engine.price_european(heston, "call", 100.0))
print(engine.price_snowball(heston, 100.0, 100.0, 0.12, 70.0, 103.0, dates))

# Heston-only European estimator; default variance shifts are [0.0, -4.0, 4.0].
print(engine.price_heston_conditional(heston, "put", 80.0))
# Use [0.0] for conditional Monte Carlo without variance-tail shifts.
print(engine.price_heston_conditional(heston, "put", 80.0, variance_shifts=[0.0]))
```

`reference_spot` is the fixed initial contract fixing; barriers are absolute
price levels. Snowball coupons are annual rates; Phoenix coupons are cash
amounts per observation. See [the cashflow tables](docs/ROADMAP.md) for the
exact contract variants. The structured methods have matching
`greeks_snowball` / `greeks_phoenix` methods with the same arguments;
scheduled Asian and continuous barrier also expose corresponding Greeks.
Greeks, analytic controls, and continuous bridge methods require GBM and
reject Heston with `ValueError`. Standard uncontrolled payoff pricing accepts
both models; `price_heston_conditional` requires Heston and a European payoff.

## Optional market tools

Use Python 3.12 and the optional `market` extra for the SPXW workflow.
Live access uses the desktop LSEG session and `LSEG_APP_KEY`; the doctor
checks actual capabilities, and saved snapshots support offline replay.
Desktop access, the SPX close, SPXW Search discovery/native daily history,
and historical IPA surface/curve requests have succeeded. The latest reviewed
capture is dated 2026-10-08: 87 quotes and seven curve expiries, with 84
accepted quotes and a converged SSVI fit on six qualifying expiries.
The 2026-10-06 snapshot remains available for historical replay.
The historical option probe reported delayed quote quality of service.

```sh
python -m pip install -e '.[market]'
python -m tools.market_validation doctor --output artifacts/market_doctor
python -m tools.market_validation run \
  --snapshot path/to/snapshot.json --curves path/to/curves.csv \
  --output artifacts/market_report

# Fit a constrained quote-derived surface entirely offline.
python -m tools.market_validation surface-fit \
  --snapshot tests/fixtures/market/lseg_spxw_20261008_surface.json \
  --surface-model ssvi --starts 8 --seed 42 --max-iter 2000 \
  --output artifacts/market_validation/ssvi_20261008
```

The CLI provides `doctor`, `fetch`, `validate`, `calibrate`, `surface-fit`,
`report`, and `run`. QuantLib supplies the independent vanilla pricing/calibration path;
the Rust Monte Carlo engine supplies separate grid/seed comparisons.
The additive `fe.black_scholes_price(model, option_type, strike)` helper
exposes the Rust closed-form GBM price for independent regression checks.
See the [market validation guide](docs/MARKET_VALIDATION.md) for live-fetch
examples, snapshot and curve conventions, command options, and output files.
The validator preserves ordinary MC results and automatically rechecks rare
Heston tails with the conditional importance estimator and finer grids.
The reviewed run's two missed tails now satisfy the declared sampling check;
their original zero estimates remain in the report.
IPA overlays require typed date/unit/model/carry evidence; ambiguous legacy
surfaces based on SDK unit assumptions are rejected. The current account's
typed supplier request is access-denied, so the report has no vendor overlay.
Use `--verify-vendor-surface` with `fetch` or a live `run` to request that
evidence when the account has the required content permission.

## Caveats

- Barrier monitoring is **discrete on the simulation grid**. For a
  continuously-monitored GBM contract, use `price_continuous_barrier`.
  Structured note knock-in remains contractually discrete.
- Greeks are finite-difference estimates. Pathwise and likelihood-ratio
  estimators converge better for the payoffs that admit them, but they need
  per-payoff derivative information, whereas bumping works on any `Payoff`
  as-is.
- Heston simulation has time-discretisation bias in addition to sampling
  error; the Monte Carlo confidence interval measures sampling error only.
- The library currently models a single underlying. Multi-asset baskets,
  Heston-specific Greeks, and American/Bermudan exercise are future work.
  Optional Python Heston calibration is part of the market tools.

## License

MIT.
