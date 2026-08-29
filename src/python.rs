//! PyO3 bindings exposing the library as a Python extension module.
//!
//! Built via maturin: `maturin develop --release --features python`.
//! Compiled out of the default build, so plain `cargo build` is
//! unaffected.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::error::PricingError;
use crate::greeks::{try_bump_and_revalue, BumpSizes, Greeks};
use crate::model::GbmModel;
use crate::payoff::{
    AsianOption, AutocallableNote, BarrierKind, BarrierOption, CliquetOption, EuropeanOption,
    LookbackOption, OptionType,
};
use crate::pricer::{EuropeanControl, McEngine, PriceResult};

/// Surface a `PricingError` as a catchable Python `ValueError` rather than
/// letting a Rust panic cross the FFI boundary as `PanicException`.
fn to_py_err(err: PricingError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn parse_option_type(s: &str) -> PyResult<OptionType> {
    match s.to_ascii_lowercase().as_str() {
        "call" | "c" => Ok(OptionType::Call),
        "put" | "p" => Ok(OptionType::Put),
        other => Err(PyValueError::new_err(format!(
            "option_type must be 'call' or 'put', got '{other}'"
        ))),
    }
}

fn parse_barrier_kind(s: &str) -> PyResult<BarrierKind> {
    let normalised = s.to_ascii_lowercase().replace('-', "_");
    match normalised.as_str() {
        "up_and_out" | "uo" => Ok(BarrierKind::UpAndOut),
        "up_and_in" | "ui" => Ok(BarrierKind::UpAndIn),
        "down_and_out" | "do" => Ok(BarrierKind::DownAndOut),
        "down_and_in" | "di" => Ok(BarrierKind::DownAndIn),
        other => Err(PyValueError::new_err(format!(
            "barrier kind must be one of \
             'up_and_out', 'up_and_in', 'down_and_out', 'down_and_in', got '{other}'"
        ))),
    }
}

// -----------------------------------------------------------------------
// GbmModel
// -----------------------------------------------------------------------

#[pyclass(name = "GbmModel", module = "fuzzy_enigma", from_py_object)]
#[derive(Clone)]
pub struct PyGbmModel {
    inner: GbmModel,
}

#[pymethods]
impl PyGbmModel {
    /// Geometric Brownian Motion under the risk-neutral measure.
    #[new]
    #[pyo3(signature = (spot, r, q, sigma, t, steps = 252))]
    fn new(spot: f64, r: f64, q: f64, sigma: f64, t: f64, steps: usize) -> PyResult<Self> {
        let inner = GbmModel::try_new(spot, r, q, sigma, t, steps).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn spot(&self) -> f64 {
        self.inner.spot
    }
    #[getter]
    fn r(&self) -> f64 {
        self.inner.risk_free_rate
    }
    #[getter]
    fn q(&self) -> f64 {
        self.inner.dividend_yield
    }
    #[getter]
    fn sigma(&self) -> f64 {
        self.inner.volatility
    }
    #[getter]
    fn t(&self) -> f64 {
        self.inner.maturity
    }
    #[getter]
    fn steps(&self) -> usize {
        self.inner.steps
    }

    fn __repr__(&self) -> String {
        let m = &self.inner;
        format!(
            "GbmModel(spot={}, r={}, q={}, sigma={}, t={}, steps={})",
            m.spot, m.risk_free_rate, m.dividend_yield, m.volatility, m.maturity, m.steps,
        )
    }
}

// -----------------------------------------------------------------------
// PriceResult
// -----------------------------------------------------------------------

#[pyclass(name = "PriceResult", module = "fuzzy_enigma", from_py_object)]
#[derive(Clone)]
pub struct PyPriceResult {
    inner: PriceResult,
}

#[pymethods]
impl PyPriceResult {
    #[getter]
    fn price(&self) -> f64 {
        self.inner.price
    }
    #[getter]
    fn std_error(&self) -> f64 {
        self.inner.std_error
    }
    #[getter]
    fn samples(&self) -> usize {
        self.inner.samples
    }

    fn confidence_95(&self) -> (f64, f64) {
        self.inner.confidence_95()
    }

    fn __repr__(&self) -> String {
        let r = &self.inner;
        let (lo, hi) = r.confidence_95();
        format!(
            "PriceResult(price={:.6}, std_error={:.6}, samples={}, ci95=[{:.6}, {:.6}])",
            r.price, r.std_error, r.samples, lo, hi,
        )
    }
}

// -----------------------------------------------------------------------
// Greeks
// -----------------------------------------------------------------------

#[pyclass(name = "Greeks", module = "fuzzy_enigma", from_py_object)]
#[derive(Clone)]
pub struct PyGreeks {
    inner: Greeks,
}

#[pymethods]
impl PyGreeks {
    #[getter]
    fn delta(&self) -> f64 {
        self.inner.delta
    }
    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma
    }
    #[getter]
    fn vega(&self) -> f64 {
        self.inner.vega
    }
    #[getter]
    fn theta(&self) -> f64 {
        self.inner.theta
    }
    #[getter]
    fn rho(&self) -> f64 {
        self.inner.rho
    }

    fn __repr__(&self) -> String {
        let g = &self.inner;
        format!(
            "Greeks(delta={:.6}, gamma={:.6}, vega={:.6}, theta={:.6}, rho={:.6})",
            g.delta, g.gamma, g.vega, g.theta, g.rho,
        )
    }
}

// -----------------------------------------------------------------------
// McEngine
// -----------------------------------------------------------------------

#[pyclass(name = "McEngine", module = "fuzzy_enigma", from_py_object)]
#[derive(Clone)]
pub struct PyMcEngine {
    inner: McEngine,
}

#[pymethods]
impl PyMcEngine {
    /// Monte Carlo engine.
    ///
    /// Parameters
    /// ----------
    /// paths : int
    ///     Total number of simulated paths. With `antithetic=True` they
    ///     are run in pairs.
    /// seed : int
    ///     Reproducibility seed.
    /// antithetic : bool
    ///     Apply antithetic variance reduction.
    /// parallel : bool
    ///     Evaluate paths across rayon threads.
    #[new]
    #[pyo3(signature = (paths = 100_000, seed = 42, antithetic = true, parallel = true))]
    fn new(paths: usize, seed: u64, antithetic: bool, parallel: bool) -> Self {
        Self {
            inner: McEngine {
                paths,
                seed,
                antithetic,
                parallel,
            },
        }
    }

    #[getter]
    fn paths(&self) -> usize {
        self.inner.paths
    }
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }
    #[getter]
    fn antithetic(&self) -> bool {
        self.inner.antithetic
    }
    #[getter]
    fn parallel(&self) -> bool {
        self.inner.parallel
    }

    /// Price a vanilla European call/put.
    fn price_european(
        &self,
        py: Python<'_>,
        model: PyRef<'_, PyGbmModel>,
        option_type: &str,
        strike: f64,
    ) -> PyResult<PyPriceResult> {
        let payoff = EuropeanOption {
            option_type: parse_option_type(option_type)?,
            strike,
        };
        let model = model.inner.clone();
        let engine = self.inner.clone();
        let inner = py
            .detach(move || engine.try_price(&model, &payoff))
            .map_err(to_py_err)?;
        Ok(PyPriceResult { inner })
    }

    /// Price an arithmetic-average Asian call/put.
    ///
    /// With `control=True` the European option at the same strike is used
    /// as a control variate, which cuts the standard error at no extra
    /// path cost. Note that the gain partly overlaps with `antithetic`:
    /// both suppress the payoff's linear component, so enabling the second
    /// one helps less than it would on its own.
    #[pyo3(signature = (model, option_type, strike, control = false))]
    fn price_asian(
        &self,
        py: Python<'_>,
        model: PyRef<'_, PyGbmModel>,
        option_type: &str,
        strike: f64,
        control: bool,
    ) -> PyResult<PyPriceResult> {
        let option_type = parse_option_type(option_type)?;
        let payoff = AsianOption {
            option_type,
            strike,
        };
        let model = model.inner.clone();
        let engine = self.inner.clone();
        let inner = py
            .detach(move || {
                if control {
                    let cv = EuropeanControl::new(&model, option_type, strike);
                    engine.try_price_with_control(&model, &payoff, &cv)
                } else {
                    engine.try_price(&model, &payoff)
                }
            })
            .map_err(to_py_err)?;
        Ok(PyPriceResult { inner })
    }

    /// Price a discretely-monitored barrier option. `kind` is one of
    /// `'up_and_out'`, `'up_and_in'`, `'down_and_out'`, `'down_and_in'`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (model, option_type, kind, strike, barrier, rebate = 0.0, control = false))]
    fn price_barrier(
        &self,
        py: Python<'_>,
        model: PyRef<'_, PyGbmModel>,
        option_type: &str,
        kind: &str,
        strike: f64,
        barrier: f64,
        rebate: f64,
        control: bool,
    ) -> PyResult<PyPriceResult> {
        let option_type = parse_option_type(option_type)?;
        let payoff = BarrierOption {
            option_type,
            kind: parse_barrier_kind(kind)?,
            strike,
            barrier,
            rebate,
        };
        let model = model.inner.clone();
        let engine = self.inner.clone();
        let inner = py
            .detach(move || {
                if control {
                    let cv = EuropeanControl::new(&model, option_type, strike);
                    engine.try_price_with_control(&model, &payoff, &cv)
                } else {
                    engine.try_price(&model, &payoff)
                }
            })
            .map_err(to_py_err)?;
        Ok(PyPriceResult { inner })
    }

    /// Price a lookback option. Pass `strike=None` for floating-strike,
    /// or a numeric strike for fixed-strike.
    #[pyo3(signature = (model, option_type, strike = None))]
    fn price_lookback(
        &self,
        py: Python<'_>,
        model: PyRef<'_, PyGbmModel>,
        option_type: &str,
        strike: Option<f64>,
    ) -> PyResult<PyPriceResult> {
        let payoff = LookbackOption {
            option_type: parse_option_type(option_type)?,
            strike,
        };
        let model = model.inner.clone();
        let engine = self.inner.clone();
        let inner = py
            .detach(move || engine.try_price(&model, &payoff))
            .map_err(to_py_err)?;
        Ok(PyPriceResult { inner })
    }

    /// Price a cliquet (ratchet) option with local & global cap/floor.
    #[pyo3(signature = (
        model,
        notional,
        local_floor,
        local_cap,
        global_floor = 0.0,
        global_cap = f64::INFINITY,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn price_cliquet(
        &self,
        py: Python<'_>,
        model: PyRef<'_, PyGbmModel>,
        notional: f64,
        local_floor: f64,
        local_cap: f64,
        global_floor: f64,
        global_cap: f64,
    ) -> PyResult<PyPriceResult> {
        let payoff = CliquetOption {
            notional,
            local_floor,
            local_cap,
            global_floor,
            global_cap,
        };
        let model = model.inner.clone();
        let engine = self.inner.clone();
        let inner = py
            .detach(move || engine.try_price(&model, &payoff))
            .map_err(to_py_err)?;
        Ok(PyPriceResult { inner })
    }

    /// Price an autocallable note. `observation_indices` is a list of
    /// 1-based path indices where the spot is observed; the last index
    /// is treated as maturity.
    #[pyo3(signature = (
        model,
        notional,
        coupon_per_period,
        autocall_barrier,
        protection_barrier,
        observation_indices,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn price_autocallable(
        &self,
        py: Python<'_>,
        model: PyRef<'_, PyGbmModel>,
        notional: f64,
        coupon_per_period: f64,
        autocall_barrier: f64,
        protection_barrier: f64,
        observation_indices: Vec<usize>,
    ) -> PyResult<PyPriceResult> {
        let payoff = AutocallableNote {
            notional,
            coupon_per_period,
            autocall_barrier,
            protection_barrier,
            observation_indices,
        };
        let model = model.inner.clone();
        let engine = self.inner.clone();
        let inner = py
            .detach(move || engine.try_price(&model, &payoff))
            .map_err(to_py_err)?;
        Ok(PyPriceResult { inner })
    }

    /// Greeks for a vanilla European by bump-and-revalue.
    ///
    /// All revaluations share this engine's seed, so the bumped and
    /// unbumped runs are driven by the same noise and the difference
    /// between them reflects the bump rather than sampling error. Vega and
    /// rho are per unit (not per point or basis point), and `theta` is
    /// dV/dT, so a longer-dated option gives a positive value.
    #[pyo3(signature = (model, option_type, strike))]
    fn greeks_european(
        &self,
        py: Python<'_>,
        model: PyRef<'_, PyGbmModel>,
        option_type: &str,
        strike: f64,
    ) -> PyResult<PyGreeks> {
        let payoff = EuropeanOption {
            option_type: parse_option_type(option_type)?,
            strike,
        };
        let model = model.inner.clone();
        let engine = self.inner.clone();
        let inner = py
            .detach(move || {
                let bumps = BumpSizes::for_model(&model);
                try_bump_and_revalue(&engine, &model, &payoff, bumps)
            })
            .map_err(to_py_err)?;
        Ok(PyGreeks { inner })
    }

    fn __repr__(&self) -> String {
        let e = &self.inner;
        format!(
            "McEngine(paths={}, seed={}, antithetic={}, parallel={})",
            e.paths, e.seed, e.antithetic, e.parallel,
        )
    }
}

#[pymodule]
fn fuzzy_enigma(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGbmModel>()?;
    m.add_class::<PyMcEngine>()?;
    m.add_class::<PyPriceResult>()?;
    m.add_class::<PyGreeks>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
