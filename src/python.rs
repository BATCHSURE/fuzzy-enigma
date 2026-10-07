//! PyO3 bindings exposing the library as a Python extension module.
//!
//! Built via maturin: `maturin develop --release --features python`.
//! Compiled out of the default build, so plain `cargo build` is
//! unaffected.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::error::PricingError;
use crate::greeks::{
    try_bump_and_revalue, try_bump_and_revalue_with, try_continuous_barrier_greeks, BumpSizes,
    Greeks,
};
use crate::model::{GbmModel, HestonModel, PathGenerator};
use crate::numerical::{geometric_asian_price, GeometricAsianControl, ScheduledAsianOption};
use crate::payoff::{
    AsianOption, AutocallableNote, BarrierKind, BarrierOption, CliquetOption, EuropeanOption,
    LookbackOption, OptionType, Payoff,
};
use crate::pricer::{EuropeanControl, McEngine, PriceResult};
use crate::structured::{PhoenixNote, SnowballNote};

#[derive(Clone)]
enum PricingModel {
    Gbm(GbmModel),
    Heston(HestonModel),
}

impl PricingModel {
    fn gbm(&self) -> Result<&GbmModel, PricingError> {
        match self {
            Self::Gbm(model) => Ok(model),
            Self::Heston(_) => Err(PricingError::invalid(
                "model",
                "this method requires GbmModel",
            )),
        }
    }
}

impl PathGenerator for PricingModel {
    fn steps(&self) -> usize {
        match self {
            Self::Gbm(m) => m.steps(),
            Self::Heston(m) => m.steps(),
        }
    }
    fn noise_dim(&self) -> usize {
        match self {
            Self::Gbm(m) => m.noise_dim(),
            Self::Heston(m) => m.noise_dim(),
        }
    }
    fn maturity(&self) -> f64 {
        match self {
            Self::Gbm(m) => m.maturity(),
            Self::Heston(m) => m.maturity(),
        }
    }
    fn risk_free_rate(&self) -> f64 {
        match self {
            Self::Gbm(m) => m.risk_free_rate(),
            Self::Heston(m) => m.risk_free_rate(),
        }
    }
    fn validate(&self) -> Result<(), PricingError> {
        match self {
            Self::Gbm(m) => m.validate(),
            Self::Heston(m) => m.validate(),
        }
    }
    fn generate(&self, normals: &[f64], out: &mut [f64]) {
        match self {
            Self::Gbm(m) => m.generate(normals, out),
            Self::Heston(m) => m.generate(normals, out),
        }
    }
}

fn extract_model(model: &Bound<'_, PyAny>) -> PyResult<PricingModel> {
    if let Ok(model) = model.extract::<PyRef<'_, PyGbmModel>>() {
        return Ok(PricingModel::Gbm(model.inner.clone()));
    }
    if let Ok(model) = model.extract::<PyRef<'_, PyHestonModel>>() {
        return Ok(PricingModel::Heston(model.inner.clone()));
    }
    Err(PyValueError::new_err(
        "model must be GbmModel or HestonModel",
    ))
}

fn extract_gbm(model: &Bound<'_, PyAny>) -> PyResult<GbmModel> {
    extract_model(model)?.gbm().cloned().map_err(to_py_err)
}

fn run_price<P: Payoff + 'static>(
    py: Python<'_>,
    engine: &McEngine,
    model: &Bound<'_, PyAny>,
    payoff: P,
) -> PyResult<PyPriceResult> {
    let model = extract_model(model)?;
    let engine = engine.clone();
    let inner = py
        .detach(move || engine.try_price(&model, &payoff))
        .map_err(to_py_err)?;
    Ok(PyPriceResult { inner })
}

fn run_greeks<P: Payoff + 'static>(
    py: Python<'_>,
    engine: &McEngine,
    model: &Bound<'_, PyAny>,
    payoff: P,
) -> PyResult<PyGreeks> {
    let model = extract_gbm(model)?;
    let engine = engine.clone();
    let inner = py
        .detach(move || {
            try_bump_and_revalue(&engine, &model, &payoff, BumpSizes::for_model(&model))
        })
        .map_err(to_py_err)?;
    Ok(PyGreeks { inner })
}

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

#[derive(Clone, Copy)]
enum AsianControl {
    None,
    European,
    Geometric,
}

fn parse_asian_control(control: &str) -> PyResult<AsianControl> {
    match control.to_ascii_lowercase().as_str() {
        "none" => Ok(AsianControl::None),
        "european" => Ok(AsianControl::European),
        "geometric" => Ok(AsianControl::Geometric),
        _ => Err(PyValueError::new_err(
            "control must be 'none', 'european', or 'geometric'",
        )),
    }
}

fn scheduled_price(
    engine: &McEngine,
    model: &PricingModel,
    payoff: &ScheduledAsianOption,
    control: AsianControl,
) -> Result<PriceResult, PricingError> {
    match control {
        AsianControl::None => engine.try_price(model, payoff),
        AsianControl::European => {
            let cv = EuropeanControl::new(model.gbm()?, payoff.option_type, payoff.strike);
            engine.try_price_with_control(model, payoff, &cv)
        }
        AsianControl::Geometric => {
            let cv = GeometricAsianControl::try_new(
                model.gbm()?,
                payoff.option_type,
                payoff.strike,
                payoff.observation_indices.clone(),
            )?;
            engine.try_price_with_control(model, payoff, &cv)
        }
    }
}

/// Closed-form, discretely observed geometric Asian option under GBM.
#[pyfunction(name = "geometric_asian_price")]
fn py_geometric_asian_price(
    model: &Bound<'_, PyAny>,
    option_type: &str,
    strike: f64,
    observation_indices: Vec<usize>,
) -> PyResult<f64> {
    geometric_asian_price(
        &extract_gbm(model)?,
        parse_option_type(option_type)?,
        strike,
        &observation_indices,
    )
    .map_err(to_py_err)
}

/// Closed-form European price under GBM, for independent market regressions.
#[pyfunction(name = "black_scholes_price")]
fn py_black_scholes_price(
    model: &Bound<'_, PyAny>,
    option_type: &str,
    strike: f64,
) -> PyResult<f64> {
    let model = extract_gbm(model)?;
    model.validate().map_err(to_py_err)?;
    if !strike.is_finite() || strike <= 0.0 {
        return Err(PyValueError::new_err("strike must be finite and positive"));
    }
    Ok(crate::analytic::bs_price(
        parse_option_type(option_type)?,
        model.spot,
        strike,
        model.risk_free_rate,
        model.dividend_yield,
        model.volatility,
        model.maturity,
    ))
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
// HestonModel
// -----------------------------------------------------------------------

#[pyclass(name = "HestonModel", module = "fuzzy_enigma", from_py_object)]
#[derive(Clone)]
pub struct PyHestonModel {
    inner: HestonModel,
}

#[pymethods]
impl PyHestonModel {
    /// Single-asset Heston model, with full-truncation Euler variance.
    /// v0 and theta are variances; xi is volatility of variance.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (spot, r, q, v0, kappa, theta, xi, rho, t, steps = 252))]
    fn new(
        spot: f64,
        r: f64,
        q: f64,
        v0: f64,
        kappa: f64,
        theta: f64,
        xi: f64,
        rho: f64,
        t: f64,
        steps: usize,
    ) -> PyResult<Self> {
        let inner = HestonModel::try_new(spot, r, q, v0, kappa, theta, xi, rho, t, steps)
            .map_err(to_py_err)?;
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
    fn v0(&self) -> f64 {
        self.inner.initial_variance
    }
    #[getter]
    fn kappa(&self) -> f64 {
        self.inner.kappa
    }
    #[getter]
    fn theta(&self) -> f64 {
        self.inner.theta
    }
    #[getter]
    fn xi(&self) -> f64 {
        self.inner.vol_of_vol
    }
    #[getter]
    fn rho(&self) -> f64 {
        self.inner.rho
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
        format!("HestonModel(spot={}, r={}, q={}, v0={}, kappa={}, theta={}, xi={}, rho={}, t={}, steps={})",
            m.spot, m.risk_free_rate, m.dividend_yield, m.initial_variance,
            m.kappa, m.theta, m.vol_of_vol, m.rho, m.maturity, m.steps)
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
        model: &Bound<'_, PyAny>,
        option_type: &str,
        strike: f64,
    ) -> PyResult<PyPriceResult> {
        let payoff = EuropeanOption {
            option_type: parse_option_type(option_type)?,
            strike,
        };
        let model = extract_model(model)?;
        let engine = self.inner.clone();
        let inner = py
            .detach(move || engine.try_price(&model, &payoff))
            .map_err(to_py_err)?;
        Ok(PyPriceResult { inner })
    }

    /// Price a Heston European call/put with conditional Monte Carlo.
    ///
    /// The independent stock Brownian driver is integrated analytically.
    /// By default a defensive importance-sampling mixture also samples the
    /// variance-driver tails. Pass `variance_shifts=[0.0]` for conditional
    /// Monte Carlo alone. Each shift is the mean of the unit-length constant
    /// variance-driver projection, measured in standard deviations.
    /// The full-truncation variance scheme and its grid error are retained.
    #[pyo3(signature = (model, option_type, strike, variance_shifts = None))]
    fn price_heston_conditional(
        &self,
        py: Python<'_>,
        model: &Bound<'_, PyAny>,
        option_type: &str,
        strike: f64,
        variance_shifts: Option<Vec<f64>>,
    ) -> PyResult<PyPriceResult> {
        let model = model
            .extract::<PyRef<'_, PyHestonModel>>()
            .map_err(|_| PyValueError::new_err("this method requires HestonModel"))?
            .inner
            .clone();
        let payoff = EuropeanOption {
            option_type: parse_option_type(option_type)?,
            strike,
        };
        let engine = self.inner.clone();
        let inner = py
            .detach(move || match variance_shifts {
                Some(shifts) => {
                    engine.try_price_heston_conditional_with_shifts(&model, &payoff, &shifts)
                }
                None => engine.try_price_heston_conditional(&model, &payoff),
            })
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
        model: &Bound<'_, PyAny>,
        option_type: &str,
        strike: f64,
        control: bool,
    ) -> PyResult<PyPriceResult> {
        let option_type = parse_option_type(option_type)?;
        let payoff = AsianOption {
            option_type,
            strike,
        };
        let model = extract_model(model)?;
        let engine = self.inner.clone();
        let inner = py
            .detach(move || {
                if control {
                    let cv = EuropeanControl::new(model.gbm()?, option_type, strike);
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
        model: &Bound<'_, PyAny>,
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
        let model = extract_model(model)?;
        let engine = self.inner.clone();
        let inner = py
            .detach(move || {
                if control {
                    let cv = EuropeanControl::new(model.gbm()?, option_type, strike);
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
        model: &Bound<'_, PyAny>,
        option_type: &str,
        strike: Option<f64>,
    ) -> PyResult<PyPriceResult> {
        let payoff = LookbackOption {
            option_type: parse_option_type(option_type)?,
            strike,
        };
        let model = extract_model(model)?;
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
        model: &Bound<'_, PyAny>,
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
        let model = extract_model(model)?;
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
        model: &Bound<'_, PyAny>,
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
        let model = extract_model(model)?;
        let engine = self.inner.clone();
        let inner = py
            .detach(move || engine.try_price(&model, &payoff))
            .map_err(to_py_err)?;
        Ok(PyPriceResult { inner })
    }

    /// Snowball with grid-monitored knock-in and scheduled knock-out.
    /// coupon_rate is annualised; reference_spot is the fixed contract fixing.
    #[allow(clippy::too_many_arguments)]
    fn price_snowball(
        &self,
        py: Python<'_>,
        model: &Bound<'_, PyAny>,
        notional: f64,
        reference_spot: f64,
        coupon_rate: f64,
        knock_in_barrier: f64,
        knock_out_barrier: f64,
        observation_indices: Vec<usize>,
    ) -> PyResult<PyPriceResult> {
        run_price(
            py,
            &self.inner,
            model,
            SnowballNote {
                notional,
                reference_spot,
                coupon_rate,
                knock_in_barrier,
                knock_out_barrier,
                observation_indices,
            },
        )
    }

    /// Phoenix with conditional cash coupons and optional coupon memory.
    /// Coupons are settled before a same-day autocall; memory expires at maturity.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (model, notional, reference_spot, coupon_per_period,
        coupon_barrier, knock_in_barrier, knock_out_barrier, coupon_indices,
        autocall_indices, memory = false))]
    fn price_phoenix(
        &self,
        py: Python<'_>,
        model: &Bound<'_, PyAny>,
        notional: f64,
        reference_spot: f64,
        coupon_per_period: f64,
        coupon_barrier: f64,
        knock_in_barrier: f64,
        knock_out_barrier: f64,
        coupon_indices: Vec<usize>,
        autocall_indices: Vec<usize>,
        memory: bool,
    ) -> PyResult<PyPriceResult> {
        run_price(
            py,
            &self.inner,
            model,
            PhoenixNote {
                notional,
                reference_spot,
                coupon_per_period,
                coupon_barrier,
                knock_in_barrier,
                knock_out_barrier,
                coupon_indices,
                autocall_indices,
                memory,
            },
        )
    }

    /// GBM Greeks holding the Snowball's contractual reference fixing constant.
    #[allow(clippy::too_many_arguments)]
    fn greeks_snowball(
        &self,
        py: Python<'_>,
        model: &Bound<'_, PyAny>,
        notional: f64,
        reference_spot: f64,
        coupon_rate: f64,
        knock_in_barrier: f64,
        knock_out_barrier: f64,
        observation_indices: Vec<usize>,
    ) -> PyResult<PyGreeks> {
        run_greeks(
            py,
            &self.inner,
            model,
            SnowballNote {
                notional,
                reference_spot,
                coupon_rate,
                knock_in_barrier,
                knock_out_barrier,
                observation_indices,
            },
        )
    }

    /// GBM Phoenix Greeks. theta rescales the whole observation schedule.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (model, notional, reference_spot, coupon_per_period,
        coupon_barrier, knock_in_barrier, knock_out_barrier, coupon_indices,
        autocall_indices, memory = false))]
    fn greeks_phoenix(
        &self,
        py: Python<'_>,
        model: &Bound<'_, PyAny>,
        notional: f64,
        reference_spot: f64,
        coupon_per_period: f64,
        coupon_barrier: f64,
        knock_in_barrier: f64,
        knock_out_barrier: f64,
        coupon_indices: Vec<usize>,
        autocall_indices: Vec<usize>,
        memory: bool,
    ) -> PyResult<PyGreeks> {
        run_greeks(
            py,
            &self.inner,
            model,
            PhoenixNote {
                notional,
                reference_spot,
                coupon_per_period,
                coupon_barrier,
                knock_in_barrier,
                knock_out_barrier,
                coupon_indices,
                autocall_indices,
                memory,
            },
        )
    }

    /// Arithmetic Asian on explicit observation indices. Controls require GBM.
    #[pyo3(signature = (model, option_type, strike, observation_indices, control = "none"))]
    fn price_asian_scheduled(
        &self,
        py: Python<'_>,
        model: &Bound<'_, PyAny>,
        option_type: &str,
        strike: f64,
        observation_indices: Vec<usize>,
        control: &str,
    ) -> PyResult<PyPriceResult> {
        let payoff = ScheduledAsianOption {
            option_type: parse_option_type(option_type)?,
            strike,
            observation_indices,
        };
        let control = parse_asian_control(control)?;
        let model = extract_model(model)?;
        let engine = self.inner.clone();
        let inner = py
            .detach(move || scheduled_price(&engine, &model, &payoff, control))
            .map_err(to_py_err)?;
        Ok(PyPriceResult { inner })
    }

    /// GBM scheduled Asian Greeks, rebuilding the control expectation on each bump.
    #[pyo3(signature = (model, option_type, strike, observation_indices, control = "none"))]
    fn greeks_asian_scheduled(
        &self,
        py: Python<'_>,
        model: &Bound<'_, PyAny>,
        option_type: &str,
        strike: f64,
        observation_indices: Vec<usize>,
        control: &str,
    ) -> PyResult<PyGreeks> {
        let payoff = ScheduledAsianOption {
            option_type: parse_option_type(option_type)?,
            strike,
            observation_indices,
        };
        let control = parse_asian_control(control)?;
        let model = extract_gbm(model)?;
        let engine = self.inner.clone();
        let inner = py
            .detach(move || {
                try_bump_and_revalue_with(&model, BumpSizes::for_model(&model), |m| {
                    Ok(
                        scheduled_price(&engine, &PricingModel::Gbm(m.clone()), &payoff, control)?
                            .price,
                    )
                })
            })
            .map_err(to_py_err)?;
        Ok(PyGreeks { inner })
    }

    /// Continuously monitored GBM single barrier, with maturity-paid rebate.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (model, option_type, kind, strike, barrier, rebate = 0.0))]
    fn price_continuous_barrier(
        &self,
        py: Python<'_>,
        model: &Bound<'_, PyAny>,
        option_type: &str,
        kind: &str,
        strike: f64,
        barrier: f64,
        rebate: f64,
    ) -> PyResult<PyPriceResult> {
        let payoff = BarrierOption {
            option_type: parse_option_type(option_type)?,
            kind: parse_barrier_kind(kind)?,
            strike,
            barrier,
            rebate,
        };
        let model = extract_gbm(model)?;
        let engine = self.inner.clone();
        let inner = py
            .detach(move || engine.try_price_continuous_barrier(&model, &payoff))
            .map_err(to_py_err)?;
        Ok(PyPriceResult { inner })
    }

    /// GBM continuous-barrier Greeks, including the crossing probability's vega.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (model, option_type, kind, strike, barrier, rebate = 0.0))]
    fn greeks_continuous_barrier(
        &self,
        py: Python<'_>,
        model: &Bound<'_, PyAny>,
        option_type: &str,
        kind: &str,
        strike: f64,
        barrier: f64,
        rebate: f64,
    ) -> PyResult<PyGreeks> {
        let payoff = BarrierOption {
            option_type: parse_option_type(option_type)?,
            kind: parse_barrier_kind(kind)?,
            strike,
            barrier,
            rebate,
        };
        let model = extract_gbm(model)?;
        let engine = self.inner.clone();
        let inner = py
            .detach(move || {
                try_continuous_barrier_greeks(
                    &engine,
                    &model,
                    &payoff,
                    BumpSizes::for_model(&model),
                )
            })
            .map_err(to_py_err)?;
        Ok(PyGreeks { inner })
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
        model: &Bound<'_, PyAny>,
        option_type: &str,
        strike: f64,
    ) -> PyResult<PyGreeks> {
        let payoff = EuropeanOption {
            option_type: parse_option_type(option_type)?,
            strike,
        };
        let model = extract_gbm(model)?;
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
    m.add_class::<PyHestonModel>()?;
    m.add_class::<PyMcEngine>()?;
    m.add_class::<PyPriceResult>()?;
    m.add_class::<PyGreeks>()?;
    m.add_function(wrap_pyfunction!(py_geometric_asian_price, m)?)?;
    m.add_function(wrap_pyfunction!(py_black_scholes_price, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
