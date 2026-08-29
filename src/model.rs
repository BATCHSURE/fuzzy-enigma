//! Asset price models used to drive path simulation.
//!
//! Only a single-factor Geometric Brownian Motion is provided here, but
//! the [`PathGenerator`] trait is intentionally minimal so additional
//! models (Heston, local-vol, jump-diffusion, multi-asset baskets) can be
//! plugged into the same Monte Carlo engine by implementing it.

use crate::error::PricingError;

/// A path generator turns a slice of standard-normal increments into one
/// realisation of the underlying asset price.
///
/// Keeping the noise external to the model is what allows the pricing
/// engine to apply antithetic variates (re-running the same path with
/// negated noise) without per-model support.
pub trait PathGenerator: Send + Sync {
    /// Number of time steps in each path. The output buffer must hold
    /// `steps + 1` values: `path[0]` is the spot, `path[steps]` is the
    /// terminal price.
    fn steps(&self) -> usize;

    /// Time to maturity in years.
    fn maturity(&self) -> f64;

    /// Risk-free rate used by the engine for discounting.
    fn risk_free_rate(&self) -> f64;

    /// Populate `out` with one path driven by the supplied normals.
    ///
    /// `normals.len()` must equal `self.steps()` and `out.len()` must
    /// equal `self.steps() + 1`.
    fn generate(&self, normals: &[f64], out: &mut [f64]);
}

/// Geometric Brownian Motion under the risk-neutral measure:
///
/// `dS_t = (r - q) S_t dt + sigma S_t dW_t`.
///
/// Simulated exactly with the log-Euler scheme so step size does not
/// introduce discretisation bias for vanilla and most exotic payoffs
/// (continuously-monitored barriers still pick up the usual discrete
/// monitoring bias - see `BarrierOption` for notes).
#[derive(Clone, Debug)]
pub struct GbmModel {
    pub spot: f64,
    pub risk_free_rate: f64,
    pub dividend_yield: f64,
    pub volatility: f64,
    pub maturity: f64,
    pub steps: usize,
}

impl GbmModel {
    /// Construct a model, panicking on invalid parameters.
    ///
    /// Use [`GbmModel::try_new`] when the parameters come from outside the
    /// program (a config file, a Python caller) and an error is more useful
    /// than a panic.
    pub fn new(
        spot: f64,
        risk_free_rate: f64,
        dividend_yield: f64,
        volatility: f64,
        maturity: f64,
        steps: usize,
    ) -> Self {
        Self::try_new(
            spot,
            risk_free_rate,
            dividend_yield,
            volatility,
            maturity,
            steps,
        )
        .expect("invalid GbmModel parameters")
    }

    /// Construct a model, returning [`PricingError`] on invalid parameters.
    pub fn try_new(
        spot: f64,
        risk_free_rate: f64,
        dividend_yield: f64,
        volatility: f64,
        maturity: f64,
        steps: usize,
    ) -> Result<Self, PricingError> {
        // Each check tests `is_finite` first so a NaN is rejected rather
        // than slipping through: every comparison against NaN is false, so
        // a bare `spot <= 0.0` would wave it past.
        if !spot.is_finite() || spot <= 0.0 {
            return Err(PricingError::invalid("spot", "must be positive"));
        }
        if !volatility.is_finite() || volatility < 0.0 {
            return Err(PricingError::invalid("volatility", "must be non-negative"));
        }
        if !maturity.is_finite() || maturity <= 0.0 {
            return Err(PricingError::invalid("maturity", "must be positive"));
        }
        if steps < 1 {
            return Err(PricingError::invalid("steps", "must be at least 1"));
        }
        if !risk_free_rate.is_finite() {
            return Err(PricingError::invalid("risk_free_rate", "must be finite"));
        }
        if !dividend_yield.is_finite() {
            return Err(PricingError::invalid("dividend_yield", "must be finite"));
        }
        Ok(Self {
            spot,
            risk_free_rate,
            dividend_yield,
            volatility,
            maturity,
            steps,
        })
    }
}

impl PathGenerator for GbmModel {
    fn steps(&self) -> usize {
        self.steps
    }
    fn maturity(&self) -> f64 {
        self.maturity
    }
    fn risk_free_rate(&self) -> f64 {
        self.risk_free_rate
    }

    fn generate(&self, normals: &[f64], out: &mut [f64]) {
        debug_assert_eq!(normals.len(), self.steps);
        debug_assert_eq!(out.len(), self.steps + 1);
        let dt = self.maturity / self.steps as f64;
        let drift =
            (self.risk_free_rate - self.dividend_yield - 0.5 * self.volatility * self.volatility)
                * dt;
        let diffusion = self.volatility * dt.sqrt();
        out[0] = self.spot;
        for i in 0..self.steps {
            out[i + 1] = out[i] * (drift + diffusion * normals[i]).exp();
        }
    }
}
