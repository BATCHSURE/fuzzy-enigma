//! Asset price models used to drive path simulation.
//!
//! Geometric Brownian Motion and Heston stochastic volatility share the
//! same single-asset path interface. The random-input dimension is
//! independent of the path length, allowing multi-factor dynamics.

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

    /// Number of independent standard normals needed for one path.
    /// Single-factor models use the default; Heston uses two per step.
    fn noise_dim(&self) -> usize {
        self.steps()
    }

    /// Validate model-specific parameters before a pricing run. The engine
    /// also checks the common time grid and discounting parameters.
    fn validate(&self) -> Result<(), PricingError> {
        Ok(())
    }

    /// Time to maturity in years.
    fn maturity(&self) -> f64;

    /// Risk-free rate used by the engine for discounting.
    fn risk_free_rate(&self) -> f64;

    /// Populate `out` with one path driven by the supplied normals.
    ///
    /// `normals.len()` must equal `self.noise_dim()` and `out.len()` must
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
        if steps.checked_add(1).is_none() {
            return Err(PricingError::invalid("steps", "path length overflows"));
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

    fn validate(&self) -> Result<(), PricingError> {
        Self::try_new(
            self.spot,
            self.risk_free_rate,
            self.dividend_yield,
            self.volatility,
            self.maturity,
            self.steps,
        )
        .map(|_| ())
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

/// Heston stochastic volatility under the risk-neutral measure:
///
/// `dS = (r-q) S dt + sqrt(v) S dW_s`,
/// `dv = kappa (theta-v) dt + vol_of_vol sqrt(v) dW_v`,
/// with `corr(dW_s, dW_v) = rho`.
///
/// Uses log-Euler for the stock and full-truncation Euler for variance.
/// The raw variance state may be negative; only its positive part enters
/// drift and diffusion. Finite time steps introduce discretisation bias.
/// The Feller condition is deliberately not enforced: full truncation
/// supports parameter sets in which the variance can reach zero.
#[derive(Clone, Debug)]
pub struct HestonModel {
    pub spot: f64,
    pub risk_free_rate: f64,
    pub dividend_yield: f64,
    pub initial_variance: f64,
    pub kappa: f64,
    pub theta: f64,
    pub vol_of_vol: f64,
    pub rho: f64,
    pub maturity: f64,
    pub steps: usize,
}

impl HestonModel {
    /// Construct a model, panicking on invalid parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        spot: f64,
        risk_free_rate: f64,
        dividend_yield: f64,
        initial_variance: f64,
        kappa: f64,
        theta: f64,
        vol_of_vol: f64,
        rho: f64,
        maturity: f64,
        steps: usize,
    ) -> Self {
        Self::try_new(
            spot,
            risk_free_rate,
            dividend_yield,
            initial_variance,
            kappa,
            theta,
            vol_of_vol,
            rho,
            maturity,
            steps,
        )
        .expect("invalid HestonModel parameters")
    }

    /// Construct a model with checked finite parameters. Zero initial or
    /// long-run variance and zero volatility of variance are valid.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        spot: f64,
        risk_free_rate: f64,
        dividend_yield: f64,
        initial_variance: f64,
        kappa: f64,
        theta: f64,
        vol_of_vol: f64,
        rho: f64,
        maturity: f64,
        steps: usize,
    ) -> Result<Self, PricingError> {
        // Reuse the common single-asset model parameter checks.
        GbmModel::try_new(spot, risk_free_rate, dividend_yield, 0.0, maturity, steps)?;
        for (name, value) in [
            ("initial_variance", initial_variance),
            ("theta", theta),
            ("vol_of_vol", vol_of_vol),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(PricingError::invalid(
                    name,
                    "must be finite and non-negative",
                ));
            }
        }
        if !kappa.is_finite() || kappa <= 0.0 {
            return Err(PricingError::invalid(
                "kappa",
                "must be finite and positive",
            ));
        }
        if !rho.is_finite() || !(-1.0..=1.0).contains(&rho) {
            return Err(PricingError::invalid("rho", "must be between -1 and 1"));
        }
        if steps.checked_mul(2).is_none() {
            return Err(PricingError::invalid("steps", "noise dimension overflows"));
        }
        Ok(Self {
            spot,
            risk_free_rate,
            dividend_yield,
            initial_variance,
            kappa,
            theta,
            vol_of_vol,
            rho,
            maturity,
            steps,
        })
    }
}

impl PathGenerator for HestonModel {
    fn steps(&self) -> usize {
        self.steps
    }

    fn noise_dim(&self) -> usize {
        self.steps.saturating_mul(2)
    }

    fn maturity(&self) -> f64 {
        self.maturity
    }

    fn risk_free_rate(&self) -> f64 {
        self.risk_free_rate
    }

    fn validate(&self) -> Result<(), PricingError> {
        Self::try_new(
            self.spot,
            self.risk_free_rate,
            self.dividend_yield,
            self.initial_variance,
            self.kappa,
            self.theta,
            self.vol_of_vol,
            self.rho,
            self.maturity,
            self.steps,
        )
        .map(|_| ())
    }

    fn generate(&self, normals: &[f64], out: &mut [f64]) {
        debug_assert_eq!(normals.len(), self.noise_dim());
        debug_assert_eq!(out.len(), self.steps + 1);
        let dt = self.maturity / self.steps as f64;
        let rho_complement = (1.0 - self.rho * self.rho).max(0.0).sqrt();
        let mut raw_variance = self.initial_variance;
        out[0] = self.spot;
        for i in 0..self.steps {
            let variance = raw_variance.max(0.0);
            let diffusion = (variance * dt).sqrt();
            let stock_normal = normals[2 * i];
            let variance_normal = self.rho * stock_normal + rho_complement * normals[2 * i + 1];
            out[i + 1] = out[i]
                * ((self.risk_free_rate - self.dividend_yield - 0.5 * variance) * dt
                    + diffusion * stock_normal)
                    .exp();
            raw_variance += self.kappa * (self.theta - variance) * dt
                + self.vol_of_vol * diffusion * variance_normal;
        }
    }
}
