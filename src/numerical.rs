//! Numerical tools for scheduled Asians and continuously monitored GBM barriers.
//!
//! The bridge estimator integrates out crossings between simulated fixings.
//! It applies to a single constant barrier under GBM, not to contractual
//! discrete monitoring dates or stochastic-volatility paths.

use crate::analytic::norm_cdf;
use crate::error::PricingError;
use crate::model::GbmModel;
use crate::payoff::{validate_observation_indices, BarrierKind, BarrierOption, OptionType, Payoff};
use crate::pricer::{ControlVariate, McEngine, PriceResult};

/// Arithmetic Asian using explicitly selected grid indices, excluding t = 0.
///
/// The average uses only the supplied observations and settles at model
/// maturity, even if its last fixing is earlier. To refine the simulation
/// grid while preserving the contract, scale the observation indices so
/// their dates remain unchanged.
#[derive(Clone, Debug)]
pub struct ScheduledAsianOption {
    pub option_type: OptionType,
    pub strike: f64,
    pub observation_indices: Vec<usize>,
}

impl Payoff for ScheduledAsianOption {
    fn evaluate(&self, path: &[f64], dt: f64, r: f64) -> f64 {
        let average = self
            .observation_indices
            .iter()
            .map(|&index| path[index])
            .sum::<f64>()
            / self.observation_indices.len() as f64;
        let maturity = dt * (path.len() - 1) as f64;
        (-r * maturity).exp() * self.option_type.intrinsic(average, self.strike)
    }

    fn validate(&self, steps: usize) -> Result<(), PricingError> {
        validate_positive("strike", self.strike)?;
        validate_observation_indices(&self.observation_indices, steps, false)
    }
}

fn validate_positive(name: &'static str, value: f64) -> Result<(), PricingError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(PricingError::invalid(name, "must be finite and positive"));
    }
    Ok(())
}

fn validate_gbm(model: &GbmModel) -> Result<(), PricingError> {
    GbmModel::try_new(
        model.spot,
        model.risk_free_rate,
        model.dividend_yield,
        model.volatility,
        model.maturity,
        model.steps,
    )?;
    Ok(())
}

/// Closed-form GBM price of a geometric-average Asian on selected dates.
///
/// If `t_i` are the observation dates, its log average is Gaussian with
/// mean `ln(S) + (r - q - sigma²/2) mean(t_i)` and variance
/// `sigma² sum(min(t_i, t_j)) / n²`. Settlement is at model maturity.
/// A single terminal observation reduces to Black-Scholes.
pub fn geometric_asian_price(
    model: &GbmModel,
    option_type: OptionType,
    strike: f64,
    observation_indices: &[usize],
) -> Result<f64, PricingError> {
    validate_gbm(model)?;
    validate_positive("strike", strike)?;
    validate_observation_indices(observation_indices, model.steps, false)?;

    let dt = model.maturity / model.steps as f64;
    let count = observation_indices.len() as f64;
    let mean_time = observation_indices
        .iter()
        .map(|&i| i as f64 * dt)
        .sum::<f64>()
        / count;
    // Sorted times give an O(n), rather than O(n²), covariance sum.
    let covariance_sum = observation_indices
        .iter()
        .enumerate()
        .map(|(i, &index)| (2.0 * (count - i as f64) - 1.0) * index as f64 * dt)
        .sum::<f64>();
    let variance = model.volatility.powi(2) * covariance_sum / count.powi(2);
    let mean = model.spot.ln()
        + (model.risk_free_rate - model.dividend_yield - 0.5 * model.volatility.powi(2))
            * mean_time;
    let discount = (-model.risk_free_rate * model.maturity).exp();
    if variance == 0.0 {
        return Ok(discount * option_type.intrinsic(mean.exp(), strike));
    }
    let std_dev = variance.sqrt();
    let d2 = (mean - strike.ln()) / std_dev;
    let d1 = d2 + std_dev;
    let forward = (mean + 0.5 * variance).exp();
    let undiscounted = match option_type {
        OptionType::Call => forward * norm_cdf(d1) - strike * norm_cdf(d2),
        OptionType::Put => strike * norm_cdf(-d2) - forward * norm_cdf(-d1),
    };
    Ok(discount * undiscounted.max(0.0))
}

/// Geometric Asian control with the same fixing schedule as the arithmetic
/// Asian being priced. Construct it from the current GBM model and rebuild
/// it for every revaluation, including Greeks, so its expectation remains
/// consistent with the simulated paths.
#[derive(Clone, Debug)]
pub struct GeometricAsianControl {
    option_type: OptionType,
    strike: f64,
    observation_indices: Vec<usize>,
    steps: usize,
    expectation: f64,
}

impl GeometricAsianControl {
    pub fn new(
        model: &GbmModel,
        option_type: OptionType,
        strike: f64,
        observation_indices: Vec<usize>,
    ) -> Self {
        Self::try_new(model, option_type, strike, observation_indices)
            .expect("invalid geometric Asian control")
    }

    pub fn try_new(
        model: &GbmModel,
        option_type: OptionType,
        strike: f64,
        observation_indices: Vec<usize>,
    ) -> Result<Self, PricingError> {
        let expectation = geometric_asian_price(model, option_type, strike, &observation_indices)?;
        Ok(Self {
            option_type,
            strike,
            observation_indices,
            steps: model.steps,
            expectation,
        })
    }
}

impl ControlVariate for GeometricAsianControl {
    fn evaluate(&self, path: &[f64], dt: f64, r: f64) -> f64 {
        let log_average = self
            .observation_indices
            .iter()
            .map(|&index| path[index].ln())
            .sum::<f64>()
            / self.observation_indices.len() as f64;
        let maturity = dt * (path.len() - 1) as f64;
        (-r * maturity).exp() * self.option_type.intrinsic(log_average.exp(), self.strike)
    }

    fn expectation(&self) -> f64 {
        self.expectation
    }

    fn validate(&self, steps: usize) -> Result<(), PricingError> {
        if steps != self.steps {
            return Err(PricingError::invalid(
                "control",
                "must be rebuilt when the model grid changes",
            ));
        }
        validate_observation_indices(&self.observation_indices, steps, false)
    }
}

struct ContinuousBarrier<'a> {
    payoff: &'a BarrierOption,
    volatility: f64,
}

impl Payoff for ContinuousBarrier<'_> {
    fn evaluate(&self, path: &[f64], dt: f64, r: f64) -> f64 {
        let is_up = matches!(
            self.payoff.kind,
            BarrierKind::UpAndOut | BarrierKind::UpAndIn
        );
        let hit_at_fixing = path.iter().any(|&s| {
            if is_up {
                s >= self.payoff.barrier
            } else {
                s <= self.payoff.barrier
            }
        });
        let survival = if hit_at_fixing {
            0.0
        } else if self.volatility == 0.0 {
            // Zero-vol GBM has a monotonic log path; safe endpoints imply
            // safe intervals without dividing by a zero diffusion.
            1.0
        } else {
            let log_barrier = self.payoff.barrier.ln();
            let interval_variance = self.volatility.powi(2) * dt;
            path.windows(2)
                .map(|pair| {
                    let a = pair[0].ln() - log_barrier;
                    let b = pair[1].ln() - log_barrier;
                    let exponent = -2.0 * a * b / interval_variance;
                    // 1 - exp(exponent), stable even for near-barrier
                    // endpoints; summing logs avoids repeated underflow.
                    (-exponent.exp_m1()).ln()
                })
                .sum::<f64>()
                .exp()
        };
        let alive_weight = match self.payoff.kind {
            BarrierKind::UpAndOut | BarrierKind::DownAndOut => survival,
            BarrierKind::UpAndIn | BarrierKind::DownAndIn => 1.0 - survival,
        };
        let maturity = dt * (path.len() - 1) as f64;
        let intrinsic = self
            .payoff
            .option_type
            .intrinsic(path[path.len() - 1], self.payoff.strike);
        (-r * maturity).exp()
            * (alive_weight * intrinsic + (1.0 - alive_weight) * self.payoff.rebate)
    }

    fn validate(&self, _steps: usize) -> Result<(), PricingError> {
        validate_positive("strike", self.payoff.strike)?;
        validate_positive("barrier", self.payoff.barrier)?;
        if !self.payoff.rebate.is_finite() || self.payoff.rebate < 0.0 {
            return Err(PricingError::invalid(
                "rebate",
                "must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

impl McEngine {
    /// Continuous GBM single-barrier price with Brownian bridge weighting.
    /// Rebates settle at maturity for both knock-in and knock-out options.
    pub fn try_price_continuous_barrier(
        &self,
        model: &GbmModel,
        payoff: &BarrierOption,
    ) -> Result<PriceResult, PricingError> {
        validate_gbm(model)?;
        self.try_price(
            model,
            &ContinuousBarrier {
                payoff,
                volatility: model.volatility,
            },
        )
    }

    /// Panicking convenience form of [`Self::try_price_continuous_barrier`].
    pub fn price_continuous_barrier(
        &self,
        model: &GbmModel,
        payoff: &BarrierOption,
    ) -> PriceResult {
        self.try_price_continuous_barrier(model, payoff)
            .expect("invalid continuous barrier pricing setup")
    }
}
