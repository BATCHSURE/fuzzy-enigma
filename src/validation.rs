//! Variance-conditioned Heston vanilla estimation for rare-tail verification.
//!
//! For each time step, rotate the original independent normals into the
//! variance driver U = rho Z_s + sqrt(1-rho²) Z_perp and its orthogonal
//! stock driver. Conditional on the full U path, the terminal log stock
//! is Gaussian. Integrating the orthogonal stock driver prices exactly
//! the same log-Euler/full-truncation discretisation as `HestonModel`.
//!
//! Defensive Gaussian shifts of U cover rare variance tails. Their exact
//! target/mixture likelihood ratio keeps the estimator unbiased. The
//! default shifts are 0, -4 and +4 along the normalised terminal Brownian
//! direction; `[0.0]` selects unshifted conditional Monte Carlo.

use crate::analytic::norm_cdf;
use crate::error::PricingError;
use crate::model::HestonModel;
use crate::payoff::{EuropeanOption, OptionType};
use crate::pricer::{summarise, validate_model, McEngine, PriceResult};

const DEFAULT_SHIFTS: [f64; 3] = [0.0, -4.0, 4.0];

fn lognormal_price(option: &EuropeanOption, log_forward: f64, variance: f64, discount: f64) -> f64 {
    if !log_forward.is_finite() || !variance.is_finite() || variance < 0.0 || !discount.is_finite()
    {
        return f64::NAN;
    }
    let forward = log_forward.exp();
    if !forward.is_finite() {
        return f64::NAN;
    }
    if variance == 0.0 {
        return discount * option.option_type.intrinsic(forward, option.strike);
    }
    let std_dev = variance.sqrt();
    let d1 = (log_forward - option.strike.ln()) / std_dev + 0.5 * std_dev;
    let d2 = d1 - std_dev;
    // Evaluate the OTM side with tail probabilities and then apply parity
    // for the ITM side, avoiding subtraction of CDFs near one.
    let call_otm = forward * norm_cdf(d1) - option.strike * norm_cdf(d2);
    let put_otm = option.strike * norm_cdf(-d2) - forward * norm_cdf(-d1);
    let value = match (option.option_type, forward >= option.strike) {
        (OptionType::Call, false) => call_otm,
        (OptionType::Call, true) => put_otm + forward - option.strike,
        (OptionType::Put, true) => put_otm,
        (OptionType::Put, false) => call_otm + option.strike - forward,
    };
    discount * value.max(0.0)
}

fn conditioned_sample(
    model: &HestonModel,
    option: &EuropeanOption,
    normals: &[f64],
    shift: f64,
    shifts: &[f64],
    proportions: &[f64],
    noise_sign: f64,
) -> Result<f64, PricingError> {
    let dt = model.maturity / model.steps as f64;
    let direction = 1.0 / (model.steps as f64).sqrt();
    let rho_complement = (1.0 - model.rho * model.rho).max(0.0).sqrt();
    let mut raw = model.initial_variance;
    let mut integrated_variance = 0.0;
    let mut correlated_log_noise = 0.0;
    let mut terminal_driver = 0.0;
    for pair in normals.chunks_exact(2) {
        let u = noise_sign * (model.rho * pair[0] + rho_complement * pair[1]) + shift * direction;
        let (variance, diffusion, next_raw) = model.variance_step(raw, dt, u);
        integrated_variance += variance * dt;
        correlated_log_noise += diffusion * u;
        terminal_driver += u * direction;
        raw = next_raw;
    }
    let log_forward = model.spot.ln()
        + (model.risk_free_rate - model.dividend_yield) * model.maturity
        + model.rho * correlated_log_noise
        - 0.5 * model.rho * model.rho * integrated_variance;
    let variance = (1.0 - model.rho * model.rho).max(0.0) * integrated_variance;
    let price = lognormal_price(
        option,
        log_forward,
        variance,
        (-model.risk_free_rate * model.maturity).exp(),
    );

    // q(U)/p(U) = sum(alpha_j exp(lambda_j L - lambda_j²/2)).
    // The zero-shift stratum is always active, so the denominator has a
    // positive finite contribution and the importance weight is bounded.
    let exponents: Vec<f64> = shifts
        .iter()
        .zip(proportions)
        .filter(|(_, alpha)| **alpha > 0.0)
        .map(|(&lambda, &alpha)| alpha.ln() + lambda * terminal_driver - 0.5 * lambda * lambda)
        .collect();
    let maximum = exponents.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_denominator = maximum
        + exponents
            .iter()
            .map(|exponent| (exponent - maximum).exp())
            .sum::<f64>()
            .ln();
    let weighted = price * (-log_denominator).exp();
    if !weighted.is_finite() || weighted < 0.0 {
        return Err(PricingError::invalid(
            "conditional_price",
            "variance-conditioned valuation overflowed",
        ));
    }
    Ok(weighted)
}

/// Price the core Heston Euler scheme with conditional integration and
/// a defensive variance-driver proposal mixture (0, -4, +4).
pub fn try_price_heston_conditional(
    engine: &McEngine,
    model: &HestonModel,
    option: &EuropeanOption,
) -> Result<PriceResult, PricingError> {
    try_price_heston_conditional_with_shifts(engine, model, option, &DEFAULT_SHIFTS)
}

/// Configure variance-driver shifts; `[0.0]` gives ordinary conditional MC.
///
/// Shifts must be finite, distinct, include zero, and have finite squares.
/// Zero is evaluated first so even small budgets retain the target-density
/// component. Stratum weights use the actual rounded sample counts. Each
/// independent sample is an antithetic pair when enabled. The pooled sample
/// SE includes between-stratum means and is conservative for this balanced
/// deterministic mixture; components are never counted as independent runs.
pub fn try_price_heston_conditional_with_shifts(
    engine: &McEngine,
    model: &HestonModel,
    option: &EuropeanOption,
    shifts: &[f64],
) -> Result<PriceResult, PricingError> {
    let (steps, noise_dim) = validate_model(model)?;
    if !option.strike.is_finite() || option.strike <= 0.0 {
        return Err(PricingError::invalid(
            "strike",
            "must be finite and positive",
        ));
    }
    if engine.paths == 0 {
        return Err(PricingError::NoPaths);
    }
    if shifts.is_empty()
        || !shifts.contains(&0.0)
        || shifts
            .iter()
            .any(|shift| !shift.is_finite() || !shift.powi(2).is_finite())
        || shifts
            .iter()
            .enumerate()
            .any(|(index, shift)| shifts[..index].contains(shift))
    {
        return Err(PricingError::invalid(
            "variance_shifts",
            "must be finite, distinct, include zero and have finite squares",
        ));
    }
    let n = if engine.antithetic {
        engine.paths.div_ceil(2)
    } else {
        engine.paths
    };
    // Deterministic variance can be integrated unconditionally, including
    // non-constant Euler variance when v0 differs from theta.
    if model.vol_of_vol == 0.0 || (model.initial_variance == 0.0 && model.theta == 0.0) {
        let dt = model.maturity / steps as f64;
        let mut raw = model.initial_variance;
        let mut integrated = 0.0;
        for _ in 0..steps {
            let (variance, _, next_raw) = model.variance_step(raw, dt, 0.0);
            integrated += variance * dt;
            raw = next_raw;
        }
        let price = lognormal_price(
            option,
            model.spot.ln() + (model.risk_free_rate - model.dividend_yield) * model.maturity,
            integrated,
            (-model.risk_free_rate * model.maturity).exp(),
        );
        if !price.is_finite() {
            return Err(PricingError::invalid(
                "conditional_price",
                "deterministic valuation overflowed",
            ));
        }
        return Ok(PriceResult {
            price,
            std_error: 0.0,
            samples: n,
        });
    }
    let ordered: Vec<f64> = std::iter::once(0.0)
        .chain(shifts.iter().copied().filter(|shift| *shift != 0.0))
        .collect();
    let proportions: Vec<f64> = (0..ordered.len())
        .map(|j| (n / ordered.len() + usize::from(j < n % ordered.len())) as f64 / n as f64)
        .collect();
    let values = engine.run_samples(n, steps, noise_dim, |normals, _path, index| {
        engine.fill_normals(normals, index);
        let shift = ordered[index % ordered.len()];
        let up = conditioned_sample(model, option, normals, shift, &ordered, &proportions, 1.0)?;
        if engine.antithetic {
            let down =
                conditioned_sample(model, option, normals, shift, &ordered, &proportions, -1.0)?;
            Ok(0.5 * (up + down))
        } else {
            Ok(up)
        }
    });
    let samples: Result<Vec<f64>, PricingError> = values.into_iter().collect();
    let result = summarise(&samples?);
    if !result.price.is_finite() || !result.std_error.is_finite() {
        return Err(PricingError::invalid(
            "conditional_price",
            "sample statistics overflowed",
        ));
    }
    Ok(result)
}

impl McEngine {
    pub fn try_price_heston_conditional(
        &self,
        model: &HestonModel,
        option: &EuropeanOption,
    ) -> Result<PriceResult, PricingError> {
        try_price_heston_conditional(self, model, option)
    }

    pub fn price_heston_conditional(
        &self,
        model: &HestonModel,
        option: &EuropeanOption,
    ) -> PriceResult {
        self.try_price_heston_conditional(model, option)
            .expect("invalid conditional Heston pricing setup")
    }

    pub fn try_price_heston_conditional_with_shifts(
        &self,
        model: &HestonModel,
        option: &EuropeanOption,
        shifts: &[f64],
    ) -> Result<PriceResult, PricingError> {
        try_price_heston_conditional_with_shifts(self, model, option, shifts)
    }
}
