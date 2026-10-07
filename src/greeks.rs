//! Greeks by bump-and-revalue under common random numbers.
//!
//! Every revaluation reuses the engine's seed, so the bumped and unbumped
//! runs share their driving noise path-for-path. The difference between
//! them is then almost entirely the effect of the bump rather than Monte
//! Carlo noise, which is what makes finite differences usable at all here:
//! independent runs would need orders of magnitude more paths to resolve a
//! small bump against their own sampling error.
//!
//! This relies on the engine deriving a distinct RNG *stream* per path
//! from a fixed key. It would be quietly wrong under a scheme where
//! changing the seed merely shifted the path sequence, because "the same
//! noise" would no longer mean the same thing across two runs.
//!
//! These are finite-difference estimates. Pathwise and likelihood-ratio
//! estimators converge better for payoffs that admit them, but they need
//! per-payoff derivative information, whereas bumping works on any
//! [`Payoff`] as-is.

use crate::error::PricingError;
use crate::model::GbmModel;
use crate::payoff::{BarrierOption, Payoff};
use crate::pricer::McEngine;

/// First- and second-order sensitivities of a price.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Greeks {
    /// `dV/dS`, sensitivity to the underlying's spot.
    pub delta: f64,
    /// `d2V/dS2`, curvature in the spot.
    pub gamma: f64,
    /// `dV/dsigma`, per unit of volatility (not per volatility point).
    pub vega: f64,
    /// `dV/dT`, with all grid observation dates scaled proportionally
    /// with maturity. Negate it for the usual calendar-theta sign.
    pub theta: f64,
    /// `dV/dr`, per unit of rate (not per basis point).
    pub rho: f64,
}

/// Bump sizes for each finite difference, in absolute units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BumpSizes {
    /// Absolute spot bump. Defaults to 1% of spot in [`BumpSizes::for_model`].
    pub spot: f64,
    /// Absolute volatility bump, e.g. `0.01` for one vol point.
    pub volatility: f64,
    /// Absolute rate bump, e.g. `1e-4` for a basis point.
    pub rate: f64,
    /// Absolute maturity bump in years.
    pub maturity: f64,
}

impl BumpSizes {
    /// Bump sizes scaled to a model: 1% of spot, one vol point, one basis
    /// point of rate, and one 252nd of a year.
    ///
    /// Finite differences trade truncation error (bump too large) against
    /// cancellation noise (bump too small). Under common random numbers
    /// the noise term is heavily suppressed, so these lean small.
    pub fn for_model(model: &GbmModel) -> Self {
        Self {
            spot: 0.01 * model.spot,
            volatility: 0.01,
            rate: 1e-4,
            maturity: (1.0_f64 / 252.0).min(0.5 * model.maturity),
        }
    }
}

/// Compute Greeks by bump-and-revalue, panicking on an invalid setup.
///
/// See [`try_bump_and_revalue`] for the fallible form.
pub fn bump_and_revalue<P>(
    engine: &McEngine,
    model: &GbmModel,
    payoff: &P,
    bumps: BumpSizes,
) -> Greeks
where
    P: Payoff + ?Sized,
{
    try_bump_and_revalue(engine, model, payoff, bumps).expect("invalid pricing setup")
}

/// Compute Greeks by bump-and-revalue.
///
/// Delta, gamma, vega and rho use central differences; theta uses a
/// backward difference in maturity. The step count and observation indices
/// stay fixed, so every observation time scales proportionally with `T`.
///
/// Every revaluation runs at the engine's own seed. Do not pass an engine
/// whose seed varies between calls: the cancellation that makes these
/// estimates usable depends on the noise being shared.
pub fn try_bump_and_revalue<P>(
    engine: &McEngine,
    model: &GbmModel,
    payoff: &P,
    bumps: BumpSizes,
) -> Result<Greeks, PricingError>
where
    P: Payoff + ?Sized,
{
    try_bump_and_revalue_with(model, bumps, |m| Ok(engine.try_price(m, payoff)?.price))
}

/// Compute GBM Greeks using a model-aware repricer.
///
/// The repricer must reuse the same engine seed, and reconstruct any
/// model-dependent control expectation or bridge payoff for every call.
/// The base model, bumps, and every perturbed model are validated before
/// the first pricing run. The low-volatility down leg is clamped at zero.
pub fn try_bump_and_revalue_with<F>(
    model: &GbmModel,
    bumps: BumpSizes,
    reprice: F,
) -> Result<Greeks, PricingError>
where
    F: Fn(&GbmModel) -> Result<f64, PricingError>,
{
    for (name, value) in [
        ("spot_bump", bumps.spot),
        ("volatility_bump", bumps.volatility),
        ("rate_bump", bumps.rate),
        ("maturity_bump", bumps.maturity),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(PricingError::invalid(name, "must be finite and positive"));
        }
    }
    let checked_model = |m: GbmModel| {
        GbmModel::try_new(
            m.spot,
            m.risk_free_rate,
            m.dividend_yield,
            m.volatility,
            m.maturity,
            m.steps,
        )
    };
    let base_model = checked_model(model.clone())?;
    let spot_up = checked_model(GbmModel {
        spot: model.spot + bumps.spot,
        ..model.clone()
    })?;
    let spot_down = checked_model(GbmModel {
        spot: model.spot - bumps.spot,
        ..model.clone()
    })?;
    let vol_up = checked_model(GbmModel {
        volatility: model.volatility + bumps.volatility,
        ..model.clone()
    })?;
    let vol_down = checked_model(GbmModel {
        volatility: (model.volatility - bumps.volatility).max(0.0),
        ..model.clone()
    })?;
    let rate_up = checked_model(GbmModel {
        risk_free_rate: model.risk_free_rate + bumps.rate,
        ..model.clone()
    })?;
    let rate_down = checked_model(GbmModel {
        risk_free_rate: model.risk_free_rate - bumps.rate,
        ..model.clone()
    })?;
    let shorter = checked_model(GbmModel {
        maturity: model.maturity - bumps.maturity,
        ..model.clone()
    })?;

    // Avoid a formally positive bump that rounds away in floating point
    // or leaves an unrepresentable denominator.
    if spot_up.spot == model.spot
        || spot_down.spot == model.spot
        || !bumps.spot.powi(2).is_finite()
        || bumps.spot.powi(2) == 0.0
    {
        return Err(PricingError::invalid(
            "spot_bump",
            "must produce finite distinct spot legs",
        ));
    }
    if vol_up.volatility == model.volatility
        || (vol_up.volatility - vol_down.volatility).is_infinite()
    {
        return Err(PricingError::invalid(
            "volatility_bump",
            "must produce finite distinct volatility legs",
        ));
    }
    if rate_up.risk_free_rate == model.risk_free_rate
        || rate_down.risk_free_rate == model.risk_free_rate
        || !(2.0 * bumps.rate).is_finite()
    {
        return Err(PricingError::invalid(
            "rate_bump",
            "must produce finite distinct rate legs",
        ));
    }
    if shorter.maturity == model.maturity {
        return Err(PricingError::invalid(
            "maturity_bump",
            "must produce a distinct shorter maturity",
        ));
    }
    let price_with = |m: &GbmModel| -> Result<f64, PricingError> {
        let price = reprice(m)?;
        if !price.is_finite() {
            return Err(PricingError::invalid(
                "price",
                "repricer must return a finite value",
            ));
        }
        Ok(price)
    };
    let base = price_with(&base_model)?;
    let h_s = bumps.spot;
    let up = price_with(&spot_up)?;
    let down = price_with(&spot_down)?;
    let delta = (up - down) / (2.0 * h_s);
    let gamma = (up - 2.0 * base + down) / (h_s * h_s);

    // Volatility cannot go negative, so for a very low base vol the down
    // leg is clamped at zero and the difference is divided by the spread
    // actually used rather than the nominal 2h.
    let vega =
        (price_with(&vol_up)? - price_with(&vol_down)?) / (vol_up.volatility - vol_down.volatility);

    let h_r = bumps.rate;
    let rho = (price_with(&rate_up)? - price_with(&rate_down)?) / (2.0 * h_r);

    let h_t = bumps.maturity;
    let theta = (base - price_with(&shorter)?) / h_t;

    if [delta, gamma, vega, theta, rho]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(PricingError::invalid(
            "greeks",
            "finite differences overflowed; adjust pricing parameters or bumps",
        ));
    }

    Ok(Greeks {
        delta,
        gamma,
        vega,
        theta,
        rho,
    })
}

/// GBM continuously monitored barrier Greeks. Each bumped valuation
/// rebuilds the bridge weights using that leg's volatility.
pub fn try_continuous_barrier_greeks(
    engine: &McEngine,
    model: &GbmModel,
    payoff: &BarrierOption,
    bumps: BumpSizes,
) -> Result<Greeks, PricingError> {
    try_bump_and_revalue_with(model, bumps, |m| {
        Ok(engine.try_price_continuous_barrier(m, payoff)?.price)
    })
}
