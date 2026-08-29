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
use crate::payoff::Payoff;
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
    /// `dV/dT`, the change in value as maturity lengthens. Negate it for
    /// the usual "value lost per unit of calendar time" sign convention.
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
/// backward difference, since stepping maturity forward past `T` would
/// re-time every observation date of a path-dependent payoff.
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
    let price_with =
        |m: &GbmModel| -> Result<f64, PricingError> { Ok(engine.try_price(m, payoff)?.price) };

    let with_spot = |s: f64| GbmModel {
        spot: s,
        ..model.clone()
    };
    let with_vol = |v: f64| GbmModel {
        volatility: v,
        ..model.clone()
    };
    let with_rate = |r: f64| GbmModel {
        risk_free_rate: r,
        ..model.clone()
    };
    let with_maturity = |t: f64| GbmModel {
        maturity: t,
        ..model.clone()
    };

    let base = price_with(model)?;

    let h_s = bumps.spot;
    let up = price_with(&with_spot(model.spot + h_s))?;
    let down = price_with(&with_spot(model.spot - h_s))?;
    let delta = (up - down) / (2.0 * h_s);
    let gamma = (up - 2.0 * base + down) / (h_s * h_s);

    // Volatility cannot go negative, so for a very low base vol the down
    // leg is clamped at zero and the difference is divided by the spread
    // actually used rather than the nominal 2h.
    let vol_up = model.volatility + bumps.volatility;
    let vol_down = (model.volatility - bumps.volatility).max(0.0);
    let vega =
        (price_with(&with_vol(vol_up))? - price_with(&with_vol(vol_down))?) / (vol_up - vol_down);

    let h_r = bumps.rate;
    let rho = (price_with(&with_rate(model.risk_free_rate + h_r))?
        - price_with(&with_rate(model.risk_free_rate - h_r))?)
        / (2.0 * h_r);

    let h_t = bumps.maturity;
    let theta = (base - price_with(&with_maturity(model.maturity - h_t))?) / h_t;

    Ok(Greeks {
        delta,
        gamma,
        vega,
        theta,
        rho,
    })
}
