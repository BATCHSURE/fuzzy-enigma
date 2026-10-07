//! Single-underlying structured notes with explicit teaching contract terms.
//!
//! Barriers are absolute price levels. `reference_spot` is the original
//! contract fixing, independent of the model's current valuation spot.
//! Knock-in monitors every simulation point, including the initial fixing;
//! coupons and early redemption observe only their configured future dates.
//! Each cashflow is discounted at its own payment time.

use crate::error::PricingError;
use crate::payoff::{validate_observation_indices, Payoff};

/// A fixed-barrier snowball note with an annualised accrued coupon.
///
/// At the first observation with `S >= knock_out_barrier`, it pays
/// `notional * (1 + coupon_rate * elapsed_years)` and terminates.
/// Without early redemption, maturity is the end of the simulated path:
/// no historical knock-in pays principal plus the full accrued coupon;
/// a historical `S <= knock_in_barrier` pays
/// `notional * min(S_T / reference_spot, 1)` without a coupon.
#[derive(Clone, Debug)]
pub struct SnowballNote {
    pub notional: f64,
    /// Original contract fixing, held fixed when the valuation spot changes.
    pub reference_spot: f64,
    /// Annualised decimal coupon rate (e.g. `0.12` for 12% per year).
    pub coupon_rate: f64,
    pub knock_in_barrier: f64,
    pub knock_out_barrier: f64,
    /// Non-empty, strictly increasing future knock-out observation indices.
    pub observation_indices: Vec<usize>,
}

impl Payoff for SnowballNote {
    fn validate(&self, steps: usize) -> Result<(), PricingError> {
        validate_positive("notional", self.notional)?;
        validate_positive("reference_spot", self.reference_spot)?;
        validate_positive("knock_in_barrier", self.knock_in_barrier)?;
        validate_positive("knock_out_barrier", self.knock_out_barrier)?;
        validate_non_negative("coupon_rate", self.coupon_rate)?;
        validate_observation_indices(&self.observation_indices, steps, false)
    }

    fn evaluate(&self, path: &[f64], dt: f64, r: f64) -> f64 {
        let mut knocked_in = false;
        let mut next_observation = 0;
        for (index, &spot) in path.iter().enumerate() {
            knocked_in |= spot <= self.knock_in_barrier;
            if self.observation_indices.get(next_observation) == Some(&index) {
                next_observation += 1;
                if spot >= self.knock_out_barrier {
                    let time = dt * index as f64;
                    return (-r * time).exp() * self.notional * (1.0 + self.coupon_rate * time);
                }
            }
        }
        let terminal = *path.last().expect("path must be non-empty");
        let maturity = dt * (path.len() - 1) as f64;
        let cashflow = if knocked_in {
            self.notional * (terminal / self.reference_spot).min(1.0)
        } else {
            self.notional * (1.0 + self.coupon_rate * maturity)
        };
        (-r * maturity).exp() * cashflow
    }
}

/// A Phoenix note with independent conditional coupon and autocall dates.
///
/// A coupon date with `S >= coupon_barrier` pays `coupon_per_period`.
/// With memory enabled it additionally pays all previously missed coupons;
/// unpaid memory expires when the contract ends. At an autocall date with
/// `S >= knock_out_barrier`, any eligible same-day coupon is paid first,
/// then principal is returned and subsequent observations are ignored.
/// Otherwise maturity principal is par if no historical knock-in occurred,
/// or `notional * min(S_T / reference_spot, 1)` after a knock-in. Coupons
/// remain independent of this principal protection.
#[derive(Clone, Debug)]
pub struct PhoenixNote {
    pub notional: f64,
    /// Original contract fixing, held fixed when the valuation spot changes.
    pub reference_spot: f64,
    /// Cash coupon amount for each scheduled period, rather than a rate.
    pub coupon_per_period: f64,
    pub coupon_barrier: f64,
    pub knock_in_barrier: f64,
    pub knock_out_barrier: f64,
    /// Non-empty, strictly increasing future coupon observation indices.
    pub coupon_indices: Vec<usize>,
    /// Non-empty, strictly increasing future autocall observation indices.
    pub autocall_indices: Vec<usize>,
    pub memory: bool,
}

impl Payoff for PhoenixNote {
    fn validate(&self, steps: usize) -> Result<(), PricingError> {
        validate_positive("notional", self.notional)?;
        validate_positive("reference_spot", self.reference_spot)?;
        validate_positive("coupon_barrier", self.coupon_barrier)?;
        validate_positive("knock_in_barrier", self.knock_in_barrier)?;
        validate_positive("knock_out_barrier", self.knock_out_barrier)?;
        validate_non_negative("coupon_per_period", self.coupon_per_period)?;
        validate_observation_indices(&self.coupon_indices, steps, false)?;
        validate_observation_indices(&self.autocall_indices, steps, false)
    }

    fn evaluate(&self, path: &[f64], dt: f64, r: f64) -> f64 {
        let mut present_value = 0.0;
        let mut knocked_in = false;
        let mut missed_coupons = 0;
        let mut next_coupon = 0;
        let mut next_autocall = 0;
        for (index, &spot) in path.iter().enumerate() {
            knocked_in |= spot <= self.knock_in_barrier;
            let coupon_date = self.coupon_indices.get(next_coupon) == Some(&index);
            let autocall_date = self.autocall_indices.get(next_autocall) == Some(&index);
            if !coupon_date && !autocall_date {
                continue;
            }
            let discount = (-r * dt * index as f64).exp();
            if coupon_date {
                next_coupon += 1;
                if spot >= self.coupon_barrier {
                    present_value +=
                        discount * self.coupon_per_period * (missed_coupons + 1) as f64;
                    missed_coupons = 0;
                } else if self.memory {
                    missed_coupons += 1;
                }
            }
            if autocall_date {
                next_autocall += 1;
                if spot >= self.knock_out_barrier {
                    return present_value + discount * self.notional;
                }
            }
        }
        let terminal = *path.last().expect("path must be non-empty");
        let principal = if knocked_in {
            self.notional * (terminal / self.reference_spot).min(1.0)
        } else {
            self.notional
        };
        let maturity = dt * (path.len() - 1) as f64;
        present_value + (-r * maturity).exp() * principal
    }
}

fn validate_positive(name: &'static str, value: f64) -> Result<(), PricingError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(PricingError::invalid(name, "must be positive and finite"));
    }
    Ok(())
}

fn validate_non_negative(name: &'static str, value: f64) -> Result<(), PricingError> {
    if !value.is_finite() || value < 0.0 {
        return Err(PricingError::invalid(
            name,
            "must be non-negative and finite",
        ));
    }
    Ok(())
}
