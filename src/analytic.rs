//! Closed-form Black-Scholes prices and Greeks.
//!
//! These serve three callers: the integration tests cross-check the Monte
//! Carlo engine against them, [`crate::pricer::EuropeanControl`] uses them
//! as the known expectation of a control variate, and the Greeks tests
//! benchmark bump-and-revalue estimates against the analytic values.
//!
//! The control-variate use is why [`norm_cdf`] is a double-precision
//! rational approximation rather than a cheaper one: a control variate
//! subtracts the analytic value from the simulated one, so any error in it
//! passes straight through into the price as bias rather than averaging
//! out over paths.

use crate::payoff::OptionType;

const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// Evaluate a polynomial at `x` by Horner's rule, highest order first.
#[inline]
fn horner(coefficients: &[f64], x: f64) -> f64 {
    coefficients.iter().fold(0.0, |acc, &c| acc * x + c)
}

/// Standard normal probability density.
#[inline]
pub fn norm_pdf(x: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Standard normal cumulative distribution function.
///
/// Hart's rational approximation in the form given by Graeme West, with a
/// continued-fraction tail beyond |x| = 7.07. Absolute accuracy is around
/// 1e-15 across the range, against ~1.5e-7 for the Abramowitz & Stegun
/// series it replaces. In the far tail the continued fraction is accurate
/// to ~1e-8 *relative*, which is far below the absolute scale that matters
/// for pricing.
pub fn norm_cdf(x: f64) -> f64 {
    let abs_x = x.abs();
    if abs_x > 37.0 {
        return if x > 0.0 { 1.0 } else { 0.0 };
    }

    let e = (-0.5 * abs_x * abs_x).exp();
    let upper_tail = if abs_x < 7.071_067_811_865_47 {
        const NUM: [f64; 7] = [
            3.526_249_659_989_11e-2,
            0.700_383_064_443_688,
            6.373_962_203_531_65,
            33.912_866_078_383,
            112.079_291_497_871,
            221.213_596_169_931,
            220.206_867_912_376,
        ];
        const DEN: [f64; 8] = [
            8.838_834_764_831_84e-2,
            1.755_667_163_182_64,
            16.064_177_579_207,
            86.780_732_202_946_1,
            296.564_248_779_674,
            637.333_633_378_831,
            793.826_512_519_948,
            440.413_735_824_752,
        ];
        e * horner(&NUM, abs_x) / horner(&DEN, abs_x)
    } else {
        let mut b = abs_x + 0.65;
        for k in [4.0, 3.0, 2.0, 1.0] {
            b = abs_x + k / b;
        }
        e / (b * 2.506_628_274_631)
    };

    if x > 0.0 {
        1.0 - upper_tail
    } else {
        upper_tail
    }
}

/// `(d1, d2)` from the Black-Scholes formula, or `None` when the contract
/// is degenerate (zero volatility or zero time), which leaves the payoff
/// deterministic and the derivatives undefined.
fn d1_d2(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> Option<(f64, f64)> {
    let vol_sqrt_t = sigma * t.sqrt();
    if vol_sqrt_t <= 0.0 || s <= 0.0 || k <= 0.0 {
        return None;
    }
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / vol_sqrt_t;
    Some((d1, d1 - vol_sqrt_t))
}

/// Deterministic payoff when volatility or time to expiry is zero: the
/// forward is known, so the option is worth its discounted intrinsic.
fn degenerate_price(option_type: OptionType, s: f64, k: f64, r: f64, q: f64, t: f64) -> f64 {
    let forward = s * ((r - q) * t).exp();
    (-r * t).exp() * option_type.intrinsic(forward, k)
}

/// Black-Scholes price of a European call.
pub fn bs_call(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
    match d1_d2(s, k, r, q, sigma, t) {
        Some((d1, d2)) => s * (-q * t).exp() * norm_cdf(d1) - k * (-r * t).exp() * norm_cdf(d2),
        None => degenerate_price(OptionType::Call, s, k, r, q, t),
    }
}

/// Black-Scholes price of a European put.
pub fn bs_put(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
    match d1_d2(s, k, r, q, sigma, t) {
        Some((d1, d2)) => k * (-r * t).exp() * norm_cdf(-d2) - s * (-q * t).exp() * norm_cdf(-d1),
        None => degenerate_price(OptionType::Put, s, k, r, q, t),
    }
}

/// Black-Scholes price dispatched on [`OptionType`].
pub fn bs_price(
    option_type: OptionType,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
) -> f64 {
    match option_type {
        OptionType::Call => bs_call(s, k, r, q, sigma, t),
        OptionType::Put => bs_put(s, k, r, q, sigma, t),
    }
}

/// Black-Scholes delta, `dV/dS`.
pub fn bs_delta(
    option_type: OptionType,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
) -> f64 {
    let Some((d1, _)) = d1_d2(s, k, r, q, sigma, t) else {
        return 0.0;
    };
    match option_type {
        OptionType::Call => (-q * t).exp() * norm_cdf(d1),
        OptionType::Put => -(-q * t).exp() * norm_cdf(-d1),
    }
}

/// Black-Scholes gamma, `d2V/dS2`. Identical for calls and puts.
pub fn bs_gamma(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
    let Some((d1, _)) = d1_d2(s, k, r, q, sigma, t) else {
        return 0.0;
    };
    (-q * t).exp() * norm_pdf(d1) / (s * sigma * t.sqrt())
}

/// Black-Scholes vega, `dV/dsigma`, per unit of volatility (not per 1%).
/// Identical for calls and puts.
pub fn bs_vega(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
    let Some((d1, _)) = d1_d2(s, k, r, q, sigma, t) else {
        return 0.0;
    };
    s * (-q * t).exp() * norm_pdf(d1) * t.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tol: f64, what: &str) {
        assert!(
            (actual - expected).abs() < tol,
            "{what}: expected {expected}, got {actual}"
        );
    }

    #[test]
    fn norm_cdf_matches_known_values() {
        assert_close(norm_cdf(0.0), 0.5, 1e-15, "Phi(0)");
        assert_close(norm_cdf(1.0), 0.841_344_746_068_543, 1e-13, "Phi(1)");
        assert_close(norm_cdf(-1.0), 0.158_655_253_931_457, 1e-13, "Phi(-1)");
        assert_close(norm_cdf(1.96), 0.975_002_104_851_780, 1e-13, "Phi(1.96)");
        assert_close(norm_cdf(-3.0), 1.349_898_031_630_09e-3, 1e-15, "Phi(-3)");
        // Deep tail: the continued-fraction branch takes over past 7.07,
        // where accuracy is relative (~1e-8) rather than absolute.
        let tail = 6.220_960_574_271_78e-16;
        assert_close(norm_cdf(-8.0), tail, 1e-7 * tail, "Phi(-8)");
    }

    #[test]
    fn norm_cdf_is_symmetric() {
        for i in 0..200 {
            let x = -10.0 + 0.1 * i as f64;
            assert_close(norm_cdf(x) + norm_cdf(-x), 1.0, 1e-14, "Phi(x)+Phi(-x)");
        }
    }

    #[test]
    fn put_call_parity_holds() {
        let (s, k, r, q, sigma, t) = (100.0, 95.0, 0.05, 0.02, 0.25, 1.5);
        let lhs = bs_call(s, k, r, q, sigma, t) - bs_put(s, k, r, q, sigma, t);
        let rhs = s * (-q * t).exp() - k * (-r * t).exp();
        assert_close(lhs, rhs, 1e-12, "C - P = S e^-qT - K e^-rT");
    }

    #[test]
    fn zero_volatility_gives_discounted_intrinsic() {
        let (s, k, r, t) = (100.0_f64, 90.0_f64, 0.03_f64, 1.0_f64);
        let expected = (-r * t).exp() * (s * (r * t).exp() - k);
        assert_close(bs_call(s, k, r, 0.0, 0.0, t), expected, 1e-12, "sigma = 0");
        assert_close(bs_put(s, k, r, 0.0, 0.0, t), 0.0, 1e-12, "sigma = 0 put");
    }

    #[test]
    fn analytic_greeks_match_finite_differences() {
        let (s, k, r, q, sigma, t) = (100.0, 105.0, 0.04, 0.01, 0.22, 0.75);
        let h = 1e-4;
        let call = OptionType::Call;

        let up = bs_price(call, s + h, k, r, q, sigma, t);
        let down = bs_price(call, s - h, k, r, q, sigma, t);
        let mid = bs_price(call, s, k, r, q, sigma, t);

        assert_close(
            bs_delta(call, s, k, r, q, sigma, t),
            (up - down) / (2.0 * h),
            1e-6,
            "delta",
        );
        assert_close(
            bs_gamma(s, k, r, q, sigma, t),
            (up - 2.0 * mid + down) / (h * h),
            1e-4,
            "gamma",
        );
        assert_close(
            bs_vega(s, k, r, q, sigma, t),
            (bs_price(call, s, k, r, q, sigma + h, t) - bs_price(call, s, k, r, q, sigma - h, t))
                / (2.0 * h),
            1e-5,
            "vega",
        );
    }
}
