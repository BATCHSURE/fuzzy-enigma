//! Contract-date, analytic-reference, and variance-reduction regressions.

use fuzzy_enigma::analytic::{bs_price, norm_cdf};
use fuzzy_enigma::greeks::{
    try_bump_and_revalue, try_bump_and_revalue_with, try_continuous_barrier_greeks,
};
use fuzzy_enigma::numerical::{geometric_asian_price, GeometricAsianControl, ScheduledAsianOption};
use fuzzy_enigma::{
    BarrierKind, BarrierOption, BumpSizes, ControlVariate, EuropeanOption, GbmModel, McEngine,
    OptionType, PathGenerator, Payoff, PricingError,
};
use std::cell::Cell;

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, actual {actual}, tolerance {tolerance}"
    );
}

fn engine(paths: usize) -> McEngine {
    McEngine::new(paths).with_seed(1234).with_parallel(true)
}

#[test]
fn scheduled_asian_uses_only_fixings_and_settles_at_maturity() {
    let payoff = ScheduledAsianOption {
        option_type: OptionType::Call,
        strike: 95.0,
        observation_indices: vec![1, 3],
    };
    close(
        payoff.evaluate(&[1_000.0, 90.0, 300.0, 110.0, 500.0], 0.25, 0.05),
        5.0 * (-0.05_f64).exp(),
        1e-12,
    );
}

#[test]
fn scheduled_asian_checks_strike_and_observation_schedule() {
    let payoff = |strike, indices| ScheduledAsianOption {
        option_type: OptionType::Put,
        strike,
        observation_indices: indices,
    };
    for strike in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(payoff(strike, vec![1]).validate(4).is_err());
    }
    assert_eq!(
        payoff(100.0, vec![]).validate(4),
        Err(PricingError::EmptyObservationSchedule)
    );
    for indices in [vec![0, 2], vec![2, 2], vec![3, 1], vec![1, 5]] {
        assert!(payoff(100.0, indices).validate(4).is_err());
    }
    assert!(payoff(100.0, vec![1, 4]).validate(4).is_ok());
}

#[test]
fn refining_grid_preserves_scheduled_fixing_dates() {
    let coarse = GbmModel::new(100.0, 0.04, 0.01, 0.25, 1.0, 4);
    let fine = GbmModel {
        steps: 12,
        ..coarse.clone()
    };
    let mut coarse_path = vec![0.0; 5];
    let mut fine_path = vec![0.0; 13];
    coarse.generate(&[0.0; 4], &mut coarse_path);
    fine.generate(&[0.0; 12], &mut fine_path);
    let coarse_payoff = ScheduledAsianOption {
        option_type: OptionType::Put,
        strike: 105.0,
        observation_indices: vec![1, 3, 4],
    };
    let fine_payoff = ScheduledAsianOption {
        observation_indices: vec![3, 9, 12],
        ..coarse_payoff.clone()
    };
    close(
        coarse_payoff.evaluate(&coarse_path, 0.25, coarse.risk_free_rate),
        fine_payoff.evaluate(&fine_path, 1.0 / 12.0, fine.risk_free_rate),
        1e-11,
    );
    close(
        geometric_asian_price(&coarse, OptionType::Put, 105.0, &[1, 3, 4]).unwrap(),
        geometric_asian_price(&fine, OptionType::Put, 105.0, &[3, 9, 12]).unwrap(),
        1e-12,
    );
}

#[test]
fn single_terminal_geometric_fixing_reduces_to_black_scholes() {
    for option_type in [OptionType::Call, OptionType::Put] {
        for volatility in [0.0, 0.3] {
            let m = GbmModel::new(100.0, 0.03, 0.02, volatility, 1.5, 24);
            close(
                geometric_asian_price(&m, option_type, 105.0, &[24]).unwrap(),
                bs_price(
                    option_type,
                    m.spot,
                    105.0,
                    m.risk_free_rate,
                    m.dividend_yield,
                    volatility,
                    m.maturity,
                ),
                1e-12,
            );
        }
    }
}

#[test]
fn zero_vol_geometric_average_uses_fixing_dates_and_final_discount() {
    let m = GbmModel::new(100.0, 0.08, 0.02, 0.0, 2.0, 8);
    let average_time = (0.25 + 0.75 + 1.25) / 3.0;
    let expected = (-m.risk_free_rate * m.maturity).exp()
        * (m.spot * ((m.risk_free_rate - m.dividend_yield) * average_time).exp() - 90.0);
    close(
        geometric_asian_price(&m, OptionType::Call, 90.0, &[1, 3, 5]).unwrap(),
        expected,
        1e-12,
    );
}

struct GeometricPayoff<'a>(&'a GeometricAsianControl);

impl Payoff for GeometricPayoff<'_> {
    fn evaluate(&self, path: &[f64], dt: f64, r: f64) -> f64 {
        self.0.evaluate(path, dt, r)
    }
    fn validate(&self, steps: usize) -> Result<(), PricingError> {
        self.0.validate(steps)
    }
}

#[test]
fn geometric_analytic_price_matches_simulation_on_irregular_fixings() {
    let m = GbmModel::new(100.0, 0.04, 0.01, 0.35, 1.0, 24);
    for option_type in [OptionType::Call, OptionType::Put] {
        let control = GeometricAsianControl::new(&m, option_type, 100.0, vec![1, 5, 11, 18, 22]);
        let result = engine(24_000).price(&m, &GeometricPayoff(&control));
        close(result.price, control.expectation(), 5.0 * result.std_error);
    }
}

#[test]
fn geometric_control_reduces_arithmetic_asian_standard_error() {
    let m = GbmModel::new(100.0, 0.04, 0.01, 0.35, 1.0, 24);
    let payoff = ScheduledAsianOption {
        option_type: OptionType::Call,
        strike: 100.0,
        observation_indices: (1..=24).collect(),
    };
    let control = GeometricAsianControl::new(
        &m,
        payoff.option_type,
        payoff.strike,
        payoff.observation_indices.clone(),
    );
    let e = engine(16_000);
    let plain = e.price(&m, &payoff);
    let controlled = e.price_with_control(&m, &payoff, &control);
    assert!(controlled.std_error < 0.15 * plain.std_error);
    close(controlled.price, plain.price, 5.0 * plain.std_error);
}

#[test]
fn geometric_control_validates_grid_and_avoids_product_overflow() {
    let m = GbmModel::new(1e200, 0.0, 0.0, 0.2, 1.0, 2);
    let control = GeometricAsianControl::new(&m, OptionType::Call, 1.0, vec![1, 2]);
    let value = control.evaluate(&[1e200; 3], 0.5, 0.0);
    assert!(value.is_finite());
    close(value / 1e200, 1.0, 1e-12);
    let changed_grid = GbmModel { steps: 4, ..m };
    let payoff = ScheduledAsianOption {
        option_type: OptionType::Call,
        strike: 1.0,
        observation_indices: vec![1, 2],
    };
    assert!(engine(10)
        .try_price_with_control(&changed_grid, &payoff, &control)
        .is_err());
}

fn barrier(option_type: OptionType, kind: BarrierKind, level: f64, rebate: f64) -> BarrierOption {
    BarrierOption {
        option_type,
        kind,
        strike: 100.0,
        barrier: level,
        rebate,
    }
}

#[test]
fn continuous_barrier_in_out_parity_includes_maturity_rebates() {
    let m = GbmModel::new(100.0, 0.04, 0.01, 0.3, 1.0, 12);
    let e = engine(4_000);
    for option_type in [OptionType::Call, OptionType::Put] {
        let vanilla = e
            .price(
                &m,
                &EuropeanOption {
                    option_type,
                    strike: 100.0,
                },
            )
            .price;
        for (inside, outside, level) in [
            (BarrierKind::UpAndIn, BarrierKind::UpAndOut, 120.0),
            (BarrierKind::DownAndIn, BarrierKind::DownAndOut, 80.0),
        ] {
            for rebate in [0.0, 2.0] {
                let p_in =
                    e.price_continuous_barrier(&m, &barrier(option_type, inside, level, rebate));
                let p_out =
                    e.price_continuous_barrier(&m, &barrier(option_type, outside, level, rebate));
                close(
                    p_in.price + p_out.price,
                    vanilla + rebate * (-m.risk_free_rate * m.maturity).exp(),
                    1e-11,
                );
            }
        }
    }
}

#[test]
fn continuous_barrier_initial_equality_counts_as_a_hit() {
    let m = GbmModel::new(100.0, 0.03, 0.0, 0.2, 1.0, 4);
    let e = engine(1_000);
    let vanilla = e
        .price(
            &m,
            &EuropeanOption {
                option_type: OptionType::Call,
                strike: 100.0,
            },
        )
        .price;
    for kind in [BarrierKind::UpAndOut, BarrierKind::DownAndOut] {
        close(
            e.price_continuous_barrier(&m, &barrier(OptionType::Call, kind, 100.0, 3.0))
                .price,
            3.0 * (-0.03_f64).exp(),
            1e-12,
        );
    }
    for kind in [BarrierKind::UpAndIn, BarrierKind::DownAndIn] {
        close(
            e.price_continuous_barrier(&m, &barrier(OptionType::Call, kind, 100.0, 3.0))
                .price,
            vanilla,
            1e-12,
        );
    }
}

#[test]
fn zero_vol_continuous_barriers_follow_deterministic_path() {
    for rate in [-0.2, 0.2] {
        let m = GbmModel::new(100.0, rate, 0.0, 0.0, 1.0, 1);
        let e = engine(4);
        for kind in [
            BarrierKind::UpAndOut,
            BarrierKind::UpAndIn,
            BarrierKind::DownAndOut,
            BarrierKind::DownAndIn,
        ] {
            for level in [80.0, 90.0, 110.0, 130.0] {
                let payoff = barrier(OptionType::Put, kind, level, 2.0);
                close(
                    e.price_continuous_barrier(&m, &payoff).price,
                    e.price(&m, &payoff).price,
                    1e-12,
                );
            }
        }
    }
}

// Independent killed-lognormal transition density (reflection principle).
// For K >= H, every positive call payoff lies in the safe half-line.
fn down_out_call_reference(m: &GbmModel, strike: f64, level: f64) -> f64 {
    let drift = m.risk_free_rate - m.dividend_yield - 0.5 * m.volatility.powi(2);
    let image_weight = (level / m.spot).powf(2.0 * drift / m.volatility.powi(2));
    let price = |spot| {
        bs_price(
            OptionType::Call,
            spot,
            strike,
            m.risk_free_rate,
            m.dividend_yield,
            m.volatility,
            m.maturity,
        )
    };
    price(m.spot) - image_weight * price(level.powi(2) / m.spot)
}

// Truncate a European call payoff at the upper barrier, then subtract
// the reflected transition density. This differs from bridge weighting.
fn up_out_call_reference(m: &GbmModel, strike: f64, level: f64) -> f64 {
    let truncated = |spot: f64| {
        let vanilla = bs_price(
            OptionType::Call,
            spot,
            strike,
            m.risk_free_rate,
            m.dividend_yield,
            m.volatility,
            m.maturity,
        );
        let beyond = bs_price(
            OptionType::Call,
            spot,
            level,
            m.risk_free_rate,
            m.dividend_yield,
            m.volatility,
            m.maturity,
        );
        let d2_level = ((spot / level).ln()
            + (m.risk_free_rate - m.dividend_yield - 0.5 * m.volatility.powi(2)) * m.maturity)
            / (m.volatility * m.maturity.sqrt());
        vanilla
            - beyond
            - (level - strike) * (-m.risk_free_rate * m.maturity).exp() * norm_cdf(d2_level)
    };
    let drift = m.risk_free_rate - m.dividend_yield - 0.5 * m.volatility.powi(2);
    truncated(m.spot)
        - (level / m.spot).powf(2.0 * drift / m.volatility.powi(2))
            * truncated(level.powi(2) / m.spot)
}

#[test]
fn continuous_barriers_match_independent_reflection_prices_on_coarse_grids() {
    for steps in [1, 16] {
        let m = GbmModel::new(100.0, 0.04, 0.01, 0.25, 1.0, steps);
        let e = engine(32_000);
        for (kind, level, reference) in [
            (
                BarrierKind::DownAndOut,
                85.0,
                down_out_call_reference(&m, 100.0, 85.0),
            ),
            (
                BarrierKind::UpAndOut,
                125.0,
                up_out_call_reference(&m, 100.0, 125.0),
            ),
        ] {
            let result =
                e.price_continuous_barrier(&m, &barrier(OptionType::Call, kind, level, 0.0));
            close(result.price, reference, 5.0 * result.std_error + 1e-4);
        }
    }
}

#[test]
fn continuous_barrier_validates_parameters_and_public_model_fields() {
    let m = GbmModel::new(100.0, 0.03, 0.0, 0.2, 1.0, 4);
    let e = engine(10);
    for level in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(e
            .try_price_continuous_barrier(
                &m,
                &barrier(OptionType::Call, BarrierKind::UpAndOut, level, 0.0)
            )
            .is_err());
    }
    for rebate in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(e
            .try_price_continuous_barrier(
                &m,
                &barrier(OptionType::Call, BarrierKind::UpAndOut, 120.0, rebate)
            )
            .is_err());
    }
    assert!(e
        .try_price_continuous_barrier(
            &GbmModel { steps: 0, ..m },
            &barrier(OptionType::Call, BarrierKind::UpAndOut, 120.0, 0.0)
        )
        .is_err());
}

#[test]
fn continuous_barrier_vega_rebuilds_bridge_volatility() {
    let m = GbmModel::new(100.0, 0.04, 0.01, 0.25, 1.0, 1);
    let payoff = barrier(OptionType::Call, BarrierKind::DownAndOut, 85.0, 0.0);
    let bumps = BumpSizes {
        volatility: 0.001,
        ..BumpSizes::for_model(&m)
    };
    let result = try_continuous_barrier_greeks(&engine(48_000), &m, &payoff, bumps).unwrap();
    let up = GbmModel {
        volatility: m.volatility + bumps.volatility,
        ..m.clone()
    };
    let down = GbmModel {
        volatility: m.volatility - bumps.volatility,
        ..m.clone()
    };
    let reference = (down_out_call_reference(&up, payoff.strike, payoff.barrier)
        - down_out_call_reference(&down, payoff.strike, payoff.barrier))
        / (2.0 * bumps.volatility);
    close(result.vega, reference, 0.75);
}

#[test]
fn controlled_greeks_rebuild_analytic_expectation_on_every_leg() {
    let m = GbmModel::new(100.0, 0.04, 0.01, 0.25, 1.0, 4);
    let payoff = ScheduledAsianOption {
        option_type: OptionType::Call,
        strike: 100.0,
        observation_indices: vec![4],
    };
    let bumps = BumpSizes::for_model(&m);
    let e = engine(256);
    let controlled = try_bump_and_revalue_with(&m, bumps, |leg| {
        let control = GeometricAsianControl::try_new(
            leg,
            payoff.option_type,
            payoff.strike,
            payoff.observation_indices.clone(),
        )?;
        Ok(e.try_price_with_control(leg, &payoff, &control)?.price)
    })
    .unwrap();
    let analytic = try_bump_and_revalue_with(&m, bumps, |leg| {
        Ok(bs_price(
            payoff.option_type,
            leg.spot,
            payoff.strike,
            leg.risk_free_rate,
            leg.dividend_yield,
            leg.volatility,
            leg.maturity,
        ))
    })
    .unwrap();
    for (actual, expected) in [
        (controlled.delta, analytic.delta),
        (controlled.gamma, analytic.gamma),
        (controlled.vega, analytic.vega),
        (controlled.rho, analytic.rho),
        (controlled.theta, analytic.theta),
    ] {
        close(actual, expected, 1e-9);
    }
}

#[test]
fn greek_invalid_bumps_are_rejected_before_repricing() {
    let m = GbmModel::new(100.0, 0.04, 0.01, 0.25, 1.0, 4);
    let normal = BumpSizes::for_model(&m);
    for bumps in [
        BumpSizes {
            spot: 0.0,
            ..normal
        },
        BumpSizes {
            spot: 100.0,
            ..normal
        },
        BumpSizes {
            spot: f64::NAN,
            ..normal
        },
        BumpSizes {
            spot: f64::MIN_POSITIVE,
            ..normal
        },
        BumpSizes {
            volatility: -0.01,
            ..normal
        },
        BumpSizes {
            volatility: f64::INFINITY,
            ..normal
        },
        BumpSizes {
            rate: 0.0,
            ..normal
        },
        BumpSizes {
            maturity: 1.0,
            ..normal
        },
    ] {
        let calls = Cell::new(0);
        let result = try_bump_and_revalue_with(&m, bumps, |_| {
            calls.set(calls.get() + 1);
            Ok(1.0)
        });
        assert!(result.is_err());
        assert_eq!(calls.get(), 0);
        assert!(try_bump_and_revalue(
            &engine(10),
            &m,
            &EuropeanOption {
                option_type: OptionType::Call,
                strike: 100.0
            },
            bumps
        )
        .is_err());
    }
}

#[test]
fn greek_zero_vol_clamp_and_scaled_maturity_remain_finite() {
    let m = GbmModel::new(100.0, 0.04, 0.01, 0.0, 1.0, 4);
    let bumps = BumpSizes::for_model(&m);
    let values =
        try_bump_and_revalue_with(&m, bumps, |leg| Ok(leg.volatility.powi(2) + leg.maturity))
            .unwrap();
    close(values.vega, bumps.volatility, 1e-12);
    close(values.theta, 1.0, 1e-12);
    assert!(try_bump_and_revalue_with(&m, bumps, |_| Ok(f64::NAN)).is_err());
}
