//! Conditional Monte Carlo verifies the discretised Heston scheme.

use fuzzy_enigma::analytic::bs_price;
use fuzzy_enigma::{EuropeanOption, HestonModel, McEngine, OptionType};

fn option(option_type: OptionType, strike: f64) -> EuropeanOption {
    EuropeanOption {
        option_type,
        strike,
    }
}

#[test]
fn deterministic_variance_integrates_the_actual_discrete_scheme() {
    // These cases distinguish the Euler variance integral from the continuous
    // mean-reversion integral, including a negative raw-variance state.
    for (v0, theta, kappa, steps) in [(0.04, 0.09, 4.0, 2), (0.04, 0.016, 10.0, 4)] {
        let maturity = 1.0;
        let dt = maturity / steps as f64;
        let mut raw_variance: f64 = v0;
        let mut integrated_variance = 0.0;
        for _ in 0..steps {
            let variance = raw_variance.max(0.0);
            integrated_variance += variance * dt;
            raw_variance += kappa * (theta - variance) * dt;
        }
        for rho in [-1.0, -0.7, 0.0, 1.0] {
            let model = HestonModel::new(
                100.0, 0.03, 0.01, v0, kappa, theta, 0.0, rho, maturity, steps,
            );
            for kind in [OptionType::Call, OptionType::Put] {
                let payoff = option(kind, 105.0);
                let result = McEngine::new(7).price_heston_conditional(&model, &payoff);
                let expected = bs_price(
                    kind,
                    100.0,
                    105.0,
                    0.03,
                    0.01,
                    (integrated_variance / maturity).sqrt(),
                    maturity,
                );
                assert!((result.price - expected).abs() < 2e-12);
                assert_eq!(result.std_error, 0.0);
                assert_eq!(result.samples, 4);
            }
        }
    }
}

#[test]
fn zero_variance_has_discounted_deterministic_cashflow() {
    let model = HestonModel::new(100.0, 0.03, 0.01, 0.0, 1.0, 0.0, 0.4, -0.7, 1.0, 16);
    let result =
        McEngine::new(31).price_heston_conditional(&model, &option(OptionType::Call, 99.0));
    let expected = 100.0 * (-0.01_f64).exp() - 99.0 * (-0.03_f64).exp();
    assert!((result.price - expected).abs() < 2e-12);
    assert_eq!(result.std_error, 0.0);
}

#[test]
fn perfect_correlation_reduces_to_the_original_payoff() {
    for rho in [-1.0, 1.0] {
        let model = HestonModel::new(100.0, 0.03, 0.01, 0.04, 1.5, 0.04, 0.4, rho, 1.0, 16);
        let engine = McEngine::new(4000).with_seed(824);
        for kind in [OptionType::Call, OptionType::Put] {
            let payoff = option(kind, 105.0);
            let conditional = engine
                .try_price_heston_conditional_with_shifts(&model, &payoff, &[0.0])
                .unwrap();
            let original = engine.price(&model, &payoff);
            assert!((conditional.price - original.price).abs() < 2e-12);
            assert!((conditional.std_error - original.std_error).abs() < 2e-12);
            assert_eq!(conditional.samples, original.samples);
        }
    }
}

#[test]
fn conditional_prices_preserve_joint_law_and_reduce_sampling_error() {
    let model = HestonModel::new(100.0, 0.03, 0.01, 0.04, 1.5, 0.04, 0.4, -0.7, 1.0, 64);
    let engine = McEngine::new(40_000).with_seed(2718);
    for kind in [OptionType::Call, OptionType::Put] {
        let payoff = option(kind, 105.0);
        let conditional = engine
            .try_price_heston_conditional_with_shifts(&model, &payoff, &[0.0])
            .unwrap();
        let original = engine.price(&model, &payoff);
        let comparison_se = conditional.std_error.hypot(original.std_error);
        assert!((conditional.price - original.price).abs() < 4.0 * comparison_se);
        assert!(conditional.std_error < original.std_error);
    }
}

#[test]
fn defensive_mixture_likelihood_recovers_known_one_step_distribution() {
    // With one step, stock variance is exactly v0 and the terminal law is
    // GBM, regardless of the subsequently updated variance. This supplies
    // an analytic oracle for the importance-sampling likelihood itself.
    let model = HestonModel::new(100.0, 0.03, 0.01, 0.04, 1.5, 0.09, 0.8, -0.7, 1.0, 1);
    let engine = McEngine::new(40_001).with_seed(137);
    for (kind, strike) in [(OptionType::Call, 120.0), (OptionType::Put, 80.0)] {
        let result = engine.price_heston_conditional(&model, &option(kind, strike));
        let expected = bs_price(kind, 100.0, strike, 0.03, 0.01, 0.2, 1.0);
        assert!((result.price - expected).abs() < 4.0 * result.std_error);
        assert!(result.std_error > 0.0);
        assert_eq!(result.samples, 20_001);
    }
}

#[test]
fn conditional_serial_parallel_runs_are_identical_and_pairs_are_samples() {
    let model = HestonModel::new(100.0, 0.03, 0.01, 0.04, 1.5, 0.04, 0.4, -0.7, 1.0, 16);
    let payoff = option(OptionType::Put, 95.0);
    for antithetic in [false, true] {
        let engine = McEngine::new(101).with_seed(42).with_antithetic(antithetic);
        let serial = engine
            .clone()
            .with_parallel(false)
            .price_heston_conditional(&model, &payoff);
        let parallel = engine
            .with_parallel(true)
            .price_heston_conditional(&model, &payoff);
        assert_eq!(serial, parallel);
        assert_eq!(serial.samples, if antithetic { 51 } else { 101 });
        assert!(serial.std_error.is_finite() && serial.std_error > 0.0);
    }
}

#[test]
fn conditional_pricing_validates_model_payoff_and_path_budget() {
    let mut model = HestonModel::new(100.0, 0.03, 0.01, 0.04, 1.5, 0.04, 0.4, -0.7, 1.0, 16);
    let payoff = option(OptionType::Call, 100.0);
    assert!(McEngine::new(0)
        .try_price_heston_conditional(&model, &payoff)
        .is_err());
    assert!(McEngine::new(2)
        .try_price_heston_conditional(&model, &option(OptionType::Call, -1.0))
        .is_err());
    model.rho = 1.01;
    assert!(McEngine::new(2)
        .try_price_heston_conditional(&model, &payoff)
        .is_err());
    model.rho = -0.7;
    for shifts in [
        vec![],
        vec![1.0],
        vec![0.0, 0.0],
        vec![0.0, f64::NAN],
        vec![0.0, f64::INFINITY],
    ] {
        assert!(McEngine::new(2)
            .try_price_heston_conditional_with_shifts(&model, &payoff, &shifts)
            .is_err());
    }
}

#[test]
fn conditional_mc_resolves_the_real_short_dated_option_tails() {
    // Frozen SPXW 2026-10-06 terms and fitted Heston parameters. Independent
    // QuantLib 1.40 adaptive Fourier integration, relative tolerance 1e-12,
    // returned 8.061898825586576e-6; 192-point Laguerre agrees within 8e-7
    // relative. The discretisation allowance is relative to this tiny price,
    // so a zero estimate cannot pass through a fixed absolute tolerance.
    let model = HestonModel::new(
        7818.93,
        0.03936302545941496,
        -0.03227861194511458,
        0.009501458131691976,
        8.598595842385889,
        0.03731037891957906,
        1.6420748428920762,
        -0.6185742577642136,
        7.0 / 365.0,
        512,
    );
    for (kind, strike, reference) in [
        (OptionType::Call, 8600.0, 8.061898825586576e-6),
        (OptionType::Put, 6300.0, 2.1262144975118262e-5),
    ] {
        let result = McEngine::new(120_000)
            .with_seed(2718)
            .price_heston_conditional(&model, &option(kind, strike));
        println!(
            "conditional tiny {kind:?}={}, SE={}, reference={reference}",
            result.price, result.std_error
        );
        assert!(result.price > 0.0);
        assert!(result.std_error > 0.0 && result.std_error < 0.20 * reference);
        assert!((result.price - reference).abs() < 4.0 * result.std_error + 0.05 * reference);
    }
}
