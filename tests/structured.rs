//! Explicit-path contract tests for the teaching snowball and Phoenix notes.

use fuzzy_enigma::{
    AutocallableNote, GbmModel, McEngine, Payoff, PhoenixNote, PricingError, SnowballNote,
};

fn snowball() -> SnowballNote {
    SnowballNote {
        notional: 100.0,
        reference_spot: 100.0,
        coupon_rate: 0.12,
        knock_in_barrier: 70.0,
        knock_out_barrier: 105.0,
        observation_indices: vec![2, 4],
    }
}

fn phoenix() -> PhoenixNote {
    PhoenixNote {
        notional: 100.0,
        reference_spot: 100.0,
        coupon_per_period: 3.0,
        coupon_barrier: 90.0,
        knock_in_barrier: 70.0,
        knock_out_barrier: 105.0,
        coupon_indices: vec![1, 2, 3, 4],
        autocall_indices: vec![2, 4],
        memory: false,
    }
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-10,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn snowball_knockout_accrues_coupon_to_actual_redemption_time() {
    let note = snowball();
    let path = [100.0, 110.0, 105.0, 50.0, 10.0];
    let expected = 106.0 * (-0.04_f64 * 0.5).exp();
    assert_close(note.evaluate(&path, 0.25, 0.04), expected);
}

#[test]
fn snowball_no_knockin_pays_principal_and_full_coupon_at_path_maturity() {
    let mut note = snowball();
    note.observation_indices = vec![1, 2];
    let path = [100.0, 95.0, 92.0, 88.0, 80.0];
    assert_close(note.evaluate(&path, 0.25, 0.04), 112.0 * (-0.04_f64).exp());
}

#[test]
fn snowball_knockin_on_non_observation_date_removes_coupon() {
    let note = snowball();
    let path = [100.0, 70.0, 80.0, 90.0, 95.0];
    assert_close(note.evaluate(&path, 0.25, 0.04), 95.0 * (-0.04_f64).exp());
}

#[test]
fn snowball_knockin_recovery_caps_principal_without_coupon() {
    let mut note = snowball();
    note.knock_out_barrier = 120.0;
    let path = [100.0, 60.0, 80.0, 100.0, 110.0];
    assert_close(note.evaluate(&path, 0.25, 0.0), 100.0);
}

#[test]
fn snowball_initial_fixing_counts_for_knockin() {
    let note = snowball();
    let path = [70.0, 80.0, 90.0, 95.0, 100.0];
    assert_close(note.evaluate(&path, 0.25, 0.0), 100.0);
}

#[test]
fn snowball_knockout_overrides_previous_knockin() {
    let note = snowball();
    let path = [100.0, 50.0, 105.0, 60.0, 40.0];
    assert_close(note.evaluate(&path, 0.25, 0.0), 106.0);
}

#[test]
fn snowball_knockout_only_on_selected_dates_including_maturity() {
    let note = snowball();
    let path = [100.0, 110.0, 100.0, 110.0, 105.0];
    assert_close(note.evaluate(&path, 0.25, 0.0), 112.0);
}

#[test]
fn phoenix_coupons_and_maturity_principal_discount_at_each_payment_time() {
    let note = phoenix();
    let path = [100.0, 95.0, 100.0, 95.0, 90.0];
    let expected = (1..=4)
        .map(|index| 3.0 * (-0.04 * 0.25 * index as f64).exp())
        .sum::<f64>()
        + 100.0 * (-0.04_f64).exp();
    assert_close(note.evaluate(&path, 0.25, 0.04), expected);
}

#[test]
fn phoenix_coupon_equality_is_eligible() {
    let note = phoenix();
    let path = [100.0, 90.0, 80.0, 80.0, 80.0];
    assert_close(note.evaluate(&path, 0.25, 0.0), 103.0);
}

#[test]
fn phoenix_memory_catches_up_all_missed_coupons_at_payment_time() {
    let mut note = phoenix();
    note.memory = true;
    let path = [100.0, 80.0, 80.0, 90.0, 80.0];
    let expected = 9.0 * (-0.04_f64 * 0.75).exp() + 100.0 * (-0.04_f64).exp();
    assert_close(note.evaluate(&path, 0.25, 0.04), expected);
}

#[test]
fn phoenix_without_memory_pays_only_current_eligible_coupon() {
    let note = phoenix();
    let path = [100.0, 80.0, 80.0, 90.0, 80.0];
    assert_close(note.evaluate(&path, 0.25, 0.0), 103.0);
}

#[test]
fn phoenix_unpaid_memory_expires_at_maturity() {
    let mut note = phoenix();
    note.memory = true;
    let path = [100.0, 95.0, 80.0, 80.0, 80.0];
    assert_close(note.evaluate(&path, 0.25, 0.0), 103.0);
}

#[test]
fn phoenix_same_day_coupon_precedes_knockout_and_stops_future_cashflows() {
    let mut note = phoenix();
    note.memory = true;
    let path = [100.0, 80.0, 105.0, 120.0, 120.0];
    assert_close(
        note.evaluate(&path, 0.25, 0.04),
        106.0 * (-0.04_f64 * 0.5).exp(),
    );
}

#[test]
fn phoenix_knockout_can_occur_between_coupon_dates() {
    let mut note = phoenix();
    note.coupon_indices = vec![1, 3, 4];
    let path = [100.0, 95.0, 105.0, 120.0, 120.0];
    let expected = 3.0 * (-0.04_f64 * 0.25).exp() + 100.0 * (-0.04_f64 * 0.5).exp();
    assert_close(note.evaluate(&path, 0.25, 0.04), expected);
}

#[test]
fn phoenix_autocall_without_coupon_eligibility_does_not_pay_memory() {
    let mut note = phoenix();
    note.memory = true;
    note.coupon_barrier = 110.0;
    let path = [100.0, 80.0, 105.0, 120.0, 120.0];
    assert_close(note.evaluate(&path, 0.25, 0.0), 100.0);
}

#[test]
fn phoenix_knockin_principal_and_maturity_coupon_are_independent() {
    let note = phoenix();
    let path = [100.0, 70.0, 80.0, 80.0, 90.0];
    assert_close(note.evaluate(&path, 0.25, 0.0), 93.0);
}

#[test]
fn phoenix_knockin_recovery_caps_principal_and_retains_eligible_coupons() {
    let mut note = phoenix();
    note.knock_out_barrier = 130.0;
    let path = [100.0, 60.0, 80.0, 95.0, 110.0];
    assert_close(note.evaluate(&path, 0.25, 0.0), 106.0);
}

#[test]
fn phoenix_initial_knockin_and_no_terminal_coupon_preserve_equity_exposure() {
    let mut note = phoenix();
    note.coupon_indices = vec![1, 2];
    let path = [70.0, 80.0, 80.0, 90.0, 90.0];
    assert_close(note.evaluate(&path, 0.25, 0.0), 90.0);
}

#[test]
fn phoenix_maturity_autocall_pays_par_after_historical_knockin() {
    let note = phoenix();
    let path = [100.0, 50.0, 80.0, 80.0, 105.0];
    assert_close(note.evaluate(&path, 0.25, 0.0), 103.0);
}

#[test]
fn fixed_reference_is_independent_of_model_valuation_spot() {
    let mut snowball = snowball();
    snowball.knock_in_barrier = 95.0;
    let mut phoenix = phoenix();
    phoenix.knock_in_barrier = 95.0;
    phoenix.coupon_per_period = 0.0;
    let engine = McEngine::new(4).with_seed(7);
    for spot in [80.0, 90.0] {
        let model = GbmModel::new(spot, 0.0, 0.0, 0.0, 1.0, 4);
        assert_close(engine.price(&model, &snowball).price, spot);
        assert_close(engine.price(&model, &phoenix).price, spot);
    }
    assert_eq!(snowball.reference_spot, 100.0);
    assert_eq!(phoenix.reference_spot, 100.0);
}

#[test]
fn new_note_schedules_reject_empty_initial_unordered_and_out_of_range_dates() {
    let invalid_schedules = [vec![], vec![0, 1], vec![2, 2], vec![3, 2], vec![1, 5]];
    for indices in invalid_schedules {
        let mut snowball = snowball();
        snowball.observation_indices = indices.clone();
        assert!(snowball.validate(4).is_err(), "schedule {indices:?}");
        let mut phoenix = phoenix();
        phoenix.coupon_indices = indices.clone();
        assert!(phoenix.validate(4).is_err(), "coupon schedule {indices:?}");
        phoenix.coupon_indices = vec![1, 2];
        phoenix.autocall_indices = indices.clone();
        assert!(
            phoenix.validate(4).is_err(),
            "autocall schedule {indices:?}"
        );
    }
    assert!(snowball().validate(4).is_ok());
    assert!(phoenix().validate(4).is_ok());
}

#[test]
fn legacy_autocallable_still_allows_initial_observation() {
    let note = AutocallableNote {
        notional: 100.0,
        coupon_per_period: 3.0,
        autocall_barrier: 100.0,
        protection_barrier: 70.0,
        observation_indices: vec![0, 4],
    };
    assert!(note.validate(4).is_ok());
    assert_close(
        note.evaluate(&[100.0, 90.0, 80.0, 70.0, 60.0], 0.25, 0.04),
        103.0,
    );
}

#[test]
fn snowball_rejects_all_invalid_numeric_parameter_domains() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 0.0] {
        for name in [
            "notional",
            "reference_spot",
            "knock_in_barrier",
            "knock_out_barrier",
        ] {
            let mut note = snowball();
            match name {
                "notional" => note.notional = invalid,
                "reference_spot" => note.reference_spot = invalid,
                "knock_in_barrier" => note.knock_in_barrier = invalid,
                "knock_out_barrier" => note.knock_out_barrier = invalid,
                _ => unreachable!(),
            }
            assert!(matches!(
                note.validate(4),
                Err(PricingError::InvalidParameter { name: got, .. }) if got == name
            ));
        }
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1] {
        let mut note = snowball();
        note.coupon_rate = invalid;
        assert!(matches!(
            note.validate(4),
            Err(PricingError::InvalidParameter {
                name: "coupon_rate",
                ..
            })
        ));
    }
    let mut note = snowball();
    note.coupon_rate = 0.0;
    assert!(note.validate(4).is_ok());
}

#[test]
fn phoenix_rejects_all_invalid_numeric_parameter_domains() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 0.0] {
        for name in [
            "notional",
            "reference_spot",
            "coupon_barrier",
            "knock_in_barrier",
            "knock_out_barrier",
        ] {
            let mut note = phoenix();
            match name {
                "notional" => note.notional = invalid,
                "reference_spot" => note.reference_spot = invalid,
                "coupon_barrier" => note.coupon_barrier = invalid,
                "knock_in_barrier" => note.knock_in_barrier = invalid,
                "knock_out_barrier" => note.knock_out_barrier = invalid,
                _ => unreachable!(),
            }
            assert!(matches!(
                note.validate(4),
                Err(PricingError::InvalidParameter { name: got, .. }) if got == name
            ));
        }
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1] {
        let mut note = phoenix();
        note.coupon_per_period = invalid;
        assert!(matches!(
            note.validate(4),
            Err(PricingError::InvalidParameter {
                name: "coupon_per_period",
                ..
            })
        ));
    }
    let mut note = phoenix();
    note.coupon_per_period = 0.0;
    assert!(note.validate(4).is_ok());
}

#[test]
fn engine_rejects_invalid_note_before_simulating() {
    let model = GbmModel::new(100.0, 0.03, 0.0, 0.2, 1.0, 4);
    let mut note = phoenix();
    note.autocall_indices = vec![5];
    assert_eq!(
        McEngine::new(4).try_price(&model, &note),
        Err(PricingError::ObservationOutOfRange { index: 5, steps: 4 })
    );
}
