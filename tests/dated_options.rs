use chrono::Duration;
use fuzzy_enigma::dated::*;
use fuzzy_enigma::payoff::{BarrierKind, OptionType};
use fuzzy_enigma::McEngine;

fn date(n: i64) -> Date {
    Date::from_ymd_opt(2026, 1, 1).unwrap() + Duration::days(n)
}
fn context(step: f64) -> ValuationContext {
    let discount = DiscountCurve::try_new(
        date(0),
        "USD".into(),
        vec![
            CurveNode {
                date: date(0),
                value: 1.0,
            },
            CurveNode {
                date: date(40),
                value: 0.98,
            },
        ],
        "test D".into(),
    )
    .unwrap();
    let forward = ForwardCurve::try_new(
        date(0),
        "USD".into(),
        "INDEX".into(),
        vec![
            CurveNode {
                date: date(0),
                value: 100.0,
            },
            CurveNode {
                date: date(30),
                value: 110.0,
            },
        ],
        "test F".into(),
    )
    .unwrap();
    ValuationContext::try_new(
        ValuationCutoff {
            as_of: date(0),
            phase: CutoffPhase::AfterFixing,
        },
        100.0,
        "USD".into(),
        "INDEX".into(),
        discount,
        Some(forward),
        "test".into(),
    )
    .unwrap()
    .with_grid(TimeGrid::try_new(date(0), vec![date(15), date(30)], step).unwrap())
    .unwrap()
}
fn barrier(kind: BarrierKind, barrier: f64, rebate: f64) -> ContextBarrierOption {
    ContextBarrierOption::try_new(
        OptionType::Call,
        kind,
        100.0,
        barrier,
        rebate,
        date(30),
        date(40),
        vec![date(0), date(30)],
        "USD".into(),
    )
    .unwrap()
}

#[test]
fn monitoring_is_contractual_not_a_numerical_refinement() {
    let ctx = context(15.0);
    let option = barrier(BarrierKind::UpAndOut, 120.0, 3.0);
    option.validate(&ctx).unwrap();
    let mut flows = Vec::new();
    option
        .cashflows(&[100.0, 150.0, 110.0], &ctx, &mut flows)
        .unwrap();
    // The unobserved interior excursion cannot knock out the option.
    assert_eq!(flows[0].amount, 10.0);
    assert_eq!(flows[0].payment_date, date(40));
    let mut fine = vec![100.0; context(1.0).grid.steps() + 1];
    fine[15] = 150.0;
    fine[30] = 110.0;
    let mut refined = Vec::new();
    option
        .cashflows(&fine, &context(1.0), &mut refined)
        .unwrap();
    assert_eq!(flows, refined);
}

#[test]
fn barrier_equalities_initial_touch_and_in_out_parity() {
    let ctx = context(15.0);
    for (out_kind, in_kind, level) in [
        (BarrierKind::UpAndOut, BarrierKind::UpAndIn, 110.0),
        (BarrierKind::DownAndOut, BarrierKind::DownAndIn, 100.0),
    ] {
        let mut out = Vec::new();
        let mut input = Vec::new();
        barrier(out_kind, level, 0.0)
            .cashflows(&[100.0, 103.0, 110.0], &ctx, &mut out)
            .unwrap();
        barrier(in_kind, level, 0.0)
            .cashflows(&[100.0, 103.0, 110.0], &ctx, &mut input)
            .unwrap();
        assert_eq!(out[0].amount, 0.0);
        assert_eq!(input[0].amount, 10.0);
        assert_eq!(input[0].amount + out[0].amount, 10.0);
    }
}

#[test]
fn payment_lag_discounted_once_without_extending_stock_simulation() {
    let ctx = context(15.0);
    let option =
        ContextEuropeanOption::try_new(OptionType::Call, 100.0, date(30), date(40), "USD".into())
            .unwrap();
    let engine = McEngine {
        paths: 100,
        seed: 42,
        antithetic: true,
        parallel: true,
    };
    let price = engine
        .try_price_context(
            &ctx,
            Some(&Dynamics::Gbm(GbmDynamics::try_new(0.0).unwrap())),
            &option,
        )
        .unwrap();
    assert!((price.estimate.price - 9.8).abs() < 1e-12);
    assert_eq!(price.expected_cashflows[0].payment_date, date(40));
    assert!((ctx.grid.maturity() - 30.0 / 365.0).abs() < 1e-14);
}

#[test]
fn invalid_dates_currency_and_past_monitoring_fail() {
    assert!(ContextEuropeanOption::try_new(
        OptionType::Call,
        100.0,
        date(30),
        date(29),
        "USD".into()
    )
    .is_err());
    assert!(ContextBarrierOption::try_new(
        OptionType::Call,
        BarrierKind::UpAndOut,
        100.0,
        120.0,
        0.0,
        date(30),
        date(40),
        vec![date(20), date(10)],
        "USD".into()
    )
    .is_err());
    assert!(ContextEuropeanOption::try_new(
        OptionType::Call,
        100.0,
        date(30),
        date(40),
        "EUR".into()
    )
    .unwrap()
    .validate(&context(15.0))
    .is_err());
    let bad = ContextBarrierOption::try_new(
        OptionType::Call,
        BarrierKind::UpAndOut,
        100.0,
        120.0,
        0.0,
        date(30),
        date(40),
        vec![date(-1), date(30)],
        "USD".into(),
    )
    .unwrap();
    assert!(bad.validate(&context(15.0)).is_err());
}
