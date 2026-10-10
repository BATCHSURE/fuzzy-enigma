use chrono::Duration;
use fuzzy_enigma::dated::*;
use fuzzy_enigma::McEngine;

fn date() -> Date {
    Date::from_ymd_opt(2024, 2, 28).unwrap()
}
fn day(n: i64) -> Date {
    date() + Duration::days(n)
}
fn context(fixing_days: &[i64], max_step_days: f64) -> ValuationContext {
    let discount = DiscountCurve::try_new(
        date(),
        "USD".into(),
        vec![
            CurveNode {
                date: date(),
                value: 1.0,
            },
            CurveNode {
                date: day(30),
                value: 0.997,
            },
            CurveNode {
                date: day(90),
                value: 0.982,
            },
            CurveNode {
                date: day(120),
                value: 0.975,
            },
        ],
        "independent test inputs".into(),
    )
    .unwrap();
    let forward = ForwardCurve::try_new(
        date(),
        "USD".into(),
        "TEST".into(),
        vec![
            CurveNode {
                date: date(),
                value: 100.0,
            },
            CurveNode {
                date: day(30),
                value: 103.0,
            },
            CurveNode {
                date: day(90),
                value: 101.0,
            },
        ],
        "independent test inputs".into(),
    )
    .unwrap();
    ValuationContext::try_new(
        ValuationCutoff {
            as_of: date(),
            phase: CutoffPhase::AfterFixing,
        },
        100.0,
        "USD".into(),
        "TEST".into(),
        discount,
        Some(forward),
        "test".into(),
    )
    .unwrap()
    .with_grid(
        TimeGrid::try_new(
            date(),
            fixing_days.iter().map(|&n| day(n)).collect(),
            max_step_days,
        )
        .unwrap(),
    )
    .unwrap()
}

#[derive(Clone)]
struct Terminal {
    fixing: Date,
    payment: Date,
    strike: Option<f64>,
    duplicates: usize,
}
impl CashflowPayoff for Terminal {
    fn validate(&self, c: &ValuationContext) -> Result<(), ContextPricingError> {
        c.grid.index(self.fixing)?;
        Ok(())
    }
    fn cashflows(
        &self,
        path: &[f64],
        c: &ValuationContext,
        out: &mut Vec<Cashflow>,
    ) -> Result<(), ContextPricingError> {
        let spot = path[c.grid.index(self.fixing)?];
        let value = self.strike.map_or(spot, |k| (spot - k).max(0.0));
        for i in 0..self.duplicates {
            out.push(Cashflow {
                id: format!("terminal{i}"),
                fixing_date: Some(self.fixing),
                payment_date: self.payment,
                amount: value,
                kind: CashflowKind::Other,
                currency: "USD".into(),
            });
        }
        Ok(())
    }
    fn future_fixing_dates(&self) -> Vec<Date> {
        vec![self.fixing]
    }
    fn payment_dates(&self) -> Vec<Date> {
        vec![self.payment]
    }
}

#[test]
fn curves_support_negative_rates_and_exact_bounded_log_interpolation() {
    let d = DiscountCurve::try_new(
        date(),
        "USD".into(),
        vec![
            CurveNode {
                date: date(),
                value: 1.0,
            },
            CurveNode {
                date: day(10),
                value: 1.01,
            },
            CurveNode {
                date: day(30),
                value: 0.985,
            },
        ],
        "explicit".into(),
    )
    .unwrap();
    assert_eq!(d.value(0.0).unwrap(), 1.0);
    assert!((d.value_on(day(20)).unwrap() - (1.01_f64 * 0.985).sqrt()).abs() < 1e-14);
    assert!(d.value(-1e-10).is_err());
    assert!(d.value_on(day(31)).is_err());
    assert!(DiscountCurve::try_new(
        date(),
        "USD".into(),
        vec![CurveNode {
            date: date(),
            value: 0.999
        }],
        "explicit".into()
    )
    .is_err());
    assert!(DiscountCurve::try_new(
        date(),
        "USD".into(),
        vec![CurveNode {
            date: day(1),
            value: 1.0
        }],
        "explicit".into()
    )
    .is_err());
    assert!(DiscountCurve::try_new(
        date(),
        "USD".into(),
        vec![
            CurveNode {
                date: date(),
                value: 1.0
            },
            CurveNode {
                date: date(),
                value: 1.0
            }
        ],
        "explicit".into()
    )
    .is_err());
    assert!(ForwardCurve::try_new(
        date(),
        "USD".into(),
        "TEST".into(),
        vec![CurveNode {
            date: date(),
            value: f64::NAN
        }],
        "explicit".into()
    )
    .is_err());
}

#[test]
fn date_grid_leap_year_exact_anchors_and_refinement_do_not_change_events() {
    let grid =
        TimeGrid::try_new(date(), vec![day(60), day(30), date(), day(1), day(30)], 1.0).unwrap();
    assert_eq!(day(1).to_string(), "2024-02-29");
    assert_eq!(grid.steps(), 60);
    for n in [0, 1, 30, 60] {
        assert_eq!(grid.times()[grid.index(day(n)).unwrap()], n as f64 / 365.0);
    }
    let coarse = TimeGrid::try_new(date(), vec![day(1), day(30), day(60)], 7.0).unwrap();
    assert_eq!(coarse.fixing_dates(), &[day(1), day(30), day(60)]);
    assert!(coarse
        .times()
        .windows(2)
        .all(|w| (w[1] - w[0]) * 365.0 <= 7.0 + 1e-12));
    assert!(TimeGrid::try_new(date(), vec![day(-1)], 1.0).is_err());
    assert!(TimeGrid::try_new(date(), vec![day(1)], 1e-100).is_err());
    assert!(grid.index(day(2)).is_err());
}

#[test]
fn context_identity_and_curve_rolls_bind_market_date_and_forward_anchor() {
    let c = context(&[30, 90], 9.0);
    let cutoff = ValuationCutoff {
        as_of: day(7),
        phase: CutoffPhase::AfterFixing,
    };
    let r = c.frozen_roll(cutoff).unwrap();
    assert_eq!(r.spot, c.spot);
    assert_eq!(r.discount_curve.value(0.0).unwrap(), 1.0);
    assert_eq!(r.forward_curve.as_ref().unwrap().value(0.0).unwrap(), 100.0);
    assert!(
        (r.discount_curve.value_on(day(90)).unwrap()
            - c.discount_curve.value_on(day(90)).unwrap()
                / c.discount_curve.value_on(day(7)).unwrap())
        .abs()
            < 1e-14
    );
    assert!(
        (r.forward_curve.as_ref().unwrap().value_on(day(90)).unwrap()
            - 100.0 * c.forward_curve.as_ref().unwrap().value_on(day(90)).unwrap()
                / c.forward_curve.as_ref().unwrap().value_on(day(7)).unwrap())
        .abs()
            < 1e-12
    );
    assert_ne!(r.identity(), c.identity());
    assert!(c.frozen_roll(c.cutoff).is_err());
    let mut invalid = c.clone();
    invalid.spot = 101.0;
    assert!(invalid.validate().is_err());
    let mut invalid = c.clone();
    invalid.currency = "EUR".into();
    assert!(invalid.validate().is_err());
}

#[test]
fn zero_volatility_tracks_nonflat_forward_at_every_numerical_point() {
    let c = context(&[17, 30, 90], 13.0);
    let generator = Dynamics::Gbm(GbmDynamics::try_new(0.0).unwrap())
        .prepare(&c)
        .unwrap();
    let mut path = vec![0.0; c.grid.steps() + 1];
    generator
        .generate(&vec![0.37; generator.noise_dim()], &mut path)
        .unwrap();
    for (&t, &spot) in c.grid.times().iter().zip(&path) {
        assert!((spot - c.forward_curve.as_ref().unwrap().value(t).unwrap()).abs() < 2e-12);
    }
    assert!(generator.generate(&[], &mut path).is_err());
}

#[test]
fn heston_retains_negative_raw_variance_across_nonuniform_intervals() {
    let c = context(&[5, 30, 90], 100.0);
    let dynamics = HestonDynamics::try_new(0.01, 2.0, 0.03, 2.0, 0.0).unwrap();
    let g = Dynamics::Heston(dynamics).prepare(&c).unwrap();
    let normals = [0.3, -4.0, -0.6, 0.1, 0.8, -0.2];
    let mut path = vec![0.0; 4];
    g.generate(&normals, &mut path).unwrap();
    let f = c.forward_curve.as_ref().unwrap();
    let dt = 5.0 / 365.0;
    let first = 100.0
        * ((f.value_on(day(5)).unwrap() / 100.0).ln() - 0.5 * 0.01 * dt + (0.01 * dt).sqrt() * 0.3)
            .exp();
    assert!((path[1] - first).abs() < 1e-12);
    // Raw variance remains below zero; projecting the stored state would restart diffusion.
    assert!(
        (path[2] / path[1] - f.value_on(day(30)).unwrap() / f.value_on(day(5)).unwrap()).abs()
            < 1e-14
    );
    assert!(
        (path[3] / path[2] - f.value_on(day(90)).unwrap() / f.value_on(day(30)).unwrap()).abs()
            < 1e-14
    );
}

#[test]
fn constant_variance_heston_reduces_to_gbm_with_matching_stock_noise() {
    let c = context(&[7, 30, 90], 17.0);
    let gbm = Dynamics::Gbm(GbmDynamics::try_new(0.2).unwrap())
        .prepare(&c)
        .unwrap();
    let heston = Dynamics::Heston(HestonDynamics::try_new(0.04, 1.7, 0.04, 0.0, -0.7).unwrap())
        .prepare(&c)
        .unwrap();
    let z: Vec<_> = (0..c.grid.steps())
        .map(|i| (i as f64 * 0.79).sin())
        .collect();
    let hz: Vec<_> = z.iter().flat_map(|&v| [v, -0.37]).collect();
    let mut g = vec![0.0; c.grid.steps() + 1];
    let mut h = g.clone();
    gbm.generate(&z, &mut g).unwrap();
    heston.generate(&hz, &mut h).unwrap();
    for (&a, &b) in g.iter().zip(&h) {
        assert!((a - b).abs() < 2e-12);
    }
}

#[test]
fn term_structure_vanilla_matches_black_with_fixing_forward_and_payment_discount() {
    let c = context(&[30, 90], 11.0);
    let payoff = Terminal {
        fixing: day(90),
        payment: day(120),
        strike: Some(100.0),
        duplicates: 1,
    };
    let sigma = 0.22;
    let dynamics = Dynamics::Gbm(GbmDynamics::try_new(sigma).unwrap());
    let result = McEngine::new(40_000)
        .with_seed(551)
        .try_price_context(&c, Some(&dynamics), &payoff)
        .unwrap();
    let t = 90.0 / 365.0;
    let d = c.discount_curve.value_on(day(120)).unwrap();
    let f = c.forward_curve.as_ref().unwrap().value_on(day(90)).unwrap();
    let r = -d.ln() / t;
    let q = r - (f / 100.0).ln() / t;
    let expected = fuzzy_enigma::analytic::bs_call(100.0, 100.0, r, q, sigma, t);
    assert!((result.estimate.price - expected).abs() < 4.0 * result.estimate.std_error + 1e-12);
    assert_eq!(result.expected_cashflows.len(), 1);
    assert!((result.expected_cashflows[0].present_value - result.estimate.price).abs() < 1e-12);
    assert!((result.expected_cashflows[0].pv_std_error - result.estimate.std_error).abs() < 1e-12);
}

#[test]
fn cashflow_covariance_parallel_reproducibility_and_antithetic_counts() {
    let c = context(&[30, 90], 12.0);
    let dyns = Dynamics::Gbm(GbmDynamics::try_new(0.2).unwrap());
    let payoff = Terminal {
        fixing: day(90),
        payment: day(90),
        strike: None,
        duplicates: 1,
    };
    let engine = McEngine::new(101).with_seed(31);
    let one = engine
        .clone()
        .with_parallel(false)
        .try_price_context(&c, Some(&dyns), &payoff)
        .unwrap();
    let parallel = engine
        .clone()
        .with_parallel(true)
        .try_price_context(&c, Some(&dyns), &payoff)
        .unwrap();
    assert_eq!(one, parallel);
    assert_eq!(one.estimate.samples, 51);
    assert_eq!(one.paths, 102);
    let doubled = engine
        .try_price_context(
            &c,
            Some(&dyns),
            &Terminal {
                duplicates: 2,
                ..payoff
            },
        )
        .unwrap();
    assert!((doubled.estimate.price - 2.0 * one.estimate.price).abs() < 1e-12);
    assert!((doubled.estimate.std_error - 2.0 * one.estimate.std_error).abs() < 1e-12);
    assert!(
        (doubled
            .expected_cashflows
            .iter()
            .map(|cf| cf.present_value)
            .sum::<f64>()
            - doubled.estimate.price)
            .abs()
            < 1e-12
    );
}

#[test]
fn heston_nonuniform_discounted_stock_expectation_matches_supplied_forward() {
    let c = context(&[7, 30, 90], 3.0);
    let dynamics = Dynamics::Heston(HestonDynamics::try_new(0.04, 1.5, 0.05, 0.5, -0.7).unwrap());
    let payoff = Terminal {
        fixing: day(90),
        payment: day(120),
        strike: None,
        duplicates: 1,
    };
    let result = McEngine::new(60_000)
        .with_seed(193)
        .try_price_context(&c, Some(&dynamics), &payoff)
        .unwrap();
    let expected = c.discount_curve.value_on(day(120)).unwrap()
        * c.forward_curve.as_ref().unwrap().value_on(day(90)).unwrap();
    assert!((result.estimate.price - expected).abs() < 4.0 * result.estimate.std_error + 1e-12);
}

#[test]
fn unsampled_payment_coverage_and_invalid_dynamics_fail_before_sampling() {
    let c = context(&[90], 30.0);
    let dyns = Dynamics::Gbm(GbmDynamics::try_new(0.2).unwrap());
    let payoff = Terminal {
        fixing: day(90),
        payment: day(121),
        strike: Some(1e10),
        duplicates: 1,
    };
    assert!(McEngine::new(1)
        .try_price_context(&c, Some(&dyns), &payoff)
        .unwrap_err()
        .to_string()
        .contains("curve_coverage"));
    assert!(GbmDynamics::try_new(f64::NAN).is_err());
    assert!(HestonDynamics::try_new(0.04, 1.0, 0.04, 0.2, 1.01).is_err());
    let mut invalid = c.clone();
    invalid.forward_curve = None;
    let payoff = Terminal {
        payment: day(90),
        ..payoff
    };
    assert!(McEngine::new(1)
        .try_price_context(&invalid, Some(&dyns), &payoff)
        .is_err());
}

#[test]
fn prepared_and_custom_generator_entrypoints_preserve_streams_and_validate_context() {
    let c = context(&[30, 90], 13.0);
    let dyns = Dynamics::Gbm(GbmDynamics::try_new(0.2).unwrap());
    let generator = dyns.prepare(&c).unwrap();
    let payoff = Terminal {
        fixing: day(90),
        payment: day(120),
        strike: Some(100.0),
        duplicates: 1,
    };
    let engine = McEngine::new(101).with_seed(991);
    let regular = engine.try_price_context(&c, Some(&dyns), &payoff).unwrap();
    let prepared = engine
        .try_price_context_with_generator(&c, &generator, &payoff)
        .unwrap();
    assert_eq!(regular, prepared);
    let mut changed = c.clone();
    changed.source.push_str(" changed provenance");
    assert!(engine
        .try_price_context_with_generator(&changed, &generator, &payoff)
        .is_err());
    let changed_grid = c
        .clone()
        .with_grid(TimeGrid::try_new(date(), vec![day(30), day(90)], 9.0).unwrap())
        .unwrap();
    assert!(engine
        .try_price_context_with_generator(&changed_grid, &generator, &payoff)
        .is_err());

    struct ConstantGenerator(TimeGrid);
    impl ContextPathGenerator for ConstantGenerator {
        fn grid(&self) -> &TimeGrid {
            &self.0
        }
        fn noise_dim(&self) -> usize {
            0
        }
        fn generate(&self, _: &[f64], out: &mut [f64]) -> Result<(), ContextPricingError> {
            out.fill(123.0);
            out[0] = 100.0;
            Ok(())
        }
    }
    let custom = engine
        .try_price_context_with_generator(&c, &ConstantGenerator(c.grid.clone()), &payoff)
        .unwrap();
    assert!((custom.estimate.price - 23.0 * 0.975).abs() < 1e-12);
    assert!(custom.estimate.std_error < 1e-12);
}

#[test]
fn dated_pricing_rejects_unresolved_se_and_finite_payoffs_with_overflowing_moments() {
    let c = context(&[90], 30.0);
    let dyns = Dynamics::Gbm(GbmDynamics::try_new(0.2).unwrap());
    let terminal = Terminal {
        fixing: day(90),
        payment: day(90),
        strike: None,
        duplicates: 1,
    };
    for paths in [0, 1, 2] {
        assert!(McEngine::new(paths)
            .try_price_context(&c, Some(&dyns), &terminal)
            .is_err());
    }
    struct Huge(Terminal);
    impl CashflowPayoff for Huge {
        fn validate(&self, c: &ValuationContext) -> Result<(), ContextPricingError> {
            self.0.validate(c)
        }
        fn cashflows(
            &self,
            path: &[f64],
            c: &ValuationContext,
            out: &mut Vec<Cashflow>,
        ) -> Result<(), ContextPricingError> {
            self.0.cashflows(path, c, out)?;
            for flow in out {
                flow.amount *= 1e299;
            }
            Ok(())
        }
        fn future_fixing_dates(&self) -> Vec<Date> {
            self.0.future_fixing_dates()
        }
        fn payment_dates(&self) -> Vec<Date> {
            self.0.payment_dates()
        }
    }
    let error = McEngine::new(8)
        .with_seed(771)
        .try_price_context(&c, Some(&dyns), &Huge(terminal))
        .unwrap_err();
    assert!(error.to_string().contains("estimated_std_error"));
}
