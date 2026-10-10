use chrono::Duration;
use fuzzy_enigma::dated::*;
use fuzzy_enigma::{HestonModel, McEngine, PathGenerator};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

fn origin() -> Date {
    Date::from_ymd_opt(2026, 1, 1).unwrap()
}
fn day(n: i64) -> Date {
    origin() + Duration::days(n)
}
fn flat_context(fixing_days: &[i64], max_step_days: f64) -> ValuationContext {
    let dates = [day(0), day(30), day(90), day(120)];
    let discount = DiscountCurve::try_new(
        origin(),
        "USD".into(),
        dates
            .iter()
            .map(|&date| CurveNode {
                date,
                value: (-0.03_f64 * year_fraction(origin(), date)).exp(),
            })
            .collect(),
        "flat test".into(),
    )
    .unwrap();
    let forward = ForwardCurve::try_new(
        origin(),
        "USD".into(),
        "TEST".into(),
        dates
            .iter()
            .map(|&date| CurveNode {
                date,
                value: 100.0 * (0.02_f64 * year_fraction(origin(), date)).exp(),
            })
            .collect(),
        "flat test".into(),
    )
    .unwrap();
    ValuationContext::try_new(
        ValuationCutoff {
            as_of: origin(),
            phase: CutoffPhase::AfterFixing,
        },
        100.0,
        "USD".into(),
        "TEST".into(),
        discount,
        Some(forward),
        "calendar test".into(),
    )
    .unwrap()
    .with_grid(
        TimeGrid::try_new(
            origin(),
            fixing_days.iter().map(|&n| day(n)).collect(),
            max_step_days,
        )
        .unwrap(),
    )
    .unwrap()
}
#[derive(Clone)]
struct Terminal {
    date: Date,
    strike: f64,
}
impl CashflowPayoff for Terminal {
    fn validate(&self, c: &ValuationContext) -> Result<(), ContextPricingError> {
        c.grid.index(self.date)?;
        Ok(())
    }
    fn cashflows(
        &self,
        path: &[f64],
        c: &ValuationContext,
        out: &mut Vec<Cashflow>,
    ) -> Result<(), ContextPricingError> {
        out.push(Cashflow {
            id: "terminal".into(),
            fixing_date: Some(self.date),
            payment_date: self.date,
            amount: (path[c.grid.index(self.date)?] - self.strike).max(0.0),
            kind: CashflowKind::Other,
            currency: "USD".into(),
        });
        Ok(())
    }
    fn future_fixing_dates(&self) -> Vec<Date> {
        vec![self.date]
    }
    fn payment_dates(&self) -> Vec<Date> {
        vec![self.date]
    }
}
struct Known(Vec<Cashflow>);
impl CashflowPayoff for Known {
    fn validate(&self, _: &ValuationContext) -> Result<(), ContextPricingError> {
        Ok(())
    }
    fn cashflows(
        &self,
        _: &[f64],
        _: &ValuationContext,
        _: &mut Vec<Cashflow>,
    ) -> Result<(), ContextPricingError> {
        Ok(())
    }
    fn known_cashflows(&self) -> &[Cashflow] {
        &self.0
    }
    fn requires_simulation(&self) -> bool {
        false
    }
}
fn fixed(payment: Date, amount: f64) -> Cashflow {
    Cashflow {
        id: "fixed".into(),
        fixing_date: None,
        payment_date: payment,
        amount,
        kind: CashflowKind::Principal,
        currency: "USD".into(),
    }
}

#[test]
fn deterministic_event_free_theta_rebases_cashflows_without_any_model_or_paths() {
    let mut base = flat_context(&[], 1.0);
    base.forward_curve = None;
    let rolled = base
        .frozen_roll(ValuationCutoff {
            as_of: day(3),
            phase: CutoffPhase::AfterFixing,
        })
        .unwrap();
    let payoff = Known(vec![fixed(day(10), 100.0)]);
    let result = McEngine::new(0)
        .try_calendar_theta(&base, &payoff, &rolled, &payoff, None, &[])
        .unwrap();
    let expected = 100.0 * ((-0.03_f64 * 7.0 / 365.0).exp() - (-0.03_f64 * 10.0 / 365.0).exp());
    assert!((result.pv_change - expected).abs() < 1e-12);
    assert!((result.per_day - expected / 3.0).abs() < 1e-12);
    assert_eq!(result.std_error, 0.0);
    assert_eq!(result.samples, 0);
    assert!(result.base_date_discounted_change.abs() < 1e-12);
    assert_eq!(result.base.method, DatedPricingMethod::Deterministic);
    assert_eq!(result.base.estimate.samples, 0);
    assert_eq!(result.base.paths, 0);
}

#[test]
fn actual_paid_cash_is_separate_from_pv_change_and_reconciles_on_base_date() {
    let mut base = flat_context(&[], 1.0);
    base.forward_curve = None;
    let rolled = base
        .frozen_roll(ValuationCutoff {
            as_of: day(10),
            phase: CutoffPhase::AfterFixing,
        })
        .unwrap();
    let flow = fixed(day(5), 100.0);
    let b = Known(vec![flow.clone()]);
    let r = Known(vec![]);
    let result = McEngine::new(0)
        .try_calendar_theta(&base, &b, &rolled, &r, None, &[flow])
        .unwrap();
    assert_eq!(result.cash_paid, 100.0);
    assert_eq!(result.rolled.estimate.price, 0.0);
    assert!((result.pv_change + 100.0 * (-0.03_f64 * 5.0 / 365.0).exp()).abs() < 1e-12);
    assert!(result.base_date_discounted_change.abs() < 1e-12);
    // A payment is never silently removed merely because its contractual date passed.
    assert!(McEngine::new(0)
        .try_price_context(&rolled, None, &b)
        .is_err());
    assert!(McEngine::new(0)
        .try_calendar_theta(&base, &b, &rolled, &r, None, &[fixed(day(11), 100.0)])
        .is_err());
}

#[test]
fn paired_gbm_theta_matches_analytic_roll_and_uses_covariance_of_complete_pvs() {
    let base = flat_context(&[90], 13.0);
    let rolled = base
        .frozen_roll(ValuationCutoff {
            as_of: day(7),
            phase: CutoffPhase::AfterFixing,
        })
        .unwrap()
        .with_grid(TimeGrid::try_new(day(7), vec![day(90)], 9.0).unwrap())
        .unwrap();
    let payoff = Terminal {
        date: day(90),
        strike: 100.0,
    };
    let dyns = Dynamics::Gbm(GbmDynamics::try_new(0.22).unwrap());
    let engine = McEngine::new(40_000).with_seed(9041);
    let result = engine
        .clone()
        .with_parallel(false)
        .try_calendar_theta(&base, &payoff, &rolled, &payoff, Some(&dyns), &[])
        .unwrap();
    let parallel = engine
        .with_parallel(true)
        .try_calendar_theta(&base, &payoff, &rolled, &payoff, Some(&dyns), &[])
        .unwrap();
    assert_eq!(result, parallel);
    let expected = fuzzy_enigma::analytic::bs_call(100.0, 100.0, 0.03, 0.01, 0.22, 83.0 / 365.0)
        - fuzzy_enigma::analytic::bs_call(100.0, 100.0, 0.03, 0.01, 0.22, 90.0 / 365.0);
    assert!((result.pv_change - expected).abs() < 4.0 * result.std_error + 1e-12);
    let independence =
        (result.base.estimate.std_error.powi(2) + result.rolled.estimate.std_error.powi(2)).sqrt();
    assert!(result.std_error < 0.5 * independence);
    let variance = result.base.estimate.std_error.powi(2)
        + result.rolled.estimate.std_error.powi(2)
        - 2.0 * result.covariance / result.samples as f64;
    assert!((result.std_error.powi(2) - variance).abs() < 1e-12);
    assert!(result.union_times.len() > base.grid.times().len());
    assert!(result.union_times.len() > rolled.grid.times().len());
}

fn aggregate(union: &[f64], leg: &ValuationContext, normals: &[f64]) -> Vec<f64> {
    let shift = year_fraction(origin(), leg.as_of());
    let mut result = vec![];
    for interval in leg.grid.times().windows(2) {
        let start = shift + interval[0];
        let end = shift + interval[1];
        for factor in 0..2 {
            let mut brownian = 0.0;
            for j in 0..union.len() - 1 {
                if union[j] >= start && union[j + 1] <= end {
                    brownian += (union[j + 1] - union[j]).sqrt() * normals[2 * j + factor];
                }
            }
            result.push(brownian / (end - start).sqrt());
        }
    }
    result
}
fn stats(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    (
        mean,
        (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt(),
    )
}

#[test]
fn heston_union_noise_keeps_both_original_euler_grids_and_normal_ordering() {
    let base = flat_context(&[90], 30.0);
    let rolled = base
        .frozen_roll(ValuationCutoff {
            as_of: day(7),
            phase: CutoffPhase::AfterFixing,
        })
        .unwrap()
        .with_grid(TimeGrid::try_new(day(7), vec![day(90)], 20.75).unwrap())
        .unwrap();
    assert_eq!(base.grid.steps(), 3);
    assert_eq!(rolled.grid.steps(), 4);
    let payoff = Terminal {
        date: day(90),
        strike: 98.0,
    };
    let dyns = Dynamics::Heston(HestonDynamics::try_new(0.04, 1.5, 0.05, 0.9, -0.7).unwrap());
    let seed = 912;
    let result = McEngine::new(20)
        .with_seed(seed)
        .with_antithetic(false)
        .try_calendar_theta(&base, &payoff, &rolled, &payoff, Some(&dyns), &[])
        .unwrap();
    // Independent legacy generators provide an oracle for each original uniform leg.
    // The union is used to generate Brownian increments, not to update variance.
    let bm = HestonModel::new(
        100.0,
        0.03,
        0.01,
        0.04,
        1.5,
        0.05,
        0.9,
        -0.7,
        90.0 / 365.0,
        3,
    );
    let rm = HestonModel::new(
        100.0,
        0.03,
        0.01,
        0.04,
        1.5,
        0.05,
        0.9,
        -0.7,
        83.0 / 365.0,
        4,
    );
    let mut bs = vec![];
    let mut rs = vec![];
    for i in 0..20 {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        rng.set_stream(i);
        let normals: Vec<f64> = (0..2 * (result.union_times.len() - 1))
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        let mut bp = vec![0.0; 4];
        let mut rp = vec![0.0; 5];
        bm.generate(&aggregate(&result.union_times, &base, &normals), &mut bp);
        rm.generate(&aggregate(&result.union_times, &rolled, &normals), &mut rp);
        bs.push((bp[3] - 98.0).max(0.0) * (-0.03_f64 * 90.0 / 365.0).exp());
        rs.push((rp[4] - 98.0).max(0.0) * (-0.03_f64 * 83.0 / 365.0).exp());
    }
    let differences: Vec<_> = bs.iter().zip(&rs).map(|(b, r)| r - b).collect();
    let (delta, se) = stats(&differences);
    assert!((result.base.estimate.price - stats(&bs).0).abs() < 1e-11);
    assert!((result.rolled.estimate.price - stats(&rs).0).abs() < 1e-11);
    assert!((result.pv_change - delta).abs() < 1e-11);
    assert!((result.std_error - se).abs() < 1e-11);
}

#[test]
fn calendar_rejects_market_shocks_and_insufficient_paired_sampling() {
    let base = flat_context(&[90], 30.0);
    let mut rolled = base
        .frozen_roll(ValuationCutoff {
            as_of: day(7),
            phase: CutoffPhase::AfterFixing,
        })
        .unwrap();
    let p = Terminal {
        date: day(90),
        strike: 100.0,
    };
    let dyns = Dynamics::Gbm(GbmDynamics::try_new(0.2).unwrap());
    assert!(McEngine::new(2)
        .try_calendar_theta(&base, &p, &rolled, &p, Some(&dyns), &[])
        .is_err());
    rolled.source.push_str("changed policy");
    assert!(McEngine::new(10)
        .try_calendar_theta(&base, &p, &rolled, &p, Some(&dyns), &[])
        .is_err());
}

#[test]
fn one_leg_can_be_deterministic_after_economic_maturity_without_forward_extension() {
    let mut base = flat_context(&[5], 2.0);
    base.forward_curve = Some(
        ForwardCurve::try_new(
            origin(),
            "USD".into(),
            "TEST".into(),
            vec![
                CurveNode {
                    date: origin(),
                    value: 100.0,
                },
                CurveNode {
                    date: day(5),
                    value: 100.0 * (0.02_f64 * 5.0 / 365.0).exp(),
                },
            ],
            "short economic coverage".into(),
        )
        .unwrap(),
    );
    let rolled = base
        .frozen_roll(ValuationCutoff {
            as_of: day(10),
            phase: CutoffPhase::AfterFixing,
        })
        .unwrap();
    assert!(rolled.forward_curve.is_none());
    let b = Terminal {
        date: day(5),
        strike: 100.0,
    };
    let r = Known(vec![fixed(day(20), 1.0)]);
    let dyns = Dynamics::Gbm(GbmDynamics::try_new(0.2).unwrap());
    let result = McEngine::new(1000)
        .with_seed(67)
        .try_calendar_theta(&base, &b, &rolled, &r, Some(&dyns), &[])
        .unwrap();
    assert_eq!(result.rolled.method, DatedPricingMethod::Deterministic);
    assert_eq!(result.rolled.estimate.samples, 0);
    assert!(result.std_error > 0.0);
    assert!((result.std_error - result.base.estimate.std_error).abs() < 1e-12);
}

#[test]
fn calendar_rejects_nonfinite_reconciliation_and_paid_cash_totals() {
    let base = flat_context(&[], 1.0);
    let rolled = base
        .frozen_roll(ValuationCutoff {
            as_of: day(3),
            phase: CutoffPhase::AfterFixing,
        })
        .unwrap();
    let base_cashflow = Known(vec![fixed(day(10), -1e308)]);
    let rolled_cashflow = Known(vec![fixed(day(10), 1e308)]);
    let error = McEngine::new(0)
        .try_calendar_theta(&base, &base_cashflow, &rolled, &rolled_cashflow, None, &[])
        .unwrap_err();
    assert!(error.to_string().contains("calendar_pv_change"));
    let empty = Known(vec![]);
    let error = McEngine::new(0)
        .try_calendar_theta(
            &base,
            &empty,
            &rolled,
            &empty,
            None,
            &[fixed(day(1), 1e308), fixed(day(2), 1e308)],
        )
        .unwrap_err();
    assert!(error.to_string().contains("cash_paid"));
}
