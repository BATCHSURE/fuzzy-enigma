//! Manual dated cashflow, history/state and actual settlement regressions.

use fuzzy_enigma::dated::{
    curves::{CurveNode, DiscountCurve, ForwardCurve, TimeGrid, ValuationContext},
    notes::{
        advance_note_state, replay_note_history, roll_note_state, ActualPayment, DatedNote,
        DatedPhoenixNote, DatedSnowballNote, EventSchedule, Fixing, NoteState, PreparedNote,
        SettlementConfirmation, StateProvenance, Termination,
    },
    Cashflow, CashflowKind, CashflowPayoff, CutoffPhase, Date, Dynamics, GbmDynamics,
    HestonDynamics, ValuationCutoff,
};
use fuzzy_enigma::{McEngine, Payoff, PhoenixNote, SnowballNote};

fn date(month: u32, day: u32) -> Date {
    Date::from_ymd_opt(2026, month, day).unwrap()
}
fn issue() -> Date {
    date(1, 1)
}
fn maturity() -> Date {
    date(5, 1)
}
fn after(as_of: Date) -> ValuationCutoff {
    ValuationCutoff {
        as_of,
        phase: CutoffPhase::AfterFixing,
    }
}
fn before(as_of: Date) -> ValuationCutoff {
    ValuationCutoff {
        as_of,
        phase: CutoffPhase::BeforeFixing,
    }
}
fn fixing(month: u32, spot: f64) -> Fixing {
    Fixing {
        date: date(month, 1),
        spot,
    }
}
fn schedule(months: &[u32], lag: u32) -> EventSchedule {
    EventSchedule::new(
        months.iter().map(|&m| date(m, 1)).collect(),
        months.iter().map(|&m| date(m, 1 + lag)).collect(),
        "explicit teaching dates".into(),
    )
    .unwrap()
}
fn snowball() -> DatedNote {
    DatedSnowballNote::new(
        100.0,
        100.0,
        0.12,
        70.0,
        105.0,
        issue(),
        "USD".into(),
        "SPX".into(),
        (1..=5).map(|m| date(m, 1)).collect(),
        schedule(&[3, 5], 3),
        schedule(&[5], 3),
    )
    .unwrap()
    .into()
}
fn phoenix(memory: bool) -> DatedNote {
    DatedPhoenixNote::new(
        100.0,
        100.0,
        3.0,
        90.0,
        70.0,
        105.0,
        issue(),
        "USD".into(),
        "SPX".into(),
        (1..=5).map(|m| date(m, 1)).collect(),
        schedule(&[2, 3, 4, 5], 2),
        schedule(&[3, 5], 3),
        schedule(&[5], 3),
        memory,
    )
    .unwrap()
    .into()
}
fn initial(note: &DatedNote) -> NoteState {
    replay_note_history(note, &[], after(issue()), &[], None).unwrap()
}
fn context(prepared: &PreparedNote, spot: f64, max_step_days: f64) -> ValuationContext {
    let as_of = prepared.state().cutoff().as_of;
    let end = date(6, 1).max(as_of);
    let t = (end - as_of).num_days() as f64 / 365.0;
    let nodes = if end == as_of {
        vec![CurveNode {
            date: as_of,
            value: 1.0,
        }]
    } else {
        vec![
            CurveNode {
                date: as_of,
                value: 1.0,
            },
            CurveNode {
                date: end,
                value: (-0.04 * t).exp(),
            },
        ]
    };
    let discount =
        DiscountCurve::try_new(as_of, "USD".into(), nodes, "flat teaching curve".into()).unwrap();
    let f_nodes = if end == as_of {
        vec![CurveNode {
            date: as_of,
            value: spot,
        }]
    } else {
        vec![
            CurveNode {
                date: as_of,
                value: spot,
            },
            CurveNode {
                date: end,
                value: spot,
            },
        ]
    };
    let forward = ForwardCurve::try_new(
        as_of,
        "USD".into(),
        "SPX".into(),
        f_nodes,
        "flat teaching carry".into(),
    )
    .unwrap();
    let ctx = ValuationContext::try_new(
        prepared.state().cutoff(),
        spot,
        "USD".into(),
        "SPX".into(),
        discount,
        Some(forward),
        "synthetic".into(),
    )
    .unwrap();
    ctx.with_grid(TimeGrid::try_new(as_of, prepared.future_fixing_dates(), max_step_days).unwrap())
        .unwrap()
}
fn manual(prepared: &PreparedNote, values: &[Fixing], step: f64) -> Vec<Cashflow> {
    let ctx = context(prepared, 100.0, step);
    prepared.validate(&ctx).unwrap();
    let mut path = vec![100.0; ctx.grid.steps() + 1];
    for fix in values {
        if let Ok(i) = ctx.grid.index(fix.date) {
            path[i] = fix.spot;
        }
    }
    let mut cashflows = prepared.known_cashflows().to_vec();
    prepared.cashflows(&path, &ctx, &mut cashflows).unwrap();
    cashflows
}
fn amount(cashflows: &[Cashflow], kind: CashflowKind) -> f64 {
    cashflows
        .iter()
        .filter(|cf| cf.kind == kind)
        .map(|cf| cf.amount)
        .sum()
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-10, "{a} != {b}");
}
fn confirmation(
    note: &DatedNote,
    kind: CashflowKind,
    fixing_date: Date,
    as_of: Date,
    payments: Vec<ActualPayment>,
    expected_payment_date: Option<Date>,
) -> SettlementConfirmation {
    SettlementConfirmation {
        cashflow_id: note.cashflow_id(kind, fixing_date).unwrap(),
        as_of,
        payments,
        expected_payment_date,
    }
}
fn receipt(id: &str, payment_date: Date, amount: f64) -> ActualPayment {
    ActualPayment {
        payment_id: id.into(),
        payment_date,
        amount,
    }
}

#[test]
fn explicit_schedules_reject_empty_duplicate_unordered_and_early_payments() {
    for (fixing_dates, payment_dates) in [
        (vec![], vec![]),
        (vec![date(2, 1)], vec![]),
        (vec![date(2, 1), date(2, 1)], vec![date(2, 3), date(2, 3)]),
        (vec![date(3, 1), date(2, 1)], vec![date(3, 3), date(2, 3)]),
        (vec![date(2, 1)], vec![date(1, 31)]),
    ] {
        assert!(EventSchedule::new(fixing_dates, payment_dates, "supplied".into()).is_err());
    }
    assert!(EventSchedule::new(vec![date(2, 1)], vec![date(2, 1)], " ".into()).is_err());
    // Separate fixing events can share one settlement date.
    assert!(EventSchedule::new(
        vec![date(2, 1), date(3, 1)],
        vec![date(3, 3), date(3, 3)],
        "supplied".into()
    )
    .is_ok());
}

#[test]
fn dated_contracts_require_issue_knockin_and_original_numeric_domains() {
    for ki_dates in [
        vec![],
        vec![date(2, 1)],
        vec![issue(), issue()],
        vec![issue(), date(6, 1)],
    ] {
        assert!(DatedSnowballNote::new(
            100.0,
            100.0,
            0.12,
            70.0,
            105.0,
            issue(),
            "USD".into(),
            "SPX".into(),
            ki_dates,
            schedule(&[3, 5], 3),
            schedule(&[5], 3)
        )
        .is_err());
    }
    for invalid in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
        assert!(DatedSnowballNote::new(
            invalid,
            100.0,
            0.12,
            70.0,
            105.0,
            issue(),
            "USD".into(),
            "SPX".into(),
            vec![issue()],
            schedule(&[3], 3),
            schedule(&[5], 3)
        )
        .is_err());
        assert!(DatedPhoenixNote::new(
            100.0,
            100.0,
            3.0,
            invalid,
            70.0,
            105.0,
            issue(),
            "USD".into(),
            "SPX".into(),
            vec![issue()],
            schedule(&[2], 2),
            schedule(&[3], 3),
            schedule(&[5], 3),
            true
        )
        .is_err());
    }
    assert!(DatedSnowballNote::new(
        100.0,
        100.0,
        -0.1,
        70.0,
        105.0,
        issue(),
        "USD".into(),
        "SPX".into(),
        vec![issue()],
        schedule(&[3], 3),
        schedule(&[5], 3)
    )
    .is_err());
    assert!(DatedSnowballNote::new(
        100.0,
        100.0,
        0.12,
        70.0,
        105.0,
        issue(),
        "USD".into(),
        "SPX".into(),
        vec![issue()],
        schedule(&[3], 3),
        schedule(&[4, 5], 3)
    )
    .is_err());
}

#[test]
fn snowball_early_autocall_uses_issue_accrual_and_payment_lag() {
    let note = snowball();
    let prepared = PreparedNote::new(note.clone(), initial(&note)).unwrap();
    let flows = manual(
        &prepared,
        &[
            fixing(2, 50.0),
            fixing(3, 105.0),
            fixing(4, 1.0),
            fixing(5, 1.0),
        ],
        10.0,
    );
    assert_eq!(flows.len(), 2);
    close(amount(&flows, CashflowKind::Principal), 100.0);
    close(
        amount(&flows, CashflowKind::Coupon),
        100.0 * 0.12 * (date(3, 1) - issue()).num_days() as f64 / 365.0,
    );
    assert!(flows.iter().all(|cf| cf.payment_date == date(3, 4)));
    assert!(flows.iter().all(|cf| cf.fixing_date == Some(date(3, 1))));
}

#[test]
fn snowball_maturity_branches_include_knockin_recovery_and_loss() {
    let note = snowball();
    let prepared = PreparedNote::new(note.clone(), initial(&note)).unwrap();
    let healthy = manual(
        &prepared,
        &[
            fixing(2, 80.0),
            fixing(3, 90.0),
            fixing(4, 100.0),
            fixing(5, 100.0),
        ],
        10.0,
    );
    close(amount(&healthy, CashflowKind::Principal), 100.0);
    close(
        amount(&healthy, CashflowKind::Coupon),
        100.0 * 0.12 * (maturity() - issue()).num_days() as f64 / 365.0,
    );
    let loss = manual(
        &prepared,
        &[
            fixing(2, 70.0),
            fixing(3, 80.0),
            fixing(4, 90.0),
            fixing(5, 95.0),
        ],
        10.0,
    );
    close(amount(&loss, CashflowKind::Principal), 95.0);
    close(amount(&loss, CashflowKind::Coupon), 0.0);
    let recovery = manual(
        &prepared,
        &[
            fixing(2, 60.0),
            fixing(3, 80.0),
            fixing(4, 90.0),
            fixing(5, 104.0),
        ],
        10.0,
    );
    close(amount(&recovery, CashflowKind::Principal), 100.0);
    close(amount(&recovery, CashflowKind::Coupon), 0.0);
}

#[test]
fn initial_original_fixing_always_counts_for_knockin() {
    let note: DatedNote = DatedSnowballNote::new(
        100.0,
        70.0,
        0.12,
        70.0,
        105.0,
        issue(),
        "USD".into(),
        "SPX".into(),
        vec![issue(), maturity()],
        schedule(&[5], 3),
        schedule(&[5], 3),
    )
    .unwrap()
    .into();
    let state = replay_note_history(&note, &[], before(issue()), &[], None).unwrap();
    assert!(state.knocked_in());
    assert_eq!(state.first_knock_in_date(), Some(issue()));
    assert_eq!(
        state.fixings(),
        &[Fixing {
            date: issue(),
            spot: 70.0
        }]
    );
}

#[test]
fn phoenix_memory_catches_up_at_eligible_payment_then_expires() {
    let note = phoenix(true);
    let prepared = PreparedNote::new(note.clone(), initial(&note)).unwrap();
    let flows = manual(
        &prepared,
        &[
            fixing(2, 80.0),
            fixing(3, 80.0),
            fixing(4, 90.0),
            fixing(5, 80.0),
        ],
        10.0,
    );
    close(amount(&flows, CashflowKind::Coupon), 9.0);
    assert_eq!(
        flows
            .iter()
            .find(|cf| cf.kind == CashflowKind::Coupon)
            .unwrap()
            .payment_date,
        date(4, 3)
    );
    let paid = confirmation(
        &note,
        CashflowKind::Coupon,
        date(4, 1),
        maturity(),
        vec![receipt("catchup", date(4, 3), 9.0)],
        None,
    );
    let state = replay_note_history(
        &note,
        &[
            fixing(2, 80.0),
            fixing(3, 80.0),
            fixing(4, 90.0),
            fixing(5, 80.0),
        ],
        after(maturity()),
        &[paid],
        None,
    )
    .unwrap();
    assert!(state.memory_coupon_ids().is_empty());
    assert_eq!(state.expired_memory_coupon_ids().len(), 1);
    assert_eq!(state.cashflows()[0].component_ids.len(), 3);
}

#[test]
fn phoenix_no_memory_coupon_and_knockin_principal_are_independent() {
    let note = phoenix(false);
    let prepared = PreparedNote::new(note.clone(), initial(&note)).unwrap();
    let flows = manual(
        &prepared,
        &[
            fixing(2, 70.0),
            fixing(3, 80.0),
            fixing(4, 80.0),
            fixing(5, 90.0),
        ],
        10.0,
    );
    close(amount(&flows, CashflowKind::Coupon), 3.0);
    close(amount(&flows, CashflowKind::Principal), 90.0);
}

#[test]
fn phoenix_same_day_coupon_precedes_autocall_and_survives_distinct_payment_dates() {
    let note = phoenix(true);
    let state = replay_note_history(
        &note,
        &[fixing(2, 80.0), fixing(3, 105.0)],
        after(date(3, 1)),
        &[],
        None,
    )
    .unwrap();
    assert_eq!(
        state.termination(),
        Some(&Termination::Redeemed {
            fixing_date: date(3, 1)
        })
    );
    assert_eq!(state.cashflows().len(), 2);
    let coupon = &state.cashflows()[0];
    close(coupon.cashflow.amount, 6.0);
    assert_eq!(coupon.cashflow.payment_date, date(3, 3));
    assert_eq!(coupon.component_ids.len(), 2);
    assert_eq!(state.cashflows()[1].cashflow.payment_date, date(3, 4));
    assert!(state.memory_coupon_ids().is_empty());
    let prepared = PreparedNote::new(note, state).unwrap();
    assert!(!prepared.requires_simulation());
    assert!(prepared.future_fixing_dates().is_empty());
    assert_eq!(prepared.known_cashflows().len(), 2);
}

#[test]
fn maturity_autocall_overrides_prior_knockin() {
    let note = phoenix(true);
    let prepared = PreparedNote::new(note.clone(), initial(&note)).unwrap();
    let flows = manual(
        &prepared,
        &[
            fixing(2, 50.0),
            fixing(3, 80.0),
            fixing(4, 80.0),
            fixing(5, 105.0),
        ],
        10.0,
    );
    close(amount(&flows, CashflowKind::Principal), 100.0);
    close(amount(&flows, CashflowKind::Coupon), 12.0);
}

#[test]
fn observation_monitoring_is_independent_of_refinement_and_unscheduled_lows() {
    let note: DatedNote = DatedSnowballNote::new(
        100.0,
        100.0,
        0.12,
        70.0,
        120.0,
        issue(),
        "USD".into(),
        "SPX".into(),
        vec![issue(), maturity()],
        schedule(&[3, 5], 3),
        schedule(&[5], 3),
    )
    .unwrap()
    .into();
    let prepared = PreparedNote::new(note.clone(), initial(&note)).unwrap();
    let mut results = Vec::new();
    for step in [31.0, 1.0] {
        let ctx = context(&prepared, 100.0, step);
        let mut path = vec![1.0; ctx.grid.steps() + 1];
        for d in prepared.future_fixing_dates() {
            path[ctx.grid.index(d).unwrap()] = 100.0;
        }
        let mut flows = Vec::new();
        prepared.cashflows(&path, &ctx, &mut flows).unwrap();
        results.push(flows);
    }
    assert_eq!(results[0], results[1]);
    assert!(amount(&results[0], CashflowKind::Coupon) > 0.0);
}

#[test]
fn full_history_and_split_history_preserve_cashflows_and_hashes() {
    let note = phoenix(true);
    let history = [
        fixing(2, 70.0),
        fixing(3, 80.0),
        fixing(4, 95.0),
        fixing(5, 90.0),
    ];
    let middle = replay_note_history(&note, &history[..2], after(date(3, 1)), &[], None).unwrap();
    assert!(middle.knocked_in());
    assert_eq!(middle.memory_coupon_ids().len(), 2);
    let split = PreparedNote::new(note.clone(), middle.clone()).unwrap();
    let split_flows = manual(&split, &history[2..], 10.0);
    let full = PreparedNote::new(note.clone(), initial(&note)).unwrap();
    assert_eq!(split_flows, manual(&full, &history, 10.0));
    let paid = confirmation(
        &note,
        CashflowKind::Coupon,
        date(4, 1),
        maturity(),
        vec![receipt("apr", date(4, 3), 9.0)],
        None,
    );
    let advanced = advance_note_state(
        &note,
        &middle,
        &history[2..],
        after(maturity()),
        std::slice::from_ref(&paid),
        None,
    )
    .unwrap();
    let replayed = replay_note_history(&note, &history, after(maturity()), &[paid], None).unwrap();
    assert_eq!(advanced, replayed);
    assert_eq!(
        advanced
            .cashflows()
            .iter()
            .map(|cf| cf.cashflow.clone())
            .collect::<Vec<_>>(),
        split_flows
    );
}

#[test]
fn before_fixing_requires_scenario_and_never_uses_valuation_spot() {
    let note = phoenix(true);
    let state =
        replay_note_history(&note, &[fixing(2, 80.0)], before(date(3, 1)), &[], None).unwrap();
    let prepared = PreparedNote::new(note.clone(), state).unwrap();
    assert!(prepared.validate(&context(&prepared, 105.0, 10.0)).is_err());
    let state = replay_note_history(
        &note,
        &[fixing(2, 80.0)],
        before(date(3, 1)),
        &[],
        Some(fixing(3, 105.0)),
    )
    .unwrap();
    assert!(state.termination().is_none());
    assert_eq!(state.provenance(), StateProvenance::FixingScenario);
    let prepared = PreparedNote::new(note.clone(), state.clone()).unwrap();
    assert!(!prepared.requires_simulation());
    assert!(prepared.future_fixing_dates().is_empty());
    let ctx = context(&prepared, 10.0, 10.0);
    prepared.validate(&ctx).unwrap();
    close(
        amount(prepared.known_cashflows(), CashflowKind::Coupon),
        6.0,
    );
    close(
        amount(prepared.known_cashflows(), CashflowKind::Principal),
        100.0,
    );
    assert_eq!(prepared.state(), &state); // Projection does not mutate actual history/cutoff.
    assert!(advance_note_state(
        &note,
        &state,
        &[fixing(3, 105.0)],
        after(date(3, 1)),
        &[],
        None,
    )
    .is_err());
    let after_state = replay_note_history(
        &note,
        &[fixing(2, 80.0), fixing(3, 105.0)],
        after(date(3, 1)),
        &[],
        None,
    )
    .unwrap();
    assert_eq!(after_state.provenance(), StateProvenance::Actual);
    assert_eq!(
        after_state
            .cashflows()
            .iter()
            .map(|cf| cf.cashflow.clone())
            .collect::<Vec<_>>(),
        prepared.known_cashflows()
    );
}

#[test]
fn today_maturity_scenario_projects_known_cashflows_with_no_future_grid() {
    let note = snowball();
    let state = replay_note_history(
        &note,
        &[fixing(2, 70.0), fixing(3, 80.0), fixing(4, 90.0)],
        before(maturity()),
        &[],
        Some(fixing(5, 90.0)),
    )
    .unwrap();
    let prepared = PreparedNote::new(note, state).unwrap();
    assert!(!prepared.requires_simulation());
    close(
        amount(prepared.known_cashflows(), CashflowKind::Principal),
        90.0,
    );
    close(
        amount(prepared.known_cashflows(), CashflowKind::Coupon),
        0.0,
    );
    assert!(prepared.future_fixing_dates().is_empty());
    prepared.validate(&context(&prepared, 1.0, 1.0)).unwrap();
}

#[test]
fn today_scenario_is_not_duplicated_when_note_remains_active() {
    let note = phoenix(true);
    let state = replay_note_history(
        &note,
        &[fixing(2, 80.0)],
        before(date(3, 1)),
        &[],
        Some(fixing(3, 90.0)),
    )
    .unwrap();
    let prepared = PreparedNote::new(note, state).unwrap();
    assert!(prepared.requires_simulation());
    assert!(!prepared.future_fixing_dates().contains(&date(3, 1)));
    let flows = manual(&prepared, &[fixing(4, 80.0), fixing(5, 80.0)], 10.0);
    close(amount(&flows, CashflowKind::Coupon), 6.0);
}

#[test]
fn after_fixing_same_day_scenario_is_explicit_and_cannot_replace_actual_history() {
    let note = snowball();
    let state = replay_note_history(
        &note,
        &[fixing(2, 80.0)],
        after(date(3, 1)),
        &[],
        Some(fixing(3, 105.0)),
    )
    .unwrap();
    assert!(state.termination().is_some());
    assert_eq!(state.provenance(), StateProvenance::FixingScenario);
    assert!(replay_note_history(
        &note,
        &[fixing(2, 80.0), fixing(3, 105.0)],
        after(date(3, 1)),
        &[],
        Some(fixing(3, 105.0))
    )
    .is_err());
}

#[test]
fn history_rejects_missing_duplicate_future_and_conflicting_issue_fixings() {
    let note = snowball();
    for history in [
        vec![],
        vec![fixing(2, 80.0), fixing(2, 80.0)],
        vec![fixing(3, 80.0), fixing(2, 80.0)],
        vec![fixing(2, 80.0), fixing(3, 80.0), fixing(4, 80.0)],
        vec![
            Fixing {
                date: issue(),
                spot: 99.0,
            },
            fixing(2, 80.0),
            fixing(3, 80.0),
        ],
    ] {
        assert!(replay_note_history(&note, &history, after(date(3, 1)), &[], None).is_err());
    }
    assert!(
        replay_note_history(&note, &[fixing(2, f64::NAN)], after(date(2, 1)), &[], None).is_err()
    );
    assert!(replay_note_history(
        &note,
        &[],
        after(date(1, 1) - chrono::Duration::days(1)),
        &[],
        None
    )
    .is_err());
}

fn partially_paid() -> (DatedNote, NoteState, Vec<SettlementConfirmation>) {
    let note = snowball();
    let as_of = date(3, 10);
    let coupon = 100.0 * 0.12 * (date(3, 1) - issue()).num_days() as f64 / 365.0;
    let confirmations = vec![
        confirmation(
            &note,
            CashflowKind::Principal,
            date(3, 1),
            as_of,
            vec![receipt("principal-1", date(3, 4), 40.0)],
            Some(date(3, 12)),
        ),
        confirmation(
            &note,
            CashflowKind::Coupon,
            date(3, 1),
            as_of,
            vec![receipt("coupon-1", date(3, 4), coupon)],
            None,
        ),
    ];
    let state = replay_note_history(
        &note,
        &[fixing(2, 80.0), fixing(3, 105.0)],
        after(as_of),
        &confirmations,
        None,
    )
    .unwrap();
    (note, state, confirmations)
}

#[test]
fn partial_actual_receipts_preserve_contractual_dates_and_remaining_claim() {
    let (note, state, _) = partially_paid();
    assert!(state.termination().is_some());
    assert_eq!(state.cashflows().len(), 2);
    assert_eq!(state.receivables().len(), 1);
    let remaining = &state.receivables()[0];
    assert_eq!(remaining.settlement_status(), "partial");
    assert_eq!(remaining.cashflow.payment_date, date(3, 4));
    assert_eq!(remaining.expected_payment_date, Some(date(3, 12)));
    close(remaining.paid_amount, 40.0);
    close(remaining.outstanding_amount, 60.0);
    let prepared = PreparedNote::new(note, state).unwrap();
    assert!(!prepared.requires_simulation());
    assert_eq!(prepared.known_cashflows()[0].payment_date, date(3, 12));
    close(prepared.known_cashflows()[0].amount, 60.0);
}

#[test]
fn overdue_missing_confirmation_expected_date_or_stale_confirmation_fails() {
    let (note, _, confirmations) = partially_paid();
    let history = [fixing(2, 80.0), fixing(3, 105.0)];
    assert!(
        replay_note_history(&note, &history, after(date(3, 10)), &[], None)
            .unwrap_err()
            .to_string()
            .contains(
                &note
                    .cashflow_id(CashflowKind::Principal, date(3, 1))
                    .unwrap()
            )
    );
    for expected in [None, Some(date(3, 9))] {
        let mut bad = confirmations.clone();
        bad[0].expected_payment_date = expected;
        assert!(replay_note_history(&note, &history, after(date(3, 10)), &bad, None).is_err());
    }
    let mut stale = confirmations.clone();
    stale[0].as_of = date(3, 9);
    assert!(replay_note_history(&note, &history, after(date(3, 10)), &stale, None).is_err());
    // An explicitly declared same-day expected payment is valid at D(0).
    let mut today = confirmations;
    today[0].expected_payment_date = Some(date(3, 10));
    let state = replay_note_history(&note, &history, after(date(3, 10)), &today, None).unwrap();
    assert_eq!(
        state.receivables()[0].expected_payment_date,
        Some(date(3, 10))
    );
}

#[test]
fn due_today_never_implicitly_means_paid_or_pending_at_today() {
    let note = snowball();
    let history = [fixing(2, 80.0), fixing(3, 105.0)];
    assert!(replay_note_history(&note, &history, after(date(3, 4)), &[], None).is_err());
    let mut confirmations = vec![
        confirmation(
            &note,
            CashflowKind::Principal,
            date(3, 1),
            date(3, 4),
            vec![],
            None,
        ),
        confirmation(
            &note,
            CashflowKind::Coupon,
            date(3, 1),
            date(3, 4),
            vec![],
            Some(date(3, 5)),
        ),
    ];
    assert!(replay_note_history(&note, &history, after(date(3, 4)), &confirmations, None).is_err());
    confirmations[0].expected_payment_date = Some(date(3, 4));
    assert!(replay_note_history(&note, &history, after(date(3, 4)), &confirmations, None).is_ok());
}

#[test]
fn settlement_validation_rejects_receipt_duplicates_overpay_and_early_receipts() {
    let (note, _, confirmations) = partially_paid();
    let history = [fixing(2, 80.0), fixing(3, 105.0)];
    for case in 0..7 {
        let mut bad = confirmations.clone();
        match case {
            0 => bad.push(bad[0].clone()),
            1 => {
                let duplicate = bad[0].payments[0].clone();
                bad[0].payments.push(duplicate);
            }
            2 => bad[0].payments[0].amount = 101.0,
            3 => bad[0].payments[0].payment_date = date(2, 28),
            4 => bad[0].payments[0].amount = f64::NAN,
            5 => bad[0].payments[0].payment_date = date(3, 11),
            6 => bad[1].expected_payment_date = Some(date(3, 12)),
            _ => unreachable!(),
        }
        assert!(
            replay_note_history(&note, &history, after(date(3, 10)), &bad, None).is_err(),
            "case {case}"
        );
    }
    let mut unknown = confirmations;
    unknown[0].cashflow_id = "unknown".into();
    assert!(replay_note_history(&note, &history, after(date(3, 10)), &unknown, None).is_err());
}

#[test]
fn actual_advance_preserves_receipts_and_can_fully_settle_redeemed_trade() {
    let (note, state, mut confirmations) = partially_paid();
    assert!(advance_note_state(&note, &state, &[], after(date(3, 11)), &[], None).is_err());
    confirmations[0].as_of = date(3, 11);
    confirmations[0]
        .payments
        .push(receipt("principal-2", date(3, 11), 60.0));
    confirmations[0].expected_payment_date = None;
    let settled =
        advance_note_state(&note, &state, &[], after(date(3, 11)), &confirmations, None).unwrap();
    assert!(settled.receivables().is_empty());
    assert!(settled
        .cashflows()
        .iter()
        .all(|entry| entry.settlement_status() == "paid"));
    let prepared = PreparedNote::new(note.clone(), settled).unwrap();
    assert!(!prepared.requires_simulation());
    assert!(prepared.known_cashflows().is_empty());
    let mut corrupt = confirmations;
    corrupt[0].payments.remove(0);
    assert!(advance_note_state(&note, &state, &[], after(date(3, 11)), &corrupt, None).is_err());
}

#[test]
fn frozen_roll_keeps_unpaid_balance_but_crossed_payment_requires_update() {
    let (note, state, mut confirmations) = partially_paid();
    let rolled = roll_note_state(&note, &state, &[], after(date(3, 11)), &[], None).unwrap();
    assert_eq!(rolled.provenance(), StateProvenance::FrozenRoll);
    assert_eq!(rolled.fixings(), state.fixings());
    assert_ne!(rolled.history_hash(), state.history_hash()); // Provenance is bound.
    close(rolled.receivables()[0].outstanding_amount, 60.0);
    assert!(roll_note_state(&note, &state, &[], after(date(3, 12)), &[], None).is_err());
    confirmations[0].as_of = date(3, 12);
    confirmations[0].expected_payment_date = Some(date(3, 14));
    let extended =
        roll_note_state(&note, &state, &[], after(date(3, 12)), &confirmations, None).unwrap();
    assert_eq!(
        extended.receivables()[0].expected_payment_date,
        Some(date(3, 14))
    );
    close(extended.receivables()[0].outstanding_amount, 60.0);
}

#[test]
fn advance_cannot_append_backdated_actual_receipts() {
    let (note, state, mut confirmations) = partially_paid();
    confirmations[0].as_of = date(3, 11);
    confirmations[0]
        .payments
        .push(receipt("backdated", date(3, 9), 10.0));
    assert!(
        advance_note_state(&note, &state, &[], after(date(3, 11)), &confirmations, None).is_err()
    );
    confirmations[0].payments.last_mut().unwrap().payment_date = date(3, 10);
    assert!(
        advance_note_state(&note, &state, &[], after(date(3, 11)), &confirmations, None).is_err()
    );
    // Full replay is the explicit correction path, without altering old state.
    let corrected = replay_note_history(
        &note,
        &[fixing(2, 80.0), fixing(3, 105.0)],
        after(date(3, 11)),
        &confirmations,
        None,
    )
    .unwrap();
    close(corrected.receivables()[0].outstanding_amount, 50.0);
    close(state.receivables()[0].outstanding_amount, 60.0);
}

#[test]
fn immutable_history_prefix_contract_binding_and_cutoff_validation() {
    let note = snowball();
    let state =
        replay_note_history(&note, &[fixing(2, 80.0)], after(date(2, 1)), &[], None).unwrap();
    assert!(advance_note_state(
        &note,
        &state,
        &[fixing(2, 85.0)],
        after(date(3, 1)),
        &[],
        None
    )
    .is_err());
    assert!(advance_note_state(&note, &state, &[], before(date(2, 1)), &[], None).is_err());
    assert!(PreparedNote::new(phoenix(true), state.clone()).is_err());
    let prepared = PreparedNote::new(note.clone(), state).unwrap();
    let mut ctx = context(&prepared, 80.0, 10.0);
    ctx.cutoff = after(date(2, 2));
    assert!(prepared.validate(&ctx).is_err());
    assert_eq!(note.reference_spot(), 100.0);
}

#[test]
fn fixed_reference_does_not_follow_market_spot_or_end_observation() {
    let note = snowball();
    let state =
        replay_note_history(&note, &[fixing(2, 70.0)], after(date(2, 1)), &[], None).unwrap();
    let prepared = PreparedNote::new(note.clone(), state).unwrap();
    for spot in [80.0, 90.0] {
        let ctx = context(&prepared, spot, 10.0);
        let path = vec![spot; ctx.grid.steps() + 1];
        let mut flows = Vec::new();
        prepared.cashflows(&path, &ctx, &mut flows).unwrap();
        close(amount(&flows, CashflowKind::Principal), spot);
    }
    assert_eq!(note.reference_spot(), 100.0);
}

#[test]
fn today_autocall_and_maturity_scenarios_price_without_paths_or_dynamics() {
    for (note, history, cutoff, scenario, principal, coupon) in [
        (
            phoenix(true),
            vec![fixing(2, 80.0)],
            before(date(3, 1)),
            fixing(3, 105.0),
            100.0,
            6.0,
        ),
        (
            snowball(),
            vec![fixing(2, 70.0), fixing(3, 80.0), fixing(4, 90.0)],
            before(maturity()),
            fixing(5, 90.0),
            90.0,
            0.0,
        ),
    ] {
        let state = replay_note_history(&note, &history, cutoff, &[], Some(scenario)).unwrap();
        let prepared = PreparedNote::new(note, state).unwrap();
        let mut ctx = context(&prepared, 1.0, 10.0);
        ctx.forward_curve = None;
        let priced = McEngine::new(0)
            .try_price_context(&ctx, None, &prepared)
            .unwrap();
        let expected = prepared
            .known_cashflows()
            .iter()
            .map(|cf| cf.amount * ctx.discount_curve.value_on(cf.payment_date).unwrap())
            .sum::<f64>();
        close(priced.estimate.price, expected);
        close(priced.estimate.std_error, 0.0);
        close(
            amount(prepared.known_cashflows(), CashflowKind::Principal),
            principal,
        );
        close(
            amount(prepared.known_cashflows(), CashflowKind::Coupon),
            coupon,
        );
    }
}

#[test]
fn overdue_partial_receivable_discounts_expected_date_once() {
    let (note, state, _) = partially_paid();
    let prepared = PreparedNote::new(note, state).unwrap();
    let mut ctx = context(&prepared, 100.0, 10.0);
    ctx.forward_curve = None;
    let priced = McEngine::new(0)
        .try_price_context(&ctx, None, &prepared)
        .unwrap();
    close(
        priced.estimate.price,
        60.0 * ctx.discount_curve.value_on(date(3, 12)).unwrap(),
    );
    close(priced.estimate.std_error, 0.0);
}

#[test]
fn separate_coupon_and_principal_payment_dates_use_their_own_discount_factors() {
    let note = phoenix(true);
    let state = replay_note_history(
        &note,
        &[fixing(2, 80.0), fixing(3, 105.0)],
        after(date(3, 1)),
        &[],
        None,
    )
    .unwrap();
    let prepared = PreparedNote::new(note, state).unwrap();
    let mut ctx = context(&prepared, 100.0, 10.0);
    ctx.discount_curve = DiscountCurve::try_new(
        date(3, 1),
        "USD".into(),
        vec![
            CurveNode {
                date: date(3, 1),
                value: 1.0,
            },
            CurveNode {
                date: date(3, 3),
                value: 0.99,
            },
            CurveNode {
                date: date(3, 4),
                value: 0.97,
            },
        ],
        "non-flat reference".into(),
    )
    .unwrap();
    let priced = McEngine::new(0)
        .try_price_context(&ctx, None, &prepared)
        .unwrap();
    close(priced.estimate.price, 6.0 * 0.99 + 100.0 * 0.97);
}

#[test]
fn dated_notes_price_under_both_models_reproducibly() {
    for note in [snowball(), phoenix(true)] {
        let prepared = PreparedNote::new(note.clone(), initial(&note)).unwrap();
        let ctx = context(&prepared, 100.0, 5.0);
        for dynamics in [
            Dynamics::Gbm(GbmDynamics::try_new(0.25).unwrap()),
            Dynamics::Heston(HestonDynamics::try_new(0.0625, 1.5, 0.0625, 0.3, -0.7).unwrap()),
        ] {
            let serial = McEngine::new(128)
                .with_seed(19)
                .with_parallel(false)
                .try_price_context(&ctx, Some(&dynamics), &prepared)
                .unwrap();
            let parallel = McEngine::new(128)
                .with_seed(19)
                .with_parallel(true)
                .try_price_context(&ctx, Some(&dynamics), &prepared)
                .unwrap();
            assert_eq!(serial.estimate.price, parallel.estimate.price);
            assert_eq!(serial.estimate.std_error, parallel.estimate.std_error);
            assert!(serial.estimate.price.is_finite() && serial.estimate.price > 0.0);
            assert!(serial.estimate.std_error.is_finite());
        }
    }
}

#[test]
fn matched_uniform_contract_dates_reproduce_legacy_cashflow_pv() {
    let dates: Vec<_> = (0..=4)
        .map(|i| issue() + chrono::Duration::days(30 * i))
        .collect();
    let coupons = EventSchedule::new(
        dates[1..].to_vec(),
        dates[1..].to_vec(),
        "uniform comparison".into(),
    )
    .unwrap();
    let calls = EventSchedule::new(
        vec![dates[2], dates[4]],
        vec![dates[2], dates[4]],
        "uniform comparison".into(),
    )
    .unwrap();
    let terminal =
        EventSchedule::new(vec![dates[4]], vec![dates[4]], "uniform comparison".into()).unwrap();
    let dated_snowball: DatedNote = DatedSnowballNote::new(
        100.0,
        100.0,
        0.12,
        70.0,
        105.0,
        issue(),
        "USD".into(),
        "SPX".into(),
        dates.clone(),
        calls.clone(),
        terminal.clone(),
    )
    .unwrap()
    .into();
    let dated_phoenix: DatedNote = DatedPhoenixNote::new(
        100.0,
        100.0,
        3.0,
        90.0,
        70.0,
        105.0,
        issue(),
        "USD".into(),
        "SPX".into(),
        dates.clone(),
        coupons,
        calls,
        terminal,
        true,
    )
    .unwrap()
    .into();
    let old_snowball = SnowballNote {
        notional: 100.0,
        reference_spot: 100.0,
        coupon_rate: 0.12,
        knock_in_barrier: 70.0,
        knock_out_barrier: 105.0,
        observation_indices: vec![2, 4],
    };
    let old_phoenix = PhoenixNote {
        notional: 100.0,
        reference_spot: 100.0,
        coupon_per_period: 3.0,
        coupon_barrier: 90.0,
        knock_in_barrier: 70.0,
        knock_out_barrier: 105.0,
        coupon_indices: vec![1, 2, 3, 4],
        autocall_indices: vec![2, 4],
        memory: true,
    };
    for path in [
        [100.0, 70.0, 80.0, 95.0, 90.0],
        [100.0, 80.0, 105.0, 1.0, 1.0],
        [100.0, 95.0, 100.0, 95.0, 100.0],
    ] {
        let history: Vec<_> = dates[1..]
            .iter()
            .zip(&path[1..])
            .map(|(&date, &spot)| Fixing { date, spot })
            .collect();
        for (note, legacy) in [
            (
                &dated_snowball,
                old_snowball.evaluate(&path, 30.0 / 365.0, 0.04),
            ),
            (
                &dated_phoenix,
                old_phoenix.evaluate(&path, 30.0 / 365.0, 0.04),
            ),
        ] {
            let prepared = PreparedNote::new(note.clone(), initial(note)).unwrap();
            let ctx = context(&prepared, 100.0, 30.0);
            let flows = manual(&prepared, &history, 30.0);
            let pv = flows
                .iter()
                .map(|cf| cf.amount * ctx.discount_curve.value_on(cf.payment_date).unwrap())
                .sum::<f64>();
            close(pv, legacy);
        }
    }
}

#[test]
fn explicitly_declared_after_fixing_scene_survives_repeated_frozen_rolls() {
    let note = snowball();
    let state = replay_note_history(
        &note,
        &[fixing(2, 80.0)],
        after(date(3, 1)),
        &[],
        Some(fixing(3, 105.0)),
    )
    .unwrap();
    let first = roll_note_state(&note, &state, &[], after(date(3, 2)), &[], None).unwrap();
    let second = roll_note_state(&note, &first, &[], after(date(3, 3)), &[], None).unwrap();
    assert_eq!(first.provenance(), StateProvenance::FrozenRoll);
    assert_eq!(second.provenance(), StateProvenance::FrozenRoll);
    assert_eq!(first.scenario_fixings(), &[fixing(3, 105.0)]);
    assert_eq!(second.scenario_fixings(), first.scenario_fixings());
    assert_eq!(first.fixings(), state.fixings());
    assert_eq!(second.history_hash(), first.history_hash());
    assert_ne!(first.history_hash(), state.history_hash());
    assert!(second.same_day_fixing_scenario().is_none());
    assert_eq!(second.termination(), state.termination());
    assert!(advance_note_state(&note, &second, &[], after(date(3, 3)), &[], None).is_err());
}

#[test]
fn explicitly_declared_before_fixing_scene_rolls_without_inventing_a_fixing() {
    let note = snowball();
    let state = replay_note_history(
        &note,
        &[fixing(2, 80.0)],
        before(date(3, 1)),
        &[],
        Some(fixing(3, 105.0)),
    )
    .unwrap();
    let base = PreparedNote::new(note.clone(), state.clone()).unwrap();
    let rolled = roll_note_state(&note, &state, &[], after(date(3, 2)), &[], None).unwrap();
    assert_eq!(rolled.scenario_fixings(), &[fixing(3, 105.0)]);
    assert!(!rolled
        .fixings()
        .iter()
        .any(|fixing| fixing.date == date(3, 1)));
    let payoff = PreparedNote::new(note.clone(), rolled.clone()).unwrap();
    assert!(!payoff.requires_simulation());
    let ctx = context(&base, 100.0, 10.0);
    let rolled_ctx = ctx.frozen_roll(rolled.cutoff()).unwrap();
    let theta = McEngine::new(0)
        .try_calendar_theta(&ctx, &base, &rolled_ctx, &payoff, None, &[])
        .unwrap();
    close(theta.base_date_discounted_change, 0.0);
    assert!(theta.pv_change > 0.0);
    // An identical resupplied pending outcome stays hypothetical.
    let repeated = roll_note_state(
        &note,
        &state,
        &[fixing(3, 105.0)],
        after(date(3, 2)),
        &[],
        None,
    )
    .unwrap();
    assert_eq!(repeated, rolled);
    assert!(roll_note_state(
        &note,
        &state,
        &[fixing(3, 104.0)],
        after(date(3, 2)),
        &[],
        None
    )
    .is_err());
    assert!(roll_note_state(
        &note,
        &state,
        &[fixing(3, 105.0), fixing(3, 105.0)],
        after(date(3, 2)),
        &[],
        None
    )
    .is_err());
}

#[test]
fn different_declared_outcomes_bind_history_identity_and_actual_replay_is_explicit() {
    let note = snowball();
    let called = replay_note_history(
        &note,
        &[fixing(2, 80.0)],
        after(date(3, 1)),
        &[],
        Some(fixing(3, 105.0)),
    )
    .unwrap();
    let alive = replay_note_history(
        &note,
        &[fixing(2, 80.0)],
        after(date(3, 1)),
        &[],
        Some(fixing(3, 104.0)),
    )
    .unwrap();
    assert_eq!(called.fixings(), alive.fixings());
    assert_ne!(called.history_hash(), alive.history_hash());
    assert!(called.termination().is_some());
    assert!(alive.termination().is_none());
    assert!(advance_note_state(&note, &called, &[], after(date(3, 2)), &[], None).is_err());
    assert!(roll_note_state(
        &note,
        &called,
        &[fixing(3, 104.0)],
        after(date(3, 2)),
        &[],
        None
    )
    .is_err());
    let actual = replay_note_history(
        &note,
        &[fixing(2, 80.0), fixing(3, 105.0)],
        after(date(3, 1)),
        &[],
        None,
    )
    .unwrap();
    assert_eq!(actual.provenance(), StateProvenance::Actual);
    assert_ne!(actual.history_hash(), called.history_hash());
    assert!(actual.scenario_fixings().is_empty());
    assert_eq!(actual.cashflows(), called.cashflows());
}

#[test]
fn successive_declared_roll_scenarios_form_an_immutable_hypothetical_prefix() {
    let note = phoenix(true);
    let base = replay_note_history(
        &note,
        &[fixing(2, 80.0)],
        after(date(3, 1)),
        &[],
        Some(fixing(3, 90.0)),
    )
    .unwrap();
    let first = roll_note_state(&note, &base, &[], after(date(3, 2)), &[], None).unwrap();
    let payment = confirmation(
        &note,
        CashflowKind::Coupon,
        date(3, 1),
        date(4, 1),
        vec![receipt("declared-coupon-paid", date(3, 3), 6.0)],
        None,
    );
    let second = roll_note_state(
        &note,
        &first,
        &[],
        after(date(4, 1)),
        &[payment],
        Some(fixing(4, 95.0)),
    )
    .unwrap();
    let third = roll_note_state(&note, &second, &[], after(date(4, 2)), &[], None).unwrap();
    assert_eq!(
        third.scenario_fixings(),
        &[fixing(3, 90.0), fixing(4, 95.0)]
    );
    assert_eq!(third.provenance(), StateProvenance::FrozenRoll);
    assert_eq!(third.history_hash(), second.history_hash());
    assert_eq!(third.fixings(), base.fixings());
    close(third.receivables()[0].outstanding_amount, 3.0);
    assert!(third.same_day_fixing_scenario().is_none());
}
