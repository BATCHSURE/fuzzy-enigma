//! Dated teaching notes, deterministic history replay and actual settlements.
//!
//! Contractual fixing dates are independent of numerical simulation points.
//! Cashflows are undiscounted; the context engine discounts each payment once.

use std::collections::{BTreeMap, BTreeSet};

use sha2::{Digest, Sha256};

use super::{
    Cashflow, CashflowKind, CashflowPayoff, ContextPricingError, CutoffPhase, Date,
    ValuationContext, ValuationCutoff,
};

type Result<T> = std::result::Result<T, ContextPricingError>;

fn invalid(name: &'static str, reason: impl Into<String>) -> ContextPricingError {
    ContextPricingError::invalid(name, reason)
}

fn positive(name: &'static str, value: f64) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid(name, "must be positive and finite"));
    }
    Ok(())
}

fn non_negative(name: &'static str, value: f64) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(invalid(name, "must be non-negative and finite"));
    }
    Ok(())
}

fn non_empty(name: &'static str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(invalid(name, "must not be empty"));
    }
    Ok(())
}

fn hash(value: impl AsRef<[u8]>) -> String {
    format!("{:x}", Sha256::digest(value.as_ref()))
}

/// Explicit, already-adjusted contractual fixing/payment pairs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventSchedule {
    fixing_dates: Vec<Date>,
    payment_dates: Vec<Date>,
    calendar_source: String,
}

impl EventSchedule {
    pub fn new(
        fixing_dates: Vec<Date>,
        payment_dates: Vec<Date>,
        calendar_source: String,
    ) -> Result<Self> {
        non_empty("calendar_source", &calendar_source)?;
        if fixing_dates.is_empty() || fixing_dates.len() != payment_dates.len() {
            return Err(invalid(
                "schedule",
                "fixing/payment dates must be non-empty and have matching lengths",
            ));
        }
        if fixing_dates.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(invalid(
                "schedule",
                "fixing dates must be strictly increasing",
            ));
        }
        for (&fixing, &payment) in fixing_dates.iter().zip(&payment_dates) {
            if payment < fixing {
                return Err(invalid("schedule", "payment date is before its fixing"));
            }
        }
        Ok(Self {
            fixing_dates,
            payment_dates,
            calendar_source,
        })
    }

    pub fn fixing_dates(&self) -> &[Date] {
        &self.fixing_dates
    }
    pub fn payment_dates(&self) -> &[Date] {
        &self.payment_dates
    }
    pub fn calendar_source(&self) -> &str {
        &self.calendar_source
    }

    fn payment_on(&self, date: Date) -> Option<Date> {
        self.fixing_dates
            .binary_search(&date)
            .ok()
            .map(|i| self.payment_dates[i])
    }
}

/// Immutable Snowball terms. Coupon accrual uses ACT/365F from issuance.
#[derive(Clone, Debug, PartialEq)]
pub struct DatedSnowballNote {
    notional: f64,
    reference_spot: f64,
    coupon_rate: f64,
    knock_in_barrier: f64,
    knock_out_barrier: f64,
    issue_date: Date,
    currency: String,
    asset: String,
    knock_in_dates: Vec<Date>,
    autocall_schedule: EventSchedule,
    maturity_schedule: EventSchedule,
    contract_hash: String,
}

impl DatedSnowballNote {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        notional: f64,
        reference_spot: f64,
        coupon_rate: f64,
        knock_in_barrier: f64,
        knock_out_barrier: f64,
        issue_date: Date,
        currency: String,
        asset: String,
        knock_in_dates: Vec<Date>,
        autocall_schedule: EventSchedule,
        maturity_schedule: EventSchedule,
    ) -> Result<Self> {
        validate_common(
            notional,
            reference_spot,
            knock_in_barrier,
            knock_out_barrier,
            issue_date,
            &currency,
            &asset,
            &knock_in_dates,
            &autocall_schedule,
            &maturity_schedule,
        )?;
        non_negative("coupon_rate", coupon_rate)?;
        let contract_hash = hash(format!(
            "dated-snowball-v1|{notional:?}|{reference_spot:?}|{coupon_rate:?}|{knock_in_barrier:?}|{knock_out_barrier:?}|{issue_date}|{currency:?}|{asset:?}|{knock_in_dates:?}|{autocall_schedule:?}|{maturity_schedule:?}"
        ));
        Ok(Self {
            notional,
            reference_spot,
            coupon_rate,
            knock_in_barrier,
            knock_out_barrier,
            issue_date,
            currency,
            asset,
            knock_in_dates,
            autocall_schedule,
            maturity_schedule,
            contract_hash,
        })
    }

    pub fn notional(&self) -> f64 {
        self.notional
    }
    pub fn reference_spot(&self) -> f64 {
        self.reference_spot
    }
    pub fn coupon_rate(&self) -> f64 {
        self.coupon_rate
    }
    pub fn knock_in_barrier(&self) -> f64 {
        self.knock_in_barrier
    }
    pub fn knock_out_barrier(&self) -> f64 {
        self.knock_out_barrier
    }
    pub fn issue_date(&self) -> Date {
        self.issue_date
    }
    pub fn currency(&self) -> &str {
        &self.currency
    }
    pub fn asset(&self) -> &str {
        &self.asset
    }
    pub fn knock_in_dates(&self) -> &[Date] {
        &self.knock_in_dates
    }
    pub fn autocall_schedule(&self) -> &EventSchedule {
        &self.autocall_schedule
    }
    pub fn maturity_schedule(&self) -> &EventSchedule {
        &self.maturity_schedule
    }
    pub fn contract_hash(&self) -> &str {
        &self.contract_hash
    }
    pub fn cashflow_id(&self, kind: CashflowKind, date: Date) -> Result<String> {
        DatedNote::Snowball(self.clone()).cashflow_id(kind, date)
    }
}

/// Immutable Phoenix terms; coupons are cash amounts per observation period.
#[derive(Clone, Debug, PartialEq)]
pub struct DatedPhoenixNote {
    notional: f64,
    reference_spot: f64,
    coupon_per_period: f64,
    coupon_barrier: f64,
    knock_in_barrier: f64,
    knock_out_barrier: f64,
    issue_date: Date,
    currency: String,
    asset: String,
    knock_in_dates: Vec<Date>,
    coupon_schedule: EventSchedule,
    autocall_schedule: EventSchedule,
    maturity_schedule: EventSchedule,
    memory: bool,
    contract_hash: String,
}

impl DatedPhoenixNote {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        notional: f64,
        reference_spot: f64,
        coupon_per_period: f64,
        coupon_barrier: f64,
        knock_in_barrier: f64,
        knock_out_barrier: f64,
        issue_date: Date,
        currency: String,
        asset: String,
        knock_in_dates: Vec<Date>,
        coupon_schedule: EventSchedule,
        autocall_schedule: EventSchedule,
        maturity_schedule: EventSchedule,
        memory: bool,
    ) -> Result<Self> {
        validate_common(
            notional,
            reference_spot,
            knock_in_barrier,
            knock_out_barrier,
            issue_date,
            &currency,
            &asset,
            &knock_in_dates,
            &autocall_schedule,
            &maturity_schedule,
        )?;
        non_negative("coupon_per_period", coupon_per_period)?;
        positive("coupon_barrier", coupon_barrier)?;
        validate_future_schedule(
            &coupon_schedule,
            issue_date,
            maturity_schedule.fixing_dates[0],
        )?;
        let contract_hash = hash(format!(
            "dated-phoenix-v1|{notional:?}|{reference_spot:?}|{coupon_per_period:?}|{coupon_barrier:?}|{knock_in_barrier:?}|{knock_out_barrier:?}|{issue_date}|{currency:?}|{asset:?}|{knock_in_dates:?}|{coupon_schedule:?}|{autocall_schedule:?}|{maturity_schedule:?}|{memory}"
        ));
        Ok(Self {
            notional,
            reference_spot,
            coupon_per_period,
            coupon_barrier,
            knock_in_barrier,
            knock_out_barrier,
            issue_date,
            currency,
            asset,
            knock_in_dates,
            coupon_schedule,
            autocall_schedule,
            maturity_schedule,
            memory,
            contract_hash,
        })
    }

    pub fn notional(&self) -> f64 {
        self.notional
    }
    pub fn reference_spot(&self) -> f64 {
        self.reference_spot
    }
    pub fn coupon_per_period(&self) -> f64 {
        self.coupon_per_period
    }
    pub fn coupon_barrier(&self) -> f64 {
        self.coupon_barrier
    }
    pub fn knock_in_barrier(&self) -> f64 {
        self.knock_in_barrier
    }
    pub fn knock_out_barrier(&self) -> f64 {
        self.knock_out_barrier
    }
    pub fn issue_date(&self) -> Date {
        self.issue_date
    }
    pub fn currency(&self) -> &str {
        &self.currency
    }
    pub fn asset(&self) -> &str {
        &self.asset
    }
    pub fn knock_in_dates(&self) -> &[Date] {
        &self.knock_in_dates
    }
    pub fn coupon_schedule(&self) -> &EventSchedule {
        &self.coupon_schedule
    }
    pub fn autocall_schedule(&self) -> &EventSchedule {
        &self.autocall_schedule
    }
    pub fn maturity_schedule(&self) -> &EventSchedule {
        &self.maturity_schedule
    }
    pub fn memory(&self) -> bool {
        self.memory
    }
    pub fn contract_hash(&self) -> &str {
        &self.contract_hash
    }
    pub fn cashflow_id(&self, kind: CashflowKind, date: Date) -> Result<String> {
        DatedNote::Phoenix(self.clone()).cashflow_id(kind, date)
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_common(
    notional: f64,
    reference: f64,
    ki: f64,
    ko: f64,
    issue: Date,
    currency: &str,
    asset: &str,
    ki_dates: &[Date],
    autocall: &EventSchedule,
    maturity: &EventSchedule,
) -> Result<()> {
    positive("notional", notional)?;
    positive("reference_spot", reference)?;
    positive("knock_in_barrier", ki)?;
    positive("knock_out_barrier", ko)?;
    non_empty("currency", currency)?;
    non_empty("asset", asset)?;
    if maturity.fixing_dates.len() != 1 || maturity.fixing_dates[0] <= issue {
        return Err(invalid(
            "maturity_schedule",
            "requires one fixing strictly after issue",
        ));
    }
    let maturity_date = maturity.fixing_dates[0];
    if ki_dates.first() != Some(&issue)
        || ki_dates.windows(2).any(|p| p[0] >= p[1])
        || ki_dates.iter().any(|date| *date > maturity_date)
    {
        return Err(invalid(
            "knock_in_dates",
            "must include issue first, increase strictly, and end by maturity",
        ));
    }
    validate_future_schedule(autocall, issue, maturity_date)
}

fn validate_future_schedule(schedule: &EventSchedule, issue: Date, maturity: Date) -> Result<()> {
    if schedule
        .fixing_dates
        .iter()
        .any(|date| *date <= issue || *date > maturity)
    {
        return Err(invalid(
            "schedule",
            "observations must be after issue and no later than maturity",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub enum DatedNote {
    Snowball(DatedSnowballNote),
    Phoenix(DatedPhoenixNote),
}

macro_rules! note_getter {
    ($name:ident, $return_type:ty) => {
        pub fn $name(&self) -> $return_type {
            match self {
                Self::Snowball(n) => n.$name(),
                Self::Phoenix(n) => n.$name(),
            }
        }
    };
}

impl DatedNote {
    note_getter!(notional, f64);
    note_getter!(reference_spot, f64);
    note_getter!(knock_in_barrier, f64);
    note_getter!(knock_out_barrier, f64);
    note_getter!(issue_date, Date);
    note_getter!(currency, &str);
    note_getter!(asset, &str);
    note_getter!(knock_in_dates, &[Date]);
    note_getter!(autocall_schedule, &EventSchedule);
    note_getter!(maturity_schedule, &EventSchedule);
    note_getter!(contract_hash, &str);

    pub fn maturity_date(&self) -> Date {
        self.maturity_schedule().fixing_dates[0]
    }

    pub fn observation_dates(&self) -> Vec<Date> {
        let mut dates = BTreeSet::from_iter(self.knock_in_dates().iter().copied());
        dates.extend(self.autocall_schedule().fixing_dates.iter().copied());
        dates.insert(self.maturity_date());
        if let Self::Phoenix(note) = self {
            dates.extend(note.coupon_schedule.fixing_dates.iter().copied());
        }
        dates.into_iter().collect()
    }

    /// Discover stable ledger IDs before supplying actual confirmations.
    pub fn cashflow_id(&self, kind: CashflowKind, date: Date) -> Result<String> {
        let principal_date =
            self.autocall_schedule().fixing_dates.contains(&date) || date == self.maturity_date();
        let coupon_date = match self {
            Self::Snowball(_) => principal_date,
            Self::Phoenix(n) => n.coupon_schedule.fixing_dates.contains(&date),
        };
        let label = match kind {
            CashflowKind::Principal if principal_date => "principal",
            CashflowKind::Coupon if coupon_date => "coupon",
            _ => {
                return Err(invalid(
                    "cashflow_id",
                    "kind/date is not a contractual cashflow event",
                ))
            }
        };
        Ok(format!("{}:{label}:{date}", self.contract_hash()))
    }
}

impl From<DatedSnowballNote> for DatedNote {
    fn from(note: DatedSnowballNote) -> Self {
        Self::Snowball(note)
    }
}
impl From<DatedPhoenixNote> for DatedNote {
    fn from(note: DatedPhoenixNote) -> Self {
        Self::Phoenix(note)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Fixing {
    pub date: Date,
    pub spot: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ActualPayment {
    pub payment_id: String,
    pub payment_date: Date,
    pub amount: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SettlementConfirmation {
    pub cashflow_id: String,
    pub as_of: Date,
    pub payments: Vec<ActualPayment>,
    pub expected_payment_date: Option<Date>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Receivable {
    /// Original contractual cashflow; its payment date is never overwritten.
    pub cashflow: Cashflow,
    /// Coupon-period IDs included in a Phoenix memory catch-up.
    pub component_ids: Vec<String>,
    pub payments: Vec<ActualPayment>,
    pub paid_amount: f64,
    pub outstanding_amount: f64,
    pub expected_payment_date: Option<Date>,
    pub confirmation_as_of: Option<Date>,
}

impl Receivable {
    pub fn settlement_status(&self) -> &'static str {
        if self.outstanding_amount == 0.0 {
            "paid"
        } else if self.paid_amount > 0.0 {
            "partial"
        } else {
            "pending"
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Termination {
    Redeemed { fixing_date: Date },
    Matured { fixing_date: Date },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateProvenance {
    Actual,
    FixingScenario,
    FrozenRoll,
}

impl StateProvenance {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Actual => "actual",
            Self::FixingScenario => "fixing_scenario",
            Self::FrozenRoll => "frozen_roll",
        }
    }
}

/// Immutable, contract-bound result of chronological history replay.
#[derive(Clone, Debug, PartialEq)]
pub struct NoteState {
    cutoff: ValuationCutoff,
    first_knock_in_date: Option<Date>,
    termination: Option<Termination>,
    memory_coupon_ids: Vec<String>,
    expired_memory_coupon_ids: Vec<String>,
    cashflows: Vec<Receivable>,
    receivables: Vec<Receivable>,
    fixings: Vec<Fixing>,
    scenario_fixings: Vec<Fixing>,
    settlements: Vec<SettlementConfirmation>,
    contract_hash: String,
    history_hash: String,
    settlement_hash: String,
    provenance: StateProvenance,
    same_day_fixing_scenario: Option<Fixing>,
}

impl NoteState {
    pub fn cutoff(&self) -> ValuationCutoff {
        self.cutoff
    }
    pub fn knocked_in(&self) -> bool {
        self.first_knock_in_date.is_some()
    }
    pub fn first_knock_in_date(&self) -> Option<Date> {
        self.first_knock_in_date
    }
    pub fn termination(&self) -> Option<&Termination> {
        self.termination.as_ref()
    }
    pub fn memory_coupon_ids(&self) -> &[String] {
        &self.memory_coupon_ids
    }
    pub fn expired_memory_coupon_ids(&self) -> &[String] {
        &self.expired_memory_coupon_ids
    }
    pub fn cashflows(&self) -> &[Receivable] {
        &self.cashflows
    }
    pub fn receivables(&self) -> &[Receivable] {
        &self.receivables
    }
    pub fn fixings(&self) -> &[Fixing] {
        &self.fixings
    }
    /// Explicit hypothetical outcomes retained by a frozen calendar roll.
    pub fn scenario_fixings(&self) -> &[Fixing] {
        &self.scenario_fixings
    }
    pub fn settlements(&self) -> &[SettlementConfirmation] {
        &self.settlements
    }
    pub fn contract_hash(&self) -> &str {
        &self.contract_hash
    }
    pub fn history_hash(&self) -> &str {
        &self.history_hash
    }
    pub fn settlement_hash(&self) -> &str {
        &self.settlement_hash
    }
    pub fn provenance(&self) -> StateProvenance {
        self.provenance
    }
    pub fn same_day_fixing_scenario(&self) -> Option<&Fixing> {
        self.same_day_fixing_scenario.as_ref()
    }

    fn validate_bound(&self, note: &DatedNote) -> Result<()> {
        if self.contract_hash != note.contract_hash()
            || self.history_hash
                != history_identity(
                    &self.fixings,
                    &self.scenario_fixings,
                    self.same_day_fixing_scenario.as_ref(),
                    self.provenance,
                )
            || self.settlement_hash != hash(format!("settlements-v1|{:?}", self.settlements))
        {
            return Err(invalid(
                "state",
                "contract/history/settlement hash does not match immutable state",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
struct EconomicState {
    first_knock_in_date: Option<Date>,
    termination: Option<Termination>,
    memory: Vec<String>,
    expired_memory: Vec<String>,
    cashflows: Vec<Receivable>,
}

fn coupon_period_id(note: &DatedNote, date: Date) -> String {
    format!("{}:coupon_period:{date}", note.contract_hash())
}

fn earn(
    note: &DatedNote,
    state: &mut EconomicState,
    kind: CashflowKind,
    date: Date,
    payment: Date,
    amount: f64,
    component_ids: Vec<String>,
) -> Result<()> {
    non_negative("cashflow_amount", amount)?;
    if amount == 0.0 {
        return Ok(());
    }
    state.cashflows.push(Receivable {
        cashflow: Cashflow {
            id: note.cashflow_id(kind, date)?,
            fixing_date: Some(date),
            payment_date: payment,
            amount,
            kind,
            currency: note.currency().to_owned(),
        },
        component_ids,
        payments: Vec::new(),
        paid_amount: 0.0,
        outstanding_amount: amount,
        expected_payment_date: Some(payment),
        confirmation_as_of: None,
    });
    Ok(())
}

fn process_observation(
    note: &DatedNote,
    date: Date,
    spot: f64,
    state: &mut EconomicState,
) -> Result<()> {
    positive("fixing", spot)?;
    if state.termination.is_some() {
        return Ok(());
    }
    if note.knock_in_dates().binary_search(&date).is_ok()
        && spot <= note.knock_in_barrier()
        && state.first_knock_in_date.is_none()
    {
        state.first_knock_in_date = Some(date);
    }
    if let DatedNote::Phoenix(n) = note {
        if let Some(payment) = n.coupon_schedule.payment_on(date) {
            let period_id = coupon_period_id(note, date);
            if spot >= n.coupon_barrier {
                let mut components = std::mem::take(&mut state.memory);
                components.push(period_id);
                earn(
                    note,
                    state,
                    CashflowKind::Coupon,
                    date,
                    payment,
                    n.coupon_per_period * components.len() as f64,
                    components,
                )?;
            } else if n.memory {
                state.memory.push(period_id);
            }
        }
    }
    if let Some(payment) = note.autocall_schedule().payment_on(date) {
        if spot >= note.knock_out_barrier() {
            earn(
                note,
                state,
                CashflowKind::Principal,
                date,
                payment,
                note.notional(),
                Vec::new(),
            )?;
            if let DatedNote::Snowball(n) = note {
                let accrual = (date - n.issue_date).num_days() as f64 / 365.0;
                earn(
                    note,
                    state,
                    CashflowKind::Coupon,
                    date,
                    payment,
                    n.notional * n.coupon_rate * accrual,
                    Vec::new(),
                )?;
            }
            state.expired_memory.append(&mut state.memory);
            state.termination = Some(Termination::Redeemed { fixing_date: date });
            return Ok(());
        }
    }
    if date == note.maturity_date() {
        let payment = note.maturity_schedule().payment_dates[0];
        let principal = if state.first_knock_in_date.is_some() {
            note.notional() * (spot / note.reference_spot()).min(1.0)
        } else {
            note.notional()
        };
        earn(
            note,
            state,
            CashflowKind::Principal,
            date,
            payment,
            principal,
            Vec::new(),
        )?;
        if let DatedNote::Snowball(n) = note {
            if state.first_knock_in_date.is_none() {
                let accrual = (date - n.issue_date).num_days() as f64 / 365.0;
                earn(
                    note,
                    state,
                    CashflowKind::Coupon,
                    date,
                    payment,
                    n.notional * n.coupon_rate * accrual,
                    Vec::new(),
                )?;
            }
        }
        state.expired_memory.append(&mut state.memory);
        state.termination = Some(Termination::Matured { fixing_date: date });
    }
    Ok(())
}

fn hash_fixings(fixings: &[Fixing]) -> String {
    let mut canonical = String::from("fixings-v1\n");
    for fixing in fixings {
        canonical.push_str(&format!("{}|{:016x}\n", fixing.date, fixing.spot.to_bits()));
    }
    hash(canonical)
}

fn history_identity(
    actual: &[Fixing],
    retained: &[Fixing],
    current: Option<&Fixing>,
    provenance: StateProvenance,
) -> String {
    let mut scenarios = retained.to_vec();
    scenarios.extend(current.cloned());
    scenarios.sort_by_key(|fixing| fixing.date);
    hash(format!(
        "note-history-v2|{}|{}|{}",
        hash_fixings(actual),
        hash_fixings(&scenarios),
        provenance.as_str()
    ))
}

fn is_processed(date: Date, cutoff: ValuationCutoff) -> bool {
    date < cutoff.as_of || (date == cutoff.as_of && cutoff.phase == CutoffPhase::AfterFixing)
}

fn canonical_fixings(
    note: &DatedNote,
    input: &[Fixing],
    cutoff: ValuationCutoff,
) -> Result<Vec<Fixing>> {
    if cutoff.as_of < note.issue_date() {
        return Err(invalid("cutoff", "valuation date is before issue"));
    }
    if input.windows(2).any(|p| p[0].date >= p[1].date) {
        return Err(invalid(
            "fixings",
            "history dates must be strictly increasing; duplicates are forbidden",
        ));
    }
    for fixing in input {
        positive("fixing", fixing.spot)?;
        if fixing.date < note.issue_date()
            || (!is_processed(fixing.date, cutoff) && fixing.date != note.issue_date())
        {
            return Err(invalid(
                "fixings",
                format!("fixing {} is outside processed history cutoff", fixing.date),
            ));
        }
        if fixing.date == note.issue_date()
            && fixing.spot.to_bits() != note.reference_spot().to_bits()
        {
            return Err(invalid(
                "reference_spot",
                "issue-date fixing conflicts with original contract reference",
            ));
        }
    }
    let mut fixings = input.to_vec();
    if fixings.first().map(|f| f.date) != Some(note.issue_date()) {
        fixings.insert(
            0,
            Fixing {
                date: note.issue_date(),
                spot: note.reference_spot(),
            },
        );
    }
    Ok(fixings)
}

fn apply_settlements(
    ledger: &mut [Receivable],
    input: &[SettlementConfirmation],
    as_of: Date,
    frozen_roll: bool,
) -> Result<Vec<SettlementConfirmation>> {
    let mut records = BTreeMap::new();
    let mut payment_ids = BTreeSet::new();
    for confirmation in input {
        non_empty("cashflow_id", &confirmation.cashflow_id)?;
        if confirmation.as_of > as_of {
            return Err(invalid(
                "settlements",
                "confirmation is after valuation date",
            ));
        }
        if records
            .insert(confirmation.cashflow_id.as_str(), confirmation)
            .is_some()
        {
            return Err(invalid("settlements", "duplicate cashflow confirmation"));
        }
        if confirmation
            .payments
            .windows(2)
            .any(|p| p[0].payment_date > p[1].payment_date)
        {
            return Err(invalid(
                "settlements",
                "actual receipts must be chronological",
            ));
        }
        for receipt in &confirmation.payments {
            non_empty("payment_id", &receipt.payment_id)?;
            positive("paid_amount", receipt.amount)?;
            if !payment_ids.insert(receipt.payment_id.as_str()) {
                return Err(invalid("settlements", "duplicate actual payment ID"));
            }
            if receipt.payment_date > confirmation.as_of {
                return Err(invalid(
                    "settlements",
                    "receipt date is after confirmation as-of",
                ));
            }
        }
    }
    for entry in ledger.iter_mut() {
        let confirmation = records.remove(entry.cashflow.id.as_str());
        if let Some(c) = confirmation {
            if c.payments
                .iter()
                .any(|p| Some(p.payment_date) < entry.cashflow.fixing_date)
            {
                return Err(invalid(
                    "settlements",
                    "actual receipt predates the earning fixing",
                ));
            }
            let paid = c.payments.iter().map(|p| p.amount).sum::<f64>();
            let tolerance = 1e-10_f64.max(entry.cashflow.amount.abs() * 1e-12);
            if !paid.is_finite() || paid > entry.cashflow.amount + tolerance {
                return Err(invalid(
                    "settlements",
                    "receipts exceed the earned cashflow",
                ));
            }
            let outstanding = (entry.cashflow.amount - paid).max(0.0);
            entry.paid_amount = paid;
            entry.outstanding_amount = if outstanding <= tolerance {
                0.0
            } else {
                outstanding
            };
            entry.payments = c.payments.clone();
            entry.confirmation_as_of = Some(c.as_of);
            if entry.outstanding_amount == 0.0 {
                if c.expected_payment_date.is_some() {
                    return Err(invalid(
                        "expected_payment_date",
                        "fully paid cashflow must not have an expected payment date",
                    ));
                }
                entry.expected_payment_date = None;
                continue;
            }
            if !frozen_roll && c.as_of != as_of && entry.cashflow.payment_date <= as_of {
                return Err(invalid(
                    "settlements",
                    format!("stale pending confirmation for {}", entry.cashflow.id),
                ));
            }
            entry.expected_payment_date = c.expected_payment_date.or_else(|| {
                (entry.cashflow.payment_date > as_of).then_some(entry.cashflow.payment_date)
            });
        } else if entry.cashflow.payment_date <= as_of {
            return Err(invalid(
                "settlements",
                format!(
                    "missing actual settlement confirmation for {}",
                    entry.cashflow.id
                ),
            ));
        }
        if entry.outstanding_amount > 0.0 {
            let expected = entry.expected_payment_date.ok_or_else(|| {
                invalid(
                    "expected_payment_date",
                    format!(
                        "overdue outstanding {} requires an expected payment date",
                        entry.cashflow.id
                    ),
                )
            })?;
            if expected < as_of {
                return Err(invalid(
                    "expected_payment_date",
                    format!(
                        "expected payment for {} must be on or after as-of",
                        entry.cashflow.id
                    ),
                ));
            }
        }
    }
    if let Some(id) = records.keys().next() {
        return Err(invalid(
            "settlements",
            format!("confirmation refers to unearned/unknown cashflow {id}"),
        ));
    }
    let mut canonical = input.to_vec();
    canonical.sort_by(|a, b| a.cashflow_id.cmp(&b.cashflow_id));
    Ok(canonical)
}

#[allow(clippy::too_many_arguments)]
fn replay(
    note: &DatedNote,
    input: &[Fixing],
    cutoff: ValuationCutoff,
    settlements: &[SettlementConfirmation],
    scenario: Option<Fixing>,
    frozen_roll: bool,
    retained_scenarios: &[Fixing],
) -> Result<NoteState> {
    let fixings = canonical_fixings(note, input, cutoff)?;
    if retained_scenarios
        .windows(2)
        .any(|pair| pair[0].date >= pair[1].date)
    {
        return Err(invalid(
            "scenario_fixings",
            "retained scenario dates must increase strictly",
        ));
    }
    for fixing in retained_scenarios {
        positive("scenario_fixings", fixing.spot)?;
        if !frozen_roll
            || !is_processed(fixing.date, cutoff)
            || fixing.date <= note.issue_date()
            || !note.observation_dates().contains(&fixing.date)
            || fixings.iter().any(|actual| actual.date == fixing.date)
        {
            return Err(invalid("scenario_fixings", "retained hypothetical fixing must belong to the processed frozen-roll prefix and cannot replace actual history"));
        }
    }
    if let Some(fixing) = &scenario {
        positive("same_day_fixing_scenario", fixing.spot)?;
        if fixing.date != cutoff.as_of
            || fixing.date == note.issue_date()
            || !note.observation_dates().contains(&fixing.date)
        {
            return Err(invalid(
                "same_day_fixing_scenario",
                "must be a valuation-day contractual observation after issue",
            ));
        }
        if fixings.iter().any(|f| f.date == fixing.date) {
            return Err(invalid(
                "same_day_fixing_scenario",
                "cannot replace an actual historical fixing",
            ));
        }
        if retained_scenarios.iter().any(|f| f.date == fixing.date) {
            return Err(invalid(
                "same_day_fixing_scenario",
                "cannot override a processed scenario fixing",
            ));
        }
    }
    let historical = BTreeMap::from_iter(fixings.iter().map(|f| (f.date, f.spot)));
    let mut economic = EconomicState::default();
    for date in note.observation_dates() {
        if economic.termination.is_some() {
            break;
        }
        if date != note.issue_date() && !is_processed(date, cutoff) {
            break;
        }
        let spot = historical
            .get(&date)
            .copied()
            .or_else(|| {
                retained_scenarios
                    .iter()
                    .find(|fixing| fixing.date == date)
                    .map(|fixing| fixing.spot)
            })
            .or_else(|| scenario.as_ref().filter(|f| f.date == date).map(|f| f.spot))
            .ok_or_else(|| {
                invalid(
                    "fixings",
                    format!("missing required historical fixing for {date}"),
                )
            })?;
        process_observation(note, date, spot, &mut economic)?;
    }
    let settlements = apply_settlements(
        &mut economic.cashflows,
        settlements,
        cutoff.as_of,
        frozen_roll,
    )?;
    let receivables = economic
        .cashflows
        .iter()
        .filter(|entry| entry.outstanding_amount > 0.0)
        .cloned()
        .collect();
    let provenance = if frozen_roll {
        StateProvenance::FrozenRoll
    } else if scenario.is_some() {
        StateProvenance::FixingScenario
    } else {
        StateProvenance::Actual
    };
    let history_hash =
        history_identity(&fixings, retained_scenarios, scenario.as_ref(), provenance);
    let settlement_hash = hash(format!("settlements-v1|{settlements:?}"));
    Ok(NoteState {
        cutoff,
        first_knock_in_date: economic.first_knock_in_date,
        termination: economic.termination,
        memory_coupon_ids: economic.memory,
        expired_memory_coupon_ids: economic.expired_memory,
        cashflows: economic.cashflows,
        receivables,
        fixings,
        scenario_fixings: retained_scenarios.to_vec(),
        settlements,
        contract_hash: note.contract_hash().to_owned(),
        history_hash,
        settlement_hash,
        provenance,
        same_day_fixing_scenario: scenario,
    })
}

pub fn replay_note_history(
    note: &DatedNote,
    fixings: &[Fixing],
    cutoff: ValuationCutoff,
    settlements: &[SettlementConfirmation],
    same_day_fixing_scenario: Option<Fixing>,
) -> Result<NoteState> {
    replay(
        note,
        fixings,
        cutoff,
        settlements,
        same_day_fixing_scenario,
        false,
        &[],
    )
}

fn advance(
    note: &DatedNote,
    state: &NoteState,
    new_fixings: &[Fixing],
    cutoff: ValuationCutoff,
    updates: &[SettlementConfirmation],
    scenario: Option<Fixing>,
    frozen_roll: bool,
) -> Result<NoteState> {
    state.validate_bound(note)?;
    if !frozen_roll && state.provenance != StateProvenance::Actual {
        return Err(invalid("state", "actual advancement of hypothetical state requires a full replay with actual fixings and settlements"));
    }
    if cutoff.as_of < state.cutoff.as_of
        || (cutoff.as_of == state.cutoff.as_of
            && state.cutoff.phase == CutoffPhase::AfterFixing
            && cutoff.phase == CutoffPhase::BeforeFixing)
    {
        return Err(invalid("cutoff", "cannot move state cutoff backwards"));
    }
    if new_fixings
        .iter()
        .any(|f| is_processed(f.date, state.cutoff) || f.date == note.issue_date())
    {
        return Err(invalid(
            "fixings",
            "new history must extend the immutable processed prefix",
        ));
    }
    if new_fixings
        .windows(2)
        .any(|pair| pair[0].date >= pair[1].date)
    {
        return Err(invalid("fixings", "new history dates must increase strictly; duplicate scenario outcomes are also forbidden"));
    }
    let mut retained = state.scenario_fixings.clone();
    let mut current_scenario = scenario;
    if frozen_roll {
        if let Some(previous) = &state.same_day_fixing_scenario {
            if is_processed(previous.date, cutoff) {
                if current_scenario
                    .as_ref()
                    .is_some_and(|fixing| fixing.date == previous.date)
                {
                    return Err(invalid(
                        "same_day_fixing_scenario",
                        "cannot override a processed scenario fixing",
                    ));
                }
                retained.push(previous.clone());
            } else if current_scenario.is_none() {
                current_scenario = Some(previous.clone());
            }
        }
    }
    let mut fixings = state.fixings.clone();
    for fixing in new_fixings {
        if let Some(declared) = retained
            .iter()
            .find(|declared| declared.date == fixing.date)
        {
            if declared.spot.to_bits() != fixing.spot.to_bits() {
                return Err(invalid("fixings", "cannot override an immutable scenario outcome; replay actual history separately"));
            }
            // An identical explicitly resupplied outcome remains hypothetical.
        } else {
            fixings.push(fixing.clone());
        }
    }
    let mut confirmations = BTreeMap::from_iter(
        state
            .settlements
            .iter()
            .cloned()
            .map(|c| (c.cashflow_id.clone(), c)),
    );
    let mut seen_updates = BTreeSet::new();
    for update in updates {
        if !seen_updates.insert(update.cashflow_id.as_str()) {
            return Err(invalid("settlements", "duplicate settlement update"));
        }
        if let Some(old) = confirmations.get(&update.cashflow_id) {
            if update.as_of < old.as_of || !update.payments.starts_with(&old.payments) {
                return Err(invalid(
                    "settlements",
                    "actual receipts and confirmations must extend their immutable prefix",
                ));
            }
        }
        if let Some(old) = state
            .cashflows
            .iter()
            .find(|entry| entry.cashflow.id == update.cashflow_id)
        {
            if !update.payments.starts_with(&old.payments)
                || update.payments[old.payments.len()..]
                    .iter()
                    .any(|payment| payment.payment_date <= state.cutoff.as_of)
            {
                return Err(invalid("settlements", "new actual receipts must be after the prior EOD cutoff; historical corrections require full replay"));
            }
        }
        confirmations.insert(update.cashflow_id.clone(), update.clone());
    }
    let confirmations: Vec<_> = confirmations.into_values().collect();
    if frozen_roll {
        for entry in &state.receivables {
            let expected = entry.expected_payment_date.expect("validated pending date");
            if expected <= cutoff.as_of
                && cutoff.as_of > state.cutoff.as_of
                && !seen_updates.contains(entry.cashflow.id.as_str())
            {
                return Err(invalid("settlements", format!("calendar roll crosses payment {} and requires an explicit settlement update", entry.cashflow.id)));
            }
        }
    }
    replay(
        note,
        &fixings,
        cutoff,
        &confirmations,
        current_scenario,
        frozen_roll,
        &retained,
    )
}

pub fn advance_note_state(
    note: &DatedNote,
    state: &NoteState,
    new_fixings: &[Fixing],
    cutoff: ValuationCutoff,
    settlement_updates: &[SettlementConfirmation],
    same_day_fixing_scenario: Option<Fixing>,
) -> Result<NoteState> {
    advance(
        note,
        state,
        new_fixings,
        cutoff,
        settlement_updates,
        same_day_fixing_scenario,
        false,
    )
}

/// Hypothetical frozen-market roll; crossed payments require explicit updates.
pub fn roll_note_state(
    note: &DatedNote,
    state: &NoteState,
    new_fixings: &[Fixing],
    cutoff: ValuationCutoff,
    settlement_updates: &[SettlementConfirmation],
    same_day_fixing_scenario: Option<Fixing>,
) -> Result<NoteState> {
    advance(
        note,
        state,
        new_fixings,
        cutoff,
        settlement_updates,
        same_day_fixing_scenario,
        true,
    )
}

/// Contract/state pair prepared for curve-aware valuation, without discounting.
#[derive(Clone, Debug)]
pub struct PreparedNote {
    note: DatedNote,
    state: NoteState,
    known: Vec<Cashflow>,
    projected: Option<EconomicState>,
}

impl PreparedNote {
    pub fn new(note: DatedNote, state: NoteState) -> Result<Self> {
        state.validate_bound(&note)?;
        let mut known: Vec<Cashflow> = state
            .receivables
            .iter()
            .map(|entry| {
                let mut cf = entry.cashflow.clone();
                cf.amount = entry.outstanding_amount;
                cf.payment_date = entry
                    .expected_payment_date
                    .expect("validated pending payment date");
                cf
            })
            .collect();
        let mut projected = None;
        if state.termination.is_none() && state.cutoff.phase == CutoffPhase::BeforeFixing {
            if let Some(scenario) = &state.same_day_fixing_scenario {
                let mut economic = EconomicState {
                    first_knock_in_date: state.first_knock_in_date,
                    memory: state.memory_coupon_ids.clone(),
                    ..EconomicState::default()
                };
                process_observation(&note, scenario.date, scenario.spot, &mut economic)?;
                known.extend(
                    economic
                        .cashflows
                        .iter()
                        .map(|entry| entry.cashflow.clone()),
                );
                projected = Some(economic);
            }
        }
        Ok(Self {
            note,
            state,
            known,
            projected,
        })
    }
    pub fn note(&self) -> &DatedNote {
        &self.note
    }
    pub fn state(&self) -> &NoteState {
        &self.state
    }
    pub fn future_fixing_dates(&self) -> Vec<Date> {
        if !self.requires_simulation() {
            return Vec::new();
        }
        self.note
            .observation_dates()
            .into_iter()
            .filter(|date| {
                *date != self.note.issue_date()
                    && !is_processed(*date, self.state.cutoff)
                    && !(self.projected.is_some() && *date == self.state.cutoff.as_of)
            })
            .collect()
    }

    /// Maximum remaining contractual payment date, including known obligations.
    pub fn payment_dates(&self) -> Vec<Date> {
        let mut dates: BTreeSet<_> = self.known.iter().map(|cf| cf.payment_date).collect();
        if self.requires_simulation() {
            for (&fixing, &payment) in self
                .note
                .autocall_schedule()
                .fixing_dates
                .iter()
                .zip(&self.note.autocall_schedule().payment_dates)
            {
                if !is_processed(fixing, self.state.cutoff) {
                    dates.insert(payment);
                }
            }
            dates.extend(self.note.maturity_schedule().payment_dates.iter().copied());
            if let DatedNote::Phoenix(n) = &self.note {
                for (&fixing, &payment) in n
                    .coupon_schedule
                    .fixing_dates
                    .iter()
                    .zip(&n.coupon_schedule.payment_dates)
                {
                    if !is_processed(fixing, self.state.cutoff) {
                        dates.insert(payment);
                    }
                }
            }
        }
        dates.into_iter().collect()
    }
}

impl CashflowPayoff for PreparedNote {
    fn validate(&self, context: &ValuationContext) -> Result<()> {
        self.state.validate_bound(&self.note)?;
        if context.cutoff != self.state.cutoff {
            return Err(invalid(
                "state",
                "stale state: cutoff does not match valuation context",
            ));
        }
        if context.currency != self.note.currency() || context.asset != self.note.asset() {
            return Err(invalid("context", "currency/asset does not match contract"));
        }
        for date in self.future_fixing_dates() {
            context.grid.index(date)?;
            if date == context.cutoff.as_of && self.state.same_day_fixing_scenario.is_none() {
                return Err(invalid("same_day_fixing_scenario", "unknown valuation-day fixing requires an explicit scenario; valuation spot is not a fixing"));
            }
        }
        Ok(())
    }

    fn known_cashflows(&self) -> &[Cashflow] {
        &self.known
    }
    fn payment_dates(&self) -> Vec<Date> {
        PreparedNote::payment_dates(self)
    }
    fn requires_simulation(&self) -> bool {
        self.state.termination.is_none()
            && self
                .projected
                .as_ref()
                .is_none_or(|state| state.termination.is_none())
    }
    fn future_fixing_dates(&self) -> Vec<Date> {
        PreparedNote::future_fixing_dates(self)
    }

    fn cashflows(
        &self,
        path: &[f64],
        context: &ValuationContext,
        out: &mut Vec<Cashflow>,
    ) -> Result<()> {
        if !self.requires_simulation() {
            return Ok(());
        }
        let mut economic = EconomicState {
            first_knock_in_date: self
                .projected
                .as_ref()
                .map_or(self.state.first_knock_in_date, |state| {
                    state.first_knock_in_date
                }),
            termination: None,
            memory: self.projected.as_ref().map_or_else(
                || self.state.memory_coupon_ids.clone(),
                |state| state.memory.clone(),
            ),
            expired_memory: Vec::new(),
            cashflows: Vec::new(),
        };
        for date in self.future_fixing_dates() {
            if economic.termination.is_some() {
                break;
            }
            let spot = if date == context.cutoff.as_of {
                self.state
                    .same_day_fixing_scenario
                    .as_ref()
                    .ok_or_else(|| {
                        invalid(
                            "same_day_fixing_scenario",
                            "valuation spot cannot supply today's unknown contractual fixing",
                        )
                    })?
                    .spot
            } else {
                *path
                    .get(context.grid.index(date)?)
                    .ok_or_else(|| invalid("path", "path is shorter than event grid"))?
            };
            process_observation(&self.note, date, spot, &mut economic)?;
        }
        out.extend(economic.cashflows.into_iter().map(|entry| entry.cashflow));
        Ok(())
    }
}
