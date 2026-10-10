//! Dated contracts, bounded deterministic curves, and context-aware pricing.
//!
//! This layer is additive: legacy uniform-grid generators and discounted
//! `Payoff` implementations retain their original APIs and random streams.

pub mod curves;
pub mod engine;
pub mod notes;

pub use chrono::NaiveDate as Date;
pub use curves::{CurveNode, DiscountCurve, ForwardCurve, TimeGrid, ValuationContext};
pub use engine::{
    try_calendar_theta, CalendarThetaResult, Cashflow, CashflowEstimate, CashflowKind,
    CashflowPayoff, ContextPathGenerator, CurvePathGenerator, DatedPriceResult, DatedPricingMethod,
    Dynamics, GbmDynamics, HestonDynamics,
};
pub use notes::{
    advance_note_state, replay_note_history, roll_note_state, ActualPayment, DatedNote,
    DatedPhoenixNote, DatedSnowballNote, EventSchedule, Fixing, NoteState, PreparedNote,
    Receivable, SettlementConfirmation, StateProvenance, Termination,
};

use sha2::{Digest, Sha256};
use std::fmt;

/// Whether the contractual fixing on the valuation date has been processed.
/// Settlement records are dated EOD records, independently of this phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CutoffPhase {
    BeforeFixing,
    AfterFixing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValuationCutoff {
    pub as_of: Date,
    pub phase: CutoffPhase,
}

impl ValuationCutoff {
    pub fn includes_fixing(&self, date: Date) -> bool {
        date < self.as_of || (date == self.as_of && self.phase == CutoffPhase::AfterFixing)
    }
}

/// Errors in the additive dated layer, separate from legacy PricingError.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContextPricingError {
    InvalidParameter { name: &'static str, reason: String },
}

impl ContextPricingError {
    pub fn invalid(name: &'static str, reason: impl Into<String>) -> Self {
        Self::InvalidParameter {
            name,
            reason: reason.into(),
        }
    }
}

impl fmt::Display for ContextPricingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter { name, reason } => {
                write!(f, "invalid dated parameter `{name}`: {reason}")
            }
        }
    }
}

impl std::error::Error for ContextPricingError {}

pub(crate) fn number(
    value: f64,
    name: &'static str,
    positive: bool,
) -> Result<(), ContextPricingError> {
    if !value.is_finite() || (positive && value <= 0.0) {
        return Err(ContextPricingError::invalid(
            name,
            "must be finite and within its numeric domain",
        ));
    }
    Ok(())
}

pub(crate) fn nonempty(value: &str, name: &'static str) -> Result<(), ContextPricingError> {
    if value.trim().is_empty() {
        return Err(ContextPricingError::invalid(name, "must be supplied"));
    }
    Ok(())
}

/// Date-based ACT/365F, including zero and negative historical intervals.
pub fn year_fraction(from: Date, to: Date) -> f64 {
    (to - from).num_days() as f64 / 365.0
}

/// A length-delimited canonical identity, independent of hash-map ordering.
pub fn canonical_identity(parts: &[String]) -> String {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update((part.len() as u64).to_be_bytes());
        hash.update(part.as_bytes());
    }
    format!("{:x}", hash.finalize())
}

pub(crate) fn float_identity(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}
