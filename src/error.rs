//! Errors raised when a model, payoff or engine is misconfigured.
//!
//! Validation happens up front - when a model is constructed and once per
//! pricing run, before any path is simulated - rather than inside the hot
//! loop. That matters for more than tidiness: a bounds check that fires
//! per-path fires once per rayon worker, which turns a single bad input
//! into a storm of panics from every thread at once.

use std::fmt;

/// Reasons a pricing request can be rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PricingError {
    /// A model or payoff parameter is outside its valid domain.
    InvalidParameter { name: &'static str, reason: String },
    /// An observation index falls outside the simulated path. The path
    /// holds `steps + 1` values, so the largest legal index is `steps`.
    ObservationOutOfRange { index: usize, steps: usize },
    /// A schedule-driven payoff was given no observation dates.
    EmptyObservationSchedule,
    /// Observation dates must be strictly increasing.
    UnorderedObservationSchedule { previous: usize, next: usize },
    /// The engine was asked for zero paths, which has no defined mean.
    NoPaths,
}

impl PricingError {
    /// Convenience constructor for [`PricingError::InvalidParameter`].
    pub fn invalid(name: &'static str, reason: impl Into<String>) -> Self {
        PricingError::InvalidParameter {
            name,
            reason: reason.into(),
        }
    }
}

impl fmt::Display for PricingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PricingError::InvalidParameter { name, reason } => {
                write!(f, "invalid parameter `{name}`: {reason}")
            }
            PricingError::ObservationOutOfRange { index, steps } => write!(
                f,
                "observation index {index} is outside the simulated path \
                 (model has {steps} steps, so the largest legal index is {steps})"
            ),
            PricingError::EmptyObservationSchedule => {
                write!(f, "observation schedule must contain at least one date")
            }
            PricingError::UnorderedObservationSchedule { previous, next } => write!(
                f,
                "observation schedule must be strictly increasing, \
                 but {next} follows {previous}"
            ),
            PricingError::NoPaths => {
                write!(f, "engine was configured with zero paths")
            }
        }
    }
}

impl std::error::Error for PricingError {}
