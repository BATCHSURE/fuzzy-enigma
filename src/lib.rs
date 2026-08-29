//! # fuzzy-enigma
//!
//! A Monte Carlo pricing library for path-dependent exotic derivatives.
//!
//! The library is organised around three traits:
//!
//! * [`PathGenerator`](model::PathGenerator) - drives a single asset path
//!   from a slice of standard-normal increments. Decoupling the noise from
//!   the model lets the engine apply variance-reduction techniques
//!   (antithetic variates) without the model knowing about them.
//! * [`Payoff`](payoff::Payoff) - maps a simulated path to its
//!   present-value cashflow.
//! * [`McEngine`](pricer::McEngine) - the Monte Carlo runner. It owns the
//!   sampling strategy (paths, RNG seed, antithetic, parallelism) and is
//!   independent of both model and payoff.
//!
//! See `examples/price_options.rs` for end-to-end usage.

pub mod analytic;
pub mod error;
pub mod greeks;
pub mod model;
pub mod payoff;
pub mod pricer;

#[cfg(feature = "python")]
mod python;

pub use error::PricingError;
pub use greeks::{bump_and_revalue, BumpSizes, Greeks};
pub use model::{GbmModel, PathGenerator};
pub use payoff::{
    AsianOption, AutocallableNote, BarrierKind, BarrierOption, CliquetOption, EuropeanOption,
    LookbackOption, OptionType, Payoff,
};
pub use pricer::{ControlVariate, EuropeanControl, McEngine, PriceResult};
