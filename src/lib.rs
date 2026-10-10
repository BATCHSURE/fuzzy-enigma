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
pub mod dated;
pub mod error;
pub mod greeks;
pub mod model;
pub mod numerical;
pub mod payoff;
pub mod pricer;
pub mod structured;
pub mod validation;

#[cfg(feature = "python")]
mod python;

pub use error::PricingError;
pub use greeks::{
    bump_and_revalue, try_bump_and_revalue, try_bump_and_revalue_with,
    try_continuous_barrier_greeks, BumpSizes, Greeks,
};
pub use model::{GbmModel, HestonModel, PathGenerator};
pub use numerical::{geometric_asian_price, GeometricAsianControl, ScheduledAsianOption};
pub use payoff::{
    AsianOption, AutocallableNote, BarrierKind, BarrierOption, CliquetOption, EuropeanOption,
    LookbackOption, OptionType, Payoff,
};
pub use pricer::{ControlVariate, EuropeanControl, McEngine, PriceResult};
pub use structured::{PhoenixNote, SnowballNote};
pub use validation::{try_price_heston_conditional, try_price_heston_conditional_with_shifts};
