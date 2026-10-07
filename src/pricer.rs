//! Monte Carlo pricing engine.
//!
//! The engine is independent of the underlying model and the payoff: it
//! only needs a [`PathGenerator`] to produce paths and a [`Payoff`] to
//! score them. It supports antithetic variance reduction and parallel
//! evaluation via rayon.

use crate::analytic;
use crate::error::PricingError;
use crate::model::{GbmModel, PathGenerator};
use crate::payoff::{EuropeanOption, OptionType, Payoff};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

/// Outcome of a Monte Carlo pricing run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PriceResult {
    /// Estimated present value.
    pub price: f64,
    /// Standard error of the estimator (sample std-dev / sqrt(n)).
    pub std_error: f64,
    /// Number of independent samples that fed the mean. With antithetic
    /// variates this is half of `McEngine::paths`, since each pair of
    /// negated paths contributes one paired sample.
    pub samples: usize,
}

impl PriceResult {
    /// 95% confidence interval (price +/- 1.96 * std_error).
    pub fn confidence_95(&self) -> (f64, f64) {
        let half = 1.959_963_984_540_054 * self.std_error;
        (self.price - half, self.price + half)
    }
}

/// Configuration for a Monte Carlo pricing run.
#[derive(Clone, Debug)]
pub struct McEngine {
    /// Total number of simulated paths. With `antithetic = true` this is
    /// rounded up to an even number and paths are generated in pairs.
    pub paths: usize,
    /// Base seed. Every path draws from its own ChaCha20 *stream* under
    /// this one key (path `i` uses stream `i`), which gives 2^64 disjoint
    /// streams per seed. Results are therefore reproducible regardless of
    /// `parallel`, and two runs at different seeds are genuinely
    /// independent rather than shifted copies of one another.
    pub seed: u64,
    /// Whether to use antithetic variates.
    pub antithetic: bool,
    /// Whether to evaluate paths in parallel via rayon.
    pub parallel: bool,
}

impl Default for McEngine {
    fn default() -> Self {
        Self {
            paths: 100_000,
            seed: 0xC0FFEE_u64,
            antithetic: true,
            parallel: true,
        }
    }
}

impl McEngine {
    pub fn new(paths: usize) -> Self {
        Self {
            paths,
            ..Self::default()
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_antithetic(mut self, on: bool) -> Self {
        self.antithetic = on;
        self
    }

    pub fn with_parallel(mut self, on: bool) -> Self {
        self.parallel = on;
        self
    }

    /// Price the given payoff under the given model.
    ///
    /// Panics if the configuration is invalid; see [`McEngine::try_price`]
    /// for the fallible form.
    pub fn price<M, P>(&self, model: &M, payoff: &P) -> PriceResult
    where
        M: PathGenerator + ?Sized,
        P: Payoff + ?Sized,
    {
        self.try_price(model, payoff)
            .expect("invalid pricing setup")
    }

    /// Price the given payoff under the given model, validating the setup
    /// first.
    ///
    /// Validation runs once, before any path is simulated, so a bad
    /// observation schedule or an empty path budget is reported as a
    /// single error instead of panicking inside every worker thread.
    pub fn try_price<M, P>(&self, model: &M, payoff: &P) -> Result<PriceResult, PricingError>
    where
        M: PathGenerator + ?Sized,
        P: Payoff + ?Sized,
    {
        let (n_steps, noise_dim) = validate_model(model)?;
        payoff.validate(n_steps)?;
        if self.paths == 0 {
            return Err(PricingError::NoPaths);
        }

        let dt = model.maturity() / n_steps as f64;
        let r = model.risk_free_rate();

        // With antithetic variates each "sample" is the average of a path
        // and its negation, so we run paths/2 sample units to keep total
        // path count consistent with the user's request.
        let n_samples = if self.antithetic {
            self.paths.div_ceil(2)
        } else {
            self.paths
        };

        let samples = self.run_samples(n_samples, n_steps, noise_dim, |normals, path, i| {
            self.fill_normals(normals, i);
            model.generate(normals, path);
            let pv1 = payoff.evaluate(path, dt, r);

            if self.antithetic {
                for x in normals.iter_mut() {
                    *x = -*x;
                }
                model.generate(normals, path);
                0.5 * (pv1 + payoff.evaluate(path, dt, r))
            } else {
                pv1
            }
        });

        Ok(summarise(&samples))
    }

    /// Price a payoff using a control variate to reduce variance.
    ///
    /// Panics if the configuration is invalid; see
    /// [`McEngine::try_price_with_control`] for the fallible form.
    pub fn price_with_control<M, P, C>(&self, model: &M, payoff: &P, control: &C) -> PriceResult
    where
        M: PathGenerator + ?Sized,
        P: Payoff + ?Sized,
        C: ControlVariate + ?Sized,
    {
        self.try_price_with_control(model, payoff, control)
            .expect("invalid pricing setup")
    }

    /// Price a payoff using a control variate, validating the setup first.
    ///
    /// The optimal coefficient `beta = Cov(X, Y) / Var(X)` is estimated
    /// from the same sample that produces the price. That is the standard
    /// approach and introduces an `O(1/n)` bias, negligible against the
    /// `O(1/sqrt(n))` standard error at any realistic path count, but it
    /// does mean the estimator is not *exactly* unbiased the way a
    /// fixed-beta control would be.
    ///
    /// If the control turns out to be uncorrelated with the payoff (or
    /// constant, so `Var(X) = 0`), beta falls back to zero and this
    /// degrades gracefully to [`McEngine::try_price`].
    pub fn try_price_with_control<M, P, C>(
        &self,
        model: &M,
        payoff: &P,
        control: &C,
    ) -> Result<PriceResult, PricingError>
    where
        M: PathGenerator + ?Sized,
        P: Payoff + ?Sized,
        C: ControlVariate + ?Sized,
    {
        let (n_steps, noise_dim) = validate_model(model)?;
        payoff.validate(n_steps)?;
        control.validate(n_steps)?;
        if self.paths == 0 {
            return Err(PricingError::NoPaths);
        }
        if !control.expectation().is_finite() {
            return Err(PricingError::invalid(
                "control expectation",
                "must be finite",
            ));
        }

        let dt = model.maturity() / n_steps as f64;
        let r = model.risk_free_rate();
        let n_samples = if self.antithetic {
            self.paths.div_ceil(2)
        } else {
            self.paths
        };

        // Payoff and control are scored on the very same paths - that
        // shared randomness is the whole source of their correlation.
        let pairs: Vec<(f64, f64)> =
            self.run_samples(n_samples, n_steps, noise_dim, |normals, path, i| {
                self.fill_normals(normals, i);
                model.generate(normals, path);
                let (y1, x1) = (payoff.evaluate(path, dt, r), control.evaluate(path, dt, r));

                if self.antithetic {
                    for v in normals.iter_mut() {
                        *v = -*v;
                    }
                    model.generate(normals, path);
                    (
                        0.5 * (y1 + payoff.evaluate(path, dt, r)),
                        0.5 * (x1 + control.evaluate(path, dt, r)),
                    )
                } else {
                    (y1, x1)
                }
            });

        let n = pairs.len() as f64;
        let mean_y = pairs.iter().map(|p| p.0).sum::<f64>() / n;
        let mean_x = pairs.iter().map(|p| p.1).sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        for &(y, x) in &pairs {
            cov += (x - mean_x) * (y - mean_y);
            var_x += (x - mean_x) * (x - mean_x);
        }
        let beta = if var_x > 0.0 { cov / var_x } else { 0.0 };

        // Re-express every sample as its control-adjusted residual, then
        // summarise exactly as the plain estimator does.
        let expectation = control.expectation();
        let adjusted: Vec<f64> = pairs
            .iter()
            .map(|&(y, x)| y - beta * (x - expectation))
            .collect();

        Ok(summarise(&adjusted))
    }

    /// Draw the standard normals for sample `i` into `normals`.
    ///
    /// Path `i` gets ChaCha20 stream `i` under the engine's key. Deriving
    /// the stream (rather than perturbing the seed) is what makes two runs
    /// at neighbouring seeds independent: `seed + 1` would otherwise just
    /// re-run this run's paths shifted by one.
    fn fill_normals(&self, normals: &mut [f64], i: usize) {
        let mut rng = ChaCha20Rng::seed_from_u64(self.seed);
        rng.set_stream(i as u64);
        for x in normals.iter_mut() {
            *x = StandardNormal.sample(&mut rng);
        }
    }

    /// Evaluate `f` over `n_samples` sample units, serially or across
    /// rayon threads.
    ///
    /// The scratch buffers are allocated once per thread and reused, which
    /// keeps the per-path heap traffic out of the hot loop. Because each
    /// sample derives its own RNG stream from its index, the result is
    /// identical - bit for bit - whether or not `parallel` is set.
    fn run_samples<F, T>(&self, n_samples: usize, n_steps: usize, noise_dim: usize, f: F) -> Vec<T>
    where
        F: Fn(&mut Vec<f64>, &mut Vec<f64>, usize) -> T + Send + Sync,
        T: Send,
    {
        let make_buffers = || (vec![0.0_f64; noise_dim], vec![0.0_f64; n_steps + 1]);

        if self.parallel {
            (0..n_samples)
                .into_par_iter()
                .map_init(make_buffers, |(normals, path), i| f(normals, path, i))
                .collect()
        } else {
            let (mut normals, mut path) = make_buffers();
            (0..n_samples)
                .map(|i| f(&mut normals, &mut path, i))
                .collect()
        }
    }
}

// -----------------------------------------------------------------------
// Control variates
// -----------------------------------------------------------------------

/// A quantity that is correlated with the payoff being priced and whose
/// expectation is known in closed form.
///
/// Given such a control `X` with known `E[X]`, the estimator
/// `mean(Y) - beta * (mean(X) - E[X])` is unbiased for `E[Y]` and has
/// lower variance than `mean(Y)` whenever `X` and `Y` are correlated.
/// This is kept separate from [`Payoff`] because a control is not a
/// tradeable payoff the engine prices - it is scaffolding used to price
/// something else.
pub trait ControlVariate: Send + Sync {
    /// Validate any observation schedule before path simulation.
    fn validate(&self, _steps: usize) -> Result<(), PricingError> {
        Ok(())
    }

    /// Present value of the control on this path, using the same
    /// convention as [`Payoff::evaluate`].
    fn evaluate(&self, path: &[f64], dt: f64, r: f64) -> f64;

    /// The control's analytically known expectation.
    fn expectation(&self) -> f64;
}

/// Validate the common model contract, including custom implementations
/// that rely on the default model-specific validation hook.
fn validate_model<M: PathGenerator + ?Sized>(model: &M) -> Result<(usize, usize), PricingError> {
    model.validate()?;
    let steps = model.steps();
    if steps == 0 {
        return Err(PricingError::invalid("steps", "must be at least 1"));
    }
    let path_len = steps
        .checked_add(1)
        .ok_or_else(|| PricingError::invalid("steps", "path length overflows"))?;
    if !model.maturity().is_finite() || model.maturity() <= 0.0 {
        return Err(PricingError::invalid(
            "maturity",
            "must be finite and positive",
        ));
    }
    if !model.risk_free_rate().is_finite() {
        return Err(PricingError::invalid("risk_free_rate", "must be finite"));
    }
    let noise_dim = model.noise_dim();
    let max_buffer_len = (isize::MAX as usize) / std::mem::size_of::<f64>();
    if path_len > max_buffer_len || noise_dim > max_buffer_len {
        return Err(PricingError::invalid(
            "steps",
            "simulation buffer size overflows",
        ));
    }
    Ok((steps, noise_dim))
}

/// A European option under [`GbmModel`] used as a control variate.
///
/// The natural control for most single-asset exotics: it is cheap to
/// evaluate on a path that has already been generated, strongly correlated
/// with Asian and barrier payoffs at the same strike, and its expectation
/// is exactly the Black-Scholes price.
#[derive(Clone, Debug)]
pub struct EuropeanControl {
    payoff: EuropeanOption,
    expectation: f64,
}

impl EuropeanControl {
    /// Build a control from the model the exotic is being priced under, so
    /// the analytic expectation is consistent with the simulation.
    pub fn new(model: &GbmModel, option_type: OptionType, strike: f64) -> Self {
        let expectation = analytic::bs_price(
            option_type,
            model.spot,
            strike,
            model.risk_free_rate,
            model.dividend_yield,
            model.volatility,
            model.maturity,
        );
        Self {
            payoff: EuropeanOption {
                option_type,
                strike,
            },
            expectation,
        }
    }
}

impl ControlVariate for EuropeanControl {
    fn evaluate(&self, path: &[f64], dt: f64, r: f64) -> f64 {
        self.payoff.evaluate(path, dt, r)
    }

    fn expectation(&self) -> f64 {
        self.expectation
    }
}

/// Mean and standard error of a set of i.i.d. sample payoffs.
fn summarise(samples: &[f64]) -> PriceResult {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = if samples.len() > 1 {
        samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    PriceResult {
        price: mean,
        std_error: (var / n).sqrt(),
        samples: samples.len(),
    }
}
