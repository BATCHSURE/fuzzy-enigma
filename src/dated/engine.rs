//! Undiscounted cashflow pricing and paired calendar rolls on Brownian union grids.

use super::{
    canonical_identity, float_identity, nonempty, number, year_fraction, ContextPricingError, Date,
    TimeGrid, ValuationContext, ValuationCutoff,
};
use crate::model::{GbmModel, HestonModel};
use crate::pricer::{summarise, McEngine, PriceResult};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CashflowKind {
    Principal,
    Coupon,
    Other,
}

impl CashflowKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Principal => "principal",
            Self::Coupon => "coupon",
            Self::Other => "other",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Cashflow {
    pub id: String,
    pub fixing_date: Option<Date>,
    pub payment_date: Date,
    pub amount: f64,
    pub kind: CashflowKind,
    pub currency: String,
}

/// Amounts emitted here are nominal, never already discounted.
pub trait CashflowPayoff: Send + Sync {
    fn validate(&self, context: &ValuationContext) -> Result<(), ContextPricingError>;
    fn cashflows(
        &self,
        path: &[f64],
        context: &ValuationContext,
        out: &mut Vec<Cashflow>,
    ) -> Result<(), ContextPricingError>;
    fn known_cashflows(&self) -> &[Cashflow] {
        &[]
    }
    fn requires_simulation(&self) -> bool {
        true
    }
    fn future_fixing_dates(&self) -> Vec<Date> {
        vec![]
    }
    /// All possible payment dates, including outcomes not sampled in this run.
    fn payment_dates(&self) -> Vec<Date> {
        self.known_cashflows()
            .iter()
            .map(|flow| flow.payment_date)
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GbmDynamics {
    pub volatility: f64,
}
impl GbmDynamics {
    pub fn try_new(volatility: f64) -> Result<Self, ContextPricingError> {
        GbmModel::try_new(1.0, 0.0, 0.0, volatility, 1.0, 1)
            .map_err(|err| ContextPricingError::invalid("gbm_dynamics", err.to_string()))?;
        Ok(Self { volatility })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HestonDynamics {
    /// Current variance at valuation, independent of note issue/history state.
    pub initial_variance: f64,
    pub kappa: f64,
    pub theta: f64,
    pub vol_of_vol: f64,
    pub rho: f64,
}
impl HestonDynamics {
    pub fn try_new(
        initial_variance: f64,
        kappa: f64,
        theta: f64,
        vol_of_vol: f64,
        rho: f64,
    ) -> Result<Self, ContextPricingError> {
        HestonModel::try_new(
            1.0,
            0.0,
            0.0,
            initial_variance,
            kappa,
            theta,
            vol_of_vol,
            rho,
            1.0,
            1,
        )
        .map_err(|err| ContextPricingError::invalid("heston_dynamics", err.to_string()))?;
        Ok(Self {
            initial_variance,
            kappa,
            theta,
            vol_of_vol,
            rho,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Dynamics {
    Gbm(GbmDynamics),
    Heston(HestonDynamics),
}
impl Dynamics {
    pub fn validate(&self) -> Result<(), ContextPricingError> {
        match *self {
            Self::Gbm(model) => {
                GbmDynamics::try_new(model.volatility)?;
            }
            Self::Heston(model) => {
                HestonDynamics::try_new(
                    model.initial_variance,
                    model.kappa,
                    model.theta,
                    model.vol_of_vol,
                    model.rho,
                )?;
            }
        }
        Ok(())
    }
    pub fn factors(&self) -> usize {
        match self {
            Self::Gbm(_) => 1,
            Self::Heston(_) => 2,
        }
    }
    pub fn identity(&self) -> String {
        let mut parts = vec!["dated_dynamics_v1".into()];
        match *self {
            Self::Gbm(model) => {
                parts.push("gbm_exact_log".into());
                parts.push(float_identity(model.volatility));
            }
            Self::Heston(model) => {
                parts.push("heston_log_euler_full_truncation_raw".into());
                parts.extend(
                    [
                        model.initial_variance,
                        model.kappa,
                        model.theta,
                        model.vol_of_vol,
                        model.rho,
                    ]
                    .map(float_identity),
                );
            }
        }
        canonical_identity(&parts)
    }
    pub fn prepare(
        &self,
        context: &ValuationContext,
    ) -> Result<CurvePathGenerator, ContextPricingError> {
        CurvePathGenerator::try_new(context, *self)
    }
}

pub trait ContextPathGenerator: Send + Sync {
    fn grid(&self) -> &TimeGrid;
    fn noise_dim(&self) -> usize;
    /// Fill exactly grid.steps() + 1 positive finite prices, starting at context.spot.
    fn generate(&self, normals: &[f64], out: &mut [f64]) -> Result<(), ContextPricingError>;
    /// Custom generators should override this identity to include their parameters.
    fn identity(&self) -> String {
        canonical_identity(&[
            "custom_context_generator_v1".into(),
            std::any::type_name::<Self>().into(),
            self.grid().identity(),
            self.noise_dim().to_string(),
        ])
    }
    fn validate(&self, context: &ValuationContext) -> Result<(), ContextPricingError> {
        validate_generator_context(context, self.grid())
    }
}

fn validate_generator_context(
    context: &ValuationContext,
    grid: &TimeGrid,
) -> Result<(), ContextPricingError> {
    context.validate()?;
    if context.grid.identity() != grid.identity() {
        return Err(ContextPricingError::invalid(
            "generator_grid",
            "differs from the valuation grid",
        ));
    }
    let forward = context.forward_curve.as_ref().ok_or_else(|| {
        ContextPricingError::invalid("forward_curve", "required for future fixing risk")
    })?;
    for &time in grid.times() {
        context.discount_curve.value(time)?;
        forward.value(time)?;
    }
    Ok(())
}

/// Precompiled nonuniform carry increments; neither dates nor curves are searched in the path loop.
#[derive(Clone, Debug)]
pub struct CurvePathGenerator {
    grid: TimeGrid,
    spot: f64,
    dynamics: Dynamics,
    dt: Vec<f64>,
    carry: Vec<f64>,
    heston: Option<HestonModel>,
    noise_dim: usize,
    context_identity: String,
}

impl CurvePathGenerator {
    pub fn try_new(
        context: &ValuationContext,
        dynamics: Dynamics,
    ) -> Result<Self, ContextPricingError> {
        context.validate()?;
        dynamics.validate()?;
        if context.grid.steps() == 0 {
            return Err(ContextPricingError::invalid(
                "simulation_grid",
                "a stochastic generator needs at least one future interval",
            ));
        }
        let forward = context.forward_curve.as_ref().ok_or_else(|| {
            ContextPricingError::invalid("forward_curve", "required for future fixing risk")
        })?;
        for &time in context.grid.times() {
            context.discount_curve.value(time)?;
        }
        let logs = context
            .grid
            .times()
            .iter()
            .map(|&t| forward.value(t).map(f64::ln))
            .collect::<Result<Vec<_>, _>>()?;
        let dt = context
            .grid
            .times()
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .collect();
        let carry = logs.windows(2).map(|pair| pair[1] - pair[0]).collect();
        let noise_dim = context
            .grid
            .steps()
            .checked_mul(dynamics.factors())
            .ok_or_else(|| ContextPricingError::invalid("noise_dim", "overflows"))?;
        let heston = match dynamics {
            Dynamics::Gbm(_) => None,
            Dynamics::Heston(model) => Some(
                HestonModel::try_new(
                    context.spot,
                    0.0,
                    0.0,
                    model.initial_variance,
                    model.kappa,
                    model.theta,
                    model.vol_of_vol,
                    model.rho,
                    context.grid.maturity(),
                    context.grid.steps(),
                )
                .map_err(|err| ContextPricingError::invalid("heston_dynamics", err.to_string()))?,
            ),
        };
        Ok(Self {
            grid: context.grid.clone(),
            spot: context.spot,
            dynamics,
            dt,
            carry,
            heston,
            noise_dim,
            context_identity: context.identity(),
        })
    }
    fn generate_with<F>(&self, out: &mut [f64], normal: F) -> Result<(), ContextPricingError>
    where
        F: Fn(usize) -> f64,
    {
        if out.len() != self.grid.steps() + 1 {
            return Err(ContextPricingError::invalid(
                "path_buffer",
                "length disagrees with the prepared grid",
            ));
        }
        out[0] = self.spot;
        let mut raw = match self.dynamics {
            Dynamics::Heston(model) => model.initial_variance,
            _ => 0.0,
        };
        for i in 0..self.grid.steps() {
            let dt = self.dt[i];
            let increment = match self.dynamics {
                Dynamics::Gbm(model) => {
                    self.carry[i] - 0.5 * model.volatility * model.volatility * dt
                        + model.volatility * dt.sqrt() * normal(i)
                }
                Dynamics::Heston(model) => {
                    let stock = normal(2 * i);
                    let variance = model.rho * stock
                        + (1.0 - model.rho * model.rho).max(0.0).sqrt() * normal(2 * i + 1);
                    let (positive, diffusion, next) = self
                        .heston
                        .as_ref()
                        .expect("prepared Heston")
                        .variance_step(raw, dt, variance);
                    raw = next;
                    if !raw.is_finite() {
                        return Err(ContextPricingError::invalid(
                            "variance_path",
                            "non-finite raw variance",
                        ));
                    }
                    self.carry[i] - 0.5 * positive * dt + diffusion * stock
                }
            };
            out[i + 1] = out[i] * increment.exp();
            number(out[i + 1], "stock_path", true)?;
        }
        Ok(())
    }
    fn generate_union(
        &self,
        union: &[f64],
        mapping: &[NoiseInterval],
        factors: usize,
        out: &mut [f64],
    ) -> Result<(), ContextPricingError> {
        self.generate_with(out, |index| {
            let interval = &mapping[index / factors];
            let factor = index % factors;
            interval
                .weights
                .iter()
                .map(|&(j, weight)| weight * union[j * factors + factor])
                .sum()
        })
    }
}

impl ContextPathGenerator for CurvePathGenerator {
    fn grid(&self) -> &TimeGrid {
        &self.grid
    }
    fn noise_dim(&self) -> usize {
        self.noise_dim
    }
    fn identity(&self) -> String {
        self.dynamics.identity()
    }
    fn validate(&self, context: &ValuationContext) -> Result<(), ContextPricingError> {
        validate_generator_context(context, &self.grid)?;
        if self.context_identity != context.identity() {
            return Err(ContextPricingError::invalid(
                "generator_context",
                "prepared carry and spot belong to a different valuation context",
            ));
        }
        Ok(())
    }
    fn generate(&self, normals: &[f64], out: &mut [f64]) -> Result<(), ContextPricingError> {
        if normals.len() != self.noise_dim || normals.iter().any(|z| !z.is_finite()) {
            return Err(ContextPricingError::invalid(
                "normal_buffer",
                "requires the expected number of finite independent normals",
            ));
        }
        self.generate_with(out, |i| normals[i])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DatedPricingMethod {
    Deterministic,
    MonteCarlo,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CashflowEstimate {
    pub id: String,
    pub fixing_date: Option<Date>,
    pub payment_date: Date,
    pub kind: CashflowKind,
    pub currency: String,
    pub expected_amount: f64,
    pub amount_std_error: f64,
    pub present_value: f64,
    pub pv_std_error: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DatedPriceResult {
    pub estimate: PriceResult,
    pub method: DatedPricingMethod,
    pub valuation_cutoff: ValuationCutoff,
    pub expected_cashflows: Vec<CashflowEstimate>,
    pub known_cashflows: Vec<Cashflow>,
    pub context_identity: String,
    pub dynamics_identity: Option<String>,
    pub grid_identity: String,
    /// Actual stock paths: antithetic budgets are rounded up to an even count.
    pub paths: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CashflowKey {
    id: String,
    fixing_date: Option<Date>,
    payment_date: Date,
    kind: CashflowKind,
    currency: String,
}
impl From<&Cashflow> for CashflowKey {
    fn from(flow: &Cashflow) -> Self {
        Self {
            id: flow.id.clone(),
            fixing_date: flow.fixing_date,
            payment_date: flow.payment_date,
            kind: flow.kind,
            currency: flow.currency.clone(),
        }
    }
}
#[derive(Clone, Debug)]
struct Sample {
    pv: f64,
    amounts: BTreeMap<CashflowKey, f64>,
}

fn validate_flow(flow: &Cashflow, context: &ValuationContext) -> Result<f64, ContextPricingError> {
    nonempty(&flow.id, "cashflow_id")?;
    number(flow.amount, "cashflow_amount", false)?;
    if flow.currency != context.currency {
        return Err(ContextPricingError::invalid(
            "cashflow_currency",
            "differs from context",
        ));
    }
    if flow
        .fixing_date
        .is_some_and(|date| flow.payment_date < date)
    {
        return Err(ContextPricingError::invalid(
            "payment_date",
            "must not precede its fixing",
        ));
    }
    context.discount_curve.value_on(flow.payment_date)
}

fn score(flows: &[Cashflow], context: &ValuationContext) -> Result<Sample, ContextPricingError> {
    let mut result = Sample {
        pv: 0.0,
        amounts: BTreeMap::new(),
    };
    for flow in flows {
        result.pv += flow.amount * validate_flow(flow, context)?;
        *result.amounts.entry(flow.into()).or_default() += flow.amount;
    }
    number(result.pv, "cashflow_pv", false)?;
    for &amount in result.amounts.values() {
        number(amount, "cashflow_total", false)?;
    }
    Ok(result)
}

fn evaluate<P: CashflowPayoff + ?Sized>(
    payoff: &P,
    context: &ValuationContext,
    path: &[f64],
) -> Result<Sample, ContextPricingError> {
    let mut flows = vec![];
    payoff.cashflows(path, context, &mut flows)?;
    score(&flows, context)
}

fn pair(mut first: Sample, second: Sample) -> Sample {
    first.pv = 0.5 * first.pv + 0.5 * second.pv;
    for amount in first.amounts.values_mut() {
        *amount *= 0.5;
    }
    for (key, amount) in second.amounts {
        *first.amounts.entry(key).or_default() += 0.5 * amount;
    }
    first
}

fn validate_payoff<P: CashflowPayoff + ?Sized>(
    context: &ValuationContext,
    payoff: &P,
) -> Result<Sample, ContextPricingError> {
    context.validate()?;
    payoff.validate(context)?;
    for date in payoff.future_fixing_dates() {
        context.grid.index(date)?;
    }
    for date in payoff.payment_dates() {
        context.discount_curve.value_on(date)?;
    }
    score(payoff.known_cashflows(), context)
}

fn result<P: CashflowPayoff + ?Sized>(
    context: &ValuationContext,
    payoff: &P,
    dynamics_identity: Option<String>,
    samples: &[Sample],
    known: &Sample,
    deterministic: bool,
    paths: usize,
) -> Result<DatedPriceResult, ContextPricingError> {
    let pvs: Vec<_> = samples.iter().map(|sample| sample.pv + known.pv).collect();
    let estimate = if deterministic {
        PriceResult {
            price: pvs[0],
            std_error: 0.0,
            samples: 0,
        }
    } else {
        summarise(&pvs)
    };
    validate_estimate(&estimate)?;
    let mut keys: BTreeSet<_> = known.amounts.keys().cloned().collect();
    for sample in samples {
        keys.extend(sample.amounts.keys().cloned());
    }
    let mut expected_cashflows = vec![];
    for key in keys {
        let known_amount = known.amounts.get(&key).copied().unwrap_or(0.0);
        let amounts: Vec<_> = samples
            .iter()
            .map(|sample| sample.amounts.get(&key).copied().unwrap_or(0.0) + known_amount)
            .collect();
        let amount = if deterministic {
            PriceResult {
                price: amounts[0],
                std_error: 0.0,
                samples: 0,
            }
        } else {
            summarise(&amounts)
        };
        validate_estimate(&amount)?;
        let discount = context.discount_curve.value_on(key.payment_date)?;
        number(amount.price * discount, "expected_cashflow_pv", false)?;
        number(
            amount.std_error * discount,
            "expected_cashflow_pv_std_error",
            false,
        )?;
        expected_cashflows.push(CashflowEstimate {
            id: key.id,
            fixing_date: key.fixing_date,
            payment_date: key.payment_date,
            kind: key.kind,
            currency: key.currency,
            expected_amount: amount.price,
            amount_std_error: amount.std_error,
            present_value: amount.price * discount,
            pv_std_error: amount.std_error * discount,
        });
    }
    Ok(DatedPriceResult {
        estimate,
        method: if deterministic {
            DatedPricingMethod::Deterministic
        } else {
            DatedPricingMethod::MonteCarlo
        },
        valuation_cutoff: context.cutoff,
        expected_cashflows,
        known_cashflows: payoff.known_cashflows().to_vec(),
        context_identity: context.identity(),
        dynamics_identity: if deterministic {
            None
        } else {
            dynamics_identity
        },
        grid_identity: context.grid.identity(),
        paths: if deterministic { 0 } else { paths },
    })
}

fn validate_estimate(estimate: &PriceResult) -> Result<(), ContextPricingError> {
    number(estimate.price, "estimated_pv", false)?;
    number(estimate.std_error, "estimated_std_error", false)?;
    if estimate.std_error < 0.0 {
        return Err(ContextPricingError::invalid(
            "estimated_std_error",
            "must not be negative",
        ));
    }
    let (lower, upper) = estimate.confidence_95();
    number(lower, "confidence_interval", false)?;
    number(upper, "confidence_interval", false)?;
    Ok(())
}

fn validate_path(path: &[f64], context: &ValuationContext) -> Result<(), ContextPricingError> {
    if path[0] != context.spot {
        return Err(ContextPricingError::invalid(
            "path_initial_spot",
            "must equal the current valuation spot",
        ));
    }
    for &spot in path {
        number(spot, "stock_path", true)?;
    }
    Ok(())
}

impl McEngine {
    pub fn try_price_context<P: CashflowPayoff + ?Sized>(
        &self,
        context: &ValuationContext,
        dynamics: Option<&Dynamics>,
        payoff: &P,
    ) -> Result<DatedPriceResult, ContextPricingError> {
        let known = validate_payoff(context, payoff)?;
        if !payoff.requires_simulation() || context.grid.steps() == 0 {
            let sample = evaluate(payoff, context, &[context.spot])?;
            return result(context, payoff, None, &[sample], &known, true, 0);
        }
        if self.paths == 0 {
            return Err(ContextPricingError::invalid(
                "paths",
                "must be positive for Monte Carlo pricing",
            ));
        }
        let dynamics = dynamics.ok_or_else(|| {
            ContextPricingError::invalid("dynamics", "required for future fixing risk")
        })?;
        let generator = dynamics.prepare(context)?;
        self.price_context_with_generator_checked(context, &generator, payoff, &known)
    }

    /// Additive entrypoint for a prepared or user-defined nonuniform path generator.
    /// This uses the same sample streams and antithetic pairing as the Dynamics entrypoint.
    pub fn try_price_context_with_generator<
        G: ContextPathGenerator + ?Sized,
        P: CashflowPayoff + ?Sized,
    >(
        &self,
        context: &ValuationContext,
        generator: &G,
        payoff: &P,
    ) -> Result<DatedPriceResult, ContextPricingError> {
        let known = validate_payoff(context, payoff)?;
        if !payoff.requires_simulation() || context.grid.steps() == 0 {
            let sample = evaluate(payoff, context, &[context.spot])?;
            return result(context, payoff, None, &[sample], &known, true, 0);
        }
        generator.validate(context)?;
        self.price_context_with_generator_checked(context, generator, payoff, &known)
    }

    fn price_context_with_generator_checked<
        G: ContextPathGenerator + ?Sized,
        P: CashflowPayoff + ?Sized,
    >(
        &self,
        context: &ValuationContext,
        generator: &G,
        payoff: &P,
        known: &Sample,
    ) -> Result<DatedPriceResult, ContextPricingError> {
        let n = if self.antithetic {
            self.paths.div_ceil(2)
        } else {
            self.paths
        };
        if n < 2 {
            return Err(ContextPricingError::invalid(
                "paths",
                "dated Monte Carlo SE requires at least two independent sample units",
            ));
        }
        let outcomes = self.run_samples(
            n,
            context.grid.steps(),
            generator.noise_dim(),
            |normals, path, i| {
                self.fill_normals(normals, i);
                generator.generate(normals, path)?;
                validate_path(path, context)?;
                let first = evaluate(payoff, context, path)?;
                if self.antithetic {
                    for z in normals.iter_mut() {
                        *z = -*z;
                    }
                    generator.generate(normals, path)?;
                    validate_path(path, context)?;
                    Ok(pair(first, evaluate(payoff, context, path)?))
                } else {
                    Ok(first)
                }
            },
        );
        let samples = outcomes
            .into_iter()
            .collect::<Result<Vec<_>, ContextPricingError>>()?;
        result(
            context,
            payoff,
            Some(generator.identity()),
            &samples,
            known,
            false,
            if self.antithetic { 2 * n } else { n },
        )
    }

    pub fn try_calendar_theta<P: CashflowPayoff + ?Sized, Q: CashflowPayoff + ?Sized>(
        &self,
        base: &ValuationContext,
        base_payoff: &P,
        rolled: &ValuationContext,
        rolled_payoff: &Q,
        dynamics: Option<&Dynamics>,
        cash_paid: &[Cashflow],
    ) -> Result<CalendarThetaResult, ContextPricingError> {
        try_calendar_theta(
            self,
            base,
            base_payoff,
            rolled,
            rolled_payoff,
            dynamics,
            cash_paid,
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CalendarThetaResult {
    pub base: DatedPriceResult,
    pub rolled: DatedPriceResult,
    pub pv_change: f64,
    pub std_error: f64,
    pub per_day: f64,
    pub per_day_std_error: f64,
    pub roll_days: i64,
    pub samples: usize,
    pub cash_paid: f64,
    /// D(delta)*rolled_PV + discounted paid cash - base_PV.
    pub base_date_discounted_change: f64,
    pub discounted_change_std_error: f64,
    pub covariance: f64,
    /// Absolute times relative to the base date; this grid couples noise only.
    pub union_times: Vec<f64>,
    pub coupling: String,
}

struct NoiseInterval {
    weights: Vec<(usize, f64)>,
}
fn noise_mapping(context: &ValuationContext, origin: Date, union: &[f64]) -> Vec<NoiseInterval> {
    let offset = year_fraction(origin, context.as_of());
    context
        .grid
        .times()
        .windows(2)
        .map(|interval| {
            let start = offset + interval[0];
            let end = offset + interval[1];
            let left = union
                .binary_search_by(|t| t.total_cmp(&start))
                .expect("union contains every leg boundary");
            let right = union
                .binary_search_by(|t| t.total_cmp(&end))
                .expect("union contains every leg boundary");
            let length = union[right] - union[left];
            NoiseInterval {
                weights: (left..right)
                    .map(|i| (i, ((union[i + 1] - union[i]) / length).sqrt()))
                    .collect(),
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn sample_union_leg<P: CashflowPayoff + ?Sized>(
    context: &ValuationContext,
    payoff: &P,
    generator: &Option<CurvePathGenerator>,
    mapping: &[NoiseInterval],
    normals: &[f64],
    path: &mut [f64],
    factors: usize,
) -> Result<Sample, ContextPricingError> {
    if let Some(generator) = generator {
        let path = &mut path[..context.grid.steps() + 1];
        generator.generate_union(normals, mapping, factors, path)?;
        evaluate(payoff, context, path)
    } else {
        evaluate(payoff, context, &[context.spot])
    }
}

/// Frozen-spot/dynamics calendar roll with paired Brownian noise.
/// Each leg retains its own update grid; refining the noise union never refines Heston's scheme.
pub fn try_calendar_theta<P: CashflowPayoff + ?Sized, Q: CashflowPayoff + ?Sized>(
    engine: &McEngine,
    base: &ValuationContext,
    base_payoff: &P,
    rolled: &ValuationContext,
    rolled_payoff: &Q,
    dynamics: Option<&Dynamics>,
    cash_paid: &[Cashflow],
) -> Result<CalendarThetaResult, ContextPricingError> {
    let base_known = validate_payoff(base, base_payoff)?;
    let rolled_known = validate_payoff(rolled, rolled_payoff)?;
    let expected = base.frozen_roll(rolled.cutoff)?;
    if expected.market_identity() != rolled.market_identity() {
        return Err(ContextPricingError::invalid(
            "roll_context",
            "calendar theta requires frozen spot and rebased base curves",
        ));
    }
    let roll_days = (rolled.as_of() - base.as_of()).num_days();
    let discount_roll = base.discount_curve.value_on(rolled.as_of())?;
    let mut cash = 0.0;
    let mut discounted_cash = 0.0;
    for flow in cash_paid {
        if flow.payment_date <= base.as_of() || flow.payment_date > rolled.as_of() {
            return Err(ContextPricingError::invalid(
                "cash_paid",
                "actual settlement date must be in (base date, rolled date]",
            ));
        }
        let discount = validate_flow(flow, base)?;
        cash += flow.amount;
        discounted_cash += flow.amount * discount;
    }
    number(cash, "cash_paid", false)?;
    number(discounted_cash, "discounted_cash_paid", false)?;
    let base_risk = base_payoff.requires_simulation() && base.grid.steps() > 0;
    let rolled_risk = rolled_payoff.requires_simulation() && rolled.grid.steps() > 0;
    if !base_risk && !rolled_risk {
        let b = engine.try_price_context(base, None, base_payoff)?;
        let r = engine.try_price_context(rolled, None, rolled_payoff)?;
        let change = r.estimate.price - b.estimate.price;
        number(change, "calendar_pv_change", false)?;
        number(
            discount_roll * r.estimate.price + discounted_cash - b.estimate.price,
            "calendar_discounted_change",
            false,
        )?;
        return Ok(CalendarThetaResult {
            base_date_discounted_change: discount_roll * r.estimate.price + discounted_cash
                - b.estimate.price,
            base: b,
            rolled: r,
            pv_change: change,
            std_error: 0.0,
            per_day: change / roll_days as f64,
            per_day_std_error: 0.0,
            roll_days,
            samples: 0,
            cash_paid: cash,
            discounted_change_std_error: 0.0,
            covariance: 0.0,
            union_times: vec![],
            coupling: "deterministic remaining cashflows; no Brownian sampling".into(),
        });
    }
    let dynamics = dynamics.ok_or_else(|| {
        ContextPricingError::invalid("dynamics", "required for at least one future-risk leg")
    })?;
    let n = if engine.antithetic {
        engine.paths.div_ceil(2)
    } else {
        engine.paths
    };
    if n < 2 {
        return Err(ContextPricingError::invalid(
            "paths",
            "paired calendar-theta SE requires at least two independent sample units",
        ));
    }
    let base_generator = if base_risk {
        Some(dynamics.prepare(base)?)
    } else {
        None
    };
    let rolled_generator = if rolled_risk {
        Some(dynamics.prepare(rolled)?)
    } else {
        None
    };
    let mut union = vec![];
    for (context, risk) in [(base, base_risk), (rolled, rolled_risk)] {
        if risk {
            let offset = year_fraction(base.as_of(), context.as_of());
            union.extend(context.grid.times().iter().map(|&t| offset + t));
        }
    }
    union.sort_by(f64::total_cmp);
    union.dedup();
    let base_mapping = if base_risk {
        noise_mapping(base, base.as_of(), &union)
    } else {
        vec![]
    };
    let rolled_mapping = if rolled_risk {
        noise_mapping(rolled, base.as_of(), &union)
    } else {
        vec![]
    };
    let factors = dynamics.factors();
    let noise_dim = (union.len() - 1)
        .checked_mul(factors)
        .ok_or_else(|| ContextPricingError::invalid("noise_dim", "union buffer overflows"))?;
    let steps = base.grid.steps().max(rolled.grid.steps());
    let outcomes = engine.run_samples(n, steps, noise_dim, |normals, path, i| {
        engine.fill_normals(normals, i);
        let b = sample_union_leg(
            base,
            base_payoff,
            &base_generator,
            &base_mapping,
            normals,
            path,
            factors,
        )?;
        let r = sample_union_leg(
            rolled,
            rolled_payoff,
            &rolled_generator,
            &rolled_mapping,
            normals,
            path,
            factors,
        )?;
        if engine.antithetic {
            for z in normals.iter_mut() {
                *z = -*z;
            }
            Ok((
                pair(
                    b,
                    sample_union_leg(
                        base,
                        base_payoff,
                        &base_generator,
                        &base_mapping,
                        normals,
                        path,
                        factors,
                    )?,
                ),
                pair(
                    r,
                    sample_union_leg(
                        rolled,
                        rolled_payoff,
                        &rolled_generator,
                        &rolled_mapping,
                        normals,
                        path,
                        factors,
                    )?,
                ),
            ))
        } else {
            Ok((b, r))
        }
    });
    let pairs = outcomes
        .into_iter()
        .collect::<Result<Vec<_>, ContextPricingError>>()?;
    let (bs, rs): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    let b = result(
        base,
        base_payoff,
        Some(dynamics.identity()),
        &bs,
        &base_known,
        !base_risk,
        if engine.antithetic { 2 * n } else { n },
    )?;
    let r = result(
        rolled,
        rolled_payoff,
        Some(dynamics.identity()),
        &rs,
        &rolled_known,
        !rolled_risk,
        if engine.antithetic { 2 * n } else { n },
    )?;
    let differences: Vec<_> = bs
        .iter()
        .zip(&rs)
        .map(|(b, r)| r.pv + rolled_known.pv - b.pv - base_known.pv)
        .collect();
    let discounted: Vec<_> = bs
        .iter()
        .zip(&rs)
        .map(|(b, r)| {
            discount_roll * (r.pv + rolled_known.pv) + discounted_cash - b.pv - base_known.pv
        })
        .collect();
    let delta = summarise(&differences);
    let reconciliation = summarise(&discounted);
    validate_estimate(&delta)?;
    validate_estimate(&reconciliation)?;
    let covariance = bs
        .iter()
        .zip(&rs)
        .map(|(x, y)| {
            (x.pv + base_known.pv - b.estimate.price) * (y.pv + rolled_known.pv - r.estimate.price)
        })
        .sum::<f64>()
        / (n - 1) as f64;
    number(covariance, "calendar_covariance", false)?;
    Ok(CalendarThetaResult{base:b,rolled:r,pv_change:delta.price,std_error:delta.std_error,per_day:delta.price/roll_days as f64,
                          per_day_std_error:delta.std_error/roll_days as f64,roll_days,samples:n,cash_paid:cash,
                          base_date_discounted_change:reconciliation.price,discounted_change_std_error:reconciliation.std_error,covariance,
                          union_times:union,coupling:"common Brownian increments on absolute-time union; each leg retains its own update grid".into()})
}
