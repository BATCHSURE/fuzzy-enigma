//! Dated European and explicitly monitored barrier cashflows.
//!
//! These context payoffs share the note engine's carry and payment discounting.
//! A numerical time-grid refinement never adds a contractual barrier fixing.

use super::{
    canonical_identity, float_identity, nonempty, number, year_fraction, Cashflow, CashflowKind,
    CashflowPayoff, ContextPricingError, Date, ValuationContext,
};
use crate::payoff::{BarrierKind, OptionType};

#[derive(Clone, Debug)]
pub struct ContextEuropeanOption {
    option_type: OptionType,
    strike: f64,
    expiry: Date,
    payment_date: Date,
    currency: String,
    cashflow_id: String,
}

impl ContextEuropeanOption {
    pub fn try_new(
        option_type: OptionType,
        strike: f64,
        expiry: Date,
        payment_date: Date,
        currency: String,
    ) -> Result<Self, ContextPricingError> {
        number(strike, "strike", true)?;
        nonempty(&currency, "currency")?;
        if payment_date < expiry {
            return Err(ContextPricingError::invalid(
                "payment_date",
                "precedes expiry",
            ));
        }
        let cashflow_id = canonical_identity(&[
            "context_european_v1".into(),
            format!("{option_type:?}"),
            float_identity(strike),
            expiry.to_string(),
            payment_date.to_string(),
            currency.clone(),
        ]);
        Ok(Self {
            option_type,
            strike,
            expiry,
            payment_date,
            currency,
            cashflow_id,
        })
    }

    fn amount(&self, path: &[f64], context: &ValuationContext) -> Result<f64, ContextPricingError> {
        let index = context.grid.index(self.expiry)?;
        let spot = path.get(index).copied().ok_or_else(|| {
            ContextPricingError::invalid("path_buffer", "does not contain expiry fixing")
        })?;
        number(spot, "stock_fixing", true)?;
        Ok(self.option_type.intrinsic(spot, self.strike))
    }

    fn emit(&self, amount: f64, out: &mut Vec<Cashflow>) {
        out.push(Cashflow {
            id: self.cashflow_id.clone(),
            fixing_date: Some(self.expiry),
            payment_date: self.payment_date,
            amount,
            kind: CashflowKind::Other,
            currency: self.currency.clone(),
        });
    }
}

impl CashflowPayoff for ContextEuropeanOption {
    fn validate(&self, context: &ValuationContext) -> Result<(), ContextPricingError> {
        if self.currency != context.currency {
            return Err(ContextPricingError::invalid(
                "currency",
                "option differs from context",
            ));
        }
        if self.expiry <= context.as_of() {
            return Err(ContextPricingError::invalid(
                "expiry",
                "must be after valuation date",
            ));
        }
        if (context.grid.maturity() - year_fraction(context.as_of(), self.expiry)).abs() > 1e-12 {
            return Err(ContextPricingError::invalid(
                "simulation_grid",
                "must end at option expiry",
            ));
        }
        context.grid.index(self.expiry)?;
        context.discount_curve.value_on(self.payment_date)?;
        Ok(())
    }
    fn cashflows(
        &self,
        path: &[f64],
        context: &ValuationContext,
        out: &mut Vec<Cashflow>,
    ) -> Result<(), ContextPricingError> {
        self.emit(self.amount(path, context)?, out);
        Ok(())
    }
    fn future_fixing_dates(&self) -> Vec<Date> {
        vec![self.expiry]
    }
    fn payment_dates(&self) -> Vec<Date> {
        vec![self.payment_date]
    }
}

#[derive(Clone, Debug)]
pub struct ContextBarrierOption {
    vanilla: ContextEuropeanOption,
    kind: BarrierKind,
    barrier: f64,
    rebate: f64,
    monitoring_dates: Vec<Date>,
}

impl ContextBarrierOption {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        option_type: OptionType,
        kind: BarrierKind,
        strike: f64,
        barrier: f64,
        rebate: f64,
        expiry: Date,
        payment_date: Date,
        monitoring_dates: Vec<Date>,
        currency: String,
    ) -> Result<Self, ContextPricingError> {
        number(barrier, "barrier", true)?;
        number(rebate, "rebate", false)?;
        if rebate < 0.0 {
            return Err(ContextPricingError::invalid(
                "rebate",
                "must be nonnegative",
            ));
        }
        if monitoring_dates.is_empty()
            || monitoring_dates.windows(2).any(|p| p[0] >= p[1])
            || monitoring_dates.iter().any(|&d| d > expiry)
        {
            return Err(ContextPricingError::invalid(
                "monitoring_dates",
                "must be nonempty, strictly increasing and no later than expiry",
            ));
        }
        let mut vanilla =
            ContextEuropeanOption::try_new(option_type, strike, expiry, payment_date, currency)?;
        let mut parts = vec![
            "context_barrier_v1".into(),
            vanilla.cashflow_id.clone(),
            format!("{kind:?}"),
            float_identity(barrier),
            float_identity(rebate),
        ];
        parts.extend(monitoring_dates.iter().map(ToString::to_string));
        vanilla.cashflow_id = canonical_identity(&parts);
        Ok(Self {
            vanilla,
            kind,
            barrier,
            rebate,
            monitoring_dates,
        })
    }
}

impl CashflowPayoff for ContextBarrierOption {
    fn validate(&self, context: &ValuationContext) -> Result<(), ContextPricingError> {
        self.vanilla.validate(context)?;
        for &date in &self.monitoring_dates {
            if date < context.as_of() {
                return Err(ContextPricingError::invalid(
                    "monitoring_dates",
                    "historical barrier fixings require a separate historical contract state",
                ));
            }
            context.grid.index(date)?;
        }
        Ok(())
    }
    fn cashflows(
        &self,
        path: &[f64],
        context: &ValuationContext,
        out: &mut Vec<Cashflow>,
    ) -> Result<(), ContextPricingError> {
        let mut hit = false;
        for &date in &self.monitoring_dates {
            let index = context.grid.index(date)?;
            let spot = path.get(index).copied().ok_or_else(|| {
                ContextPricingError::invalid("path_buffer", "does not contain monitoring fixing")
            })?;
            number(spot, "stock_fixing", true)?;
            hit |= match self.kind {
                BarrierKind::UpAndIn | BarrierKind::UpAndOut => spot >= self.barrier,
                BarrierKind::DownAndIn | BarrierKind::DownAndOut => spot <= self.barrier,
            };
        }
        let alive = match self.kind {
            BarrierKind::UpAndOut | BarrierKind::DownAndOut => !hit,
            BarrierKind::UpAndIn | BarrierKind::DownAndIn => hit,
        };
        let amount = if alive {
            self.vanilla.amount(path, context)?
        } else {
            self.rebate
        };
        self.vanilla.emit(amount, out);
        Ok(())
    }
    fn future_fixing_dates(&self) -> Vec<Date> {
        let mut dates = self.monitoring_dates.clone();
        dates.push(self.vanilla.expiry);
        dates.sort_unstable();
        dates.dedup();
        dates
    }
    fn payment_dates(&self) -> Vec<Date> {
        vec![self.vanilla.payment_date]
    }
}
