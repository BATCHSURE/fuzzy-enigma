//! Positive, bounded log-linear dated curves and nonuniform simulation grids.

use super::{
    canonical_identity, float_identity, nonempty, number, year_fraction, ContextPricingError, Date,
    ValuationCutoff,
};

#[derive(Clone, Debug, PartialEq)]
pub struct CurveNode {
    pub date: Date,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct Curve {
    as_of: Date,
    currency: String,
    source: String,
    nodes: Vec<CurveNode>,
    times: Vec<f64>,
}

impl Curve {
    fn new(
        as_of: Date,
        currency: String,
        nodes: Vec<CurveNode>,
        source: String,
    ) -> Result<Self, ContextPricingError> {
        nonempty(&currency, "currency")?;
        nonempty(&source, "curve_source")?;
        if nodes.is_empty() || nodes[0].date != as_of {
            return Err(ContextPricingError::invalid(
                "curve_nodes",
                "an explicit valuation-date anchor is required",
            ));
        }
        for (i, node) in nodes.iter().enumerate() {
            number(node.value, "curve_value", true)?;
            if i > 0 && node.date <= nodes[i - 1].date {
                return Err(ContextPricingError::invalid(
                    "curve_nodes",
                    "dates must be strictly increasing",
                ));
            }
        }
        let times = nodes
            .iter()
            .map(|node| year_fraction(as_of, node.date))
            .collect();
        Ok(Self {
            as_of,
            currency,
            source,
            nodes,
            times,
        })
    }

    fn value(&self, t: f64) -> Result<f64, ContextPricingError> {
        number(t, "curve_time", false)?;
        let last = *self.times.last().expect("validated nonempty nodes");
        if t < 0.0 || t > last {
            return Err(ContextPricingError::invalid(
                "curve_coverage",
                format!("time {t} outside [0, {last}]; extrapolation is disabled"),
            ));
        }
        let i = self.times.partition_point(|&time| time < t);
        if self.times[i] == t {
            return Ok(self.nodes[i].value);
        }
        let alpha = (t - self.times[i - 1]) / (self.times[i] - self.times[i - 1]);
        let result =
            ((1.0 - alpha) * self.nodes[i - 1].value.ln() + alpha * self.nodes[i].value.ln()).exp();
        number(result, "interpolated_curve_value", true)?;
        Ok(result)
    }

    fn identity(&self, kind: &str, asset: &str) -> String {
        let mut parts = vec![
            "dated_curve_v1".into(),
            kind.into(),
            self.as_of.to_string(),
            self.currency.clone(),
            asset.into(),
            self.source.clone(),
            "ACT/365F;log_linear;no_extrapolation".into(),
        ];
        for node in &self.nodes {
            parts.push(node.date.to_string());
            parts.push(float_identity(node.value));
        }
        canonical_identity(&parts)
    }
}

/// Discount factors need not be decreasing and can exceed one.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscountCurve {
    inner: Curve,
}

impl DiscountCurve {
    pub fn try_new(
        as_of: Date,
        currency: String,
        nodes: Vec<CurveNode>,
        source: String,
    ) -> Result<Self, ContextPricingError> {
        let inner = Curve::new(as_of, currency, nodes, source)?;
        if inner.nodes[0].value != 1.0 {
            return Err(ContextPricingError::invalid(
                "discount_anchor",
                "D(0) must equal 1",
            ));
        }
        Ok(Self { inner })
    }
    pub fn as_of(&self) -> Date {
        self.inner.as_of
    }
    pub fn currency(&self) -> &str {
        &self.inner.currency
    }
    pub fn source(&self) -> &str {
        &self.inner.source
    }
    pub fn nodes(&self) -> &[CurveNode] {
        &self.inner.nodes
    }
    pub fn value(&self, t: f64) -> Result<f64, ContextPricingError> {
        self.inner.value(t)
    }
    pub fn value_on(&self, date: Date) -> Result<f64, ContextPricingError> {
        self.value(year_fraction(self.as_of(), date))
    }
    pub fn identity(&self) -> String {
        self.inner.identity("discount", "")
    }
    pub fn rebase(&self, new_as_of: Date) -> Result<Self, ContextPricingError> {
        let anchor = self.value_on(new_as_of)?;
        let mut nodes = vec![CurveNode {
            date: new_as_of,
            value: 1.0,
        }];
        nodes.extend(
            self.nodes()
                .iter()
                .filter(|node| node.date > new_as_of)
                .map(|node| CurveNode {
                    date: node.date,
                    value: node.value / anchor,
                }),
        );
        Self::try_new(
            new_as_of,
            self.currency().into(),
            nodes,
            format!("frozen_market_rebase({})", self.source()),
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ForwardCurve {
    inner: Curve,
    asset: String,
}

impl ForwardCurve {
    pub fn try_new(
        as_of: Date,
        currency: String,
        asset: String,
        nodes: Vec<CurveNode>,
        source: String,
    ) -> Result<Self, ContextPricingError> {
        nonempty(&asset, "asset")?;
        Ok(Self {
            inner: Curve::new(as_of, currency, nodes, source)?,
            asset,
        })
    }
    pub fn as_of(&self) -> Date {
        self.inner.as_of
    }
    pub fn currency(&self) -> &str {
        &self.inner.currency
    }
    pub fn asset(&self) -> &str {
        &self.asset
    }
    pub fn source(&self) -> &str {
        &self.inner.source
    }
    pub fn nodes(&self) -> &[CurveNode] {
        &self.inner.nodes
    }
    pub fn value(&self, t: f64) -> Result<f64, ContextPricingError> {
        self.inner.value(t)
    }
    pub fn value_on(&self, date: Date) -> Result<f64, ContextPricingError> {
        self.value(year_fraction(self.as_of(), date))
    }
    pub fn identity(&self) -> String {
        self.inner.identity("forward", self.asset())
    }
    pub fn rebase(&self, new_as_of: Date, new_spot: f64) -> Result<Self, ContextPricingError> {
        number(new_spot, "rolled_spot", true)?;
        let anchor = self.value_on(new_as_of)?;
        let mut nodes = vec![CurveNode {
            date: new_as_of,
            value: new_spot,
        }];
        nodes.extend(
            self.nodes()
                .iter()
                .filter(|node| node.date > new_as_of)
                .map(|node| CurveNode {
                    date: node.date,
                    value: new_spot * (node.value / anchor),
                }),
        );
        Self::try_new(
            new_as_of,
            self.currency().into(),
            self.asset().into(),
            nodes,
            format!("frozen_market_rebase({})", self.source()),
        )
    }
}

const MAX_GRID_STEPS: usize = 1_000_000;

/// Contractual dates are exact anchors; numerical substeps can be fractional days.
#[derive(Clone, Debug, PartialEq)]
pub struct TimeGrid {
    as_of: Date,
    times: Vec<f64>,
    fixing_dates: Vec<Date>,
    fixing_indices: Vec<usize>,
    max_step_days: f64,
}

impl TimeGrid {
    pub fn try_new(
        as_of: Date,
        mut fixing_dates: Vec<Date>,
        max_step_days: f64,
    ) -> Result<Self, ContextPricingError> {
        number(max_step_days, "max_step_days", true)?;
        fixing_dates.sort_unstable();
        fixing_dates.dedup();
        if fixing_dates.first().is_some_and(|&date| date < as_of) {
            return Err(ContextPricingError::invalid(
                "grid_dates",
                "historical dates cannot enter a future simulation grid",
            ));
        }
        let mut times = vec![0.0];
        let mut fixing_indices = Vec::with_capacity(fixing_dates.len());
        let mut previous = 0.0;
        for &date in &fixing_dates {
            let t = year_fraction(as_of, date);
            let ratio = (t - previous) * 365.0 / max_step_days;
            let nearest = ratio.round();
            let ratio = if (ratio - nearest).abs() <= 16.0 * f64::EPSILON * ratio.abs().max(1.0) {
                nearest
            } else {
                ratio
            };
            let intervals = ratio.ceil();
            if !intervals.is_finite()
                || intervals > MAX_GRID_STEPS as f64
                || times.len().saturating_add(intervals as usize) > MAX_GRID_STEPS + 1
            {
                return Err(ContextPricingError::invalid(
                    "grid_size",
                    "refinement exceeds the one-million-step limit",
                ));
            }
            let intervals = intervals as usize;
            for j in 1..=intervals {
                let next = if j == intervals {
                    t
                } else {
                    previous + (t - previous) * j as f64 / intervals as f64
                };
                if next <= *times.last().expect("nonempty grid") {
                    return Err(ContextPricingError::invalid(
                        "grid_size",
                        "refinement cannot be represented by increasing floating-point times",
                    ));
                }
                times.push(next);
            }
            fixing_indices.push(times.len() - 1);
            previous = t;
        }
        Ok(Self {
            as_of,
            times,
            fixing_dates,
            fixing_indices,
            max_step_days,
        })
    }
    pub fn as_of(&self) -> Date {
        self.as_of
    }
    pub fn times(&self) -> &[f64] {
        &self.times
    }
    pub fn fixing_dates(&self) -> &[Date] {
        &self.fixing_dates
    }
    pub fn max_step_days(&self) -> f64 {
        self.max_step_days
    }
    pub fn index(&self, date: Date) -> Result<usize, ContextPricingError> {
        self.fixing_dates
            .binary_search(&date)
            .map(|i| self.fixing_indices[i])
            .map_err(|_| {
                ContextPricingError::invalid(
                    "grid_anchor",
                    format!("fixing date {date} is not an exact contractual grid anchor"),
                )
            })
    }
    pub fn steps(&self) -> usize {
        self.times.len() - 1
    }
    pub fn maturity(&self) -> f64 {
        *self.times.last().expect("nonempty grid")
    }
    pub fn identity(&self) -> String {
        let mut parts = vec![
            "dated_grid_v1".into(),
            self.as_of.to_string(),
            float_identity(self.max_step_days),
        ];
        parts.extend(self.times.iter().map(|&v| float_identity(v)));
        for date in &self.fixing_dates {
            parts.push(date.to_string());
        }
        canonical_identity(&parts)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ValuationContext {
    pub cutoff: ValuationCutoff,
    pub spot: f64,
    pub currency: String,
    pub asset: String,
    pub discount_curve: DiscountCurve,
    pub forward_curve: Option<ForwardCurve>,
    pub source: String,
    pub grid: TimeGrid,
}

impl ValuationContext {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        cutoff: ValuationCutoff,
        spot: f64,
        currency: String,
        asset: String,
        discount_curve: DiscountCurve,
        forward_curve: Option<ForwardCurve>,
        source: String,
    ) -> Result<Self, ContextPricingError> {
        let result = Self {
            cutoff,
            spot,
            currency,
            asset,
            discount_curve,
            forward_curve,
            source,
            grid: TimeGrid::try_new(cutoff.as_of, vec![], 1.0)?,
        };
        result.validate()?;
        Ok(result)
    }
    pub fn as_of(&self) -> Date {
        self.cutoff.as_of
    }
    pub fn validate(&self) -> Result<(), ContextPricingError> {
        number(self.spot, "spot", true)?;
        nonempty(&self.currency, "currency")?;
        nonempty(&self.asset, "asset")?;
        if self.discount_curve.as_of() != self.as_of()
            || self.discount_curve.currency() != self.currency
            || self.grid.as_of() != self.as_of()
        {
            return Err(ContextPricingError::invalid(
                "context",
                "discount/grid as-of or currency differs from context",
            ));
        }
        if let Some(curve) = &self.forward_curve {
            if curve.as_of() != self.as_of()
                || curve.currency() != self.currency
                || curve.asset() != self.asset
                || curve.value(0.0)? != self.spot
            {
                return Err(ContextPricingError::invalid(
                    "context",
                    "forward as-of, currency, asset or F(0) differs from context",
                ));
            }
        }
        Ok(())
    }
    pub fn with_grid(&self, grid: TimeGrid) -> Result<Self, ContextPricingError> {
        let mut result = self.clone();
        result.grid = grid;
        result.validate()?;
        Ok(result)
    }
    pub fn market_identity(&self) -> String {
        canonical_identity(&[
            "dated_market_v1".into(),
            self.as_of().to_string(),
            format!("{:?}", self.cutoff.phase),
            float_identity(self.spot),
            self.currency.clone(),
            self.asset.clone(),
            self.discount_curve.identity(),
            self.forward_curve
                .as_ref()
                .map(ForwardCurve::identity)
                .unwrap_or_default(),
            self.source.clone(),
        ])
    }
    pub fn identity(&self) -> String {
        canonical_identity(&[self.market_identity(), self.grid.identity()])
    }
    pub fn frozen_roll(&self, cutoff: ValuationCutoff) -> Result<Self, ContextPricingError> {
        self.validate()?;
        if cutoff.as_of <= self.as_of() {
            return Err(ContextPricingError::invalid(
                "roll_date",
                "must advance by at least one calendar day",
            ));
        }
        let discount = self.discount_curve.rebase(cutoff.as_of)?;
        let dates: Vec<_> = self
            .grid
            .fixing_dates()
            .iter()
            .copied()
            .filter(|&date| date >= cutoff.as_of)
            .collect();
        let forward = self
            .forward_curve
            .as_ref()
            .filter(|curve| {
                !dates.is_empty()
                    || cutoff.as_of <= curve.nodes().last().expect("nonempty curve").date
            })
            .map(|curve| curve.rebase(cutoff.as_of, self.spot))
            .transpose()?;
        let result = Self::try_new(
            cutoff,
            self.spot,
            self.currency.clone(),
            self.asset.clone(),
            discount,
            forward,
            format!("frozen_spot_dynamics_roll({})", self.source),
        )?;
        result.with_grid(TimeGrid::try_new(
            cutoff.as_of,
            dates,
            self.grid.max_step_days(),
        )?)
    }
}
