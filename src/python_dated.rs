//! Native, additive bindings for dated curves and outstanding notes.

use chrono::NaiveDate;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use crate::dated::*;
use crate::pricer::McEngine;

fn error(err: ContextPricingError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Date-only input deliberately rejects datetime, which inherits from date.
fn date(value: &Bound<'_, PyAny>) -> PyResult<NaiveDate> {
    let text = if value.is_instance_of::<PyString>() {
        value.extract::<String>()?
    } else {
        let module = value.py().import("datetime")?;
        if value.is_instance(&module.getattr("datetime")?)?
            || !value.is_instance(&module.getattr("date")?)?
        {
            return Err(PyValueError::new_err(
                "expected an ISO date or datetime.date; datetime.datetime is not supported",
            ));
        }
        value.call_method0("isoformat")?.extract::<String>()?
    };
    if text.len() != 10 {
        return Err(PyValueError::new_err(
            "date must have the canonical YYYY-MM-DD format",
        ));
    }
    let parsed = NaiveDate::parse_from_str(&text, "%Y-%m-%d")
        .map_err(|_| PyValueError::new_err("date must have the canonical YYYY-MM-DD format"))?;
    if parsed.format("%Y-%m-%d").to_string() != text {
        return Err(PyValueError::new_err(
            "date must have the canonical YYYY-MM-DD format",
        ));
    }
    Ok(parsed)
}

fn dates(values: &Bound<'_, PyAny>) -> PyResult<Vec<NaiveDate>> {
    values.try_iter()?.map(|value| date(&value?)).collect()
}

fn strings(values: &[NaiveDate]) -> Vec<String> {
    values.iter().map(ToString::to_string).collect()
}

#[pyclass(
    name = "ValuationCutoff",
    module = "fuzzy_enigma",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(super) struct PyCutoff {
    pub inner: ValuationCutoff,
}

#[pymethods]
impl PyCutoff {
    #[new]
    #[pyo3(signature = (as_of, phase = "after_fixing"))]
    fn new(as_of: &Bound<'_, PyAny>, phase: &str) -> PyResult<Self> {
        let phase = match phase {
            "before_fixing" => CutoffPhase::BeforeFixing,
            "after_fixing" => CutoffPhase::AfterFixing,
            _ => {
                return Err(PyValueError::new_err(
                    "phase must be before_fixing or after_fixing",
                ))
            }
        };
        Ok(Self {
            inner: ValuationCutoff {
                as_of: date(as_of)?,
                phase,
            },
        })
    }
    #[getter]
    fn as_of(&self) -> String {
        self.inner.as_of.to_string()
    }
    #[getter]
    fn phase(&self) -> &'static str {
        phase_name(self.inner.phase)
    }
    fn __repr__(&self) -> String {
        format!("ValuationCutoff({}, {})", self.as_of(), self.phase())
    }
}

fn phase_name(phase: CutoffPhase) -> &'static str {
    match phase {
        CutoffPhase::BeforeFixing => "before_fixing",
        CutoffPhase::AfterFixing => "after_fixing",
    }
}

#[pyclass(
    name = "EventSchedule",
    module = "fuzzy_enigma",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(super) struct PySchedule {
    inner: EventSchedule,
}

#[pymethods]
impl PySchedule {
    #[new]
    fn new(
        fixing_dates: &Bound<'_, PyAny>,
        payment_dates: &Bound<'_, PyAny>,
        calendar_source: String,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: EventSchedule::new(dates(fixing_dates)?, dates(payment_dates)?, calendar_source)
                .map_err(error)?,
        })
    }
    #[getter]
    fn fixing_dates(&self) -> Vec<String> {
        strings(self.inner.fixing_dates())
    }
    #[getter]
    fn payment_dates(&self) -> Vec<String> {
        strings(self.inner.payment_dates())
    }
    #[getter]
    fn calendar_source(&self) -> String {
        self.inner.calendar_source().to_owned()
    }
}

#[pyclass(
    name = "DatedSnowballNote",
    module = "fuzzy_enigma",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(super) struct PySnowball {
    pub inner: DatedSnowballNote,
}

#[pymethods]
impl PySnowball {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        notional: f64,
        reference_spot: f64,
        coupon_rate: f64,
        knock_in_barrier: f64,
        knock_out_barrier: f64,
        issue_date: &Bound<'_, PyAny>,
        currency: String,
        asset: String,
        knock_in_dates: &Bound<'_, PyAny>,
        autocall_schedule: PyRef<'_, PySchedule>,
        maturity_schedule: PyRef<'_, PySchedule>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: DatedSnowballNote::new(
                notional,
                reference_spot,
                coupon_rate,
                knock_in_barrier,
                knock_out_barrier,
                date(issue_date)?,
                currency,
                asset,
                dates(knock_in_dates)?,
                autocall_schedule.inner.clone(),
                maturity_schedule.inner.clone(),
            )
            .map_err(error)?,
        })
    }
    #[getter]
    fn notional(&self) -> f64 {
        self.inner.notional()
    }
    #[getter]
    fn reference_spot(&self) -> f64 {
        self.inner.reference_spot()
    }
    #[getter]
    fn coupon_rate(&self) -> f64 {
        self.inner.coupon_rate()
    }
    #[getter]
    fn knock_in_barrier(&self) -> f64 {
        self.inner.knock_in_barrier()
    }
    #[getter]
    fn knock_out_barrier(&self) -> f64 {
        self.inner.knock_out_barrier()
    }
    #[getter]
    fn issue_date(&self) -> String {
        self.inner.issue_date().to_string()
    }
    #[getter]
    fn currency(&self) -> String {
        self.inner.currency().to_owned()
    }
    #[getter]
    fn asset(&self) -> String {
        self.inner.asset().to_owned()
    }
    #[getter]
    fn knock_in_dates(&self) -> Vec<String> {
        strings(self.inner.knock_in_dates())
    }
    #[getter]
    fn autocall_schedule(&self) -> PySchedule {
        PySchedule {
            inner: self.inner.autocall_schedule().clone(),
        }
    }
    #[getter]
    fn maturity_schedule(&self) -> PySchedule {
        PySchedule {
            inner: self.inner.maturity_schedule().clone(),
        }
    }
    #[getter]
    fn contract_hash(&self) -> String {
        self.inner.contract_hash().to_owned()
    }
    fn cashflow_id(&self, kind: &str, fixing_date: &Bound<'_, PyAny>) -> PyResult<String> {
        self.inner
            .cashflow_id(cashflow_kind(kind)?, date(fixing_date)?)
            .map_err(error)
    }
}

#[pyclass(
    name = "DatedPhoenixNote",
    module = "fuzzy_enigma",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(super) struct PyPhoenix {
    pub inner: DatedPhoenixNote,
}

#[pymethods]
impl PyPhoenix {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (notional, reference_spot, coupon_per_period, coupon_barrier, knock_in_barrier, knock_out_barrier,
        issue_date, currency, asset, knock_in_dates, coupon_schedule, autocall_schedule, maturity_schedule, memory = false))]
    fn new(
        notional: f64,
        reference_spot: f64,
        coupon_per_period: f64,
        coupon_barrier: f64,
        knock_in_barrier: f64,
        knock_out_barrier: f64,
        issue_date: &Bound<'_, PyAny>,
        currency: String,
        asset: String,
        knock_in_dates: &Bound<'_, PyAny>,
        coupon_schedule: PyRef<'_, PySchedule>,
        autocall_schedule: PyRef<'_, PySchedule>,
        maturity_schedule: PyRef<'_, PySchedule>,
        memory: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: DatedPhoenixNote::new(
                notional,
                reference_spot,
                coupon_per_period,
                coupon_barrier,
                knock_in_barrier,
                knock_out_barrier,
                date(issue_date)?,
                currency,
                asset,
                dates(knock_in_dates)?,
                coupon_schedule.inner.clone(),
                autocall_schedule.inner.clone(),
                maturity_schedule.inner.clone(),
                memory,
            )
            .map_err(error)?,
        })
    }
    #[getter]
    fn notional(&self) -> f64 {
        self.inner.notional()
    }
    #[getter]
    fn reference_spot(&self) -> f64 {
        self.inner.reference_spot()
    }
    #[getter]
    fn coupon_per_period(&self) -> f64 {
        self.inner.coupon_per_period()
    }
    #[getter]
    fn coupon_barrier(&self) -> f64 {
        self.inner.coupon_barrier()
    }
    #[getter]
    fn knock_in_barrier(&self) -> f64 {
        self.inner.knock_in_barrier()
    }
    #[getter]
    fn knock_out_barrier(&self) -> f64 {
        self.inner.knock_out_barrier()
    }
    #[getter]
    fn memory(&self) -> bool {
        self.inner.memory()
    }
    #[getter]
    fn issue_date(&self) -> String {
        self.inner.issue_date().to_string()
    }
    #[getter]
    fn currency(&self) -> String {
        self.inner.currency().to_owned()
    }
    #[getter]
    fn asset(&self) -> String {
        self.inner.asset().to_owned()
    }
    #[getter]
    fn knock_in_dates(&self) -> Vec<String> {
        strings(self.inner.knock_in_dates())
    }
    #[getter]
    fn coupon_schedule(&self) -> PySchedule {
        PySchedule {
            inner: self.inner.coupon_schedule().clone(),
        }
    }
    #[getter]
    fn autocall_schedule(&self) -> PySchedule {
        PySchedule {
            inner: self.inner.autocall_schedule().clone(),
        }
    }
    #[getter]
    fn maturity_schedule(&self) -> PySchedule {
        PySchedule {
            inner: self.inner.maturity_schedule().clone(),
        }
    }
    #[getter]
    fn contract_hash(&self) -> String {
        self.inner.contract_hash().to_owned()
    }
    fn cashflow_id(&self, kind: &str, fixing_date: &Bound<'_, PyAny>) -> PyResult<String> {
        self.inner
            .cashflow_id(cashflow_kind(kind)?, date(fixing_date)?)
            .map_err(error)
    }
}

fn note(value: &Bound<'_, PyAny>) -> PyResult<DatedNote> {
    if let Ok(value) = value.extract::<PyRef<'_, PySnowball>>() {
        return Ok(DatedNote::Snowball(value.inner.clone()));
    }
    if let Ok(value) = value.extract::<PyRef<'_, PyPhoenix>>() {
        return Ok(DatedNote::Phoenix(value.inner.clone()));
    }
    Err(PyValueError::new_err(
        "contract must be DatedSnowballNote or DatedPhoenixNote",
    ))
}

#[pyclass(
    name = "ActualPayment",
    module = "fuzzy_enigma",
    frozen,
    from_py_object
)]
#[derive(Clone)]
struct PyPayment {
    inner: ActualPayment,
}

#[pymethods]
impl PyPayment {
    #[new]
    fn new(payment_id: String, payment_date: &Bound<'_, PyAny>, amount: f64) -> PyResult<Self> {
        if payment_id.trim().is_empty() || !amount.is_finite() || amount <= 0.0 {
            return Err(PyValueError::new_err(
                "payment ID must be nonempty and amount positive and finite",
            ));
        }
        Ok(Self {
            inner: ActualPayment {
                payment_id,
                payment_date: date(payment_date)?,
                amount,
            },
        })
    }
    #[getter]
    fn payment_id(&self) -> String {
        self.inner.payment_id.clone()
    }
    #[getter]
    fn payment_date(&self) -> String {
        self.inner.payment_date.to_string()
    }
    #[getter]
    fn amount(&self) -> f64 {
        self.inner.amount
    }
}

#[pyclass(
    name = "SettlementConfirmation",
    module = "fuzzy_enigma",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(super) struct PySettlement {
    pub inner: SettlementConfirmation,
}

#[pymethods]
impl PySettlement {
    #[new]
    #[pyo3(signature = (cashflow_id, as_of, payments = None, expected_payment_date = None))]
    fn new(
        cashflow_id: String,
        as_of: &Bound<'_, PyAny>,
        payments: Option<Vec<PyPayment>>,
        expected_payment_date: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if cashflow_id.trim().is_empty() {
            return Err(PyValueError::new_err("cashflow ID must be nonempty"));
        }
        Ok(Self {
            inner: SettlementConfirmation {
                cashflow_id,
                as_of: date(as_of)?,
                payments: payments
                    .unwrap_or_default()
                    .into_iter()
                    .map(|p| p.inner)
                    .collect(),
                expected_payment_date: expected_payment_date.map(date).transpose()?,
            },
        })
    }
    #[getter]
    fn cashflow_id(&self) -> String {
        self.inner.cashflow_id.clone()
    }
    #[getter]
    fn as_of(&self) -> String {
        self.inner.as_of.to_string()
    }
    #[getter]
    fn expected_payment_date(&self) -> Option<String> {
        self.inner.expected_payment_date.map(|d| d.to_string())
    }
    #[getter]
    fn payments(&self) -> Vec<PyPayment> {
        self.inner
            .payments
            .iter()
            .cloned()
            .map(|inner| PyPayment { inner })
            .collect()
    }
}

fn fixings(values: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Fixing>> {
    let Some(values) = values else {
        return Ok(Vec::new());
    };
    values
        .try_iter()?
        .map(|item| {
            let item = item?;
            let pair = item.extract::<(Bound<'_, PyAny>, f64)>()?;
            Ok(Fixing {
                date: date(&pair.0)?,
                spot: pair.1,
            })
        })
        .collect()
}

fn fixing(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Fixing>> {
    value
        .map(|value| {
            let (d, spot) = value.extract::<(Bound<'_, PyAny>, f64)>()?;
            Ok(Fixing {
                date: date(&d)?,
                spot,
            })
        })
        .transpose()
}

#[pyclass(name = "NoteState", module = "fuzzy_enigma", frozen, from_py_object)]
#[derive(Clone)]
pub(super) struct PyState {
    pub inner: NoteState,
}

fn ledger(py: Python<'_>, entries: &[Receivable]) -> PyResult<Vec<Py<PyDict>>> {
    entries
        .iter()
        .map(|entry| {
            let item = PyDict::new(py);
            item.set_item("id", &entry.cashflow.id)?;
            item.set_item("kind", format!("{:?}", entry.cashflow.kind).to_lowercase())?;
            item.set_item(
                "fixing_date",
                entry.cashflow.fixing_date.map(|d| d.to_string()),
            )?;
            item.set_item(
                "contractual_payment_date",
                entry.cashflow.payment_date.to_string(),
            )?;
            item.set_item(
                "expected_payment_date",
                entry.expected_payment_date.map(|d| d.to_string()),
            )?;
            item.set_item("amount", entry.cashflow.amount)?;
            item.set_item("paid_amount", entry.paid_amount)?;
            item.set_item("outstanding_amount", entry.outstanding_amount)?;
            item.set_item("currency", &entry.cashflow.currency)?;
            item.set_item("component_ids", &entry.component_ids)?;
            item.set_item(
                "confirmation_as_of",
                entry.confirmation_as_of.map(|d| d.to_string()),
            )?;
            let payments: PyResult<Vec<_>> = entry
                .payments
                .iter()
                .map(|payment| {
                    let item = PyDict::new(py);
                    item.set_item("payment_id", &payment.payment_id)?;
                    item.set_item("payment_date", payment.payment_date.to_string())?;
                    item.set_item("amount", payment.amount)?;
                    Ok(item.unbind())
                })
                .collect();
            item.set_item("payments", payments?)?;
            Ok(item.unbind())
        })
        .collect()
}

#[pymethods]
impl PyState {
    #[getter]
    fn cutoff(&self) -> PyCutoff {
        PyCutoff {
            inner: self.inner.cutoff(),
        }
    }
    #[getter]
    fn knocked_in(&self) -> bool {
        self.inner.knocked_in()
    }
    #[getter]
    fn first_knock_in_date(&self) -> Option<String> {
        self.inner.first_knock_in_date().map(|d| d.to_string())
    }
    #[getter]
    fn termination(&self) -> &'static str {
        match self.inner.termination() {
            None => "active",
            Some(Termination::Redeemed { .. }) => "redeemed",
            Some(Termination::Matured { .. }) => "matured",
        }
    }
    #[getter]
    fn memory_coupon_ids(&self) -> Vec<String> {
        self.inner.memory_coupon_ids().to_vec()
    }
    #[getter]
    fn expired_memory_coupon_ids(&self) -> Vec<String> {
        self.inner.expired_memory_coupon_ids().to_vec()
    }
    #[getter]
    fn contract_hash(&self) -> String {
        self.inner.contract_hash().to_owned()
    }
    #[getter]
    fn history_hash(&self) -> String {
        self.inner.history_hash().to_owned()
    }
    #[getter]
    fn settlement_hash(&self) -> String {
        self.inner.settlement_hash().to_owned()
    }
    #[getter]
    fn provenance(&self) -> &'static str {
        match self.inner.provenance() {
            StateProvenance::Actual => "actual",
            StateProvenance::FixingScenario => "fixing_scenario",
            StateProvenance::FrozenRoll => "frozen_roll",
        }
    }
    #[getter]
    fn fixings(&self) -> Vec<(String, f64)> {
        self.inner
            .fixings()
            .iter()
            .map(|f| (f.date.to_string(), f.spot))
            .collect()
    }
    #[getter]
    fn scenario_fixings(&self) -> Vec<(String, f64)> {
        self.inner
            .scenario_fixings()
            .iter()
            .map(|f| (f.date.to_string(), f.spot))
            .collect()
    }
    #[getter]
    fn same_day_fixing_scenario(&self) -> Option<(String, f64)> {
        self.inner
            .same_day_fixing_scenario()
            .map(|f| (f.date.to_string(), f.spot))
    }
    #[getter]
    fn ledger(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        ledger(py, self.inner.cashflows())
    }
    #[getter]
    fn known_receivables(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        ledger(py, self.inner.receivables())
    }
    fn __repr__(&self) -> String {
        format!(
            "NoteState({}, {}, {})",
            self.cutoff().as_of(),
            self.termination(),
            self.provenance()
        )
    }
}

#[pyfunction(name = "replay_note_history")]
#[pyo3(signature = (contract, fixings, cutoff, *, settlements = None, same_day_fixing_scenario = None))]
fn replay(
    py: Python<'_>,
    contract: &Bound<'_, PyAny>,
    fixings: &Bound<'_, PyAny>,
    cutoff: PyRef<'_, PyCutoff>,
    settlements: Option<Vec<PySettlement>>,
    same_day_fixing_scenario: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyState> {
    let contract = note(contract)?;
    let fixings = self::fixings(Some(fixings))?;
    let settlements: Vec<_> = settlements
        .unwrap_or_default()
        .into_iter()
        .map(|s| s.inner)
        .collect();
    let scenario = fixing(same_day_fixing_scenario)?;
    let cutoff = cutoff.inner;
    let inner = py
        .detach(move || replay_note_history(&contract, &fixings, cutoff, &settlements, scenario))
        .map_err(error)?;
    Ok(PyState { inner })
}

#[pyfunction(name = "advance_note_state")]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (contract, state, fixings, cutoff, *, settlements = None, same_day_fixing_scenario = None))]
fn advance(
    py: Python<'_>,
    contract: &Bound<'_, PyAny>,
    state: PyRef<'_, PyState>,
    fixings: &Bound<'_, PyAny>,
    cutoff: PyRef<'_, PyCutoff>,
    settlements: Option<Vec<PySettlement>>,
    same_day_fixing_scenario: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyState> {
    let contract = note(contract)?;
    let state = state.inner.clone();
    let fixings = self::fixings(Some(fixings))?;
    let settlements: Vec<_> = settlements
        .unwrap_or_default()
        .into_iter()
        .map(|s| s.inner)
        .collect();
    let scenario = fixing(same_day_fixing_scenario)?;
    let cutoff = cutoff.inner;
    let inner = py
        .detach(move || {
            advance_note_state(&contract, &state, &fixings, cutoff, &settlements, scenario)
        })
        .map_err(error)?;
    Ok(PyState { inner })
}

fn nodes(input_dates: &Bound<'_, PyAny>, values: Vec<f64>) -> PyResult<Vec<CurveNode>> {
    let dates = dates(input_dates)?;
    if dates.len() != values.len() {
        return Err(PyValueError::new_err(
            "curve dates and values must have equal length",
        ));
    }
    Ok(dates
        .into_iter()
        .zip(values)
        .map(|(date, value)| CurveNode { date, value })
        .collect())
}

#[pyclass(
    name = "DiscountCurve",
    module = "fuzzy_enigma",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(super) struct PyDiscount {
    inner: DiscountCurve,
}

#[pymethods]
impl PyDiscount {
    #[new]
    fn new(
        as_of: &Bound<'_, PyAny>,
        currency: String,
        dates: &Bound<'_, PyAny>,
        values: Vec<f64>,
        source: String,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: DiscountCurve::try_new(date(as_of)?, currency, nodes(dates, values)?, source)
                .map_err(error)?,
        })
    }
    #[getter]
    fn as_of(&self) -> String {
        self.inner.as_of().to_string()
    }
    #[getter]
    fn currency(&self) -> String {
        self.inner.currency().to_owned()
    }
    #[getter]
    fn source(&self) -> String {
        self.inner.source().to_owned()
    }
    #[getter]
    fn dates(&self) -> Vec<String> {
        self.inner
            .nodes()
            .iter()
            .map(|n| n.date.to_string())
            .collect()
    }
    #[getter]
    fn values(&self) -> Vec<f64> {
        self.inner.nodes().iter().map(|n| n.value).collect()
    }
    #[getter]
    fn identity(&self) -> String {
        self.inner.identity()
    }
    fn value(&self, payment_date: &Bound<'_, PyAny>) -> PyResult<f64> {
        self.inner.value_on(date(payment_date)?).map_err(error)
    }
    fn value_at_time(&self, t: f64) -> PyResult<f64> {
        self.inner.value(t).map_err(error)
    }
}

#[pyclass(name = "ForwardCurve", module = "fuzzy_enigma", frozen, from_py_object)]
#[derive(Clone)]
pub(super) struct PyForward {
    inner: ForwardCurve,
}

#[pymethods]
impl PyForward {
    #[new]
    fn new(
        as_of: &Bound<'_, PyAny>,
        currency: String,
        asset: String,
        dates: &Bound<'_, PyAny>,
        values: Vec<f64>,
        source: String,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ForwardCurve::try_new(
                date(as_of)?,
                currency,
                asset,
                nodes(dates, values)?,
                source,
            )
            .map_err(error)?,
        })
    }
    #[getter]
    fn as_of(&self) -> String {
        self.inner.as_of().to_string()
    }
    #[getter]
    fn currency(&self) -> String {
        self.inner.currency().to_owned()
    }
    #[getter]
    fn asset(&self) -> String {
        self.inner.asset().to_owned()
    }
    #[getter]
    fn source(&self) -> String {
        self.inner.source().to_owned()
    }
    #[getter]
    fn dates(&self) -> Vec<String> {
        self.inner
            .nodes()
            .iter()
            .map(|n| n.date.to_string())
            .collect()
    }
    #[getter]
    fn values(&self) -> Vec<f64> {
        self.inner.nodes().iter().map(|n| n.value).collect()
    }
    #[getter]
    fn identity(&self) -> String {
        self.inner.identity()
    }
    fn value(&self, fixing_date: &Bound<'_, PyAny>) -> PyResult<f64> {
        self.inner.value_on(date(fixing_date)?).map_err(error)
    }
    fn value_at_time(&self, t: f64) -> PyResult<f64> {
        self.inner.value(t).map_err(error)
    }
}

#[pyclass(name = "TimeGrid", module = "fuzzy_enigma", frozen, from_py_object)]
#[derive(Clone)]
pub(super) struct PyGrid {
    pub inner: TimeGrid,
}

#[pymethods]
impl PyGrid {
    #[new]
    #[pyo3(signature = (as_of, fixing_dates, max_step_days = 1.0))]
    fn new(
        as_of: &Bound<'_, PyAny>,
        fixing_dates: &Bound<'_, PyAny>,
        max_step_days: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: TimeGrid::try_new(date(as_of)?, dates(fixing_dates)?, max_step_days)
                .map_err(error)?,
        })
    }
    #[getter]
    fn as_of(&self) -> String {
        self.inner.as_of().to_string()
    }
    #[getter]
    fn fixing_dates(&self) -> Vec<String> {
        strings(self.inner.fixing_dates())
    }
    #[getter]
    fn times(&self) -> Vec<f64> {
        self.inner.times().to_vec()
    }
    #[getter]
    fn steps(&self) -> usize {
        self.inner.steps()
    }
    #[getter]
    fn identity(&self) -> String {
        self.inner.identity()
    }
}

#[pyclass(
    name = "ValuationContext",
    module = "fuzzy_enigma",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(super) struct PyContext {
    pub inner: ValuationContext,
}

#[pymethods]
impl PyContext {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (cutoff, spot, currency, asset, discount_curve, forward_curve = None, source = ""))]
    fn new(
        cutoff: PyRef<'_, PyCutoff>,
        spot: f64,
        currency: String,
        asset: String,
        discount_curve: PyRef<'_, PyDiscount>,
        forward_curve: Option<PyRef<'_, PyForward>>,
        source: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ValuationContext::try_new(
                cutoff.inner,
                spot,
                currency,
                asset,
                discount_curve.inner.clone(),
                forward_curve.map(|v| v.inner.clone()),
                source.to_owned(),
            )
            .map_err(error)?,
        })
    }
    #[getter]
    fn cutoff(&self) -> PyCutoff {
        PyCutoff {
            inner: self.inner.cutoff,
        }
    }
    #[getter]
    fn spot(&self) -> f64 {
        self.inner.spot
    }
    #[getter]
    fn currency(&self) -> String {
        self.inner.currency.clone()
    }
    #[getter]
    fn asset(&self) -> String {
        self.inner.asset.clone()
    }
    #[getter]
    fn source(&self) -> String {
        self.inner.source.clone()
    }
    #[getter]
    fn discount_curve(&self) -> PyDiscount {
        PyDiscount {
            inner: self.inner.discount_curve.clone(),
        }
    }
    #[getter]
    fn forward_curve(&self) -> Option<PyForward> {
        self.inner
            .forward_curve
            .clone()
            .map(|inner| PyForward { inner })
    }
    #[getter]
    fn grid(&self) -> PyGrid {
        PyGrid {
            inner: self.inner.grid.clone(),
        }
    }
    #[getter]
    fn identity(&self) -> String {
        self.inner.identity()
    }
    fn with_grid(&self, grid: PyRef<'_, PyGrid>) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_grid(grid.inner.clone())
                .map_err(error)?,
        })
    }
    fn frozen_roll(&self, cutoff: PyRef<'_, PyCutoff>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.frozen_roll(cutoff.inner).map_err(error)?,
        })
    }
}

#[pyclass(name = "GbmDynamics", module = "fuzzy_enigma", frozen, from_py_object)]
#[derive(Clone)]
pub(super) struct PyGbmDynamics {
    inner: GbmDynamics,
}

#[pymethods]
impl PyGbmDynamics {
    #[new]
    fn new(volatility: f64) -> PyResult<Self> {
        Ok(Self {
            inner: GbmDynamics::try_new(volatility).map_err(error)?,
        })
    }
    #[getter]
    fn volatility(&self) -> f64 {
        self.inner.volatility
    }
}

#[pyclass(
    name = "HestonDynamics",
    module = "fuzzy_enigma",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(super) struct PyHestonDynamics {
    inner: HestonDynamics,
}

#[pymethods]
impl PyHestonDynamics {
    #[new]
    fn new(v0: f64, kappa: f64, theta: f64, xi: f64, rho: f64) -> PyResult<Self> {
        Ok(Self {
            inner: HestonDynamics::try_new(v0, kappa, theta, xi, rho).map_err(error)?,
        })
    }
    #[getter]
    fn v0(&self) -> f64 {
        self.inner.initial_variance
    }
    #[getter]
    fn kappa(&self) -> f64 {
        self.inner.kappa
    }
    #[getter]
    fn theta(&self) -> f64 {
        self.inner.theta
    }
    #[getter]
    fn xi(&self) -> f64 {
        self.inner.vol_of_vol
    }
    #[getter]
    fn rho(&self) -> f64 {
        self.inner.rho
    }
}

fn dynamics(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Dynamics>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if let Ok(value) = value.extract::<PyRef<'_, PyGbmDynamics>>() {
        return Ok(Some(Dynamics::Gbm(value.inner)));
    }
    if let Ok(value) = value.extract::<PyRef<'_, PyHestonDynamics>>() {
        return Ok(Some(Dynamics::Heston(value.inner)));
    }
    Err(PyValueError::new_err(
        "dated pricing requires GbmDynamics or HestonDynamics; legacy models are not accepted",
    ))
}

fn expected_cashflows(py: Python<'_>, values: &[CashflowEstimate]) -> PyResult<Vec<Py<PyDict>>> {
    values
        .iter()
        .map(|value| {
            let item = PyDict::new(py);
            item.set_item("id", &value.id)?;
            item.set_item("fixing_date", value.fixing_date.map(|d| d.to_string()))?;
            item.set_item("payment_date", value.payment_date.to_string())?;
            item.set_item("kind", format!("{:?}", value.kind).to_lowercase())?;
            item.set_item("currency", &value.currency)?;
            item.set_item("expected_amount", value.expected_amount)?;
            item.set_item("amount_std_error", value.amount_std_error)?;
            item.set_item("present_value", value.present_value)?;
            item.set_item("pv_std_error", value.pv_std_error)?;
            Ok(item.unbind())
        })
        .collect()
}

#[pyclass(
    name = "DatedPriceResult",
    module = "fuzzy_enigma",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(super) struct PyDatedResult {
    inner: DatedPriceResult,
    state: NoteState,
}

#[pymethods]
impl PyDatedResult {
    #[getter]
    fn price(&self) -> f64 {
        self.inner.estimate.price
    }
    #[getter]
    fn std_error(&self) -> f64 {
        self.inner.estimate.std_error
    }
    #[getter]
    fn samples(&self) -> usize {
        self.inner.estimate.samples
    }
    #[getter]
    fn paths(&self) -> usize {
        self.inner.paths
    }
    #[getter]
    fn method(&self) -> &'static str {
        match self.inner.method {
            DatedPricingMethod::Deterministic => "deterministic",
            DatedPricingMethod::MonteCarlo => "mc",
        }
    }
    #[getter]
    fn state(&self) -> PyState {
        PyState {
            inner: self.state.clone(),
        }
    }
    #[getter]
    fn cutoff(&self) -> PyCutoff {
        PyCutoff {
            inner: self.inner.valuation_cutoff,
        }
    }
    #[getter]
    fn known_receivables(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        ledger(py, self.state.receivables())
    }
    #[getter]
    fn expected_cashflows(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        expected_cashflows(py, &self.inner.expected_cashflows)
    }
    #[getter]
    fn context_identity(&self) -> String {
        self.inner.context_identity.clone()
    }
    #[getter]
    fn dynamics_identity(&self) -> Option<String> {
        self.inner.dynamics_identity.clone()
    }
    #[getter]
    fn grid_identity(&self) -> String {
        self.inner.grid_identity.clone()
    }
    fn confidence_95(&self) -> (f64, f64) {
        self.inner.estimate.confidence_95()
    }
    fn __repr__(&self) -> String {
        format!(
            "DatedPriceResult(price={:.6}, std_error={:.6}, method={})",
            self.price(),
            self.std_error(),
            self.method()
        )
    }
}

fn attach_grid(
    context: ValuationContext,
    payoff: &PreparedNote,
    grid: Option<TimeGrid>,
) -> PyResult<ValuationContext> {
    let grid = match grid {
        Some(grid) => grid,
        None if payoff.future_fixing_dates().is_empty() => {
            TimeGrid::try_new(context.as_of(), Vec::new(), 1.0).map_err(error)?
        }
        None if context.grid.steps() > 0 => context.grid.clone(),
        None => {
            TimeGrid::try_new(context.as_of(), payoff.future_fixing_dates(), 1.0).map_err(error)?
        }
    };
    context.with_grid(grid).map_err(error)
}

pub(super) fn price(
    py: Python<'_>,
    engine: &McEngine,
    contract: DatedNote,
    context: &PyContext,
    dynamics: Option<&Bound<'_, PyAny>>,
    state: &PyState,
    grid: Option<&PyGrid>,
) -> PyResult<PyDatedResult> {
    let payoff = PreparedNote::new(contract, state.inner.clone()).map_err(error)?;
    let context = attach_grid(
        context.inner.clone(),
        &payoff,
        grid.map(|g| g.inner.clone()),
    )?;
    let dynamics = self::dynamics(dynamics)?;
    let engine = engine.clone();
    let inner = py
        .detach(move || engine.try_price_context(&context, dynamics.as_ref(), &payoff))
        .map_err(error)?;
    Ok(PyDatedResult {
        inner,
        state: state.inner.clone(),
    })
}

fn cashflow_kind(kind: &str) -> PyResult<CashflowKind> {
    match kind {
        "coupon" => Ok(CashflowKind::Coupon),
        "principal" => Ok(CashflowKind::Principal),
        _ => Err(PyValueError::new_err(
            "cashflow kind must be coupon or principal",
        )),
    }
}

#[pyclass(
    name = "CalendarThetaResult",
    module = "fuzzy_enigma",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(super) struct PyCalendarResult {
    inner: CalendarThetaResult,
    base_state: NoteState,
    rolled_state: NoteState,
}

#[pymethods]
impl PyCalendarResult {
    #[getter]
    fn base_price(&self) -> f64 {
        self.inner.base.estimate.price
    }
    #[getter]
    fn rolled_price(&self) -> f64 {
        self.inner.rolled.estimate.price
    }
    #[getter]
    fn pv_change(&self) -> f64 {
        self.inner.pv_change
    }
    #[getter]
    fn std_error(&self) -> f64 {
        self.inner.std_error
    }
    #[getter]
    fn theta_per_day(&self) -> f64 {
        self.inner.per_day
    }
    #[getter]
    fn theta_per_year(&self) -> f64 {
        self.inner.per_day * 365.0
    }
    #[getter]
    fn theta_per_day_std_error(&self) -> f64 {
        self.inner.per_day_std_error
    }
    #[getter]
    fn theta_per_year_std_error(&self) -> f64 {
        self.inner.per_day_std_error * 365.0
    }
    #[getter]
    fn roll_days(&self) -> i64 {
        self.inner.roll_days
    }
    #[getter]
    fn samples(&self) -> usize {
        self.inner.samples
    }
    #[getter]
    fn cash_paid(&self) -> f64 {
        self.inner.cash_paid
    }
    #[getter]
    fn cash_adjusted_change(&self) -> f64 {
        self.inner.base_date_discounted_change
    }
    #[getter]
    fn cash_adjusted_std_error(&self) -> f64 {
        self.inner.discounted_change_std_error
    }
    #[getter]
    fn covariance(&self) -> f64 {
        self.inner.covariance
    }
    #[getter]
    fn union_times(&self) -> Vec<f64> {
        self.inner.union_times.clone()
    }
    #[getter]
    fn coupling(&self) -> String {
        self.inner.coupling.clone()
    }
    #[getter]
    fn rolled_state(&self) -> PyState {
        PyState {
            inner: self.rolled_state.clone(),
        }
    }
    #[getter]
    fn base_result(&self) -> PyDatedResult {
        PyDatedResult {
            inner: self.inner.base.clone(),
            state: self.base_state.clone(),
        }
    }
    #[getter]
    fn rolled_result(&self) -> PyDatedResult {
        PyDatedResult {
            inner: self.inner.rolled.clone(),
            state: self.rolled_state.clone(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn calendar(
    py: Python<'_>,
    engine: &McEngine,
    contract: &Bound<'_, PyAny>,
    context: &PyContext,
    dynamics: Option<&Bound<'_, PyAny>>,
    state: &PyState,
    roll_date: Option<&Bound<'_, PyAny>>,
    fixings: Option<&Bound<'_, PyAny>>,
    settlements: Option<Vec<PySettlement>>,
    same_day_fixing_scenario: Option<&Bound<'_, PyAny>>,
    grid: Option<&PyGrid>,
    rolled_grid: Option<&PyGrid>,
) -> PyResult<PyCalendarResult> {
    let contract = note(contract)?;
    let dynamics = self::dynamics(dynamics)?;
    let to = match roll_date {
        Some(value) => date(value)?,
        None => context
            .inner
            .as_of()
            .succ_opt()
            .ok_or_else(|| PyValueError::new_err("roll date overflows calendar range"))?,
    };
    let cutoff = ValuationCutoff {
        as_of: to,
        phase: CutoffPhase::AfterFixing,
    };
    let updates: Vec<_> = settlements
        .unwrap_or_default()
        .into_iter()
        .map(|s| s.inner)
        .collect();
    let base_state = state.inner.clone();
    let rolled_state = roll_note_state(
        &contract,
        &base_state,
        &self::fixings(fixings)?,
        cutoff,
        &updates,
        fixing(same_day_fixing_scenario)?,
    )
    .map_err(error)?;
    let base_payoff = PreparedNote::new(contract.clone(), base_state.clone()).map_err(error)?;
    let rolled_payoff = PreparedNote::new(contract, rolled_state.clone()).map_err(error)?;
    let base_context = attach_grid(
        context.inner.clone(),
        &base_payoff,
        grid.map(|g| g.inner.clone()),
    )?;
    let rolled_context = base_context.frozen_roll(cutoff).map_err(error)?;
    let rolled_context = attach_grid(
        rolled_context,
        &rolled_payoff,
        rolled_grid.map(|g| g.inner.clone()),
    )?;
    // Only newly confirmed actual receipts belong in roll cash P&L.
    let old_ids: std::collections::BTreeSet<_> = base_state
        .cashflows()
        .iter()
        .flat_map(|cf| cf.payments.iter().map(|p| p.payment_id.as_str()))
        .collect();
    let mut paid = Vec::new();
    for entry in rolled_state.cashflows() {
        for receipt in &entry.payments {
            if old_ids.contains(receipt.payment_id.as_str()) {
                continue;
            }
            if receipt.payment_date <= base_context.as_of() || receipt.payment_date > to {
                return Err(PyValueError::new_err("new calendar-roll receipts must fall in (base date, roll date]; replay corrected history separately"));
            }
            paid.push(Cashflow {
                id: receipt.payment_id.clone(),
                fixing_date: entry.cashflow.fixing_date,
                payment_date: receipt.payment_date,
                amount: receipt.amount,
                kind: entry.cashflow.kind,
                currency: entry.cashflow.currency.clone(),
            });
        }
    }
    let engine = engine.clone();
    let inner = py
        .detach(move || {
            try_calendar_theta(
                &engine,
                &base_context,
                &base_payoff,
                &rolled_context,
                &rolled_payoff,
                dynamics.as_ref(),
                &paid,
            )
        })
        .map_err(error)?;
    Ok(PyCalendarResult {
        inner,
        base_state,
        rolled_state,
    })
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCutoff>()?;
    m.add_class::<PySchedule>()?;
    m.add_class::<PySnowball>()?;
    m.add_class::<PyPhoenix>()?;
    m.add_class::<PyPayment>()?;
    m.add_class::<PySettlement>()?;
    m.add_class::<PyState>()?;
    m.add_class::<PyDiscount>()?;
    m.add_class::<PyForward>()?;
    m.add_class::<PyGrid>()?;
    m.add_class::<PyContext>()?;
    m.add_class::<PyGbmDynamics>()?;
    m.add_class::<PyHestonDynamics>()?;
    m.add_class::<PyDatedResult>()?;
    m.add_class::<PyCalendarResult>()?;
    m.add_function(wrap_pyfunction!(replay, m)?)?;
    m.add_function(wrap_pyfunction!(advance, m)?)?;
    Ok(())
}
