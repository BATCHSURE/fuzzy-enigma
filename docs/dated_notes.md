# Dated curves and outstanding notes

The dated API values single-underlying Snowball and Phoenix teaching contracts
after issuance. It keeps original terms, replays supplied historical fixings,
and discounts each remaining payment on its settlement date. GBM and Heston
dynamics are supported. Existing uniform-grid products and their Python calls
keep their original conventions.

The [outstanding-note notebook](../notebooks/outstanding_notes.ipynb) uses
committed, credential-free LSEG spot/carry snapshots with **synthetic note
terms, fixings and settlement history**. It is not a valuation of an observed
issued trade. The bridge does not request new data or calibrate a model.

## Curves, dates and dynamics

`DiscountCurve` and `ForwardCurve` accept an as-of date, currency, dated nodes,
and source text. A forward curve also identifies the underlying asset.
Discount factors and forwards must be positive and finite; `D > 1` is valid
for negative rates. The curves require the origin anchors `D(0) = 1` and
`F(0) = spot` and use log-linear interpolation within their supplied range.
They reject extrapolation. The context checks date, currency, asset and spot
consistency.

```python
import fuzzy_enigma as fe
from tools.market_validation import context_from_snapshot

grid = fe.TimeGrid("2026-10-08", ["2026-12-31", "2027-03-31"], max_step_days=7)
context, audit = context_from_snapshot(
    "tests/fixtures/market/lseg_spxw_20261008_surface.json",
    currency="USD", time_grid=grid,
)
gbm = fe.GbmDynamics(volatility=0.20)
heston = fe.HestonDynamics(v0=0.04, kappa=1.5, theta=0.04, xi=0.4, rho=-0.7)
```

Dates are Gregorian dates, supplied as exact `YYYY-MM-DD` strings or Python
`datetime.date` values. Intraday `datetime.datetime` inputs are rejected.
ACT/365F converts actual day differences to simulation and discount times,
including leap days. All future contractual fixing dates belong to the
`TimeGrid`; numerical refinement retains them. Same-day economic events share
one grid point and remain separate events. Supplied schedules carry a calendar
source; automatic holiday or business-day adjustment is outside this API.

The log-stock carry increment is `log(F(t_next) / F(t))`. GBM subtracts
`0.5 * sigma^2 * dt` and adds its normal shock. Heston uses the positive part
of the raw variance in log-stock drift/diffusion and full-truncation Euler
variance dynamics. Current Heston parameters, including `v0`, are explicit
valuation inputs; historical stock fixings cannot reconstruct current variance.

Curve support must reach every simulated fixing and every outstanding payment.
A payment lag can extend beyond the final stock fixing without extending the
stock simulation. A known unpaid receivable needs only a discount curve;
fully settled contracts have zero remaining PV and require no simulation.

## Teaching terms and event order

The initial reference fixing is a contract term, independent of current spot.
Barriers are absolute prices. Knock-in is observed on its explicit contractual
dates. The issue date is the mandatory first knock-in observation and uses the
original reference fixing. Extra numerical grid points do not introduce
additional knock-in observations.

| Event | Rule |
|---|---|
| Knock-in | `spot <= knock_in_barrier` sets a permanent historical flag |
| Phoenix coupon | `spot >= coupon_barrier` qualifies; eligible memory is paid with the current coupon |
| Autocall | `spot >= knock_out_barrier` redeems the note |
| Same-day order | Knock-in, coupon eligibility/memory, autocall or maturity principal, then settlement |
| After redemption | Stop economic observations; retain existing payment obligations |

Snowball early redemption returns principal and accrues its annual coupon from
the **original issue date** to redemption. At maturity without knock-in it
returns principal plus that accrued coupon. After knock-in without redemption,
principal is `notional * min(final_spot / reference_spot, 1)` with no coupon.
Accrual uses ACT/365F.

Phoenix coupons are cash amounts per observation. Missed coupons accumulate
only when memory is enabled. An eligible coupon pays the current period plus
remembered periods; unpaid memory expires at maturity or redemption. Principal
is independent of coupons: redemption returns principal; maturity returns
principal unless historical knock-in exposes the same capped downside ratio.
Eligible maturity coupons remain separate payments. Contractual payment dates
may differ from their fixing dates.

## History, cutoff and settlement

`ValuationCutoff(as_of, phase)` explicitly uses `before_fixing` or
`after_fixing`. Before fixing, that day's observation remains future; after
fixing, history must supply it. Context spot does not silently substitute for a
missing contractual fixing. The original reference fixing is already known
from the contract; history supplies other required observed prices. Replay
checks the contract, original reference, processed cutoff and observation set;
missing fixings or incompatible/stale states raise `ValueError`.

An explicit `same_day_fixing_scenario=(date, spot)` can condition a valuation-day
event. Under a before-fixing cutoff it processes that hypothetical fixing before
any future simulation, without changing the returned input state's before-fixing
cutoff or recording it as actual history. Conditional cashflows appear in the
valuation result's `expected_cashflows`; a scenario that redeems or matures the
note needs no random paths or dynamics. Under an after-fixing cutoff the same
argument can explicitly fill a missing current-day fixing, but cannot replace
an actual supplied fixing. State provenance identifies scenario inputs.
Hypothetical fixing outcomes remain separate from actual history and are
preserved by later frozen calendar rolls. They are never silently promoted to
actual observations: advancing a hypothetical state as actual history is
rejected. Once realised fixings are available, reconstruct confirmed state with
a full actual-history replay.

State separates historical knock-in, coupon memory, redemption and the cashflow
ledger. The ledger includes each obligation's stable identifier, component
identifiers, kind, fixing date, contractual payment date, expected payment date,
amount, paid amount, outstanding amount, currency and explicit payment records.
Coupon and principal obligations remain separate. Tables may sum them by
payment date for display while preserving their identifiers and components.

Passing a scheduled payment date does **not** mark an obligation paid. Supply
explicit settlement records, including partial payments. Replay rejects unknown
identifiers, duplicate payments, overpayments and inconsistent currencies.
Currency consistency applies to the contract and context; settlement records
inherit their identified obligation's currency rather than accepting another
currency field. Advancing a state preserves the complete immutable receipt
prefix. For an already-earned obligation, a newly added receipt must be after
the prior state's EOD date; historical corrections require a full replay.
Paid cash remains audit history and contributes no current PV. For an overdue
unpaid balance, explicitly supply an expected settlement date on or after the
valuation date; an explicitly declared same-day balance uses `D(0) = 1` without
being marked paid. The API rejects an absent or past expected date rather than
discounting at a negative time. On-time unpaid balances retain their contractual payment
date. Known balances are discounted once on their effective payment date.

Matured and redeemed notes can therefore retain value. A note with all amounts
settled returns exactly zero. Deterministic known receivables use no random
paths; their sampling error is zero.

## Calendar roll and theta

The existing Greeks method's `theta` remains schedule-scaled `dV/dT`.
`calendar_theta` instead advances the valuation date with fixed contract dates
and reports the roll interval, change in remaining PV, cash paid and total P&L.
It freezes spot and the supplied GBM/Heston dynamics. It rebases the same dated
curves:

```text
D_rolled(t) = D_old(delta + t) / D_old(delta)
F_rolled(t) = spot * F_old(delta + t) / F_old(delta)
```

No surface roll or recalibration is performed. Common random numbers make the
base and rolled Monte Carlo estimates comparable, but the reported sampling
error does not measure Heston discretisation or model uncertainty. An event-free
roll needs no invented prices. Crossing a contractual observation requires
supplied realised fixings or explicitly labelled scenario fixings. Explicit
settlements determine cash paid during the interval; the routine does not infer
settlement from the calendar. Paid cash reconciles separately from PV change.
`cash_adjusted_change` measures this reconciliation in the original valuation
numeraire: `D_old(delta) * rolled_PV + sum(D_old(receipt_date) * receipt_amount)
- base_PV`. A deterministic unchanged receivable has zero value change on this
basis even when its later-date PV rises as payment approaches.
Paired calendar estimates use a union of Brownian noise intervals while each
leg retains its own update grid. A leg's finite-sample price can therefore differ
from a standalone price call with the same seed; the paired change and its
sampling error are computed from those joint samples.

## Offline market bridge and provenance

`context_from_snapshot(snapshot_or_path, *, currency, time_grid=None,
phase="after_fixing")` returns `(context, audit)`. It verifies the existing
`dataset_sha256`, preserves captured spot and every positive-tenor D/F node,
and adds derived origin anchors. The caller must declare currency; recorded
currency metadata must agree. The curated curves do not independently certify
currency, so the audit labels this declaration.

Audit metadata preserves the parent dataset/capture hashes, dated source,
derived curve/context hashes, anchor construction, interpolation convention
and coverage. When a grid is supplied it attaches that pricing default and also
hashes the actual refined times and contractual fixing dates. The derived curve hash differs from the original
SSVI curve hash because it includes origin anchors and context conventions.
Hash checks detect changed inputs; they do not certify market correctness.

The committed captures are EOD Close observations. The bridge uses
`after_fixing` and rejects `before_fixing` for Close snapshots, because the
recorded Close cannot supply a prior intraday spot. Intraday expiry timestamps
remain unverified. The examples use explicitly dated teaching schedules rather
than claiming vendor settlement-time verification.

The bridge imports no LSEG SDK, opens no session, runs no quote inversion or
calibration, and writes no context/state artifact. A model's dynamics remain
separate inputs; a fitted European volatility surface does not define a
path-dependent note's dynamics. The optional bridge is repo tooling, while the
dated curve and note APIs are part of the native ABI-stable Python extension.

## Verification

```sh
python tests/test_dated.py
python -m unittest discover -s tests -p 'test_market_curves.py' -v
python tools/run_notebooks.py --notebook notebooks/outstanding_notes.ipynb --write
# Keep checked-in outputs untouched while saving a fresh execution separately:
python tools/run_notebooks.py --output-dir artifacts/dated-notebooks
```

The independent checks cover flat-curve compatibility, non-flat manual
cashflows, actual-day carry, schedule refinement, historical state, partial and
delayed settlement, deterministic receivables and calendar-roll reconciliation.
Offline bridge regressions replay both dated LSEG fixtures, verify exact nodes
and independent interpolation, check hashes and metadata, and reject unsupported
coverage. CI runs the dated native smoke test with Python 3.9 and 3.14 and loads
the same ABI-stable wheel on Python 3.9–3.14. The market/notebook job uses Python
3.12 and saved fixtures.

The completed phase 6 checks comprise 132 Rust tests, 80 offline market tests,
formatting and both all-targets Clippy configurations. One final ABI-stable wheel
passes all Python smoke scripts on 3.12 and 3.14. All six notebooks execute in
fresh kernels with 30 code cells and 16 PNG outputs; the outstanding-note example
has seven code cells and three visually checked charts.
