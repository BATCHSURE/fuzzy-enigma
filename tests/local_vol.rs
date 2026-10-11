use chrono::Duration;
use fuzzy_enigma::dated::*;
use fuzzy_enigma::local_vol::*;
use fuzzy_enigma::McEngine;

fn origin() -> Date {
    Date::from_ymd_opt(2026, 10, 8).unwrap()
}
fn day(n: i64) -> Date {
    origin() + Duration::days(n)
}

fn artifact(eta: f64) -> LocalVolArtifact {
    let times = vec![0.0, 90.0 / 365.0, 180.0 / 365.0, 1.0];
    let values: Vec<_> = times.iter().map(|&t| 0.04 * t).collect();
    let tangents = pchip_tangents(&times, &values).unwrap();
    let nodes: Vec<_> = times
        .iter()
        .zip(values)
        .zip(tangents)
        .map(|((&t, theta), tangent)| ThetaNode { t, theta, tangent })
        .collect();
    let coefficients = pchip_coefficients(&nodes);
    let mut a = LocalVolArtifact {
        schema_version: 1,
        model: "regularized_ssvi_local_vol".into(),
        as_of: origin().to_string(),
        currency: "USD".into(),
        asset: "TEST".into(),
        spot: 100.0,
        rho: -0.4,
        eta,
        theta_floor: 0.0001,
        epsilon: 1e-6,
        min_denominator: 1e-12,
        nodes,
        coefficients,
        curves: [0, 90, 180, 365]
            .into_iter()
            .map(|n| SurfaceCurveNode {
                date: day(n).to_string(),
                discount_factor: (-0.03 * n as f64 / 365.0).exp(),
                forward: 100.0 * (0.02 * n as f64 / 365.0).exp(),
                source: "synthetic independent flat carry".into(),
            })
            .collect(),
        coverage: times[1..]
            .iter()
            .flat_map(|&t| [-0.35, 0.35].map(|k| CoveragePoint { k, t }))
            .collect(),
        source_surface_sha256: "1".repeat(64),
        source_dataset_sha256: "2".repeat(64),
        source_curve_sha256: "3".repeat(64),
        interpolation: "pchip_theta_c1".into(),
        short_time: "global_shifted_phi".into(),
        wings: "analytic_ssvi_all_real_k".into(),
        time_domain: "bounded_no_long_extrapolation".into(),
        fit_status: "accepted".into(),
        source_fit_status: "synthetic_reference".into(),
        artifact_sha256: String::new(),
    };
    a.artifact_sha256 = a.content_identity();
    a
}
fn model(eta: f64) -> LocalVolModel {
    LocalVolModel::try_new(artifact(eta)).unwrap()
}
fn context(max_step: f64) -> ValuationContext {
    let a = artifact(0.0);
    let d = DiscountCurve::try_new(
        origin(),
        "USD".into(),
        a.curves
            .iter()
            .map(|n| CurveNode {
                date: Date::parse_from_str(&n.date, "%Y-%m-%d").unwrap(),
                value: n.discount_factor,
            })
            .collect(),
        "test carry".into(),
    )
    .unwrap();
    let f = ForwardCurve::try_new(
        origin(),
        "USD".into(),
        "TEST".into(),
        a.curves
            .iter()
            .map(|n| CurveNode {
                date: Date::parse_from_str(&n.date, "%Y-%m-%d").unwrap(),
                value: n.forward,
            })
            .collect(),
        "test carry".into(),
    )
    .unwrap();
    ValuationContext::try_new(
        ValuationCutoff {
            as_of: origin(),
            phase: CutoffPhase::AfterFixing,
        },
        100.0,
        "USD".into(),
        "TEST".into(),
        d,
        Some(f),
        "test".into(),
    )
    .unwrap()
    .with_grid(TimeGrid::try_new(origin(), vec![day(365)], max_step).unwrap())
    .unwrap()
}

#[derive(Clone)]
struct Terminal(Option<f64>);
impl CashflowPayoff for Terminal {
    fn validate(&self, c: &ValuationContext) -> Result<(), ContextPricingError> {
        c.grid.index(day(365))?;
        Ok(())
    }
    fn cashflows(
        &self,
        p: &[f64],
        c: &ValuationContext,
        out: &mut Vec<Cashflow>,
    ) -> Result<(), ContextPricingError> {
        let terminal = p[c.grid.index(day(365))?];
        out.push(Cashflow {
            id: "terminal".into(),
            fixing_date: Some(day(365)),
            payment_date: day(365),
            amount: self.0.map_or(terminal, |k| (terminal - k).max(0.0)),
            kind: CashflowKind::Other,
            currency: "USD".into(),
        });
        Ok(())
    }
    fn future_fixing_dates(&self) -> Vec<Date> {
        vec![day(365)]
    }
    fn payment_dates(&self) -> Vec<Date> {
        vec![day(365)]
    }
}

#[test]
fn flat_surface_recovers_constant_variance_including_origin_and_wings() {
    let m = model(0.0);
    for t in [-0.0, 0.0, 1e-12, 0.1, 90.0 / 365.0, 0.5, 1.0] {
        for k in [-1000.0, -2.0, 0.0, 2.0, 1000.0] {
            let d = m.derivatives(k, t).unwrap();
            assert!((d.w - 0.04 * t).abs() < 1e-14);
            assert!((d.local_variance - 0.04).abs() < 1e-14);
            assert_eq!(d.g, 1.0);
            assert_eq!(d.w_k, 0.0);
            assert_eq!(d.w_kk, 0.0);
        }
    }
}

#[test]
fn shifted_phi_has_finite_origin_limit_and_positive_global_wings() {
    let m = model(0.6);
    for k in [-1000.0, -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1000.0] {
        let zero = m.derivatives(k, 0.0).unwrap();
        assert!(zero.g >= 0.25 - 1e-14);
        assert!(zero.local_variance.is_finite() && zero.local_variance >= 0.0);
        let tiny = m.derivatives(k, 1e-12).unwrap();
        assert!(
            (zero.local_variance - tiny.local_variance).abs() < 1e-5 * zero.local_variance.max(1.0)
        );
        for t in [0.02, 0.3, 0.8, 1.0] {
            let d = m.derivatives(k, t).unwrap();
            assert!(d.g > 0.0 && d.w_t >= 0.0 && d.local_variance >= 0.0);
        }
    }
}

#[test]
fn analytic_derivatives_match_finite_differences_at_well_conditioned_points() {
    let m = model(0.6);
    let hk = 1e-5;
    let ht = 1e-6;
    for (k, t) in [(-0.25, 0.15), (0.0, 0.4), (0.12, 0.7)] {
        let d = m.derivatives(k, t).unwrap();
        let w = m.total_variance(k, t).unwrap();
        let left = m.total_variance(k - hk, t).unwrap();
        let right = m.total_variance(k + hk, t).unwrap();
        let dk = (right - left) / (2.0 * hk);
        let dkk = (right - 2.0 * w + left) / (hk * hk);
        let dt = (m.total_variance(k, t + ht).unwrap() - m.total_variance(k, t - ht).unwrap())
            / (2.0 * ht);
        assert!((dk - d.w_k).abs() < 1e-6 * d.w_k.abs().max(1e-4));
        assert!((dkk - d.w_kk).abs() < 1e-4 * d.w_kk.abs().max(1e-4));
        assert!((dt - d.w_t).abs() < 1e-6 * d.w_t.abs().max(1e-4));
    }
}

#[test]
fn pchip_is_c1_at_knots_and_records_nonuniform_polynomials() {
    let mut a = artifact(0.3);
    a.nodes[1].theta = 0.008;
    a.nodes[2].theta = 0.022;
    let tangents = pchip_tangents(
        &a.nodes.iter().map(|n| n.t).collect::<Vec<_>>(),
        &a.nodes.iter().map(|n| n.theta).collect::<Vec<_>>(),
    )
    .unwrap();
    for (n, d) in a.nodes.iter_mut().zip(tangents) {
        n.tangent = d;
    }
    a.coefficients = pchip_coefficients(&a.nodes);
    a.artifact_sha256 = a.content_identity();
    let m = LocalVolModel::try_new(a).unwrap();
    for knot in &m.artifact().nodes[1..3] {
        let left = m.derivatives(-0.1, knot.t - 1e-9).unwrap();
        let center = m.derivatives(-0.1, knot.t).unwrap();
        let right = m.derivatives(-0.1, knot.t + 1e-9).unwrap();
        assert!((left.w_t - right.w_t).abs() < 1e-7);
        assert!((center.theta_t - knot.tangent).abs() < 1e-14);
    }
}

#[test]
fn monotone_pchip_plateaus_have_zero_variance_without_roundoff_clipping() {
    let mut a = artifact(0.3);
    a.nodes[1].theta = 0.008;
    a.nodes[2].theta = 0.008;
    let times: Vec<_> = a.nodes.iter().map(|n| n.t).collect();
    let values: Vec<_> = a.nodes.iter().map(|n| n.theta).collect();
    for (node, slope) in a
        .nodes
        .iter_mut()
        .zip(pchip_tangents(&times, &values).unwrap())
    {
        node.tangent = slope;
    }
    a.coefficients = pchip_coefficients(&a.nodes);
    a.artifact_sha256 = a.content_identity();
    let m = LocalVolModel::try_new(a).unwrap();
    for epsilon in [1e-5, 1e-10, 1e-15] {
        assert!(
            m.derivatives(-0.1, times[1] - epsilon)
                .unwrap()
                .local_variance
                >= 0.0
        );
        assert!(
            m.derivatives(-0.1, times[2] + epsilon)
                .unwrap()
                .local_variance
                >= 0.0
        );
    }
    for t in [times[1], (times[1] + times[2]) / 2.0, times[2]] {
        let d = m.derivatives(0.1, t).unwrap();
        assert_eq!(d.theta, 0.008);
        assert_eq!(d.theta_t, 0.0);
        assert_eq!(d.local_variance, 0.0);
    }
}

#[test]
fn serialization_revalidates_hash_status_curves_and_interpolation() {
    let m = model(0.3);
    assert_eq!(LocalVolModel::from_json(&m.to_json().unwrap()).unwrap(), m);
    let mut a = artifact(0.3);
    a.eta = 2.0;
    assert!(LocalVolModel::try_new(a.clone()).is_err());
    a.artifact_sha256 = a.content_identity();
    assert!(LocalVolModel::try_new(a).is_err());
    let mut a = artifact(0.3);
    a.epsilon = 0.0;
    a.artifact_sha256 = a.content_identity();
    assert!(LocalVolModel::try_new(a).is_err());
    let mut a = artifact(0.3);
    a.min_denominator = 0.0;
    a.artifact_sha256 = a.content_identity();
    assert!(LocalVolModel::try_new(a).is_err());
    let mut a = artifact(0.3);
    a.fit_status = "incomplete".into();
    a.artifact_sha256 = a.content_identity();
    assert!(LocalVolModel::try_new(a).is_err());
    let mut a = artifact(0.3);
    a.coefficients[0].b += 1e-5;
    a.artifact_sha256 = a.content_identity();
    assert!(LocalVolModel::try_new(a).is_err());
    let mut a = artifact(0.3);
    a.curves[0].forward = 99.0;
    a.artifact_sha256 = a.content_identity();
    assert!(LocalVolModel::try_new(a).is_err());
    let mut value: serde_json::Value = serde_json::from_str(&m.to_json().unwrap()).unwrap();
    value["unknown_policy"] = true.into();
    assert!(LocalVolModel::from_json(&value.to_string()).is_err());
}

#[test]
fn boundary_queries_and_mismatched_context_fail_explicitly() {
    let m = model(0.3);
    for (k, t) in [
        (0.0, -1e-12),
        (0.0, 1.0 + 1e-12),
        (f64::NAN, 0.2),
        (0.0, f64::INFINITY),
    ] {
        assert!(m.derivatives(k, t).is_err());
    }
    let mut c = context(1.0);
    m.validate_context(&c).unwrap();
    c.spot = 99.0;
    assert!(m.validate_context(&c).is_err());
    let mut c = context(1.0);
    c.asset = "OTHER".into();
    assert!(m.validate_context(&c).is_err());
    let c = context(1.0)
        .frozen_roll(ValuationCutoff {
            as_of: day(1),
            phase: CutoffPhase::AfterFixing,
        })
        .unwrap();
    assert!(m.validate_context(&c).is_err());
}

#[test]
fn constant_local_vol_tracks_gbm_with_same_noise_and_forward_carry() {
    let c = context(7.0);
    let lv = model(0.0).prepare(&c).unwrap();
    let gbm = Dynamics::Gbm(GbmDynamics::try_new(0.2).unwrap())
        .prepare(&c)
        .unwrap();
    assert_eq!(lv.noise_dim(), gbm.noise_dim());
    let z: Vec<_> = (0..lv.noise_dim())
        .map(|i| (i as f64 * 1.7).sin())
        .collect();
    let mut p = vec![0.0; c.grid.steps() + 1];
    let mut q = p.clone();
    lv.generate(&z, &mut p).unwrap();
    gbm.generate(&z, &mut q).unwrap();
    for (&a, &b) in p.iter().zip(&q) {
        assert!((a - b).abs() < 1e-11);
    }
    assert_eq!(
        lv.path_diagnostics(&p).unwrap().unwrap().evaluations,
        c.grid.steps()
    );
}

#[test]
fn zero_variance_local_vol_tracks_forward_exactly() {
    let mut a = artifact(0.0);
    for n in &mut a.nodes {
        n.theta = 0.0;
        n.tangent = 0.0;
    }
    a.coefficients = pchip_coefficients(&a.nodes);
    a.artifact_sha256 = a.content_identity();
    let d = LocalVolModel::try_new(a).unwrap();
    let c = context(30.0);
    let g = d.prepare(&c).unwrap();
    let mut p = vec![0.0; c.grid.steps() + 1];
    g.generate(&vec![1.0; g.noise_dim()], &mut p).unwrap();
    for (&s, &t) in p.iter().zip(c.grid.times()) {
        assert!((s - c.forward_curve.as_ref().unwrap().value(t).unwrap()).abs() < 1e-10);
    }
}

#[test]
fn local_vol_engine_reproducible_diagnostics_count_antithetic_paths() {
    let d = model(0.4);
    let c = context(14.0);
    let e = McEngine::new(2000).with_seed(123).with_parallel(false);
    let a = e
        .try_price_local_vol_context(&c, &d, &Terminal(Some(100.0)))
        .unwrap();
    let b = e
        .with_parallel(true)
        .try_price_local_vol_context(&c, &d, &Terminal(Some(100.0)))
        .unwrap();
    assert_eq!(a, b);
    let stats = a.path_diagnostics.unwrap();
    assert_eq!(stats.evaluations, a.paths * c.grid.steps());
    assert_eq!(stats.paths_with_extensions, a.paths);
    assert!(
        stats.short_time_evaluations > 0
            && stats.outside_observed_evaluations >= stats.short_time_evaluations
    );
}

#[test]
fn nonconstant_local_vol_preserves_forward_expectation_with_declared_sampling_tolerance() {
    let d = model(0.4);
    let c = context(3.0);
    let e = McEngine::new(30_000).with_seed(345);
    let p = e
        .try_price_local_vol_context(&c, &d, &Terminal(None))
        .unwrap();
    let reference = c.forward_curve.as_ref().unwrap().value(1.0).unwrap()
        * c.discount_curve.value(1.0).unwrap();
    // Log-Euler's conditional martingale property gives this target at any
    // grid; the predeclared statistical criterion is four standard errors.
    assert!((p.estimate.price - reference).abs() < 4.0 * p.estimate.std_error);
}

#[test]
fn local_vol_rejects_undefined_rolled_surface_context_and_preserves_copy_dynamics() {
    let c = context(7.0);
    let rolled = c
        .frozen_roll(ValuationCutoff {
            as_of: day(1),
            phase: CutoffPhase::AfterFixing,
        })
        .unwrap();
    let d = model(0.3);
    let e = McEngine::new(100);
    let err = e
        .try_price_local_vol_context(&rolled, &d, &Terminal(None))
        .unwrap_err();
    assert!(err.to_string().contains("date/currency/asset"));
    fn requires_copy<T: Copy>(_: T) {}
    let gbm = Dynamics::Gbm(GbmDynamics::try_new(0.2).unwrap());
    requires_copy(gbm);
    requires_copy(gbm);
}
