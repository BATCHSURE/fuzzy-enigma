//! A versioned, curve-bound smooth SSVI representation and Dupire evaluator.
//!
//! The explicitly declared shifted-phi regularisation is part of the model,
//! not a fallback applied after a failed derivative query. It gives a finite
//! analytic limit at the origin while retaining the SSVI sufficient static
//! arbitrage bounds. No strike clipping or maturity extrapolation is used.

use crate::dated::{
    canonical_identity, float_identity, year_fraction, ContextPricingError, Date, ValuationContext,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const ARTIFACT_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThetaNode {
    pub t: f64,
    pub theta: f64,
    pub tangent: f64,
}

/// Coefficients in normalised interval coordinate u=(t-t_i)/(t_{i+1}-t_i).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThetaPolynomial {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SurfaceCurveNode {
    pub date: String,
    pub discount_factor: f64,
    pub forward: f64,
    pub source: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoveragePoint {
    pub k: f64,
    pub t: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalVolArtifact {
    pub schema_version: u32,
    pub model: String,
    pub as_of: String,
    pub currency: String,
    pub asset: String,
    pub spot: f64,
    pub rho: f64,
    pub eta: f64,
    pub theta_floor: f64,
    pub epsilon: f64,
    pub min_denominator: f64,
    pub nodes: Vec<ThetaNode>,
    pub coefficients: Vec<ThetaPolynomial>,
    pub curves: Vec<SurfaceCurveNode>,
    pub coverage: Vec<CoveragePoint>,
    pub source_surface_sha256: String,
    pub source_dataset_sha256: String,
    pub source_curve_sha256: String,
    pub interpolation: String,
    pub short_time: String,
    pub wings: String,
    pub time_domain: String,
    pub fit_status: String,
    pub source_fit_status: String,
    pub artifact_sha256: String,
}

impl LocalVolArtifact {
    /// Content identity uses IEEE-754 bits and length-delimited strings, so
    /// Python and Rust do not depend on each other's JSON float formatting.
    pub fn content_identity(&self) -> String {
        let mut fields = vec![
            "local_vol_artifact_v1".into(),
            self.as_of.clone(),
            self.currency.clone(),
            self.asset.clone(),
        ];
        fields.extend(
            [
                self.spot,
                self.rho,
                self.eta,
                self.theta_floor,
                self.epsilon,
                self.min_denominator,
            ]
            .map(float_identity),
        );
        fields.extend([
            self.source_surface_sha256.clone(),
            self.source_dataset_sha256.clone(),
            self.source_curve_sha256.clone(),
            self.interpolation.clone(),
            self.short_time.clone(),
            self.wings.clone(),
            self.time_domain.clone(),
            self.fit_status.clone(),
            self.source_fit_status.clone(),
        ]);
        fields.push(self.nodes.len().to_string());
        for node in &self.nodes {
            fields.extend([node.t, node.theta, node.tangent].map(float_identity));
        }
        fields.push(self.coefficients.len().to_string());
        for c in &self.coefficients {
            fields.extend([c.a, c.b, c.c, c.d].map(float_identity));
        }
        fields.push(self.curves.len().to_string());
        for node in &self.curves {
            fields.extend([
                node.date.clone(),
                float_identity(node.discount_factor),
                float_identity(node.forward),
                node.source.clone(),
            ]);
        }
        fields.push(self.coverage.len().to_string());
        for point in &self.coverage {
            fields.extend([point.k, point.t].map(float_identity));
        }
        canonical_identity(&fields)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceDerivatives {
    pub theta: f64,
    pub theta_t: f64,
    pub w: f64,
    pub w_k: f64,
    pub w_kk: f64,
    pub w_t: f64,
    pub g: f64,
    pub local_variance: f64,
    pub short_time_extension: bool,
    pub wing_extension: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LocalVolModel {
    artifact: LocalVolArtifact,
    as_of: Date,
    curve_times: Vec<f64>,
    hull: Vec<(f64, f64)>,
}

fn invalid(reason: impl Into<String>) -> ContextPricingError {
    ContextPricingError::invalid("local_vol_surface", reason)
}
fn finite(x: f64) -> Result<(), ContextPricingError> {
    if x.is_finite() {
        Ok(())
    } else {
        Err(invalid("non-finite numeric input or derivative"))
    }
}
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 2e-12 * a.abs().max(b.abs()).max(1.0)
}
fn coefficient_close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-14_f64.max(1e-11 * a.abs().max(b.abs()))
}
fn date(s: &str) -> Result<Date, ContextPricingError> {
    let d = Date::parse_from_str(s, "%Y-%m-%d")
        .map_err(|_| invalid("dates must be canonical ISO date-only strings"))?;
    if d.format("%Y-%m-%d").to_string() != s {
        return Err(invalid("dates must be canonical ISO date-only strings"));
    }
    Ok(d)
}
fn hash_ok(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

/// Fritsch-Butland PCHIP tangents, matching SciPy PchipInterpolator.
pub fn pchip_tangents(times: &[f64], values: &[f64]) -> Result<Vec<f64>, ContextPricingError> {
    if times.len() != values.len() || times.len() < 2 {
        return Err(invalid("PCHIP needs matching nonempty intervals"));
    }
    let h: Vec<_> = times.windows(2).map(|p| p[1] - p[0]).collect();
    let delta: Vec<_> = values
        .windows(2)
        .zip(&h)
        .map(|(p, &h)| (p[1] - p[0]) / h)
        .collect();
    if times.iter().chain(values).any(|v| !v.is_finite())
        || h.iter().any(|&v| v <= 0.0)
        || delta.iter().any(|&v| v < 0.0)
    {
        return Err(invalid(
            "PCHIP needs finite increasing times and monotone variances",
        ));
    }
    if times.len() == 2 {
        return Ok(vec![delta[0], delta[0]]);
    }
    let n = times.len();
    let mut d = vec![0.0; n];
    fn endpoint(h0: f64, h1: f64, m0: f64, m1: f64) -> f64 {
        let d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
        if d * m0 <= 0.0 {
            0.0
        } else if m0 * m1 <= 0.0 && d.abs() > 3.0 * m0.abs() {
            3.0 * m0
        } else {
            d
        }
    }
    d[0] = endpoint(h[0], h[1], delta[0], delta[1]);
    d[n - 1] = endpoint(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
    for i in 1..n - 1 {
        if delta[i - 1] > 0.0 && delta[i] > 0.0 {
            let w1 = 2.0 * h[i] + h[i - 1];
            let w2 = h[i] + 2.0 * h[i - 1];
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
        }
    }
    Ok(d)
}

pub fn pchip_coefficients(nodes: &[ThetaNode]) -> Vec<ThetaPolynomial> {
    nodes
        .windows(2)
        .map(|n| {
            let h = n[1].t - n[0].t;
            let change = n[1].theta - n[0].theta;
            ThetaPolynomial {
                a: n[0].theta,
                b: h * n[0].tangent,
                c: 3.0 * change - h * (2.0 * n[0].tangent + n[1].tangent),
                d: -2.0 * change + h * (n[0].tangent + n[1].tangent),
            }
        })
        .collect()
}

impl LocalVolModel {
    pub fn from_json(json: &str) -> Result<Self, ContextPricingError> {
        let artifact: LocalVolArtifact = serde_json::from_str(json)
            .map_err(|e| invalid(format!("invalid versioned JSON: {e}")))?;
        Self::try_new(artifact)
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ContextPricingError> {
        Self::from_json(
            &std::fs::read_to_string(path)
                .map_err(|e| invalid(format!("cannot read artifact: {e}")))?,
        )
    }
    pub fn to_json(&self) -> Result<String, ContextPricingError> {
        serde_json::to_string_pretty(&self.artifact).map_err(|e| invalid(e.to_string()))
    }
    pub fn artifact(&self) -> &LocalVolArtifact {
        &self.artifact
    }
    pub fn identity(&self) -> String {
        self.artifact.artifact_sha256.clone()
    }
    pub fn as_of(&self) -> Date {
        self.as_of
    }
    pub fn maturity(&self) -> f64 {
        self.artifact.nodes.last().expect("validated nodes").t
    }
    pub fn first_quote_time(&self) -> f64 {
        self.artifact.nodes[1].t
    }
    pub fn extension_policy(&self) -> String {
        "global_shifted_phi; analytic_ssvi_all_real_k; bounded_no_long_extrapolation".into()
    }
    /// Prepare the additive ContextPathGenerator without extending the legacy
    /// Copy GBM/Heston Dynamics enum or changing its random-number streams.
    pub fn prepare(
        &self,
        context: &ValuationContext,
    ) -> Result<crate::dated::CurvePathGenerator, ContextPricingError> {
        crate::dated::CurvePathGenerator::try_new_local_vol(
            context,
            std::sync::Arc::new(self.clone()),
        )
    }

    pub fn try_new(a: LocalVolArtifact) -> Result<Self, ContextPricingError> {
        if a.schema_version != ARTIFACT_VERSION
            || a.model != "regularized_ssvi_local_vol"
            || a.interpolation != "pchip_theta_c1"
            || a.short_time != "global_shifted_phi"
            || a.wings != "analytic_ssvi_all_real_k"
            || a.time_domain != "bounded_no_long_extrapolation"
            || a.fit_status != "accepted"
            || !matches!(
                a.source_fit_status.as_str(),
                "converged" | "synthetic_reference"
            )
        {
            return Err(invalid(
                "unsupported schema/model/interpolation/extension/acceptance policy",
            ));
        }
        if ![
            &a.source_surface_sha256,
            &a.source_dataset_sha256,
            &a.source_curve_sha256,
            &a.artifact_sha256,
        ]
        .iter()
        .all(|s| hash_ok(s))
            || a.artifact_sha256 != a.content_identity()
        {
            return Err(invalid("missing or inconsistent source/content hashes"));
        }
        let as_of = date(&a.as_of)?;
        if a.currency.len() != 3
            || !a.currency.bytes().all(|c| c.is_ascii_uppercase())
            || a.asset.trim().is_empty()
        {
            return Err(invalid(
                "explicit ISO currency and nonempty asset are required",
            ));
        }
        for x in [
            a.spot,
            a.rho,
            a.eta,
            a.theta_floor,
            a.epsilon,
            a.min_denominator,
        ] {
            finite(x)?;
        }
        if a.spot <= 0.0
            || a.rho.abs() > 0.999
            || a.eta < 0.0
            || a.theta_floor <= 0.0
            || a.epsilon <= 0.0
            || a.epsilon >= 1.0
            || a.min_denominator <= 0.0
            || a.min_denominator >= 0.25
            || a.eta * (1.0 + a.rho.abs()) > 2.0 - a.epsilon
        {
            return Err(invalid(
                "invalid parameters or global static-arbitrage sufficient constraints",
            ));
        }
        if a.nodes.len() < 2 || a.coefficients.len() + 1 != a.nodes.len() {
            return Err(invalid(
                "origin and at least one expiry with matching coefficients are required",
            ));
        }
        if a.nodes[0].t != 0.0 || a.nodes[0].theta != 0.0 {
            return Err(invalid("origin must be t=theta=0"));
        }
        for (i, node) in a.nodes.iter().enumerate() {
            for x in [node.t, node.theta, node.tangent] {
                finite(x)?;
            }
            if node.t < 0.0
                || node.theta < 0.0
                || node.theta > 4.0
                || node.tangent < 0.0
                || (i > 0 && (node.t <= a.nodes[i - 1].t || node.theta < a.nodes[i - 1].theta))
            {
                return Err(invalid("invalid monotone total-variance nodes or tangents"));
            }
        }
        let tangents = pchip_tangents(
            &a.nodes.iter().map(|n| n.t).collect::<Vec<_>>(),
            &a.nodes.iter().map(|n| n.theta).collect::<Vec<_>>(),
        )?;
        if a.nodes
            .iter()
            .zip(tangents)
            .any(|(n, d)| !coefficient_close(n.tangent, d))
        {
            return Err(invalid(
                "recorded tangents differ from declared PCHIP construction",
            ));
        }
        for (supplied, expected) in a.coefficients.iter().zip(pchip_coefficients(&a.nodes)) {
            for (x, y) in [supplied.a, supplied.b, supplied.c, supplied.d]
                .into_iter()
                .zip([expected.a, expected.b, expected.c, expected.d])
            {
                finite(x)?;
                if !coefficient_close(x, y) {
                    return Err(invalid(
                        "recorded interpolation coefficients are inconsistent",
                    ));
                }
            }
        }
        if a.curves.len() < 2 {
            return Err(invalid("origin and future carry nodes are required"));
        }
        let mut curve_times = Vec::new();
        for (i, node) in a.curves.iter().enumerate() {
            let t = year_fraction(as_of, date(&node.date)?);
            finite(node.discount_factor)?;
            finite(node.forward)?;
            if node.source.trim().is_empty()
                || node.discount_factor <= 0.0
                || node.forward <= 0.0
                || t < 0.0
                || (i > 0 && t <= curve_times[i - 1])
            {
                return Err(invalid("invalid dated carry nodes"));
            }
            curve_times.push(t);
        }
        if curve_times[0] != 0.0
            || a.curves[0].discount_factor != 1.0
            || a.curves[0].forward != a.spot
            || *curve_times.last().unwrap() < a.nodes.last().unwrap().t
        {
            return Err(invalid(
                "missing carry origin, mismatched spot or insufficient carry coverage",
            ));
        }
        if a.coverage.is_empty() {
            return Err(invalid("observed quote coverage must be declared"));
        }
        for point in &a.coverage {
            finite(point.k)?;
            finite(point.t)?;
            if point.t < a.nodes[1].t || point.t > a.nodes.last().unwrap().t {
                return Err(invalid("quote coverage lies outside fitted time support"));
            }
        }
        let hull = convex_hull(a.coverage.iter().map(|p| (p.k, p.t)).collect());
        let model = Self {
            artifact: a,
            as_of,
            curve_times,
            hull,
        };
        // This scan is a numerical supplement; the global sufficient bounds
        // above, rather than a finite scan, justify the analytic wings.
        for pair in model.artifact.nodes.windows(2) {
            for t in [pair[0].t, 0.5 * (pair[0].t + pair[1].t), pair[1].t] {
                for k in [
                    -1000.0, -100.0, -10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0, 100.0, 1000.0,
                ] {
                    model.derivatives(k, t)?;
                }
            }
        }
        Ok(model)
    }

    pub fn theta(&self, t: f64) -> Result<(f64, f64), ContextPricingError> {
        finite(t)?;
        if t < 0.0 || t > self.maturity() {
            return Err(invalid(
                "time lies outside bounded support; maturity extrapolation is disabled",
            ));
        }
        let nodes = &self.artifact.nodes;
        if t == 0.0 {
            return Ok((nodes[0].theta, nodes[0].tangent));
        }
        if let Ok(i) = nodes.binary_search_by(|n| n.t.total_cmp(&t)) {
            return Ok((nodes[i].theta, nodes[i].tangent));
        }
        let i = nodes.partition_point(|n| n.t < t) - 1;
        let h = nodes[i + 1].t - nodes[i].t;
        let u = (t - nodes[i].t) / h;
        let c = &self.artifact.coefficients[i];
        let theta = ((c.d * u + c.c) * u + c.b) * u + c.a;
        // The equivalent Bernstein derivative avoids subtracting large,
        // almost equal polynomial terms next to a zero-slope plateau.
        let delta = (nodes[i + 1].theta - nodes[i].theta) / h;
        let left = nodes[i].tangent;
        let right = nodes[i + 1].tangent;
        let dt = (1.0 - u).powi(2) * left
            + 2.0 * u * (1.0 - u) * (3.0 * delta - left - right)
            + u * u * right;
        finite(theta)?;
        finite(dt)?;
        if theta < 0.0 || dt < 0.0 {
            return Err(invalid(
                "negative interpolated variance or calendar derivative",
            ));
        }
        Ok((theta, dt))
    }
    pub fn observed(&self, k: f64, t: f64) -> bool {
        inside_hull((k, t), &self.hull)
    }
    pub fn total_variance(&self, k: f64, t: f64) -> Result<f64, ContextPricingError> {
        Ok(self.derivatives(k, t)?.w)
    }
    pub fn local_variance(&self, k: f64, t: f64) -> Result<f64, ContextPricingError> {
        Ok(self.derivatives(k, t)?.local_variance)
    }
    pub fn derivatives(&self, k: f64, t: f64) -> Result<SurfaceDerivatives, ContextPricingError> {
        finite(k)?;
        let (theta, theta_t) = self.theta(t)?;
        let p = &self.artifact;
        let phi = p.eta / (theta + p.theta_floor).sqrt() / (1.0 + theta).sqrt();
        let phi_theta = -0.5 * phi * (1.0 / (theta + p.theta_floor) + 1.0 / (1.0 + theta));
        let x = phi * k;
        finite(x)?;
        let q = (x + p.rho).hypot((1.0 - p.rho * p.rho).sqrt());
        let a = 0.5 * (1.0 + p.rho * x + q);
        let ax = 0.5 * (p.rho + (x + p.rho) / q);
        let akk = 0.5 * (1.0 - p.rho * p.rho) * phi * phi / q.powi(3);
        let ak = ax * phi;
        let w = theta * a;
        let w_k = theta * ak;
        let w_kk = theta * akk;
        let w_t = (a + theta * ax * k * phi_theta) * theta_t;
        // Algebraically equivalent to the usual g expression, but remains
        // defined at theta=0 and avoids dividing two vanishing quantities.
        let g = (1.0 - k * ak / (2.0 * a)).powi(2)
            - theta * ak * ak / (4.0 * a)
            - theta * theta * ak * ak / 16.0
            + theta * akk / 2.0;
        for value in [phi, phi_theta, a, ax, akk, w, w_k, w_kk, w_t, g] {
            finite(value)?;
        }
        if a <= 0.0 || w_t < 0.0 || g <= p.min_denominator {
            return Err(invalid("negative calendar variance or unstable Dupire denominator; no clipping or fallback is applied"));
        }
        let local_variance = w_t / g;
        finite(local_variance)?;
        Ok(SurfaceDerivatives {
            theta,
            theta_t,
            w,
            w_k,
            w_kk,
            w_t,
            g,
            local_variance,
            short_time_extension: t < self.first_quote_time(),
            wing_extension: t >= self.first_quote_time() && !self.observed(k, t),
        })
    }

    fn curve_value(&self, t: f64, discount: bool) -> Result<f64, ContextPricingError> {
        if !t.is_finite() || t < 0.0 || t > *self.curve_times.last().unwrap() {
            return Err(invalid("missing bounded artifact carry coverage"));
        }
        let value = |i: usize| {
            if discount {
                self.artifact.curves[i].discount_factor
            } else {
                self.artifact.curves[i].forward
            }
        };
        if t == 0.0 {
            return Ok(value(0));
        }
        if let Ok(i) = self.curve_times.binary_search_by(|x| x.total_cmp(&t)) {
            return Ok(value(i));
        }
        let i = self.curve_times.partition_point(|&x| x < t) - 1;
        let weight = (t - self.curve_times[i]) / (self.curve_times[i + 1] - self.curve_times[i]);
        Ok(((1.0 - weight) * value(i).ln() + weight * value(i + 1).ln()).exp())
    }
    pub fn validate_context(&self, context: &ValuationContext) -> Result<(), ContextPricingError> {
        context.validate()?;
        if context.as_of() != self.as_of
            || context.currency != self.artifact.currency
            || context.asset != self.artifact.asset
            || context.spot != self.artifact.spot
        {
            return Err(invalid(
                "surface date/currency/asset/origin spot differs from valuation context",
            ));
        }
        let forward = context
            .forward_curve
            .as_ref()
            .ok_or_else(|| invalid("local volatility requires an explicit forward curve"))?;
        if context.grid.maturity() > self.maturity() {
            return Err(invalid("simulation grid exceeds surface maturity coverage"));
        }
        // Compare the union of breakpoints and simulation nodes. This checks
        // the complete log-linear carry trajectory, not merely end values.
        let mut times = self.curve_times.clone();
        times.extend(
            context
                .discount_curve
                .nodes()
                .iter()
                .map(|n| year_fraction(self.as_of, n.date))
                .filter(|&t| t <= *self.curve_times.last().unwrap()),
        );
        times.extend(
            forward
                .nodes()
                .iter()
                .map(|n| year_fraction(self.as_of, n.date))
                .filter(|&t| t <= *self.curve_times.last().unwrap()),
        );
        times.extend(context.grid.times());
        for t in times {
            if !close(context.discount_curve.value(t)?, self.curve_value(t, true)?)
                || !close(forward.value(t)?, self.curve_value(t, false)?)
            {
                return Err(invalid(
                    "surface carry differs from the supplied valuation curves",
                ));
            }
        }
        Ok(())
    }
}

fn cross(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}
fn convex_hull(mut p: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    p.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    p.dedup();
    if p.len() < 3 {
        return p;
    }
    let mut lo = Vec::new();
    for &v in &p {
        while lo.len() >= 2 && cross(lo[lo.len() - 2], lo[lo.len() - 1], v) <= 0.0 {
            lo.pop();
        }
        lo.push(v);
    }
    let mut hi = Vec::new();
    for &v in p.iter().rev() {
        while hi.len() >= 2 && cross(hi[hi.len() - 2], hi[hi.len() - 1], v) <= 0.0 {
            hi.pop();
        }
        hi.push(v);
    }
    lo.pop();
    hi.pop();
    lo.extend(hi);
    lo
}
fn inside_hull(p: (f64, f64), h: &[(f64, f64)]) -> bool {
    if h.len() == 1 {
        return (p.0 - h[0].0).abs() < 1e-12 && (p.1 - h[0].1).abs() < 1e-12;
    }
    if h.len() == 2 {
        return cross(h[0], h[1], p).abs() < 1e-12
            && p.0 >= h[0].0.min(h[1].0) - 1e-12
            && p.0 <= h[0].0.max(h[1].0) + 1e-12
            && p.1 >= h[0].1.min(h[1].1) - 1e-12
            && p.1 <= h[0].1.max(h[1].1) + 1e-12;
    }
    h.len() >= 3 && (0..h.len()).all(|i| cross(h[i], h[(i + 1) % h.len()], p) >= -1e-12)
}
