//! Heston validation and independent numerical references.
use fuzzy_enigma::analytic::bs_call;
use fuzzy_enigma::model::{GbmModel, HestonModel, PathGenerator};
use fuzzy_enigma::{EuropeanControl, EuropeanOption, McEngine, OptionType, Payoff, PricingError};

fn model(steps: usize) -> HestonModel {
    HestonModel::new(100.0, 0.03, 0.01, 0.04, 1.5, 0.04, 0.4, -0.7, 1.0, steps)
}

fn call() -> EuropeanOption {
    EuropeanOption {
        option_type: OptionType::Call,
        strike: 100.0,
    }
}

#[test]
fn generic_model_contract_is_checked_before_simulation() {
    struct CustomModel {
        steps: usize,
        maturity: f64,
        rate: f64,
        noise: usize,
    }
    impl PathGenerator for CustomModel {
        fn steps(&self) -> usize {
            self.steps
        }
        fn noise_dim(&self) -> usize {
            self.noise
        }
        fn maturity(&self) -> f64 {
            self.maturity
        }
        fn risk_free_rate(&self) -> f64 {
            self.rate
        }
        fn generate(&self, _: &[f64], _: &mut [f64]) {
            panic!("invalid custom model must be rejected before simulation");
        }
    }
    for (steps, maturity, rate, noise) in [
        (0, 1.0, 0.03, 0),
        (1, 0.0, 0.03, 1),
        (1, f64::NAN, 0.03, 1),
        (1, 1.0, f64::INFINITY, 1),
        (1, 1.0, 0.03, usize::MAX),
        (usize::MAX, 1.0, 0.03, 1),
    ] {
        let model = CustomModel {
            steps,
            maturity,
            rate,
            noise,
        };
        assert!(McEngine::new(1).try_price(&model, &call()).is_err());
    }
}

#[test]
fn gbm_preserves_the_original_chacha_stream_and_antithetic_estimator() {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};
    let model = GbmModel::new(100.0, 0.03, 0.01, 0.2, 1.0, 4);
    assert_eq!(model.noise_dim(), model.steps);
    for antithetic in [false, true] {
        let engine = McEngine::new(19).with_seed(47).with_antithetic(antithetic);
        let result = engine.price(&model, &call());
        let n = if antithetic { 10 } else { 19 };
        let mut samples = Vec::new();
        for i in 0..n {
            let mut rng = ChaCha20Rng::seed_from_u64(47);
            rng.set_stream(i as u64);
            let mut normals: Vec<f64> = (0..model.steps)
                .map(|_| StandardNormal.sample(&mut rng))
                .collect();
            let mut path = vec![0.0; model.steps + 1];
            model.generate(&normals, &mut path);
            let first = call().evaluate(&path, 0.25, 0.03);
            samples.push(if antithetic {
                for normal in &mut normals {
                    *normal = -*normal;
                }
                model.generate(&normals, &mut path);
                0.5 * (first + call().evaluate(&path, 0.25, 0.03))
            } else {
                first
            });
        }
        let mean = samples.iter().sum::<f64>() / n as f64;
        let variance = samples
            .iter()
            .map(|sample| (sample - mean).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;
        assert_eq!(result.price, mean);
        assert_eq!(result.std_error, (variance / n as f64).sqrt());
        assert_eq!(result.samples, n);
    }
}

#[test]
fn gbm_plain_and_controlled_prices_match_pre_heston_snapshots() {
    // Captured by building HEAD's original source in an isolated temporary
    // crate before the noise_dim change: seed=42, paths=2000, antithetic.
    // Exact snapshots are specific to macOS/aarch64's transcendental math;
    // other platforms use a tight numerical bound plus the portable legacy
    // stream/estimator bitwise regression above.
    let model = GbmModel::new(100.0, 0.03, 0.0, 0.25, 1.0, 12);
    let control = EuropeanControl::new(&model, OptionType::Call, 100.0);
    for parallel in [false, true] {
        let engine = McEngine::new(2_000)
            .with_seed(42)
            .with_antithetic(true)
            .with_parallel(parallel);
        for (result, price_bits, error_bits) in [
            (
                engine.price(&model, &call()),
                4_622_649_201_644_832_719,
                4_599_490_776_013_517_232,
            ),
            (
                engine.price_with_control(&model, &call(), &control),
                4_622_578_242_554_875_880,
                4_393_630_416_385_507_573,
            ),
        ] {
            assert_eq!(result.samples, 1_000);
            assert!((result.price - f64::from_bits(price_bits)).abs() < 1e-12);
            assert!((result.std_error - f64::from_bits(error_bits)).abs() < 1e-12);
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                assert_eq!(result.price.to_bits(), price_bits);
                assert_eq!(result.std_error.to_bits(), error_bits);
            }
        }
    }
}

#[test]
fn full_truncation_retains_negative_raw_variance() {
    // First step makes raw variance -0.17. Two subsequent steps add 0.1
    // each while using zero diffusion; only step four uses variance 0.03.
    // Projecting the stored state to zero would resume diffusion sooner.
    let model = HestonModel::new(100.0, 0.0, 0.0, 0.04, 1.0, 0.4, 3.0, 0.0, 1.0, 4);
    let normals = [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let mut path = [0.0; 5];
    model.generate(&normals, &mut path);
    let first = 100.0 * (-0.005_f64).exp();
    assert!((path[1] - first).abs() < 1e-12);
    assert_eq!(path[1], path[2]);
    assert_eq!(path[2], path[3]);
    let terminal = first * (-0.5_f64 * 0.03 * 0.25 + (0.03_f64 * 0.25).sqrt()).exp();
    assert!((path[4] - terminal).abs() < 1e-12);
}

#[test]
fn heston_parameter_boundaries_and_public_mutation_are_checked() {
    assert!(HestonModel::try_new(100.0, -0.01, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 1.0, 1).is_ok());
    assert!(HestonModel::try_new(100.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1).is_ok());
    // Violating 2*kappa*theta >= xi^2 is supported by full truncation.
    assert!(HestonModel::try_new(100.0, 0.0, 0.0, 0.04, 1.0, 0.04, 2.0, 0.0, 1.0, 1).is_ok());
    let engine = McEngine::new(1);
    for field in [
        "spot",
        "risk_free_rate",
        "dividend_yield",
        "initial_variance",
        "kappa",
        "theta",
        "vol_of_vol",
        "rho",
        "maturity",
    ] {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut m = model(1);
            match field {
                "spot" => m.spot = value,
                "risk_free_rate" => m.risk_free_rate = value,
                "dividend_yield" => m.dividend_yield = value,
                "initial_variance" => m.initial_variance = value,
                "kappa" => m.kappa = value,
                "theta" => m.theta = value,
                "vol_of_vol" => m.vol_of_vol = value,
                "rho" => m.rho = value,
                "maturity" => m.maturity = value,
                _ => unreachable!(),
            }
            assert!(
                matches!(engine.try_price(&m, &call()), Err(PricingError::InvalidParameter { name, .. }) if name == field)
            );
        }
    }
    for (field, value) in [
        ("spot", 0.0),
        ("initial_variance", -0.01),
        ("kappa", 0.0),
        ("theta", -0.01),
        ("vol_of_vol", -0.01),
        ("rho", 1.01),
        ("rho", -1.01),
        ("maturity", 0.0),
    ] {
        let mut m = model(1);
        match field {
            "spot" => m.spot = value,
            "initial_variance" => m.initial_variance = value,
            "kappa" => m.kappa = value,
            "theta" => m.theta = value,
            "vol_of_vol" => m.vol_of_vol = value,
            "rho" => m.rho = value,
            "maturity" => m.maturity = value,
            _ => unreachable!(),
        }
        assert!(engine.try_price(&m, &call()).is_err());
    }
    for steps in [0, usize::MAX / 2 + 1, usize::MAX] {
        let mut m = model(1);
        m.steps = steps;
        assert!(engine.try_price(&m, &call()).is_err());
    }
    let mut gbm = GbmModel::new(100.0, 0.03, 0.0, 0.2, 1.0, 1);
    gbm.volatility = -0.2;
    assert!(engine.try_price(&gbm, &call()).is_err());
}

#[test]
fn heston_stock_and_variance_correlation_at_endpoints() {
    for rho in [-1.0, 1.0] {
        let model = HestonModel::new(100.0, 0.0, 0.0, 0.04, 1.0, 0.04, 0.2, rho, 1.0, 2);
        let mut first = [0.0; 3];
        let mut second = [0.0; 3];
        model.generate(&[1.0, -10.0, 1.0, 10.0], &mut first);
        model.generate(&[1.0, 10.0, 1.0, -10.0], &mut second);
        assert_eq!(
            first, second,
            "rho = +/-1 must ignore the independent normal"
        );
    }
}

#[test]
fn constant_variance_heston_matches_gbm_path_and_black_scholes() {
    let gbm = GbmModel::new(100.0, 0.03, 0.01, 0.2, 1.0, 4);
    let heston = HestonModel::new(100.0, 0.03, 0.01, 0.04, 1.5, 0.04, 0.0, -0.7, 1.0, 4);
    let normals = [0.5, -1.0, 0.25, 0.0];
    let mut doubled = Vec::new();
    for z in normals {
        doubled.extend([z, 3.0]);
    }
    let mut gbm_path = [0.0; 5];
    let mut heston_path = [0.0; 5];
    gbm.generate(&normals, &mut gbm_path);
    heston.generate(&doubled, &mut heston_path);
    for (a, b) in gbm_path.into_iter().zip(heston_path) {
        assert!((a - b).abs() < 1e-12);
    }
    let result = McEngine::new(40_000).with_seed(21).price(&heston, &call());
    let expected = bs_call(100.0, 100.0, 0.03, 0.01, 0.2, 1.0);
    assert!((result.price - expected).abs() <= 4.0 * result.std_error);
}

#[test]
fn heston_serial_parallel_reproducibility_and_discounted_martingale() {
    struct DiscountedStock;
    impl Payoff for DiscountedStock {
        fn evaluate(&self, path: &[f64], dt: f64, r: f64) -> f64 {
            path[path.len() - 1] * (-r * dt * (path.len() - 1) as f64).exp()
        }
    }
    let model = model(64);
    let engine = McEngine::new(40_000).with_seed(37);
    let parallel = engine
        .clone()
        .with_parallel(true)
        .price(&model, &DiscountedStock);
    let serial = engine.with_parallel(false).price(&model, &DiscountedStock);
    assert_eq!(parallel, serial);
    let expected = model.spot * (-model.dividend_yield * model.maturity).exp();
    assert!((parallel.price - expected).abs() < 4.0 * parallel.std_error);
}

#[test]
fn heston_vanilla_matches_independent_fourier_reference_across_grids() {
    let reference = fourier_call(&model(1), 100.0, 8_000);
    let finer_integral = fourier_call(&model(1), 100.0, 16_000);
    assert!(
        (reference - finer_integral).abs() < 1e-7,
        "Fourier quadrature must converge"
    );
    // Fourier integration is independent of the full-truncation path
    // scheme. Allow sampling error and a decreasing discretisation budget.
    for (steps, discretisation_tolerance) in [(16, 0.3), (64, 0.1), (256, 0.04)] {
        let result = McEngine::new(60_000)
            .with_seed(2718)
            .price(&model(steps), &call());
        println!(
            "steps={steps}, MC={}, se={}, Fourier={reference}",
            result.price, result.std_error
        );
        assert!(
            (result.price - reference).abs() < 4.0 * result.std_error + discretisation_tolerance
        );
    }
}

// Test-only complex arithmetic avoids adding a numerical dependency to the
// library. Characteristic function from Heston (1993), with the stable
// negative-exponential representation and risk-neutral Fourier inversion:
// https://doi.org/10.1093/rfs/6.2.327
#[derive(Clone, Copy)]
struct Complex {
    re: f64,
    im: f64,
}
impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn exp(self) -> Self {
        let amplitude = self.re.exp();
        Self::new(amplitude * self.im.cos(), amplitude * self.im.sin())
    }
    fn ln(self) -> Self {
        Self::new(self.re.hypot(self.im).ln(), self.im.atan2(self.re))
    }
    fn sqrt(self) -> Self {
        let modulus = self.re.hypot(self.im);
        if self.re >= 0.0 {
            let real = ((modulus + self.re) * 0.5).sqrt();
            Self::new(
                real,
                if real == 0.0 {
                    0.0
                } else {
                    self.im / (2.0 * real)
                },
            )
        } else {
            let imaginary = ((modulus - self.re) * 0.5).sqrt().copysign(self.im);
            Self::new(self.im / (2.0 * imaginary), imaginary)
        }
    }
}
impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }
}
impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }
}
impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }
}
impl std::ops::Div for Complex {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        let denominator = other.re * other.re + other.im * other.im;
        Self::new(
            (self.re * other.re + self.im * other.im) / denominator,
            (self.im * other.re - self.re * other.im) / denominator,
        )
    }
}
impl std::ops::Mul<f64> for Complex {
    type Output = Self;
    fn mul(self, factor: f64) -> Self {
        Self::new(self.re * factor, self.im * factor)
    }
}

fn characteristic(model: &HestonModel, u: Complex) -> Complex {
    let one = Complex::new(1.0, 0.0);
    let iu = Complex::new(0.0, 1.0) * u;
    let xi_squared = model.vol_of_vol * model.vol_of_vol;
    let b = Complex::new(model.kappa, 0.0) - iu * (model.rho * model.vol_of_vol);
    let d = (b * b + (u * u + iu) * xi_squared).sqrt();
    let g = (b - d) / (b + d);
    let decay = (d * -model.maturity).exp();
    let c = ((b - d) * model.maturity - ((one - g * decay) / (one - g)).ln() * 2.0)
        * (model.kappa * model.theta / xi_squared);
    let variance =
        (b - d) * (one - decay) / (one - g * decay) * (model.initial_variance / xi_squared);
    (iu * (model.spot.ln() + (model.risk_free_rate - model.dividend_yield) * model.maturity)
        + c
        + variance)
        .exp()
}

fn fourier_call(model: &HestonModel, strike: f64, intervals: usize) -> f64 {
    assert_eq!(intervals % 2, 0);
    let upper = 160.0;
    let lower = 1e-7;
    let width = (upper - lower) / intervals as f64;
    let forward =
        model.spot * ((model.risk_free_rate - model.dividend_yield) * model.maturity).exp();
    let integrand = |u: f64, shift: f64| {
        let phase = Complex::new(0.0, -u * strike.ln()).exp();
        let characteristic = characteristic(model, Complex::new(u, shift));
        (phase * characteristic / Complex::new(0.0, u)).re
            / if shift == -1.0 { forward } else { 1.0 }
    };
    let probability = |shift: f64| {
        let mut total = integrand(lower, shift) + integrand(upper, shift);
        for i in 1..intervals {
            total +=
                if i % 2 == 0 { 2.0 } else { 4.0 } * integrand(lower + i as f64 * width, shift);
        }
        0.5 + total * width / (3.0 * std::f64::consts::PI)
    };
    model.spot * (-model.dividend_yield * model.maturity).exp() * probability(-1.0)
        - strike * (-model.risk_free_rate * model.maturity).exp() * probability(0.0)
}
