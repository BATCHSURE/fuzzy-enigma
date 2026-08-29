//! End-to-end example: price the full menu of exotic payoffs against the
//! same Geometric Brownian Motion underlying.
//!
//! Run with:  `cargo run --release --example price_options`

use fuzzy_enigma::{
    bump_and_revalue, AsianOption, AutocallableNote, BarrierKind, BarrierOption, BumpSizes,
    CliquetOption, EuropeanControl, EuropeanOption, GbmModel, LookbackOption, McEngine, OptionType,
    Payoff,
};

fn main() {
    let spot = 100.0;
    let r = 0.03;
    let q = 0.0;
    let sigma = 0.25;
    let t = 1.0;
    let steps = 252;

    let model = GbmModel::new(spot, r, q, sigma, t, steps);
    let engine = McEngine::default()
        .with_seed(42)
        .with_antithetic(true)
        .with_parallel(true);

    let payoffs: Vec<(&str, Box<dyn Payoff>)> = vec![
        (
            "European call K=100",
            Box::new(EuropeanOption {
                option_type: OptionType::Call,
                strike: 100.0,
            }),
        ),
        (
            "Asian call K=100",
            Box::new(AsianOption {
                option_type: OptionType::Call,
                strike: 100.0,
            }),
        ),
        (
            "Up-and-out call K=100 B=130",
            Box::new(BarrierOption {
                option_type: OptionType::Call,
                kind: BarrierKind::UpAndOut,
                strike: 100.0,
                barrier: 130.0,
                rebate: 0.0,
            }),
        ),
        (
            "Down-and-in put  K=100 B=80",
            Box::new(BarrierOption {
                option_type: OptionType::Put,
                kind: BarrierKind::DownAndIn,
                strike: 100.0,
                barrier: 80.0,
                rebate: 0.0,
            }),
        ),
        (
            "Floating-strike lookback call",
            Box::new(LookbackOption {
                option_type: OptionType::Call,
                strike: None,
            }),
        ),
        (
            "Cliquet 5% cap / -3% floor",
            Box::new(CliquetOption {
                notional: 100.0,
                local_floor: -0.03,
                local_cap: 0.05,
                global_floor: 0.0,
                global_cap: f64::INFINITY,
            }),
        ),
        (
            "Autocallable note (quarterly)",
            Box::new(AutocallableNote {
                notional: 100.0,
                coupon_per_period: 2.0,
                autocall_barrier: 100.0,
                protection_barrier: 70.0,
                observation_indices: vec![63, 126, 189, 252],
            }),
        ),
    ];

    println!(
        "Spot={spot}  r={r}  q={q}  sigma={sigma}  T={t}y  steps={steps}  paths={}",
        engine.paths
    );
    println!("{:-<70}", "");
    println!(
        "{:<35} {:>12} {:>12} {:>9}",
        "payoff", "price", "std-err", "95% CI"
    );

    for (name, payoff) in &payoffs {
        let res = engine.price(&model, payoff.as_ref());
        let (lo, hi) = res.confidence_95();
        println!(
            "{:<35} {:>12.4} {:>12.4} [{:>5.2},{:>5.2}]",
            name, res.price, res.std_error, lo, hi
        );
    }

    // Variance reduction: the European at the same strike is strongly
    // correlated with the Asian payoff, and its expectation is exactly the
    // Black-Scholes price, so it makes a free control variate.
    let asian = AsianOption {
        option_type: OptionType::Call,
        strike: 100.0,
    };
    let control = EuropeanControl::new(&model, OptionType::Call, 100.0);
    let plain = engine.price(&model, &asian);
    let controlled = engine.price_with_control(&model, &asian, &control);

    println!();
    println!("Control variate on the Asian call, same path budget");
    println!("{:-<70}", "");
    println!(
        "{:<35} {:>12.4} {:>12.4}",
        "without control", plain.price, plain.std_error
    );
    println!(
        "{:<35} {:>12.4} {:>12.4}",
        "with European control", controlled.price, controlled.std_error
    );
    println!(
        "{:<35} {:>12.2}x",
        "standard error cut by",
        plain.std_error / controlled.std_error
    );

    // Greeks by bump-and-revalue. Every revaluation reuses the engine's
    // seed, so the bumped runs share their noise with the base run and the
    // differences are not swamped by Monte Carlo error.
    let euro = EuropeanOption {
        option_type: OptionType::Call,
        strike: 100.0,
    };
    let greeks = bump_and_revalue(&engine, &model, &euro, BumpSizes::for_model(&model));

    println!();
    println!("Greeks for the European call (bump-and-revalue, common random numbers)");
    println!("{:-<70}", "");
    for (name, value) in [
        ("delta  dV/dS", greeks.delta),
        ("gamma  d2V/dS2", greeks.gamma),
        ("vega   dV/dsigma", greeks.vega),
        ("theta  dV/dT", greeks.theta),
        ("rho    dV/dr", greeks.rho),
    ] {
        println!("{name:<35} {value:>12.4}");
    }
}
