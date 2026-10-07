//! Structured notes, numerical controls and Heston in one reproducible run.
//! Run: cargo run --release --example advanced_pricing

use std::time::Instant;

use fuzzy_enigma::{
    try_continuous_barrier_greeks, BarrierKind, BarrierOption, BumpSizes, EuropeanOption, GbmModel,
    GeometricAsianControl, HestonModel, McEngine, OptionType, PhoenixNote, PriceResult,
    ScheduledAsianOption, SnowballNote,
};

fn report(label: &str, result: PriceResult, elapsed: f64) {
    println!(
        "{label:<30} price={:.6} SE={:.6} samples={} elapsed={elapsed:.3}s",
        result.price, result.std_error, result.samples,
    );
}

fn main() {
    let gbm = GbmModel::new(100.0, 0.03, 0.0, 0.25, 1.0, 252);
    let engine = McEngine::new(40_000).with_seed(42);
    let dates = vec![63, 126, 189, 252];
    println!("GBM {gbm:?}; engine {engine:?}");

    let snowball = SnowballNote {
        notional: 100.0,
        reference_spot: 100.0,
        coupon_rate: 0.12,
        knock_in_barrier: 70.0,
        knock_out_barrier: 103.0,
        observation_indices: dates.clone(),
    };
    let phoenix = PhoenixNote {
        notional: 100.0,
        reference_spot: 100.0,
        coupon_per_period: 2.0,
        coupon_barrier: 80.0,
        knock_in_barrier: 60.0,
        knock_out_barrier: 105.0,
        coupon_indices: dates.clone(),
        autocall_indices: dates.clone(),
        memory: true,
    };
    let start = Instant::now();
    report(
        "Snowball",
        engine.price(&gbm, &snowball),
        start.elapsed().as_secs_f64(),
    );
    let start = Instant::now();
    report(
        "Phoenix (memory)",
        engine.price(&gbm, &phoenix),
        start.elapsed().as_secs_f64(),
    );

    let asian = ScheduledAsianOption {
        option_type: OptionType::Call,
        strike: 100.0,
        observation_indices: dates.clone(),
    };
    let control = GeometricAsianControl::new(&gbm, OptionType::Call, 100.0, dates);
    let start = Instant::now();
    report(
        "Scheduled Asian plain",
        engine.price(&gbm, &asian),
        start.elapsed().as_secs_f64(),
    );
    let start = Instant::now();
    report(
        "Scheduled Asian geometric CV",
        engine.price_with_control(&gbm, &asian, &control),
        start.elapsed().as_secs_f64(),
    );

    let barrier = BarrierOption {
        option_type: OptionType::Call,
        kind: BarrierKind::UpAndOut,
        strike: 100.0,
        barrier: 130.0,
        rebate: 0.0,
    };
    let start = Instant::now();
    report(
        "Continuous up-and-out",
        engine.price_continuous_barrier(&gbm, &barrier),
        start.elapsed().as_secs_f64(),
    );
    let greeks = try_continuous_barrier_greeks(&engine, &gbm, &barrier, BumpSizes::for_model(&gbm))
        .expect("valid Greeks setup");
    println!("Continuous barrier Greeks {greeks:?}");

    let heston = HestonModel::new(100.0, 0.03, 0.0, 0.04, 1.5, 0.04, 0.4, -0.7, 1.0, 252);
    println!("Heston {heston:?}");
    let start = Instant::now();
    report(
        "Heston Snowball",
        engine.price(&heston, &snowball),
        start.elapsed().as_secs_f64(),
    );
    let start = Instant::now();
    report(
        "Heston European",
        engine.price(
            &heston,
            &EuropeanOption {
                option_type: OptionType::Call,
                strike: 100.0,
            },
        ),
        start.elapsed().as_secs_f64(),
    );
}
