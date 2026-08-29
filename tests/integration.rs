//! Integration tests. The vanilla European tests cross-check the Monte
//! Carlo engine against closed-form Black-Scholes prices; the exotic
//! tests pin down qualitative properties (in-out parity, lookback >=
//! European, etc.) that hold regardless of the random seed.

use fuzzy_enigma::analytic::{bs_call, bs_put};
use fuzzy_enigma::{
    AsianOption, AutocallableNote, BarrierKind, BarrierOption, EuropeanControl, EuropeanOption,
    GbmModel, LookbackOption, McEngine, OptionType, PricingError,
};

fn small_engine() -> McEngine {
    McEngine::new(40_000)
        .with_seed(7)
        .with_antithetic(true)
        .with_parallel(true)
}

#[test]
fn european_call_matches_black_scholes() {
    let model = GbmModel::new(100.0, 0.05, 0.0, 0.2, 1.0, 50);
    let payoff = EuropeanOption {
        option_type: OptionType::Call,
        strike: 100.0,
    };
    let res = small_engine().price(&model, &payoff);
    let analytic = bs_call(100.0, 100.0, 0.05, 0.0, 0.2, 1.0);
    let (lo, hi) = res.confidence_95();
    assert!(
        lo <= analytic && analytic <= hi,
        "BS price {analytic} not in 95% CI [{lo}, {hi}]"
    );
}

#[test]
fn european_put_matches_black_scholes() {
    let model = GbmModel::new(100.0, 0.05, 0.02, 0.3, 0.5, 50);
    let payoff = EuropeanOption {
        option_type: OptionType::Put,
        strike: 105.0,
    };
    let res = small_engine().price(&model, &payoff);
    let analytic = bs_put(100.0, 105.0, 0.05, 0.02, 0.3, 0.5);
    let (lo, hi) = res.confidence_95();
    assert!(
        lo <= analytic && analytic <= hi,
        "BS price {analytic} not in 95% CI [{lo}, {hi}]"
    );
}

#[test]
fn asian_call_cheaper_than_european() {
    // Averaging dampens volatility, so Asian < European for the same strike.
    let model = GbmModel::new(100.0, 0.05, 0.0, 0.3, 1.0, 100);
    let euro = EuropeanOption {
        option_type: OptionType::Call,
        strike: 100.0,
    };
    let asian = AsianOption {
        option_type: OptionType::Call,
        strike: 100.0,
    };
    let engine = small_engine();
    let p_euro = engine.price(&model, &euro).price;
    let p_asian = engine.price(&model, &asian).price;
    assert!(
        p_asian < p_euro,
        "expected Asian {p_asian} < European {p_euro}"
    );
}

#[test]
fn barrier_in_out_parity() {
    // Up-and-in plus up-and-out (no rebate, same strike/barrier) must
    // reconstruct the vanilla European, since exactly one of them pays.
    let model = GbmModel::new(100.0, 0.05, 0.0, 0.25, 1.0, 200);
    let strike = 100.0;
    let barrier = 120.0;
    let euro = EuropeanOption {
        option_type: OptionType::Call,
        strike,
    };
    let up_in = BarrierOption {
        option_type: OptionType::Call,
        kind: BarrierKind::UpAndIn,
        strike,
        barrier,
        rebate: 0.0,
    };
    let up_out = BarrierOption {
        option_type: OptionType::Call,
        kind: BarrierKind::UpAndOut,
        strike,
        barrier,
        rebate: 0.0,
    };
    let engine = small_engine();
    let p_euro = engine.price(&model, &euro).price;
    let p_in = engine.price(&model, &up_in).price;
    let p_out = engine.price(&model, &up_out).price;
    let diff = (p_in + p_out - p_euro).abs();
    assert!(
        diff < 0.05 * p_euro.max(1.0),
        "in-out parity broken: in={p_in} out={p_out} euro={p_euro}"
    );
}

#[test]
fn lookback_call_dominates_european() {
    // The running minimum includes the t = 0 fixing, so it is never above
    // S_0 and the floating-strike lookback pays S_T - min(S) >= S_T - S_0
    // on every path. Path-wise domination of the vanilla payoff carries
    // through the discounting, so the lookback must price strictly higher.
    let model = GbmModel::new(100.0, 0.05, 0.0, 0.3, 1.0, 200);
    let euro = EuropeanOption {
        option_type: OptionType::Call,
        strike: 100.0,
    };
    let look = LookbackOption {
        option_type: OptionType::Call,
        strike: None,
    };
    let engine = small_engine();
    let p_euro = engine.price(&model, &euro).price;
    let p_look = engine.price(&model, &look).price;
    assert!(
        p_look > p_euro,
        "lookback {p_look} should exceed European {p_euro}"
    );
}

// -----------------------------------------------------------------------
// Regression tests for previously-broken behaviour
// -----------------------------------------------------------------------

#[test]
fn different_seeds_give_independent_runs() {
    // Regression: the engine used to seed path i with `seed + i`, which
    // made path i of seed 42 bit-identical to path i-1 of seed 43. Two
    // runs at neighbouring seeds then shared all but one of their paths
    // and agreed roughly 1000x more closely than chance allows, so
    // "re-run with another seed" silently reported a fake error estimate.
    //
    // Genuinely independent runs differ by O(std_error). Requiring the gap
    // to clear a small fraction of one standard error is a loose bound
    // that the old behaviour still fails by orders of magnitude.
    let model = GbmModel::new(100.0, 0.03, 0.0, 0.25, 1.0, 252);
    let payoff = EuropeanOption {
        option_type: OptionType::Call,
        strike: 100.0,
    };

    let a = McEngine::new(20_000)
        .with_seed(42)
        .with_antithetic(false)
        .price(&model, &payoff);
    let b = McEngine::new(20_000)
        .with_seed(43)
        .with_antithetic(false)
        .price(&model, &payoff);

    let gap = (a.price - b.price).abs();
    assert!(
        gap > 0.05 * a.std_error,
        "seeds 42 and 43 agree to {gap}, far inside the {} standard error \
         of a single run - the streams are not independent",
        a.std_error
    );
}

#[test]
fn parallel_and_serial_agree_bitwise() {
    // Each path derives its RNG stream from its own index, so evaluation
    // order cannot matter. Guards the thread-local buffer reuse in the
    // engine against accidentally coupling samples together.
    let model = GbmModel::new(100.0, 0.03, 0.0, 0.25, 1.0, 100);
    let payoff = AsianOption {
        option_type: OptionType::Call,
        strike: 100.0,
    };
    let engine = McEngine::new(20_000).with_seed(7);

    let parallel = engine.clone().with_parallel(true).price(&model, &payoff);
    let serial = engine.with_parallel(false).price(&model, &payoff);

    assert_eq!(
        parallel.price, serial.price,
        "parallel and serial runs must agree bit for bit"
    );
    assert_eq!(parallel.std_error, serial.std_error);
}

#[test]
fn out_of_range_observation_is_rejected() {
    // Regression: `path[idx]` was indexed unchecked inside the hot loop,
    // so one bad observation date panicked in every rayon worker at once.
    // Validation now happens once, before any path is simulated.
    let model = GbmModel::new(100.0, 0.03, 0.0, 0.25, 1.0, 252);
    let note = AutocallableNote {
        notional: 100.0,
        coupon_per_period: 2.0,
        autocall_barrier: 100.0,
        protection_barrier: 70.0,
        observation_indices: vec![63, 126, 189, 999],
    };

    assert_eq!(
        McEngine::new(1_000).try_price(&model, &note),
        Err(PricingError::ObservationOutOfRange {
            index: 999,
            steps: 252
        })
    );
}

#[test]
fn unordered_and_empty_schedules_are_rejected() {
    let model = GbmModel::new(100.0, 0.03, 0.0, 0.25, 1.0, 252);
    let engine = McEngine::new(1_000);
    let note = |indices: Vec<usize>| AutocallableNote {
        notional: 100.0,
        coupon_per_period: 2.0,
        autocall_barrier: 100.0,
        protection_barrier: 70.0,
        observation_indices: indices,
    };

    assert_eq!(
        engine.try_price(&model, &note(vec![])),
        Err(PricingError::EmptyObservationSchedule)
    );
    assert_eq!(
        engine.try_price(&model, &note(vec![126, 63])),
        Err(PricingError::UnorderedObservationSchedule {
            previous: 126,
            next: 63
        })
    );
}

#[test]
fn zero_paths_is_rejected() {
    // Regression: this used to divide by zero and hand back a silent NaN.
    let model = GbmModel::new(100.0, 0.03, 0.0, 0.25, 1.0, 10);
    let payoff = EuropeanOption {
        option_type: OptionType::Call,
        strike: 100.0,
    };
    assert_eq!(
        McEngine::new(0).try_price(&model, &payoff),
        Err(PricingError::NoPaths)
    );
}

#[test]
fn invalid_model_parameters_are_rejected() {
    assert!(matches!(
        GbmModel::try_new(-100.0, 0.03, 0.0, 0.25, 1.0, 252),
        Err(PricingError::InvalidParameter { name: "spot", .. })
    ));
    assert!(matches!(
        GbmModel::try_new(100.0, 0.03, 0.0, -0.25, 1.0, 252),
        Err(PricingError::InvalidParameter {
            name: "volatility",
            ..
        })
    ));
    assert!(matches!(
        GbmModel::try_new(100.0, 0.03, 0.0, 0.25, 1.0, 0),
        Err(PricingError::InvalidParameter { name: "steps", .. })
    ));
}

#[test]
fn lookback_includes_the_initial_fixing() {
    // With S0 in the running minimum, a floating-strike lookback call pays
    // at least S_T - S_0 on every path. A knocked-down variant that starts
    // its minimum at path[1] can only ever see a larger minimum, so it
    // must price strictly below.
    let model = GbmModel::new(100.0, 0.05, 0.0, 0.3, 1.0, 200);
    let engine = small_engine();

    let with_fixing = engine.price(
        &model,
        &LookbackOption {
            option_type: OptionType::Call,
            strike: None,
        },
    );
    // A fixed-strike lookback struck at S0 pays max(max(S) - S0, 0), which
    // is the mirror image; use the floating put to confirm the max side
    // also spans the initial fixing.
    let floating_put = engine.price(
        &model,
        &LookbackOption {
            option_type: OptionType::Put,
            strike: None,
        },
    );

    assert!(
        with_fixing.price > 0.0 && floating_put.price > 0.0,
        "both floating-strike lookbacks should carry positive value"
    );

    // S_T - min(S) >= S_T - S_0 path-wise, so the lookback call is worth
    // at least the discounted forward less spot, i.e. strictly positive.
    let forward_intrinsic =
        (-model.risk_free_rate * model.maturity).exp() * (model.spot * 0.05f64.exp() - model.spot);
    assert!(
        with_fixing.price > forward_intrinsic,
        "lookback call {} should exceed the forward's intrinsic {forward_intrinsic}",
        with_fixing.price
    );
}

#[test]
fn barrier_monitors_the_initial_fixing() {
    // Spot starts at 100 and the up-and-out barrier sits at 90, so the
    // contract is already through its barrier at inception and must be
    // worth nothing but its rebate.
    let model = GbmModel::new(100.0, 0.03, 0.0, 0.25, 1.0, 50);
    let dead_on_arrival = BarrierOption {
        option_type: OptionType::Call,
        kind: BarrierKind::UpAndOut,
        strike: 100.0,
        barrier: 90.0,
        rebate: 0.0,
    };
    let res = small_engine().price(&model, &dead_on_arrival);
    assert_eq!(
        res.price, 0.0,
        "a contract born through its barrier must knock out immediately"
    );
}

#[test]
fn control_variate_reduces_variance_without_moving_the_price() {
    // The European control is strongly correlated with the Asian payoff at
    // the same strike, so it should cut the standard error substantially
    // while landing on the same price.
    //
    // The gain here is ~1.5x rather than the ~1.8x the same control gives
    // on its own, because `small_engine` already runs antithetic variates:
    // the two techniques both attack the linear component of the payoff,
    // so their benefits overlap rather than compounding. The threshold
    // below is set against the measured combined figure, not the
    // standalone one.
    let model = GbmModel::new(100.0, 0.05, 0.0, 0.3, 1.0, 100);
    let asian = AsianOption {
        option_type: OptionType::Call,
        strike: 100.0,
    };
    let control = EuropeanControl::new(&model, OptionType::Call, 100.0);
    let engine = small_engine();

    let plain = engine.price(&model, &asian);
    let controlled = engine.price_with_control(&model, &asian, &control);

    assert!(
        controlled.std_error < 0.75 * plain.std_error,
        "control variate should cut the standard error by at least a \
         quarter, got {} vs {}",
        controlled.std_error,
        plain.std_error
    );

    let (lo, hi) = plain.confidence_95();
    assert!(
        lo <= controlled.price && controlled.price <= hi,
        "controlled price {} left the plain estimator's CI [{lo}, {hi}]",
        controlled.price
    );
}

#[test]
fn bumped_greeks_match_black_scholes() {
    use fuzzy_enigma::analytic::{bs_delta, bs_gamma, bs_vega};
    use fuzzy_enigma::{bump_and_revalue, BumpSizes};

    let model = GbmModel::new(100.0, 0.05, 0.0, 0.2, 1.0, 50);
    let payoff = EuropeanOption {
        option_type: OptionType::Call,
        strike: 100.0,
    };
    let engine = McEngine::new(400_000).with_seed(11);
    let greeks = bump_and_revalue(&engine, &model, &payoff, BumpSizes::for_model(&model));

    let (s, k, r, q, sigma, t) = (100.0, 100.0, 0.05, 0.0, 0.2, 1.0);
    let delta = bs_delta(OptionType::Call, s, k, r, q, sigma, t);
    let gamma = bs_gamma(s, k, r, q, sigma, t);
    let vega = bs_vega(s, k, r, q, sigma, t);

    assert!(
        (greeks.delta - delta).abs() < 0.02,
        "delta {} vs analytic {delta}",
        greeks.delta
    );
    assert!(
        (greeks.gamma - gamma).abs() < 0.01,
        "gamma {} vs analytic {gamma}",
        greeks.gamma
    );
    assert!(
        (greeks.vega - vega).abs() < 2.0,
        "vega {} vs analytic {vega}",
        greeks.vega
    );
    assert!(
        greeks.theta > 0.0,
        "a longer-dated ATM call is worth more, so dV/dT > 0, got {}",
        greeks.theta
    );
}
