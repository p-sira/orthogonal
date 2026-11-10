//! Independent 100-digit trigonometric/hyperbolic endpoint references.
use orthogonal::{eval_chebyt, eval_chebyu, try_eval_chebyt, try_eval_chebyu};

#[test]
fn endpoint_values_match_high_precision_references() {
    let mut count = 0;
    for line in include_str!("fixtures/chebyshev_endpoints.csv")
        .lines()
        .skip(1)
    {
        let fields: Vec<_> = line.split(',').collect();
        let n: i32 = fields[1].parse().unwrap();
        let x: f64 = fields[2].parse().unwrap();
        let expected: f64 = fields[3].parse().unwrap();
        for sign in [-1.0, 1.0] {
            let expected = if sign < 0.0 && n % 2 != 0 {
                -expected
            } else {
                expected
            };
            let actual = match fields[0] {
                "T" => try_eval_chebyt(n, sign * x).unwrap(),
                "U" => try_eval_chebyu(n, sign * x).unwrap(),
                _ => panic!("unknown family"),
            };
            if expected.is_infinite() {
                assert_eq!(actual, expected, "{line}, sign={sign}");
            } else {
                // Oscillatory values near roots require an absolute bound;
                // growing exterior values use a relative bound.
                let tolerance = if x < 1.0 {
                    5e-11 * expected.abs().max(1.0)
                } else {
                    3e-13 * expected.abs()
                };
                assert!(
                    actual.is_finite() && (actual - expected).abs() <= tolerance,
                    "{line}, sign={sign}: actual={actual:.17e}, expected={expected:.17e}"
                );
            }
        }
        count += 1;
    }
    assert_eq!(count, 224);
}

#[test]
fn audited_million_degree_cases_retain_double_precision() {
    for (actual, expected) in [
        (
            eval_chebyt(1_000_000, 0.9999999999_f64),
            -0.0049692473059826815,
        ),
        (
            eval_chebyu(1_000_000, 1.0000000000000002_f64),
            1_000_075.016_733_839,
        ),
    ] {
        assert!(
            (actual - expected).abs() <= 4e-15 * expected.abs().max(1.0),
            "actual={actual:.17e}, expected={expected:.17e}"
        );
    }
}
