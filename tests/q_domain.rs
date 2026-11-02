//! Independent real Legendre-Q references from mpmath 1.3.0, at 430 dps.
use orthogonal::{Error, lqmn, lqn};

fn assert_relative(actual: f64, expected: f64, tolerance: f64, description: &str) {
    if expected == 0.0 {
        assert_eq!(actual, 0.0, "{description}");
    } else {
        assert!(
            actual.is_finite() && ((actual - expected) / expected).abs() <= tolerance,
            "{description}: got {actual:.17e}, expected {expected:.17e}, relative error {:.3e}",
            (actual / expected - 1.0).abs()
        );
    }
}

#[test]
fn high_precision_ferrers_and_exterior_values_derivatives() {
    let mut cases = 0;
    for line in include_str!("fixtures/q_domain_expected.csv")
        .lines()
        .skip(1)
    {
        let fields: Vec<_> = line.split(',').collect();
        assert_eq!(fields.len(), 6, "{line}");
        let m = fields[0].parse::<usize>().unwrap();
        let n = fields[1].parse::<usize>().unwrap();
        let x = fields[2].parse::<f64>().unwrap();
        let value = fields[3].parse::<f64>().unwrap();
        let derivative = fields[4].parse::<f64>().unwrap();
        let tolerance = fields[5].parse::<f64>().unwrap();
        let (q, d) = lqmn(m, n, x).unwrap_or_else(|err| {
            panic!("lqmn({m},{n},{x}) failed: {err}");
        });
        let context = format!("Q_{n}^{m}({x})");
        assert_relative(q[m][n], value, tolerance, &context);
        assert_relative(
            d[m][n],
            derivative,
            tolerance,
            &format!("{context} derivative"),
        );
        cases += 1;
    }
    assert_eq!(cases, 20);
}

#[test]
fn exterior_reflection_and_degree_zero_row() {
    for (m, n, x) in [(1, 50, 1.01), (3, 15, 2.0), (5, 25, 10.0)] {
        let (pos, pd) = lqmn(m, n, x).unwrap();
        let (neg, nd) = lqmn(m, n, -x).unwrap();
        let (q, d) = lqn(n, x).unwrap();
        for degree in 0..=n {
            assert_relative(pos[0][degree], q[degree], 3e-12, "Q^0 equals lqn");
            assert_relative(pd[0][degree], d[degree], 3e-12, "Q^0 derivative equals lqn");
            for order in 0..=m {
                let reflection = if (degree + order) % 2 == 0 { -1.0 } else { 1.0 };
                assert_eq!(neg[order][degree], reflection * pos[order][degree]);
                assert_eq!(nd[order][degree], -reflection * pd[order][degree]);
            }
        }
    }
}

#[test]
fn bounded_large_degree_and_numerical_overflow() {
    let (q, d) = lqn(4096, 2.0).unwrap();
    assert_eq!(q[4096], 0.0);
    assert_eq!(d[4096], 0.0);
    assert!(q[0] > 0.0);
    assert!(matches!(lqn(4097, 2.0), Err(Error::InvalidInput(_))));
    assert!(matches!(lqmn(4097, 1, 2.0), Err(Error::InvalidInput(_))));
    assert!(matches!(lqmn(1000, 4096, 2.0), Err(Error::InvalidInput(_))));
    assert!(matches!(
        lqmn(300, 0, 1.01),
        Err(Error::NonFiniteEvaluation)
    ));
}
