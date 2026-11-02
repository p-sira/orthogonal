//! Independent high-degree and real Legendre Q exterior checks generated with
//! mpmath 1.3.0 (110-digit precision), using the hypergeometric Q formula.
use orthogonal::*;

fn assert_relative(actual: f64, expected: f64, tolerance: f64, context: &str) {
    if expected == 0.0 {
        assert_eq!(actual, 0.0, "{context}");
    } else {
        assert!(
            actual.is_finite() && ((actual - expected) / expected).abs() <= tolerance,
            "{context}: actual={actual:.17e}, expected={expected:.17e}, relative={:.4e}",
            ((actual - expected) / expected).abs()
        );
    }
}

#[test]
fn independent_high_degree_and_legendre_domain_references() {
    let mut counts = [0; 6];
    for line in include_str!("fixtures/limits_expected.csv").lines().skip(1) {
        let values: Vec<_> = line.split(',').collect();
        assert_eq!(values.len(), 6);
        let x: f64 = values[3].parse().unwrap();
        let reference: f64 = values[4].parse().unwrap();
        let context = format!(
            "{} degree={} order={} x={x}",
            values[0], values[1], values[2]
        );
        match values[0] {
            "legendre" => {
                let n: i32 = values[1].parse().unwrap();
                assert_relative(eval_legendre(n, x), reference, 2e-10, &context);
                counts[0] += 1;
            }
            "jacobi" => {
                let n: i32 = values[1].parse().unwrap();
                assert_relative(eval_jacobi(n, 0.2_f64, 0.3, x), reference, 2e-10, &context);
                counts[3] += 1;
            }
            "gegenbauer" => {
                let n: i32 = values[1].parse().unwrap();
                assert_relative(eval_gegenbauer(n, 0.8_f64, x), reference, 2e-10, &context);
                counts[4] += 1;
            }
            "laguerre" => {
                let n: i32 = values[1].parse().unwrap();
                assert_relative(eval_genlaguerre(n, 0.4_f64, x), reference, 2e-10, &context);
                counts[5] += 1;
            }
            "lpmv" => {
                let degree: f64 = values[1].parse().unwrap();
                let order: i32 = values[2].parse().unwrap();
                assert_relative(lpmv(order, degree, x), reference, 2e-9, &context);
                counts[1] += 1;
            }
            "q" => {
                let n: usize = values[1].parse().unwrap();
                let (q, d) = lqn(n, x).unwrap();
                assert_relative(q[n], reference, 2e-10, &context);
                assert_relative(d[n], values[5].parse().unwrap(), 2e-10, &context);
                counts[2] += 1;
            }
            other => panic!("unexpected case {other}"),
        }
    }
    assert_eq!(counts, [13, 100, 70, 3, 3, 3]);
}

#[test]
fn high_degree_endpoint_recurrence_and_tables() {
    let max = 1_000_000;
    assert_eq!(eval_legendre(max, 1.0), 1.0);
    assert_eq!(eval_legendre(max - 1, -1.0), -1.0);
    assert_eq!(eval_chebyt(max, -1.0), 1.0);
    assert_eq!(eval_chebyu(max, 1.0), 1_000_001.0);
    assert_eq!(eval_genlaguerre(max, 0.0, 0.0), 1.0);
    assert_eq!(eval_jacobi(max, 0.0, 0.0, 1.0), 1.0);
    assert_eq!(eval_gegenbauer(max, 0.5, 1.0), 1.0);
    assert_eq!(eval_hermite(4097, 0.0_f64), 0.0);
    assert_eq!(eval_hermitenorm(4097, 0.0_f64), 0.0);
    assert!((eval_chebyt(8192, 0.5_f64) + 0.5).abs() < 2e-12);
    assert!(eval_legendre(max + 1, 0.3_f64).is_nan());
    assert!(matches!(
        legendre_p_all(max as usize + 1, 0.3),
        Err(Error::InvalidInput(_))
    ));
    // Independently evaluate P_n(1-2z) as the terminating Gauss series
    // 2F1(-n,n+1;1;z); n=1e6 while z~5e-13 needs only ~20 terms.
    let x = 1.0 - 1e-12;
    let z = (1.0 - x) * 0.5;
    let mut term = 1.0;
    let mut reference = 1.0;
    for k in 1..=60 {
        let kf = k as f64;
        term *= ((kf - 1.0 - max as f64) * (max as f64 + kf) * z) / (kf * kf);
        reference += term;
    }
    assert_relative(eval_legendre(max, x), reference, 2e-10, "P_1000000 near +1");
    assert_relative(
        eval_legendre(max, -x),
        reference,
        2e-10,
        "P_1000000 near -1",
    );

    // P_2k(0) = (-1)^k product_{j=1}^k (2j-1)/(2j), P_2k+1(0)=0.
    let n = 65536;
    let table = legendre_p_all(n, 0.0).unwrap();
    assert_eq!(table.len(), n + 1);
    let mut expected = 1.0;
    for k in 1..=n / 2 {
        expected *= -((2 * k - 1) as f64) / (2 * k) as f64;
        assert_relative(table[2 * k], expected, 3e-12, &format!("P_{}(0)", 2 * k));
        assert_eq!(table[2 * k - 1], 0.0);
    }
    let x = 0.000001;
    let table = legendre_p_all(8192, x).unwrap();
    for k in [0, 1, 17, 128, 4096, 8192] {
        assert_relative(
            table[k],
            eval_legendre(k as i32, x),
            3e-10,
            &format!("P_{k}({x})"),
        );
    }
}

#[test]
fn exterior_branch_symmetries_domains_and_normalization() {
    for x in [1.0 + f64::EPSILON, 1.01, 2.0, 10.0, 1e100] {
        let (pos, dp) = lqn(32, x).unwrap();
        let (neg, dn) = lqn(32, -x).unwrap();
        for n in 0..=32 {
            assert_eq!(neg[n], if n % 2 == 0 { -pos[n] } else { pos[n] });
            assert_eq!(dn[n], if n % 2 == 0 { dp[n] } else { -dp[n] });
        }
        let (values, derivatives) = lqmn(0, 32, x).unwrap();
        assert_eq!(values[0], pos);
        assert_eq!(derivatives[0], dp);
    }
    for x in [1.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(lqn(0, x).is_err());
        assert!(lqmn(0, 2, x).is_err());
    }
    assert!(lqmn(1, 2, 2.0).is_ok());
    assert!(lqn(129, 0.3).is_ok());
    assert!(lqn(4096, 0.3).is_ok());
    assert!(lqmn(2, 129, 0.3).is_ok());
    assert!(lqn(4097, 0.3).is_err());
    assert!(lqmn(2, 4097, 0.3).is_err());
    assert!(lqmn(4096, 4096, 0.3).is_err());
    assert_eq!(eval_sh_jacobi(0, -0.25, 0.5, 0.3), 1.0);
    assert!(eval_sh_jacobi(0, -1.0, 0.5, 0.3).is_nan());
}

#[test]
fn extended_gaussian_rules_integrate_polynomials_and_have_positive_weights() {
    // Exact orthogonality moments cross-check large orders independently of
    // the tridiagonal eigensolver's internal eigenpairs and recurrence.
    for n in [512, 1024] {
        for (label, result, mass, mean, second) in [
            ("legendre", roots_legendre(n), 2.0, 0.0, 1.0 / 3.0),
            (
                "jacobi",
                roots_jacobi(n, 0.3, 0.7),
                1.6309532725293936, // 4*B(1.3,1.7)
                (0.7 - 0.3) / 3.0,
                3.16 / 12.0,
            ),
            ("laguerre", roots_laguerre(n), 1.0, 1.0, 2.0),
            (
                "hermite",
                roots_hermite(n),
                std::f64::consts::PI.sqrt(),
                0.0,
                0.5,
            ),
            (
                "hermitenorm",
                roots_hermitenorm(n),
                (2.0 * std::f64::consts::PI).sqrt(),
                0.0,
                1.0,
            ),
        ] {
            let (nodes, weights) = result.unwrap_or_else(|e| panic!("{label} n={n}: {e}"));
            assert!(nodes.windows(2).all(|v| v[0] < v[1]), "{label} n={n}");
            assert!(
                weights.iter().all(|w| w.is_finite() && *w >= 0.0),
                "{label} n={n}"
            );
            let sum: f64 = weights.iter().sum();
            let avg: f64 = nodes.iter().zip(&weights).map(|(x, w)| x * w).sum::<f64>() / sum;
            let sq: f64 = nodes
                .iter()
                .zip(&weights)
                .map(|(x, w)| x * x * w)
                .sum::<f64>()
                / sum;
            assert!(
                (sum / mass - 1.0).abs() < 3e-10,
                "{label} n={n}, mass={sum}"
            );
            assert!((avg - mean).abs() < 5e-12, "{label} n={n}, mean={avg}");
            assert!((sq - second).abs() < 5e-12, "{label} n={n}, second={sq}");
        }
    }
}
