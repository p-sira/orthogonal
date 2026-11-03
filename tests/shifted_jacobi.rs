//! Shifted Jacobi must agree with an independent 110-digit Jacobi/Gamma
//! reference even when its monic normalization is tiny or p is near -1.
use orthogonal::{eval_sh_jacobi, try_eval_sh_jacobi};

#[test]
fn monic_shifted_jacobi_matches_independent_mpmath() {
    let mut count = 0;
    for line in include_str!("fixtures/shifted_jacobi.csv").lines().skip(1) {
        let fields: Vec<_> = line.split(',').collect();
        assert_eq!(fields.len(), 5);
        let n: i32 = fields[0].parse().unwrap();
        let p: f64 = fields[1].parse().unwrap();
        let q: f64 = fields[2].parse().unwrap();
        let x: f64 = fields[3].parse().unwrap();
        let expected: f64 = fields[4].parse().unwrap();
        assert!(expected.is_finite() && expected != 0.0);
        let got = try_eval_sh_jacobi(n, p, q, x).unwrap();
        let relative = ((got - expected) / expected).abs();
        assert!(
            got.is_finite() && relative <= 2e-9,
            "n={n},p={p},q={q},x={x}: got={got:.17e}, expected={expected:.17e}, rel={relative:.3e}"
        );
        count += 1;
    }
    assert!(count >= 350, "expected broad parameter and degree coverage");
}

#[test]
fn normalization_endpoints_and_domains() {
    for (p, q) in [(2.5, 1.5), (0.0, 0.5), (-0.8, 0.1), (1e5, 3e4)] {
        assert_eq!(eval_sh_jacobi(0, p, q, 0.3), 1.0);
        let value = eval_sh_jacobi(1, p, q, 0.4);
        let wanted = 0.4 - q / (p + 1.0);
        assert!((value - wanted).abs() <= 3.0 * f64::EPSILON);
    }
    for (p, q) in [(-1.0, 0.1), (2.0, 0.0), (0.2, 1.3)] {
        assert!(try_eval_sh_jacobi(2, p, q, 0.4).is_err());
    }
    assert!(try_eval_sh_jacobi(2, 2.5, 1.5, f64::NAN).is_err());
    assert!(try_eval_sh_jacobi(-1, 2.5, 1.5, 0.4).is_err());
    assert!(try_eval_sh_jacobi(1_000_001, 2.5, 1.5, 0.4).is_err());
    for n in 1..=5 {
        assert_eq!(eval_sh_jacobi(n, 2.5, 1.5, f64::INFINITY), f64::INFINITY);
        assert_eq!(
            eval_sh_jacobi(n, 2.5, 1.5, f64::NEG_INFINITY),
            if n % 2 == 0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            }
        );
    }
}

#[test]
fn extremely_large_beta_parameters_do_not_overflow_recurrence_coefficients() {
    // For fixed n and p -> infinity with q/p -> r, the beta measure tends
    // to a point mass at r, and monic shifted Jacobi tends to (x-r)^n.
    let p = 1e308;
    let q = 9e307;
    for n in 1..=8 {
        for x in [0.0_f64, 0.2, 0.5, 1.0] {
            let expected = (x - 0.9).powi(n);
            let got = try_eval_sh_jacobi(n, p, q, x).unwrap();
            let relative = ((got - expected) / expected).abs();
            assert!(
                relative <= 5e-13,
                "n={n},x={x}: got={got}, expected={expected}, relative={relative}"
            );
        }
    }
}
