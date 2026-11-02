//! Independent Ferrers P references from mpmath 1.3.0 with 110 decimal digits:
//! mp.legenp(v, m, mp.mpf(x), type=2).
//!
//! The decimal inputs printed below are the exact round-tripped f64 x and v.
use orthogonal::lpmv;

// Independent inputs and expected values are kept in separate CSV files.
fn references() -> Vec<(i32, f64, f64, f64)> {
    let inputs = include_str!("fixtures/lpmv_domain_inputs.csv");
    let expected = include_str!("fixtures/lpmv_domain_expected.csv");
    let cases: Vec<_> = inputs.lines().skip(1).collect();
    let values: Vec<_> = expected.lines().skip(1).collect();
    assert_eq!(cases.len(), values.len());
    assert_eq!(cases.len(), 81);
    cases
        .into_iter()
        .zip(values)
        .enumerate()
        .map(|(index, (case, value))| {
            let case: Vec<_> = case.split(',').collect();
            let value: Vec<_> = value.split(',').collect();
            assert_eq!(case.len(), 4);
            assert_eq!(value.len(), 2);
            assert_eq!(case[0].parse::<usize>().unwrap(), index);
            assert_eq!(value[0].parse::<usize>().unwrap(), index);
            (
                case[1].parse().unwrap(),
                case[2].parse().unwrap(),
                case[3].parse().unwrap(),
                value[1].parse().unwrap(),
            )
        })
        .collect()
}

#[test]
fn independent_noninteger_domain_references() {
    for (order, degree, x, expected) in references() {
        let actual = lpmv(order, degree, x);
        let relative = ((actual - expected) / expected).abs();
        assert!(
            actual.is_finite() && relative <= 2e-8,
            "m={order}, v={degree}, x={x}: actual={actual:.17e}, expected={expected:.17e}, rel={relative:.3e}"
        );
    }
}

#[test]
fn degree_symmetry_and_domain_edges() {
    for (order, degree, x, _) in references() {
        let a = lpmv(order, degree, x);
        let b = lpmv(order, -degree - 1.0, x);
        // At v=1e-8 the degree reflected around -1/2 loses ~6e-17
        // absolute precision in its floating-point representation.
        let tolerance = if degree.abs() < 1e-6 { 1e-8 } else { 3e-13 };
        assert!(
            (a - b).abs() <= tolerance * a.abs().max(b.abs()),
            "m={order}, v={degree}, x={x}, P={a}, reflected={b}"
        );
    }
    for m in [-4097, 4097] {
        assert!(lpmv(m, 2.5, 0.0).is_nan());
    }
    assert!(lpmv(0, 100000.25, 0.0).is_nan());
    assert!(lpmv(0, 0.5, 1.0001).is_nan());
    assert!(lpmv(0, f64::NAN, 0.0).is_nan());
    assert_eq!(lpmv(0, 0.5, 1.0), 1.0);
    assert_eq!(lpmv(1, 0.5, 1.0), 0.0);
    assert_eq!(lpmv(-1, 0.5, 1.0), 0.0);
    assert!(lpmv(0, 0.5, -1.0).is_infinite());
    assert!(lpmv(1, 0.5, -1.0).is_infinite());
    assert!(lpmv(-1, 0.5, -1.0).is_infinite());
}
