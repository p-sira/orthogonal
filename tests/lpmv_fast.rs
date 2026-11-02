//! The low-order-seeded fast path must preserve nonzero high-precision values,
//! including small, oscillatory, and negative-order Ferrers P references.
use orthogonal::lpmv;

#[test]
fn accelerated_lpmv_matches_independent_mpmath() {
    let inputs = include_str!("fixtures/lpmv_fast_inputs.csv");
    let expected = include_str!("fixtures/lpmv_fast_expected.csv");
    let mut count = 0;
    for (source, reference) in inputs.lines().skip(1).zip(expected.lines().skip(1)) {
        let source: Vec<_> = source.split(',').collect();
        let reference: Vec<_> = reference.split(',').collect();
        assert_eq!(source.len(), 4);
        assert_eq!(reference.len(), 2);
        assert_eq!(source[0], reference[0]);
        let m: i32 = source[1].parse().unwrap();
        let v: f64 = source[2].parse().unwrap();
        let x: f64 = source[3].parse().unwrap();
        let wanted: f64 = reference[1].parse().unwrap();
        let observed = lpmv(m, v, x);
        assert!(wanted != 0.0, "expected a nonzero reference: {source:?}");
        let error = ((observed - wanted) / wanted).abs();
        assert!(
            observed.is_finite() && error <= 1e-8,
            "m={m},v={v},x={x}: got={observed:.17e} expected={wanted:.17e} rel={error:.3e}"
        );
        count += 1;
    }
    assert_eq!(count, 72);
}
