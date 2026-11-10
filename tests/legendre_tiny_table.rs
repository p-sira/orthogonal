//! Independent 100-digit finite-binomial references for the tiny-x Legendre table.
//! Regenerate the separate input and expected CSVs with
//! `python3 tests/scipy/generate_legendre_tiny.py`.

#[test]
fn legendre_tiny_table_matches_high_precision_binomial_series() {
    let inputs = include_str!("fixtures/legendre_tiny_inputs.csv");
    let expected = include_str!("fixtures/legendre_tiny_expected.csv");
    let mut last_x = None;
    let mut table = Vec::new();
    let mut count = 0;
    for (input, reference) in inputs.lines().skip(1).zip(expected.lines().skip(1)) {
        let (degree, x) = input.split_once(',').unwrap();
        let degree: usize = degree.parse().unwrap();
        let x: f64 = x.parse().unwrap();
        if last_x != Some(x) {
            table = orthogonal::legendre_p_all(4096, x).unwrap();
            last_x = Some(x);
        }
        let expected: f64 = reference.parse().unwrap();
        let actual = table[degree];
        assert!(
            (actual - expected).abs() < 3e-13 * expected.abs().max(1e-30),
            "n={degree}, x={x:e}: actual={actual:.17e}, expected={expected:.17e}"
        );
        count += 1;
    }
    assert_eq!(count, 102);
}
