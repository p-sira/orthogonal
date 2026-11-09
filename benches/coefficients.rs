//! Benchmarks for coefficient construction and Gaussian quadrature (`roots_*`).
use divan::black_box;

fn main() {
    divan::main();
}

const COEFF_DEGREES: &[usize] = &[5, 20];
const QUAD_DEGREES: &[usize] = &[10, 100, 500];

mod coefficients {
    use super::*;

    #[divan::bench(args = [64, 128, 256])]
    fn legendre_large(n: usize) -> Vec<f64> {
        orthogonal::legendre(black_box(n), false).unwrap()
    }

    #[divan::bench(args = [64, 128, 256])]
    fn jacobi_large(n: usize) -> Vec<f64> {
        orthogonal::jacobi(black_box(n), 0.5, 0.25, false).unwrap()
    }

    #[divan::bench(args = COEFF_DEGREES)]
    fn legendre(n: usize) -> Vec<f64> {
        orthogonal::legendre(black_box(n), false).unwrap()
    }

    #[divan::bench(args = COEFF_DEGREES)]
    fn jacobi(n: usize) -> Vec<f64> {
        orthogonal::jacobi(black_box(n), 0.5, 0.25, false).unwrap()
    }

    #[divan::bench(args = COEFF_DEGREES)]
    fn hermite(n: usize) -> Vec<f64> {
        orthogonal::hermite(black_box(n), false).unwrap()
    }

    #[divan::bench(args = COEFF_DEGREES)]
    fn laguerre(n: usize) -> Vec<f64> {
        orthogonal::laguerre(black_box(n), false).unwrap()
    }

    #[divan::bench(args = COEFF_DEGREES)]
    fn chebyt(n: usize) -> Vec<f64> {
        orthogonal::chebyt(black_box(n), false).unwrap()
    }
}

mod quadrature {
    use super::*;

    #[divan::bench(args = QUAD_DEGREES)]
    fn roots_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
        orthogonal::roots_legendre(black_box(n)).unwrap()
    }

    #[divan::bench(args = QUAD_DEGREES)]
    fn roots_jacobi(n: usize) -> (Vec<f64>, Vec<f64>) {
        orthogonal::roots_jacobi(black_box(n), 0.5, 0.25).unwrap()
    }

    #[divan::bench(args = QUAD_DEGREES)]
    fn roots_gegenbauer(n: usize) -> (Vec<f64>, Vec<f64>) {
        orthogonal::roots_gegenbauer(black_box(n), 0.75).unwrap()
    }

    #[divan::bench(args = QUAD_DEGREES)]
    fn roots_laguerre(n: usize) -> (Vec<f64>, Vec<f64>) {
        orthogonal::roots_laguerre(black_box(n)).unwrap()
    }

    #[divan::bench(args = QUAD_DEGREES)]
    fn roots_genlaguerre(n: usize) -> (Vec<f64>, Vec<f64>) {
        orthogonal::roots_genlaguerre(black_box(n), 1.5).unwrap()
    }

    #[divan::bench(args = QUAD_DEGREES)]
    fn roots_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
        orthogonal::roots_hermite(black_box(n)).unwrap()
    }

    #[divan::bench(args = QUAD_DEGREES)]
    fn roots_chebyt(n: usize) -> (Vec<f64>, Vec<f64>) {
        orthogonal::roots_chebyt(black_box(n)).unwrap()
    }
}
