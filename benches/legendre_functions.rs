//! Benchmarks for associated Legendre functions and spherical harmonics.
use divan::black_box;

fn main() {
    divan::main();
}

const TABLE_SIZES: &[usize] = &[8, 32, 128];

mod scalar {
    use super::*;

    #[divan::bench(args = [(10, 3), (100, 50)])]
    fn assoc_legendre_p((n, m): (u32, i32)) -> f64 {
        orthogonal::assoc_legendre_p(black_box(n), black_box(m), black_box(0.4_f64)).unwrap()
    }

    #[divan::bench(args = [(10, 3), (100, 50)])]
    fn sph_legendre_p((n, m): (u32, i32)) -> f64 {
        orthogonal::sph_legendre_p(black_box(n), black_box(m), black_box(1.1_f64)).unwrap()
    }

    #[divan::bench(args = [(10, 3), (100, 50)])]
    fn sph_harm_y((n, m): (u32, i32)) -> num_complex::Complex<f64> {
        orthogonal::sph_harm_y(
            black_box(n),
            black_box(m),
            black_box(1.1_f64),
            black_box(0.7),
        )
        .unwrap()
    }

    #[divan::bench(args = [(3, 10.0), (20, 55.5)])]
    fn lpmv((m, v): (i32, f64)) -> f64 {
        orthogonal::lpmv(black_box(m), black_box(v), black_box(0.4))
    }
}

mod table {
    use super::*;

    #[divan::bench(args = TABLE_SIZES)]
    fn assoc_legendre_p_all(n: usize) -> Vec<Vec<f64>> {
        orthogonal::assoc_legendre_p_all(black_box(n), black_box(n), black_box(0.4)).unwrap()
    }

    #[divan::bench(args = TABLE_SIZES)]
    fn sph_legendre_p_all(n: usize) -> Vec<Vec<f64>> {
        orthogonal::sph_legendre_p_all(black_box(n), black_box(n), black_box(1.1)).unwrap()
    }

    #[divan::bench(args = TABLE_SIZES)]
    fn sph_harm_y_all(n: usize) -> Vec<Vec<num_complex::Complex64>> {
        orthogonal::sph_harm_y_all(black_box(n), black_box(n), black_box(1.1), black_box(0.7))
            .unwrap()
    }

    #[divan::bench(args = TABLE_SIZES)]
    fn lqn(n: usize) -> (Vec<f64>, Vec<f64>) {
        orthogonal::lqn(black_box(n), black_box(0.4)).unwrap()
    }

    #[divan::bench(args = [8, 32])]
    fn lqmn(n: usize) -> orthogonal::ValueDerivativeMatrices {
        orthogonal::lqmn(black_box(n), black_box(n), black_box(0.4)).unwrap()
    }
}
