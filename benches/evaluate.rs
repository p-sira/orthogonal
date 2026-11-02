//! Benchmarks for scalar polynomial evaluation (`eval_*`).
use divan::black_box;
use orthogonal::*;

fn main() {
    divan::main();
}

/// Degrees covering the closed-form, short recurrence and long recurrence paths.
const DEGREES: &[i32] = &[5, 50, 500];

/// Evaluation points in the orthogonality interval [-1, 1].
fn points() -> Vec<f64> {
    (0..64).map(|i| -0.99 + 1.98 * i as f64 / 63.0).collect()
}

/// Evaluation points on the half-line used by Laguerre and Hermite.
fn positive_points() -> Vec<f64> {
    (0..64).map(|i| 0.1 + 10.0 * i as f64 / 63.0).collect()
}

macro_rules! eval_bench {
    ($name:ident, $points:ident, |$n:ident, $x:ident| $call:expr) => {
        #[divan::bench(args = DEGREES)]
        fn $name(bencher: divan::Bencher, $n: i32) {
            let xs = $points();
            bencher.bench_local(|| {
                let $n = black_box($n);
                let mut acc = 0.0;
                for &$x in black_box(&xs) {
                    acc += $call;
                }
                acc
            });
        }
    };
}

eval_bench!(legendre, points, |n, x| eval_legendre(n, x));
eval_bench!(jacobi, points, |n, x| eval_jacobi(n, 0.5, 0.25, x));
eval_bench!(gegenbauer, points, |n, x| eval_gegenbauer(n, 0.75, x));
eval_bench!(chebyt, points, |n, x| eval_chebyt(n, x));
eval_bench!(chebyu, points, |n, x| eval_chebyu(n, x));
eval_bench!(laguerre, positive_points, |n, x| eval_laguerre(n, x));
eval_bench!(genlaguerre, positive_points, |n, x| eval_genlaguerre(
    n, 1.5, x
));
eval_bench!(hermite, points, |n, x| eval_hermite(n, x));
eval_bench!(hermitenorm, points, |n, x| eval_hermitenorm(n, x));
eval_bench!(sh_jacobi, points, |n, x| eval_sh_jacobi(
    n,
    2.5,
    1.5,
    0.5 * (x + 1.0)
));

#[divan::bench(args = [10, 1000, 100_000])]
fn legendre_p_all(n: usize) -> Vec<f64> {
    orthogonal::legendre_p_all(black_box(n), black_box(0.3)).unwrap()
}
