//! # orthogonal
//!
//! Classical real orthogonal polynomials, associated Legendre functions and
//! Gaussian quadrature. Coefficient construction and Gaussian rules use the
//! same three-term recurrences. No Sci crate is needed.
//!
//! All polynomial coefficients are ordered from highest to lowest power;
//! quadrature nodes are sorted increasing. Their weights correspond to the
//! unnormalized orthogonality measure and integrate polynomials through degree
//! `2*n - 1`, subject to rounding.
//!
//! ```
//! let p = orthogonal::legendre(2, false)?;
//! assert_eq!(p, vec![1.5, 0.0, -0.5]);
//! let (x, w) = orthogonal::roots_legendre(3)?;
//! assert_eq!(x.len(), 3);
//! assert!((w.iter().sum::<f64>() - 2.0).abs() < 1e-13);
//! # Ok::<(), orthogonal::Error>(())
//! ```
#![forbid(unsafe_code)]

use num_traits::Float;
use std::fmt;

mod associated;
mod coefficients;
mod evaluate;
mod ferrers;
pub use ferrers::lpmv;

/// Two matrices of real associated Legendre Q values and first derivatives.
pub type ValueDerivativeMatrices = (Vec<Vec<f64>>, Vec<Vec<f64>>);

pub use associated::{assoc_legendre_p, sph_harm_y, sph_legendre_p};
pub use coefficients::*;

/// Failure to construct a polynomial or Gaussian rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Input degree, order, parameter or matrix shape is invalid.
    InvalidInput(&'static str),
    /// Matrix dimensions do not agree.
    DimensionMismatch,
    /// Intermediate numerical values overflowed or were nonfinite.
    NonFiniteEvaluation,
    /// The numerical procedure did not converge within the work limit.
    NumericalFailure(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidInput(text) | Error::NumericalFailure(text) => f.write_str(text),
            Error::DimensionMismatch => f.write_str("input dimensions do not agree"),
            Error::NonFiniteEvaluation => f.write_str("nonfinite numerical evaluation"),
        }
    }
}
impl std::error::Error for Error {}

/// Result returned by checked coefficient constructors and quadrature rules.
pub type Result<T> = std::result::Result<T, Error>;

// The scalar backends deliberately distinguish a NaN (undefined) from an
// infinity (which may be a legitimate overflow result for a polynomial).
pub(crate) type StrErr = &'static str;
pub(crate) fn finish<T: Float>(value: f64, error: StrErr) -> std::result::Result<T, StrErr> {
    if value.is_nan() {
        Err(error)
    } else {
        T::from(value).ok_or("floating-point conversion failed")
    }
}

// Lanczos approximation (g=7; real positive arguments only). This local
// implementation replaces the Sci gamma/beta helpers. The quadrature weights
// involve gamma(a+1) or beta(a+1,b+1) for a,b > -1.
fn gammaln(z: f64) -> f64 {
    const C: [f64; 9] = [
        0.9999999999998099,
        676.5203681218851,
        -1259.1392167224028,
        771.3234287776531,
        -176.6150291621406,
        12.507343278686905,
        -0.13857109526572012,
        9.984369578019572e-6,
        1.5056327351493116e-7,
    ];
    if !z.is_finite() || z <= 0.0 {
        return f64::NAN;
    }
    if z < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().ln()
            - gammaln(1.0 - z);
    }
    let zz = z - 1.0;
    let mut sum = C[0];
    for (i, c) in C[1..].iter().enumerate() {
        sum += c / (zz + i as f64 + 1.0);
    }
    let t = zz + 7.5;
    0.918_938_533_204_672_8 + (zz + 0.5) * t.ln() - t + sum.ln()
}

fn gamma(z: f64) -> f64 {
    gammaln(z).exp()
}
fn beta(a: f64, b: f64) -> f64 {
    (gammaln(a) + gammaln(b) - gammaln(a + b)).exp()
}

// In the tiny-x Legendre expansion, the beta argument is an integer and the
// second parameter is +/-1/2. These recurrences avoid signed-gamma plumbing.
fn signed_half_beta(a: f64, b: f64) -> f64 {
    debug_assert!(a >= 1.0 && a.fract() == 0.0 && (b == 0.5 || b == -0.5));
    let mut value = if b < 0.0 { -2.0 } else { 2.0 };
    for k in 1..a as usize {
        let k = k as f64;
        value *= k / (k + b);
    }
    value
}

macro_rules! evaluator {
    ($(#[$meta:meta])* $name:ident($n:ident $(, $arg:ident)*)) => {
        $(#[$meta])*
        #[inline]
        pub fn $name<T: Float>($n: i32, $($arg: T),*) -> T {
            if !(0..=1_000_000).contains(&$n) { return T::nan(); }
            evaluate::$name($n, $($arg),*).unwrap_or_else(|_| T::nan())
        }
    };
}

evaluator! {
/// Jacobi polynomial `P_n^(alpha,beta)(x)`, orthogonal with weight `(1-x)^alpha (1+x)^beta`, `alpha,beta > -1`.
eval_jacobi(n, alpha, beta, x) }
evaluator! {
/// Legendre polynomial `P_n(x)` defined by `P_0=1`, `P_1=x`, `(n+1)P_{n+1}=(2n+1)xP_n-nP_{n-1}`.
eval_legendre(n, x) }
evaluator! {
/// Gegenbauer polynomial `C_n^alpha(x)`, with generating function `(1-2xt+t²)^(-alpha)`.
eval_gegenbauer(n, alpha, x) }
evaluator! {
/// Generalized Laguerre polynomial `L_n^alpha(x)`, orthogonal with weight `x^alpha exp(-x)` for `alpha > -1`.
eval_genlaguerre(n, alpha, x) }
evaluator! {
/// Laguerre polynomial `L_n(x)=L_n^0(x)`.
eval_laguerre(n, x) }
evaluator! {
/// Physicists' Hermite polynomial `H_n(x)` with `H_0=1`, `H_1=2x`, `H_{n+1}=2xH_n-2nH_{n-1}`.
eval_hermite(n, x) }
evaluator! {
/// Probabilists' Hermite polynomial `He_n(x)` with `He_0=1`, `He_1=x`, `He_{n+1}=xHe_n-nHe_{n-1}`.
eval_hermitenorm(n, x) }
evaluator! {
/// Chebyshev polynomial `T_n(x)=cos(n arccos x)` for `-1<=x<=1`.
eval_chebyt(n, x) }
evaluator! {
/// Chebyshev polynomial `U_n(x)=sin((n+1)arccos x)/sin(arccos x)` for `-1<x<1`.
eval_chebyu(n, x) }

/// Scaled Chebyshev first kind, `C_n(x)=2 T_n(x/2)`.
#[inline]
pub fn eval_chebyc(n: i32, x: f64) -> f64 {
    2.0 * eval_chebyt(n, x / 2.0)
}
/// Scaled Chebyshev second kind, `S_n(x)=U_n(x/2)`.
#[inline]
pub fn eval_chebys(n: i32, x: f64) -> f64 {
    eval_chebyu(n, x / 2.0)
}
/// Shifted Chebyshev first kind, `T_n(2x-1)`.
#[inline]
pub fn eval_sh_chebyt(n: i32, x: f64) -> f64 {
    eval_chebyt(n, 2.0 * x - 1.0)
}
/// Shifted Chebyshev second kind, `U_n(2x-1)`.
#[inline]
pub fn eval_sh_chebyu(n: i32, x: f64) -> f64 {
    eval_chebyu(n, 2.0 * x - 1.0)
}
/// Shifted Legendre polynomial, `P_n(2x-1)`.
#[inline]
pub fn eval_sh_legendre(n: i32, x: f64) -> f64 {
    eval_legendre(n, 2.0 * x - 1.0)
}
/// Shifted Jacobi `G_n^(p,q)(x) = n! Γ(n+p)/Γ(2n+p) P_n^(p-q,q-1)(2x-1)`.
/// Requires `0<=n<=1_000_000`, `q>0`, `p-q>-1`, and finite `p,q`.
pub fn eval_sh_jacobi(n: i32, p: f64, q: f64, x: f64) -> f64 {
    if !(0..=1_000_000).contains(&n)
        || !p.is_finite()
        || !q.is_finite()
        || q <= 0.0
        || p - q <= -1.0
    {
        return f64::NAN;
    }
    if n == 0 {
        return 1.0;
    }
    let nf = n as f64;
    (gammaln(nf + 1.0) + gammaln(nf + p) - gammaln(2.0 * nf + p)).exp()
        * eval_jacobi(n, p - q, q - 1.0, 2.0 * x - 1.0)
}

/// All Legendre polynomial values `[P_0(x), ..., P_n(x)]` for `n <= 1_000_000`.
///
/// Takes `O(n)` time and memory; the recurrence is stable on `[-1, 1]`.
/// Outside this interval, values can overflow for large degrees.
/// NaN inputs produce a table of NaNs.
pub fn legendre_p_all(n: usize, x: f64) -> Result<Vec<f64>> {
    if n > 1_000_000 {
        return Err(Error::InvalidInput("degree must not exceed 1,000,000"));
    }
    Ok(evaluate::legendre_all(n, x))
}

/// Alias for the integer-degree Legendre polynomial `P_n(x)`.
#[inline]
pub fn legendre_p<T: Float>(n: i32, x: T) -> T {
    eval_legendre(n, x)
}

/// Values `P_n^m(x)` for degrees `0..=n` and orders `0..=m, -m..=-1`.
///
/// Uses the Condon–Shortley phase. Requires `n,m <= 128` and `-1 <= x <= 1`.
/// Each row is a degree and each column an order (positive first, then negative).
pub fn assoc_legendre_p_all(n: usize, m: usize, x: f64) -> Result<Vec<Vec<f64>>> {
    if n > 128 || m > 128 || !(-1.0..=1.0).contains(&x) {
        return Err(Error::InvalidInput(
            "associated Legendre table dimensions or domain invalid",
        ));
    }
    Ok(associated::assoc_legendre_table(n, m, x))
}

/// Spherically normalized associated Legendre values through degree `n` and
/// order `m`, at polar angle `theta` in radians.
///
/// `Y_n^m(theta,phi) = sph_legendre_p(n,m,theta)*exp(i*m*phi)`.
/// Requires `n,m <= 128` and `0 <= theta <= pi`.
pub fn sph_legendre_p_all(n: usize, m: usize, theta: f64) -> Result<Vec<Vec<f64>>> {
    if n > 128 || m > 128 || !(0.0..=std::f64::consts::PI).contains(&theta) {
        return Err(Error::InvalidInput(
            "spherical Legendre table dimensions or domain invalid",
        ));
    }
    Ok(associated::sph_legendre_table(n, m, theta))
}

/// Complex spherical harmonics `Y_n^m(theta,phi)` for degrees `0..=n` and
/// orders `0..=m, -m..=-1`, including the Condon–Shortley phase.
///
/// Requires `n,m <= 128`, `theta` in `[0,pi]`, and finite azimuth `phi`.
pub fn sph_harm_y_all(
    n: usize,
    m: usize,
    theta: f64,
    phi: f64,
) -> Result<Vec<Vec<num_complex::Complex64>>> {
    if n > 128 || m > 128 || !(0.0..=std::f64::consts::PI).contains(&theta) || !phi.is_finite() {
        return Err(Error::InvalidInput(
            "spherical harmonics table dimensions or domain invalid",
        ));
    }
    Ok(associated::sph_harm_table(n, m, theta, phi))
}

/// Real Legendre functions of the second kind `Q_0(x),...,Q_n(x)` and
/// their derivatives, in ascending degree order.
///
/// On `-1 < x < 1`, the Ferrers branch uses `Q_0(x)=atanh(x)`. On the real
/// exterior `|x| > 1`, `Q_0(x)=atanh(1/x)` is the real-valued continuation;
/// `Q_n(-x)=(-1)^(n+1)Q_n(x)`. The recurrence is
/// `(n+1)Q_(n+1)=(2n+1)x Q_n-n Q_(n-1)` on either branch.
///
/// Requires `n <= 128` and finite `x != ±1`; these endpoints are logarithmic
/// singularities. The exterior branch uses a continued fraction for the
/// minimal solution to prevent the growing Legendre P solution from corrupting
/// high-degree Q values. Results may underflow to zero at very large `|x|`.
///
/// # Examples
/// ```
/// let (q, d) = orthogonal::lqn(1, 0.3)?;
/// assert!((q[0] - 0.3_f64.atanh()).abs() < 1e-14);
/// assert_eq!(q.len(), d.len());
/// let (q, _) = orthogonal::lqn(1, 2.0)?;
/// assert!((q[0] - 0.5 * 3.0_f64.ln()).abs() < 1e-14);
/// # Ok::<(), orthogonal::Error>(())
/// ```
pub fn lqn(n: usize, x: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if n > 128 || !x.is_finite() || x.abs() == 1.0 {
        return Err(Error::InvalidInput(
            "Legendre Q requires n<=128, finite x != ±1",
        ));
    }
    let mut q = vec![0.0; n + 1];
    let mut d = vec![0.0; n + 1];
    let denominator = (1.0 - x) * (1.0 + x);
    if x.abs() < 1.0 {
        q[0] = x.atanh();
        if n > 0 {
            q[1] = x * q[0] - 1.0;
        }
        for k in 2..=n {
            let kf = k as f64;
            q[k] = ((2.0 * kf - 1.0) * x * q[k - 1] - (kf - 1.0) * q[k - 2]) / kf;
        }
    } else {
        let a = x.abs();
        // log1p avoids reciprocal rounding near |x|=1 and avoids catastrophic
        // cancellation in atanh(1/x) as |x| tends to infinity.
        let q0 = 0.5 * (2.0 / (a - 1.0)).ln_1p();
        q[0] = q0;
        let acosh = a.acosh();
        if (n as f64) * acosh <= 1.0 {
            // Very near the logarithmic singularity, the ordinary recurrence
            // is well conditioned over n steps and needs no backward iteration.
            if n > 0 {
                q[1] = a * q0 - 1.0;
            }
            for k in 2..=n {
                let kf = k as f64;
                q[k] = ((2.0 * kf - 1.0) * a * q[k - 1] - (kf - 1.0) * q[k - 2]) / kf;
            }
        } else {
            // Pincherle's minimal-solution continued fraction:
            // Q_k/Q_(k-1) = k / ((2k+1)*x - (k+1)*Q_(k+1)/Q_k).
            // Truncation error decreases exponentially with acosh(x).
            let extra = (20.0 / acosh).ceil().max(32.0) as usize;
            let mut ratios = vec![0.0; n + 1];
            let mut next_ratio = 0.0;
            for k in (1..=n + extra).rev() {
                let kf = k as f64;
                next_ratio = (kf / a) / ((2.0 * kf + 1.0) - (kf + 1.0) * next_ratio / a);
                if k <= n {
                    ratios[k] = next_ratio;
                }
            }
            for k in 1..=n {
                q[k] = q[k - 1] * ratios[k];
            }
        }
        // Q_n(-x)=(-1)^(n+1) Q_n(x).
        if x < 0.0 {
            for (k, value) in q.iter_mut().enumerate() {
                if k % 2 == 0 {
                    *value = -*value;
                }
            }
        }
    }
    d[0] = 1.0 / denominator;
    for k in 1..=n {
        d[k] = (k as f64) * (q[k - 1] - x * q[k]) / denominator;
    }
    if q.iter().chain(&d).any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok((q, d))
}

/// Associated real Ferrers `Q_k^j(x)` and their first derivatives.
///
/// The returned `(values,derivatives)` matrices have `m+1` order rows and
/// `n+1` degree columns, with the associated (Condon–Shortley) phase.
/// Requires `m,n <= 128`, `-1 < x < 1` for positive orders. For `m=0`,
/// also supports the real exterior `|x|>1` as [`lqn`]. At the endpoints,
/// these functions diverge and are undefined.
///
/// # References
/// NIST DLMF Chapter 14 and [SciPy `lqmn`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.lqmn.html).
pub fn lqmn(m: usize, n: usize, x: f64) -> Result<ValueDerivativeMatrices> {
    if n > 128 || m > 128 || !x.is_finite() || x.abs() == 1.0 || (m > 0 && x.abs() > 1.0) {
        return Err(Error::InvalidInput(
            "associated Legendre Q requires m,n<=128; positive orders require -1<x<1",
        ));
    }
    if m == 0 {
        let (values, slopes) = lqn(n, x)?;
        return Ok((vec![values], vec![slopes]));
    }
    let mut values = vec![vec![0.0; n + 1]; m + 1];
    let mut slopes = vec![vec![0.0; n + 1]; m + 1];
    let denominator = (1.0 - x) * (1.0 + x);
    let root = denominator.sqrt();
    let q0 = x.atanh();
    values[0][0] = q0;
    if n > 0 {
        values[0][1] = x * q0 - 1.0;
    }
    if m > 0 {
        values[1][0] = -1.0 / root;
        if n > 0 {
            values[1][1] = -root * (q0 + x / denominator);
        }
    }
    for (order, row) in values.iter_mut().enumerate().take(m.min(1) + 1) {
        for degree in 2..=n {
            let d = degree as f64;
            let o = order as f64;
            row[degree] =
                ((2.0 * d - 1.0) * x * row[degree - 1] - (d + o - 1.0) * row[degree - 2]) / (d - o);
        }
    }
    for order in 2..=m {
        let o = order as f64;
        let (earlier, later) = values.split_at_mut(order);
        let previous = &earlier[order - 1];
        let before = &earlier[order - 2];
        for (degree, out) in later[0].iter_mut().enumerate() {
            let d = degree as f64;
            *out = -2.0 * (o - 1.0) * x / root * previous[degree]
                - (d + o - 1.0) * (d - o + 2.0) * before[degree];
        }
    }
    slopes[0][0] = 1.0 / denominator;
    for degree in 1..=n {
        slopes[0][degree] =
            degree as f64 * (values[0][degree - 1] - x * values[0][degree]) / denominator;
    }
    for order in 1..=m {
        let o = order as f64;
        for degree in 0..=n {
            let d = degree as f64;
            slopes[order][degree] = o * x / denominator * values[order][degree]
                + (o + d) * (d - o + 1.0) / root * values[order - 1][degree];
        }
    }
    if values
        .iter()
        .chain(&slopes)
        .flatten()
        .any(|x| !x.is_finite())
    {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok((values, slopes))
}
