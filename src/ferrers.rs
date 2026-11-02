//! Limited real-degree Ferrers associated Legendre function.

/// Evaluate `P_v^m(x)` with Condon–Shortley phase for real degree `v` and
/// integer order `m`.
///
/// For noninteger `v`, the defining Gauss hypergeometric series is used:
///
/// `P_v^m(x) = ((-v)_m(v+1)_m / (2^m m!)) (1-x²)^(m/2) ₂F₁(-v+m,v+m+1;m+1;(1-x)/2)`
/// for `m>=0`; negative orders use the regularized continuation.
/// For integral `v>=0`, the standard stable polynomial recurrence is used.
///
/// # Domain and special values
///
/// Integral degree: `0<=v<=1_000_000`, `-1<=x<=1`, with integer order.
/// Noninteger degree: `|v|<=32`, `|m|<=12`, `-0.9<=x<=1`.
/// Values outside the supported real domain or whose bounded hypergeometric
/// series fails to converge return NaN. Noninteger degrees near the negative branch endpoint are excluded because
/// direct series summation is ill-conditioned. The positive endpoint x=1
/// is regular: P_v(1)=1 and P_v^m(1)=0 for nonzero integer m.
/// Values are dimensionless and are not spherically normalized.
///
/// # Numerical behavior
///
/// A double-precision recurrence terminates after at most 20,000 terms;
/// loss of relative precision is possible near zeros or for noninteger degrees.
///
/// # References
/// NIST DLMF 14.3 and [SciPy lpmv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.lpmv.html).
///
/// # Examples
/// ```
/// assert!((orthogonal::lpmv(1, 2.0, 0.3) + (1.0_f64 - 0.3*0.3).sqrt()*3.0*0.3).abs() < 1e-14);
/// assert!((orthogonal::lpmv(0, 0.0, 0.3) - 1.0).abs() < 1e-14);
/// ```
pub fn lpmv(m: i32, v: f64, x: f64) -> f64 {
    if !v.is_finite() || !x.is_finite() || x.abs() > 1.0 {
        return f64::NAN;
    }
    if v >= 0.0 && v.fract() == 0.0 && v <= 1_000_000.0 {
        return super::assoc_legendre_p(v as i32, m, x).unwrap_or(f64::NAN);
    }
    // Ferrers symmetry P_{-v-1}^m = P_v^m.
    let v = if v < -0.5 { -v - 1.0 } else { v };
    if v < -0.5 || v > 32.0 || m.unsigned_abs() > 12 || x < -0.9 {
        return f64::NAN;
    }
    let order = m.unsigned_abs() as usize;
    let z = (1.0 - x) * 0.5;
    let mut coefficient = 1.0;
    let (a, b, c) = if m >= 0 {
        for j in 0..order {
            let j = j as f64;
            coefficient *= (-v + j) * (v + 1.0 + j) / (2.0 * (j + 1.0));
        }
        coefficient *= ((1.0 - x) * (1.0 + x)).powf(order as f64 * 0.5);
        (
            -v + order as f64,
            v + order as f64 + 1.0,
            order as f64 + 1.0,
        )
    } else {
        for j in 1..=order {
            coefficient /= j as f64;
        }
        coefficient *= ((1.0 - x) / (1.0 + x)).powf(order as f64 * 0.5);
        (-v, v + 1.0, order as f64 + 1.0)
    };
    if coefficient == 0.0 {
        return 0.0;
    }
    let mut term = 1.0;
    let mut sum = 1.0_f64;
    for k in 0..20_000 {
        let k = k as f64;
        term *= ((a + k) * (b + k) * z) / ((c + k) * (k + 1.0));
        if !term.is_finite() || !sum.is_finite() {
            return f64::NAN;
        }
        let next = sum + term;
        if term.abs() <= 4.0 * f64::EPSILON * next.abs().max(1e-300) || next == sum {
            return coefficient * next;
        }
        sum = next;
    }
    f64::NAN
}
