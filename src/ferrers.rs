//! Real-degree Ferrers associated Legendre functions.

use std::f64::consts::PI;

const MAX_DEGREE: f64 = 100_000.0;
const MAX_ORDER: u32 = 4096;
const MAX_SERIES: usize = 20_000;

/// The real Ferrers associated Legendre function `P_v^m(x)`, including the
/// Condon–Shortley phase for integer order `m`.
///
/// # Definition and conventions
/// For `m >= 0`, with `z=(1-x)/2`,
///
/// `P_v^m(x) = [(-v)_m (v+1)_m / (2^m m!)] (1-x²)^(m/2) ₂F₁(m-v,m+v+1;m+1;z)`.
///
/// For `m < 0`, the hypergeometric continuation is
///
/// `P_v^(-q)(x) = ((1-x)/(1+x))^(q/2) ₂F₁(-v,v+1;q+1;z)/q!`.
///
/// The degree symmetry `P_{-v-1}^m = P_v^m` is used for negative degrees.
/// Values are dimensionless and not spherically normalized.
///
/// # Domain and special values
/// Finite real degrees `|v| <= 100000`, integer orders `|m| <= 4096`,
/// and real `-1 <= x <= 1`. Integer nonnegative degrees use the polynomial
/// evaluator. Noninteger degrees have `P_v(1)=1` and `P_v^m(1)=0` for
/// nonzero integer `m`. At `x=-1`, noninteger-degree functions are singular.
/// Invalid arguments or an ill-conditioned, nonconvergent series return NaN.
/// Values outside the representable double-precision range may overflow.
///
/// # Numerical behavior
/// A hypergeometric series near `x=1`, its logarithmic continuation near
/// `x=-1`, and the three-term degree recurrence are combined. Recurrences
/// require at most 100000 degree steps and series at most 20000 terms.
/// Large orders, highly oscillatory solutions and near zeros may lose
/// relative precision.
///
/// # Examples
/// ```
/// assert!((orthogonal::lpmv(1, 2.0, 0.3) + 3.0*0.3*(1.0_f64-0.3*0.3).sqrt()).abs() < 1e-14);
/// assert!((orthogonal::lpmv(0, 0.5, 0.3) - 0.7009385309696583).abs() < 1e-14);
/// ```
///
/// # References
/// NIST DLMF §§14.3 and 14.10; [SciPy lpmv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.lpmv.html).
pub fn lpmv(m: i32, v: f64, x: f64) -> f64 {
    if !v.is_finite()
        || !x.is_finite()
        || x.abs() > 1.0
        || v.abs() > MAX_DEGREE
        || m.unsigned_abs() > MAX_ORDER
    {
        return f64::NAN;
    }
    // Ferrers symmetry also maps negative integral degrees to polynomials.
    let degree = if v < -0.5 { -v - 1.0 } else { v };
    if degree >= 0.0 && degree.fract() == 0.0 {
        return super::assoc_legendre_p(degree as i32, m, x).unwrap_or(f64::NAN);
    }
    if x == 1.0 {
        return if m == 0 { 1.0 } else { 0.0 };
    }
    if x == -1.0 {
        if m >= 0 {
            return f64::INFINITY.copysign(-(PI * degree).sin());
        }
        let order = m.unsigned_abs() as f64;
        let sign = if degree < order {
            1.0
        } else if ((degree - order).floor() as i64) % 2 == 0 {
            -1.0
        } else {
            1.0
        };
        return f64::INFINITY.copysign(sign);
    }

    let order = m.unsigned_abs() as usize;
    if m < 0 && order <= 128 {
        // Normalize the positive-order function by the exact gamma ratio.
        // This avoids an unstable high-degree recurrence for negative orders
        // and the slowly decaying Gauss tail close to x=-1.
        let positive = positive_order(degree, order, x);
        if positive.is_finite() && positive != 0.0 {
            let mut log_ratio = 0.0;
            let mut sign = if order.is_multiple_of(2) { 1.0 } else { -1.0 };
            for j in 1..=order {
                let factor = degree - order as f64 + j as f64;
                if factor == 0.0 {
                    return f64::NAN;
                }
                sign *= factor.signum();
                log_ratio -= factor.abs().ln();
                log_ratio -= (degree + j as f64).ln();
            }
            return sign * (positive.abs().ln() + log_ratio).exp() * positive.signum();
        }
    }
    if m < 0 {
        return degree_recurrence(degree, order, x, false);
    }
    positive_order(degree, order, x)
}

fn positive_order(degree: f64, order: usize, x: f64) -> f64 {
    if x < -0.35 {
        if degree <= 10.0 {
            return raised_order(degree, order, x);
        }
        let p0 = degree_recurrence(degree, 0, x, true);
        if order == 0 {
            return p0;
        }
        let p1 = degree_recurrence(degree, 1, x, true);
        return raise_from_two(degree, order, x, p0, p1);
    }
    let fractional = if degree < 0.0 { degree } else { degree.fract() };
    let integer = degree.max(0.0).floor() as usize;
    if integer <= order + 1 {
        return gauss_seed(degree, order, x, true);
    }
    // Seed immediately above the order, so the degree recurrence does not
    // cross its nearly singular denominator at degree = m-1.
    let start = order;
    let (p0, p1) = (
        gauss_seed(start as f64 + fractional, order, x, true),
        gauss_seed(start as f64 + fractional + 1.0, order, x, true),
    );
    advance_degree(integer, start, fractional, order as i32, x, p0, p1)
}

fn degree_recurrence(degree: f64, order: usize, x: f64, positive: bool) -> f64 {
    let integer = degree.max(0.0).floor() as usize;
    let fractional = if degree < 0.0 { degree } else { degree.fract() };
    let (p0, p1) = if positive {
        (
            raised_order(fractional, order, x),
            raised_order(fractional + 1.0, order, x),
        )
    } else {
        (
            gauss_seed(fractional, order, x, false),
            gauss_seed(fractional + 1.0, order, x, false),
        )
    };
    advance_degree(
        integer,
        0,
        fractional,
        if positive {
            order as i32
        } else {
            -(order as i32)
        },
        x,
        p0,
        p1,
    )
}

fn advance_degree(
    integer: usize,
    start: usize,
    fractional: f64,
    order: i32,
    x: f64,
    mut p0: f64,
    mut p1: f64,
) -> f64 {
    if integer == start {
        return p0;
    }
    for n in start + 2..=integer {
        let v = n as f64 + fractional;
        let next = ((2.0 * v - 1.0) * x * p1 - (v + order as f64 - 1.0) * p0) / (v - order as f64);
        p0 = p1;
        p1 = next;
    }
    p1
}

// A real digamma function needed only at noninteger arguments close to zero,
// where the logarithmic continuation carries an explicit gamma pole.
fn digamma(mut z: f64) -> f64 {
    let mut shift = 0.0;
    while z < 8.0 {
        shift -= 1.0 / z;
        z += 1.0;
    }
    let inverse = 1.0 / z;
    let inverse_squared = inverse * inverse;
    shift + z.ln()
        - 0.5 * inverse
        - inverse_squared
            * (1.0 / 12.0
                - inverse_squared
                    * (1.0 / 120.0 - inverse_squared * (1.0 / 252.0 - inverse_squared / 240.0)))
}

// Compute the regular P_v and its first Condon–Shortley order from a
// logarithmic continuation of 2F1(-v,v+1;1;1-t) (DLMF 15.8.10).
fn near_minus_one(v: f64, x: f64) -> (f64, f64) {
    let t = 0.5 * (1.0 + x);
    let a = -v;
    let b = v + 1.0;
    let mut coeff = 1.0;
    let mut bracket = 2.0 * digamma(1.0) - digamma(a) - digamma(b) - t.ln();
    let mut sum = bracket;
    let mut derivative = -1.0;
    for k in 1..=MAX_SERIES {
        let kf = k as f64;
        coeff *= (a + kf - 1.0) * (b + kf - 1.0) * t / (kf * kf);
        bracket += 2.0 / kf - 1.0 / (a + kf - 1.0) - 1.0 / (b + kf - 1.0);
        let term = coeff * bracket;
        let slope = coeff * (kf * bracket - 1.0);
        sum += term;
        derivative += slope;
        if !sum.is_finite() || !derivative.is_finite() {
            return (f64::NAN, f64::NAN);
        }
        if k >= 8
            && term.abs() <= f64::EPSILON * sum.abs()
            && slope.abs() <= f64::EPSILON * derivative.abs()
        {
            break;
        }
    }
    let factor = -(PI * v).sin() / PI;
    let p = factor * sum;
    let p1 = -0.5 * ((1.0 - x) * (1.0 + x)).sqrt() * factor * derivative / t;
    (p, p1)
}

fn raised_order(v: f64, order: usize, x: f64) -> f64 {
    let (p0, p1) = near_minus_one(v, x);
    if order == 0 {
        return p0;
    }
    if order == 1 {
        return p1;
    }
    // The derivative of t^k (D_k - ln(t)) can be evaluated exactly for
    // integer order. The terms k<m form a finite singular polynomial;
    // k>=m give a regular logarithmic remainder (DLMF 15.8.10).
    // Factor out (m-1)! * ((1-t)/t)^(m/2) to keep the series scaled.
    let t = 0.5 * (1.0 + x);
    let mf = order as f64;
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 0..order - 1 {
        let kf = k as f64;
        term *= -((kf - v) * (kf + v + 1.0) * t) / ((kf + 1.0) * (mf - kf - 1.0));
        sum += term;
    }
    // A_m is the coefficient of t^m in 2F1(-v,v+1;1;t).
    let mut a_m = 1.0;
    let mut d_m = 2.0 * digamma(1.0) - digamma(-v) - digamma(v + 1.0);
    let mut harmonic = 0.0;
    for k in 1..=order {
        let kf = k as f64;
        a_m *= (-v + kf - 1.0) * (v + kf) * t / (kf * kf);
        d_m += 2.0 / kf - 1.0 / (-v + kf - 1.0) - 1.0 / (v + kf);
        harmonic += 1.0 / kf;
    }
    let mut regular_weight = mf * a_m;
    for k in order..order + MAX_SERIES {
        let kf = k as f64;
        let correction = regular_weight * (d_m - t.ln() - harmonic);
        sum += if order.is_multiple_of(2) {
            correction
        } else {
            -correction
        };
        if !sum.is_finite() {
            return f64::NAN;
        }
        if k >= order + 8 && correction.abs() <= 2.0 * f64::EPSILON * sum.abs() {
            break;
        }
        regular_weight *= ((kf - v) * (kf + v + 1.0) * t) / ((kf + 1.0) * (kf + 1.0 - mf));
        d_m += 2.0 / (kf + 1.0) - 1.0 / (kf - v) - 1.0 / (kf + v + 1.0);
        harmonic += 1.0 / (kf + 1.0) - 1.0 / (kf + 1.0 - mf);
    }
    let mut log_factor = 0.5 * mf * ((1.0 - t).ln() - t.ln());
    for k in 1..order {
        log_factor += (k as f64).ln();
    }
    let coefficient = -(PI * v).sin() / PI;
    (log_factor + sum.abs().ln()).exp() * coefficient * sum.signum()
}

fn raise_from_two(v: f64, order: usize, x: f64, mut p0: f64, mut p1: f64) -> f64 {
    if order == 0 {
        return p0;
    }
    let root = ((1.0 - x) * (1.0 + x)).sqrt();
    for m in 1..order {
        let mf = m as f64;
        let p2 = -2.0 * mf * x / root * p1 - (v + mf) * (v - mf + 1.0) * p0;
        p0 = p1;
        p1 = p2;
    }
    p1
}

fn gauss_seed(v: f64, order: usize, x: f64, positive: bool) -> f64 {
    let z = 0.5 * (1.0 - x);
    let mf = order as f64;
    let (a, b, c) = if positive {
        (mf - v, mf + v + 1.0, mf + 1.0)
    } else {
        (-v, v + 1.0, mf + 1.0)
    };
    let mut log_coefficient = 0.0;
    let mut sign = 1.0;
    if positive {
        for k in 0..order {
            let kf = k as f64;
            let numerator = (kf - v) * (kf + v + 1.0);
            if numerator == 0.0 {
                return 0.0;
            }
            sign *= numerator.signum();
            log_coefficient += numerator.abs().ln() - (2.0 * (kf + 1.0)).ln();
        }
        log_coefficient += 0.5 * mf * ((1.0 - x) * (1.0 + x)).ln();
    } else {
        for k in 1..=order {
            log_coefficient -= (k as f64).ln();
        }
        log_coefficient += 0.5 * mf * ((1.0 - x).ln() - (1.0 + x).ln());
    }
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut converged = false;
    for k in 0..MAX_SERIES {
        let kf = k as f64;
        term *= (a + kf) * (b + kf) * z / ((c + kf) * (kf + 1.0));
        let next = sum + term;
        if !next.is_finite() || !term.is_finite() {
            return f64::NAN;
        }
        if term.abs() <= 2.0 * f64::EPSILON * next.abs() {
            sum = next;
            converged = true;
            break;
        }
        sum = next;
    }
    if !converged || sum == 0.0 {
        return f64::NAN;
    }
    sign * sum.signum() * (log_coefficient + sum.abs().ln()).exp()
}
