//! Integer-degree classical orthogonal polynomials.
//!
//! Kernels follow xsf/orthogonal_eval.h and evaluate in double precision.
//! Degrees with absolute value above 1,000,000 return an error to bound recurrence work.
use crate::{Error, Result, finish};
#[allow(unused_imports)]
use num_traits::Float;

macro_rules! polynomial {
    ($(#[$meta:meta])* $name:ident($n:ident, $($arg:ident),+) $body:block) => {
        $(#[$meta])*
        ///
        /// Returns an error for NaN input, undefined results, or |n| > 1,000,000.
        pub fn $name<T: Float>($n: i32, $($arg: T),+) -> Result<T> {
            if $n.unsigned_abs() > 1_000_000 {
                return Err(Error::InvalidInput(concat!(stringify!($name), ": Invalid argument or degree exceeds 1,000,000.")));
            }
            $(if $arg.is_nan() {
                return Err(Error::InvalidInput(concat!(stringify!($name), ": Arguments cannot be NAN.")));
            }
            let $arg = $arg.to_f64().ok_or(Error::InvalidInput("Polynomial: Cannot represent argument as f64."))?;)+
            let value: f64 = $body;
            finish(value, Error::InvalidInput(concat!(stringify!($name), ": Undefined real result.")))
        }
    };
}

fn hermite(n: i32, x: f64, scale: f64) -> f64 {
    if n < 0 {
        return f64::NAN;
    }
    if n == 0 {
        return 1.0;
    }
    if x.is_infinite() {
        return if n % 2 == 0 { f64::INFINITY } else { x };
    }
    // Odd polynomials vanish exactly at zero, including after even terms overflow.
    if x == 0.0 {
        if n % 2 != 0 {
            return x;
        }
        return (1..n)
            .step_by(2)
            .fold(1.0, |value, k| -scale * k as f64 * value);
    }
    if n == 10 && scale == 1.0 && x.abs() <= 3.0 {
        // He_10(x) is even. Evaluating its degree-five polynomial in y=x²
        // avoids nine dependent Clenshaw steps. Center the polynomial in
        // three ranges to limit cancellation near the roots; retain Clenshaw
        // outside this interval, where the large polynomial terms grow.
        let y = x * x;
        if y < 1.0 {
            return ((((y - 45.0) * y + 630.0) * y - 3150.0) * y + 4725.0) * y - 945.0;
        }
        if y < 4.0 {
            let z = y - 2.0;
            return ((((z - 35.0) * z + 310.0) * z - 370.0) * z - 1675.0) * z + 257.0;
        }
        let z = y - 6.0;
        return ((((z - 15.0) * z - 90.0) * z + 630.0) * z + 2565.0) * z - 459.0;
    }
    let (mut previous, mut current) = (0.0, 1.0);
    for k in (2..=n).rev() {
        let next = scale * x * current - scale * k as f64 * previous;
        previous = current;
        current = next;
    }
    scale * x * current - scale * previous
}

#[inline]
fn chebyshev(n: i32, x: f64, second: bool) -> f64 {
    if n == 10 && x.abs() < 1.0 {
        // For the common even degree, reflection does not change the value.
        // Apply T_10 = 2*T_5² - 1 and U_10 = 2*T_5*U_5 - 1 directly.
        let y = x * x;
        let t5 = x * ((16.0 * y - 20.0) * y + 5.0);
        let partner = if second {
            x * ((32.0 * y - 32.0) * y + 6.0)
        } else {
            t5
        };
        return 2.0 * t5 * partner - 1.0;
    }
    let (n, sign) = if n < 0 && second {
        if n == -1 {
            return 0.0;
        }
        (-n - 2, -1.0)
    } else {
        (n.abs(), 1.0)
    };
    if n == 0 {
        return sign;
    }
    if x.is_infinite() {
        return sign * if n % 2 == 0 { f64::INFINITY } else { x };
    }
    if x.abs() == 1.0 {
        let parity = if x < 0.0 && n % 2 != 0 { -1.0 } else { 1.0 };
        return sign * parity * if second { n as f64 + 1.0 } else { 1.0 };
    }
    let parity = if x < 0.0 && n % 2 != 0 { -1.0 } else { 1.0 };
    let x = x.abs();
    if n >= 64 && x < 1.0 {
        // Binary doubling computes (T_k(x), U_{k-1}(x)) in O(log n).
        // Keeping both polynomials avoids division by 1-x² near endpoints.
        // The short recurrence below retains its low overhead for small n.
        let target = if second { n + 1 } else { n } as u32;
        let mut t = 1.0;
        let mut u = 0.0;
        let mut bit = 1 << (31 - target.leading_zeros());
        let delta = (x - 1.0) * (x + 1.0);
        while bit != 0 {
            // (T_{2k}, U_{2k-1}) from (T_k, U_{k-1}).
            let doubled_t = 2.0 * t * t - 1.0;
            let doubled_u = 2.0 * t * u;
            if target & bit != 0 {
                // T_{k+1}=x T_k+(x²-1) U_{k-1}; U_k=T_k+x U_{k-1}.
                t = x * doubled_t + delta * doubled_u;
                u = doubled_t + x * doubled_u;
            } else {
                t = doubled_t;
                u = doubled_u;
            }
            bit >>= 1;
        }
        return sign * parity * if second { u } else { t };
    }
    let (mut previous, mut current) = (1.0, if second { 2.0 * x } else { x });
    let twice = 2.0 * x;
    if x < 1.0 {
        // |T_n|, |U_n| <= n + 1 on [-1, 1], so the recurrence cannot overflow.
        for _ in 1..n {
            (previous, current) = (current, twice * current - previous);
        }
    } else {
        for _ in 1..n {
            (previous, current) = (current, twice * current - previous);
            // For x > 1 the magnitudes grow monotonically; the final sign is known.
            if current.is_infinite() {
                break;
            }
        }
    }
    sign * parity * current
}

polynomial! {
    /// Physicists' Hermite polynomial H_n(x), for n >= 0.
    eval_hermite(n, x) { hermite(n, x, 2.0) }
}
polynomial! {
    /// Probabilists' Hermite polynomial He_n(x), for n >= 0.
    eval_hermitenorm(n, x) { hermite(n, x, 1.0) }
}
polynomial! {
    /// Chebyshev polynomial T_n(x), with T_{-n} = T_n.
    #[inline]
    eval_chebyt(n, x) { chebyshev(n, x, false) }
}
polynomial! {
    /// Chebyshev polynomial U_n(x), with U_{-1} = 0 and U_{-n-2} = -U_n.
    #[inline]
    eval_chebyu(n, x) { chebyshev(n, x, true) }
}

// Precompute the first recurrence ratios once. The first 64 degrees are
// especially common, and using multiplication avoids division in each step.
// Higher degrees continue with the ordinary recurrence.
const LEGENDRE_RECURRENCE_RATIOS: [(f64, f64); 64] = {
    let mut ratios = [(0.0, 0.0); 64];
    let mut i = 0;
    while i < ratios.len() {
        let k = (i + 1) as f64;
        ratios[i] = ((2.0 * k + 1.0) / (k + 1.0), k / (k + 1.0));
        i += 1;
    }
    ratios
};

fn legendre(n: i32, x: f64) -> f64 {
    let n = if n < 0 { -(n + 1) } else { n };
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }
    let sign = if x < 0.0 && n % 2 != 0 { -1.0 } else { 1.0 };
    let x = x.abs();
    if x == 1.0 {
        return sign;
    }
    if x.is_infinite() {
        return sign * f64::INFINITY;
    }
    if x < 1e-5 {
        // xsf's power series avoids subtracting adjacent, nearly equal values.
        let a = n / 2;
        let mut term = if a % 2 == 0 { 1.0 } else { -1.0 };
        term *= if n % 2 == 0 {
            -2.0 / crate::signed_half_beta(a as f64 + 1.0, -0.5)
        } else {
            2.0 * x / crate::signed_half_beta(a as f64 + 1.0, 0.5)
        };
        let mut sum = 0.0;
        for k in 0..=a {
            sum += term;
            term *= -2.0 * x * x * (a - k) as f64 * (2 * n + 1 - 2 * a + 2 * k) as f64
                / ((n + 1 - 2 * a + 2 * k) as f64 * (n + 2 - 2 * a + 2 * k) as f64);
            if term.abs() <= f64::EPSILON * sum.abs() {
                break;
            }
        }
        return sign * sum;
    }
    if x < 0.9 {
        // The ordinary three-term recurrence is stable well inside [-1, 1].
        // It is substantially cheaper than the difference recurrence below,
        // which is retained near x=1 to avoid cancellation of P_n and P_{n-1}.
        let (mut previous, mut value) = (1.0, x);
        let table_steps = ((n - 1) as usize).min(LEGENDRE_RECURRENCE_RATIOS.len());
        for &(a, b) in &LEGENDRE_RECURRENCE_RATIOS[..table_steps] {
            let next = a * x * value - b * previous;
            previous = value;
            value = next;
        }
        for k in (table_steps as i32 + 1)..n {
            let k = k as f64;
            let next = ((2.0 * k + 1.0) * x * value - k * previous) / (k + 1.0);
            previous = value;
            value = next;
        }
        return sign * value;
    }
    let (mut difference, mut value) = (x - 1.0, x);
    for k in 1..n {
        let k = k as f64;
        difference = ((2.0 * k + 1.0) * (x - 1.0) * value + k * difference) / (k + 1.0);
        value += difference;
        if value.is_infinite() {
            break;
        }
    }
    sign * value
}

/// P_0(x), ..., P_n(x) in one pass of the recurrence used by [`legendre`];
/// entries match scalar evaluation within floating-point rounding. For tiny
/// |x| and n <= 4096, the scalar near-zero series is retained. Larger tables
/// use the O(n) three-term recurrence to avoid quadratic work.
#[inline]
pub(crate) fn legendre_all(n: usize, x: f64) -> Vec<f64> {
    let per_degree = |k: usize| {
        if x.is_nan() {
            f64::NAN
        } else {
            legendre(k as i32, x)
        }
    };
    let a = x.abs();
    if x.is_nan() || (a < 1e-5 && n <= 4096) || a == 1.0 || a.is_infinite() || n < 2 {
        return (0..=n).map(per_degree).collect();
    }
    if a < 0.9 {
        // Fill a fixed-length output slice instead of checking a Vec's spare
        // capacity on every degree. The signed recurrence already has the
        // required parity, so no per-degree sign correction is necessary.
        let mut table = vec![0.0; n + 1];
        table[0] = 1.0;
        table[1] = x;
        let (mut previous, mut value) = (1.0, x);
        let table_steps = (n - 1).min(LEGENDRE_RECURRENCE_RATIOS.len());
        for (slot, &(coefficient, prior_coefficient)) in table[2..table_steps + 2]
            .iter_mut()
            .zip(&LEGENDRE_RECURRENCE_RATIOS[..table_steps])
        {
            let next = coefficient * x * value - prior_coefficient * previous;
            previous = value;
            value = next;
            *slot = value;
        }
        for (offset, slot) in table[table_steps + 2..].iter_mut().enumerate() {
            let k = table_steps + 1 + offset;
            let kf = k as f64;
            let next = ((2.0 * kf + 1.0) * x * value - kf * previous) / (kf + 1.0);
            previous = value;
            value = next;
            *slot = value;
        }
        return table;
    }
    let mut out = Vec::with_capacity(n + 1);
    out.push(1.0);
    out.push(x);
    let (mut difference, mut value) = (a - 1.0, a);
    for k in 1..n {
        if !value.is_infinite() {
            let kf = k as f64;
            difference = ((2.0 * kf + 1.0) * (a - 1.0) * value + kf * difference) / (kf + 1.0);
            value += difference;
        }
        let degree = k + 1;
        let sign = if x < 0.0 && degree % 2 != 0 {
            -1.0
        } else {
            1.0
        };
        out.push(sign * value);
    }
    out
}

fn endpoint_binomial(n: i32, alpha: f64) -> f64 {
    (1..=n).fold(1.0, |value, k| value * ((alpha + k as f64) / k as f64))
}

fn laguerre(n: i32, alpha: f64, x: f64) -> f64 {
    if !alpha.is_finite() || alpha <= -1.0 {
        return f64::NAN;
    }
    if n < 0 {
        return 0.0;
    }
    if n == 0 {
        return 1.0;
    }
    if x.is_infinite() {
        return if x > 0.0 && n % 2 != 0 {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    // Retain the rounding residual of alpha+1 when the linear polynomial cancels.
    let sum = alpha + 1.0;
    let error = if alpha.abs() <= 1.0 {
        alpha - (sum - 1.0)
    } else {
        1.0 - (sum - alpha)
    };
    let initial = (sum - x) + error;
    if n == 1 {
        return initial;
    }
    if alpha == 0.0 {
        // The normalization is exactly one for L_n = L_n^(0).
        // Avoid recomputing its n-factor product on every scalar call.
        if x == 0.0 {
            return 1.0;
        }
        let mut difference = -x;
        let mut value = 1.0 + difference;
        for k in 1..n {
            let k = k as f64;
            difference = -x / (k + 1.0) * value + k / (k + 1.0) * difference;
            value += difference;
        }
        return value;
    }
    let scale = endpoint_binomial(n, alpha);
    if x == 0.0 {
        return scale;
    }
    if scale.is_finite() {
        let mut difference = -x / (alpha + 1.0);
        let mut value = 1.0 + difference;
        for k in 1..n {
            let k = k as f64;
            difference = -x / (k + alpha + 1.0) * value + k / (k + alpha + 1.0) * difference;
            value += difference;
        }
        return scale * value;
    }
    // Avoid normalization overflow times a tiny value by falling back to the
    // unnormalized three-term recurrence when the scale cannot be represented.
    let (mut previous, mut value) = (1.0, initial);
    for k in 1..n {
        let k = k as f64;
        let next = ((2.0 * k + 1.0 + alpha - x) / (k + 1.0)) * value
            - ((k + alpha) / (k + 1.0)) * previous;
        previous = value;
        value = next;
    }
    value
}

fn gegenbauer(n: i32, alpha: f64, x: f64) -> f64 {
    if !alpha.is_finite() {
        return f64::NAN;
    }
    // Match xsf's convention at alpha = 0, including n = 0.
    if alpha == 0.0 || n < 0 {
        return 0.0;
    }
    if n == 0 {
        return 1.0;
    }
    if alpha < 0.0 && alpha == alpha.floor() && n as f64 > -2.0 * alpha {
        return 0.0;
    }
    if x.abs() == 1.0 {
        let value = (0..n).fold(1.0, |value, k| {
            value * ((2.0 * alpha + k as f64) / (k as f64 + 1.0))
        });
        return if x < 0.0 && n % 2 != 0 { -value } else { value };
    }
    let (mut previous, mut value) = (1.0, 2.0 * alpha * x);
    for k in 1..n {
        let k = k as f64;
        let next = (2.0 * (k + alpha) / (k + 1.0)) * x * value
            - (((k - 1.0) + 2.0 * alpha) / (k + 1.0)) * previous;
        previous = value;
        value = next;
    }
    value
}

fn jacobi(n: i32, alpha: f64, beta: f64, x: f64) -> f64 {
    if n < 0 || !alpha.is_finite() || !beta.is_finite() || alpha <= -1.0 || beta <= -1.0 {
        return f64::NAN;
    }
    if n == 0 {
        return 1.0;
    }
    // Reflection preserves parity and uses the positive endpoint consistently.
    if x < 0.0 {
        return if n % 2 == 0 { 1.0 } else { -1.0 } * jacobi(n, beta, alpha, -x);
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }
    if x == 1.0 {
        return endpoint_binomial(n, alpha);
    }
    let sum = alpha + beta;
    let (mut previous, mut value) = (1.0, 0.5 * ((alpha - beta) + (sum + 2.0) * x));
    for k in 1..n {
        let k = k as f64;
        let t = 2.0 * k + sum;
        let denominator = 2.0 * (k + 1.0) * (k + sum + 1.0) * t;
        let next = ((t + 1.0) * (t * (t + 2.0) * x + (alpha - beta) * sum) * value
            - 2.0 * (k + alpha) * (k + beta) * (t + 2.0) * previous)
            / denominator;
        previous = value;
        value = next;
    }
    value
}

// Monic shifted-Jacobi recurrence. Cancelling the binomial normalization in
// each step avoids overflow of both numerator and denominator at large degree.
//
// The weight is x^(q-1) (1-x)^(p-q), so the Jacobi parameters on [-1,1]
// are alpha=p-q, beta=q-1. The usual *monic* Jacobi recurrence transferred
// to x=(t+1)/2 is G_(k+1)=(x-A_k)G_k-B_k G_(k-1), with
// A_k = (1 + (2q-p-1)(p-1)/[(2k+p-1)(2k+p+1)])/2,
// B_k = k(k+p-q)(k+q-1)(k+p-1)/
//       [(2k+p-1)^2(2k+p-2)(2k+p)].
// The k=1 B_k expression has a removable 0/0 at p=0, hence its
// separately simplified form. Factor ratios to avoid overflow at large k.
pub(crate) fn shifted_jacobi_monic(n: i32, p: f64, q: f64, x: f64) -> f64 {
    debug_assert!(n > 0 && q > 0.0 && p - q > -1.0);
    let mut previous = 1.0;
    let mut value = x - q / (p + 1.0);
    for k in 1..n {
        let k = k as f64;
        let t = 2.0 * k + p - 1.0;
        let a = if k == 1.0 {
            // The general expression loses digits when p -> -1.
            (2.0 + (q / (p + 1.0)) * (p - 1.0)) / (p + 3.0)
        } else {
            // Divide first: 2*q can overflow even though the coefficient
            // stays bounded when p and q are both very large.
            let u = t + 2.0;
            0.5 * (1.0 + (2.0 * (q / u) - (p + 1.0) / u) * ((p - 1.0) / t))
        };
        let b = if k == 1.0 {
            let u = p + 1.0;
            (q / u) * ((u - q) / u) / (p + 2.0)
        } else {
            (k / t) * ((k + p - q) / t) * ((k + q - 1.0) / (t - 1.0)) * ((k + p - 1.0) / (t + 1.0))
        };
        let next = (x - a).mul_add(value, -b * previous);
        previous = value;
        value = next;
    }
    value
}

polynomial! {
    /// Legendre polynomial P_n(x), with P_{-n-1} = P_n.
    eval_legendre(n, x) { legendre(n, x) }
}
polynomial! {
    /// Generalized Laguerre polynomial L_n^(alpha)(x), for finite alpha > -1.
    /// Negative degrees return zero.
    eval_genlaguerre(n, alpha, x) { laguerre(n, alpha, x) }
}
polynomial! {
    /// Laguerre polynomial L_n(x). Negative degrees return zero.
    eval_laguerre(n, x) { laguerre(n, 0.0, x) }
}
polynomial! {
    /// Gegenbauer polynomial C_n^(alpha)(x), for finite alpha.
    /// Negative degrees and alpha = 0 return zero, following xsf.
    eval_gegenbauer(n, alpha, x) { gegenbauer(n, alpha, x) }
}
polynomial! {
    /// Jacobi polynomial P_n^(alpha,beta)(x), for n >= 0 and finite alpha,beta > -1.
    eval_jacobi(n, alpha, beta, x) { jacobi(n, alpha, beta, x) }
}
