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
    if n >= 64 && (0.99..1.1).contains(&x) {
        // Near x=1, doubling amplifies rounding in 2*T_k²-1, while the
        // linear recurrence accumulates error over many nearly equal terms.
        // acos/acosh retain the small distance from the endpoint. The
        // identities T_n=cos(nθ), U_n=sin((n+1)θ)/sin(θ) (and their
        // hyperbolic counterparts) avoid both sources of error.
        let degree = n as f64 + if second { 1.0 } else { 0.0 };
        let value = if x < 1.0 {
            let theta = x.acos();
            if second {
                (degree * theta).sin() / theta.sin()
            } else {
                (degree * theta).cos()
            }
        } else {
            let theta = x.acosh();
            if second {
                // sinh(theta) < 1 in this interval, so numerator overflow
                // implies that the final value also overflows.
                (degree * theta).sinh() / theta.sinh()
            } else {
                (degree * theta).cosh()
            }
        };
        return sign * parity * value;
    }
    if n >= 64 {
        // Binary doubling computes (T_k(x), U_{k-1}(x)) in O(log n).
        // Keeping both polynomials avoids division by 1-x² near endpoints.
        // For exterior x >= 1.1, all terms grow with the same sign, so
        // doubling is stable and avoids a potentially million-step recurrence.
        // Endpoint arguments use the identities above. Small degrees retain
        // the low overhead of the linear recurrence below.
        if x >= 1e150 {
            return sign * parity * f64::INFINITY;
        }
        // Convert before adding one: U_n uses T_(n+1), and n=i32::MAX
        // must not overflow the signed degree when doubling in O(log n).
        let target = n as u32 + u32::from(second);
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
        term *= if a >= 512 {
            // The first Maclaurin coefficient contains C(2a,a)/4^a.
            // Computing it by a half-beta product takes O(a) divisions even
            // when the remaining tiny-x series needs only a few terms.
            // The central-binomial asymptotic is accurate to a few ulps for
            // a >= 512; preserve the direct product for smaller degrees.
            let reciprocal = 1.0 / a as f64;
            let central = (1.0
                + reciprocal
                    * (-0.125
                        + reciprocal
                            * (1.0 / 128.0
                                + reciprocal * (5.0 / 1024.0 - reciprocal * (21.0 / 32768.0)))))
                / (std::f64::consts::PI * a as f64).sqrt();
            if n % 2 == 0 {
                central
            } else {
                n as f64 * central * x
            }
        } else if n % 2 == 0 {
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
/// entries match scalar evaluation within floating-point rounding. The O(n)
/// recurrence is stable throughout the interior interval, including tiny
/// nonzero |x|; exactly zero uses its even-degree product identity.
#[inline]
pub(crate) fn legendre_all(n: usize, x: f64) -> Vec<f64> {
    if x == 0.0 {
        // P_(2k+1)(0) = 0 and P_(2k)(0) = -(2k-1)/(2k) P_(2k-2)(0).
        // Avoid the quadratic repeated near-zero series; preserve the sign
        // of -0 for P_1, whose scalar evaluator returns x directly.
        let mut table = vec![0.0; n + 1];
        table[0] = 1.0;
        if n > 0 {
            table[1] = x;
        }
        let mut even = 1.0;
        for degree in (2..=n).step_by(2) {
            let k = degree as f64;
            even *= -(k - 1.0) / k;
            table[degree] = even;
        }
        return table;
    }
    let per_degree = |k: usize| {
        if x.is_nan() {
            f64::NAN
        } else {
            legendre(k as i32, x)
        }
    };
    let a = x.abs();
    if x.is_nan() || a == 1.0 || a.is_infinite() || n < 2 {
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
    if x == 0.0 {
        return endpoint_binomial(n, alpha);
    }
    // Build the binomial normalization alongside the difference recurrence.
    // Both previously traversed all n degrees separately. If normalization
    // overflows, restart with the unnormalized three-term recurrence below.
    let mut scale = alpha + 1.0;
    let mut difference = -x / scale;
    let mut value = 1.0 + difference;
    for k in 1..n {
        let k = k as f64;
        difference = -x / (k + alpha + 1.0) * value + k / (k + alpha + 1.0) * difference;
        value += difference;
        scale *= (alpha + k + 1.0) / (k + 1.0);
        if !scale.is_finite() {
            break;
        }
    }
    if scale.is_finite() {
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
    // P_n^(0,0) = P_n. Use the cheaper Legendre recurrence on its stable
    // interval; retain Jacobi's own near-zero path and signed-zero behavior.
    if alpha == 0.0 && beta == 0.0 && x.abs() >= 1e-5 {
        return legendre(n, x);
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
    if n == 1 {
        return value;
    }
    // Peel k=1: the general expressions lose digits when p -> -1 and the
    // B_1 expression has a removable 0/0 at p=0.
    {
        let u = p + 1.0;
        let a = (2.0 + (q / u) * (p - 1.0)) / (p + 3.0);
        let b = (q / u) * ((u - q) / u) / (p + 2.0);
        let next = (x - a) * value - b * previous;
        previous = value;
        value = next;
    }
    // For ordinary-sized parameters, use A_k = 1/2 + C/(t(t+2)) with the
    // loop-invariant C = (2q-p-1)(p-1)/2, and the factored B_k numerator over
    // t^2(t^2-1), where t = 2k+p-1. With p,q <= 1e50 and n <= 1e6, these
    // degree-4 products remain far below f64::MAX. For huge finite
    // parameters, use factored ratios instead so no intermediate overflows.
    //
    // Plain multiply/subtract is used rather than `mul_add`: without a
    // hardware FMA target feature, `mul_add` is a software library call.
    if p <= 1e50 && q <= 1e50 {
        let c = 0.5 * ((2.0 * q - p - 1.0) * (p - 1.0));
        let (pq, q1, p1) = (p - q, q - 1.0, p - 1.0);
        for k in 2..n {
            let k = k as f64;
            let t = 2.0 * k + p1;
            let t2 = t * t;
            let a = 0.5 + c / (t2 + 2.0 * t);
            let b = (k * (k + p1) * (k + pq) * (k + q1)) / (t2 * (t2 - 1.0));
            let next = (x - a) * value - b * previous;
            previous = value;
            value = next;
        }
    } else {
        for k in 2..n {
            let k = k as f64;
            let t = 2.0 * k + p - 1.0;
            // Divide first: 2*q can overflow even though the coefficient
            // stays bounded when p and q are both very large.
            let u = t + 2.0;
            let a = 0.5 * (1.0 + (2.0 * (q / u) - (p + 1.0) / u) * ((p - 1.0) / t));
            let b = (k / t)
                * ((k + p - q) / t)
                * ((k + q - 1.0) / (t - 1.0))
                * ((k + p - 1.0) / (t + 1.0));
            let next = (x - a) * value - b * previous;
            previous = value;
            value = next;
        }
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

#[cfg(test)]
mod zero_legendre_table_tests {
    use super::{legendre, legendre_all};

    #[test]
    fn tiny_x_high_degree_coefficient_matches_independent_binomial_product() {
        // P_(2k)(0) = (-1)^k C(2k,k)/4^k and
        // P_(2k+1)'(0) = (-1)^k (2k+1) C(2k,k)/4^k.
        // Multiplying the finite binomial product provides an independent
        // reference for the asymptotic coefficient used by large degrees.
        for k in [512_i32, 1024, 4096, 50_000, 499_999] {
            let central = (1..=k).fold(1.0, |value, j| {
                value * ((2 * j - 1) as f64 / (2 * j) as f64)
            });
            let parity = if k % 2 == 0 { 1.0 } else { -1.0 };
            let expected_even = parity * central;
            let actual_even = legendre(2 * k, 0.0);
            assert!(
                (actual_even - expected_even).abs() <= 3e-12 * expected_even.abs(),
                "even degree={}, actual={actual_even}, expected={expected_even}",
                2 * k
            );
            // At |x|=1e-14 the omitted x^3 term is negligible even at the
            // maximum supported degree, so the linear coefficient dominates.
            for x in [-1e-14, 1e-14] {
                let expected_odd = parity * (2 * k + 1) as f64 * central * x;
                let actual_odd = legendre(2 * k + 1, x);
                assert!(
                    (actual_odd - expected_odd).abs() <= 3e-12 * expected_odd.abs(),
                    "odd degree={}, x={x}, actual={actual_odd}, expected={expected_odd}",
                    2 * k + 1
                );
            }
        }
    }

    #[test]
    fn exact_zero_table_preserves_scalar_values_and_signed_zeros() {
        for &x in &[0.0_f64, -0.0] {
            let table = legendre_all(1024, x);
            assert_eq!(table.len(), 1025);
            for (n, &value) in table.iter().enumerate() {
                let scalar = legendre(n as i32, x);
                if n == 1 || (n % 2 != 0) {
                    assert_eq!(value.to_bits(), scalar.to_bits(), "n={n}, x={x:?}");
                } else {
                    assert!(
                        (value - scalar).abs() <= 3e-13 * scalar.abs().max(1.0),
                        "n={n}, x={x:?}, scalar={scalar}, table={value}"
                    );
                }
            }
            for &(n, expected) in &[
                (0, 1.0_f64),
                (2, -0.5),
                (4, 0.375),
                (6, -0.3125),
                (8, 0.2734375),
            ] {
                assert_eq!(table[n].to_bits(), expected.to_bits());
            }
        }
    }
}

#[cfg(test)]
mod optimized_evaluation_tests {
    use super::{chebyshev, jacobi, laguerre};

    #[test]
    fn chebyshev_exterior_doubling_matches_independent_three_term_recurrence() {
        for x in [1.1_f64, 1.25, 2.0] {
            for second in [false, true] {
                for n in [64, 65, 256, 257, 512] {
                    let (mut prior, mut current) = (1.0, if second { 2.0 * x } else { x });
                    for _ in 1..n {
                        (prior, current) = (current, 2.0 * x * current - prior);
                    }
                    for sign in [1.0, -1.0] {
                        let value = chebyshev(n, sign * x, second);
                        let expected = if sign < 0.0 && n % 2 != 0 {
                            -current
                        } else {
                            current
                        };
                        assert!(
                            value.is_finite() && (value - expected).abs() <= 3e-12 * expected.abs(),
                            "n={n}, x={}, second={second}, value={value}, expected={expected}",
                            sign * x
                        );
                    }
                }
            }
        }
        assert_eq!(chebyshev(64, 1e200, false), f64::INFINITY);
        assert_eq!(chebyshev(65, -1e200, true), f64::NEG_INFINITY);
        // The U_n doubling path evaluates T_(n+1): its exponent needs u32
        // so n=i32::MAX stays valid in debug builds.
        assert_eq!(chebyshev(i32::MAX, 1.1, true), f64::INFINITY);
        assert_eq!(chebyshev(i32::MAX, -1.1, true), f64::NEG_INFINITY);
    }

    #[test]
    fn generalized_laguerre_single_pass_matches_finite_binomial_series() {
        for alpha in [0.2_f64, 1.5, 10.0] {
            for x in [0.1_f64, 0.75, 2.0] {
                for n in [2_i32, 3, 5, 10, 20] {
                    let mut term =
                        (1..=n).fold(1.0, |value, k| value * (alpha + k as f64) / k as f64);
                    let mut expected = term;
                    for k in 1..=n {
                        term *= -x * (n - k + 1) as f64 / (k as f64 * (alpha + k as f64));
                        expected += term;
                    }
                    let value = laguerre(n, alpha, x);
                    assert!(
                        (value - expected).abs() <= 2e-10 * expected.abs().max(1.0),
                        "n={n}, alpha={alpha}, x={x}, value={value}, expected={expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn jacobi_legendre_identity_preserves_endpoints_and_zero() {
        for x in [-1.0_f64, -0.7, -0.2, 0.2, 0.7, 1.0] {
            let expected = (5.0 * x * x * x - 3.0 * x) * 0.5;
            assert!((jacobi(3, 0.0, 0.0, x) - expected).abs() < 2e-15);
        }
        assert_eq!(jacobi(1, 0.0, 0.0, -0.0).to_bits(), 0.0_f64.to_bits());
        assert_eq!(jacobi(1, 0.0, 0.0, 0.0).to_bits(), 0.0_f64.to_bits());
    }
}
