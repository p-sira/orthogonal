//! Integer-degree associated Legendre functions and spherical harmonics.
use crate::{Error, Result, finish};
use core::f64::consts::PI;
use num_complex::Complex;
use num_traits::Float;

const ERROR: Error = Error::InvalidInput(
    "Associated Legendre functions require degree in 0..=1,000,000 and a finite argument in their domain.",
);

fn validate_degree(n: i32) -> Result<u32> {
    if !(0..=1_000_000).contains(&n) {
        Err(ERROR)
    } else {
        Ok(n as u32)
    }
}

// Spherical normalization: sqrt((2n+1)/(4pi)*(n-m)!/(n+m)!).
// Keep the diagonal in logarithmic form and rescale the recurrence, so tiny
// diagonals can recover in later degrees and large factorials never overflow.
fn associated(n: u32, m: i32, x: f64, sine: f64, spherical: bool) -> f64 {
    if m == 0 {
        let value = crate::evaluate::eval_legendre(n as i32, x).unwrap_or(f64::NAN);
        return if spherical {
            value * ((2.0 * n as f64 + 1.0) / (4.0 * PI)).sqrt()
        } else {
            value
        };
    }
    if let Some(value) = associated_direct(n, m, x, sine, spherical) {
        return value;
    }
    let mut value = f64::NAN;
    associated_degrees(n, m, x, sine, spherical, n, |_, v| value = v);
    value
}

/// Unnormalized degree recurrence for moderate degree and order:
/// P_m^m = (-1)^m (2m-1)!! sin^m, then
/// (l-m) P_l^m = (2l-1) x P_(l-1)^m - (l+m-1) P_(l-2)^m (DLMF 14.10.3),
/// with the factorial ratios formed as products. Within these bounds the
/// values cannot overflow, so the logarithmic scaling of
/// `associated_degrees` and its logarithms and exponential are unnecessary.
/// Returns `None` when the result is zero or not a normal number, leaving
/// those cases to the scaled recurrence.
fn associated_direct(n: u32, m: i32, x: f64, sine: f64, spherical: bool) -> Option<f64> {
    let order = m.unsigned_abs();
    if order > 40 || n > 200 || order > n || sine < 1e-6 {
        return None;
    }
    let mut diagonal = sine.powi(order as i32);
    for k in 1..=order {
        diagonal *= (2 * k - 1) as f64;
    }
    if order % 2 == 1 {
        diagonal = -diagonal;
    }
    let (mut previous, mut current) = (0.0, diagonal);
    for degree in order + 1..=n {
        let next = ((2 * degree - 1) as f64 * x * current - (degree + order - 1) as f64 * previous)
            / (degree - order) as f64;
        previous = current;
        current = next;
    }
    // (n-|m|)!/(n+|m|)! as a product of reciprocals of n-|m|+1..=n+|m|.
    let ratio = || {
        let mut product = 1.0;
        for k in n - order + 1..=n + order {
            product /= k as f64;
        }
        product
    };
    let parity = if m < 0 && order % 2 == 1 { -1.0 } else { 1.0 };
    let value = if spherical {
        parity * current * ((2.0 * n as f64 + 1.0) / (4.0 * PI) * ratio()).sqrt()
    } else if m < 0 {
        parity * current * ratio()
    } else {
        current
    };
    value.is_normal().then_some(value)
}

// Reuse the moderate-order recurrence for both scalars and tables. The
// unnormalized values and factorial ratios remain normal within these bounds.
fn associated_direct_degrees(
    n: u32,
    m: i32,
    x: f64,
    sine: f64,
    spherical: bool,
    first: u32,
    mut emit: impl FnMut(u32, f64),
) -> bool {
    let order = m.unsigned_abs();
    if order > 40 || n > 200 || order > n || sine < 1e-6 {
        return false;
    }
    let diagonal = direct_diagonal(order, sine);
    let mut finish = |degree: u32, current: f64| {
        emit(degree, direct_value(degree, m, current, spherical));
    };
    let (mut previous, mut current) = (0.0, diagonal);
    if order >= first {
        finish(order, current);
    }
    for degree in order + 1..=n {
        let next = ((2 * degree - 1) as f64 * x * current - (degree + order - 1) as f64 * previous)
            / (degree - order) as f64;
        previous = current;
        current = next;
        if degree >= first {
            finish(degree, current);
        }
    }
    true
}

#[inline]
fn direct_diagonal(order: u32, sine: f64) -> f64 {
    let mut diagonal = sine.powi(order as i32);
    for k in 1..=order {
        diagonal *= (2 * k - 1) as f64;
    }
    if order % 2 == 1 { -diagonal } else { diagonal }
}

#[inline]
fn direct_value(degree: u32, m: i32, current: f64, spherical: bool) -> f64 {
    let order = m.unsigned_abs();
    // (degree-|m|)!/(degree+|m|)!, without overflowing factorials.
    let ratio = || {
        let mut product = 1.0;
        for k in degree - order + 1..=degree + order {
            product /= k as f64;
        }
        product
    };
    let parity = if m < 0 && order % 2 == 1 { -1.0 } else { 1.0 };
    if spherical {
        parity * current * ((2.0 * degree as f64 + 1.0) / (4.0 * PI) * ratio()).sqrt()
    } else if m < 0 {
        parity * current * ratio()
    } else {
        current
    }
}

/// Evaluates P_k^m for every degree k in `first..=n` in a single pass of the
/// degree recurrence, passing `(k, value)` to `emit`. Each value is
/// bit-identical to a separate evaluation at degree k.
fn associated_degrees(
    n: u32,
    m: i32,
    x: f64,
    sine: f64,
    spherical: bool,
    first: u32,
    mut emit: impl FnMut(u32, f64),
) {
    let order = m.unsigned_abs();
    for degree in first..=n.min(order.saturating_sub(1)) {
        if degree < order {
            emit(degree, 0.0);
        }
    }
    if order > n {
        return;
    }
    let first = first.max(order);
    if order != 0 && associated_direct_degrees(n, m, x, sine, spherical, first, &mut emit) {
        return;
    }
    if order == 0 {
        let values = crate::evaluate::legendre_all(n as usize, x);
        for degree in first..=n {
            let value = values[degree as usize];
            emit(
                degree,
                if spherical {
                    value * ((2.0 * degree as f64 + 1.0) / (4.0 * PI)).sqrt()
                } else {
                    value
                },
            );
        }
        return;
    }
    if sine == 0.0 {
        for degree in first..=n {
            emit(degree, 0.0);
        }
        return;
    }
    // Π (1 + 1/(2k)) grows only like sqrt(order), so the product is formed
    // directly and needs a single logarithm.
    let mut diagonal = 1.0;
    for k in 1..=order {
        diagonal *= 1.0 + 0.5 / k as f64;
    }
    let half_log_four_pi = 0.5 * (4.0 * PI).ln();
    let mut logarithm = -half_log_four_pi + order as f64 * sine.abs().ln() + 0.5 * diagonal.ln();
    let finalize = |degree: u32, current: f64, logarithm: f64| {
        if x == 0.0 && (degree - order) % 2 == 1 {
            return 0.0;
        }
        let mut logarithm = logarithm;
        if !spherical {
            // (4π/(2n+1)) ((n+m)!/(n-m)!)^(±1) is accumulated as products,
            // taking a logarithm only before the product could overflow.
            logarithm += half_log_four_pi;
            let width = 2.0 * degree as f64 + 1.0;
            let mut product = if m < 0 { width } else { 1.0 / width };
            let mut sum = 0.0;
            for k in degree - order + 1..=degree + order {
                if product > 1e280 {
                    sum += product.ln();
                    product = 1.0;
                }
                product *= k as f64;
            }
            sum += product.ln();
            logarithm += if m < 0 { -0.5 * sum } else { 0.5 * sum };
        }
        let parity = if (m < 0) && order % 2 == 1 { -1.0 } else { 1.0 };
        let value = if (-700.0..=700.0).contains(&logarithm) {
            current.abs() * logarithm.exp()
        } else {
            (current.abs().ln() + logarithm).exp()
        };
        parity * value.copysign(current)
    };
    let mut previous = 0.0;
    let mut current = if order % 2 == 1 { -1.0 } else { 1.0 };
    if first <= order {
        emit(order, finalize(order, current, logarithm));
    }
    for degree in order + 1..=n {
        let l = degree as f64;
        let q = order as f64;
        let denominator = (l - q) * (l + q);
        let a = ((4.0 * l * l - 1.0) / denominator).sqrt();
        let b = (((2.0 * l + 1.0) * ((l - 1.0 - q) * (l - 1.0 + q)))
            / ((2.0 * l - 3.0) * denominator))
            .sqrt();
        let next = a * x * current - b * previous;
        previous = current;
        current = next;
        let magnitude = current.abs().max(previous.abs());
        if magnitude > 1e150 {
            current *= 1e-150;
            previous *= 1e-150;
            logarithm += 150.0 * core::f64::consts::LN_10;
        } else if magnitude != 0.0 && magnitude < 1e-150 {
            current *= 1e150;
            previous *= 1e150;
            logarithm -= 150.0 * core::f64::consts::LN_10;
        }
        if degree >= first {
            emit(degree, finalize(degree, current, logarithm));
        }
    }
}

// Fill both signs of a moderate order with a shared unnormalized recurrence.
#[inline]
fn direct_table_order(
    table: &mut [Vec<f64>],
    n: usize,
    m: usize,
    order: usize,
    x: f64,
    sine: f64,
    spherical: bool,
) {
    let mut ratio = 1.0;
    for k in 1..=2 * order {
        ratio /= k as f64;
    }
    let mut previous = 0.0;
    let mut current = direct_diagonal(order as u32, sine);
    let parity = if order.is_multiple_of(2) { 1.0 } else { -1.0 };
    let negative_column = 2 * m + 1 - order;
    for (degree, row) in table.iter_mut().enumerate().skip(order) {
        let positive = if spherical {
            current * ((2 * degree + 1) as f64 / (4.0 * PI) * ratio).sqrt()
        } else {
            current
        };
        row[order] = positive;
        row[negative_column] = if spherical {
            parity * positive
        } else {
            parity * current * ratio
        };
        if degree < n {
            let next_degree = degree + 1;
            let next = ((2 * next_degree - 1) as f64 * x * current
                - (next_degree + order - 1) as f64 * previous)
                / (next_degree - order) as f64;
            previous = current;
            current = next;
            ratio *= (next_degree - order) as f64 / (next_degree + order) as f64;
        }
    }
}

// High orders need the normalized recurrence: direct factorial products can
// underflow before the eventual polynomial value becomes representable. Keep
// the shared recurrence in logarithmic scale and restore both signs of each
// order at once. For unnormalized values, update (l+m)!/(l-m)! as l advances
// instead of rebuilding its 2m factors for every table entry.
fn scaled_table_order(
    table: &mut [Vec<f64>],
    n: usize,
    m: usize,
    order: usize,
    x: f64,
    sine: f64,
    spherical: bool,
) {
    let mut diagonal = 1.0;
    for k in 1..=order {
        diagonal *= 1.0 + 0.5 / k as f64;
    }
    let half_log_four_pi = 0.5 * (4.0 * PI).ln();
    let mut logarithm = -half_log_four_pi + order as f64 * sine.abs().ln() + 0.5 * diagonal.ln();
    let mut previous = 0.0;
    let mut current = if order.is_multiple_of(2) { 1.0 } else { -1.0 };
    let negative_column = 2 * m + 1 - order;
    let parity = if order.is_multiple_of(2) { 1.0 } else { -1.0 };

    // The factorial ratio is only needed for the unnormalized table. Keep a
    // bounded mantissa and a logarithmic scale even when (2m)! overflows.
    let mut factorial = 1.0;
    let mut factorial_log_scale = 0.0;
    if !spherical {
        for k in 1..=2 * order {
            if factorial > 1e280 {
                factorial_log_scale += factorial.ln();
                factorial = 1.0;
            }
            factorial *= k as f64;
        }
    }
    let rescale = |current: f64, logarithm: f64| {
        let magnitude = if (-700.0..=700.0).contains(&logarithm) {
            current.abs() * logarithm.exp()
        } else {
            (current.abs().ln() + logarithm).exp()
        };
        magnitude.copysign(current)
    };

    for (degree, row) in table.iter_mut().enumerate().take(n + 1).skip(order) {
        if degree > order {
            let l = degree as f64;
            let q = order as f64;
            let denominator = (l - q) * (l + q);
            let a = ((4.0 * l * l - 1.0) / denominator).sqrt();
            let b = (((2.0 * l + 1.0) * ((l - 1.0 - q) * (l - 1.0 + q)))
                / ((2.0 * l - 3.0) * denominator))
                .sqrt();
            let next = a * x * current - b * previous;
            previous = current;
            current = next;
            let magnitude = current.abs().max(previous.abs());
            if magnitude > 1e150 {
                current *= 1e-150;
                previous *= 1e-150;
                logarithm += 150.0 * core::f64::consts::LN_10;
            } else if magnitude != 0.0 && magnitude < 1e-150 {
                current *= 1e150;
                previous *= 1e150;
                logarithm -= 150.0 * core::f64::consts::LN_10;
            }
            if !spherical {
                if factorial > 1e280 {
                    factorial_log_scale += factorial.ln();
                    factorial = 1.0;
                }
                factorial *= (degree + order) as f64 / (degree - order) as f64;
            }
        }
        // The degree/order parity zeros are exact even after rescaling.
        if x == 0.0 && (degree - order) % 2 == 1 {
            row[order] = 0.0;
            row[negative_column] = 0.0;
        } else if spherical {
            let positive = rescale(current, logarithm);
            row[order] = positive;
            row[negative_column] = parity * positive;
        } else {
            let half_factorial_log = 0.5 * (factorial_log_scale + factorial.ln());
            let centered = logarithm + half_log_four_pi - 0.5 * ((2 * degree + 1) as f64).ln();
            row[order] = rescale(current, centered + half_factorial_log);
            row[negative_column] = parity * rescale(current, centered - half_factorial_log);
        }
    }
}

/// Table `[degree][order]` of P_k^j for k in 0..=n and orders 0..=m followed
/// by -m..=-1, computing each order with one degree recurrence.
pub(crate) fn associated_table(
    n: usize,
    m: usize,
    x: f64,
    sine: f64,
    spherical: bool,
) -> Vec<Vec<f64>> {
    let mut table = vec![vec![0.0; 2 * m + 1]; n + 1];
    // Positive and negative orders use the same degree recurrence. For
    // moderate degrees, fill both columns together and update the factorial
    // ratio (l-m)!/(l+m)! as the degree increases, instead of recomputing
    // that product and the recurrence for every negative-order column.
    if (n <= 200 && m <= 40 && sine >= 1e-6) || (n <= 128 && m <= 128) {
        associated_degrees(n as u32, 0, x, sine, spherical, 0, |degree, value| {
            table[degree as usize][0] = value;
        });
        for order in 1..=m.min(n) {
            if order <= 40 && n <= 200 && sine >= 1e-6 {
                direct_table_order(&mut table, n, m, order, x, sine, spherical);
            } else if sine != 0.0 {
                scaled_table_order(&mut table, n, m, order, x, sine, spherical);
            }
        }
        return table;
    }
    let orders = (0..=m as i32).chain(-(m as i32)..0);
    for (column, order) in orders.enumerate() {
        associated_degrees(n as u32, order, x, sine, spherical, 0, |degree, value| {
            table[degree as usize][column] = value;
        });
    }
    table
}

/// [`associated_table`] for Ferrers functions, with NaN outside `-1 <= x <= 1`.
pub(crate) fn assoc_legendre_table(n: usize, m: usize, x: f64) -> Vec<Vec<f64>> {
    if n > 1_000_000 || !(-1.0..=1.0).contains(&x) {
        return vec![vec![f64::NAN; 2 * m + 1]; n + 1];
    }
    associated_table(n, m, x, ((1.0 - x) * (1.0 + x)).sqrt(), false)
}

/// [`associated_table`] with spherical normalization at polar angle `theta`.
pub(crate) fn sph_legendre_table(n: usize, m: usize, theta: f64) -> Vec<Vec<f64>> {
    if n > 1_000_000 || !(0.0..=PI).contains(&theta) {
        return vec![vec![f64::NAN; 2 * m + 1]; n + 1];
    }
    let (sine, cosine) = theta.sin_cos();
    let sine = if theta == PI { 0.0 } else { sine };
    associated_table(n, m, cosine, sine, true)
}

/// Ferrers associated Legendre P_n^m(x), including the Condon–Shortley phase.
///
/// Requires integer `0 <= n <= 1_000_000` and `-1 <= x <= 1`. Negative orders use
/// P_n^(-m) = (-1)^m (n-m)!/(n+m)! P_n^m; `|m| > n` returns zero.
/// Factorials and tiny diagonal values are handled without intermediate overflow.
/// Returns [`Error::InvalidInput`] for inputs outside the domain and
/// [`Error::NonFiniteEvaluation`] if the computed result is undefined.
pub fn assoc_legendre_p<T: Float>(n: i32, m: i32, x: T) -> Result<T> {
    let n = validate_degree(n)?;
    let x = x.to_f64().ok_or(ERROR)?;
    if !(-1.0..=1.0).contains(&x) {
        return Err(ERROR);
    }
    finish(
        associated(n, m, x, ((1.0 - x) * (1.0 + x)).sqrt(), false),
        Error::NonFiniteEvaluation,
    )
}

/// Spherically normalized associated Legendre function at polar angle `theta`.
///
/// Equals sqrt((2n+1)/(4pi)*(n-m)!/(n+m)!) P_n^m(cos(theta)).
/// Includes the Condon–Shortley phase. Requires `0 <= theta <= pi` and
/// `0 <= n <= 1_000_000`; orders outside `-n..=n` return zero.
/// Returns [`Error::InvalidInput`] for inputs outside the domain and
/// [`Error::NonFiniteEvaluation`] if the computed result is undefined.
pub fn sph_legendre_p<T: Float>(n: i32, m: i32, theta: T) -> Result<T> {
    let n = validate_degree(n)?;
    let theta = if theta == T::from(PI).ok_or(ERROR)? {
        PI
    } else {
        theta.to_f64().ok_or(ERROR)?
    };
    if !(0.0..=PI).contains(&theta) {
        return Err(ERROR);
    }
    let (sine, cosine) = theta.sin_cos();
    // pi denotes the exact south pole even though sin(pi) is rounded away from zero.
    let sine = if theta == PI { 0.0 } else { sine };
    finish(
        associated(n, m, cosine, sine, true),
        Error::NonFiniteEvaluation,
    )
}

/// Complex spherical harmonic Y_n^m(theta,phi), with polar angle first.
///
/// Uses spherical normalization and the Condon–Shortley phase. Requires
/// `0 <= theta <= pi`, finite azimuth `phi`, and `0 <= n <= 1_000_000`.
/// Invalid arguments return [`Error::InvalidInput`].
pub fn sph_harm_y<T: Float>(n: i32, m: i32, theta: T, phi: T) -> Result<Complex<T>> {
    let amplitude = sph_legendre_p(n, m, theta)?;
    let phi = phi.to_f64().ok_or(ERROR)?;
    if !phi.is_finite() {
        return Err(ERROR);
    }
    let (sine, cosine) = harmonic_phase(m, phi);
    Ok(Complex::new(
        amplitude * T::from(cosine).ok_or(ERROR)?,
        amplitude * T::from(sine).ok_or(ERROR)?,
    ))
}

/// (sin(m phi), cos(m phi)) as used by [`sph_harm_y`], for finite `phi`.
fn harmonic_phase(m: i32, phi: f64) -> (f64, f64) {
    let phase = m as f64 * phi;
    if phase.is_finite() {
        phase.sin_cos()
    } else {
        let (sine, cosine) = phi.sin_cos();
        let phase = num_complex::Complex64::new(cosine, sine).powu(m.unsigned_abs());
        (if m < 0 { -phase.im } else { phase.im }, phase.re)
    }
}

/// Table of Y_k^j(theta, phi) laid out as [`associated_table`], with NaN
/// entries for arguments outside the domain of [`sph_harm_y`].
pub(crate) fn sph_harm_table(
    n: usize,
    m: usize,
    theta: f64,
    phi: f64,
) -> Vec<Vec<num_complex::Complex64>> {
    let nan = num_complex::Complex64::new(f64::NAN, f64::NAN);
    if !phi.is_finite() {
        return vec![vec![nan; 2 * m + 1]; n + 1];
    }
    let amplitudes = sph_legendre_table(n, m, theta);
    // For each nonzero order, exp(-imφ) is the conjugate of exp(imφ).
    // This halves the trigonometric work and preserves signed zeros of sin.
    let mut phases = Vec::with_capacity(2 * m + 1);
    for order in 0..=m {
        phases.push(harmonic_phase(order as i32, phi));
    }
    for order in (1..=m).rev() {
        let (sine, cosine) = phases[order];
        phases.push((-sine, cosine));
    }
    amplitudes
        .into_iter()
        .map(|row| {
            row.into_iter()
                .zip(&phases)
                .map(|(amplitude, &(sine, cosine))| {
                    if amplitude.is_nan() {
                        nan
                    } else {
                        num_complex::Complex64::new(amplitude * cosine, amplitude * sine)
                    }
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64) {
        assert!(actual.is_finite() && expected.is_finite());
        let scale = actual.abs().max(expected.abs());
        assert!(
            (actual - expected).abs() <= 5e-11 * scale + 1e-320,
            "actual={actual:e}, expected={expected:e}"
        );
    }

    #[test]
    fn all_order_tables_match_independent_scalar_paths() {
        for n in [41_usize, 64, 128] {
            let m = n;
            let degrees = [0, 1, n / 2, n - 1, n];
            let orders = [0, 1, 2, 39, 40, 41, m / 2, m - 1, m];
            for x in [
                -1.0,
                -0.999_999_999_9,
                -0.4,
                -0.0,
                0.0,
                0.4,
                0.999_999_999_9,
                1.0,
            ] {
                let table = assoc_legendre_table(n, m, x);
                for degree in degrees {
                    for order in orders {
                        if order > m {
                            continue;
                        }
                        close(
                            table[degree][order],
                            assoc_legendre_p(degree as i32, order as i32, x).unwrap(),
                        );
                        if order > 0 {
                            close(
                                table[degree][2 * m + 1 - order],
                                assoc_legendre_p(degree as i32, -(order as i32), x).unwrap(),
                            );
                        }
                    }
                }
            }
            for theta in [0.0, 1e-8, 0.7, PI / 2.0, PI - 1e-8, PI] {
                let table = sph_legendre_table(n, m, theta);
                for degree in degrees {
                    for order in orders {
                        if order > m {
                            continue;
                        }
                        close(
                            table[degree][order],
                            sph_legendre_p(degree as i32, order as i32, theta).unwrap(),
                        );
                        if order > 0 {
                            close(
                                table[degree][2 * m + 1 - order],
                                sph_legendre_p(degree as i32, -(order as i32), theta).unwrap(),
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn all_order_tables_preserve_exact_zeros_and_phase() {
        let table = assoc_legendre_table(128, 128, -0.0);
        assert_eq!(table[42][41].to_bits(), 0.0_f64.to_bits());
        assert_eq!(table[42][216].to_bits(), 0.0_f64.to_bits());
        for theta in [0.0, 0.7, PI] {
            for phi in [-0.0, 0.0, -0.8, 1e308] {
                let all = sph_harm_table(48, 48, theta, phi);
                for degree in [0, 1, 40, 41, 48] {
                    for order in [-48, -41, -1, 0, 1, 41, 48] {
                        let expected = sph_harm_y(degree, order, theta, phi).unwrap();
                        let column = if order >= 0 {
                            order as usize
                        } else {
                            (97 + order) as usize
                        };
                        let actual = all[degree as usize][column];
                        close(actual.re, expected.re);
                        close(actual.im, expected.im);
                    }
                }
            }
        }
    }
}
