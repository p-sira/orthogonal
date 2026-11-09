//! Polynomial coefficients and Gaussian rules use the same three-term recurrence.

use crate::{Error, Result};

// Implicit symmetric tridiagonal QL with Wilkinson shifts (EISPACK IMTQL2).
// Quadrature needs only the first component of each eigenvector (Golub-Welsch),
// while Lamé polynomials need the complete eigensystem. Keep the same shifted
// iteration for both, but accumulate a single row for the quadrature case.
// Scaling avoids overflow in shifts and Givens rotations; work is bounded.
/// Eigenvalues and eigenvectors of a real symmetric tridiagonal matrix.
///
/// Entries of `off` describe the `n-1` subdiagonal values; the returned
/// eigenvectors are arranged as vectors, in increasing eigenvalue order.
/// Orders from 1 through 1024 are accepted.
#[cfg(test)]
pub(crate) fn eigensystem(diagonal: &[f64], off: &[f64]) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let (values, rows) = tridiagonal_eigensystem(diagonal, off, true)?;
    Ok((
        values,
        (0..diagonal.len())
            .map(|i| rows.iter().map(|row| row[i]).collect())
            .collect(),
    ))
}

fn first_eigenvector_components(diagonal: &[f64], off: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
    let (values, mut rows) = tridiagonal_eigensystem(diagonal, off, false)?;
    Ok((values, rows.pop().unwrap()))
}

// Accumulate eigenvector rows, sorted consistently with the eigenvalues.
// With all_vectors=false, only row zero is retained, avoiding unnecessary
// O(n^3) eigenvector rotations and O(n^2) eigenvector storage.
fn tridiagonal_eigensystem(
    diagonal: &[f64],
    off: &[f64],
    all_vectors: bool,
) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = diagonal.len();
    if n == 0 || n > 1024 {
        return Err(Error::InvalidInput("tridiagonal order must be in 1..=1024"));
    }
    if off.len() != n - 1 {
        return Err(Error::DimensionMismatch);
    }
    if diagonal.iter().chain(off).any(|x| !x.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    let scale = diagonal
        .iter()
        .chain(off)
        .fold(0.0_f64, |value, &x| value.max(x.abs()));
    if scale == 0.0 {
        let mut first = vec![0.0; n];
        first[0] = 1.0;
        if !all_vectors {
            return Ok((vec![0.0; n], vec![first]));
        }
        let vectors = (0..n)
            .map(|j| (0..n).map(|i| f64::from(i == j)).collect())
            .collect();
        return Ok((vec![0.0; n], vectors));
    }
    let mut d: Vec<_> = diagonal.iter().map(|&x| x / scale).collect();
    let mut e: Vec<_> = off.iter().map(|&x| x / scale).collect();
    e.push(0.0);
    let mut vectors = vec![vec![0.0; n]; if all_vectors { n } else { 1 }];
    for (i, row) in vectors.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    for l in 0..n {
        let mut iterations = 0;
        loop {
            let mut m = l;
            while m + 1 < n {
                if e[m].abs() <= f64::EPSILON * (d[m].abs() + d[m + 1].abs()) {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            if iterations == 128 {
                return Err(Error::NumericalFailure(
                    "symmetric tridiagonal eigensolver did not converge",
                ));
            }
            iterations += 1;

            // Wilkinson shift, choosing the sign to avoid cancellation.
            let shift = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let denominator = shift + shift.hypot(1.0).copysign(shift);
            let mut g = d[m] - d[l] + e[l] / denominator;
            let mut sine = 1.0;
            let mut cosine = 1.0;
            let mut correction = 0.0;
            for i in (l..m).rev() {
                let f = sine * e[i];
                let b = cosine * e[i];
                let radius = f.hypot(g);
                if radius == 0.0 {
                    e[i + 1] = 0.0;
                    sine = 0.0;
                    cosine = 1.0;
                } else {
                    e[i + 1] = radius;
                    sine = f / radius;
                    cosine = g / radius;
                }
                g = d[i + 1] - correction;
                let rotation = (d[i] - g) * sine + 2.0 * cosine * b;
                correction = sine * rotation;
                d[i + 1] = g + correction;
                g = cosine * rotation - b;
                for row in &mut vectors {
                    let f = row[i + 1];
                    row[i + 1] = sine * row[i] + cosine * f;
                    row[i] = cosine * row[i] - sine * f;
                }
            }
            d[l] -= correction;
            e[l] = g;
            e[m] = 0.0;
            if !d[l].is_finite() || !g.is_finite() {
                return Err(Error::NonFiniteEvaluation);
            }
        }
    }
    for value in &mut d {
        *value *= scale;
    }
    if d.iter().any(|x| !x.is_finite()) || vectors.iter().flatten().any(|x| !x.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    let mut indices: Vec<_> = (0..n).collect();
    indices.sort_by(|&a, &b| d[a].total_cmp(&d[b]));
    Ok((
        indices.iter().map(|&i| d[i]).collect(),
        vectors
            .iter()
            .map(|row| indices.iter().map(|&i| row[i]).collect())
            .collect(),
    ))
}

// Coefficients are ascending internally. The public representation is descending.
// Reuse a scratch buffer between recurrence steps instead of allocating for
// every degree. Keep the accumulation order to preserve roundoff behavior.
fn advance_into(
    previous: &[f64],
    before: &[f64],
    x: f64,
    constant: f64,
    subtract: f64,
    out: &mut Vec<f64>,
) {
    out.clear();
    out.resize(previous.len() + 1, 0.0);
    for (i, &value) in previous.iter().enumerate() {
        out[i] += constant * value;
        out[i + 1] += x * value;
    }
    for (i, &value) in before.iter().enumerate() {
        out[i] -= subtract * value;
    }
}

// One Horner step for p(scale*x+shift), in descending index order so the
// previous coefficients remain available until their contributions are used.
// The lower index contributes first, as in advance_into.
fn compose_step(composed: &mut Vec<f64>, coefficient: f64, scale: f64, shift: f64) {
    let len = composed.len();
    composed.push(0.0);
    composed[len] = 0.0 + scale * composed[len - 1];
    for i in (1..len).rev() {
        composed[i] = (0.0 + scale * composed[i - 1]) + shift * composed[i];
    }
    composed[0] = 0.0 + shift * composed[0];
    composed[0] += coefficient;
}

#[derive(Clone, Copy)]
enum Family {
    Jacobi(f64, f64),
    Laguerre(f64),
    Hermite(bool),
    Chebyt,
    Chebyu,
    Gegenbauer(f64),
}

fn coefficients(
    n: usize,
    family: Family,
    monic: bool,
    scale: f64,
    shift: f64,
    factor: f64,
) -> Result<Vec<f64>> {
    if n > 256 {
        return Err(Error::InvalidInput("polynomial degree must not exceed 256"));
    }
    let mut prev = Vec::with_capacity(n + 1);
    prev.push(1.0);
    let mut before = Vec::with_capacity(n + 1);
    let mut next = Vec::with_capacity(n + 1);
    for k in 1..=n {
        let kf = k as f64;
        let (x, c, b) = match family {
            Family::Jacobi(a, b) if k == 1 => ((a + b + 2.0) / 2.0, (a - b) / 2.0, 0.0),
            Family::Jacobi(a, b) => {
                let t = 2.0 * kf + a + b;
                let denominator = 2.0 * kf * (kf + a + b) * (t - 2.0);
                (
                    (t - 1.0) * t * (t - 2.0) / denominator,
                    (t - 1.0) * (a * a - b * b) / denominator,
                    2.0 * (kf + a - 1.0) * (kf + b - 1.0) * t / denominator,
                )
            }
            Family::Laguerre(a) => (-1.0 / kf, (2.0 * kf - 1.0 + a) / kf, (kf - 1.0 + a) / kf),
            Family::Hermite(true) => (1.0, 0.0, kf - 1.0),
            Family::Hermite(false) => (2.0, 0.0, 2.0 * (kf - 1.0)),
            Family::Chebyt if k == 1 => (1.0, 0.0, 0.0),
            Family::Chebyt | Family::Chebyu => (2.0, 0.0, 1.0),
            Family::Gegenbauer(a) => (2.0 * (kf - 1.0 + a) / kf, 0.0, (kf - 2.0 + 2.0 * a) / kf),
        };
        advance_into(&prev, &before, x, c, b, &mut next);
        std::mem::swap(&mut before, &mut prev);
        std::mem::swap(&mut prev, &mut next);
    }
    // The identity transform applies to most polynomial families. It needs
    // neither coefficient arithmetic nor the O(n^2) Horner composition.
    let mut composed = if scale == 1.0 && shift == 0.0 {
        prev
    } else {
        let mut result = Vec::with_capacity(n + 2);
        result.push(0.0);
        for &coefficient in prev.iter().rev() {
            compose_step(&mut result, coefficient, scale, shift);
        }
        result.truncate(n + 1);
        result
    };
    let divisor = if monic { composed[n] } else { 1.0 / factor };
    for c in &mut composed {
        *c /= divisor;
    }
    if composed.iter().any(|x| !x.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    composed.reverse();
    Ok(composed)
}

fn quadrature(
    n: usize,
    family: Family,
    scale: f64,
    shift: f64,
    weight_scale: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if n == 0 || n > 1024 {
        return Err(Error::InvalidInput("quadrature order must be in 1..=1024"));
    }
    let mut diagonal = Vec::with_capacity(n);
    let mut off = Vec::with_capacity(n - 1);
    let mass = match family {
        Family::Jacobi(a, b) => (2.0_f64).powf(a + b + 1.0) * crate::beta(a + 1.0, b + 1.0),
        Family::Laguerre(a) => crate::gamma(a + 1.0),
        Family::Hermite(true) => (2.0 * std::f64::consts::PI).sqrt(),
        Family::Hermite(false) => std::f64::consts::PI.sqrt(),
        _ => unreachable!(),
    };
    for i in 0..n {
        let k = i as f64;
        diagonal.push(match family {
            Family::Jacobi(a, b) if i == 0 => (b - a) / (a + b + 2.0),
            Family::Jacobi(a, b) => (b * b - a * a) / ((2.0 * k + a + b) * (2.0 * k + a + b + 2.0)),
            Family::Laguerre(a) => 2.0 * k + a + 1.0,
            _ => 0.0,
        });
        if i > 0 {
            off.push(match family {
                Family::Jacobi(a, b) if i == 1 => {
                    (4.0 * (1.0 + a) * (1.0 + b) / ((a + b + 2.0).powi(2) * (a + b + 3.0))).sqrt()
                }
                Family::Jacobi(a, b) => {
                    let t = 2.0 * k + a + b;
                    (4.0 * k * (k + a) * (k + b) * (k + a + b) / (t * t * (t * t - 1.0))).sqrt()
                }
                Family::Laguerre(a) => (k * (k + a)).sqrt(),
                Family::Hermite(true) => k.sqrt(),
                Family::Hermite(false) => (k / 2.0).sqrt(),
                _ => unreachable!(),
            });
        }
    }
    let (nodes, first_components) = first_eigenvector_components(&diagonal, &off)?;
    let weights: Vec<_> = first_components
        .iter()
        .map(|v| v * v * mass * weight_scale)
        .collect();
    if weights.iter().any(|x| !x.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok((
        nodes.into_iter().map(|x| scale * x + shift).collect(),
        weights,
    ))
}

// Gauss-Chebyshev rules have explicit nodes and weights. Construct the pairs
// symmetrically: this halves the trigonometric work, preserves exact symmetry,
// and avoids the O(n^2) tridiagonal eigensolver used by general Jacobi rules.
// The first kind integrates (1-x²)^(-1/2); the second integrates sqrt(1-x²).
// The caller supplies the same node transform and weight multiplier as the
// general quadrature backend.
fn chebyshev_quadrature(
    n: usize,
    second: bool,
    scale: f64,
    shift: f64,
    weight_scale: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if n == 0 || n > 1024 {
        return Err(Error::InvalidInput("quadrature order must be in 1..=1024"));
    }
    let denominator = (n + usize::from(second)) as f64;
    let step = std::f64::consts::PI / denominator;
    let weight = step * weight_scale;
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    for i in 0..n / 2 {
        let angle = (i as f64 + if second { 1.0 } else { 0.5 }) * step;
        let (left, right, w) = if shift == 0.5 {
            // For the shifted [0,1] rules, (1-cos(angle))/2 cancels near
            // zero. The half-angle identity retains relative node accuracy.
            let (sine, cosine) = if second {
                (0.5 * angle).sin_cos()
            } else {
                ((0.5 * angle).sin(), 1.0)
            };
            let left = sine * sine;
            let s = 2.0 * sine * cosine;
            (left, 1.0 - left, weight * if second { s * s } else { 1.0 })
        } else {
            let (sine, cosine) = if second {
                angle.sin_cos()
            } else {
                (1.0, angle.cos())
            };
            let offset = scale * cosine;
            (shift - offset, shift + offset, weight * sine * sine)
        };
        nodes[i] = left;
        nodes[n - 1 - i] = right;
        weights[i] = w;
        weights[n - 1 - i] = w;
    }
    if !n.is_multiple_of(2) {
        // For odd orders the central node is exactly zero before translation.
        // Its second-kind sine factor is exactly one.
        nodes[n / 2] = shift;
        weights[n / 2] = weight;
    }
    Ok((nodes, weights))
}

/// Returns coefficients of the Jacobi P_n^(alpha,beta) polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight (1−x)^alpha(1+x)^beta on `[−1,1]` integrable.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = orthogonal::jacobi(3,0.5,0.3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy jacobi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jacobi.html).
pub fn jacobi(n: usize, alpha: f64, beta: f64, monic: bool) -> Result<Vec<f64>> {
    if !(alpha.is_finite() && beta.is_finite() && alpha > -1.0 && beta > -1.0) {
        return Err(Error::InvalidInput(
            "orthogonal polynomial parameters are outside the domain",
        ));
    }
    coefficients(n, Family::Jacobi(alpha, beta), monic, 1.0, 0.0, 1.0)
}

/// Gaussian nodes and weights for (1−x)^alpha(1+x)^beta on `[−1,1]`.
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=1024`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = orthogonal::roots_jacobi(3,0.5,0.3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_jacobi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_jacobi.html).
pub fn roots_jacobi(n: usize, alpha: f64, beta: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if !(alpha.is_finite() && beta.is_finite() && alpha > -1.0 && beta > -1.0) {
        return Err(Error::InvalidInput(
            "quadrature parameters are outside the domain",
        ));
    }
    quadrature(n, Family::Jacobi(alpha, beta), 1.0, 0.0, 1.0)
}

/// Returns coefficients of the Legendre P_n polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight unit weight on `[−1,1]` integrable.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = orthogonal::legendre(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy legendre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.legendre.html).
pub fn legendre(n: usize, monic: bool) -> Result<Vec<f64>> {
    coefficients(n, Family::Jacobi(0.0, 0.0), monic, 1.0, 0.0, 1.0)
}

/// Gaussian nodes and weights for unit weight on `[−1,1]`.
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=1024`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = orthogonal::roots_legendre(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_legendre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_legendre.html).
pub fn roots_legendre(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    quadrature(n, Family::Jacobi(0.0, 0.0), 1.0, 0.0, 1.0)
}

/// Returns coefficients of the Gegenbauer C_n^alpha polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight (1−x²)^(alpha−1/2) on `[−1,1]` integrable.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = orthogonal::gegenbauer(3,0.5,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy gegenbauer](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gegenbauer.html).
pub fn gegenbauer(n: usize, alpha: f64, monic: bool) -> Result<Vec<f64>> {
    if !(alpha.is_finite() && alpha > -0.5) {
        return Err(Error::InvalidInput(
            "orthogonal polynomial parameters are outside the domain",
        ));
    }
    coefficients(n, Family::Gegenbauer(alpha), monic, 1.0, 0.0, 1.0)
}

/// Gaussian nodes and weights for (1−x²)^(alpha−1/2) on `[−1,1]`.
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=1024`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = orthogonal::roots_gegenbauer(3,0.5)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_gegenbauer](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_gegenbauer.html).
pub fn roots_gegenbauer(n: usize, alpha: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if !(alpha.is_finite() && alpha > -0.5) {
        return Err(Error::InvalidInput(
            "quadrature parameters are outside the domain",
        ));
    }
    quadrature(n, Family::Jacobi(alpha - 0.5, alpha - 0.5), 1.0, 0.0, 1.0)
}

/// Returns coefficients of the generalized Laguerre L_n^alpha polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight x^alpha exp(−x) on [0,∞) integrable.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = orthogonal::genlaguerre(3,0.5,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy genlaguerre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.genlaguerre.html).
pub fn genlaguerre(n: usize, alpha: f64, monic: bool) -> Result<Vec<f64>> {
    if !(alpha.is_finite() && alpha > -1.0) {
        return Err(Error::InvalidInput(
            "orthogonal polynomial parameters are outside the domain",
        ));
    }
    coefficients(n, Family::Laguerre(alpha), monic, 1.0, 0.0, 1.0)
}

/// Gaussian nodes and weights for x^alpha exp(−x) on [0,∞).
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=1024`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = orthogonal::roots_genlaguerre(3,0.5)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_genlaguerre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_genlaguerre.html).
pub fn roots_genlaguerre(n: usize, alpha: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if !(alpha.is_finite() && alpha > -1.0) {
        return Err(Error::InvalidInput(
            "quadrature parameters are outside the domain",
        ));
    }
    quadrature(n, Family::Laguerre(alpha), 1.0, 0.0, 1.0)
}

/// Returns coefficients of the Laguerre L_n polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight exp(−x) on [0,∞) integrable.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = orthogonal::laguerre(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy laguerre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.laguerre.html).
pub fn laguerre(n: usize, monic: bool) -> Result<Vec<f64>> {
    coefficients(n, Family::Laguerre(0.0), monic, 1.0, 0.0, 1.0)
}

/// Gaussian nodes and weights for exp(−x) on [0,∞).
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=1024`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = orthogonal::roots_laguerre(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_laguerre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_laguerre.html).
pub fn roots_laguerre(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    quadrature(n, Family::Laguerre(0.0), 1.0, 0.0, 1.0)
}

/// Returns coefficients of the physicists' Hermite H_n polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight exp(−x²) on (−∞,∞) integrable.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = orthogonal::hermite(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy hermite](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hermite.html).
pub fn hermite(n: usize, monic: bool) -> Result<Vec<f64>> {
    coefficients(n, Family::Hermite(false), monic, 1.0, 0.0, 1.0)
}

/// Gaussian nodes and weights for exp(−x²) on (−∞,∞).
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=1024`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = orthogonal::roots_hermite(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_hermite](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_hermite.html).
pub fn roots_hermite(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    quadrature(n, Family::Hermite(false), 1.0, 0.0, 1.0)
}

/// Returns coefficients of the probabilists' Hermite He_n polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight exp(−x²/2) on (−∞,∞) integrable.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = orthogonal::hermitenorm(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy hermitenorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hermitenorm.html).
pub fn hermitenorm(n: usize, monic: bool) -> Result<Vec<f64>> {
    coefficients(n, Family::Hermite(true), monic, 1.0, 0.0, 1.0)
}

/// Gaussian nodes and weights for exp(−x²/2) on (−∞,∞).
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=1024`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = orthogonal::roots_hermitenorm(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_hermitenorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_hermitenorm.html).
pub fn roots_hermitenorm(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    quadrature(n, Family::Hermite(true), 1.0, 0.0, 1.0)
}

/// Returns coefficients of the Chebyshev T_n polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight (1−x²)^(-1/2) on `[−1,1]` integrable.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = orthogonal::chebyt(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy chebyt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chebyt.html).
pub fn chebyt(n: usize, monic: bool) -> Result<Vec<f64>> {
    coefficients(n, Family::Chebyt, monic, 1.0, 0.0, 1.0)
}

/// Gaussian nodes and weights for (1−x²)^(-1/2) on `[−1,1]`.
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=1024`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = orthogonal::roots_chebyt(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_chebyt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_chebyt.html).
pub fn roots_chebyt(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    chebyshev_quadrature(n, false, 1.0, 0.0, 1.0)
}

/// Returns coefficients of the Chebyshev U_n polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight √(1−x²) on `[−1,1]` integrable.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = orthogonal::chebyu(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy chebyu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chebyu.html).
pub fn chebyu(n: usize, monic: bool) -> Result<Vec<f64>> {
    coefficients(n, Family::Chebyu, monic, 1.0, 0.0, 1.0)
}

/// Gaussian nodes and weights for √(1−x²) on `[−1,1]`.
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=1024`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = orthogonal::roots_chebyu(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_chebyu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_chebyu.html).
pub fn roots_chebyu(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    chebyshev_quadrature(n, true, 1.0, 0.0, 1.0)
}

/// Returns coefficients of the scaled Chebyshev 2T_n(x/2) polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight (1−x²/4)^(-1/2) on `[−2,2]` integrable.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = orthogonal::chebyc(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy chebyc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chebyc.html).
pub fn chebyc(n: usize, monic: bool) -> Result<Vec<f64>> {
    coefficients(n, Family::Chebyt, monic, 0.5, 0.0, 2.0)
}

/// Gaussian nodes and weights for (1−x²/4)^(-1/2) on `[−2,2]`.
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=1024`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = orthogonal::roots_chebyc(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_chebyc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_chebyc.html).
pub fn roots_chebyc(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    chebyshev_quadrature(n, false, 2.0, 0.0, 2.0)
}

/// Returns coefficients of the scaled Chebyshev U_n(x/2) polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight √(1−x²/4) on `[−2,2]` integrable.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = orthogonal::chebys(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy chebys](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chebys.html).
pub fn chebys(n: usize, monic: bool) -> Result<Vec<f64>> {
    coefficients(n, Family::Chebyu, monic, 0.5, 0.0, 1.0)
}

/// Gaussian nodes and weights for √(1−x²/4) on `[−2,2]`.
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=1024`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = orthogonal::roots_chebys(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_chebys](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_chebys.html).
pub fn roots_chebys(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    chebyshev_quadrature(n, true, 2.0, 0.0, 2.0)
}

/// Returns coefficients of the shifted Chebyshev T_n(2x−1) polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight (x(1−x))^(-1/2) on `[0,1]` integrable.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = orthogonal::sh_chebyt(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy sh_chebyt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sh_chebyt.html).
pub fn sh_chebyt(n: usize, monic: bool) -> Result<Vec<f64>> {
    coefficients(n, Family::Chebyt, monic, 2.0, -1.0, 1.0)
}

/// Gaussian nodes and weights for (x(1−x))^(-1/2) on `[0,1]`.
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=1024`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = orthogonal::roots_sh_chebyt(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_sh_chebyt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_sh_chebyt.html).
pub fn roots_sh_chebyt(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    chebyshev_quadrature(n, false, 0.5, 0.5, 1.0)
}

/// Returns coefficients of the shifted Chebyshev U_n(2x−1) polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight √(x(1−x)) on `[0,1]` integrable.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = orthogonal::sh_chebyu(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy sh_chebyu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sh_chebyu.html).
pub fn sh_chebyu(n: usize, monic: bool) -> Result<Vec<f64>> {
    coefficients(n, Family::Chebyu, monic, 2.0, -1.0, 1.0)
}

/// Gaussian nodes and weights for √(x(1−x)) on `[0,1]`.
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=1024`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = orthogonal::roots_sh_chebyu(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_sh_chebyu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_sh_chebyu.html).
pub fn roots_sh_chebyu(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    chebyshev_quadrature(n, true, 0.5, 0.5, 0.25)
}

/// Returns coefficients of the shifted Legendre P_n(2x−1) polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight unit weight on `[0,1]` integrable.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = orthogonal::sh_legendre(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy sh_legendre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sh_legendre.html).
pub fn sh_legendre(n: usize, monic: bool) -> Result<Vec<f64>> {
    coefficients(n, Family::Jacobi(0.0, 0.0), monic, 2.0, -1.0, 1.0)
}

/// Gaussian nodes and weights for unit weight on `[0,1]`.
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=1024`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = orthogonal::roots_sh_legendre(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_sh_legendre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_sh_legendre.html).
pub fn roots_sh_legendre(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    quadrature(n, Family::Jacobi(0.0, 0.0), 0.5, 0.5, 0.5)
}

/// Returns coefficients of the shifted Jacobi G_n^(p,q) polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight (1−x)^(p−q) x^(q−1) on `[0,1]` integrable.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = orthogonal::sh_jacobi(3,2.5,1.5,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy sh_jacobi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sh_jacobi.html).
pub fn sh_jacobi(n: usize, p: f64, q: f64, monic: bool) -> Result<Vec<f64>> {
    if !(p.is_finite() && q.is_finite() && q > 0.0 && p - q > -1.0) {
        return Err(Error::InvalidInput(
            "orthogonal polynomial parameters are outside the domain",
        ));
    }
    coefficients(
        n,
        Family::Jacobi(p - q, q - 1.0),
        monic,
        2.0,
        -1.0,
        (crate::gammaln(n as f64 + 1.0) + crate::gammaln(n as f64 + p)
            - crate::gammaln(2.0 * n as f64 + p))
        .exp(),
    )
}

/// Gaussian nodes and weights for (1−x)^(p−q) x^(q−1) on `[0,1]`.
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=1024`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = orthogonal::roots_sh_jacobi(3,2.5,1.5)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),orthogonal::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_sh_jacobi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_sh_jacobi.html).
pub fn roots_sh_jacobi(n: usize, p: f64, q: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if !(p.is_finite() && q.is_finite() && q > 0.0 && p - q > -1.0) {
        return Err(Error::InvalidInput(
            "quadrature parameters are outside the domain",
        ));
    }
    quadrature(
        n,
        Family::Jacobi(p - q, q - 1.0),
        0.5,
        0.5,
        2.0_f64.powf(-p),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // Retain the allocation-per-step implementation as a numerical reference.
    // This checks the exact rounding and signed zeros of coefficient generation,
    // including the transformed and monic families.
    fn original_advance(
        previous: &[f64],
        before: &[f64],
        x: f64,
        constant: f64,
        subtract: f64,
    ) -> Vec<f64> {
        let mut out = vec![0.0; previous.len() + 1];
        for (i, &value) in previous.iter().enumerate() {
            out[i] += constant * value;
            out[i + 1] += x * value;
        }
        for (i, &value) in before.iter().enumerate() {
            out[i] -= subtract * value;
        }
        out
    }

    fn original_coefficients(
        n: usize,
        family: Family,
        monic: bool,
        scale: f64,
        shift: f64,
        factor: f64,
    ) -> Result<Vec<f64>> {
        let mut prev = vec![1.0];
        let mut before = vec![];
        for k in 1..=n {
            let kf = k as f64;
            let (x, c, b) = match family {
                Family::Jacobi(a, b) if k == 1 => ((a + b + 2.0) / 2.0, (a - b) / 2.0, 0.0),
                Family::Jacobi(a, b) => {
                    let t = 2.0 * kf + a + b;
                    let denominator = 2.0 * kf * (kf + a + b) * (t - 2.0);
                    (
                        (t - 1.0) * t * (t - 2.0) / denominator,
                        (t - 1.0) * (a * a - b * b) / denominator,
                        2.0 * (kf + a - 1.0) * (kf + b - 1.0) * t / denominator,
                    )
                }
                Family::Laguerre(a) => (-1.0 / kf, (2.0 * kf - 1.0 + a) / kf, (kf - 1.0 + a) / kf),
                Family::Hermite(true) => (1.0, 0.0, kf - 1.0),
                Family::Hermite(false) => (2.0, 0.0, 2.0 * (kf - 1.0)),
                Family::Chebyt if k == 1 => (1.0, 0.0, 0.0),
                Family::Chebyt | Family::Chebyu => (2.0, 0.0, 1.0),
                Family::Gegenbauer(a) => {
                    (2.0 * (kf - 1.0 + a) / kf, 0.0, (kf - 2.0 + 2.0 * a) / kf)
                }
            };
            let next = original_advance(&prev, &before, x, c, b);
            before = prev;
            prev = next;
        }
        let mut composed = vec![0.0];
        for &coefficient in prev.iter().rev() {
            composed = original_advance(&composed, &[], scale, shift, 0.0);
            composed[0] += coefficient;
        }
        composed.truncate(n + 1);
        let divisor = if monic { composed[n] } else { 1.0 / factor };
        for c in &mut composed {
            *c /= divisor;
        }
        if composed.iter().any(|x| !x.is_finite()) {
            return Err(Error::NonFiniteEvaluation);
        }
        composed.reverse();
        Ok(composed)
    }

    #[test]
    fn reused_coefficient_buffers_match_original_bits() {
        let families = [
            (Family::Jacobi(0.0, 0.0), 1.0, 0.0, 1.0),
            (Family::Jacobi(0.5, -0.5), 1.0, 0.0, 1.0),
            (Family::Jacobi(-0.99, 2.0), 2.0, -1.0, 1.0),
            (Family::Laguerre(0.0), 1.0, 0.0, 1.0),
            (Family::Laguerre(2.25), 1.0, 0.0, 1.0),
            (Family::Hermite(true), 1.0, 0.0, 1.0),
            (Family::Hermite(false), 1.0, 0.0, 1.0),
            (Family::Chebyt, 1.0, 0.0, 1.0),
            (Family::Chebyu, 1.0, 0.0, 1.0),
            (Family::Chebyt, 0.5, 0.0, 2.0),
            (Family::Chebyu, 0.5, 0.0, 1.0),
            (Family::Chebyt, 2.0, -1.0, 1.0),
            (Family::Chebyu, 2.0, -1.0, 1.0),
            (Family::Gegenbauer(0.25), 1.0, 0.0, 1.0),
        ];
        for (family, scale, shift, factor) in families {
            for n in [0, 1, 2, 3, 8, 16, 32, 64, 128, 256] {
                for monic in [false, true] {
                    let actual = coefficients(n, family, monic, scale, shift, factor);
                    let reference = original_coefficients(n, family, monic, scale, shift, factor);
                    match (actual, reference) {
                        (Ok(actual), Ok(reference)) => {
                            assert_eq!(actual.len(), reference.len());
                            for (i, (&a, &b)) in actual.iter().zip(&reference).enumerate() {
                                assert_eq!(a.to_bits(), b.to_bits(), "n={n}, i={i}");
                            }
                        }
                        (Err(a), Err(b)) => {
                            assert_eq!(std::mem::discriminant(&a), std::mem::discriminant(&b));
                        }
                        (actual, reference) => {
                            panic!(
                                "n={n}, monic={monic}: actual={actual:?}, reference={reference:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn reused_horner_step_preserves_signed_zero_and_special_values() {
        let inputs = [
            vec![0.0, -0.0, 1.0, -1.0],
            vec![
                f64::MIN_POSITIVE,
                f64::MIN_POSITIVE / 2.0,
                -f64::MIN_POSITIVE,
            ],
            vec![f64::MAX, -f64::MAX, f64::INFINITY],
            vec![f64::NAN, f64::NEG_INFINITY, 0.0],
        ];
        for input in inputs {
            for (scale, shift, coefficient) in [
                (1.0, -0.0, -0.0),
                (0.5, 0.0, -1.0),
                (2.0, -1.0, 0.0),
                (-1.0, 1.0, f64::INFINITY),
            ] {
                let mut actual = input.clone();
                compose_step(&mut actual, coefficient, scale, shift);
                let mut reference = original_advance(&input, &[], scale, shift, 0.0);
                reference[0] += coefficient;
                for (&a, &b) in actual.iter().zip(&reference) {
                    if b.is_nan() {
                        assert!(a.is_nan());
                    } else {
                        assert_eq!(a.to_bits(), b.to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn gaussian_rules_integrate_independent_polynomial_moments() {
        // Exact moments through degree 2n-1 independently exercise sorted
        // eigenvalues, the first eigenvector components, and weight scaling.
        for n in [1, 2, 3, 5, 8, 16] {
            let (nodes, weights) = roots_legendre(n).unwrap();
            assert!(nodes.windows(2).all(|x| x[0] < x[1]));
            assert!(weights.iter().all(|&w| w > 0.0));
            for k in 0..(2 * n) {
                let moment: f64 = nodes
                    .iter()
                    .zip(&weights)
                    .map(|(&x, &w)| w * x.powi(k as i32))
                    .sum();
                let expected = if k % 2 == 0 {
                    2.0 / (k + 1) as f64
                } else {
                    0.0
                };
                assert!((moment - expected).abs() < 2e-12, "Legendre n={n}, k={k}");
            }
        }
        for n in [1, 2, 3, 6] {
            let (nodes, weights) = roots_jacobi(n, 1.0, 0.0).unwrap();
            for k in 0..(2 * n) {
                let moment: f64 = nodes
                    .iter()
                    .zip(&weights)
                    .map(|(&x, &w)| w * x.powi(k as i32))
                    .sum();
                let expected = if k % 2 == 0 {
                    2.0 / (k + 1) as f64
                } else {
                    -2.0 / (k + 2) as f64
                };
                assert!((moment - expected).abs() < 2e-12, "Jacobi n={n}, k={k}");
            }
        }
        for n in [1, 2, 3, 5, 8] {
            let (nodes, weights) = roots_laguerre(n).unwrap();
            let mut factorial = 1.0;
            for k in 0..(2 * n) {
                if k > 0 {
                    factorial *= k as f64;
                }
                let moment: f64 = nodes
                    .iter()
                    .zip(&weights)
                    .map(|(&x, &w)| w * x.powi(k as i32))
                    .sum();
                assert!(
                    ((moment - factorial) / factorial).abs() < 3e-12,
                    "Laguerre n={n}, k={k}"
                );
            }
        }
    }

    #[test]
    fn chebyshev_closed_form_rules_match_generic_jacobi_eigensolver() {
        // The previous public implementations all used Golub-Welsch with
        // Jacobi parameters (+/-1/2, +/-1/2). Keep that independent numerical
        // backend as a reference for every scaling/translation variant.
        for n in [1, 2, 3, 4, 8, 32, 128, 256] {
            for second in [false, true] {
                let alpha = if second { 0.5 } else { -0.5 };
                let (reference_nodes, reference_weights) =
                    quadrature(n, Family::Jacobi(alpha, alpha), 1.0, 0.0, 1.0).unwrap();
                for variant in 0..3 {
                    let (scale, shift, weight_scale, rule) = match (second, variant) {
                        (false, 0) => (1.0, 0.0, 1.0, roots_chebyt(n)),
                        (false, 1) => (2.0, 0.0, 2.0, roots_chebyc(n)),
                        (false, _) => (0.5, 0.5, 1.0, roots_sh_chebyt(n)),
                        (true, 0) => (1.0, 0.0, 1.0, roots_chebyu(n)),
                        (true, 1) => (2.0, 0.0, 2.0, roots_chebys(n)),
                        (true, _) => (0.5, 0.5, 0.25, roots_sh_chebyu(n)),
                    };
                    let (nodes, weights) = rule.unwrap();
                    assert!(nodes.windows(2).all(|pair| pair[0] < pair[1]));
                    for i in 0..n {
                        let expected_node = scale * reference_nodes[i] + shift;
                        let expected_weight = weight_scale * reference_weights[i];
                        assert!(
                            (nodes[i] - expected_node).abs() <= 3e-12 * (1.0 + expected_node.abs()),
                            "node n={n}, second={second}, variant={variant}, i={i}: {} vs {}",
                            nodes[i],
                            expected_node
                        );
                        assert!(
                            (weights[i] - expected_weight).abs()
                                <= 3e-12 * (1.0 + expected_weight.abs()),
                            "weight n={n}, second={second}, variant={variant}, i={i}: {} vs {}",
                            weights[i],
                            expected_weight
                        );
                        assert_eq!(weights[i], weights[n - 1 - i]);
                    }
                    if n % 2 != 0 {
                        assert_eq!(nodes[n / 2], shift);
                    }
                }
            }
        }
    }

    #[test]
    fn chebyshev_closed_form_rules_integrate_exact_moments_to_high_order() {
        // Independent beta-integral moments: E_T[x^(2k)] = pi * (2k-1)!!/(2k)!!
        // and E_U[x^(2k)] = (pi/2) * product_{j=1}^k (2j-1)/(2j+2).
        // Undo the affine node map so one identity covers all six public rules.
        for n in [1, 2, 3, 5, 16, 128, 512, 1024] {
            for (second, scale, shift, weight_scale, rule) in [
                (false, 1.0, 0.0, 1.0, roots_chebyt(n)),
                (true, 1.0, 0.0, 1.0, roots_chebyu(n)),
                (false, 2.0, 0.0, 2.0, roots_chebyc(n)),
                (true, 2.0, 0.0, 2.0, roots_chebys(n)),
                (false, 0.5, 0.5, 1.0, roots_sh_chebyt(n)),
                (true, 0.5, 0.5, 0.25, roots_sh_chebyu(n)),
            ] {
                let (nodes, weights) = rule.unwrap();
                assert_eq!(nodes.len(), n);
                assert!(nodes.windows(2).all(|pair| pair[0] < pair[1]));
                assert!(weights.iter().all(|&w| w > 0.0 && w.is_finite()));
                let mut even_moment =
                    std::f64::consts::PI * if second { 0.5 } else { 1.0 } * weight_scale;
                for degree in 0..=(2 * n - 1).min(24) {
                    if degree > 0 && degree % 2 == 0 {
                        let k = degree / 2;
                        even_moment *=
                            (2 * k - 1) as f64 / (2 * k + if second { 2 } else { 0 }) as f64;
                    }
                    let expected = if degree % 2 == 0 { even_moment } else { 0.0 };
                    let actual: f64 = nodes
                        .iter()
                        .zip(&weights)
                        .map(|(&node, &weight)| {
                            weight * ((node - shift) / scale).powi(degree as i32)
                        })
                        .sum();
                    assert!(
                        (actual - expected).abs() < 7e-12 * (1.0 + expected.abs()),
                        "moment n={n}, second={second}, scale={scale}, degree={degree}: {actual} vs {expected}"
                    );
                }
            }
        }
        for n in [0, 1025] {
            assert!(roots_chebyt(n).is_err());
            assert!(roots_chebyu(n).is_err());
            assert!(roots_chebyc(n).is_err());
            assert!(roots_chebys(n).is_err());
            assert!(roots_sh_chebyt(n).is_err());
            assert!(roots_sh_chebyu(n).is_err());
        }
    }

    #[test]
    fn shifted_chebyshev_endpoint_nodes_keep_relative_precision() {
        // Independent Taylor expansions avoid the subtraction in
        // (1-cos(theta))/2 and expose cancellation near the left endpoint.
        let n = 1024;
        for second in [false, true] {
            let (nodes, weights) = if second {
                roots_sh_chebyu(n).unwrap()
            } else {
                roots_sh_chebyt(n).unwrap()
            };
            let denominator = (n + usize::from(second)) as f64;
            let theta = std::f64::consts::PI * if second { 1.0 } else { 0.5 } / denominator;
            let z = theta * theta;
            let expected_left = 0.25 * z * (1.0 - z / 12.0 + z * z / 360.0);
            assert!(
                ((nodes[0] - expected_left) / expected_left).abs() < 5e-14,
                "leftmost node n={n}, second={second}: {} vs {}",
                nodes[0],
                expected_left
            );
            assert_eq!(nodes[n - 1], 1.0 - nodes[0]);
            if second {
                let expected_weight = std::f64::consts::PI / denominator
                    * 0.25
                    * z
                    * (1.0 - z / 3.0 + 2.0 * z * z / 45.0);
                assert!(
                    ((weights[0] - expected_weight) / expected_weight).abs() < 5e-14,
                    "leftmost weight n={n}: {} vs {}",
                    weights[0],
                    expected_weight
                );
            }
        }
    }

    #[test]
    fn quadrature_first_components_match_full_eigensystem() {
        // Both paths use identical shifts and sorting; the optimization skips
        // unused eigenvector rows rather than changing eigenvalue algorithms.
        for n in [1, 2, 3, 16, 64, 128, 256] {
            let diagonal: Vec<_> = (0..n).map(|i| 0.15 * (i as f64).sin()).collect();
            let off: Vec<_> = (0..n - 1).map(|i| 0.25 + 0.01 * ((i % 7) as f64)).collect();
            let (full_values, full_vectors) = eigensystem(&diagonal, &off).unwrap();
            let (first_values, first_components) =
                first_eigenvector_components(&diagonal, &off).unwrap();
            assert_eq!(first_values, full_values, "n={n}");
            for (i, &component) in first_components.iter().enumerate() {
                assert_eq!(component, full_vectors[i][0], "n={n}, i={i}");
            }
        }
        let (values, first) = first_eigenvector_components(&[0.0; 3], &[0.0; 2]).unwrap();
        assert_eq!(values, vec![0.0; 3]);
        assert_eq!(first, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn shifted_tridiagonal_eigenpairs_have_small_residuals_and_are_orthogonal() {
        // The Toeplitz tridiagonal spectrum is 2 - 2*cos(k*pi/(n+1)).
        // In particular, the smallest eigenvalues of the 256-row case are
        // much smaller than the entries of the matrix.
        for n in [1, 2, 16, 64, 256] {
            let diagonal = vec![2.0; n];
            let off = vec![-1.0; n - 1];
            let (values, vectors) = eigensystem(&diagonal, &off).unwrap();
            assert_eq!(values.len(), n);
            assert_eq!(vectors.len(), n);
            assert!(values.windows(2).all(|pair| pair[0] <= pair[1]));
            for (k, (&value, vector)) in values.iter().zip(&vectors).enumerate() {
                let expected =
                    2.0 - 2.0 * (((k + 1) as f64 * std::f64::consts::PI) / (n + 1) as f64).cos();
                assert!(
                    (value - expected).abs() < 3e-12,
                    "n={n}, k={k}, eigenvalue={value}, expected={expected}"
                );
                let mut squared_norm = 0.0;
                for i in 0..n {
                    let left = if i > 0 { -vector[i - 1] } else { 0.0 };
                    let right = if i + 1 < n { -vector[i + 1] } else { 0.0 };
                    let residual = left + 2.0 * vector[i] + right - value * vector[i];
                    assert!(
                        residual.abs() < 3e-12,
                        "n={n}, k={k}, i={i}, residual={residual:e}"
                    );
                    squared_norm += vector[i] * vector[i];
                }
                assert!((squared_norm - 1.0).abs() < 3e-12);
            }
            // Check the full basis at modest orders and representative
            // separated/adjacent eigenpairs at the largest supported order.
            let selected: Vec<_> = if n <= 64 {
                (0..n).collect()
            } else {
                vec![0, 1, 63, 64, 127, 128, 254, 255]
            };
            for &i in &selected {
                for &j in &selected {
                    let dot = vectors[i]
                        .iter()
                        .zip(&vectors[j])
                        .map(|(x, y)| x * y)
                        .sum::<f64>();
                    assert!(
                        (dot - f64::from(i == j)).abs() < 3e-12,
                        "n={n}, i={i}, j={j}, dot={dot}"
                    );
                }
            }
        }
    }

    #[test]
    fn shifted_tridiagonal_handles_scaling_degeneracy_and_invalid_data() {
        assert!(matches!(eigensystem(&[], &[]), Err(Error::InvalidInput(_))));
        assert!(matches!(
            eigensystem(&[1.], &[1.]),
            Err(Error::DimensionMismatch)
        ));
        assert_eq!(eigensystem(&[5.], &[]).unwrap(), (vec![5.], vec![vec![1.]]));
        assert!(matches!(
            eigensystem(&[f64::NAN], &[]),
            Err(Error::NonFiniteEvaluation)
        ));
        assert!(matches!(
            eigensystem(&[1., 1.], &[f64::INFINITY]),
            Err(Error::NonFiniteEvaluation)
        ));
        let (zero_values, zero_vectors) = eigensystem(&[0.; 4], &[0.; 3]).unwrap();
        assert_eq!(zero_values, [0.; 4]);
        for (i, vector) in zero_vectors.iter().enumerate() {
            for (j, &entry) in vector.iter().enumerate() {
                assert_eq!(entry, f64::from(i == j));
            }
        }
        for scale in [1e-200, 1e200] {
            let (values, vectors) = eigensystem(&[2.0 * scale; 2], &[scale]).unwrap();
            for (value, expected) in values.iter().zip([1., 3.]) {
                assert!((value / scale - expected).abs() < 3e-14);
            }
            assert!((vectors[0][0].abs() - std::f64::consts::FRAC_1_SQRT_2).abs() < 3e-14);
        }
    }
}
