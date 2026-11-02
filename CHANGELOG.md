# Changelog

## Unreleased

**Breaking Changes**
- `assoc_legendre_p`, `sph_legendre_p`, and `sph_harm_y`: Accept signed `i32` degrees instead of `u32`, consistently with scalar polynomial evaluators. Negative degrees are rejected with `Error::InvalidInput` rather than requiring callers to convert a signed degree; nonnegative degrees use the same internal recurrences.

**Tests**
- Verify signed degree interoperability, rejection of negative and out-of-range degrees, and existing SciPy reference results.

## 0.1

### 0.1.0

**New Features**
- Initial standalone extraction of classical orthogonal polynomials, associated Legendre functions, spherical harmonics, and Gaussian quadrature.
- Include scalar evaluators, checked `try_eval_*` variants, coefficient constructors, and quadrature rules, with independent SciPy reference fixtures.
