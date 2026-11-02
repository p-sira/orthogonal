# Changelog

## Unreleased

**Bug Fixes**
- `lqn` and `lqmn`: Restore Sci-compatible degree and order limits up to 4096 and the established matrix-size bound; support the real exterior associated-Q branch for positive orders.
- `lpmv`: Restore the documented Sci real-degree and integer-order input domain, including noninteger degrees near the negative endpoint.

**Breaking Changes**
- `assoc_legendre_p`, `sph_legendre_p`, and `sph_harm_y`: Accept signed `i32` degrees instead of `u32`, consistently with scalar polynomial evaluators. Negative degrees are rejected with `Error::InvalidInput` rather than requiring callers to convert a signed degree; nonnegative degrees use the same internal recurrences.

**Tests**
- Verify signed degree interoperability, rejection of negative and out-of-range degrees, and existing SciPy reference results.

## 0.1

### 0.1.0

**New Features**
- Initial standalone extraction of classical orthogonal polynomials, associated Legendre functions, spherical harmonics, and Gaussian quadrature.
- Include scalar evaluators, checked `try_eval_*` variants, coefficient constructors, and quadrature rules, with independent SciPy reference fixtures.
