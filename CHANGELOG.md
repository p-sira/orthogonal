# Changelog

## 0.2

### 0.2.1

**Performance Improvements**
- `eval_sh_jacobi`, `try_eval_sh_jacobi`: Evaluate monic shifted-Jacobi polynomials directly by a three-term recurrence, removing three log-gamma evaluations and intermediate normalization per call. Use direct, low-division coefficients for ordinary parameters and overflow-safe factored ratios for extreme parameters, improving short and long recurrences.

**Bug Fixes**
- Preserve small, representable shifted-Jacobi endpoint values near admissible parameter boundaries with direct product formulas, avoiding recurrence cancellation.

**Tests**
- Compare shifted-Jacobi values with independently generated 110-digit mpmath Jacobi/Gamma references across degrees, parameter boundaries, endpoints, and interior/exterior arguments.

### 0.2.0

**Performance**
- `lpmv`: Avoid expensive high-order hypergeometric seeds for well-conditioned interior arguments by combining a stable degree recurrence with one associated-order recurrence; preserve direct evaluation near the positive endpoint to avoid cancellation.
- Keep previously supported and newly supported `lpmv` inputs in separate benchmark groups; the original benchmark measured `NaN` rejection, not a valid function evaluation.

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
