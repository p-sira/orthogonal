# Changelog

## 0.2

### 0.2.2
**Performance Improvements**
- Reuse recurrence buffers during polynomial coefficient construction; avoid redundant polynomial transforms for identity mappings ([#6](https://github.com/p-sira/orthogonal/pull/6)). 
- Build associated-Legendre and spherical-harmonic tables by shared per-order recurrences, and evaluate Legendre tables at zero by an exact O(n) even-degree recurrence ([#6](https://github.com/p-sira/orthogonal/pull/6)). 
- Generate all six Chebyshev Gaussian quadrature rules with linear-time closed-form nodes and weights, including cancellation-free shifted endpoints. Use one linear-time Legendre recurrence for tiny nonzero arguments, a constant-time asymptotic coefficient for large-degree scalar values near zero, and logarithmic-time Chebyshev doubling for exterior arguments.
- Avoid redundant normalization passes in generalized Laguerre recurrence; reuse Legendre evaluation for Jacobi parameters alpha=beta=0 where stable.

**Testing**
- Add larger-degree CodSpeed workloads and independent parity checks ([#6](https://github.com/p-sira/orthogonal/pull/6)).
- Verify quadrature against the former eigensolver and analytic moments, including high-order shifted endpoints; check tiny-argument Legendre values against 100-digit fixtures, high-degree binomial coefficients, exterior Chebyshev and Laguerre identities.

### 0.2.1
**Performance Improvements**
- `eval_sh_jacobi`, `try_eval_sh_jacobi`: Evaluate monic shifted-Jacobi polynomials directly by a three-term recurrence, removing three log-gamma evaluations and intermediate normalization per call. Use direct, low-division coefficients for ordinary parameters and overflow-safe factored ratios for extreme parameters, improving short and long recurrences ([#5](https://github.com/p-sira/orthogonal/pull/5)).

**Bug Fixes**
- Preserve small, representable shifted-Jacobi endpoint values near admissible parameter boundaries with direct product formulas, avoiding recurrence cancellation ([#5](https://github.com/p-sira/orthogonal/pull/5)).

**Tests**
- Compare shifted-Jacobi values with independently generated 110-digit mpmath Jacobi/Gamma references across degrees, parameter boundaries, endpoints, and interior/exterior arguments ([#5](https://github.com/p-sira/orthogonal/pull/5)).

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
