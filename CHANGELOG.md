# Changelog

## 0.2.2

### Performance

- Reduce allocations and redundant transforms when constructing polynomial coefficients ([#6](https://github.com/p-sira/orthogonal/pull/6)).
- Accelerate associated-Legendre and spherical-harmonic tables, and Legendre tables at zero ([#6](https://github.com/p-sira/orthogonal/pull/6)).
- Generate all six Chebyshev Gaussian quadrature rules in linear time, with more accurate shifted endpoints ([#6](https://github.com/p-sira/orthogonal/pull/6)).
- Accelerate Legendre evaluation near zero, Chebyshev evaluation outside [-1, 1], generalized Laguerre evaluation, and Jacobi evaluation with zero parameters ([#6](https://github.com/p-sira/orthogonal/pull/6)).

### Fixed

- Improve high-degree Chebyshev accuracy near both endpoints, inside and outside [-1, 1] ([#7](https://github.com/p-sira/orthogonal/pull/7)).
- Prevent integer overflow when evaluating Chebyshev polynomials of the second kind at the maximum supported degree ([#6](https://github.com/p-sira/orthogonal/pull/6)).

## 0.2.1

### Performance

- Accelerate `eval_sh_jacobi` and `try_eval_sh_jacobi` with a direct monic recurrence ([#5](https://github.com/p-sira/orthogonal/pull/5)).

### Fixed

- Preserve small shifted-Jacobi endpoint values near parameter boundaries ([#5](https://github.com/p-sira/orthogonal/pull/5)).

## 0.2.0

### Breaking changes

- Change degree arguments from `u32` to `i32` in `assoc_legendre_p`, `sph_legendre_p`, and `sph_harm_y`. Negative degrees return `Error::InvalidInput`.

### Performance

- Accelerate `lpmv` for well-conditioned interior arguments.

### Fixed

- Restore Sci-compatible degree and order limits up to 4096 in `lqn` and `lqmn`, and support the real exterior associated-Q branch for positive orders.
- Restore the documented real-degree and integer-order domain of `lpmv`, including noninteger degrees near the negative endpoint.

## 0.1.0

### Added

- Initial standalone release of classical orthogonal polynomials, associated Legendre functions, spherical harmonics, and Gaussian quadrature.
- Scalar evaluators, checked `try_eval_*` variants, polynomial coefficient constructors, and quadrature rules.
