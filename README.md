# orthogonal

Standalone Rust implementation of classical orthogonal polynomials, associated Legendre functions, and Gaussian quadrature.

The library provides:
* polynomial coefficient constructors (`legendre`, `jacobi`, `gegenbauer`, `genlaguerre`, `hermite`, `chebyt`, `chebyu` and variants),
* scalar evaluators (`eval_legendre`, `eval_jacobi`, etc.),
* Ferrers, associated Legendre Q and spherical Legendre functions,
* spherical harmonics, and
* Gaussian quadrature nodes/weights (`roots_*`).

No `sci` or `sci-special` dependency is required.

```rust
use orthogonal::{eval_legendre, legendre, roots_legendre};

let p = legendre(3, false).unwrap(); // [2.5, 0, -1.5, 0]
assert!((eval_legendre(3, 0.25_f64) + 0.3359375).abs() < 1e-14);

let (x, w) = roots_legendre(8).unwrap();
assert!((w.iter().sum::<f64>() - 2.0).abs() < 1e-13);
```

Polynomial coefficients are returned in **descending** powers. The scalar evaluators accept `f32` and `f64` inputs and return NaN on invalid arguments. Coefficient constructors and quadrature return `Result` with an explicit error type. Quadrature supports orders `1..=1024`, coefficient construction supports degrees `0..=256`, and scalar evaluation supports degrees `0..=1_000_000` (with linear work for most families). `legendre_p_all` likewise supports degree `1_000_000` using linear time and memory. Quadrature weights integrate the respective unnormalized family weight on its domain. Near multiple roots, under extreme parameter values, and at high degrees, floating-point accuracy may degrade. Consult the per-function rustdoc for the underlying weight, conventions, and restrictions.

Independent SciPy reference generation (pinned versions) is in `tests/scipy/generate.py`; inputs and expected values are stored separately in `tests/fixtures/` so ordinary `cargo test` needs no Python.

The package is based on orthogonal polynomial implementations extracted from [Sci](https://github.com/p-sira/sci) under the BSD 3-Clause license, with mathematical definitions and conventions cross-checked against the [NIST DLMF Chapter 18](https://dlmf.nist.gov/18) and SciPy. The coefficient and QL routines derive from the Sci implementations; scalar recurrences are derived from xsf/SciPy-style formulas. License: [BSD-3-Clause](LICENSE).

## Mathematical conventions

| Family | Classical weight | Domain |
| --- | --- | --- |
| Jacobi `P_n^(α,β)` | `(1-x)^α (1+x)^β` | `[-1,1]`, `α,β>-1` |
| Legendre `P_n` | `1` | `[-1,1]` |
| Gegenbauer `C_n^λ` | `(1-x²)^(λ-1/2)` | `[-1,1]`, `λ>-1/2` |
| Laguerre `L_n^α` | `x^α exp(-x)` | `[0,∞)`, `α>-1` |
| Physicists' Hermite `H_n` | `exp(-x²)` | `(-∞,∞)` |
| Probabilists' Hermite `He_n` | `exp(-x²/2)` | `(-∞,∞)` |
| Chebyshev `T_n`, `U_n` | `(1-x²)^(-1/2)` and `sqrt(1-x²)` | `[-1,1]` |

`chebyc`, `chebys`, and shifted families evaluate these same polynomials after affine transformations. `sh_jacobi` follows SciPy's `(p,q)` convention, with `p-q>-1`, `q>0`. Associated Ferrers P includes the Condon–Shortley phase; spherical Legendre P includes spherical-harmonic normalization; spherical harmonic arguments are `(n,m,theta,phi)` with polar angle first, in radians. `lqn` implements real second-kind Legendre Q for finite `x` with `|x|!=1` (interior Ferrers and exterior real branches); `lqmn` also accepts exterior arguments for order zero. Positive associated Q orders remain restricted to the Ferrers interior `-1<x<1`. Both APIs bound degree/order by 128; the endpoints are singular. The `lpmv` real-degree continuation uses bounded hypergeometric series for moderate noninteger degree/order and `-0.9<=x<=1` (see its rustdoc). For all these APIs, no runtime Python is needed.

The construction's tridiagonal QL solver is private, with its own unit tests. Source algorithm provenance and published mathematical references remain in Rustdoc, and no external linked numerical backend is required.
