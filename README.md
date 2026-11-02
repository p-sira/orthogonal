# orthogonal

<p align="center">
    <a href="https://opensource.org/license/BSD-3-clause" style="text-decoration:none">
        <img src="https://img.shields.io/badge/License-BSD--3--Clause-brightgreen.svg" alt="License">
    </a>
    <a href="https://crates.io/crates/orthogonal" style="text-decoration:none">
        <img src="https://img.shields.io/crates/v/orthogonal" alt="Crate">
    </a>
    <a href="https://docs.rs/orthogonal" style="text-decoration:none">
        <img src="https://img.shields.io/badge/Docs-docs.rs-blue" alt="Documentation">
    </a>
    <a href="https://app.codspeed.io/p-sira/orthogonal?utm_source=badge" style="text-decoration:none">
        <img src="https://img.shields.io/endpoint?url=https://codspeed.io/badge.json" alt="CodSpeed"/>
    </a>
</p>

Pure Rust library for classical orthogonal polynomials, associated Legendre functions, spherical harmonics, and Gaussian quadrature.

- Evaluate polynomials with `eval_*` or checked `try_eval_*` functions.
- Construct polynomial coefficients with `legendre`, `jacobi`, and other families.
- Compute Gaussian quadrature nodes and weights with `roots_*`.

```rust
use orthogonal::{eval_legendre, legendre, roots_legendre, try_eval_jacobi};

let coefficients = legendre(2, false).unwrap();
assert_eq!(coefficients, vec![1.5, 0.0, -0.5]);
assert_eq!(eval_legendre(2, 0.5_f64), -0.125);

let value = try_eval_jacobi(3, 0.5_f64, 0.25, 0.2).unwrap();
assert!(value.is_finite());

let (nodes, weights) = roots_legendre(3).unwrap();
assert_eq!(nodes.len(), weights.len());
```

`eval_*` functions return `NaN` for invalid inputs. Checked evaluations, coefficient constructors, and quadrature return `Result<_, orthogonal::Error>`. Coefficients are in descending powers; quadrature nodes are sorted ascending.

The definitions are cross-checked against SciPy and the [NIST DLMF](https://dlmf.nist.gov/18).
