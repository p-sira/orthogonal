#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["mpmath==1.3.0"]
# ///
"""Generate tests/fixtures/q_domain_expected.csv with independent Legendre-Q values.

Use mpmath's Ferrers (type=2) convention in (-1, 1); the real exterior
branch is its type=3 value at positive |x|, reflected with (-1)^(n+m+1).
Evaluating type=3 directly at negative x uses a different odd-order branch.
430-digit precision resolves derivatives at x=1e200.
"""

import csv
import os
from pathlib import Path
import tempfile

from mpmath import mp

CASES = [
    (0, 0, 0.3, "2e-12"),
    (1, 0, 0.3, "2e-12"),
    (2, 1, 0.3, "2e-12"),
    (3, 2, -0.7, "2e-12"),
    (1, 0, 2.0, "2e-12"),
    (1, 0, -2.0, "2e-12"),
    (2, 1, 2.0, "2e-12"),
    (2, 1, -2.0, "2e-12"),
    (3, 2, 2.0, "2e-12"),
    (3, 2, -2.0, "2e-12"),
    (5, 64, 2.0, "2e-11"),
    (20, 20, 2.0, "2e-11"),
    (64, 1023, 1.001, "2e-10"),
    (0, 4096, 1.0 + 2.0**-52, "3e-10"),
    (1, 4096, 1.0 + 2.0**-52, "3e-10"),
    (2, 4096, 1.0 + 1e-12, "3e-10"),
    (8, 4096, 1.0 + 1e-12, "3e-10"),
    (100, 0, 1e200, "2e-10"),
    (100, 1, 1e154, "2e-10"),
    (100, 1, 1e200, "2e-10"),
]


def reference(order, degree, x):
    # Very large exterior arguments need extra precision to resolve the
    # cancellation in Q and its derivatives. Ordinary arguments are faster.
    mp.dps = 430 if abs(x) >= 1e100 else 110
    a = mp.mpf(abs(x))
    mode = 3 if abs(x) > 1 else 2
    argument = a if mode == 3 else mp.mpf(x)
    value = mp.re(mp.legenq(degree, order, argument, type=mode))
    following_order = mp.re(mp.legenq(degree, order + 1, argument, type=mode))
    if mode == 3:
        root = mp.sqrt((argument - 1) * (argument + 1))
        derivative = order * argument / (root * root) * value + following_order / root
    else:
        root = mp.sqrt((1 - argument) * (1 + argument))
        derivative = -order * argument / (root * root) * value - following_order / root
    if x < -1:
        parity = (-1) ** (degree + order + 1)
        value *= parity
        derivative *= -parity
    return format(float(value), ".17g"), format(float(derivative), ".17g")


def main():
    path = Path(__file__).resolve().parent.parent / "fixtures" / "q_domain_expected.csv"
    rows = [("order", "degree", "x", "value", "derivative", "tolerance")]
    for m, n, x, tol in CASES:
        v, d = reference(m, n, x)
        rows.append((m, n, repr(x), v, d, tol))
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", delete=False,
        ) as out:
            temp_path = Path(out.name)
            writer = csv.writer(out, lineterminator="\n")
            writer.writerows(rows)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
