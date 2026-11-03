#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["mpmath==1.3.0"]
# ///
"""Independent 100-digit Jacobi/Gamma references for monic shifted Jacobi.

Compute the exact binary64 inputs in mpmath, not their rounded display
decimal strings; the resulting CSV tests the public f64 API's actual inputs.
The reference uses mpmath's Jacobi polynomial and gamma functions rather
than the three-term recurrence used by the Rust implementation.
"""

import csv
import math
import os
from pathlib import Path
import tempfile

from mpmath import mp


PARAMETERS = [
    (2.5, 1.5),
    (0.0, 0.5),  # k=1 B coefficient has a removable 0/0
    (1.0, 0.5),
    (0.2, 0.01),
    (-0.8, 0.1),
    (-0.999999, 0.0000005),
    (1e-8, 0.9),
    (1.0000001, 1.999999),
    (10.0, 0.01),
    (1e5, 3e4),
]


def cases():
    for p, q in PARAMETERS:
        for n in (0, 1, 2, 3, 10, 32, 64):
            for x in (-0.9, 0.0, 0.29, 0.77, 1.0, 1.5):
                yield n, p, q, x
    for n in (128, 256, 512, 1024, 4096):
        yield n, 2.5, 1.5, 1.5
    for n in (128, 256, 512):
        yield n, 2.5, 1.5, 0.5


def reference(n, p, q, x):
    if n == 0:
        return 1.0  # The ratio and Gamma(p) cancel even at p=0.
    mp.dps = 110
    pp, qq, xx = (mp.mpf(value) for value in (p, q, x))
    ratio = mp.factorial(n) * mp.gamma(n + pp) / mp.gamma(2 * n + pp)
    value = ratio * mp.jacobi(n, pp - qq, qq - 1, 2 * xx - 1)
    answer = float(value)
    return answer


def main():
    path = Path(__file__).resolve().parent.parent / "fixtures" / "shifted_jacobi.csv"
    rows = [("n", "p", "q", "x", "expected")]
    for n, p, q, x in cases():
        value = reference(n, p, q, x)
        if not math.isfinite(value) or value == 0.0:
            continue  # Avoid unrepresentable true values in relative tests.
        rows.append((n, repr(p), repr(q), repr(x), format(value, ".17g")))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", delete=False,
        ) as out:
            temporary = Path(out.name)
            writer = csv.writer(out, lineterminator="\n")
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Wrote {len(rows)-1} independent reference cases to {path}")


if __name__ == "__main__":
    main()
