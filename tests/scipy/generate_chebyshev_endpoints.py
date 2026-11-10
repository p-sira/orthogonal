"""Generate 100-digit Chebyshev endpoint references using trigonometric identities.

Requires mpmath. Inputs are converted from binary64 exactly before evaluation.
Run from any directory with Python >= 3.11.
"""

import csv
import math
from pathlib import Path

import mpmath as mp


def main() -> None:
    mp.mp.dps = 100
    target = Path(__file__).resolve().parents[1] / "fixtures" / "chebyshev_endpoints.csv"
    arguments = (
        math.nextafter(0.99, 0.0),
        0.99,
        0.999,
        1.0 - 1e-10,
        math.nextafter(1.0, 0.0),
        math.nextafter(1.0, math.inf),
        1.0 + 1e-10,
        1.000001,
        1.01,
        math.nextafter(1.1, 0.0),
        1.1,
    )
    with target.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("family", "n", "x", "expected"))
        for n in (63, 64, 65, 1024, 999_999, 1_000_000):
            # Include a binary64 input close to a T_n root near the endpoint.
            for x in (*arguments, math.cos(math.pi / (2 * n))):
                exact_x = mp.mpf(x)
                if x < 1:
                    theta = mp.acos(exact_x)
                    first = mp.cos(n * theta)
                    second = mp.sin((n + 1) * theta) / mp.sin(theta)
                else:
                    theta = mp.acosh(exact_x)
                    first = mp.cosh(n * theta)
                    second = mp.sinh((n + 1) * theta) / mp.sinh(theta)
                for family, value in (("T", first), ("U", second)):
                    writer.writerow((family, n, repr(x), repr(float(value))))


if __name__ == "__main__":
    main()
