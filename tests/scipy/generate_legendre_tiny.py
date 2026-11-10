"""Generate high-precision Legendre references without calling the Rust kernel.

Uses Python's decimal arithmetic and the exact finite binomial expansion

    P_n(x) = 2**(-n) sum_k (-1)**k (2n-2k)! /
             (k! (n-k)! (n-2k)!) * x**(n-2k).

Terms are summed from the lowest power to avoid cancellation at tiny |x|.
No third-term recurrence or external packages are used. Run with Python >= 3.11.
"""

import csv
from decimal import Decimal, localcontext
from math import comb
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "fixtures"
DEGREES = (0, 1, 2, 3, 5, 16, 17, 64, 65, 127, 128, 1023, 1024, 2047, 2048, 4095, 4096)
ARGUMENTS = (-9.9e-6, -1e-12, 1e-12, 1e-8, 1e-6, 9.9e-6)


def legendre_binomial(n: int, x: Decimal) -> Decimal:
    a, parity = divmod(n, 2)
    if parity:
        coefficient = Decimal(a + 1) * Decimal(comb(2 * a + 2, a + 1)) / Decimal(1 << n)
    else:
        coefficient = Decimal(comb(2 * a, a)) / Decimal(1 << n)
    if a % 2:
        coefficient = -coefficient

    term = coefficient * x**parity
    result = Decimal(0)
    for j in range(a + 1):
        result += term
        if j == a:
            break
        numerator = (n + parity + 2 * j + 2) * (n + parity + 2 * j + 1) * (a - j)
        denominator = (a + parity + j + 1) * (parity + 2 * j + 2) * (parity + 2 * j + 1)
        term *= -Decimal(numerator) / Decimal(denominator) * x * x
        if abs(term) < Decimal("1e-92") * max(abs(result), Decimal("1e-20")):
            break
    return result


def main() -> None:
    ROOT.mkdir(exist_ok=True)
    with localcontext() as ctx:
        ctx.prec = 100
        with (ROOT / "legendre_tiny_inputs.csv").open("w", newline="") as input_file:
            with (ROOT / "legendre_tiny_expected.csv").open("w", newline="") as expected_file:
                inputs = csv.writer(input_file)
                expected = csv.writer(expected_file)
                inputs.writerow(["n", "x"])
                expected.writerow(["value"])
                for x in ARGUMENTS:
                    # Exact decimal representation of the same binary64 input
                    # parsed by the Rust tests, including its rounding residue.
                    binary64_text = repr(x)
                    for n in DEGREES:
                        inputs.writerow([n, binary64_text])
                        expected.writerow([str(legendre_binomial(n, Decimal(x)))])


if __name__ == "__main__":
    main()
