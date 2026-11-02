#!/usr/bin/env python3
"""Regenerate independent Ferrers P expected values using mpmath 1.3.0.

Run with a Python environment containing mpmath==1.3.0:

    python tests/scipy/generate_lpmv_domain.py

Each decimal input is converted through binary64 before passing it to mpmath,
so the high-precision reference evaluates the exact f64 input Rust receives.
"""

from __future__ import annotations

import csv
from pathlib import Path

import mpmath as mp


def main() -> None:
    if mp.__version__ != "1.3.0":
        raise RuntimeError(f"Requires mpmath==1.3.0; installed {mp.__version__}")

    folder = Path(__file__).resolve().parents[1] / "fixtures"
    inputs = folder / "lpmv_domain_inputs.csv"
    output = folder / "lpmv_domain_expected.csv"
    mp.mp.dps = 110
    with inputs.open(newline="") as handle:
        cases = list(csv.DictReader(handle))
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "expected"])
        for case in cases:
            order = int(case["order"])
            degree = mp.mpf(float(case["degree"]))
            x = mp.mpf(float(case["x"]))
            expected = float(mp.legenp(degree, order, x, type=2))
            writer.writerow([case["id"], repr(expected)])

    print(f"Wrote {len(cases)} high-precision reference values to {output}")


if __name__ == "__main__":
    main()
