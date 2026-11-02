#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["mpmath==1.3.0"]
# ///
"""High-precision Ferrers P references for the low-order-seeded fast path.

Evaluate the exact binary64 degree and argument with mpmath 1.3.0 at 110 dps.
Inputs and expected outputs are independent fixture files. Regenerate using
`uv run --script tests/scipy/generate_lpmv_fast.py`.
"""

import csv
from pathlib import Path
import mpmath as mp

assert mp.__version__ == "1.3.0"
mp.mp.dps = 110
folder = Path(__file__).resolve().parents[1] / "fixtures"
inputs = folder / "lpmv_fast_inputs.csv"
expected = folder / "lpmv_fast_expected.csv"
# All degree/order pairs enter the fast path; large degrees, negative orders,
# near-endpoint, and cancellation-heavy oscillatory inputs are represented.
pairs = [(5.5, 2), (12.25, 3), (30.5, 10), (55.5, 20),
         (100.25, 5), (100.25, 35), (200.25, 20), (500.5, 25),
         (1000.25, 5)]
xs = (-0.3, 0.0, 0.2, 0.4, 0.8, 0.999999)
cases = [(m, degree, x) for degree, m in pairs for x in xs]
cases += [(-m, degree, x) for degree, m in pairs for x in (0.4, 0.999999)]
with inputs.open('w', newline='') as source, expected.open('w', newline='') as output:
    input_writer = csv.writer(source, lineterminator='\n')
    reference_writer = csv.writer(output, lineterminator='\n')
    input_writer.writerow(('id', 'order', 'degree', 'x'))
    reference_writer.writerow(('id', 'expected'))
    for index, (m, v, x) in enumerate(cases):
        result = float(mp.legenp(mp.mpf(v), m, mp.mpf(x), type=2))
        input_writer.writerow((index, m, repr(v), repr(x)))
        reference_writer.writerow((index, repr(result)))
print(f"Wrote {len(cases)} high precision Ferrers P references")
