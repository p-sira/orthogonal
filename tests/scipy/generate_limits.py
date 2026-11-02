# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1", "mpmath==1.3.0"]
# ///
"""Independent arbitrary-precision boundary and high-degree references.

Run with pinned SciPy / NumPy / mpmath: uv run --offline tests/scipy/generate_limits.py.
Sources: mpmath Legendre P / Ferrers P, and the DLMF 14.3.7 hypergeometric
representation of exterior Legendre Q (avoids forward-recurrence instability).
"""
import csv
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy

assert (np.__version__, scipy.__version__, mp.__version__) == ("2.5.4", "1.18.1", "1.3.0")
mp.mp.dps = 110
OUT = Path(__file__).resolve().parents[1] / "fixtures" / "limits_expected.csv"


def exterior_q(n, x):
    # DLMF 14.3.7 / Gauss hypergeometric exterior representation for x>1.
    # m=0: Q_n(x) = 2^n (n!)^2 / (2n+1)! / x^(n+1)
    #                 * 2F1((n+1)/2,(n+2)/2;n+3/2;1/x^2).
    coefficient = 2**n * mp.factorial(n)**2 / mp.factorial(2*n+1)
    return coefficient / x**(n+1) * mp.hyp2f1(mp.mpf(n+1)/2, mp.mpf(n+2)/2,
                                               mp.mpf(n)+mp.mpf('1.5'), 1/x**2)


with OUT.open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['kind', 'n', 'm', 'x', 'value', 'slope'])
    for n, xs in [
        (4097, ('0', '0.3', '0.999999', '1.001')),
        (8192, ('0', '0.8', '-0.999999', '1.000001')),
        (16384, ('0', '0.45', '0.9999999999')),
        (32768, ('0', '0.9')),
    ]:
        for x in xs:
            value = mp.legenp(n, 0, mp.mpf(float(x)))
            writer.writerow(['legendre', n, 0, x, repr(float(value)), ''])
    for n, x in [(4097, '0.2'), (8192, '0.45'), (16384, '-0.3')]:
        x_mpf = mp.mpf(float(x))
        writer.writerow(['jacobi', n, 0, x,
                         repr(float(mp.jacobi(n, mp.mpf(float('0.2')),
                                              mp.mpf(float('0.3')), x_mpf))), ''])
        writer.writerow(['gegenbauer', n, 0, x,
                         repr(float(mp.gegenbauer(n, mp.mpf(float('0.8')), x_mpf))), ''])
    for n, x in [(4097, '0.2'), (8192, '2'), (16384, '5')]:
        writer.writerow(['laguerre', n, 0, x,
                         repr(float(mp.laguerre(n, mp.mpf(float('0.4')),
                                                mp.mpf(float(x))))), ''])
    for nu in ('0.5', '-0.25', '3.125', '12.5', '31.5'):
        for m in (-3, -1, 0, 1, 3):
            for x in ('0.91', '0.999', '0.999999999999', '1'):
                value = mp.mpf(0) if x == "1" and m != 0 else mp.legenp(mp.mpf(nu), m, mp.mpf(float(x)))
                writer.writerow(['lpmv', nu, m, x, repr(float(value)), ''])
    for n in (0, 1, 2, 8, 32, 64, 128):
        for x in ('1.00000001', '1.01', '1.5', '2', '10'):
            a = mp.mpf(float(x))
            q = exterior_q(n, a)
            d = 1 / (1-a*a) if n == 0 else n * (exterior_q(n-1,a) - a*q) / (1-a*a)
            for sign in (-1, 1):
                # Q_n(-x)=(-1)^(n+1) Q_n(x), Q_n'(-x)=(-1)^n Q_n'(x).
                writer.writerow(['q', n, 0, repr(float(sign*a)),
                                 repr(float(q if sign > 0 else (-1)**(n+1)*q)),
                                 repr(float(d if sign > 0 else (-1)**n*d))])
print(OUT)
