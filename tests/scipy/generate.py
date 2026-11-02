# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Write independently generated reference data for the `orthogonal` crate.

Run `uv run --offline tests/scipy/generate.py` from the crate root, using pinned
NumPy 2.5.4 and SciPy 1.18.1. Save mathematical inputs and expected results in
separate CSV files; never import the Rust implementation. The reference values
come from SciPy's own polynomial, quadrature, and Legendre APIs.
"""

import csv
from pathlib import Path
import numpy as np
import scipy
from scipy import special

ROOT = Path(__file__).resolve().parents[1] / "fixtures"
ROOT.mkdir(exist_ok=True)
assert (np.__version__, scipy.__version__) == ("2.5.4", "1.18.1")

FAMILIES = {
    "jacobi": (0.3, 0.7),
    "legendre": (0.0, 0.0),
    "gegenbauer": (0.8, 0.0),
    "genlaguerre": (0.4, 0.0),
    "laguerre": (0.0, 0.0),
    "hermite": (0.0, 0.0),
    "hermitenorm": (0.0, 0.0),
    "chebyt": (0.0, 0.0),
    "chebyu": (0.0, 0.0),
    "chebyc": (0.0, 0.0),
    "chebys": (0.0, 0.0),
    "sh_chebyt": (0.0, 0.0),
    "sh_chebyu": (0.0, 0.0),
    "sh_legendre": (0.0, 0.0),
    "sh_jacobi": (2.5, 1.5),
}
# Actual Sci API calls, with shape parameters passed only where applicable.
# The dictionaries cover all classical scalar and construction APIs.
def family_args(name, a, b):
    if name in ("jacobi", "sh_jacobi"):
        return (a, b)
    if name in ("genlaguerre", "gegenbauer"):
        return (a,)
    return ()


def format_values(values):
    return ";".join(repr(float(v)) for v in np.asarray(values).ravel())


with (ROOT / "inputs.csv").open("w", newline="") as input_file, (ROOT / "expected.csv").open("w", newline="") as expected_file:
    inputs = csv.writer(input_file)
    expected = csv.writer(expected_file)
    inputs.writerow(["id", "kind", "name", "n", "a", "b", "x", "monic"])
    expected.writerow(["id", "values"])
    counter = 0

    def record(kind, name, n, a, b, x, monic, values):
        global counter
        id_ = counter
        counter += 1
        inputs.writerow([id_, kind, name, n, a, b, x, int(monic)])
        expected.writerow([id_, format_values(values)])

    for name, (a, b) in FAMILIES.items():
        for n in (0, 1, 2, 3, 5, 8, 12):
            for monic in (False, True):
                poly = getattr(special, name)(n, *family_args(name,a,b), monic=monic)
                record("coeff", name, n, a, b, 0.0, monic, poly.c)
        for n in (1, 2, 3, 5, 8, 16, 32, 64, 128, 256):
            nodes, weights = getattr(special, "roots_"+name)(n, *family_args(name,a,b))
            record("roots", name, n, a, b, 0.0, False, np.r_[nodes, weights])
        for n in (0, 1, 2, 3, 5, 8, 12, 16):
            for x in (-0.9, -0.3, 0.0, 0.25, 0.77, 1.0):
                value = getattr(special, "eval_"+name)(n, *family_args(name,a,b), x)
                record("eval", name, n, a, b, x, False, [value])

    # Ferrers associated Legendre with Condon-Shortley phase and normalized
    # spherical-associated values at polar angle theta.
    for n in (0, 1, 2, 5, 12, 24):
        for m in (-n, -min(n, 3), 0, min(n, 3), n):
            for x in (-0.8, -0.25, 0.0, 0.3, 0.8):
                record("assoc", "assoc_legendre_p", n, m, 0.0, x, False, [special.lpmv(m, n, x)])
            for theta in (0.1, 0.7, 1.3, 2.7):
                # Taking phi=0 exposes real normalized associated Legendre.
                amplitude = special.sph_harm_y(n,m,theta,0.0).real
                record("spherical", "sph_legendre_p", n, m, 0.0, theta, False, [amplitude])
    for n in (0,1,2,3,6,12,24):
        for x in (-0.8,-0.25,0.0,0.35,0.8):
            values, derivatives = special.lqn(n,x)
            record("q", "lqn", n, 0.0,0.0,x,False,np.r_[values,derivatives])

    for m in (0,1,2,3,5):
        for n in (0,1,2,4,8):
            for x in (-0.6,0.0,0.45,0.8):
                values, derivatives = special.lqmn(m,n,x)
                record("assoc_q", "lqmn", n, float(m),0.0,x,False,np.r_[values.ravel(),derivatives.ravel()])
    for degree in (-0.5,-0.25,0.5,2.5,5.25,9.5):
        for m in (-3,-1,0,1,3):
            for x in (-0.8,-0.3,0.0,0.4,0.8):
                record("lpmv", "lpmv", m,degree,0.0,x,False,[special.lpmv(m,degree,x)])
print(f"Generated {counter} independent SciPy cases in {ROOT}")
