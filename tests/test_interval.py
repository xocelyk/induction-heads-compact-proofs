"""Checks of interval.py against exact rational arithmetic (fractions) and high-precision exp (decimal).

Run: python tests/test_interval.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np
from decimal import Decimal, getcontext
from fractions import Fraction as Fr
from interval import Iv, exp_pt, exp, softmax_box, lp_min_lo, lp_max_hi, sum_dn, sum_up, sqrt_recip, ratio_min_lo, ratio_max_hi, _LN2
from proof_v4 import ratio_min

getcontext().prec = 60
rng = np.random.default_rng(0)
F = lambda a: [Fr(float(v)) for v in np.ravel(a)]


def test_ln2():
    ln2 = Decimal(2).ln()
    assert Decimal(float(_LN2.lo)) < ln2 < Decimal(float(_LN2.hi))


def test_arith():
    for scale in (1.0, 1e-8, 1e8):
        a, b = rng.normal(size=2000) * scale, rng.normal(size=2000)
        A = Iv.pt(a) + Iv.pt(b) * 1e-3; B = Iv.pt(b) - Iv.pt(a) * 1e-3     # genuine intervals
        for op, ex in ((A + B, lambda x, y: x + y), (A - B, lambda x, y: x - y), (A * B, lambda x, y: x * y)):
            # exact results at the four corners must lie inside
            for xa in (A.lo, A.hi):
                for xb in (B.lo, B.hi):
                    for l, h, u, v in zip(F(op.lo), F(op.hi), F(xa), F(xb)):
                        assert l <= ex(u, v) <= h


def test_sum():
    for n in (1, 3, 32, 1000):
        x = rng.normal(size=(50, n)) * 10.0 ** rng.integers(-12, 12, size=(50, n))
        lo, hi = sum_dn(x, -1), sum_up(x, -1)
        for r in range(50):
            s = sum(F(x[r]))
            assert Fr(float(lo[r])) <= s <= Fr(float(hi[r]))


def test_dot():
    A, M = rng.normal(size=(4, 5, 32)), rng.normal(size=(32, 7))
    C = Iv.pt(A).dot(M)
    assert C.shape == (4, 5, 7)
    for i in range(4):
        for j in range(5):
            for k in range(7):
                s = sum(u * v for u, v in zip(F(A[i, j]), F(M[:, k])))
                assert Fr(float(C.lo[i, j, k])) <= s <= Fr(float(C.hi[i, j, k]))
    assert np.max((C.hi - C.lo) / np.abs(A @ M).clip(1e-3)) < 1e-11


def test_exp():
    x = np.concatenate([rng.uniform(-80, 80, 3000), rng.uniform(-1e-3, 1e-3, 500), [0.0, -700.0, 700.0, -699.9, -0.5 * float(_LN2.lo)]])
    e = exp_pt(x)
    for xi, l, h in zip(x, e.lo, e.hi):
        t = Decimal(float(xi)).exp()
        assert Decimal(float(l)) <= t <= Decimal(float(h)), xi
    assert np.max((e.hi - e.lo) / e.lo) < 1e-12
    e = exp_pt(np.array([-800.0, -1e9]))
    assert np.all(e.lo == 0) and np.all(e.hi > 0) and np.all(e.hi < 1e-300)
    try:
        exp_pt(np.array([701.0])); assert False
    except FloatingPointError:
        pass


def test_sqrt_recip():
    assert sqrt_recip(16).lo <= 0.25 <= sqrt_recip(16).hi
    for c in (2, 3, 10, 48):
        s = sqrt_recip(c)
        assert Fr(float(s.lo)) ** 2 * c < 1 < Fr(float(s.hi)) ** 2 * c


def test_softmax_box():
    for _ in range(200):
        M = int(rng.integers(1, 8))
        c, w = rng.normal(size=M) * 5, rng.uniform(0, 2, size=M)
        l, u = softmax_box(Iv(c - w, c + w))
        assert np.all(l <= u)
        for _ in range(20):
            s = [Decimal(float(v)) for v in rng.uniform(c - w, c + w)]
            if rng.random() < 0.3:                         # extreme corner: one score high, the rest low
                j = int(rng.integers(M)); s = [Decimal(float(c[m] + w[m] if m == j else c[m] - w[m])) for m in range(M)]
            e = [v.exp() for v in s]; tot = sum(e)
            for m in range(M):
                assert Decimal(float(l[m])) <= e[m] / tot <= Decimal(float(u[m]))


def _lp_exact(V, l, u):
    """Exact greedy LP in rationals: min sum V w, l <= w <= u, sum w = 1."""
    V, l, u = F(V), F(l), F(u)
    R = 1 - sum(l); val = sum(a * b for a, b in zip(V, l))
    for m in sorted(range(len(V)), key=lambda m: V[m]):
        t = min(u[m] - l[m], R); val += V[m] * t; R -= t
    assert R == 0
    return val


def test_lp():
    for _ in range(300):
        M = int(rng.integers(1, 9))
        w = rng.dirichlet(np.ones(M))
        l = w * rng.uniform(0, 1, M); u = np.minimum(1, w + rng.uniform(0, 1, M) * (1 - w))
        if M == 1: l = u = np.ones(1)
        V = rng.normal(size=M) * 10
        lo, hi = lp_min_lo(V, l, u), lp_max_hi(V, l, u)
        ex_lo, ex_hi = _lp_exact(V, l, u), -_lp_exact(-V, l, u)
        assert Fr(float(lo)) <= ex_lo and ex_hi <= Fr(float(hi))
        assert float(ex_lo) - lo < 1e-11 and hi - float(ex_hi) < 1e-11      # and tight


def test_ratio():
    import itertools
    for _ in range(300):
        M, Tn = int(rng.integers(1, 5)), 4
        S, V, mask = rng.normal(size=(M, Tn)) * 3, rng.normal(size=(M, Tn)) * 5, rng.random((M, Tn)) < 0.7
        mask[:, 0] = True
        e = [[Decimal(float(S[m, t])).exp() for t in range(Tn)] for m in range(M)]
        vals = [sum(e[m][t[m]] * Decimal(float(V[m, t[m]])) for m in range(M)) / sum(e[m][t[m]] for m in range(M))
                for t in itertools.product(range(Tn), repeat=M) if all(mask[m, t[m]] for m in range(M))]
        g = ratio_min(S, V, mask)
        lo = ratio_min_lo(Iv.pt(S), V, mask, g); hi = ratio_max_hi(Iv.pt(S), V, mask, -ratio_min(S, -V, mask))
        tol = Decimal(10) ** -40                           # the reference quotients are themselves rounded at 60 digits
        assert Decimal(float(lo)) <= min(vals) + tol and max(vals) - tol <= Decimal(float(hi))
        assert float(min(vals)) - lo < 1e-9 and hi - float(max(vals)) < 1e-9      # and tight
        assert Decimal(float(ratio_min_lo(Iv.pt(S), V, mask, g + 1.0))) <= min(vals) + tol          # a wrong guess stays sound
        assert ratio_min_lo(Iv.pt(S), V, mask, g + 1.0) == np.where(mask, V, np.inf).min()


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
