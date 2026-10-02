"""Outward-rounded interval arithmetic on numpy float64 arrays.

An `Iv` holds arrays lo <= hi and encloses a real-valued array elementwise. Every operation returns an
enclosure of the exact real result, using only these facts about the floating-point environment:

  * np.add / subtract / multiply / divide are single IEEE-754 operations rounded to nearest, so the exact
    result lies between the two floats adjacent to the computed one (`dn`, `up` step to those).
  * np.sum performs additions in some order. For any order, |fl(sum x) - sum x| <= g * sum|x| with
    g = (n-1)u / (1 - (n-1)u), u = 2**-53 (Higham, Accuracy and Stability of Numerical Algorithms, sec. 4.2).
  * np.minimum / maximum / nextafter / ldexp (away from under/overflow), negation and indexing are exact.

Nothing here relies on BLAS, einsum or the accuracy of libm: contractions are an elementwise product
followed by `sum`, and exp is computed from the basic operations.
"""
import numpy as np

U = 2.0 ** -53


def dn(x):
    return np.nextafter(x, -np.inf)


def up(x):
    return np.nextafter(x, np.inf)


def sum_dn(x, axis=-1):
    """Float array <= the exact sum of float array x along axis."""
    n = x.shape[axis]
    err = up((n * 2 * U) * up(np.sum(np.abs(x), axis=axis)))
    return dn(np.sum(x, axis=axis) - err)


def sum_up(x, axis=-1):
    return -sum_dn(-x, axis)


class Iv:
    __slots__ = ('lo', 'hi')

    def __init__(self, lo, hi):
        self.lo, self.hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)

    @staticmethod
    def pt(a):
        a = np.asarray(a, dtype=np.float64)
        return Iv(a, a)

    @staticmethod
    def of(a):
        return a if isinstance(a, Iv) else Iv.pt(a)

    @property
    def shape(self):
        return self.lo.shape

    def __getitem__(self, idx):
        return Iv(self.lo[idx], self.hi[idx])

    def __neg__(self):
        return Iv(-self.hi, -self.lo)

    def __add__(self, o):
        o = Iv.of(o)
        return Iv(dn(self.lo + o.lo), up(self.hi + o.hi))

    __radd__ = __add__

    def __sub__(self, o):
        o = Iv.of(o)
        return Iv(dn(self.lo - o.hi), up(self.hi - o.lo))

    def __rsub__(self, o):
        return Iv.of(o) - self

    def __mul__(self, o):
        o = Iv.of(o)
        p = [self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi]
        lo = np.minimum(np.minimum(p[0], p[1]), np.minimum(p[2], p[3]))
        hi = np.maximum(np.maximum(p[0], p[1]), np.maximum(p[2], p[3]))
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            raise FloatingPointError('non-finite interval product')
        return Iv(dn(lo), up(hi))

    __rmul__ = __mul__

    def div_pos(self, c):
        """Divide by a positive float c (exact divisor)."""
        assert c > 0
        return Iv(dn(self.lo / c), up(self.hi / c))

    def recip(self):
        """1 / self for strictly positive intervals."""
        assert np.all(self.lo > 0)
        return Iv(dn(1.0 / self.hi), up(1.0 / self.lo))

    def sum(self, axis=-1):
        return Iv(sum_dn(self.lo, axis), sum_up(self.hi, axis))

    def min(self, axis=-1):
        return Iv(self.lo.min(axis), self.hi.min(axis))

    def max(self, axis=-1):
        return Iv(self.lo.max(axis), self.hi.max(axis))

    def dot(self, M):
        """Contract the last axis with the first axis of M (array or Iv): [..., i] x [i, ...] -> [..., ...]."""
        M = Iv.of(M)
        a = Iv(self.lo.reshape(self.shape + (1,) * (M.lo.ndim - 1)), self.hi.reshape(self.shape + (1,) * (M.lo.ndim - 1)))
        return (a * M).sum(axis=self.lo.ndim - 1)

    def width(self):
        return float(np.max(self.hi - self.lo))


_LN2 = Iv(float.fromhex('0x1.62e42fefa39efp-1'), np.nextafter(float.fromhex('0x1.62e42fefa39efp-1'), 1.0))
_EXP_TERMS = 20
_EXP_REM = 2e-26          # >= e^0.5 * 0.5^21 / 21!, the Taylor remainder for |r| <= 0.5
_EXP_CLIP = 700.0


def exp_pt(x):
    """Enclosure of exp(x) for a float array x: exp(x) = 2^k exp(r), r = x - k ln 2, Taylor series in r."""
    x = np.asarray(x, dtype=np.float64)
    if np.any(x > _EXP_CLIP) or np.any(np.isnan(x)):
        raise FloatingPointError('exp argument out of range')
    xc = np.maximum(x, -_EXP_CLIP)
    k = np.rint(xc / _LN2.lo)
    r = Iv.pt(xc) - Iv.pt(k) * _LN2
    assert np.all(np.maximum(np.abs(r.lo), np.abs(r.hi)) <= 0.5)
    p = Iv.pt(np.ones_like(xc))
    for j in range(_EXP_TERMS, 0, -1):
        p = (r * p).div_pos(float(j)) + 1.0
    p = Iv(dn(p.lo - _EXP_REM), up(p.hi + _EXP_REM))
    ki = k.astype(np.int64)
    lo, hi = np.ldexp(p.lo, ki), np.ldexp(p.hi, ki)      # exact: |k| <= 1010, results are normal
    return Iv(np.where(x < -_EXP_CLIP, 0.0, lo), hi)


def exp(a):
    """Enclosure of exp over an interval (exp is increasing)."""
    a = Iv.of(a)
    return Iv(exp_pt(a.lo).lo, exp_pt(a.hi).hi)


def exp_dn(x):
    return exp_pt(x).lo


def exp_up(x):
    return exp_pt(x).hi


def sqrt_recip(c):
    """Enclosure of 1 / sqrt(c) for a positive integer c."""
    s = float(np.sqrt(float(c)))
    S = Iv.pt(s) if s == int(s) and int(s) ** 2 == c else Iv(dn(s), up(s))   # IEEE sqrt is correctly rounded
    return S.recip()


def softmax_box(S):
    """S: Iv [..., M] enclosing the scores of one attention row. Returns float arrays (l, u), each [..., M],
    with l <= softmax(s) <= u elementwise for every real score vector s inside S."""
    M = S.shape[-1]
    D = Iv(S.lo[..., None, :], S.hi[..., None, :]) - Iv(S.lo[..., :, None], S.hi[..., :, None])   # [..., m, m2] = S[m2] - S[m]
    E = exp(D)
    eye = np.eye(M, dtype=bool)
    E = Iv(np.where(eye, 0.0, E.lo), np.where(eye, 0.0, E.hi))
    w = (E.sum(-1) + 1.0).recip()
    return np.maximum(w.lo, 0.0), np.minimum(w.hi, 1.0)


def lp_min_lo(V, wl, wu):
    """Float array <= sum_m v_m w_m for every real v >= V and every w with wl <= w <= wu, sum(w) = 1.

    V: [..., M] floats; wl, wu: broadcastable to V, wl >= 0. Uses weak duality: for any lam,
    sum v_m w_m = lam + sum (v_m - lam) w_m >= lam + sum min((V_m - lam) wl_m, (V_m - lam) wu_m).
    lam is taken from the greedy primal solution (the marginal item), where the bound is tight in exact
    arithmetic. Soundness does not depend on how lam was chosen.
    """
    wl = np.broadcast_to(wl, V.shape); wu = np.broadcast_to(wu, V.shape)
    assert np.all(wl >= 0) and np.all(wu >= wl)
    order = np.argsort(V, -1)
    Vs = np.take_along_axis(V, order, -1)
    cap = np.cumsum(np.take_along_axis(wu - wl, order, -1), -1)
    R = 1 - wl.sum(-1, keepdims=True)
    j = np.minimum((cap < R).sum(-1, keepdims=True), V.shape[-1] - 1)
    lam = np.take_along_axis(Vs, j, -1)                   # [..., 1]
    d = dn(V - lam)
    t = dn(np.minimum(d * wl, d * wu))
    return dn(sum_dn(t, -1) + lam[..., 0])


def lp_max_hi(V, wl, wu):
    return -lp_min_lo(-V, wl, wu)


def ratio_min_lo(S, V, mask, guess):
    """Float array [...] <= min over token choices of the softmax-weighted mean  sum_m e^{s_m} v_m / sum_m e^{s_m},
    where position m picks one allowed token t (mask[..., m, t]) and gets score s_m in S[..., m, t], value v_m >= V[..., m, t].

    S: Iv, V: float array, mask: bool array, all broadcastable to [..., M, Tn]; guess: float array [...].
    The mean is >= gam whenever sum_m min_t w_m(t) (V_m(t) - gam) >= 0 for weights w = e^{s - c} > 0. This is
    checked in interval arithmetic at gam slightly below `guess` (an uncertified estimate of the minimum, e.g.
    from proof_v4.ratio_min). If no such gam passes, the smallest allowed value is returned, which is always valid.
    """
    shape = np.broadcast_shapes(S.shape, V.shape, mask.shape)
    V = np.broadcast_to(V, shape); mask = np.broadcast_to(mask, shape)
    w = exp(S - S.hi.max())
    w = Iv(np.broadcast_to(w.lo, shape), np.broadcast_to(w.hi, shape))
    out = np.where(mask, V, np.inf).min((-1, -2))
    done = np.zeros(out.shape, bool)
    for delta in (1e-11, 1e-9, 1e-7, 1e-5, 1e-3):
        gam = guess - delta * (1 + np.abs(guess))
        d = dn(V - gam[..., None, None])
        term = np.where(mask, dn(np.minimum(d * w.lo, d * w.hi)), np.inf).min(-1)
        ok = (sum_dn(term, -1) >= 0) & ~done
        out = np.where(ok, np.maximum(out, gam), out); done |= ok
        if done.all(): break
    return out


def ratio_max_hi(S, V, mask, guess):
    return -ratio_min_lo(S, -V, mask, -guess)
