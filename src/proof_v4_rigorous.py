"""The v4 bound (proof_v4.py) recomputed with outward-rounded interval arithmetic (interval.py).

See proof_v3_rigorous.py for what is certified and for the naming convention (_lo / _hi). The exact
weighted-mean minimum of v4 is certified by interval.ratio_min_lo: the float Dinkelbach iteration only supplies
a candidate value, and a value slightly below it is then verified with a sign check in interval arithmetic.
"""
import numpy as np, pickle, sys, time
from interval import dn, up, exp_dn, exp_up, lp_min_lo, lp_max_hi, sum_up, ratio_min_lo, ratio_max_hi
from proof_v3_rigorous import build, Group, row_box
from proof_v4 import Rows as FloatRows, ratio_min


def rmin_lo(S, V, mask):
    return ratio_min_lo(S, V, mask, ratio_min(S.lo, V, mask))


def rmax_hi(S, V, mask):
    return ratio_max_hi(S, V, mask, -ratio_min(S.lo, -V, mask))


class Rows(FloatRows):
    """As proof_v4.Rows, with s1 an Iv; h_lo / h_hi take and return one-sided float bounds."""
    def h(s, L, r, lo):
        L = L[..., :r + 1, :]
        out = [(rmin_lo if lo else rmax_hi)(S, L, mask) for S, mask in (s.row(r, tq) for tq in s.toks[r])]
        return np.min(out, 0) if lo else np.max(out, 0)


def certify(T, s1, q, i, a, debug=False):
    g = Rows(T, s1, q, i, a); n = T['n']; d = T['d']; f = n - 1
    box = Group(T, s1, q, i, a)
    SK, SH = T['SK'][q], T['SH'][q]                       # Iv [m, t, m2, t2]
    corr_lo = dn(SK.lo[:, :, i + 1, a] + g.h_lo(SH.lo, i + 1))   # [m, t]
    Sf, Mf = g.row(f, q)                                  # Iv [n, d], mask [n, d]
    gaps = {}
    for k in range(n):
        if k == i + 1: continue
        if k == f:
            Q_hi = up(SK.hi[:, :, f, q][..., None, None] + SH.hi)   # [m, t, m', t']
            mask = np.broadcast_to(Mf, (n, d, n, d)).copy()
            ar = np.arange(n)
            mask[ar, :, ar, :] = np.eye(d, dtype=bool)[None] & Mf[:, :, None]
            mask[~Mf] = Mf
            R_hi = rmax_hi(Sf, Q_hi, mask)                # [m, t]
            gaps[k] = rmin_lo(Sf, dn(corr_lo - R_hi), Mf); continue
        U = np.array([-1]) if k == 0 else g.toks[k - 1]; V = g.toks[k]
        Vg = np.broadcast_to(V[None, :], (len(U), len(V))); Ug = np.broadcast_to(U[:, None], (len(U), len(V)))
        cols = []
        for m2 in range(k + 1):
            if m2 == k: cols.append(SH.hi[:, :, k, :][..., Vg])
            elif m2 == k - 1: cols.append(SH.hi[:, :, k - 1, :][..., Ug])
            else: cols.append(np.broadcast_to(SH.hi[:, :, m2, g.toks[m2]].max(-1)[..., None, None], SH.shape[:2] + Vg.shape))
        lk, uk = row_box(s1, box, k, U, V)                # [|U|, |V|, k+1]
        vk_hi = up(SK.hi[:, :, k, :][..., Vg] + lp_max_hi(np.stack(cols, -1), lk, uk))
        diff_lo = dn(corr_lo[..., None, None] - vk_hi)    # [m, t, |U|, |V|]
        mask = g.fixed(f, {k: Vg, k - 1: Ug} if k >= 1 else {k: Vg}, Vg.shape)
        val = rmin_lo(Sf, np.moveaxis(diff_lo, (0, 1), (2, 3)), mask)   # [|U|, |V|]
        if k >= 1:
            val = np.where(Ug == Vg, np.inf, val)         # tokens distinct
        gaps[k] = val.min()
    keys = [k for k in range(n) if k != i + 1]
    G = np.array([gaps[k] for k in keys])
    u_oth = np.minimum(up(1 / dn(1 + exp_dn(G))), 1.0)
    l_c = float(dn(1 / up(1 + sum_up(exp_up(-G)))))
    l = np.concatenate([[l_c], np.zeros(len(keys))]); u = np.concatenate([[1.0], u_oth])
    bs = np.array([t for t in range(d) if t != a])
    dlt_lo = lambda M: np.moveaxis((M[..., a][..., None] - M[..., bs]).lo, -1, 0)   # [|bs|, ...]
    t1 = dlt_lo(T['xU'][f, q]); t2 = g.h_lo(dlt_lo(T['oxU']), f)
    def c_lb(k):
        if k == f: return dn(dlt_lo(T['xW'][f, q]) + g.h_lo(dlt_lo(T['oxW']), f))
        return dn(dlt_lo(T['xW'][k][g.toks[k]]).min(1) + g.h_lo(dlt_lo(T['oxW']), k))
    C = np.stack([c_lb(i + 1)] + [c_lb(k) for k in keys], -1)   # [|bs|, n]
    margin = dn(dn(t1 + t2) + lp_min_lo(C, l, u))
    if debug: return dict(gaps=gaps, beta=l_c, margin=margin.min())
    return margin.min() > 0, margin.min(), l_c


def run(path, verbose=True):
    p = pickle.load(open(path, 'rb'))
    t0 = time.time(); T = build(p); s1 = T['s1']
    d, n = T['d'], T['n']; ok = tot = 0; M = []; Bt = []
    for q in range(d):
        for a in range(d):
            if a == q: continue
            for i in range(n - 2):
                c, mg, b = certify(T, s1, q, i, a); ok += c; tot += 1; M.append(mg); Bt.append(b)
    M, Bt = np.array(M), np.array(Bt)
    if verbose:
        print(f'rigorous: certified {ok}/{tot} groups -> accuracy lower bound {ok/tot:.4f} ({time.time()-t0:.1f}s)')
        print(f'margin lb: min {M.min():.2f} median {np.median(M):.2f}, smallest certified {M[M > 0].min():.4f}; '
              f'beta lb: min {Bt.min():.3f} median {np.median(Bt):.3f}')
    return ok / tot, T, s1


if __name__ == '__main__':
    run(sys.argv[1])
