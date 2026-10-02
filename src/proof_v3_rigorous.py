"""The v3 bound (proof_v3.py) recomputed with outward-rounded interval arithmetic (interval.py).

A group is certified only if a floating-point number that is provably <= the real-arithmetic logit margin
is positive. "Real-arithmetic" refers to the function defined by the stored float64 weights with exact
arithmetic and a hard causal mask. Same relaxations as proof_v3; the differences are

  * every table built from the weights is an interval enclosure of its exact value;
  * attention boxes are rounded outward (softmax_box);
  * each LP value is replaced by a dual bound, which is valid for any multiplier (lp_min_lo / lp_max_hi),
    so the result does not rely on the floating-point greedy solution being optimal.

Names ending in _lo / _hi are float arrays that are lower / upper bounds on the corresponding real quantity.
Interface matches proof_v3 (run, certify), so sound.py works with either module.
"""
import numpy as np, pickle, sys, time
from interval import Iv, dn, up, exp_dn, exp_up, sqrt_recip, softmax_box, lp_min_lo, lp_max_hi, sum_up


def build(p):
    E, P = p['E'], p['P']
    d, dm = E.shape; n = P.shape[0]; dh = p['Q1'].shape[1]; f = n - 1
    sc = sqrt_recip(dh)
    x = Iv.pt(E[None, :, :]) + Iv.pt(P[:, None, :])       # [m, t, dm] residual atoms
    ox = x.dot(p['V1']).dot(p['O1'])                      # [m, t, dm] layer-1 OV images
    # layer-1 scores s1[k, tk, m, t]
    xq, xk = x.dot(p['Q1']), x.dot(p['K1'])
    s1 = (Iv(xq.lo[:, :, None, None, :], xq.hi[:, :, None, None, :]) * xk).sum(-1) * sc
    # layer-2 query vectors for each h_f atom c = (m, t): qv[q, m, t, :] = (x[f, q] + ox[m, t]) Q2
    xf = x[f]
    qv = (Iv(xf.lo[:, None, None, :], xf.hi[:, None, None, :]) + ox).dot(p['Q2'])   # [q, m, t, dh]
    qv = Iv(qv.lo[:, :, :, None, None, :], qv.hi[:, :, :, None, None, :])
    SK = (qv * x.dot(p['K2'])).sum(-1) * sc               # [q, m, t, m2, t2] score vs key r0 atom
    SH = (qv * ox.dot(p['K2'])).sum(-1) * sc              # [q, m, t, m2, t2] score vs key layer-1 atom
    xv, oxv = x.dot(p['V2']).dot(p['O2']), ox.dot(p['V2']).dot(p['O2'])
    return dict(SK=SK, SH=SH, xW=xv.dot(p['U']), oxW=oxv.dot(p['U']), xU=x.dot(p['U']), oxU=ox.dot(p['U']),
                s1=s1, d=d, n=n)


def pos_vals(L, toks, r, red):
    """L: float [..., m, t]. Reduce over the allowed tokens at each position m <= r -> [..., r+1]."""
    return np.stack([red(L[..., m, toks[m]], -1) for m in range(r + 1)], -1)


class Group:
    def __init__(s, T, s1, q, i, a):
        d, n = T['d'], T['n']; s.f = n - 1
        D = np.array([t for t in range(d) if t not in (q, a)])
        s.toks = [np.array([q]) if m == i else np.array([a]) if m == i + 1 else D for m in range(n - 1)] + [np.array([q])]
        s.lu = {0: (np.ones(1), np.ones(1))}
        for r in range(1, n):
            tq = s.toks[r]
            lo = [s1.lo[r, tq][:, m, s.toks[m]].min(-1) if m != r else s1.lo[r, tq, r, tq] for m in range(r + 1)]
            hi = [s1.hi[r, tq][:, m, s.toks[m]].max(-1) if m != r else s1.hi[r, tq, r, tq] for m in range(r + 1)]
            l, u = softmax_box(Iv(np.stack(lo, -1), np.stack(hi, -1)))   # [|tq|, r+1]
            s.lu[r] = (l.min(0), u.max(0))

    def h_lo(s, L, r):
        return lp_min_lo(pos_vals(L, s.toks, r, np.min), *s.lu[r])

    def h_hi(s, L, r):
        return lp_max_hi(pos_vals(L, s.toks, r, np.max), *s.lu[r])


def row_box(s1, g, k, U, V):
    """Attention box for layer-1 row k with token v at k and u at k-1, other positions relaxed. [|U|, |V|, k+1]."""
    if k == 0:
        o = np.ones((len(U), len(V), 1)); return o, o
    lo, hi = [], []
    for m in range(k + 1):
        if m == k:
            sl, sh = (z[k, V, k, V][None, :] for z in (s1.lo, s1.hi))
        elif m == k - 1:
            sl, sh = (z[k][np.ix_(V, [k - 1], U)][:, 0, :].T for z in (s1.lo, s1.hi))
        else:
            sl = s1.lo[k][V][:, m, g.toks[m]].min(1)[None, :]; sh = s1.hi[k][V][:, m, g.toks[m]].max(1)[None, :]
        lo.append(np.broadcast_to(sl, (len(U), len(V)))); hi.append(np.broadcast_to(sh, (len(U), len(V))))
    return softmax_box(Iv(np.stack(lo, -1), np.stack(hi, -1)))


def certify(T, s1, q, i, a, debug=False):
    g = Group(T, s1, q, i, a); f = g.f; n = T['n']; d = T['d']
    SK, SH = T['SK'][q], T['SH'][q]                       # Iv [m, t, m2, t2]
    corr_lo = dn(SK.lo[:, :, i + 1, a] + g.h_lo(SH.lo, i + 1))   # [m, t]: score of the correct key, per h_f atom
    gaps = {}
    for k in range(n):
        if k == i + 1: continue
        if k == f:
            Q_hi = up(SK.hi[:, :, f, q][..., None, None] + SH.hi)   # [m, t, m', t']
            vals = pos_vals(Q_hi, g.toks, f, np.max)                # [m, t, m']
            ar_n, ar_d = np.arange(n)[:, None], np.arange(d)[None, :]
            vals[ar_n, ar_d, ar_n] = Q_hi[ar_n, ar_d, ar_n, ar_d]   # same atom on both sides: token is shared
            R_hi = lp_max_hi(vals, *g.lu[f])
            gaps[k] = lp_min_lo(pos_vals(dn(corr_lo - R_hi), g.toks, f, np.min), *g.lu[f]); continue
        U = np.array([-1]) if k == 0 else g.toks[k - 1]; V = g.toks[k]
        # enumeration grid over (u = token at k-1, v = token at k): trailing dims [|U|, |V|]
        Vg = np.broadcast_to(V[None, :], (len(U), len(V))); Ug = np.broadcast_to(U[:, None], (len(U), len(V)))
        sk_hi = SK.hi[:, :, k, :][..., Vg]                # [m, t, |U|, |V|]
        cols = []
        for m2 in range(k + 1):
            if m2 == k:
                cols.append(SH.hi[:, :, k, :][..., Vg])
            elif m2 == k - 1:
                cols.append(SH.hi[:, :, k - 1, :][..., Ug])
            else:
                cols.append(np.broadcast_to(SH.hi[:, :, m2, g.toks[m2]].max(-1)[..., None, None], sk_hi.shape))
        lk, uk = row_box(s1, g, k, U, V)                  # [|U|, |V|, k+1]
        vk_hi = up(sk_hi + lp_max_hi(np.stack(cols, -1), lk, uk))
        diff_lo = dn(corr_lo[..., None, None] - vk_hi)    # [m, t, |U|, |V|]
        iu, iv = np.arange(len(U))[:, None], np.arange(len(V))[None, :]
        gcols = []
        for m in range(n):
            if m == k:
                gcols.append(diff_lo[m][Vg, iu, iv])
            elif m == k - 1:
                gcols.append(diff_lo[m][Ug, iu, iv])
            else:
                gcols.append(diff_lo[m][g.toks[m]].min(0))
        val = lp_min_lo(np.stack(gcols, -1), *g.lu[f])    # [|U|, |V|]
        if k >= 1:
            val = np.where(Ug == Vg, np.inf, val)         # tokens distinct
        gaps[k] = val.min()
    keys = [k for k in range(n) if k != i + 1]
    G = np.array([gaps[k] for k in keys])
    # layer-2 attention polytope over [i+1] + keys
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
