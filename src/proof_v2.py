"""v2: every attention row relaxed to {w in simplex : l <= w <= u} with per-position bounds
derived from score intervals; linear functionals bounded by exact greedy LP over that polytope."""
import numpy as np, pickle, sys, time
from proof_v1 import build

def lp_min(V, l, u):
    """min sum_m w_m V[...,m] s.t. l<=w<=u, sum w = 1 (greedy fractional knapsack)."""
    order = np.argsort(V, -1)
    Vs = np.take_along_axis(V, order, -1)
    cap = np.broadcast_to(u - l, V.shape); cap = np.take_along_axis(cap, order, -1)
    R = 1 - l.sum(-1)[..., None]
    prev = np.concatenate([np.zeros(cap.shape[:-1] + (1,)), np.cumsum(cap, -1)[..., :-1]], -1)
    alloc = np.clip(R - prev, 0, cap)
    return (V * l).sum(-1) + (Vs * alloc).sum(-1)

def lp_max(V, l, u):
    return -lp_min(-V, l, u)

class Group:
    def __init__(s, T, s1, q, i, a):
        s.T, s.q, s.i, s.a = T, q, i, a
        d, n = T['d'], T['n']; s.f = n - 1
        D = np.array([t for t in range(d) if t not in (q, a)])
        s.toks = [np.array([q]) if m == i else np.array([a]) if m == i + 1 else D for m in range(n - 1)] + [np.array([q])]
        s.lu = {}
        for r in range(1, n):
            ls, us = [], []
            for tq in s.toks[r]:
                smin = np.array([s1[r, tq, m, s.toks[m]].min() if m != r else s1[r, tq, r, tq] for m in range(r + 1)])
                smax = np.array([s1[r, tq, m, s.toks[m]].max() if m != r else s1[r, tq, r, tq] for m in range(r + 1)])
                u = np.array([1 / (1 + sum(np.exp(smin[m2] - smax[m]) for m2 in range(r + 1) if m2 != m)) for m in range(r + 1)])
                l = np.array([1 / (1 + sum(np.exp(smax[m2] - smin[m]) for m2 in range(r + 1) if m2 != m)) for m in range(r + 1)])
                ls.append(l); us.append(u)
            s.lu[r] = (np.min(ls, 0), np.max(us, 0))
        s.lu[0] = (np.ones(1), np.ones(1))

    def pos_vals(s, L, r, red):
        return np.stack([red(L[..., m, s.toks[m]], -1) for m in range(r + 1)], -1)

    def h_lo(s, L, r):
        return lp_min(s.pos_vals(L, r, np.min), *s.lu[r])

    def h_hi(s, L, r):
        return lp_max(s.pos_vals(L, r, np.max), *s.lu[r])

def certify(T, s1, q, i, a, debug=False):
    g = Group(T, s1, q, i, a); f = g.f; n = T['n']; d = T['d']
    SK, SH = T['SK'][q], T['SH'][q]                       # [m, t, m2, t2]
    corr = SK[:, :, i + 1, a] + g.h_lo(SH, i + 1)         # [m, t] (h_f atom)
    gaps = {}
    for k in range(n):
        if k == i + 1: continue
        if k == f:
            qv = T['QV'][q]                               # [m, t, dm]
            keys = T['x'][f, q][None, None] + T['ox']     # [m', t', dm]
            Qmat = np.einsum('mte,nse->mtns', qv, keys)   # [m,t,m',t']
            R = g.h_hi(Qmat, f)                           # [m,t]
            gm = g.pos_vals(corr - R, f, np.min)
        else:
            vk = SK[:, :, k, g.toks[k]].max(-1) + g.h_hi(SH, k)
            gm = g.pos_vals(corr - vk, f, np.min)
        gaps[k] = lp_min(gm, *g.lu[f])
    beta = 1 / (1 + sum(np.exp(-x) for x in gaps.values()))
    bs = np.array([t for t in range(d) if t != a])
    dlt = lambda M: np.moveaxis(M[..., a][..., None] - M[..., bs], -1, 0)   # [|bs|, ...]
    t1 = dlt(T['xU'][f, q])
    t2 = g.h_lo(dlt(T['oxU']), f)
    def c_lb(k):
        if k == f:
            return dlt(T['xW'][f, q]) + g.h_lo(dlt(T['oxW']), f)
        return dlt(T['xW'][k][g.toks[k]]).min(1) + g.h_lo(dlt(T['oxW']), k)
    c_corr = c_lb(i + 1)
    c_oth = np.min([c_lb(k) for k in range(n) if k != i + 1], 0)
    margin = t1 + t2 + np.minimum(c_corr, c_oth + beta * (c_corr - c_oth))
    if debug: return dict(gaps=gaps, beta=beta, margin=margin.min())
    return margin.min() > 0, margin.min(), beta

def run(path):
    p = pickle.load(open(path, 'rb'))
    t0 = time.time(); T = build(p)
    x = T['x']; dh = p['Q1'].shape[1]
    s1 = np.einsum('kah,mbh->kamb', x @ p['Q1'], x @ p['K1']) / np.sqrt(dh)
    d, n = T['d'], T['n']; ok = tot = 0; M = []; Bt = []
    for q in range(d):
        for a in range(d):
            if a == q: continue
            for i in range(n - 2):
                c, mg, b = certify(T, s1, q, i, a); ok += c; tot += 1; M.append(mg); Bt.append(b)
    M, Bt = np.array(M), np.array(Bt)
    print(f'certified {ok}/{tot} groups -> accuracy lower bound {ok/tot:.4f} ({time.time()-t0:.1f}s)')
    print(f'margin lb: min {M.min():.2f} median {np.median(M):.2f}; beta lb: min {Bt.min():.3f} median {np.median(Bt):.3f}')
    return T, s1

if __name__ == '__main__':
    run(sys.argv[1])
