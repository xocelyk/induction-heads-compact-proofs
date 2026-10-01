"""v3: v2 + (a) for each distractor key k, enumerate the tokens at positions (k-1, k) jointly and keep
them consistent across key residual, key's layer-1 atoms and the query's layer-1 atoms;
(b) layer-2 attention relaxed to a box polytope from per-key gaps, logits bounded by LP."""
import numpy as np, pickle, sys, time
from proof_v1 import build
from proof_v2 import Group, lp_min, lp_max

def pos_vals_fixed(g, L, r, red, fix):
    """like Group.pos_vals but positions in `fix` (dict m -> index array broadcast over trailing enum dims)."""
    cols = []
    for m in range(r + 1):
        if m in fix:
            cols.append(fix[m](L[..., m, :]))
        else:
            cols.append(red(L[..., m, g.toks[m]], -1))
    cols = np.broadcast_arrays(*cols)
    return np.stack(cols, -1)

def row_box(s1, g, k, U, V):
    """attention box for row k with token v at k (row query) and u at k-1, others relaxed."""
    if k == 0:
        o = np.ones((len(U), len(V), 1)); return o, o
    smin, smax = [], []
    for m in range(k + 1):
        if m == k:
            sv = s1[k, V, k, V][None, :] * np.ones((len(U), 1)); smin.append(sv); smax.append(sv)
        elif m == k - 1:
            sv = s1[k][np.ix_(V, [k - 1], U)][:, 0, :].T; smin.append(sv); smax.append(sv)
        else:
            blk = s1[k][V][:, m, g.toks[m]]               # [|V|, |T_m|]
            smin.append(blk.min(1)[None, :] * np.ones((len(U), 1))); smax.append(blk.max(1)[None, :] * np.ones((len(U), 1)))
    smin = np.stack(smin, -1); smax = np.stack(smax, -1)   # [|U|, |V|, k+1]
    M = k + 1; l = np.empty_like(smin); u = np.empty_like(smin)
    for m in range(M):
        o = [m2 for m2 in range(M) if m2 != m]
        u[..., m] = 1 / (1 + np.exp(smin[..., o] - smax[..., m:m + 1]).sum(-1))
        l[..., m] = 1 / (1 + np.exp(smax[..., o] - smin[..., m:m + 1]).sum(-1))
    return l, u

def certify(T, s1, q, i, a, debug=False):
    g = Group(T, s1, q, i, a); f = g.f; n = T['n']; d = T['d']
    SK, SH = T['SK'][q], T['SH'][q]                       # [m, t, m2, t2]
    corr = SK[:, :, i + 1, a] + g.h_lo(SH, i + 1)         # [m, t]
    gaps = {}
    for k in range(n):
        if k == i + 1: continue
        if k == f:
            Qc = T.setdefault('Qcache', {})
            if q not in Qc:
                Qc[q] = np.einsum('mte,nse->mtns', T['QV'][q], T['x'][f, q][None, None] + T['ox'])
            Qmat = Qc[q]
            vals = g.pos_vals(Qmat, f, np.max)            # [m, t, m']
            diag = Qmat[np.arange(n)[:, None], np.arange(d)[None, :], np.arange(n)[:, None], np.arange(d)[None, :]]
            vals[np.arange(n)[:, None], np.arange(d)[None, :], np.arange(n)[:, None]] = diag
            R = lp_max(vals, *g.lu[f])
            gaps[k] = lp_min(g.pos_vals(corr - R, f, np.min), *g.lu[f]); continue
        if k == 0:
            U = np.array([-1]); V = g.toks[0]
        else:
            U = g.toks[k - 1]; V = g.toks[k]
        # enumeration grid over (u = token at k-1, v = token at k): trailing dims [|U|, |V|]
        Vg = V[None, :]; Ug = U[:, None]
        # key score upper bound for each h_f atom (m,t) and each (u,v): SK part + h_k part
        sk = SK[:, :, k, :][..., Vg * np.ones_like(Ug)]  # [m, t, |U|, |V|]
        fixk = {k: (lambda Lm: Lm[..., Vg * np.ones_like(Ug)])}
        if k >= 1:
            fixk[k - 1] = lambda Lm: Lm[..., Ug * np.ones_like(Vg)]
        # values of SH[m,t, m2, t2] at positions of row k, with tokens fixed at k-1, k
        Lh = SH[:, :, None, None, :, :]                   # [m,t,1,1,m2,t2]
        cols = []
        for m2 in range(k + 1):
            if m2 == k:
                cols.append(SH[:, :, k, :][..., Vg * np.ones_like(Ug)])
            elif m2 == k - 1:
                cols.append(SH[:, :, k - 1, :][..., Ug * np.ones_like(Vg)])
            else:
                cols.append(np.broadcast_to(SH[:, :, m2, g.toks[m2]].max(-1)[..., None, None], sk.shape))
        lk, uk = row_box(s1, g, k, U, V)                  # [|U|, |V|, k+1]
        hk_hi = lp_max(np.stack(cols, -1), lk, uk)        # [m, t, |U|, |V|]
        vk = sk + hk_hi
        diff = corr[..., None, None] - vk                 # [m, t, |U|, |V|]
        gcols = []
        for m in range(n):
            if m == k:
                gcols.append(diff[m][Vg * np.ones_like(Ug), np.arange(len(U))[:, None], np.arange(len(V))[None, :]])
            elif m == k - 1 and k >= 1:
                gcols.append(diff[m][Ug * np.ones_like(Vg), np.arange(len(U))[:, None], np.arange(len(V))[None, :]])
            else:
                gcols.append(diff[m][g.toks[m]].min(0))
        gm = np.stack(gcols, -1)                          # [|U|, |V|, n]
        val = lp_min(gm, *g.lu[f])
        if k >= 1:
            val = np.where(Ug == Vg, np.inf, val)         # tokens distinct
        gaps[k] = val.min()
    keys = [k for k in range(n) if k != i + 1]
    G = np.array([gaps[k] for k in keys])
    # layer-2 attention polytope over [i+1] + keys
    u_oth = 1 / (1 + np.exp(G))
    l_c = 1 / (1 + np.exp(-G).sum())
    l = np.concatenate([[l_c], np.zeros(len(keys))]); u = np.concatenate([[1.0], u_oth])
    bs = np.array([t for t in range(d) if t != a])
    dlt = lambda M: np.moveaxis(M[..., a][..., None] - M[..., bs], -1, 0)
    t1 = dlt(T['xU'][f, q]); t2 = g.h_lo(dlt(T['oxU']), f)
    def c_lb(k):
        if k == f: return dlt(T['xW'][f, q]) + g.h_lo(dlt(T['oxW']), f)
        return dlt(T['xW'][k][g.toks[k]]).min(1) + g.h_lo(dlt(T['oxW']), k)
    C = np.stack([c_lb(i + 1)] + [c_lb(k) for k in keys], -1)   # [|bs|, n]
    margin = t1 + t2 + lp_min(C, l, u)
    if debug: return dict(gaps=gaps, beta=l_c, margin=margin.min())
    return margin.min() > 0, margin.min(), l_c

def run(path, verbose=True):
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
    if verbose:
        print(f'certified {ok}/{tot} groups -> accuracy lower bound {ok/tot:.4f} ({time.time()-t0:.1f}s)')
        print(f'margin lb: min {M.min():.2f} median {np.median(M):.2f}; beta lb: min {Bt.min():.3f} median {np.median(Bt):.3f}')
    return ok / tot, T, s1

if __name__ == '__main__':
    run(sys.argv[1])
