"""Compact proof: certify groups (q, i, a) = (query token, match position, answer token).

Within a group, the 3 distractor tokens are free (relaxed to range independently over V\\{q,a}).
Attention rows are relaxed to polytopes:
  * layer-1 row at key position k in 1..n-2: A1[k,k-1] >= alpha (lower bound, pessimized), rest arbitrary
  * layer-1 row at final position f: arbitrary distribution (it is diffuse in practice)
  * layer-2 row at f: A2[i+1] >= beta (derived bound), rest arbitrary
Every bilinear/affine quantity is bounded at polytope vertices, which is exact for the relaxation.
"""
import numpy as np, pickle, sys, time

def build(p):
    E, P = p['E'], p['P']
    d, dm = E.shape; n = P.shape[0]; dh = p['Q1'].shape[1]
    x = E[None, :, :] + P[:, None, :]                     # [m, t, dm] residual atoms
    OV1 = p['V1'] @ p['O1']
    ox = x @ OV1                                          # [m, t, dm] layer-1 OV images
    # layer-1 scores s1[k, tk, m, t]
    s1 = np.einsum('kah,mbh->kamb', x @ p['Q1'], x @ p['K1']) / np.sqrt(dh)
    # alpha[k, tk, tp]: lower bound on A1[k, k-1] given tokens at k and k-1, others arbitrary
    alpha = np.ones((n, d, d))
    for k in range(1, n - 1):
        for tk in range(d):
            base = s1[k, tk, k - 1, :]                    # [tp]
            other = np.exp(s1[k, tk, k, tk] - base)
            for m in range(k - 1):
                other = other + np.exp(s1[k, tk, m, :].max() - base)
            alpha[k, tk, :] = 1 / (1 + other)
    B2 = p['Q2'] @ p['K2'].T / np.sqrt(dh)
    f = n - 1
    # query vectors given h_f vertex c=(m,t): QV[q, m, t, :] = (x[f,q] + ox[m,t]) B2
    QV = np.einsum('qmtd,de->qmte', x[f][:, None, None, :] + ox[None], B2)
    SK = np.einsum('qmte,ase->qmtas', QV, x)              # score vs key r0 atom
    SH = np.einsum('qmte,ase->qmtas', QV, ox)             # score vs key layer-1 atom
    W = p['V2'] @ p['O2'] @ p['U']                        # layer-2 OV -> logits
    T = dict(x=x, ox=ox, alpha=alpha, QV=QV, SK=SK, SH=SH,
             xW=x @ W, oxW=ox @ W, xU=x @ p['U'], oxU=ox @ p['U'], d=d, n=n)
    return T


def h_bounds(L, k, toks, alpha_k):
    """Bounds on L(h_k) where L is [m, t] table of a linear functional on layer-1 atoms.
    Returns (lo, hi) arrays broadcasting over any leading dims of L (L shape [..., n, d])."""
    if k == 0:
        v = L[..., 0, toks[0]]
        return v.min(-1), v.max(-1)
    prev = L[..., k - 1, toks[k - 1]]                     # [..., |T_{k-1}|]
    oth = [L[..., m, toks[m]] for m in range(k + 1) if m != k - 1]
    oth = np.concatenate(oth, -1)
    omin, omax = oth.min(-1)[..., None], oth.max(-1)[..., None]
    lo = np.minimum(prev, alpha_k * prev + (1 - alpha_k) * omin).min(-1)
    hi = np.maximum(prev, alpha_k * prev + (1 - alpha_k) * omax).max(-1)
    return lo, hi


def certify(T, q, i, a, debug=False):
    d, n = T['d'], T['n']; f = n - 1
    D = np.array([t for t in range(d) if t != q and t != a])
    toks = [np.array([q]) if m == i else np.array([a]) if m == i + 1 else D for m in range(n - 1)] + [np.array([q])]
    al = lambda k: 1.0 if k == 0 else T['alpha'][k][np.ix_(toks[k], toks[k - 1])].min()
    # h_f vertices: all atoms (m, t in toks[m])
    verts = [(m, t) for m in range(n) for t in toks[m]]
    vm = np.array([v[0] for v in verts]); vt = np.array([v[1] for v in verts])
    SK = T['SK'][q][vm, vt]                               # [V, m, t]
    SH = T['SH'][q][vm, vt]                               # [V, m, t]
    # correct key i+1
    lo_h, _ = h_bounds(SH, i + 1, toks, al(i + 1))
    corr = SK[:, i + 1, a] + lo_h                         # [V]
    gaps = {}
    for k in range(n):
        if k == i + 1:
            continue
        if k == f:
            # s2(f) <= max_b QV[c] . (x_f + ox_b)
            qv = T['QV'][q][vm, vt]                       # [V, dm]
            keys = T['x'][f, q][None] + T['ox'][vm, vt]   # [V(b), dm]
            s_ff = qv @ keys.T                            # [V(c), V(b)]
            gaps[k] = (corr[:, None] - s_ff).min()
        else:
            _, hi_h = h_bounds(SH, k, toks, al(k))
            sk = SK[:, k, toks[k]].max(-1) + hi_h
            gaps[k] = (corr - sk).min()
    beta = 1 / (1 + sum(np.exp(-g) for g in gaps.values()))
    # logits: margin for answer a vs each wrong b
    bs = np.array([t for t in range(d) if t != a])
    dlt = lambda M: M[..., a][..., None] - M[..., bs]    # [..., |bs|]
    t1 = dlt(T['xU'][f, q])
    t2 = dlt(T['oxU'][vm, vt]).min(0)
    def c_lb(k):
        if k == f:
            return dlt(T['xW'][f, q]) + dlt(T['oxW'][vm, vt]).min(0)
        rx = dlt(T['xW'][k, toks[k]]).min(0)
        Lh = np.moveaxis(dlt(T['oxW']), -1, 0)            # [|bs|, m, t]
        lo, _ = h_bounds(Lh, k, toks, al(k))
        return rx + lo
    c_corr = c_lb(i + 1)
    c_oth = np.min([c_lb(k) for k in range(n) if k != i + 1], 0)
    attn = np.minimum(c_corr, c_oth + beta * (c_corr - c_oth))
    margin = t1 + t2 + attn
    if debug:
        return dict(gaps=gaps, beta=beta, margin=margin.min())
    return margin.min() > 0, margin.min(), beta


if __name__ == '__main__':
    p = pickle.load(open(sys.argv[1], 'rb'))
    t0 = time.time()
    T = build(p)
    d, n = T['d'], T['n']
    ok = 0; tot = 0; margins = []; betas = []
    for q in range(d):
        for a in range(d):
            if a == q: continue
            for i in range(n - 2):
                c, mg, b = certify(T, q, i, a)
                ok += c; tot += 1; margins.append(mg); betas.append(b)
    margins = np.array(margins); betas = np.array(betas)
    print(f'certified {ok}/{tot} groups -> accuracy lower bound {ok/tot:.4f}  ({time.time()-t0:.1f}s)')
    print(f'margin lb: min {margins.min():.2f} median {np.median(margins):.2f}; beta lb: min {betas.min():.3f} median {np.median(betas):.3f}')
    print(f"alpha lb on key positions: {[round(float(T['alpha'][k].min()),3) for k in range(1,n-1)]}")
