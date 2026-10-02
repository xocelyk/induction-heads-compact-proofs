"""v4: v3 with a tighter treatment of layer-1 attention rows.

v2/v3 bound a quantity of the form  sum_m A[m] V_m  (A = one layer-1 attention row) by an LP in which the
attention weights range over a box, independently of the tokens that determine V_m. But the weight on position m
is softmax of a score that depends only on the token at m (the row's own token being fixed), so

    sum_m A[m] V_m  =  sum_m e^{s_m(t_m)} V_m(t_m) / sum_m e^{s_m(t_m)} .

When V_m depends only on t_m as well, the minimum of this ratio over independent token choices per position can
be computed exactly (`ratio_min`, Dinkelbach's iteration for fractional programs). This ties each attention
weight to the token that produces it. It is used for
  * the final-position row f, in every layer-2 score gap;
  * rows i+1 and f and the key rows in the terms h_lo / h_hi (correct-key score, logit contributions).
The key row inside each distractor gap keeps v3's per-(u, v) box. Distinctness of tokens is used where it is
free: a token fixed by the enumeration is excluded from the other distractor positions.
"""
import numpy as np, pickle, sys, time
from proof_v1 import build
from proof_v2 import Group, lp_min, lp_max
from proof_v3 import row_box


def ratio_min(S, V, mask, iters=100):
    """Exact min over one allowed token per position of sum_m e^{s_m} v_m / sum_m e^{s_m}.
    S, V, mask broadcastable to [..., M, Tn]; mask[..., m, t] says token t is allowed at position m."""
    S, V, mask = np.broadcast_arrays(S, V, mask)
    Sm = np.where(mask, S, -np.inf)
    w = np.exp(Sm - Sm.max((-1, -2), keepdims=True))
    Vz = np.where(mask, V, 0.0)
    pick = lambda t: (np.take_along_axis(w, t, -1)[..., 0], np.take_along_axis(Vz, t, -1)[..., 0])
    ws, vs = pick(np.where(mask, V, np.inf).argmin(-1)[..., None])
    g = (ws * vs).sum(-1) / ws.sum(-1)
    for _ in range(iters):
        # best response to the current ratio g: each position minimises w (v - g)
        ws, vs = pick(np.where(mask, w * (Vz - g[..., None, None]), np.inf).argmin(-1)[..., None])
        g2 = (ws * vs).sum(-1) / ws.sum(-1)
        done = np.all(g2 >= g - 1e-13); g = np.minimum(g, g2)
        if done: break
    return g


def ratio_max(S, V, mask):
    return -ratio_min(S, -V, mask)


class Rows:
    """Scores and allowed-token masks of the layer-1 attention rows for the group (q, i, a)."""
    def __init__(s, T, s1, q, i, a):
        d, n = T['d'], T['n']; s.s1, s.d, s.n = s1, d, n
        D = np.array([t for t in range(d) if t not in (q, a)])
        s.toks = [np.array([q]) if m == i else np.array([a]) if m == i + 1 else D for m in range(n - 1)] + [np.array([q])]
        s.base = np.zeros((n, d), bool)
        for m in range(n): s.base[m, s.toks[m]] = True
        s.free = np.array([m not in (i, i + 1, n - 1) for m in range(n)])   # positions holding a distractor

    def row(s, r, tq):
        """Scores [r+1, d] and mask [r+1, d] for row r whose own token is tq."""
        mask = s.base[:r + 1].copy()
        mask[r] = False; mask[r, tq] = True
        if s.free[r]: mask[:r][s.free[:r], tq] = False    # distinct tokens
        return s.s1[r, tq, :r + 1, :], mask

    def fixed(s, r, fix, shape):
        """Mask [*shape, r+1, d] with position m forced to token grid fix[m] (arrays of `shape`) and those
        tokens excluded from the other distractor positions."""
        td = np.arange(s.d)
        mask = np.broadcast_to(s.base[:r + 1], shape + (r + 1, s.d)).copy()
        for m, tok in fix.items():
            mask[..., m, :] = td == tok[..., None]
            for m2 in range(r + 1):
                if s.free[m2] and m2 not in fix: mask[..., m2, :] &= td != tok[..., None]
        return mask

    def h(s, L, r, lo):
        L = L[..., :r + 1, :]
        out = [(ratio_min if lo else ratio_max)(S, L, mask) for S, mask in (s.row(r, tq) for tq in s.toks[r])]
        return np.min(out, 0) if lo else np.max(out, 0)

    def h_lo(s, L, r):
        """Lower bound on sum_m A1[r, m] L[..., m, t_m] over the group."""
        return s.h(L, r, True)

    def h_hi(s, L, r):
        return s.h(L, r, False)


def certify(T, s1, q, i, a, debug=False):
    g = Rows(T, s1, q, i, a); n = T['n']; d = T['d']; f = n - 1
    box = Group(T, s1, q, i, a)                           # v2 boxes, used by row_box
    SK, SH = T['SK'][q], T['SH'][q]                       # [m, t, m2, t2]
    corr = SK[:, :, i + 1, a] + g.h_lo(SH, i + 1)         # [m, t]
    Sf, Mf = g.row(f, q)                                  # [n, d]
    gaps = {}
    for k in range(n):
        if k == i + 1: continue
        if k == f:
            Qc = T.setdefault('Qcache', {})
            if q not in Qc:
                Qc[q] = np.einsum('mte,nse->mtns', T['QV'][q], T['x'][f, q][None, None] + T['ox'])
            Qmat = Qc[q]                                  # [m, t, m', t']
            # for query atom (m, t): max of sum_m' A1[f, m'] Qmat[m, t, m', t'], atom m' = m sharing the token t
            mask = np.broadcast_to(Mf, (n, d, n, d)).copy()
            ar = np.arange(n)
            mask[ar, :, ar, :] = np.eye(d, dtype=bool)[None] & Mf[:, :, None]
            mask[~Mf] = Mf                                # atoms (m, t) outside the group: any valid mask, never used
            R = ratio_max(Sf, Qmat, mask)                 # [m, t]
            gaps[k] = ratio_min(Sf, corr - R, Mf); continue
        U = np.array([-1]) if k == 0 else g.toks[k - 1]; V = g.toks[k]
        # enumeration grid over (u = token at k-1, v = token at k): trailing dims [|U|, |V|]
        Vg = np.broadcast_to(V[None, :], (len(U), len(V))); Ug = np.broadcast_to(U[:, None], (len(U), len(V)))
        cols = []
        for m2 in range(k + 1):
            if m2 == k: cols.append(SH[:, :, k, :][..., Vg])
            elif m2 == k - 1: cols.append(SH[:, :, k - 1, :][..., Ug])
            else: cols.append(np.broadcast_to(SH[:, :, m2, g.toks[m2]].max(-1)[..., None, None], SH.shape[:2] + Vg.shape))
        lk, uk = row_box(s1, box, k, U, V)                # [|U|, |V|, k+1]
        vk = SK[:, :, k, :][..., Vg] + lp_max(np.stack(cols, -1), lk, uk)
        diff = corr[..., None, None] - vk                 # [m, t, |U|, |V|]
        mask = g.fixed(f, {k: Vg, k - 1: Ug} if k >= 1 else {k: Vg}, Vg.shape)
        val = ratio_min(Sf, np.moveaxis(diff, (0, 1), (2, 3)), mask)   # [|U|, |V|]
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
