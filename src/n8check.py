"""For longer sequences: sampled accuracy, brute-force throughput, and per-group soundness on random groups."""
import numpy as np, pickle, sys, time, itertools
import proof_v3
from model import forward, sample, D, N
path = sys.argv[1]; ngroups = int(sys.argv[2]) if len(sys.argv) > 2 else 8
p = pickle.load(open(path, 'rb')); dh = p['Q1'].shape[1]; f = N - 1
rng = np.random.default_rng(123)
X, y = sample(rng, 400000)
t = time.time(); lg = forward(p, X, np_=np); el = time.time() - t
from math import perm
total = perm(D, N - 1) * (N - 2)
print(f'sampled acc {(lg.argmax(1)==y).mean():.6f} on 400k; throughput {400000/el:.0f}/s; '
      f'full input space {total:.3e} -> brute force est {total/(400000/el)/3600:.2f} h')
from proof_v1 import build
T = build(p); x = T['x']
s1 = np.einsum('kah,mbh->kamb', x @ p['Q1'], x @ p['K1']) / np.sqrt(dh)
B2 = p['Q2'] @ p['K2'].T / np.sqrt(dh)
viol = 0; worst = np.inf
for _ in range(ngroups):
    q, a = rng.choice(D, 2, replace=False); i = int(rng.integers(0, N - 2))
    Ds = [t for t in range(D) if t not in (q, a)]
    keep = np.array(list(itertools.permutations(Ds, N - 3)))
    Xg = np.empty((len(keep), N), int); Xg[:, i] = q; Xg[:, i + 1] = a; Xg[:, f] = q
    Xg[:, [m for m in range(N - 1) if m not in (i, i + 1)]] = keep
    mg = np.inf; gmin = {}; a2 = np.inf
    for s in range(0, len(Xg), 100000):
        lg, c = forward(p, Xg[s:s+100000], np_=np, ret=True)
        bs = [t for t in range(D) if t != a]
        mg = min(mg, (lg[:, [a]] - lg[:, bs]).min())
        s2 = np.einsum('bd,de,bke->bk', c['r1'][:, f], B2, c['r1'])
        for k in range(N):
            if k != i + 1: gmin[k] = min(gmin.get(k, np.inf), (s2[:, i+1] - s2[:, k]).min())
        a2 = min(a2, c['A2'][:, i+1].min())
    dbg = proof_v3.certify(T, s1, q, i, a, debug=True)
    for k, gl in dbg['gaps'].items():
        viol += gl > gmin[k] + 1e-6; worst = min(worst, gmin[k] - gl)
    viol += dbg['margin'] > mg + 1e-6; viol += dbg['beta'] > a2 + 1e-9
    print(f'group q={q} i={i} a={a}: {len(Xg)} inputs, margin lb {dbg["margin"]:.2f} vs actual {mg:.2f}')
print(f'soundness: {ngroups} groups, violations {viol}, min(actual-bound) {worst:.4f}')
