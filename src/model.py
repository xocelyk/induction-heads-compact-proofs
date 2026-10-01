"""2-layer attention-only transformer (no LayerNorm, no MLP) on a finite induction task.

Task: vocab d, length n. x_0..x_{n-2} are distinct tokens; x_{n-1} = x_i for some i in [0, n-3].
Target at the final position: x_{i+1}. Inputs uniform over all such sequences.
"""
import autograd.numpy as anp
import numpy as np
from autograd import grad
import itertools, pickle, sys, time

import os
D, N, DM, DH = int(os.environ.get('D', 16)), int(os.environ.get('N', 6)), 32, 16


def sample(rng, B, d=D, n=N):
    X = np.empty((B, n), dtype=np.int64)
    for b in range(B):
        X[b, :n - 1] = rng.choice(d, n - 1, replace=False)
    i = rng.integers(0, n - 2, size=B)
    X[:, n - 1] = X[np.arange(B), i]
    y = X[np.arange(B), i + 1]
    return X, y


def init(rng, d=D, n=N, dm=DM, dh=DH):
    s = lambda *sh: rng.normal(0, 1 / np.sqrt(sh[0]), sh)
    return dict(E=s(d, dm), P=s(n, dm),
                Q1=s(dm, dh), K1=s(dm, dh), V1=s(dm, dh), O1=s(dh, dm),
                Q2=s(dm, dh), K2=s(dm, dh), V2=s(dm, dh), O2=s(dh, dm),
                U=s(dm, d))


def softmax(z, ax=-1, np_=anp):
    z = z - np_.max(z, axis=ax, keepdims=True)
    e = np_.exp(z)
    return e / np_.sum(e, axis=ax, keepdims=True)


def forward(p, X, np_=anp, ret=False):
    B, n = X.shape
    dh = p['Q1'].shape[1]
    mask = np.triu(np.full((n, n), -1e9), 1)
    r0 = p['E'][X] + p['P'][None]
    s1 = np_.einsum('bqh,bkh->bqk', r0 @ p['Q1'], r0 @ p['K1']) / np.sqrt(dh) + mask
    A1 = softmax(s1, np_=np_)
    r1 = r0 + np_.einsum('bqk,bkh->bqh', A1, r0 @ p['V1']) @ p['O1']
    q2 = r1[:, -1] @ p['Q2']
    s2 = np_.einsum('bh,bkh->bk', q2, r1 @ p['K2']) / np.sqrt(dh)
    A2 = softmax(s2, np_=np_)
    r2 = r1[:, -1] + np_.einsum('bk,bkh->bh', A2, r1 @ p['V2']) @ p['O2']
    logits = r2 @ p['U']
    if ret:
        return logits, dict(A1=A1, A2=A2, r0=r0, r1=r1)
    return logits


def loss(p, X, y):
    lg = forward(p, X)
    lg = lg - anp.max(lg, axis=1, keepdims=True)
    lp = lg - anp.log(anp.sum(anp.exp(lg), axis=1, keepdims=True))
    return -anp.mean(lp[np.arange(len(y)), y])


def train(seed=0, steps=6000, B=256, lr=3e-3, wd=1e-4):
    rng = np.random.default_rng(seed)
    p = init(rng)
    g = grad(loss)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    b1, b2 = 0.9, 0.98
    t0 = time.time()
    for t in range(1, steps + 1):
        X, y = sample(rng, B)
        gr = g(p, X, y)
        for k in p:
            m[k] = b1 * m[k] + (1 - b1) * gr[k]
            v[k] = b2 * v[k] + (1 - b2) * gr[k] ** 2
            p[k] = p[k] - lr * (m[k] / (1 - b1 ** t)) / (np.sqrt(v[k] / (1 - b2 ** t)) + 1e-8) - lr * wd * p[k]
        if t % 500 == 0:
            Xv, yv = sample(rng, 2000)
            acc = (np.argmax(forward(p, Xv, np_=np), 1) == yv).mean()
            print(f'step {t} loss {loss(p, X, y):.4f} acc {acc:.4f} ({time.time()-t0:.0f}s)', flush=True)
    return p


if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    p = train(seed, steps)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'weights',
                       f'params_s{seed}.pkl' if N == 6 else f'params_n{N}_s{seed}.pkl')
    with open(out, 'wb') as f:
        pickle.dump(p, f)
