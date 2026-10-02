"""Compare the interval version of a bound with its float version, group by group.

The two compute the same relaxation, so they should agree to rounding error; a larger difference would
mean the two implementations differ. Run: python tests/check_rigorous_vs_float.py weights/params_s0.pkl [proof_v4]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np
import importlib
name = sys.argv[2] if len(sys.argv) > 2 else 'proof_v4'
flt, rig = importlib.import_module(name), importlib.import_module(name + '_rigorous')

path = sys.argv[1]
_, Tf, s1f = flt.run(path, verbose=False)
_, Tr, s1r = rig.run(path, verbose=False)
for name in ('SK', 'SH', 'xW', 'oxW', 'xU', 'oxU'):
    I = Tr[name]
    assert np.all(I.lo <= Tf[name]) and np.all(Tf[name] <= I.hi), name      # float tables sit inside the enclosures
    print(f'{name}: max enclosure width {I.width():.2e}')
d, n = Tr['d'], Tr['n']
dm = db = dg = 0.0; flips = 0
for q in range(d):
    for a in range(d):
        if a == q: continue
        for i in range(n - 2):
            x, y = flt.certify(Tf, s1f, q, i, a, debug=True), rig.certify(Tr, s1r, q, i, a, debug=True)
            dm = max(dm, abs(x['margin'] - y['margin'])); db = max(db, abs(x['beta'] - y['beta']))
            dg = max(dg, max(abs(x['gaps'][k] - y['gaps'][k]) for k in x['gaps']))
            flips += (x['margin'] > 0) != (y['margin'] > 0)
print(f'max |float - rigorous|: margin {dm:.2e}, gaps {dg:.2e}, beta {db:.2e}; certification changes {flips}')
assert dm < 1e-6 and dg < 1e-6 and db < 1e-6
