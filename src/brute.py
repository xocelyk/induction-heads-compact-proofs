import numpy as np, pickle, itertools, time, sys
from model import forward, D, N
def all_inputs(d=D, n=N):
    perms = np.array(list(itertools.permutations(range(d), n-1)), dtype=np.int64)
    out=[]
    for i in range(n-2):
        X = np.concatenate([perms, perms[:, i:i+1]], 1)
        out.append((X, perms[:, i+1], i))
    return out
if __name__=='__main__':
    p = pickle.load(open(sys.argv[1],'rb'))
    t=time.time(); tot=0; cor=0; minmargin=np.inf
    for X,y,i in all_inputs():
        for s in range(0, len(X), 200000):
            lg = forward(p, X[s:s+200000], np_=np)
            yy=y[s:s+200000]
            c = lg[np.arange(len(yy)), yy]
            lg2=lg.copy(); lg2[np.arange(len(yy)), yy]=-np.inf
            mg = c - lg2.max(1)
            cor += (mg>0).sum(); tot += len(yy); minmargin=min(minmargin, mg.min())
    print(f'brute force: acc {cor/tot:.6f} on {tot} inputs, min margin {minmargin:.3f}, {time.time()-t:.0f}s')
