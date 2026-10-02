"""Soundness check: every bound must be <= the true value over the group's inputs."""
import numpy as np, pickle, sys, itertools
from model import forward, D, N
import importlib
mod = importlib.import_module(sys.argv[2] if len(sys.argv)>2 else 'proof_v4')
_, T, s1 = mod.run(sys.argv[1], verbose=False)
p = pickle.load(open(sys.argv[1],'rb')); dh=p['Q1'].shape[1]
B2=p['Q2']@p['K2'].T/np.sqrt(dh); f=N-1
perms = np.array(list(itertools.permutations(range(D), N-3)))
viol=0; worst=np.inf; cnt=0
for q in range(D):
    for a in range(D):
        if a==q: continue
        keep = perms[~np.isin(perms,[q,a]).any(1)]
        for i in range(N-2):
            X=np.empty((len(keep),N),int); X[:,i]=q; X[:,i+1]=a; X[:,f]=q
            others=[m for m in range(N-1) if m not in (i,i+1)]; X[:,others]=keep
            lg,c=forward(p,X,np_=np,ret=True)
            bs=[t for t in range(D) if t!=a]
            mg=(lg[:,[a]]-lg[:,bs]).min()
            s2=np.einsum('bd,de,bke->bk',c['r1'][:,f],B2,c['r1'])
            dbg=mod.certify(T,s1,q,i,a,debug=True)
            for k,gl in dbg['gaps'].items():
                act=(s2[:,i+1]-s2[:,k]).min(); worst=min(worst, act-gl)
                if gl>act+1e-6: viol+=1
            if dbg['margin']>mg+1e-6: viol+=1
            if dbg['beta']>c['A2'][:,i+1].min()+1e-9: viol+=1
            worst=min(worst, mg-dbg['margin']); cnt+=1
print(f'groups checked {cnt}, violations {viol}, min(actual - bound) {worst:.4f}')
