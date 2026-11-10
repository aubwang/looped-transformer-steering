"""Small hand-computed checks; no experiment inputs required."""
import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('explore',Path(__file__).with_name('mon_explore_scores.py'))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def main():
    y=np.array([0,0,1,1]); s=np.array([0.,2.,1.,3.])
    assert m.auc(y,s)==.75
    assert m.auc(y,np.ones(4))==.5
    assert m.auc(np.zeros(4),s) is None
    assert np.allclose(m.weighted_auc(y,s,np.array([[1,1,1,1],[2,0,1,1]])),[.75,1.])
    x=np.array([0,0,0,1,1,1,1,1]); z=np.array([0,0,1,0,1,1,1,1])
    counts,phi,odds=m.association(x,z)
    assert counts==[2,1,1,4] and odds==8
    assert np.isclose(phi,7/15)
    assert m.association(np.zeros(4),np.zeros(4))[1:] == (None,None)
    assert m.align([3,1,2],[1,2,3]).tolist()==[2,0,1]
    assert m.align([3,1],[1,2,3],True).tolist()==[2,0]
    for a,b in [([1,1],[1,2]),([1,3],[1,2]),([1],[1,2])]:
        try: m.align(a,b)
        except ValueError: pass
        else: raise AssertionError('bad alignment accepted')
    bad=m.paired_bootstrap(y,s,y,s,np.zeros(4,dtype=int),200,1)
    assert bad['interval'] is None and bad['undefined_bootstrap_draws']>0
    good=m.paired_bootstrap(y,s,y,s,np.zeros(4,dtype=int),200,1,True)
    assert good['interval']==[0.,0.] and not good['undefined_bootstrap_draws']
    assert m.fold_auc(y,s,np.array([0,1,0,1]))==1.
    assert np.isclose(m.spearman(s,2*s),1.)
    print('ALL CHECKS PASSED')

if __name__=='__main__': main()
