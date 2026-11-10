"""Small deterministic synthetic chunk test; no real data modified."""
import importlib.util
import json
from pathlib import Path
import tempfile
import numpy as np

spec=importlib.util.spec_from_file_location('lateshift',Path(__file__).with_name('lateshift_existing.py'))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); run=root/'mon_main-20261110-test'; shard=run/'L32'/'s0'; shard.mkdir(parents=True)
        ts=m.T_STD+[48,63,64]; pred=np.ones((4,len(ts),5),dtype=np.int32)
        # RR, RW, WR, WW; third and fourth start with wrong token 2.
        pred[2:,:,0]=2
        j=ts.index(32); pred[1,j:,0]=2; pred[2,j:,0]=1; pred[3,j:,0]=3
        lp=np.zeros_like(pred,dtype=float); lp[:,:,1]=-2
        kw=dict(pid=np.arange(4),ans_tok=np.ones(4,dtype=np.int32),T=np.array(ts),top_id=pred,top_lp=lp,lp_ans=np.zeros((4,len(ts))))
        np.savez(shard/'c0_deep.npz',**kw)
        # Identical duplicate must not change any denominator.
        np.savez(shard/'c1_deep.npz',**kw)
        counts={'add3':{str(t):[int((pred[:,ts.index(t),0]==1).sum())*2,8] for t in m.T_STD}}
        (run/'containers.json').write_text(json.dumps({'L32:0':{'accuracy_counts':counts}}))
        r=m.main(root,root/'out'); x=r['models']['L32']['pooled']
        assert x['n']==4 and x['changes']['8,32']['rate']==.75
        assert x['changes']['32,64']['rate']==0
        assert all(x['transitions'][k]==1 for k in ['right_right','right_wrong','wrong_right','wrong_wrong'])
        assert x['transitions']['wrong_wrong_changed']['rate']==1
        assert x['accuracy']['8']['rate']==.5 and x['accuracy']['32']['rate']==.5
        assert x['settled_by']['24']['rate']==.25
        assert x['margins']['changed']['8']['median']==2
        assert r['audit']['duplicates']==4 and not r['audit']['accuracy_mismatches']
        assert r['audit']['accuracy_checks']==12
    print('Synthetic test passed')

if __name__=='__main__': test()
