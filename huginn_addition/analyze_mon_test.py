"""Synthetic integration checks; no Modal, GPU, model downloads, or network."""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_v]='1'
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import json
import tempfile
import time
import numpy as np
from scipy.special import ndtri
import analyze_mon as mon


def synthetic(root,n=400,d=48):
    rng=np.random.default_rng(1901); items=[]; chunks={m:[] for m in ('L4','L32','D')}
    for fi,fam in enumerate(('add3','addsub')):
        pid=fi*100000+np.arange(n); y=rng.integers(0,2,n)
        ans=np.full(n,123,dtype=np.int32)
        for i,p in enumerate(pid):
            items.append(dict(pid=int(p),fam=fam,i=i+mon.N_SCREEN,kind='deep' if i<80 else 'std',
                              ops=[12,34,77],ans=123,prompt='12 + 34 + 77 = ',valid='num'))
        latent=rng.normal(size=(n,d)); latent[:,0]+=np.sqrt(2)*ndtri(.90)*y if fi==0 else 0
        for m in chunks:
            base=latent.copy() if m!='D' else rng.normal(size=(n,d))
            base[:,3]=0
            steps=np.full((n,65),np.nan); steps[:,1]=rng.integers(700,1300,n)/1024
            for t in range(2,65):
                ratio=.5 if fi==0 else min(.55+.015*t,.99)
                steps[:,t]=steps[:,t-1]*ratio
            offsets=np.nancumsum(steps,axis=1); trajectory=base[:,None,:]+np.eye(d)[3][None,None,:]*offsets[:,:,None]
            if m=='L4':
                trajectory[:,:9,[0,1]]=trajectory[:,:9,[1,0]]
            top=np.where(y,456,123).astype(np.int32)
            # Readout confidence is deliberately uninformative.
            if m=='D':
                h=np.repeat(base[:,None,:],17,axis=1).astype('f4')
                z=dict(h_last=h.astype('f2'),h_prev=h[:,[8,12,16]].astype('f2'))
                for k in ('dl_full','dl_last','nl_full','nl_last'):
                    z[k]=np.ones((n,17));
                    if k.startswith('dl'): z[k][:,0]=np.nan
                z.update(readout(rng,top,17,'lens_')); z.update(readout(rng,top,None,'ro_'))
            else:
                x=trajectory[:,mon.T_DEEP].astype('f4')
                z=dict(x_last=x.astype('f2'),x_prev=trajectory[:,mon.T_PREV].astype('f2'),
                       post=trajectory[:,mon.R_POST].astype('f2'),pre=np.repeat(base[:,None,:],2,axis=1).astype('f2'))
                for suffix in ('full','last'):
                    z['d_'+suffix]=steps.copy(); z['e_'+suffix]=steps+np.roll(steps,1,axis=1); z['e_'+suffix][:,:2]=np.nan
                    z['n_'+suffix]=np.linalg.norm(trajectory,axis=2)
                    z['cos_'+suffix]=np.ones((n,65)); z['cos_'+suffix][:,:2]=np.nan
                if m=='L32':
                    z['n_pos']=np.full((n,4,160),np.nan,dtype='f4')
                    z['n_pos'][:,:,:8]=z['n_full'][:,mon.R_NPOS,None]/np.sqrt(8)
                    z['d_pos']=np.full((n,65,160),np.nan,dtype='f4')
                    z['d_pos'][:,1:,:8]=0; z['d_pos'][:,1:,0]=steps[:,1:]
                    z['disp_full']=np.full((n,5,2),np.nan)
                    for j,r in enumerate(mon.R_DISP):
                        for k,target in enumerate((32,64)):
                            if r<target: z['disp_full'][:,j,k]=np.linalg.norm(trajectory[:,target]-trajectory[:,r],axis=1)
                z.update(readout(rng,top,15,''))
            z.update(pid=pid,ans_tok=ans,plen=np.full(n,8,dtype='i4'),tok_hash=pid+99)
            for shard in (0,1):
                for deep in (True,False):
                    ix=np.flatnonzero((np.arange(n)%2==shard)&((np.arange(n)<80)==deep))
                    c={k:v[ix].copy() for k,v in z.items()}
                    ci=ix[ix<12]; key='h_last' if m=='D' else 'x_last'
                    c['pid32']=pid[ci]; c[key+'32']=(h[ci] if m=='D' else x[ci]).astype('f4')
                    if m!='D':
                        ts=mon.T_DEEP if deep else mon.T_STD; c['T']=np.array(ts)
                        for k in mon.READOUT: c[k]=c[k][:,:len(ts)]
                        c[key+'32']=c[key+'32'][:,:len(ts)]
                        for k in mon.DYNAMICS:
                            if k in c: c[k]=c[k][:,:65 if deep else 34]
                        if not deep and 'disp_full' in c: c['disp_full'][:,:,1]=np.nan
                    path=root/m/f's{shard}'; path.mkdir(parents=True,exist_ok=True)
                    np.savez(path/f'c{fi}_{"deep" if deep else "std"}.npz',**c)
            for shard in (0,1):
                meta={'status':'complete','code_sha256':'synthetic','valid_ids':{'num':list(range(1000)),'abcd':[2001,2002,2003,2004]},**{k:getattr(mon,k) for k in ('FAMS','T_STD','T_DEEP','T_PREV','R_POST','D_PREV','R_DISP','R_NPOS')}}
                (root/m/f's{shard}'/'meta.json').write_text(json.dumps(meta))
    bad=root/'L32'/'s2'; bad.mkdir(); (bad/'meta.json').write_text('{"status":"partial"}')
    dest=root/'items.json'; dest.write_text(json.dumps({'items':items}))
    for mp in root.glob('*/s*/meta.json'):
        meta=json.loads(mp.read_text()); meta['cfg']={'items_sha256':mon.sha256(dest)}; mp.write_text(json.dumps(meta))
    return dest


def readout(rng,top,nt,prefix):
    n=len(top); shape=(n,) if nt is None else (n,nt)
    ids=np.full((*shape,5),789,dtype='i4'); ids[...,0]=top if nt is None else top[:,None]
    lp=np.sort(rng.normal(-3,.2,(*shape,5)),axis=-1)[...,::-1].astype('f4')
    return {prefix+'top_id':ids,prefix+'top_lp':lp,prefix+'ent':rng.uniform(1,2,shape).astype('f4'),
            prefix+'valid_lm':rng.normal(-1,.1,shape).astype('f4'),prefix+'lp_ans':np.full(shape,-2,dtype='f4'),
            prefix+'opt_lp':np.full((*shape,4),np.nan,dtype='f4')}


def expect_assert(fn):
    try: fn()
    except AssertionError: return
    raise AssertionError('loader accepted invalid input')


def check_two_step():
    # Linear symmetric map with eigenvalues of both signs: the two-step extrapolation with the last observed
    # ratio is a lower bound for the displacement, and the bound with the true factor is an upper bound.
    rng=np.random.default_rng(3); lam=np.array([-.8,-.5,.3,.6]); n=500; T=33
    x=np.zeros((n,T,4)); x[:,0]=rng.normal(size=(n,4))
    for t in range(1,T): x[:,t]=x[:,t-1]*lam+1.
    e=np.full((n,T),np.nan); e[:,2:]=np.linalg.norm(x[:,2:]-x[:,:-2],axis=2)
    for r in (4,8,16):
        q,b=mon.rule_two_step(e,np.ones((n,T)),r); disp=np.linalg.norm(x[:,32]-x[:,r],axis=1); m=(32-r)//2
        upper=e[:,r]*.64*(1-.64**m)/(1-.64)
        assert np.all(b<=disp*(1+1e-9)) and np.all(disp<=upper*(1+1e-9)),r


def check_conformal():
    # Coverage among items with a forecast must be about 95% even when many items have none.
    rng=np.random.default_rng(5); n=40000; pred=rng.lognormal(size=n); pred[rng.random(n)<.4]=np.inf
    realized=np.where(np.isfinite(pred),pred,1.)*rng.lognormal(sigma=.7,size=n)
    c=mon.conformal(realized,{'p':pred,'one':np.ones(n)},rng.integers(0,5,n),0)
    assert abs(c['p']['coverage']-.95)<.01 and abs(c['one']['coverage']-.95)<.01,c
    assert abs(c['p']['forecast_fraction']-.6)<.02,c


def check_bootstrap():
    # Rare class in a tiny fold causes at least one undefined draw; none may be dropped.
    z={'scores':np.arange(4.),'y':np.array([0,0,0,1]),'folds':np.zeros(4,dtype=int),'pid':np.arange(4)}
    a=mon.Analysis('add3',0); a.contrast('rare',[(1,z)])
    v=a.rows['rare']; assert v['undefined_bootstrap_draws']>0 and np.isnan(v['interval']).all()
    assert 'lost a class' in v['interval_unavailable_reason']
    original=mon.paired_bootstrap
    def one_bad(stat,groups,n,seed):
        draws=np.ones(n); draws[0]=np.nan
        return stat(np.arange(len(groups))),np.nan,np.nan,draws
    try:
        mon.paired_bootstrap=one_bad; a.contrast('rare',[(1,z)])
        assert a.rows['rare']['undefined_bootstrap_draws']==1 and np.isnan(a.rows['rare']['interval']).all()
    finally: mon.paired_bootstrap=original
    mean=mon.aggregate([a.rows,a.rows],{'rare':[a.differences['rare']]*2},2)
    assert mean['rare']['interval']==[None,None] and 'interval_unavailable_reason' in mean['rare']


def check_identity():
    # The decoding of a predicted token must not depend on which gold answers occur in the data.
    assert mon.VALID and mon.SETTINGS['output_identity_tokens'].startswith('valid')
    a=mon.subset(mon.DATA['L32'],mon.DATA['L32']['pid'][:6]); a['top_id']=a['top_id'].copy()
    a['top_id'][:3,mon.T_STD.index(32),0]=[999,7,5000]  # 999 is nobody's gold answer; 5000 is not an answer token
    ident,_=mon.problem_features(a,'L32'); units,tens,hundreds,unknown=ident[:,:10],ident[:,10:20],ident[:,20:30],ident[:,-1]
    assert units[0,9]==tens[0,9]==hundreds[0,9]==1 and unknown[0]==0
    assert units[1,7]==tens[1,0]==hundreds[1,0]==1 and unknown[1]==0
    assert unknown[2]==1 and not ident[2,:-1].any()
    expect_assert(lambda: mon.token_answers({'num':[1]*1000,'abcd':[2001,2002,2003,2004]}))


def check_contrast_counts():
    # A failed frozen source probe counts in the contrast that uses its transferred scores.
    class Probe:
        def __init__(self,ok): self.converged=ok
    rng=np.random.default_rng(8); n=200; y=np.arange(n)%2; a=mon.Analysis('add3',0)
    term=lambda bad:{'scores':rng.normal(size=n)+y,'y':y,'folds':np.arange(n)//2%5,'pid':np.arange(n),'models':[Probe(i>=bad) for i in range(5)]}
    a.contrast('c',[(1,term(0)),(-1,term(2))]); assert a.rows['c']['nonconverged_outer_fits']==2,a.rows['c']


def check_floor():
    # An item without saved per-position norms (older data) falls back to the whole-state norm for single
    # positions AND prefixes: steps of 1e-9 against a norm of 100 are rounding noise and must not be used.
    n,L=3,10; dp=np.full((n,65,160),np.nan); t=np.arange(1,65)[None,:,None]
    dp[:,1:,:L]=.9**t*(1+.01*np.arange(L))[None,None,:]; dp[2,1:,:L]*=1e-9
    n_pos=np.full((n,4,160),np.nan); n_pos[:2,:,:L]=100/np.sqrt(L)
    out=mon.position_rates({'d_pos':dp,'n_pos':n_pos,'n_full':np.full((n,65),100.)})['8-16']
    assert out['floor']=={'per_position_norm_items':2,'whole_state_norm_items':1},out['floor']
    assert out['items_used']==2 and out['prefix_items_used']==2,out
    n_pos[2,:,:L]=1e-12  # with its own (tiny) norms the same item is above the floor
    out=mon.position_rates({'d_pos':dp,'n_pos':n_pos,'n_full':np.full((n,65),100.)})['8-16']
    assert out['items_used']==3 and out['prefix_items_used']==3 and out['floor']['whole_state_norm_items']==0,out


def check_missing():
    # A family with no items, and two models with no problems in common, are skipped with a reason, not a crash.
    a=mon.Analysis('arc_easy',0); assert a.local=={} and a.common(['L32','D']) is None
    for name in mon.ANALYSES: getattr(a,name)()
    saved=mon.DATA
    try:
        mon.DATA={m:dict(mon.subset(saved[m],saved[m]['pid'][i::2]),copies=saved[m]['copies']) for i,m in enumerate(('L32','D'))}
        a=mon.Analysis('add3',0); assert a.common(['L32','D']) is None
        for name in ('m1a','m1b','m1c','m1x'): getattr(a,name)()
        assert any(v.get('reason')=='no items in common' for v in a.skips),a.skips
        # a missing target family makes the mean over targets unavailable
        mon.DATA=saved; mon.SETTINGS['fams']=['add3','addsub','arc_easy']; a=mon.Analysis('add3',0); a.m1x()
        v=a.rows['mean_over_targets']; assert v=={'targets':1,'missing_targets':['arc_easy']},v
    finally: mon.DATA=saved; mon.SETTINGS['fams']=['add3','addsub']


def check_aggregate():
    out=mon.aggregate([{'a':{'x':1.,'y':2.}},{'a':{'x':3.}}],{'a':[np.ones(200)]},2)
    assert out['a']['x']==2 and out['a']['y'] is None and out['incomplete']['a/y']==1
    assert out['a']['interval']==[None,None] and out['a']['incomplete'] and out['a']['families']==1


def main():
    start=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='analyze-mon-test-') as tmp:
        root=Path(tmp)/'runs'; items=synthetic(root)
        data,prov=mon.load_runs([root]); assert len(prov['skipped_shards'])==1
        assert data['L32']['x_last'].dtype==np.float16
        assert data['L32']['x_last'].shape[1]==len(mon.T_STD)
        assert len(data['L32']['deep']['pid'])==160
        assert 'disp_full' not in data['L4'] and 'disp_full' in data['L32']
        expect_assert(lambda:mon.load_runs([root,root]))
        path=root/'D'/'s0'/'c0_deep.npz'
        with np.load(path) as z: saved={k:z[k] for k in z.files}
        changed={k:v.copy() for k,v in saved.items()}; changed['tok_hash'][0]+=1; np.savez(path,**changed)
        expect_assert(lambda:mon.load_runs([root])); np.savez(path,**saved)
        manifest={'families':['add3','addsub'],'n_main':400,'shards':2,'items_sha256':mon.sha256(items),'cfg':{},'code_sha256':'synthetic',
                  'analysis_sha256':{name:mon.sha256(Path(mon.__file__).with_name(name)) for name in ('analyze_mon.py','mon_probe.py')}}
        manifest_path=root/'main_manifest.json'; manifest_path.write_text(json.dumps(manifest))
        bad_manifest=dict(manifest,items_sha256='wrong'); bad_path=root/'bad_manifest.json'; bad_path.write_text(json.dumps(bad_manifest))
        expect_assert(lambda:mon.main(['--runs',str(root),'--items',str(items),'--out',str(Path(tmp)/'bad'),'--manifest',str(bad_path)]))
        # Optional n_pos merges with NaN rows for older chunks.
        path_optional=root/'L32'/'s0'/'c0_std.npz'
        with np.load(path_optional) as z: optional={k:z[k] for k in z.files}
        np.savez(path_optional,**{k:v for k,v in optional.items() if k!='n_pos'})
        mixed,_=mon.load_runs([root]); assert np.isnan(mixed['L32']['n_pos'][80]).all()
        np.savez(path_optional,**optional)
        results=mon.main(['--smoke','--runs',str(root),'--items',str(items),'--out',str(Path(tmp)/'out'),'--jobs','4','--manifest',str(manifest_path)])
        assert results['provenance']['analysis_matches_manifest'] is True
        assert all(v['complete'] and v['missing']==0 for v in results['provenance']['completeness'].values())
        def rows(name,fam='add3'): return results['analyses'][name]['0'][fam]['estimates']
        r=rows('m1a'); null=rows('m1a','addsub')
        assert abs(r['table/L32/L32']['auroc']-.85)<.10,r
        assert abs(null['table/L32/L32']['auroc']-.5)<.12,null
        assert r['column']['difference']>.10,r['column']
        with np.load(Path(tmp)/'out'/'scores.npz') as ss:
            fs=[ss[f'm1a/add3/0/table/{row}/{col}/folds'] for row in ('L32','D') for col in ('L32','D')]
            assert all(np.array_equal(fs[0],f) for f in fs)
        for row in ('L32','D'):
            for col in ('L32','D'):
                v=r[f'table/{row}/{col}']; assert v['interval'][0]<=v['auroc']<=v['interval'][1],v
        assert 'nonconverged_outer_fits' in r['column']
        r=rows('m3a')
        assert 'transfer_interval' in r['L32/32->4']
        assert 'transfer_interval' in results['analyses']['m3a']['0']['family_mean']['estimates']['L32/32->4']
        assert abs(r['L32/32->4']['transfer_auroc']-r['L32/32->4']['native_auroc'])<.04
        assert r['L4/32->4']['native_auroc']-r['L4/32->4']['transfer_auroc']>.15
        for fam in ('add3','addsub'):
            r=rows('t1',fam)
            for budget in (4,8,16):
                v=r[f'L32/{budget}']
                assert v['classification_transfer']['slack']>=-1e-12
                assert v['quantization_allowance'] is not None and v['allowance_over_movement'] is not None
                assert 'score_violation_beyond_quantization' in v
                assert v['rounding_subset']['n']==12 and v['rounding_subset']['max_score_error_over_q_r']<=1+1e-6
                for c in v['score_conformal'].values():
                    assert c['decision_changed']<=c['decision_change_bound']+1e-12
                    assert c['ranking_folds_used']>0
                    assert c['auroc_change_fixed_labels']<=c['close_pair_mass']+c['violated_pair_mass']+1e-12
                    assert 'stable_and_violated' in c and 'coverage_with_allowance' in c and 'fraction_forecast_stable_with_allowance' in c
                if fam=='add3': assert v['path_violation']==0 and v['exact_violation']==0,v
                else:
                    assert v['path_violation']>.8,v
                    cov=v['state_conformal']['scaled_rule']['coverage']; assert .88<=cov<=1,cov
                    sc=v['score_conformal']; assert len(sc)==9 and len(v['state_conformal'])==5,sc.keys()
                    for k,c in sc.items():
                        if c['coverage'] is not None: assert .85<=c['coverage']<=1,(k,c)
            for model in ('L32','D'):
                auc=rows('m1s',fam)[model+'/permutation']['auroc']; assert abs(auc-.5)<.12,auc
        pr=rows('dyn')['L32']['position_rate']; assert set(pr)=={'8-16','16-24','24-32'},pr.keys()
        assert all(v['floor']['per_position_norm_items']>0 and v['floor']['whole_state_norm_items']==0 for v in pr.values())
        assert all('prefix_largest_shortfall_below_running_max' in v and 'last_position_minus_whole_state' in v for v in pr.values())
        x=rows('m1x'); v=x['to/addsub']
        assert set(x)>={'L32/source','D/source','to/addsub','mean_over_targets'} and 'to/add3' not in x,x.keys()
        for m in ('L32','D'):  # the target family has no signal: transfer is at chance, inside its own interval
            assert v[m+'_interval'][0]<=v[m+'_auroc']<=v[m+'_interval'][1] and abs(v[m+'_auroc']-.5)<.12,v
        assert v['difference_interval'][0]<=v['difference']<=v['difference_interval'][1]
        assert x['mean_over_targets']['targets']==1 and x['mean_over_targets']['L32_minus_own_family']<-.2,x['mean_over_targets']
        assert x['mean_over_targets']['missing_targets']==[] and abs(v['L32_auroc_single_probe']-.5)<.12
        assert all(rows('m3a')[k]['source_nonconverged']==0 for k in rows('m3a') if '->' in k and 'matrix' not in k and not k.endswith('transfer'))
        assert isinstance(rows('m1a')['table/L32/L32']['inner_nonconverged'],int)
        check_bootstrap(); check_aggregate(); check_conformal(); check_two_step()
        check_identity(); check_contrast_counts(); check_floor(); check_missing()
        print(f'ALL CHECKS PASSED ({time.perf_counter()-start:.2f}s)',flush=True)


if __name__=='__main__': main()
