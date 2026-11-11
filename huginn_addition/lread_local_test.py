"""Offline CPU checks: /tmp/toyenv/bin/python huginn_addition/lread_local_test.py."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import lread_app as app
import analyze_lread as an
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(2); torch.manual_seed(73)
checks=0

def ok(value,label):
    global checks
    assert value,label
    checks+=1; print('PASS',label,flush=True)


def close(a,b,tol=1e-8):
    return torch.allclose(a,b,atol=tol,rtol=tol)


root=Path(sys.argv[1] if len(sys.argv)>1 else '/tmp/obs_test')
cfg=AutoConfig.from_pretrained(root/'tinyL',trust_remote_code=True,local_files_only=True)
cfg.n_embd=16; cfg.n_heads=2; cfg.num_key_value_heads=1; cfg.head_dim=8; cfg.intermediate_size=32
M=AutoModelForCausalLM.from_config(cfg,trust_remote_code=True,torch_dtype=torch.float64).eval().requires_grad_(False)
app.patch_precision(M)
tok=AutoTokenizer.from_pretrained(root/'tok',local_files_only=True)
prep,step,read=app.harness(M,'cpu')
items=[p for p in app.make_items() if p['split']=='eval']
ids=torch.tensor([[app.BOS]+tok.encode(items[0]['prompt'],add_special_tokens=False)])
with torch.no_grad(): e,fc=prep(ids)
x0=app.start(M,items[:1],ids.shape[1],11000000,'cpu').double()
tokens=[app.one_token(tok,str(70+k)) for k in range(10)]
x,base=app.base_state(step,read,x0,e,fc,tokens)
q=x[0,-1].detach()
def own(q): return step(torch.cat((x[:,:-1],q[None,None]),1),e,fc)[0,-1]
J=torch.autograd.functional.jacobian(own,q)
for task in ('A','B'):
    tt=tokens if task=='A' else [app.one_token(tok,w) for w in (' True',' False')]
    U,gn,G,pairs=app.directions(M,tok,read,x,fc,tt,task,items[0]['pid'],rows=7)
    U2,gn2,G2,pairs2=app.directions(M,tok,read,x,fc,tt,task,items[0]['pid'],rows=7)
    ok(torch.equal(U,U2) and np.array_equal(pairs,pairs2),'determinism '+task)
    f=app.families(task)
    ok(np.array_equal(np.bincount(f),[10 if task=='A' else 1,8,8,8,4,12,8]),'family counts '+task)
    ok(close(U.norm(dim=-1),torch.ones(len(U),dtype=torch.float64),1e-12),'unit norms '+task)
    out=app.selfgain(step,x,e,fc,U,rows=7)
    w1=U@J; w2=w1@J
    for family in range(7):
        ok(close(out['gram3'][f==family,0,1],(U*w1).sum(-1)[f==family]),f'dense alpha {task} family {family}')
    ok(close(out['x1'],U@w1.T),'dense x1 '+task)
    ok(close(out['x2'],U@w2.T),'dense x2 '+task)
    ok(float(out['adjoint_rel_err'].max())<1e-4,'FD adjoints '+task)
    C,_=app.coefficients(tok,read(x,fc)[1][0].detach(),tt,task,items[0]['pid'])
    for k in range(8):
        coeff=C[(10 if task=='A' else 1)+16+k]
        def obj(q): return (read(torch.cat((x[:,:-1],q[None,None]),1),fc)[1][0]*coeff).sum()
        dense=torch.autograd.functional.jacobian(obj,q)
        ok(close(U[np.flatnonzero(f==3)[k]],dense/dense.norm()),f'dense C3 {task} {k}')
    ok(close(U[f==3]**2,U[f==6]**2),'C3flip energies '+task)
    if task=='A': ok(close(G,app.readout_grads(read,x,fc,tokens)[0]),'lever task gradients identical')

# Actual production context path, G1, and a deadline between two contexts.
z,tm=app.run_set(M,tok,items[:1],'A',dict(rows=7),time.monotonic()+120)
ok(z['done'].all() and max(tm['G1'])<1e-6,'tiny harness G1 and run_set')
# Deterministic interruption during the second context exercises the same
# TimeoutError recovery as the wall-clock checks, without a timing race.
import unittest.mock
real_direction=app.directions
calls=0
def stop_second(*args,**kwargs):
    global calls
    calls+=1
    if calls==2: raise TimeoutError('second context expired')
    return real_direction(*args,**kwargs)
with unittest.mock.patch.object(app,'directions',stop_second):
    partial,_=app.run_set(M,tok,items[:2],'A',dict(rows=7),time.monotonic()+120)
ok(partial['done'].tolist()==[True,False],'deadline preserves completed context only')
meta=dict(cfg=dict(smoke=False,n_A=2),gates=dict(G1=[0.]))
ok(not an.analyze_set(partial,meta,'A')['gates']['PASS'],'partial analysis refuses reading')
calls=0
def fail_second(*args,**kwargs):
    global calls
    calls+=1
    if calls==2: raise AssertionError('planted failure')
    return real_direction(*args,**kwargs)
with unittest.mock.patch.object(app,'directions',fail_second):
    broken,bt_=app.run_set(M,tok,items[:2],'A',dict(rows=7),time.monotonic()+120)
ok(broken['done'].tolist()==[True,False] and 'planted failure' in bt_['error'],'a failing context keeps the completed ones and reports the error')
expired,_=app.run_set(M,tok,items[:2],'A',dict(rows=7),time.monotonic()-1)
ok(not expired['done'].any(),'already expired deadline')

# Truth prompts retain variable lengths and use the specified pid and truth label.
truth=json.loads((Path(__file__).resolve().parent.parent/'results/freeprobe-20261107-130520/items.json').read_text())
bitems=[dict(p,pid=900000+p['uid']) for p in truth[:2]]
bz,bt=app.run_set(M,tok,bitems,'B',dict(rows=7),time.monotonic()+120)
ok(bz['done'].all() and max(bt['G1'])<1e-6,'truth run_set and G1')
ok(np.array_equal(bz['pid'],[900000+p['uid'] for p in truth[:2]]),'truth pids')

# Controlled, physically realizable dense-Jacobian worlds (large d permits a
# low variance probe family while the read family has a planted signed gain).
f=app.families('B'); n=6; S=len(f); d=128
rng=np.random.default_rng(17)
baseU=np.eye(d)[:S]
for world,expected,tau,delta in [('deficit','STRONG DEFICIT',.4,-.3),
                                ('positive','NOT DETECTED',.4,.2),
                                ('zero','NOT DETECTED',0.,.001)]:
    UU=np.broadcast_to(baseU,(n,S,d)).copy()
    JJ=np.zeros((d,d)); JJ[np.arange(S),np.arange(S)]=tau
    JJ[np.flatnonzero(f<=3),np.flatnonzero(f<=3)]+=delta
    JJ[np.flatnonzero(f==6),np.flatnonzero(f==6)]+=delta
    # Nonzero symmetric off-diagonals ensure positive Proposition R variance.
    for k in range(S): JJ[k,(k+S)%d]=.1; JJ[(k+S)%d,k]=.1
    W1=UU@JJ; W2=W1@JJ; CV=np.stack((UU,W1,W2),2)
    gg=CV@CV.transpose(0,1,3,2)
    syn=dict(gram3=gg,ugram=UU@UU.transpose(0,2,1),x1=UU@W1.transpose(0,2,1),x2=UU@W2.transpose(0,2,1),
             G_task=UU[:,:1],fam=f,done=np.ones(n,bool),adjoint_rel_err=np.zeros((n,S)),gnorm=np.ones((n,S)),
             cos_x=np.zeros((n,S)),drift=np.zeros(n),xnorm=np.ones(n),argmax=np.zeros(n,int),
             correct=np.ones(n,bool),C_tokens=np.zeros((n,16,2),int),pid=np.arange(n),uid=np.arange(n))
    mt=dict(cfg=dict(smoke=False,n_B=n),gates=dict(G1=[0.]))
    result=an.analyze_set(syn,mt,'B')
    ok(result['gates']['PASS'],'synthetic gates '+world)
    reading=result['main']['primary']
    ok(reading['reading']==expected,'synthetic reading '+world)
    if world=='positive': ok(reading['sublabel']=='POSITIVE EXCESS','positive sublabel')
    if world=='zero': ok(reading['tau_near_zero'] and reading['meaning']==an.ZERO,'near-zero writeup meaning')
    if world=='deficit': ok(reading['meaning']==an.STRONG,'strong writeup meaning')
    syn['done'][-1]=False
    ok('main' not in an.analyze_set(syn,mt,'B'),'incomplete set no statistics '+world)

# Matching is by ids, and verifies problem cells; round trip tests the CLI path.
with tempfile.TemporaryDirectory() as directory:
    p=Path(directory)
    z2,_=app.run_set(M,tok,items[:2],'A',dict(rows=7),time.monotonic()+120)   # two distinct problems, reference stored reversed
    ok(z2['done'].all() and len(set(z2['pid'].tolist()))==2 and np.abs(z2['G_task'][0]-z2['G_task'][1]).max()>0,'two distinct contexts')
    np.savez(p/'llever.npz',G=z2['G_task'][::-1],gradient_pid=z2['pid'][::-1],pid=z2['pid'][::-1],xq=z2['xq'][::-1])
    (p/'items.json').write_text(json.dumps(items[:2]))
    ok(an.identity(z2,p)[0]<1e-15 and an.identity(z2,p)[1]<1e-15,'identity by pid/cell (gradients and state), permuted reference')
    np.savez(p/'wrong.npz',a=1); q=Path(directory)/'swapped'; q.mkdir()
    np.savez(q/'llever.npz',G=z2['G_task'],gradient_pid=z2['pid'][::-1],pid=z2['pid'][::-1],xq=z2['xq'])
    (q/'items.json').write_text(json.dumps(items[:2]))
    ok(an.identity(z2,q)[0]>1e-6 and an.identity(z2,q)[1]>1e-9,'identity rejects a reference matched to the wrong problems')
    bad=dict(z2,G_task=z2['G_task']+1)
    ok(an.identity(bad,p)[0]>1e-6,'identity rejects altered gradients')
    bad2=dict(z2); bad2['xq']=z2['xq']*(1+1e-6); ok(an.identity(bad2,p)[1]>1e-9,'identity rejects a different state')
    meta=dict(cfg=dict(smoke=True,n_A=1,n_B=1),gates=dict(G1=[0.]))
    arrays={s+'_'+k:v for s in ('A','B') for k,v in z.items()}
    np.savez(p/'lread.npz',**arrays); (p/'meta.json').write_text(json.dumps(meta))
    with contextlib.redirect_stdout(io.StringIO()): summary=an.main(p)
    ok(summary['meaning_for_writeup']=='SMOKE: no reading' and (p/'lread_summary.json').exists(),'summary file and smoke no reading')
print(f'LOCAL TEST PASSED ({checks} checks)',flush=True)
