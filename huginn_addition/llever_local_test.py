"""CPU tests. Optional argument: directory containing tinyL/ and tok/ fixtures."""
import ast
import inspect
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import llever_app as app
import analyze_llever as an
import make_llever_pushes as builder

torch.set_num_threads(2)
torch.manual_seed(73)
checks = 0


def ok(c, label):
    global checks
    assert c, label
    checks += 1; print('PASS',label,flush=True)


def rel(a,b): return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-300))


# Import source functions without executing the reference modules or their Modal apps.
for fn,names in [('mon_app.py',['one_token','start','harness','centered_error']),('obs_app.py',['patch_norms'])]:
    text = (Path(__file__).parent/fn).read_text(); tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            ok(inspect.getsource(getattr(app,node.name)).strip()==ast.get_source_segment(text,node).strip(), 'source equality '+node.name)
items = app.make_items(); ok(items==app.make_items(),'deterministic generator')
ok(len(items)==180 and len({p['pid'] for p in items})==180,'unique ids')
ok(len({(p['ctx'],p['a'],p['u']) for p in items})==180,'distinct cells')
for split,n in [('eval',40),('train',40),('fit',100)]:
    pp=[p for p in items if p['split']==split]; ok(len(pp)==n,'split '+split)
    if split!='fit': ok(all(pp[j^1]['u']==p['u'] and pp[j^1]['ctx']!=p['ctx'] and pp[(j+2)%40]['u']!=p['u'] for j,p in enumerate(pp)),'ordering '+split)
ok(all(p['x']+p['y']==p['ans']==70+p['u'] for p in items),'query arithmetic')
ok(all(int(l.split('=')[1])==sum(map(int,l.split('=')[0].split('+'))) for p in items for l in p['prompt'].splitlines()[:4]),'demo arithmetic')
ok(len({''.join('#' if c.isdigit() else c for c in p['prompt']) for p in items})==1,'equal character structure')
query_operands={v for p in items for v in (p['x'],p['y'])}
ok(all(a not in query_operands and b not in query_operands and not 70<=a+b<=79 for ctx in app.DEMOS for a,b in ctx),'demo disjointness')

# A smooth contraction with a known nonzero fixed point and exact Jacobian.
d=12; n=2; c=10; dtype=torch.float64
W=torch.randn(d,d,dtype=dtype)*.045
xref=torch.linspace(.12,.35,d,dtype=dtype)
J=torch.diag(1-xref*xref)@W
x=xref.expand(n,3,d).clone(); e=torch.zeros_like(x); fc=()
H=torch.randn(c,d,dtype=dtype)
H=H-H.mean(0)
def step(st,e,fc): return torch.tanh((st-xref)@W.T+torch.atanh(xref))
def read(st,fc):
    lg=st[:,-1]@H.T
    _=lg.softmax(-1)  # same logits -> softmax interface as the production readout
    return st,lg
base,b=app.base_state(step,read,x+.001,e,fc,list(range(c)))
ok(rel(base,x)<1e-12,'base state converges')
g=app.readout_grads(read,base,fc,list(range(c)))
ok(rel(g,H.expand(n,-1,-1))<1e-14,'readout gradient')
T=48; A=torch.zeros_like(J); term=torch.eye(d,dtype=dtype)
for _ in range(T): term=J@term; A+=term
atg,tail=app.chain(step,base,e,fc,g,T,'vjp',chunk=7)
P=torch.randn(4,d,dtype=dtype); P/=P.norm(dim=-1,keepdim=True)
v=P[None].expand(n,-1,-1)
ap,tailp=app.chain(step,base,e,fc,v,T,'jvp',chunk=3)
ok(rel(atg,H[None]@A)<1e-8,'VJP analytic sum J^1 through J^48')
ok(rel(ap,v@A.T)<1e-8,'FD JVP analytic sum J^1 through J^48')
keep={1:None,2:None,8:None}; ap2,_=app.chain(step,base,e,fc,v,T,'jvp',chunk=3,keep=keep)
ok(torch.equal(ap2,ap),'partial sums do not change the full sum')
for tt in keep:
    At=torch.zeros_like(J); term=torch.eye(d,dtype=dtype)
    for _ in range(tt): term=J@term; At+=term
    ok(rel(keep[tt],v@At.T)<1e-8,f'partial sum J^1 through J^{tt}')
# matched-time linear response: forced response after t loops against G (J + ... + J^t) P at a small dose
zs=app.forced(step,read,base,e,fc,list(range(c)),torch.cat([torch.zeros(1,d,dtype=dtype),P*1e-4]),T,chunk=5)
for tt in (1,2,8):
    i=list(zs['times']).index(tt); o=zs['Z'][:,1:,i]-zs['Z'][:,:1,i]
    Hc=H-H.mean(0,keepdim=True); ok(rel(o,1e-4*(keep[tt]@Hc.T))<1e-2,f'forced response at t={tt} equals the partial-sum prediction')
ok(rel(atg@P.T,g@ap.transpose(1,2))<1e-8,'adjoint identity')
responses=[]
for dose in (1e-4,8.):
    pushes=torch.cat([torch.zeros(1,d,dtype=dtype),P*dose])
    z=app.forced(step,read,base,e,fc,list(range(c)),pushes,T,chunk=5)
    obs=z['Z'][:,1:,-1]-z['Z'][:,:1,-1]
    prediction=(g@ap.transpose(1,2)).transpose(1,2)*dose
    responses.append(rel(obs,prediction))
ok(responses[0]<.01 and responses[1]>.01,f'forced response small/large dose relative errors {responses}')
ok(float(tail.max())<1e-15 and float(tailp.max())<1e-15,'Neumann tails')

# Synthetic arrays exercise the actual analysis, including gates, CIs and all predictors.
rng=np.random.default_rng(11); n=40; d=20
center=np.eye(10)-np.ones((10,10))/10
G=np.broadcast_to(np.c_[center,np.zeros((10,10))],(80,10,d)).copy()
PL=np.eye(d)[10:]; P=np.vstack([PL,np.eye(d)[:10],PL,PL[:2]])
AP=rng.normal(0,.25,(80,10,d)); AP[:,:,:10]+=np.eye(10)[None]*2
# Satisfy exact train adjoint by defining the last ten AtG coordinates.
AtG=np.zeros((40,10,d)); AtG[:,:,10:]=np.einsum('nkd,nmd->nkm',G[40:],AP[40:])
zt=dict(AtG=AtG,G=G[40:],tail=np.zeros((40,10)),gradient_pid=np.arange(40,80))
zp=dict(G=G,AP=AP,tail=np.zeros((80,10)),pid=np.arange(80),split=np.array(['eval']*40+['train']*40))
rows=app.row_spec([f'{f}{k}' for f in 'LGC' for k in range(10)]+['R0','R1'])
times=np.array([1,2,4,8,16,48]); full=np.einsum('nmd,nkd->nkm',G[:40],AP[:40])
metas=[dict(status='complete',gates=dict(G1=[0.,0.])) for _ in range(3)]
def synth(kind):
    Z=np.zeros((n,len(rows),6,10))
    for r,(k,rho) in enumerate(rows[1:],1):
        if k<10: mat=full[:,k]
        elif k<20: mat=full[:,k-10] if kind=='equal' else center[k-10][None]*.1
        else: mat=rng.normal(size=(n,10))*.01
        if kind=='nonlinear' and k<10: mat=mat[:,::-1]*3
        Z[:,r]=rho*mat[:,None]
    return dict(Z=Z,pid=np.arange(40),ans=70+(np.arange(40)%20)//2,ans_token=np.arange(40),
                top=np.broadcast_to(np.arange(40)[:,None],(40,len(rows))),last_drift=np.zeros((40,len(rows))),
                row_push=np.array([k for k,r in rows]),row_rho=np.array([r for k,r in rows]),times=times)
for kind,want in [('linear','HUGINN PATTERN REPLICATES'),('equal','RESPONSE SHAPE PREDICTED, NO DEMONSTRATED LEVER ADVANTAGE'),('nonlinear','PATTERN NOT REPLICATED'),('broken','INCONCLUSIVE')]:
    ze=synth(kind)
    if kind=='broken': ze['last_drift'][:]=1.
    result=an.analyze(zt,zp,ze,metas,dict(P=P,m_ref=np.array(1.)))
    ok(result['verdict']==want,'synthetic '+want)
    ok(set(result['doses']['0.03']['P3'])=={'FULL','AVERAGED','SWAPPED','SWAPPED_SAME_CLASS','IDENTITY'},'all five predictors '+kind)

# The real Modal-independent pipeline, using a tiny instance of the pinned architecture.
if len(sys.argv)>1:
    from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
    root=Path(sys.argv[1]); config=AutoConfig.from_pretrained(root/'tinyL',trust_remote_code=True)
    config.n_embd=16; config.n_heads=2; config.num_key_value_heads=1; config.head_dim=8; config.intermediate_size=32
    M=AutoModelForCausalLM.from_config(config,trust_remote_code=True,torch_dtype=torch.float64).eval().requires_grad_(False)
    precision=app.patch_precision(M)
    tok=AutoTokenizer.from_pretrained(root/'tok')
    lengths={len(tok.encode(p['prompt'],add_special_tokens=False)) for p in items}
    ok(len(lengths)==1,'real tokenizer all 180 equal token lengths')
    out,meta=app.smoke(M,tok,dict(T=48,chunk=4),time.monotonic()+500)
    ok(max(meta['G1'])<=1e-6,'tiny Raven G1 at 4 and 32')
    ok(all(np.isfinite(v).all() and v.dtype==np.float64 for v in out.values() if v.dtype.kind=='f'),'tiny Raven all three stages finite float64')
    print('TINY RAVEN',json.dumps(meta),flush=True)
else: print('SKIP tiny Raven: pass fixture directory containing tinyL/ and tok/')
# Push builder plus on-disk analysis round trip, including strict JSON serialization.
import contextlib
import hashlib
import io
import tempfile
tritems=[p for p in items if p['split']=='train']
fititems=[p for p in items if p['split']=='fit']
train_data=dict(zt, split=np.array(['train']*40+['fit']*100),
    pid=np.array([p['pid'] for p in tritems+fititems]),
    ans=np.array([p['ans'] for p in tritems+fititems]),
    norms=np.ones(140)*3, xq=rng.normal(size=(140,d)))
train_data['gradient_pid']=train_data['pid'][:40]
built=builder.build(train_data)
ok(built['P'].shape==(32,20) and np.allclose(np.linalg.norm(built['P'],axis=1),1)
   and built['m_ref']==3, 'push builder unit families and m_ref')
with tempfile.TemporaryDirectory() as tmp:
    roots=[Path(tmp)/q for q in ('train','pred','eval')]
    for root in roots: root.mkdir()
    push=dict(P=P,m_ref=np.array(1.))
    np.savez(roots[2]/'pushes.npz',**push,T=48)
    sha=hashlib.sha256((roots[2]/'pushes.npz').read_bytes()).hexdigest()
    for root,z in zip(roots,(zt,zp,synth('linear'))):
        np.savez(root/'llever.npz',**z)
        (root/'meta.json').write_text(json.dumps(dict(status='complete',cfg=dict(T=48),
            gates=dict(G1=[0.,0.]),push_file_sha256=sha)))
    with contextlib.redirect_stdout(io.StringIO()): an.main(*roots)
    result=json.loads((roots[2]/'analysis_summary.json').read_text())
    ok(result['verdict']=='HUGINN PATTERN REPLICATES','analysis file IO and JSON round trip')
    expected=float(np.diagonal(full,axis1=1,axis2=2).mean()*.03)
    ok(abs(result['doses']['0.03']['P1']['mean_target_effect']['L']-expected)<1e-14,
       'target-effect axes and problem averaging')
ze=synth('linear'); ze['Z'][0,1,0,0]=np.nan
result=an.analyze(zt,zp,ze,metas,dict(P=P,m_ref=np.array(1.)))
ok(result['verdict']=='INCONCLUSIVE' and json.dumps(an.json_clean(result),allow_nan=False), 'nonfinite gate emits valid JSON')
ze=synth('linear'); ze['times']=np.append(ze['times'],96)
ze['Z']=np.concatenate([ze['Z'],ze['Z'][:,:,-1:]],axis=2)
result=an.analyze(zt,zp,ze,metas,dict(P=P,m_ref=np.array(1.)))
ok(result['verdict']=='HUGINN PATTERN REPLICATES' and result['primary']['t']==96 and result['doses']['0.03']['prediction_time']==96,
   'T96: primary time is the endpoint (design Draft 1.1)')
try:
    app.check_deadline(time.monotonic()-1)
    ok(False,'expired deadline')
except TimeoutError: ok(True,'internal deadline raises')
print(f'LOCAL TEST PASSED ({checks} checks)')
