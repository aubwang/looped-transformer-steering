"""Offline CPU regression tests. Never writes the production prompt file."""
import sys
sys.dont_write_bytecode = True
import copy
import io
import json
import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch
import numpy as np
import torch
import analyze_flicker as a
with patch.dict(sys.modules, {'modal': None}):
    import flicker_app as app

checks = 0
def ok(value, name):
    global checks
    assert value, name
    checks += 1
    print('PASS', name)

prompt_path = a.ROOT/'results/flicker_prompts.json'
before = prompt_path.read_bytes()
prompts = json.loads(before)
with patch.object(Path, 'write_text', side_effect=AssertionError('test must not write prompts')):
    ok(a.build_prompts(a.ROOT) == prompts, 'pure builder equals checked-in 44 prompts')

top = np.zeros((64, 1, 3), int); top[:, 0] = [[10+b%2, 11-b%2, 12] for b in range(449,513)]
d = np.full((512,1), .001); lag = np.full((64,1), .001)
ok(a.classify(top,d,lag)[0] == 'FLICKER', 'settled strict alternation')
for kind in ('constant','repeat','three','low_drift','lag','halves'):
    t, dd, ll = top.copy(), d.copy(), lag.copy()
    if kind == 'constant': t[:,0,0] = 10
    if kind == 'repeat': t[3,0,0] = t[2,0,0]
    if kind == 'three': t[3,0,0] = 13
    if kind == 'low_drift': dd *= .01
    if kind == 'lag': ll *= 100
    if kind == 'halves': dd[-32:] *= 1.05
    ok(a.classify(t,dd,ll)[0] == ('NONE' if kind == 'constant' else 'UNRESOLVED'), kind)

# Mixed lengths and authoritative prediction positions.
z = {k: [] for k in ('uid','source','ids','pred_pos','steps','drift','lag2_tail','top_ids','top_lp','repro_ok','repro_mismatches','ref_preflight_ok','tok_str','top_tok_str','digit_argmax_last','full_vocab_agrees')}
z.update(K=512,tail=64)
for i,p in enumerate(prompts):
    m = len(p['digit_info']); strings = ['x']*(1+i%3)+['1','+','2','=',' 3','\n','5','=',' 1']+['2']*(m-1)
    n = len(strings); pp,_ = a.parsed_answers(strings)
    row = dict(uid=p['uid'],source=p['source'],ids=list(range(n)),pred_pos=pp,steps=512,
        drift=np.tile(d,(1,n)),lag2_tail=np.tile(lag,(1,n)),top_ids=np.tile(top,(1,n,1)),
        top_lp=np.tile(np.array([[[-.1,-.12,-.2]]]),(64,n,1)),repro_ok=True,repro_mismatches=0,
        ref_preflight_ok=p['source']=='census',tok_str=strings,top_tok_str={},digit_argmax_last=[1]*m,full_vocab_agrees=[False]*m)
    for k in row: z[k].append(row[k])
meta = dict(ok=True, validation=[dict(argmax_agree=1,max_abs=0)],cfg=dict(reproduce=True),run_id='scan-test',model_id='model',revision='rev')
ok(a.scan_gates(z,meta)['PASS'], 'complete mixed-length scan passes; digit agreement independent of full vocabulary')
for kind in ('empty','subset','tail','reference','steps','nan','role','source','ok'):
    zz,mm = copy.deepcopy(z),copy.deepcopy(meta)
    if kind in ('empty','subset'):
        for k,v in zz.items():
            if isinstance(v,list): zz[k] = v[:0 if kind=='empty' else 43]
    elif kind=='tail': zz['tail']=63
    elif kind=='reference': zz['ref_preflight_ok'][0]=False
    elif kind=='steps': zz['steps'][0]=511
    elif kind=='nan': zz['top_lp'][0][0,0,0]=np.nan
    elif kind=='role': zz['pred_pos'][0]=[0]*len(zz['pred_pos'][0])
    elif kind=='source': zz['source'][0]='screen'
    else: mm['ok']=False
    rr=a.analyze_scan(zz,mm)
    ok(not rr['gates']['PASS'] and not rr['hits'], kind+' fails gates')
rr=a.roles(dict(ids=list(range(10)),tok_str=['1','=',' nope',' 2','\n','2','=',' 4','5','6'],pred_pos=[6,7,8]))
ok(rr[1]=='other' and rr[2]=='demo answer' and rr[6:9]==['final answer']*3 and rr[-1]=='other','digit-only demo and last-position roles')
ok(a.roles(dict(ids=[0,1,2],tok_str=['=','3','x'],pred_pos=[1]))[1]=='final answer','pred_pos authoritative even when parse disagrees')
summary=a.analyze_scan(z,meta)
ok(len(summary['hits'])==4 and all(h['scan_run']=='scan-test' and len(h['scan_pair_lp'])==8 for h in summary['hits']), 'hit provenance and pair LP export')
zz=copy.deepcopy(z)
for t in zz['top_ids']: t[:,:,0]=10
ok(a.analyze_scan(zz,meta)['outcome']=='NO FLICKER POSITION','zero flicker outcome')

hit=dict(uid=1,p=1,ids=[2,3,4],n_full=3,tokens=[10,11],scan_run='scan-test',model_id='model',revision='rev',
         ids_sha256=hashlib.sha256(b'[2,3,4]').hexdigest(),smallest_lead=.01,
         scan_top1={str(b):10+b%2 for b in range(509,513)},
         scan_pair_lp={str(b):[-.1,-.11] for b in range(505,513)})
hits=[hit]
def readout(b,reverse=False):
    return dict(budget=b,top_ids=[10+(b+int(reverse))%2,11-(b+int(reverse))%2,12],top_lp=[-.1,-.11,-.2],pair_lp=[-.1,-.11])
row=dict(hit,C1=[readout(b) for b in range(509,513)],C2=[dict(seed=seed,budgets=list(range(505,513)),
    top_ids=[10+b%2 for b in range(505,513)],top_lp=[[-.1,-.11]]*8,pair_lp=[[-.1,-.11] for _ in range(8)]) for seed in range(5)],
    C3=[dict(text='a'),dict(text='b')])
cm=dict(model_id='model',revision='rev',precision='float32',start_policy=app.START_POLICY,hits_sha256=app.hits_sha(hits),cfg=dict(K=512,budgets=[509,510,511,512],tail_c2=8,seeds=list(range(5))))
conf=dict(meta=cm,hits=[row])
f64=dict(meta=dict(cm,precision='float64 patched',patch=dict(norm_modules=1,norm_source_sha256='a',rotary_source_sha256='b',dtypes={'logits':'torch.float64'}),activation=dict(max_abs_difference=1e-6,max_abs_value=1)),hits=[dict(hit,C4=[readout(b,True) for b in range(505,513)])])
def result(c=conf,f=f64,hh=hits): return a.analyze_confirmation(c,hh,f)
ok(result()['hits'][0]['confirmed'],'C4 phase reversal passes')
ok(result()['hits'][0]['C2'][0]['token_at_512']==10 and result()['hits'][0]['C3'][1]['text']=='b','C2 starts and C3 texts reported')
for kind in ('ids','budgets','sha','precision','patch','c64_ids','c64_budgets','zero_lead','repeat'):
    c,f=copy.deepcopy(conf),copy.deepcopy(f64)
    if kind=='ids': c['hits'][0]['ids']=[9]
    elif kind=='budgets': c['meta']['cfg']['budgets']=[508,509,510,511]
    elif kind=='sha': c['meta']['hits_sha256']='bad'
    elif kind=='precision': c['meta']['precision']='float64'
    elif kind=='patch': del f['meta']['patch']
    elif kind=='c64_ids': f['hits'][0]['ids']=[9]
    elif kind=='c64_budgets': f['hits'][0]['C4'][0]['budget']=1
    elif kind=='zero_lead': c['hits'][0]['C1'][0]['top_lp']=[-.1,-.1,-.2]
    else: f['hits'][0]['C4'][0]['top_ids']=f['hits'][0]['C4'][1]['top_ids']
    ok(not result(c,f)['hits'][0]['confirmed'],kind+' rejected')
for lead,delta,label in [(5e-5,0,'NUMERICALLY UNRESOLVED'),(.01,.002,'NUMERICALLY UNRESOLVED'),(.01,.0002,'CONFIRMED')]:
    hh=copy.deepcopy(hits);hh[0]['smallest_lead']=lead
    c,f=copy.deepcopy(conf),copy.deepcopy(f64)
    for lp in c['hits'][0]['C2'][0]['pair_lp']: lp[0]+=delta
    for obj in (c,f): obj['meta']['hits_sha256']=app.hits_sha(hh)
    r=result(c,f,hh)['hits'][0]
    ok(abs(r['delta']-delta)<1e-12 and r['label']==label,f'guard lead={lead} delta={delta}')
ok(result(f=None)['hits'][0]['label']=='CONFIRMATION INCOMPLETE','missing C4 incomplete')
c=copy.deepcopy(conf);c['hits'][0]['C1'][0]['pair_lp']=[]
ok(result(c)['hits'][0]['label']=='NUMERICALLY UNRESOLVED','missing pair gap fails guard')

sm=dict(ok=True,code_sha256='sha',load_s=10,loop_s=.1,tail_loop_s=.2)
for label,m,expected in [('missing',None,False),('sha',dict(sm,code_sha256='bad'),False),('not ok',dict(sm,ok=False),False),('slow',dict(sm,loop_s=1),False),('pass',sm,True)]:
    ok(app.production_allowed('scan',m,'sha')[0]==expected,'production '+label)
ok(app.deadline_at(10,100,480,340)==440 and app.deadline_at(10,200,480,340)==445,'deadline min and reserve')
ok(app.canonical_hits(hits)==a.canonical_hits(json.dumps(hits,indent=2)),'one canonical wire representation')

# A fake defining module and literal source on disk let inspect audit both casts.
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'tiny_double.py'
    path.write_text('''import torch
class RMSNorm(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.eps=1e-6; self.weight=torch.nn.Parameter(torch.ones(4,dtype=torch.float64))
    def forward(self,x):
        return (x.float()*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+self.eps)).type_as(x)*self.weight

def apply_rotary_emb_complex_like(q,k,freqs_cis):
    x=torch.cat([q,k],dim=2).unflatten(-1,(-1,2)).float()
    f=freqs_cis
    return torch.split(torch.stack([x[...,0]*f[...,0]-x[...,1]*f[...,1],x[...,1]*f[...,0]+x[...,0]*f[...,1]],-1).flatten(3).type_as(q),q.shape[2],dim=2)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.norm=RMSNorm()
''')
    import importlib.util
    spec=importlib.util.spec_from_file_location('tiny_double',path);mod=importlib.util.module_from_spec(spec);sys.modules['tiny_double']=mod;spec.loader.exec_module(mod)
    model=mod.Model(); x=torch.tensor([[1.123456789,2.3,3.4,4.5]],dtype=torch.float64)
    original=model.norm(x); float_original=model.norm(x.float())
    q=x.reshape(1,1,1,4); k=q*.3; f=torch.tensor([[[[[.9,.2],[.8,.3]]]]]).reshape(1,1,1,2,2)
    old_rot=mod.apply_rotary_emb_complex_like(q.float(),k.float(),f)
    evidence=app.patch_double(SimpleNamespace(model=model))
    norm=model.norm(x)
    ok(norm.dtype==torch.float64 and torch.equal(norm,x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+1e-6)*model.norm.weight),'double norm exact formula')
    qr,kr=mod.apply_rotary_emb_complex_like(q,k,f)
    xx=torch.cat([q,k],2).unflatten(-1,(-1,2));ff=f.double()
    want=torch.stack([xx[...,0]*ff[...,0]-xx[...,1]*ff[...,1],xx[...,1]*ff[...,0]+xx[...,0]*ff[...,1]],-1).flatten(3)
    ok(qr.dtype==kr.dtype==torch.float64 and torch.equal(torch.cat([qr,kr],2),want),'double rotary exact formula with original constants')
    ok(torch.equal(model.norm(x.float()),float_original) and all(torch.equal(u,v) for u,v in zip(mod.apply_rotary_emb_complex_like(q.float(),k.float(),f),old_rot)),'float32 original paths retained')
    ok(not torch.equal(original,norm) and evidence['norm_modules']==1 and len(evidence['rotary_source_sha256'])==64,'patch activation and hashes')

class FakeModel(torch.nn.Module):
    def __init__(self): super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1))
    def embed_inputs(self,ids): return torch.zeros((*ids.shape,2)),None
    def initialize_state(self,e): return torch.zeros_like(e)
    def forward(self,ids,input_states,num_steps):
        logits=torch.zeros((*ids.shape,20));logits[:,:,num_steps%2]=2
        return SimpleNamespace(logits=logits)
class FakeH:
    def __init__(self,*args,**kwargs):
        self.model=FakeModel();self.dev='cpu';self.d=2;self.NL=5;self.SPD=self.DIG=list(range(10));self.tok=SimpleNamespace(decode=lambda ids:''.join(str(t) for t in ids))
    def prep(self,problems):
        for p in problems:p.update(ids=[2,3,4],pred_pos=[1])
    def validate(self,problems):return [dict(argmax_agree=1,max_abs=0)]
    def init_state(self,e,uid,seed):return torch.zeros_like(e)
    def batch(self,problems,seeds):
        ids=torch.tensor([p['ids'] for p in problems]);e,_=self.model.embed_inputs(ids);return ids,e,e.clone()
    def step(self,e,s,k):return s+1
    def coda_logits(self,s,g):
        logits=torch.zeros((*g.shape,20));logits[:,:,int(s[0,0,0])%2]=2;return logits
app.H=FakeH
app.metadata=lambda cfg,began,tc,validation=None,**kw:dict(cfg=cfg,validation=validation,**kw)
real_save=app.save_result
app.save_result=lambda result,*args,**kw:result
fake_prompt=dict(uid=1,source='screen',ref_answer_argmax_last=[0])
cfg=dict(K=8,tail=4,compute_s=10,reproduce=False,preflight=False)
s=app.scan(json.dumps([fake_prompt]),cfg,'fake')
with np.load(io.BytesIO(s['npz']),allow_pickle=True) as arr:
    ok(arr['steps'][0]==8 and arr['top_ids'][0].shape==(4,3,3),'CPU scan complete shapes')
    ok(arr['digit_argmax_last'][0]==[0] and s['meta']['loop_s']>=0 and s['meta']['tail_loop_s']>=0,'digit readout and loop timing')
s=app.scan(json.dumps([fake_prompt]),dict(cfg,compute_s=-1),'fake')
ok(not s['meta']['ok'] and 'incomplete' in s['meta']['flags'],'scan deadline preserves result')
fc=dict(K=8,budgets=[5,6,7,8],seeds=[0,1,2,3,4],tail_c2=4,n_new=2,compute_s=10)
fh=dict(uid=1,p=1,ids=[2,3,4],n_full=3,tokens=[0,1])
r=app.confirm(app.canonical_hits([fh]),fc,'fake')
ok(r['meta']['ok'] and len(r['hits'][0]['C1'])==4 and len(r['hits'][0]['C2'])==5,'CPU C1 and five batched seeds')
ok(r['hits'][0]['C1'][0]['pair_lp'] and r['hits'][0]['C2'][0]['pair_lp'],'pair LP saved')
ok([r['token_ids'] for r in r['hits'][0]['C3']]==[[1,1],[0,0]],'C3 branches diverge independently')
class FailingH(FakeH):
    def step(self,e,s,k): raise RuntimeError('later stage failure')
app.H=FailingH
r=app.confirm(app.canonical_hits([fh]),fc,'fake')
ok(len(r['hits'][0]['C1'])==4 and not r['meta']['ok'],'C2/C3 failure preserves C1')
app.H=FakeH
with patch.object(Path,'mkdir',side_effect=OSError('volume unavailable')):
    payload=dict(meta=dict(ok=True,flags=[]),hits=[])
    ok(real_save(payload,'fake') is payload and not payload['meta']['ok'] and 'volume_error' in payload['meta'],'failed volume write returns payload')
# Explicit references, duplicate UID and unreadable shard all fail preflight.
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)
    for j,run in enumerate(app.CENSUS_RUNS):
        path=root/run;path.mkdir()
        np.savez(path/'census.npz',uid=[j+1],ids=[[2,3,4]],pred_pos=[[1]],steps=[512],ro_argmax=np.zeros((1,64,1),int))
    pp=[dict(uid=1,source='census',ids=[2,3,4],pred_pos=[1])]
    refs,hashes,pre,errors=app.load_references(pp,root)
    ok(pre[1] and len(hashes)==4 and not errors,'reference preflight manifest and hashes')
    np.savez(root/app.CENSUS_RUNS[1]/'census.npz',uid=[1],ids=[[2,3,4]],pred_pos=[[1]],steps=[512],ro_argmax=np.zeros((1,64,1),int))
    ok(bool(app.load_references(pp,root)[3]),'duplicate reference uid fails')
    (root/app.CENSUS_RUNS[2]/'census.npz').write_bytes(b'broken')
    ok(bool(app.load_references(pp,root)[3]),'unreadable reference fails even with available uid')
    ok(not app.load_references([dict(pp[0],uid=99)],root)[2][99],'missing census uid fails')
for kind in ('model','revision','start','double_sha','double_precision','double_zero_lead'):
    c,f=copy.deepcopy(conf),copy.deepcopy(f64)
    if kind in ('model','revision'): c['meta']['model_id' if kind=='model' else 'revision']='wrong'
    elif kind=='start': c['meta']['start_policy']='wrong'
    elif kind=='double_sha': f['meta']['hits_sha256']='wrong'
    elif kind=='double_precision': f['meta']['precision']='float64'
    else: f['hits'][0]['C4'][0]['top_lp']=[-.1,-.1,-.2]
    ok(not result(c,f)['hits'][0]['confirmed'],kind+' confirmation rejection')
c=copy.deepcopy(conf)
c['hits'][0]['C1'][0]['pair_lp']=[-.102,-.11]
c['hits'][0]['C2'][0]['pair_lp'][4]=[-.098,-.11]
ok(abs(result(c)['hits'][0]['delta']-.004)<1e-12,'delta spans all three computations')
# Admission uses the entry deadline even when model setup consumed the allowance.
r=app.confirm(app.canonical_hits([fh]),dict(fc,compute_s=-1),'fake')
ok(not r['meta']['ok'] and not r['hits'][0]['C1'],'C1 deadline denies indivisible forward')
# Genuine local head keeps float64 rather than using H.coda_logits.
head = SimpleNamespace(ln_f=torch.nn.Identity(),model=SimpleNamespace(
    freqs_cis=torch.ones(1,4,1),transformer=SimpleNamespace(coda=[]),lm_head=torch.nn.Linear(2,4).double()))
lg=app.coda_logits64(head,torch.ones(1,3,2,dtype=torch.float64),torch.tensor([[2]]))
ok(lg.dtype==torch.float64 and lg.shape==(1,1,4),'local coda readout retains float64')
# Exercise C4 orchestration separately from the audited real patch above.
class DoubleH(FakeH):
    def __init__(self,*args,**kw):
        super().__init__(); self.model.double(); self.patched=False
    def step(self,e,s,k): return s+1+(1e-7 if self.patched else 0)
def fake_patch(h):
    h.patched=True
    return dict(norm_modules=1,dtypes={},norm_source_sha256='test',rotary_source_sha256='test')
def fake_head(h,s,g): return h.coda_logits(s,g).double()
app.H=DoubleH
with patch.object(app,'patch_double',side_effect=fake_patch), patch.object(app,'coda_logits64',side_effect=fake_head):
    r=app.confirm64(app.canonical_hits([fh]),fc,'fake')
ok([r['budget'] for r in r['hits'][0]['C4']]==list(range(1,9)) and r['meta']['hits_sha256']==app.hits_sha([fh]),'C4 budgets and binding')
class NonDigitH(FakeH):
    def coda_logits(self,s,g):
        logits=super().coda_logits(s,g)
        logits[:,:,19]=10
        return logits
app.H=NonDigitH
r=app.scan(json.dumps([fake_prompt]),dict(K=512,tail=64,compute_s=10,reproduce=True,preflight=False),'fake')
with np.load(io.BytesIO(r['npz']),allow_pickle=True) as arr:
    ok(arr['repro_ok'][0] and arr['digit_argmax_last'][0]==[0] and arr['full_vocab_agrees'][0]==[False],
       'production digit reproduction passes despite non-digit vocabulary winner')
class LoadFailure:
    def __init__(self,*args,**kw): raise RuntimeError('load failure')
app.H=LoadFailure
for function in (app.confirm,app.confirm64):
    r=function(app.canonical_hits([fh]),fc,'fake')
    ok(not r['meta']['ok'] and r['meta']['flags'],'setup failure returned by '+function.__name__)
ok(prompt_path.read_bytes()==before,'test did not write production prompts')
# Flicker2 deterministic manifest and chunk analysis, entirely offline.
import make_flicker2_prompts as builder
p2 = builder.build()
with patch.object(Path, 'write_text', side_effect=AssertionError('builder must be pure')):
    ok(p2 == builder.build() == json.loads((a.ROOT/'results/flicker2_screen_prompts.json').read_text()),
       'flicker2 builder deterministic and equals checked-in file without writing')
ok(len(p2)==400 and len({p['uid'] for p in p2})==400 and len({(p['a'],p['b']) for p in p2})==400,
   'flicker2 400 unique prompts and pairs')
excluded={(p['a'],p['b']) for name in ('census_prompts_all.json','screen_prompts.json')
          for p in json.loads((a.ROOT/'results'/name).read_text())}
ok(not excluded.intersection((p['a'],p['b']) for p in p2),'flicker2 excludes earlier final sums')
for i,c in enumerate(builder.CONTEXTS):
    group=p2[100*i:100*(i+1)];ctx=next(p for p in prompts if p['uid']==c)
    demo=ctx['prompt'][:ctx['prompt'].rfind('\n')+1]
    ok(all(p['context_uid']==c and p['uid']==30000+100*i+j and p['pid']==3000+100*i+j and
           len(str(p['a']))==len(str(p['b']))==(3 if j<50 else 4) and
           p['prompt']==demo+f"{p['a']} + {p['b']} =" and p['answer']==f" {p['a']+p['b']}"
           for j,p in enumerate(group)),f'flicker2 context {c} order lengths and text')

def fake_chunks(root, manifest, K, tail, size):
    data={k:[] for k in ('uid','source','ids','pred_pos','steps','drift','lag2_tail','top_ids','top_lp','tok_str','top_tok_str')}
    for p in manifest:
        n=50; strings=['x']*n
        strings[2:5]=['=','1','\n'];strings[46:50]=['=','1','2','x']
        ids=np.arange(n); ids[47:49]=[10,10]
        ti=np.full((tail,n,3),12);ti[:,:,0]=10;ti[:,:,1]=11
        row=dict(uid=p['uid'],source=p['source'],ids=ids,pred_pos=[46,47],steps=K,
            drift=np.full((K,n),.001),lag2_tail=np.full((tail,n),.001),top_ids=ti,
            top_lp=np.tile([-.1,-.2,-.3],(tail,n,1)),tok_str=strings,top_tok_str={'10':'1','11':'2','20':'x','21':'y'})
        for k in data:data[k].append(row[k])
    return data

def write_chunks(root,data,K,tail,size):
    runids=[]
    for i,start in enumerate(range(0,len(data['uid']),size)):
        part={}
        for k,v in data.items():
            arr=np.empty(len(v[start:start+size]),object);arr[:]=v[start:start+size];part[k]=arr
        np.savez(root/f'scan_{i}.npz',K=K,tail=tail,**part)
        runids.append(f'chunk-{i}')
        (root/f'meta_{i}.json').write_text(json.dumps(dict(ok=True,run_id=runids[-1],code_sha256='sha',
            validation=[dict(argmax_agree=1,max_abs=0)],model_id='model',revision='rev',cfg=dict(K=K,tail=tail))))
    (root/'meta.json').write_text(json.dumps(dict(ok=True,chunk_run_ids=runids,code_sha256='sha')))

with tempfile.TemporaryDirectory() as tmp, patch.object(a, 'stage_provenance', return_value={}), patch.object(builder, 'build', side_effect=lambda:mf), patch.object(a, 'selection_gates', return_value={}):
    root=Path(tmp)
    mf=[dict(p,digit_info=[{},{}]) for p in p2]
    data=fake_chunks(root,mf,32,8,200)
    # Equal smallest statistics must sort by UID, independent of input sorting implementation.
    for i,lp in enumerate(data['top_lp']):lp[:,46,1]=-.1-(.001 if i in (8,9) else .1+i*.0001)
    write_chunks(root,data,32,8,200)
    summary, selected=a.analyze_screen2(root,mf)
    ok(summary['gates']['PASS'] and len(selected)==88 and [p['uid'] for p in selected[:5]]==[20178,20147,20146,30008,30009],
       'screen selection controls first 85 ranked with UID tie break')
    ok([p['screen_rank'] for p in selected[3:]]==list(range(1,86)),'screen ranks one-based')
    (root/'scan_1.npz').rename(root/'missing.npz')
    ok(not a.analyze_screen2(root,mf)[0]['gates']['PASS'],'screen missing chunk gate')
    (root/'missing.npz').rename(root/'scan_1.npz')
    data['uid'][0],data['uid'][1]=data['uid'][1],data['uid'][0]
    write_chunks(root,data,32,8,200)
    ok(not a.analyze_screen2(root,mf)[0]['gates']['PASS'],'screen wrong order gate')
    write_chunks(root,data,31,8,200)
    ok(not a.analyze_screen2(root,mf)[0]['gates']['configuration'],'screen wrong K gate')

with tempfile.TemporaryDirectory() as tmp, patch.object(a, 'stage_provenance', return_value={}), patch.object(builder, 'build', side_effect=lambda:mf), patch.object(a, 'selection_gates', return_value={}):
    root=Path(tmp)
    mf=[dict(p,digit_info=[{},{}]) for p in selected]
    data=fake_chunks(root,mf,512,64,44)
    refs={'flicker':[]}
    def flip(i,p,tokens=(10,11),lead=.1,swing=.02):
        for b in range(64):
            t=tokens[b%2];other=tokens[1-b%2]
            data['top_ids'][i][b,p]=[t,other,99]
            data['top_lp'][i][b,p]=[-.1,-.1-lead-(swing if b%2 else 0),-2]
    for i,pos in enumerate((40,41,44)):
        flip(i,pos);refs['flicker'].append(dict(uid=mf[i]['uid'],p=pos,tokens=[10,11]))
    flip(3,46,lead=.02);flip(4,47,lead=.04);flip(5,47,tokens=(20,21),lead=.5)
    flip(6,47,lead=.06);data['top_ids'][6][:,46,0]=11
    flip(7,47,lead=.08)
    flip(3,2);flip(4,2)
    # Force a shared context for the demo deduplication check.
    mf[4]['context_uid']=mf[3]['context_uid']
    write_chunks(root,data,512,64,44)
    out=a.analyze_scan2(root,mf,refs)
    ok(out['gates']['PASS'],'scan2 positive controls pass')
    ah={r['uid']:r for r in out['answer_hits']}
    ok(ah[mf[3]['uid']]['sub_label']=='RIGHT/WRONG DIGIT' and ah[mf[5]['uid']]['sub_label']=='WRONG/WRONG' and
       ah[mf[3]['uid']]['right_parity']=='odd','answer labels and budget parity')
    ok(ah[mf[4]['uid']]['earlier_digits_right'] and not ah[mf[6]['uid']]['earlier_digits_right'],
       'earlier digits right true and false')
    ok(len(out['demo_answer_hits'])==1 and len(out['demo_answer_hits'][0]['prompts'])==2,'demo flicker deduplicated per context')
    ok([h['uid'] for h in out['hits']]==[mf[i]['uid'] for i in (7,6,4,3)],'hit selection right/wrong first then decreasing lead distinct prompts')
    ok('hits_continue' not in out, 'scan does not export runnable continuation')
    ok(not {(r['uid'],r['p']) for r in out['near_misses']} & {(r['uid'],r['p']) for r in out['answer_hits']}, 'near misses exclude hits')
    ok(all(c['share']==1 and c['swing']['min']==0 and c['swing']['max']>0 for c in out['carry_over'].values()
           if (c['swing']['max'] or 0)>0),'carry over settled shares and swing distribution')
    manifest_path=root/'manifest.json';manifest_path.write_text(json.dumps(mf))
    analyze = a.analyze_scan2
    with patch.object(a,'analyze_scan2',side_effect=lambda r,m,**kw:analyze(r,m,refs,**kw)), patch.object(sys,'argv',
            ['analyze_flicker.py','--scan2',str(root),'--manifest',str(manifest_path),'--screen',str(root)]), patch('sys.stdout',new=io.StringIO()):
        a.main()
    ok(not (root/'flicker_hits_continue.json').exists() and json.loads((root/'flicker_hits.json').read_text())==out['hits'], 'scan2 CLI writes hits only')
    flip(4,2,tokens=(20,21))
    data['drift'][5][:,47]=0
    write_chunks(root,data,512,64,44)
    described=a.analyze_scan2(root,mf,refs)
    ok(sorted(pair['count'] for pair in described['demo_answer_hits'][0]['pairs'])==[1,1],
       'shared demo position reports each observed pair and its count')
    context=str(mf[5]['context_uid'])
    ok(described['carry_over'][context]['settled_prompts']==described['carry_over'][context]['prompts']-1,
       'carry over requires every final answer position settled')
    data['drift'][5][:,47]=.001
    flip(7,47,tokens=(10,20),lead=.8)
    write_chunks(root,data,512,64,44)
    other=a.analyze_scan2(root,mf,refs)
    ok([h['uid'] for h in other['hits']]==[mf[i]['uid'] for i in (6,4,3,7)] and
       other['hits'][-1]['sublabel']=='RIGHT/OTHER', 'three-way selection priority overrides lead')
    for i in (3,4,6):data['top_ids'][i][:,:,0]=10
    write_chunks(root,data,512,64,44)
    ok(a.analyze_scan2(root,mf,refs)['outcome']=='ANSWER HITS FOUND (RIGHT/OTHER, NO RIGHT/WRONG DIGIT)',
       'right other only outcome wording')
    data['top_ids'][7][:,:,0]=10
    write_chunks(root,data,512,64,44)
    ok(a.analyze_scan2(root,mf,refs)['outcome']=='ANSWER HITS FOUND (WRONG/WRONG ONLY)', 'wrong wrong only outcome wording')
    data['top_ids'][5][:,:,0]=10
    write_chunks(root,data,512,64,44)
    ok(a.analyze_scan2(root,mf,refs)['outcome']=='NO ANSWER-POSITION FLICKER','no answer hit outcome wording')
    data['top_ids'][0][:,40,0]=10
    write_chunks(root,data,512,64,44)
    bad=a.analyze_scan2(root,mf,refs)
    ok(not bad['gates']['positive_controls'] and not bad['hits'],'control not FLICKER rejects all hits')
    flip(0,40,tokens=(10,13))
    write_chunks(root,data,512,64,44)
    ok(not a.analyze_scan2(root,mf,refs)['gates']['positive_controls'],'control wrong token pair rejects')

ok(app.production_allowed('scan',sm,'sha',24,8,200/44)[0] and
   not app.production_allowed('scan',dict(sm,loop_s=.2,tail_loop_s=.4),'sha',448,64,200/44)[0], 'production batch scale applies to both loop times')
app.H=FakeH
r=app.confirm(app.canonical_hits([fh]),dict(fc,parts=['C1','C2']),'fake')
ok(r['meta']['ok'] and not r['hits'][0]['C3'] and 'incomplete' not in r['meta']['flags'], 'C1 C2 only no C3 and no incomplete flag')
r=app.confirm(app.canonical_hits([fh]),dict(fc,parts=['C1','C2'],compute_s=-1),'fake')
ok(not r['meta']['ok'] and 'incomplete' in r['meta']['flags'],'C1 C2 only incomplete required parts detected')
c=copy.deepcopy(conf);c['meta']['cfg']['parts']=['C1','C2'];c['hits'][0].pop('C3')
ok(a.analyze_confirmation(c,hits,f64)['hits'][0]['confirmed'],'confirmation analysis accepts C1 C2 without C3')
app.H=FakeH
s=app.scan(json.dumps([dict(uid=30000,source='flicker2')]),cfg,'fake')
ok(s['meta']['ok'],'flicker2 scan needs no references with reproduction and preflight disabled')

# Local entrypoint batching, with remote calls replaced by an in-process recorder.
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp); promptfile=root/'prompts.json';promptfile.write_text(json.dumps(p2))
    smokedir=root/'flicker_scan_smoke-20261111-000000';smokedir.mkdir()
    sha=hashlib.sha256(Path(app.__file__).read_bytes()).hexdigest()
    (smokedir/'meta.json').write_text(json.dumps(dict(sm,code_sha256=sha)))
    calls=[]
    def remote(rows,cfg,run):
        calls.append((json.loads(rows),cfg,run))
        return dict(npz=b'fake',meta=dict(ok=True,run_id=run))
    with patch.object(app,'scan',SimpleNamespace(remote=remote)),patch('sys.stdout',new=io.StringIO()):
        app.run_scan2('screen2',str(promptfile),root)
    dest=next(root.glob('flicker2_screen-*'))
    parent=json.loads((dest/'meta.json').read_text())
    ok([len(c[0]) for c in calls]==[200,200] and all(c[1]==dict(K=32,tail=8,compute_s=200,reproduce=False,preflight=False) for c in calls)
       and parent['ok'] and parent['prompts_sha256']==hashlib.sha256(promptfile.read_bytes()).hexdigest(),
       'screen2 sequential chunks configuration and parent provenance')
    calls.clear();promptfile.write_text(json.dumps(p2[:88]))
    with patch.object(app,'scan',SimpleNamespace(remote=remote)),patch('sys.stdout',new=io.StringIO()):
        app.run_scan2('scan2',str(promptfile),root)
    ok([len(c[0]) for c in calls]==[44,44] and all(c[1]==dict(K=512,tail=64,compute_s=340,reproduce=False,preflight=False) for c in calls),
       'scan2 sequential chunks configuration')
# A full screen uses real prompt rows and decoded text, with no provenance mocks.
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);path=root/'manifest.json';path.write_text(json.dumps(p2))
    data={k:[] for k in ('uid','source','ids','pred_pos','steps','drift','lag2_tail','top_ids','top_lp','tok_str','top_tok_str')}
    for p in p2:
        strings=['<|begin_of_text|>']+list(p['prompt']+p['answer']);n=len(strings)
        row=dict(uid=p['uid'],source=p['source'],ids=list(range(n)),pred_pos=a.parsed_answers(strings)[0],
            steps=32,drift=np.ones((32,n)),lag2_tail=np.ones((8,n)),top_ids=np.tile([1,2,3],(8,n,1)),
            top_lp=np.tile([-.1,-.2,-.3],(8,n,1)),tok_str=strings,top_tok_str={})
        for k in data:data[k].append(row[k])
    write_chunks(root,data,32,8,200)
    for i in range(2):
        mp=root/f'meta_{i}.json';m=json.loads(mp.read_text());m['cfg']={'K':32,'tail':8};mp.write_text(json.dumps(m))
    mp=root/'meta.json';m=json.loads(mp.read_text());m['prompts_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();mp.write_text(json.dumps(m))
    ss,selected_real=a.analyze_screen2(root,p2,prompts_path=path)
    ok(ss['gates']['PASS'] and len(selected_real)==88,'full screen passes all provenance and numerical shape gates')
    m['prompts_sha256']='changed';mp.write_text(json.dumps(m))
    ss,sel=a.analyze_screen2(root,p2,prompts_path=path)
    ok(not sel and ss['outcome']=='SEARCH INVALID OR INCOMPLETE' and 'prompts_sha256' in ss['failed_gates'],
       'full screen hash gate fails closed')

# Independently exercise all provenance gates without the numerical-fixture mocks.
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp); path=root/'prompts.json'
    manifest=copy.deepcopy(p2)
    path.write_text(json.dumps(manifest))
    parent=dict(prompts_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    metas=[dict(ok=True,model_id='model',revision='rev',code_sha256='sha',cfg={'K':32}) for _ in range(2)]
    chunks=[dict(uid=[p['uid'] for p in manifest[i:i+200]]) for i in (0,200)]
    zz=dict(tok_str=[['<|begin_of_text|>',p['prompt'],p['answer']] for p in manifest])
    def provenance(): return a.stage_provenance(root,manifest,zz,chunks,metas,parent,200,path)
    ok(all(provenance().values()),'real manifest provenance passes')
    for key in ('revision','model_id','code_sha256','cfg','ok'):
        saved=metas[1][key];metas[1][key]='changed'
        ok(not provenance()['chunk_consistency'],'mismatched chunk '+key)
        metas[1][key]=saved
    parent['prompts_sha256']='wrong'
    ok(not provenance()['prompts_sha256'],'prompt file byte hash mismatch')
    parent['prompts_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    chunks[0]['uid'][0]=-1
    ok(not provenance()['chunk_uids'],'chunk uid differs from manifest slice')
    chunks[0]['uid'][0]=manifest[0]['uid']
    zz['tok_str'][0]=['<|begin_of_text|>changed']
    ok(not provenance()['text_spelling'],'changed text same uid fails spelling')
    zz['tok_str'][0]=['<|alternate_marker|>'+manifest[0]['prompt'],manifest[0]['answer']]
    ok(provenance()['text_spelling'],'one leading marker inside first token stripped')
    zz['tok_str'][0]=[manifest[0]['prompt'],manifest[0]['answer']]
    ok(provenance()['text_spelling'],'no leading marker uses full join')
    manifest[0]['context_uid']=-1
    # Even a matching replacement hash cannot bypass the deterministic builder gate.
    path.write_text(json.dumps(manifest));parent['prompts_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with patch.object(a,'load_chunks',return_value=({},[],{'PASS':True})):
        summary,_=a.analyze_screen2(root,manifest,prompts_path=path)
    ok(not summary['gates']['builder_manifest'],'changed context rejected by builder')
    with patch.object(a,'load_chunks',return_value=({},[],{'PASS':True})):
        changed=copy.deepcopy(p2);changed[0]['prompt']+='changed'
        summary,_=a.analyze_screen2(root,changed,prompts_path=path)
    ok(not summary['gates']['builder_manifest'],'changed prompt same uid rejected by builder')
    expected=prompts[:0]+[next(p for p in prompts if p['uid']==u) for u in (20178,20147,20146)]+p2[:85]
    with patch.object(a,'analyze_screen2',return_value=({'gates':{'PASS':True}},expected)):
        ok(all(a.selection_gates(expected,root).values()),'exact recomputed screen selection passes')
        wrong=copy.deepcopy(expected);wrong[-1]=p2[85]
        ok(not a.selection_gates(wrong,root)['screen_selection'],'different 85 selection rejected')
        wrong=copy.deepcopy(expected);wrong[3]['context_uid']=-1
        ok(not a.selection_gates(wrong,root)['screen_selection'],'selected context changed rejected')
        wrong=copy.deepcopy(expected);wrong[0]['prompt']+='changed'
        ok(not a.selection_gates(wrong,root)['control_manifest'],'control text must match exactly')

for right,flags,label in [(True,[True,True],'RIGHT/WRONG DIGIT'),(True,[True,False],'RIGHT/OTHER'),
                          (False,[True,True],'WRONG/WRONG'),(False,[False,False],'WRONG/WRONG')]:
    ok(a.answer_sublabel(right,flags)==label,'digit sublabel '+str((right,flags)))
for token,want in [(' 3',True),('9 ',True),('33',False),('x',False),('\n',False)]:
    ok(a.is_digit_token(token)==want,'digit token '+repr(token))

# Fixed-pair gaps retain sign and omit budgets where either token drops out.
ti=np.array([[[2,1,3]],[[4,3,2]],[[1,2,3]],[[2,1,3]]])
lp=np.array([[[-.1,-.3,-.9]],[[-.1,-.2,-.3]],[[-.1,-.5,-.9]],[[-.1,-.7,-.9]]])
pair,even,odd,swing=a.pair_statistics(ti,lp,0,509)
ok(pair==[2,1] and abs(even-.6)<1e-12 and abs(odd+.1)<1e-12 and abs(swing-.7)<1e-12,
   'fixed pair signed gap parity mean ignores dropout')
ti[0,0]=ti[2,0]=[4,3,2]
ok(np.isnan(a.pair_statistics(ti,lp,0,509)[-1]),'swing NaN when a parity has no saved pair')

newhits=[dict(hit,uid=i,first_answer_pos=0,a=1,b=2,c=3) for i in range(5)]
labels={'hits':[dict(label=x) for x in ['NOT CONFIRMED','CONFIRMED','CONFIRMED','CONFIRMED','CONFIRMED']]}
ch=a.continuation_hits(newhits,labels)
ok([h['uid'] for h in ch]==[1,2] and all(h['p']==0 and not h['tokens'] and not h['scan_top1'] and not h['scan_pair_lp'] for h in ch),
   'only first two confirmed hits exported in selection order')
with tempfile.TemporaryDirectory() as tmp:
    a.write_continuation_hits(tmp,newhits,labels)
    ok(json.loads((Path(tmp)/'flicker_hits_continue.json').read_text())==ch,'confirmation continuation file written')
    a.write_continuation_hits(tmp,newhits,{'hits':[{'label':'NOT CONFIRMED'}]*5})
    ok(json.loads((Path(tmp)/'flicker_hits_continue.json').read_text())==[],'no confirmed hits exports empty continuation')

with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);cr=root/'confirm';cr.mkdir();dr=root/'double';dr.mkdir()
    hh=[dict(hit,first_answer_pos=0,a=1,b=2,c=3)]
    cc,dd=copy.deepcopy(conf),copy.deepcopy(f64)
    for payload in (cc,dd):
        payload['meta']['hits_sha256']=a.hits_sha(hh)
        payload['hits'][0].update(hh[0])
    (cr/'confirm.json').write_text(json.dumps(cc));(dr/'confirm.json').write_text(json.dumps(dd))
    hp=root/'hits.json';hp.write_text(json.dumps(hh))
    argv=['analyze_flicker.py',str(cr),'--hits',str(hp)]
    with patch.object(sys,'argv',argv),patch('sys.stdout',new=io.StringIO()):a.main()
    ok(not (cr/'flicker_hits_continue.json').exists(),'new confirmation without c64 writes no continuation')
    with patch.object(sys,'argv',argv+['--c64',str(dr)]),patch('sys.stdout',new=io.StringIO()):a.main()
    ok(json.loads((cr/'flicker_hits_continue.json').read_text())==a.continuation_hits(hh, a.analyze_confirmation(cc,hh,dd)),
       'confirmation CLI with c64 writes filtered continuation')

for texts,statuses,flag in [((' 3\nignored',' 4\n'),['CORRECT','INCORRECT'],True),
                           (('3','x\n'),['TRUNCATED','MALFORMED'],False),
                           (('\n',' 03 \n'),['MALFORMED','CORRECT'],False),
                           (('4\n','3\n'),['INCORRECT','CORRECT'],True)]:
    cont=dict(meta=dict(ok=True,hits_sha256=a.hits_sha(ch),cfg=dict(K=512,n_new=7,parts=['C3'])),
              hits=[dict(h,C3=[dict(budget=b,text=t,token_ids=list(range(7))) for b,t in zip((511,512),texts)]) for h in ch])
    out=a.analyze_continue2(cont,ch)
    ok(out['gates']['PASS'] and [b['status'] for b in out['hits'][0]['branches']]==statuses and
       out['hits'][0]['right_at_one_parity_wrong_at_other']==flag,'continuation text policy '+repr(texts))
    ok(out['hits'][0]['branches'][0]['text']==texts[0] and out['hits'][0]['branches'][0]['token_ids']==list(range(7)),
       'continuation keeps raw text and token ids including intra-token newline')
for change in ('sha','row','ok','K','n_new','parts','branches'):
    bad=copy.deepcopy(cont)
    if change=='sha':bad['meta']['hits_sha256']='bad'
    elif change=='row':bad['hits'][0]['uid']=-1
    elif change=='ok':bad['meta']['ok']=False
    elif change=='branches':bad['hits'][0]['C3'][0]['budget']=510
    else:bad['meta']['cfg'][change]=None
    ok(not a.analyze_continue2(bad,ch)['gates']['PASS'],'continuation rejects '+change)

ok(not app.production_allowed('screen2',dict(sm,load_s=0,loop_s=1,tail_loop_s=1),'sha',150,20)[0],
   'screen own compute allowance denies 170 seconds even when total fits')
ok(not app.production_allowed('scan2',dict(sm,load_s=300,loop_s=.1,tail_loop_s=.1),'sha')[0],
   'total allowance denies even when loops fit')
ok(app.production_allowed('scan2',dict(sm,load_s=76,loop_s=.5,tail_loop_s=.5),'sha')[0],
   'both admission inequalities inclusive boundary')
for raises in (False,True):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);path=root/'prompts.json';path.write_text(json.dumps(p2))
        sd=root/'flicker_scan_smoke-0';sd.mkdir()
        sha=hashlib.sha256(Path(app.__file__).read_bytes()).hexdigest()
        (sd/'meta.json').write_text(json.dumps(dict(sm,code_sha256=sha)))
        calls=[]
        def fail_remote(*args):
            calls.append(args)
            if raises:raise RuntimeError('remote failure')
            return dict(npz=b'failed payload',meta=dict(ok=False,flags=['failed']))
        with patch.object(app,'scan',SimpleNamespace(remote=fail_remote)),patch('sys.stdout',new=io.StringIO()):
            try:app.run_scan2('screen2',str(path),root)
            except SystemExit:pass
            else:raise AssertionError('failed stage must exit nonzero')
        dest=next(root.glob('flicker2_screen-*'));parent=json.loads((dest/'meta.json').read_text())
        ok(len(calls)==1 and parent['ok'] is False and parent['failed_chunk']==0 and parent['failure_reason'],
           'first failed chunk stops stage '+str(raises))
        saved=json.loads((dest/'meta_0.json').read_text())
        ok(('remote failure' in saved.get('error','')) if raises else (dest/'scan_0.npz').read_bytes()==b'failed payload',
           'exception or failed payload saved '+str(raises))

# Historical analysis must remain byte-for-byte equivalent as JSON values.
oldscan=a.ROOT/'results/flicker_scan-20261111-022450'
oldconfirm=a.ROOT/'results/flicker_confirm-20261111-023055'
old64=a.ROOT/'results/flicker_confirm64-20261111-023903'
with np.load(oldscan/'scan.npz',allow_pickle=True) as zz:
    old=a.analyze_scan(zz,json.loads((oldscan/'meta.json').read_text()),oldscan.name)
ok(old==json.loads((oldscan/'flicker_summary.json').read_text()),'first experiment scan summary unchanged')
oldhits=json.loads((oldscan/'flicker_hits.json').read_text())
oldout=a.analyze_confirmation(json.loads((oldconfirm/'confirm.json').read_text()),oldhits,json.loads((old64/'confirm.json').read_text()))
ok(oldout==json.loads((oldconfirm/'confirm_summary.json').read_text()) and
   [r['label'] for r in oldout['hits']]==['CONFIRMED']*3,'first experiment confirmation unchanged: three CONFIRMED')
with tempfile.TemporaryDirectory() as tmp:
    with patch.object(sys,'argv',['analyze_flicker.py',str(oldconfirm),'--c64',str(old64),'--hits',str(oldscan/'flicker_hits.json'),'--output-dir',tmp]),patch('sys.stdout',new=io.StringIO()):
        a.main()
    ok(not (Path(tmp)/'flicker_hits_continue.json').exists(),'old confirmation CLI writes no continuation')
ok((a.ROOT/'results/flicker2_screen_prompts.json').read_bytes()==json.dumps(p2,indent=2).encode()+b'\n',
   'screen prompt bytes remain canonical and unchanged')
print(f'LOCAL TEST PASSED ({checks} checks)',flush=True)
