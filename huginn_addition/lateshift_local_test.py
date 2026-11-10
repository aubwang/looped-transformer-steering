"""Offline CPU checks: run with an environment containing torch, numpy, scipy, modal."""
import inspect
import re
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import mon_app as mon
import lateshift_app as app
import analyze_lateshift as analysis


def verify_demo(fam, text):
    if fam in ('add6','chain6','mul2add','paren'):
        expr,ans=text.rsplit(' = ',1); assert eval(expr)==int(ans)
    elif fam in ('var4','var4s'):
        lines=text.splitlines(); target,ans=lines.pop().split(': '); values={}
        while lines:
            progressed=False
            for line in lines[:]:
                name,expr=line.split(' = ')
                try: value=eval(expr,{},values)
                except NameError: continue
                values[name]=value; lines.remove(line); progressed=True
            assert progressed
        assert values[target[-1]]==int(ans)
    elif fam=='ptr3':
        mapping={int(a):int(b) for a,b in re.findall(r'f\((\d+)\) = (\d+)',text)}
        u,ans=map(int,re.search(r'f\(f\(f\((\d+)\)\)\) = (\d+)',text).groups())
        assert mapping[mapping[mapping[u]]]==ans
    elif fam=='swap':
        initial=re.findall(r'(Alice|Bob|Carol) has the (\w+) ball',text); state=dict(initial)
        for a,b in re.findall(r'(Alice|Bob|Carol) and (Alice|Bob|Carol) swap balls',text): state[a],state[b]=state[b],state[a]
        who,ans=re.search(r'Now (\w+) has the (\w+) ball',text).groups(); assert state[who]==ans
    elif fam=='count':
        numbers=list(map(int,text.splitlines()[0].split()[1:])); target,ans=map(int,re.search(r'does (\d+) appear\? Answer: (\d+)',text).groups()); assert numbers.count(target)==ans
    elif fam=='sort2': assert sorted(map(int,text.splitlines()[0].split()[1:]))[-2]==int(text.split(': ')[-1])
    elif fam=='bool':
        expr,ans=text.rsplit(' is ',1); assert str(eval(expr))==ans
    elif fam=='acc5':
        value=int(re.search(r'Start with (\d+)',text)[1])
        for op,v in re.findall(r'(Add|Subtract|Multiply by) (\d+)',text):
            v=int(v); value=value+v if op=='Add' else value-v if op=='Subtract' else value*v
        assert value==int(text.split('Result: ')[-1])
    else:
        value=int(re.search(r'has (\d+)',text)[1])
        for sentence in text.split('. ')[1:-1]:
            if 'doubles' in sentence: value*=2
            else:
                v=int(re.search(r'\d+',sentence)[0]); value+= -v if any(w in sentence for w in ['gives away','loses','sells']) else v
        assert value==int(text.split('Answer: ')[-1])


def check_generators():
    for fam in app.FAMS[:15]:
        items=app.generate(fam,80)
        assert items==app.generate(fam,80)==app.generate(fam,380)[:80], fam
        queries=[it.get('query',it['prompt'].splitlines()[-1]) for it in items]
        assert len(set(queries))==len(items), fam
        owners=[]
        for it in items:
            assert it['pid']==100000*(100+app.FAMS.index(fam))+it['i']
            d=it.get('detail',{}); q=it.get('query',''); ans=it['ans'].strip()
            if fam=='add3':
                for line in it['prompt'].splitlines():
                    expr,value=line.split('='); expected=eval(expr)
                    if value.strip():
                        assert expected==int(value); assert not any(mon.arith_key('add3',list(map(int,re.findall(r'\d+',expr))))==mon.arith_key('add3',x['ops']) for x in items)
                    else: assert expected==int(ans)
            elif fam in ('add6','chain6','mul2add','paren','bool'):
                expected=eval(d['expr']); assert str(expected)==ans
                if fam=='bool': assert len(re.findall(r'\b(?:not|and|or)\b',d['expr']))==5 and 'not' in d['expr']
                if fam=='add6': assert all(10<=v<=99 for v in map(int,re.findall(r'\d+',d['expr'])))
                if fam=='chain6':
                    parts=d['expr'].split(); assert parts.count('+')>=2 and parts.count('-')>=2
                    for end in range(1,len(parts)+1,2): assert 1<=eval(' '.join(parts[:end]))<=999
            elif fam in ('var4','var4s'):
                values={}; lines=q.splitlines()[:-1]
                while lines:
                    for line in lines[:]:
                        name,expr=line.split(' = ')
                        try: values[name]=eval(expr,{},values)
                        except NameError: continue
                        assert 1<=values[name]<=999; lines.remove(line)
                assert values[d['letters'][-1]]==int(ans)
                assert len(set(d['letters']))==4
                if fam=='var4s': assert d['order']!=list(range(4))
            elif fam=='ptr3':
                mapping={int(a):int(b) for a,b in re.findall(r'f\((\d+)\) = (\d+)',q)}; u=d['u']; visited=[]
                for _ in range(8): visited.append(u); u=mapping[u]
                assert u==d['u'] and len(set(visited))==8
                u=d['u']
                for _ in range(3): u=mapping[u]
                assert u==int(ans)
            elif fam=='swap':
                state=dict(zip(['Alice','Bob','Carol'],d['initial'])); original=state.copy()
                for a,b in re.findall(r'(Alice|Bob|Carol) and (Alice|Bob|Carol) swap balls',q): state[a],state[b]=state[b],state[a]
                person=q.split(' Now ')[-1].split()[0]; assert state[person]==ans
                owners.append(list(original.values()).index(ans)); assert len(d['swaps'])==4
            elif fam in ('acc5','story2','story4'):
                if fam=='acc5':
                    value=int(re.search(r'Start with (\d+)',q)[1]); ops=re.findall(r'(Add|Subtract|Multiply by) (\d+)',q)
                    for op,v in ops:
                        v=int(v); value=value+v if op=='Add' else value-v if op=='Subtract' else value*v; assert 1<=value<=999
                    assert len(ops)==5
                else:
                    value=int(re.search(r'has (\d+)',q)[1]); sentences=q.split('. ')[1:-1]
                    for sentence in sentences:
                        if 'doubles' in sentence: value*=2
                        else:
                            v=int(re.search(r'\d+',sentence)[0]); value+= -v if any(s in sentence for s in ['gives away','loses','sells']) else v
                        assert 1<=value<=999
                    assert len(sentences)==int(fam[-1])
                assert value==int(ans)
            elif fam=='count': assert q.splitlines()[0].split()[1:].count(str(d['target']))==int(ans) and 1<=int(ans)<=6
            elif fam=='sort2': assert sorted(map(int,q.splitlines()[0].split()[1:]))[-2]==int(ans) and len(set(d['values']))==6
            if fam!='add3':
                for demo in it['demos']:
                    verify_demo(fam,demo)
                    assert not any(demo.startswith(query) for query in queries)
            if it['kind']=='num': assert 0<=int(ans)<=999
        if fam=='bool': assert sum(x['ans']==' True' for x in items)==40
        if fam=='swap': assert max(owners.count(j) for j in range(3))-min(owners.count(j) for j in range(3))<25
    print('15 synthetic families: answers, ranges, uniqueness, demos, balance and 80/380 prefix stability passed')
    class Tok:
        def encode(self,s,add_special_tokens=False): return [1] if s.startswith(' ') and s.strip() in 'ABCD' else [1]*min(len(s)//20+1,200)
    rows=[dict(id=str(i),question=f'Question number {i}',choices=['one','two','three','four'],labels=list('ABCD'),answerKey='A') for i in range(1000)]
    tok=Tok(); a=app.dataset_items('arc_chal',rows,tok,80); assert a==app.dataset_items('arc_chal',rows,tok,380)[:80]
    for it in a:
        assert it['mc']['id'] not in it['mc']['demo_ids']
        assert it['ans']==' '+'ABCD'[it['mc']['permutation'].index(0)]
    for fam in ['gsm','svamp']:
        rows=[dict(question=f'{i} apples?',answer=f'work #### {i%100}',_split='train' if i<10 else 'test') if fam=='gsm' else dict(Body=f'{i} apples.',Question='How many?',Answer=i%100) for i in range(500)]
        a=app.dataset_items(fam,rows,tok,80); assert a==app.dataset_items(fam,rows,tok,80)==app.dataset_items(fam,rows,tok,380)[:80]
        assert len({x['prompt'] for x in a})==80
    print('3 dataset generators: fixture selection, determinism and prefix stability passed (real tokenizer checks deferred)')


class Block(torch.nn.Module):
    def forward(self,x,*args):
        # An independent causal mixing operation; future right padding cannot affect earlier tokens.
        return .7*x+.03*x.cumsum(1)/torch.arange(1,x.shape[1]+1,device=x.device)[None,:,None]


class Emb(torch.nn.Embedding):
    def forward(self,ids): return super().forward(ids%29)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.config=SimpleNamespace(init_values={'std':.02},n_embd=8,test_time_noise=0,injection_type='linear'); self.emb_scale=1.
        self.transformer=torch.nn.Module(); t=self.transformer
        t.wte=Emb(29,8); t.prelude=torch.nn.ModuleList([Block() for _ in range(4)]); t.core_block=torch.nn.ModuleList([Block() for _ in range(6)]); t.coda=torch.nn.ModuleList([Block() for _ in range(4)]); t.adapter=torch.nn.Linear(16,8); t.ln_f=torch.nn.LayerNorm(8); self.lm_head=torch.nn.Linear(8,29)
    def rotary_emb(self,e,p): return None
    def forward(self,ids,input_states=None,num_steps=None):
        t=self.transformer; e=t.wte(ids)
        for b in t.prelude: e=b(e)
        if input_states is None: x=e
        else:
            x=input_states
            for _ in range(num_steps):
                x=t.adapter(torch.cat([x,e],-1))
                for b in t.core_block: x=b(x)
        for b in t.coda: x=b(x)
        return SimpleNamespace(logits=self.lm_head(t.ln_f(x)))
    def model(self,ids,use_cache=False):  # the D path reads the final-norm hidden state at each problem's own last position
        t=self.transformer; x=t.wte(ids)
        for b in list(t.prelude)+list(t.coda): x=b(x)
        return SimpleNamespace(last_hidden_state=t.ln_f(x))


def rejects(fn):
    try: fn()
    except AssertionError: return
    raise AssertionError('broken gate accepted')


def check_model():
    torch.manual_seed(4); torch.set_num_threads(1); M=Tiny().eval(); dev=torch.device('cpu'); its=[{'pid':1,'fam':'add3'},{'pid':2,'fam':'add3'}]; ids=[[1,2,3],[2,3,4,5,6]]
    for key in ['L32','D']:
        gates={'G1':[],'G2':[]}
        with torch.no_grad():
            z=app.batch_read(M,key,its,ids,[3,4],dev,gates,True,float('inf'))
            for j in range(2):
                single=app.batch_read(M,key,its[j:j+1],ids[j:j+1],[3+j],dev,{'G1':[],'G2':[]},True,float('inf'))
                for k in z: assert np.allclose(z[k][j:j+1],single[k],atol=1e-6,equal_nan=True),(key,k)
        assert max(x['error'] for x in gates['G1' if key=='L32' else 'G2'])<1e-3
        app.check_chunk(z); bad={k:v.copy() for k,v in z.items()}; bad['ent'][0,0]=np.nan; rejects(lambda:app.check_chunk(bad))
    x=app.start(M,[dict(its[0],plen=3),dict(its[1],plen=5)],5,11000000,dev)
    assert torch.equal(x[0,:3],mon.start(M,its[:1],3,11000000,dev)[0]) and torch.count_nonzero(x[0,3:])==0
    real=app.harness
    def broken(M,dev):
        prep,step,read=real(M,dev)
        def wrong(x,fc,positions=None):
            c,l=read(x,fc,positions); return c,l+torch.arange(l.shape[-1])*.1
        return prep,step,wrong
    app.harness=broken
    try: rejects(lambda:app.batch_read(M,'L32',its,ids,[3,4],dev,{'G1':[],'G2':[]},True,float('inf')))
    finally: app.harness=real
    class BadD(Tiny):
        def model(self,ids,use_cache=False):  # the batched D path disagrees with the single-problem forward: G2 must fire
            out=super().model(ids)
            if ids.shape[0]>1: out.last_hidden_state=out.last_hidden_state+torch.arange(8)*.1
            return out
    rejects(lambda:app.batch_read(BadD(),'D',its,ids,[3,4],dev,{'G1':[],'G2':[]},True,float('inf')))
    class Tok:
        def encode(self,s,add_special_tokens=False): return [int(s)] if s.isdigit() else [1,2,3]
        def decode(self,ids): return ' '
    tok=Tok(); rejects(lambda:app.tokenize(tok,[dict(prompt='q ',kind='word',ans=' bad')]))
    testitems=[dict(it,i=j,kind='num',prompt='q ',ans='3',chance=0) for j,it in enumerate(its)]
    with tempfile.TemporaryDirectory() as tmp:
        meta=app.run_collect('L32',M,tok,testitems,{'limit_s':30},tmp,dev,lambda:None,time.monotonic())
        assert meta['status']=='complete' and meta['items_committed']==2
        partial=app.run_collect('D',M,tok,testitems,{'limit_s':0},tmp,dev,lambda:None,time.monotonic())
        assert partial['status']=='partial'
        app.harness=broken
        try: bad=app.run_collect('L32',M,tok,testitems,{'limit_s':30},tmp,dev,lambda:None,time.monotonic())
        finally: app.harness=real
        assert bad['status']=='gate_fail' and not bad['chunks']
    print('tiny causal model: padded/single readouts, exact starts, G1–G4 rejection, persistence and deadline passed')


def check_analysis():
    families=[('qual',.2,0,.5,.5,0,'QUALIFIES'),('border',.1,0,.5,.5,0,'BORDERLINE'),('lowL',.2,0,.09,.5,0,'CHANGES BUT UNUSABLE'),('highL',.2,0,.91,.5,0,'CHANGES BUT UNUSABLE'),('lowD',.2,0,.5,.09,0,'CHANGES BUT UNUSABLE'),('highD',.2,0,.5,.91,0,'CHANGES BUT UNUSABLE'),('flicker',.2,.1,.5,.5,0,'CHANGES BUT UNUSABLE'),('settled',.09,0,.5,.5,0,'SETTLED')]
    n=100; count=n*len(families); pred=np.ones((count,len(app.T),5),dtype=np.int32); ans=np.zeros(count,dtype=np.int32); items=[]
    D=np.ones((count,1,5),dtype=np.int32)
    for f,(name,c,fl,acc,dacc,ch,v) in enumerate(families):
        off=f*n; pred[off:off+round(n*acc),:,0]=0; D[off:off+round(n*dacc),:,0]=0
        for j in range(round(c*n)): pred[off+j,app.T.index(8),0]=2
        for j in range(round(fl*n)): pred[off+j,app.T.index(31),0]=2
        items += [dict(pid=off+j,fam=name,chance=ch) for j in range(n)]
    L=dict(pid=np.arange(count),ans_tok=ans,T=np.array(app.T),top_id=pred,top_lp=np.tile(np.array([0,-1,-2,-3,-4.]),(count,len(app.T),1)),rel_step=np.tile(np.r_[np.nan,np.ones(33)],(count,1)))
    dz=dict(pid=np.arange(count),ans_tok=ans,top_id=D)
    r=analysis.analyze(L,dz,items)
    for name,c,fl,a,d,ch,v in families:
        assert r['families'][name]['verdict']==v,(name,r['families'][name]['verdict'])
        assert r['families'][name]['changes']['8,32']['rate']==c
    for acc in [.1,.9]: assert analysis.verdict(.15,.05,acc,acc,0,100)=='QUALIFIES'
    assert analysis.verdict(.149999,0,.5,.5,0,100)=='BORDERLINE'
    assert analysis.verdict(.099999,0,.5,.5,0,100)=='SETTLED'
    assert analysis.verdict(.15,.050001,.5,.5,0,100)=='CHANGES BUT UNUSABLE'
    assert analysis.verdict(.1,0,.5,.5,0,300,True)=='ACCEPTED'
    assert analysis.verdict(.1,0,.5,.5,0,100,True)=='NOT ACCEPTED'
    assert analysis.verdict(.099,0,.5,.5,0,300,True)=='NOT ACCEPTED'
    with tempfile.TemporaryDirectory() as tmp:
        np.savez(Path(tmp,'L32.npz'),**L); np.savez(Path(tmp,'D.npz'),**dz)
        import json
        for j,it in enumerate(items): it['i']=j%100
        Path(tmp,'items.json').write_text(json.dumps(items))
        cont={k:dict(status='complete',cfg=dict(i0=0,n=100,fams='')) for k in ('L32','D')}
        Path(tmp,'containers.json').write_text(json.dumps(cont)); res=analysis.main(tmp,str(Path(tmp,'items.json')))
        assert Path(tmp,'summary.md').exists() and res['next_step'].startswith('confirm') and res['candidates'][0]=='qual'
        # Draft 1.1: an incomplete screen never yields the negative conclusion; an empty collection is handled.
        keep=np.array([it['fam'] not in ('qual','border') for it in items]); keep[:40]=True  # 'qual' has 40 < 48 problems, 'border' none
        np.savez(Path(tmp,'L32.npz'),**{k:(v if k=='T' else v[keep]) for k,v in L.items()})
        res=analysis.main(tmp,str(Path(tmp,'items.json')))
        assert res['next_step'].startswith('screen incomplete') and set(res['insufficient'])=={'qual','border'}, res['next_step']
        np.savez(Path(tmp,'L32.npz')); res=analysis.main(tmp,str(Path(tmp,'items.json')))
        assert res['next_step'].startswith('screen incomplete') and len(res['insufficient'])==len(families)
        full={k:(v if k=='T' else v[np.array([it['fam'] not in ('qual','border') for it in items])]) for k,v in L.items()}
        np.savez(Path(tmp,'L32.npz'),**full); its2=[it for it in items if it['fam'] not in ('qual','border')]
        Path(tmp,'items.json').write_text(json.dumps(its2)); res=analysis.main(tmp,str(Path(tmp,'items.json')))
        assert res['next_step'].startswith('no candidate among adequately sampled families'), res['next_step']
    print('synthetic analysis: all verdicts, failing conditions, threshold boundaries, confirmation Wilson rule and reports passed')


def main():
    for name in ['arith_key','make_arith','stable_hash','one_token','make_mcq','centered_error']:
        assert inspect.getsource(getattr(app,name))==inspect.getsource(getattr(mon,name)),name
    print('all 6 unchanged copied functions are source-identical')
    check_generators(); check_model(); check_analysis()
    import os
    fixture=os.environ.get('LATESHIFT_TOKENIZER')
    if fixture:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(fixture,local_files_only=True)
        for fam in app.FAMS[:15]: app.tokenize(tok,app.generate(fam,380))
        print('cached tokenizer fixture: all 5700 synthetic prompts pass G3; pinned remote tokenizer still requires build')
    else: print('Tokenizer G3 checks deferred to build_items (or set LATESHIFT_TOKENIZER for an offline fixture)')
    print('ALL CHECKS PASSED')


if __name__=='__main__': main()
