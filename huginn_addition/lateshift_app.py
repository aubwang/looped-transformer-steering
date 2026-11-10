"""Fixed-design late-shift screen. No collection occurs on import."""
from __future__ import annotations
import hashlib
import inspect
import json
import os
import random
import time
import io
import base64
from pathlib import Path
import modal

MODELS = {
    "L32": ("smcleish/Recurrent-Llama-3.2-train-recurrence-32", "a5f6f126e9d9302445346844f1d7d29d111a06eb", 32),
    "D": ("smcleish/Llama-3.2-non-recurrent-posttrained", "004e96ace360c1a37c08f4555227136e845087b3", 0),
}
BOS = 128000
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.6.0", "transformers==4.53.1", "safetensors", "huggingface_hub[hf_transfer]",
                      "numpy", "pandas", "pyarrow")
         .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"}))
FAMS = ['add3','add6','chain6','mul2add','paren','var4','var4s','ptr3','swap','acc5','count','sort2','bool','story2','story4','gsm','svamp','arc_chal']
T = [1,2,3,4,6,8,12,16,24,31,32,33]
BATCH = 16  # four round-robin rounds at n = 64, so a deadline stop leaves coverage in steps of 16 per family
CHANCE = dict(ptr3=1/8, swap=1/3, count=1/6, sort2=1/6, bool=1/2, arc_chal=1/4)
app = modal.App("lateshift-screen")
hf_cache = modal.Volume.from_name("huginn-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("huginn-results", create_if_missing=True)


# Copied from mon_app.py unchanged.
def arith_key(fam, ops):
    a, b, c = ops[:3]
    if fam in ("add3", "add4", "add3w"):
        return tuple(sorted(ops))
    if fam == "addsub":
        return tuple(sorted((a, b))), c
    if fam == "sub3":
        return a, tuple(sorted((b, c)))
    return tuple(ops)


# Copied from mon_app.py unchanged.
def make_arith(fam, n_total, seed):
    rng = random.Random(seed)

    def draw():
        while True:
            if fam in ("add3", "add4", "add3w"):
                lo, hi = (100, 799) if fam == "add3w" else (10, 99)
                ops = [rng.randint(lo, hi) for _ in range(4 if fam == "add4" else 3)]
                ans = sum(ops); q = " + ".join(map(str, ops)) + " = "
            elif fam == "addsub":
                ops = [rng.randint(100, 899), rng.randint(100, 899), rng.randint(10, 999)]
                a, b, c = ops; ans = a + b - c; q = f"{a} + {b} - {c} = "
            elif fam == "muladd":
                ops = [rng.randint(12, 99), rng.randint(3, 9), rng.randint(10, 99)]
                a, b, c = ops; ans = a * b + c; q = f"{a} * {b} + {c} = "
            else:
                assert fam == "sub3"
                ops = [rng.randint(200, 999), rng.randint(10, 99), rng.randint(10, 99)]
                a, b, c = ops; ans = a - b - c; q = f"{a} - {b} - {c} = "
            if 1 <= ans <= 999:
                return ops, ans, q

    queries, keys = [], set()
    while len(queries) < n_total:
        p = draw(); k = arith_key(fam, p[0])
        if k not in keys:
            keys.add(k); queries.append(p)
    items = []
    for i, (ops, ans, q) in enumerate(queries):
        demos, seen = [], set()
        while len(demos) < 4:
            op, a, text = draw(); k = arith_key(fam, op)
            if k not in keys and k not in seen:
                seen.add(k); demos.append(text + str(a) + "\n")
        items.append(dict(pid=100000 * FAMS.index(fam) + i, fam=fam, i=i, kind="std", ops=ops,
                          ans=ans, prompt="".join(demos) + q, valid="num"))
    return items


# Copied from mon_app.py unchanged.
def stable_hash(s):
    return int(hashlib.sha256(str(s).encode()).hexdigest()[:8], 16)


# Copied from mon_app.py unchanged.
def one_token(tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    assert len(ids) == 1, (text, ids)
    return ids[0]


# Copied from mon_app.py unchanged.
def make_mcq(rows, tok, fam, n_total, seed, max_tokens=160):
    counts = dict(rows_total=len(rows)); seen, good = set(), []
    four = [r for r in rows if len(r["choices"]) == len(r["labels"]) == 4 and r["answerKey"] in r["labels"]]
    counts["four_option"] = len(four)
    for r in sorted(four, key=lambda r: str(r["id"])):
        if r["question"] not in seen:
            seen.add(r["question"]); good.append(r)
    counts["deduplicated"] = len(good)
    random.Random(seed).shuffle(good)

    def render(r):
        p = list(range(4)); random.Random(seed * 1000003 + stable_hash(r["id"])).shuffle(p)
        correct = r["labels"].index(r["answerKey"]); letter = "ABCD"[p.index(correct)]
        text = f'Question: {r["question"]}\n' + "".join(f'{l}. {r["choices"][j]}\n' for l, j in zip("ABCD", p)) + "Answer:"
        return text, letter, p

    pool = good[:200]; items, hist = [], {}; examined = 0
    for r in good[200:]:
        if len(items) >= n_total:
            break
        examined += 1
        rng = random.Random(seed * 1000003 + stable_hash(r["id"]))
        demos = rng.sample(pool, 2); q, ans, perm = render(r)
        prompt = "\n".join(render(d)[0] + " " + render(d)[1] for d in demos) + "\n" + q
        L = 1 + len(tok.encode(prompt, add_special_tokens=False))
        if L > max_tokens:
            continue
        one_token(tok, " " + ans)
        i = len(items)
        items.append(dict(pid=100000 * FAMS.index(fam) + i, fam=fam, i=i, kind="std", ans=ans,
                          prompt=prompt, valid="abcd", mc=dict(id=r["id"], permutation=perm,
                          demo_ids=[d["id"] for d in demos])))
        hist[str(L)] = hist.get(str(L), 0) + 1
    counts.update(examined=examined, kept=len(items), length_histogram=hist)
    return items, counts


# Copied from mon_app.py (read generalised for right padding).
def harness(M, dev):
    import torch
    c = M.config
    assert c.test_time_noise == 0 and c.injection_type == "linear" and M.emb_scale == 1.0
    assert len(M.transformer.prelude) == 4 and len(M.transformer.core_block) == 6 and len(M.transformer.coda) == 4
    dum = torch.tensor(0, dtype=torch.long)

    def prep(ids):
        e = M.transformer.wte(ids); fc = M.rotary_emb(e, torch.arange(ids.shape[1], device=dev)[None])
        bi = torch.tensor(-1, dtype=torch.long)
        for blk in M.transformer.prelude:
            bi += 1; e = blk(e, fc, bi, None, None)
        return e, fc

    def step(x, e, fc):
        x = M.transformer.adapter(torch.cat([x, e], -1))
        for blk in M.transformer.core_block:
            x = blk(x, fc, dum, None, None)
        return x

    def read(x, fc, positions=None):
        for blk in M.transformer.coda:
            x = blk(x, fc, dum, None, None)
        last = x[:, -1] if positions is None else x[torch.arange(len(x), device=x.device), positions]
        return x, M.lm_head(M.transformer.ln_f(last)).double()
    return prep, step, read


# Copied from mon_app.py unchanged.
def centered_error(a, b):
    a, b = a.double(), b.double()
    return float(((a - a.mean(-1, keepdim=True)) - (b - b.mean(-1, keepdim=True))).abs().max())

# Copied from mon_app.start; draw at each real length, then zero-pad.
def start(M, items, L, seed, dev):
    import torch
    std = float(M.config.init_values['std']); xs = []
    for it in items:
        length = it.get('plen', L)
        g = torch.Generator('cpu').manual_seed(seed + it['pid'])
        x = torch.empty((length, M.config.n_embd), dtype=torch.float32)
        torch.nn.init.trunc_normal_(x, std=std, a=-3*std, b=3*std, generator=g)
        xs.append(torch.nn.functional.pad(x, (0, 0, 0, L-length)))
    return torch.stack(xs).to(dev)


def draw_problem(fam, rng, index):
    """Return query, answer text, independently checkable generation metadata."""
    while True:
        d = {}; word = False
        if fam in ('add6', 'chain6', 'mul2add', 'paren'):
            if fam == 'add6':
                a = [rng.randint(10,99) for _ in range(6)]; q = ' + '.join(map(str,a))
            elif fam == 'chain6':
                # NOTE(design): follow prose random signs, treating the table's signs as an example.
                a = [rng.randint(100,500)] + [rng.randint(10,99) for _ in range(5)]
                signs = [rng.choice('+-') for _ in range(5)]
                if min(signs.count('+'),signs.count('-')) < 2: continue
                q = str(a[0]); value = a[0]; ok = True
                for sign,v in zip(signs,a[1:]):
                    q += f' {sign} {v}'; value += v if sign=='+' else -v
                    ok &= 1 <= value <= 999
                if not ok: continue
            elif fam == 'mul2add':
                a,c = [rng.randint(11,40) for _ in range(2)]; b,dv = [rng.randint(3,9) for _ in range(2)]
                q = f'{a} * {b} + {c} * {dv}'
            else:
                a,b = [rng.randint(10,60) for _ in range(2)]; c=rng.randint(2,8); dv=rng.randint(10,99)
                q=f'({a} + {b}) * {c} - {dv}'
            ans=eval(q); d={'expr':q}; q += ' = '
        elif fam in ('var4','var4s'):
            letters=rng.sample(list('abcdefghijklmnopqrstuvwxyz'),4); values=[rng.randint(10,99)]; lines=[f'{letters[0]} = {values[0]}']; ops=[]
            for j in range(1,4):
                sign=rng.choice('+-'); v=rng.randint(10,99); ops.append([sign,v]); values.append(values[-1]+(v if sign=='+' else -v))
                lines.append(f'{letters[j]} = {letters[j-1]} {sign} {v}')
            if not all(1<=v<=999 for v in values): continue
            order=list(range(4))
            if fam=='var4s':
                while order==list(range(4)): rng.shuffle(order)
            q='\n'.join(lines[j] for j in order)+f'\nValue of {letters[-1]}: '; ans=values[-1]
            d=dict(letters=letters,initial=values[0],ops=ops,order=order)
        elif fam=='ptr3':
            cycle=rng.sample(range(10,100),8); mapping=dict(zip(cycle,cycle[1:]+cycle[:1])); lines=[f'f({u}) = {v}' for u,v in mapping.items()]; rng.shuffle(lines)
            u=rng.choice(cycle); ans=mapping[mapping[mapping[u]]]; q='\n'.join(lines)+f'\nf(f(f({u}))) = '; d=dict(mapping=mapping,u=u)
        elif fam=='swap':
            names=['Alice','Bob','Carol']; colors=rng.sample(['red','blue','green','yellow','black','white'],3); state=colors[:]; swaps=[]
            q=f'Alice has the {colors[0]} ball, Bob has the {colors[1]} ball, and Carol has the {colors[2]} ball.'
            for _ in range(4):
                a,b=rng.sample(range(3),2); swaps.append([a,b]); state[a],state[b]=state[b],state[a]; q+=f' {names[a]} and {names[b]} swap balls.'
            who=rng.randrange(3); q+=f' Now {names[who]} has the'; ans=state[who]; word=True; d=dict(initial=colors,swaps=swaps,who=who)
        elif fam in ('acc5','story2','story4'):
            initial=rng.randint(10,99); value=initial; ops=[]; multiplies=0
            for _ in range(5 if fam=='acc5' else int(fam[-1])):
                choices=['+','-']+(['*'] if multiplies<(2 if fam=='acc5' else int(fam=='story4')) else [])
                op=rng.choice(choices); v=rng.randint(2,3) if op=='*' and fam=='acc5' else (2 if op=='*' else rng.randint(10,99))
                value= value*v if op=='*' else value+(v if op=='+' else -v); ops.append([op,v]); multiplies+=op=='*'
                if not 1<=value<=999: break
            if len(ops)!=(5 if fam=='acc5' else int(fam[-1])) or not 1<=value<=999: continue
            ans=value; d=dict(initial=initial,ops=ops)
            if fam=='acc5': q=f'Start with {initial}. '+ ' '.join(f'{ {"+":"Add","-":"Subtract","*":"Multiply by"}[op]} {v}.' for op,v in ops)+' Result: '
            else:
                name=rng.choice(['Alice','Bob','Carol','David','Emma']); things=rng.choice(['apples','coins','books','marbles'])
                q=f'Question: {name} has {initial} {things}.'
                for op,v in ops:
                    if op=='*': q+=f' {name} doubles the number of {things}.'
                    elif op=='+': q+=f' {name} {rng.choice(["buys","finds","receives"])} {v} more {things}.'
                    else: q+=f' {name} {rng.choice(["gives away","loses","sells"])} {v} {things}.'
                q+=f' How many {things} does {name} have now?\nAnswer: '
        elif fam=='count':
            target=rng.randrange(10); c=rng.randint(1,6); values=[target]*c+[rng.choice([v for v in range(10) if v!=target]) for _ in range(12-c)]; rng.shuffle(values)
            q='List: '+' '.join(map(str,values))+f'\nHow many times does {target} appear? Answer: '; ans=c; d=dict(values=values,target=target)
        elif fam=='sort2':
            values=rng.sample(range(10,100),6); q='Numbers: '+' '.join(map(str,values))+'\nSecond largest: '; ans=sorted(values)[-2]; d=dict(values=values)
        else:
            assert fam=='bool'
            def tree(n):
                if not n: return rng.choice(['True','False'])
                op=rng.choice(['not','and','or'])
                if op=='not': return f'not ( {tree(n-1)} )'
                left=rng.randrange(n); return f'( {tree(left)} {op} {tree(n-1-left)} )'
            expr=tree(5)
            if 'not' not in expr or bool(eval(expr)) != bool(index%2): continue
            ans=str(eval(expr)); q=expr+' is'; word=True; d=dict(expr=expr)
        if not word and not 0<=int(ans)<=999: continue
        return q, (' ' if word else '')+str(ans), d


def generate(fam, n_total):
    fi=FAMS.index(fam); rng=random.Random(20261110*100+fi)
    # add3's unchanged helper draws demos after all queries, which is not prefix stable.
    if fam=='add3':
        # Reuse unchanged arithmetic rendering/draw; reserve all requested screen+confirmation indices.
        # NOTE(design): canonical 380-item pool reconciles unchanged helper with 80/380 prefix stability.
        assert n_total<=380, 'add3 canonical screen + confirmation pool is 380'
        old=make_arith('add3',380,20261110*100)
        return [dict(it,pid=100000*(100+fi)+i,kind='num',ans=str(it['ans']),chance=0) for i,it in enumerate(old[:n_total])]
    k=2 if fam in ('ptr3','swap','story2','story4') else (4 if fam in ('add6','chain6','mul2add','paren','bool') else 3)
    sep='\n\n' if fam in ('var4','var4s','ptr3','swap','count','sort2','story2','story4') else '\n'
    # Canonical pool of 380 queries drawn first (prefix-stable for any n_total <= 380); then, per query, its own k worked
    # examples (as in mon_app.make_arith), none equal to any query of the pool or to another example of the same prompt.
    assert n_total<=380, 'canonical screen + confirmation pool is 380'
    seen=set(); queries=[]
    while len(queries)<380:
        q,a,d=draw_problem(fam,rng,len(queries))
        if q in seen: continue
        seen.add(q); queries.append((q,a,d))
    out=[]
    for i,(q,a,d) in enumerate(queries[:n_total]):
        demos=[]; used=set()
        while len(demos)<k:
            dq,da,_=draw_problem(fam,rng,len(demos))
            if dq in seen or dq in used: continue
            used.add(dq); demos.append(dq+da+(' ball.' if fam=='swap' else ''))
        out.append(dict(pid=100000*(100+fi)+i,fam=fam,i=i,prompt=sep.join(demos+[q]),ans=a,
                        kind='word' if a.startswith(' ') else 'num',chance=CHANCE.get(fam,0),query=q,demos=demos,detail=d))
    return out


def tokenize(tok, items):
    ids=[]; answers=[]
    for it in items:
        v=[BOS]+tok.encode(it['prompt'],add_special_tokens=False)
        assert len(v)<=320, 'G3 length'
        if it['kind']=='num': assert tok.decode(v[-1:])==' ' and not it['ans'].startswith(' '), 'G3 number'
        else: assert not it['prompt'].endswith(' ') and it['ans'].startswith(' '), 'G3 word'
        ids.append(v); answers.append(one_token(tok,it['ans']))
    return ids,answers


def dataset_items(fam, rows, tok, n_total):
    fi=FAMS.index(fam); rng=random.Random(20261110*100+fi)
    if fam=='arc_chal':
        its,counts=make_mcq(rows,tok,fam,n_total,20261110*100+fi,max_tokens=320)
        return [dict(it,pid=100000*(100+fi)+it['i'],kind='word',ans=' '+it['ans'],chance=1/4) for it in its]
    eligible=[]; seen=set()
    for r in rows:
        if fam=='gsm': q=r['question']; raw=str(r['answer']).split('####')[-1].strip().replace(',','')
        else: q=str(r['Body']).strip()+' '+str(r['Question']).strip(); raw=str(r['Answer']).strip().replace(',','')
        try: value=float(raw)
        except ValueError: continue
        if not value.is_integer() or not 0<=value<=999 or q in seen: continue
        length=len(tok.encode(q,add_special_tokens=False))
        if length>110: continue
        seen.add(q); eligible.append((length,q,str(int(value)),r.get('_split','')))
    pool=sorted([x for x in eligible if fam!='gsm' or x[3]=='train'],key=lambda x:(x[0],x[1]))[:2]
    assert len(pool)==2, 'two eligible demos required'
    queries=[x for x in eligible if x not in pool and (fam!='gsm' or x[3]=='test')]; rng.shuffle(queries)
    demos=[f'Question: {x[1]}\nAnswer: {x[2]}' for x in pool]; out=[]
    for _,q,a,_ in queries:
        prompt='\n\n'.join(demos+[f'Question: {q}\nAnswer: '])
        if 1+len(tok.encode(prompt,add_special_tokens=False))>320: continue
        i=len(out); out.append(dict(pid=100000*(100+fi)+i,fam=fam,i=i,prompt=prompt,ans=a,kind='num',chance=0))
        if len(out)==n_total: break
    return out


@app.function(image=image, volumes={'/cache':hf_cache}, timeout=600)
def build_items(cfg):
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer
    import pandas as pd
    tok=AutoTokenizer.from_pretrained(MODELS['L32'][0],revision=MODELS['L32'][1]); api=HfApi()
    n_total=cfg.get('n_total',380); items=[]; counts={'datasets':{}}
    for fam in FAMS:
        if fam not in ('gsm','svamp','arc_chal'): items.extend(generate(fam,n_total)); continue
        repo={'gsm':'openai/gsm8k','svamp':'ChilleD/SVAMP','arc_chal':'allenai/ai2_arc'}[fam]
        try:
            sha=api.dataset_info(repo).sha; files=api.list_repo_files(repo,repo_type='dataset',revision=sha)
            counts['datasets'][fam]={'repo':repo,'sha':sha}; rows=[]
            for fn in sorted(files):
                if not fn.endswith('.parquet') or (fam=='gsm' and not fn.startswith('main/')) or (fam=='arc_chal' and not fn.startswith('ARC-Challenge/')): continue
                path=hf_hub_download(repo,fn,repo_type='dataset',revision=sha)
                for r in pd.read_parquet(path).to_dict('records'):
                    r['_split']='train' if 'train' in fn else ('test' if 'test' in fn else 'validation')
                    if fam=='arc_chal': r=dict(r,choices=list(r['choices']['text']),labels=list(r['choices']['label']))
                    rows.append(r)
            assert rows, 'no parquet rows'
        except Exception as ex:
            counts['datasets'].setdefault(fam,{'repo':repo})['dropped']=repr(ex); continue
        its=dataset_items(fam,rows,tok,n_total)
        items.extend(its)
    ids,_=tokenize(tok,items)
    for it,v in zip(items,ids): it['plen']=len(v)
    for fam in FAMS:
        lengths=[it['plen'] for it in items if it['fam']==fam]; counts[fam]={'kept':len(lengths),'length_histogram':{str(l):lengths.count(l) for l in sorted(set(lengths))}}
    n=80
    while n>64 and sum(it['plen']*33 for it in items if it['i']<n)/14000>330: n-=8
    counts.update(n=n,token_loops=sum(it['plen']*33 for it in items if it['i']<n),G3=True)
    hf_cache.commit(); return dict(items=items,counts=counts)


def check_chunk(z):
    import numpy as np
    for k,v in z.items():
        # NOTE(design): G4 excludes the explicitly required NaN sentinel at rel_step[:,0].
        if k=='rel_step':
            assert np.isnan(v[:,0]).all(), 'G4 sentinel'
            v=v[:,1:]
        assert np.isfinite(v).all(), 'G4 '+k


def batch_read(M, key, its, IDS, ANS, dev, gates, gate, deadline):
    import torch
    lengths=[len(v) for v in IDS]; L=max(lengths)
    ids=torch.tensor([v+[BOS]*(L-len(v)) for v in IDS],device=dev)
    pos=torch.tensor(lengths,device=dev)-1; rows=torch.arange(len(its),device=dev)
    ans=torch.tensor(ANS,device=dev); results=[]
    if key=='L32':
        prep,step,read=harness(M,dev); e,fc=prep(ids)
        batch_items=[dict(it,plen=l) for it,l in zip(its,lengths)]
        x=start(M,batch_items,L,11000000,dev)
        rel=torch.full((len(its),34),float('nan'),device=dev)
    for t in (range(1,34) if key=='L32' else [1]):
        if time.monotonic()>=deadline: raise TimeoutError('deadline')
        if key=='L32':
            new=step(x,e,fc); a=new[rows,pos].double(); b=x[rows,pos].double()
            rel[:,t]=((a-b).norm(dim=-1)/a.norm(dim=-1)).float(); x=new
            if t not in T: continue
            _,logits=read(x,fc,pos)
        else: logits=M.lm_head(M.model(ids,use_cache=False).last_hidden_state[rows,pos]).double()  # final norm is inside M.model; G2 checks it
        if gate and (key=='D' or t in (4,32)):
            for j in sorted({0,len(its)-1}):  # batches are length-sorted: the most padded and the unpadded problem
                single=torch.tensor([IDS[j]],device=dev)
                kwargs={} if key=='D' else dict(input_states=start(M,[its[j]],lengths[j],11000000,dev),num_steps=t)
                ref=M(single,**kwargs).logits[:,-1]
                err=centered_error(logits[j:j+1],ref); g='G2' if key=='D' else 'G1'
                gates[g].append(dict(fam=its[0]['fam'],t=t,pid=its[j]['pid'],error=err,plen=lengths[j],batch_len=L))
                assert err<=1e-3, f'{g} {err}'
        lp=logits.double().log_softmax(-1); top_lp,top_id=lp.topk(5,-1)
        results.append(dict(top_id=top_id.int(),top_lp=top_lp.float(),lp_ans=lp[rows,ans].float(),ent=(-(lp.exp()*lp).sum(-1)).float()))
    z={k:torch.stack([r[k] for r in results],1).cpu().numpy() for k in results[0]}
    if key=='L32': z['rel_step']=rel.cpu().numpy()
    return z


def run_collect(key,M,tok,items,cfg,o,dev,commit,T0):
    import numpy as np
    import torch
    os.makedirs(o,exist_ok=True); deadline=T0+cfg['limit_s']; chunks=[]; timing=[]; status='complete'; error=None
    gates=dict(G1=[],G2=[],G3=False,G4=False); stored=[]
    try:
        IDS,ANS=tokenize(tok,items); gates['G3']=True
        queues={f:sorted([j for j,it in enumerate(items) if it['fam']==f],key=lambda j:len(IDS[j])) for f in FAMS}
        # Gate families (Draft 1.1): one uniform-length family and three variable-length ones, so that G1/G2 compare
        # right-padded problems (plen < batch length) with the model's own unpadded forward.
        first={f:[len(IDS[j]) for j in queues[f][:BATCH]] for f in FAMS if queues[f]}
        variable=sorted((f for f in first if len(set(first[f]))>1),key=lambda f:-max(first[f]))[:3]  # longest contexts first
        gate_fams=variable+[f for f in first if len(set(first[f]))==1][:1]
        gates['selected_families']=gate_fams; visited=set()
        with torch.no_grad():
            while any(queues.values()):
                for fam in FAMS:
                    if not queues[fam]: continue
                    if time.monotonic()>=deadline: raise TimeoutError('deadline')
                    bi=queues[fam][:BATCH]; queues[fam]=queues[fam][BATCH:]; its=[items[j] for j in bi]; t0=time.monotonic()
                    z=batch_read(M,key,its,[IDS[j] for j in bi],[ANS[j] for j in bi],dev,gates,fam in gate_fams and fam not in visited,deadline)
                    visited.add(fam)
                    z.update(pid=np.array([it['pid'] for it in its],dtype=np.int64),family=np.full(len(bi),FAMS.index(fam),dtype=np.int32),
                             plen=np.array([len(IDS[j]) for j in bi],dtype=np.int32),ans_tok=np.array([ANS[j] for j in bi],dtype=np.int32))
                    if key=='L32': z['T']=np.array(T,dtype=np.int64)
                    check_chunk(z); gates['G4']=True; name=f'chunk{len(chunks):04d}.npz'; np.savez_compressed(os.path.join(o,name),**z)
                    chunks.append(name); stored.append(z)
                    if len(chunks)%18==0: commit()  # about once per round: a volume commit costs 1-2 s (smoke run), final commit below
                    timing.append([fam,'std',int(z['plen'].max()),len(bi),33 if key=='L32' else 16,time.monotonic()-t0])
        gates['G4']=True
        recs=gates['G2' if key=='D' else 'G1']
        assert {r['fam'] for r in recs}==set(gate_fams) and (not variable or any(r['plen']<r['batch_len'] for r in recs)), 'gate coverage: no right-padded problem was compared'
    except TimeoutError: status='partial'
    except AssertionError as ex: status='gate_fail'; error=str(ex)
    meta=dict(status=status,error=error,model=key,hf_id=MODELS[key][0],revision=MODELS[key][1],cfg=cfg,gates=gates,
              code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              forward_source_sha256=hashlib.sha256(inspect.getsource(type(M).forward).encode()).hexdigest(),
              dtype='float32',chunks=chunks,items_requested=len(items),items_committed=sum(len(z['pid']) for z in stored),
              timing=dict(batches=timing,model_load_seconds=cfg.get('model_load_seconds',0),total_seconds=time.monotonic()-T0))
    Path(o,'meta.json').write_text(json.dumps(meta,indent=1)); commit()
    merged={k:(stored[0][k] if k=='T' else np.concatenate([z[k] for z in stored])) for k in stored[0]} if stored else {}
    data=io.BytesIO(); np.savez_compressed(data,**merged)
    return dict(meta,arrays_npz=base64.b64encode(data.getvalue()).decode())


@app.function(image=image,gpu='A100-40GB',volumes={'/cache':hf_cache,'/results':results_vol},timeout=500)
def collect(key,items_json,cfg,run_id):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    assert key in ('L32','D'); assert 0<cfg['limit_s']<=580; T0=time.monotonic(); hf_id,rev,_=MODELS[key]
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    tok=AutoTokenizer.from_pretrained(hf_id,revision=rev)
    extra=dict(attn_implementation='sdpa') if key=='D' else dict(trust_remote_code=True)
    M=AutoModelForCausalLM.from_pretrained(hf_id,revision=rev,torch_dtype=torch.float32,**extra).to('cuda').eval()
    hf_cache.commit(); cfg=dict(cfg,model_load_seconds=time.monotonic()-T0)
    return run_collect(key,M,tok,json.loads(items_json),cfg,f'/results/{run_id}/{key}',torch.device('cuda'),results_vol.commit,T0)


@app.local_entrypoint()
def main(mode: str, items: str='', fams: str='', n: int=80, i0: int=0):
    stamp=time.strftime('%Y%m%d-%H%M%S',time.gmtime())
    assert mode in ('build','run','smoke')
    if mode=='build':
        result=build_items.remote(dict(n_total=380)); dest=Path(__file__).resolve().parent.parent/'results'/f'lateshift_items-{stamp}'; dest.mkdir(parents=True)
        raw=json.dumps(result,indent=1).encode(); Path(dest,'items.json').write_bytes(raw)
        Path(dest,'items.json.sha256').write_text(hashlib.sha256(raw).hexdigest()+'\n'); print(json.dumps(result['counts'],indent=1)); return
    raw=Path(items).read_bytes(); master=json.loads(raw); all_items=master['items'] if isinstance(master,dict) else master
    if n==80 and isinstance(master,dict): n=master.get('counts',{}).get('n',n)
    selected=[it for it in all_items if i0<=it['i']<i0+(4 if mode=='smoke' else n) and (not fams or it['fam'] in fams.split(','))]
    assert selected, 'no items selected'
    # Projection from the actual padded batches (length-sorted, BATCH per family), Draft 1.1.
    loops=0
    for f in FAMS:
        ls=sorted(it['plen'] for it in selected if it['fam']==f)
        loops+=sum(max(ls[a:a+BATCH])*len(ls[a:a+BATCH])*33 for a in range(0,len(ls),BATCH))
    assert loops/14000<=345, f'projected L32 seconds {loops/14000:.1f} > 345'
    run_id=f'lateshift{"_smoke" if mode=="smoke" else ""}-{stamp}'; dest=Path(__file__).resolve().parent.parent/'results'/run_id; dest.mkdir(parents=True)
    # Internal deadlines from function start: 450 s (L32) + 90 s (D) = 540 s, 60 s reserve for container start; total <= 600 GPU-seconds.
    cfg=dict(items_sha256=hashlib.sha256(raw).hexdigest(),projected_token_loops=loops,seed=11000000,batch=BATCH,n=4 if mode=='smoke' else n,i0=i0,mode=mode,fams=fams)
    calls={key:collect.spawn(key,json.dumps(selected),dict(cfg,limit_s=450 if key=='L32' else 90),run_id) for key in ('L32','D')}; results={}; bad=False
    for key,call in calls.items():
        try:
            res=call.get(); Path(dest,key+'.npz').write_bytes(base64.b64decode(res.pop('arrays_npz'))); results[key]=res
            batches=res['timing']['batches']; seconds=sum(b[-1] for b in batches)
            print(key,res['status'],'batch seconds',seconds,'token-loops/s',sum(b[2]*b[3]*b[4] for b in batches)/seconds if seconds else 0)
            bad |= res['status']!='complete'
        except Exception as ex: results[key]={'status':'failed','error':repr(ex)}; bad=True
    Path(dest,'containers.json').write_text(json.dumps(results,indent=1)); print(dest)
    if bad: raise SystemExit(1)
