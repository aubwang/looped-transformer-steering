"""Fixed-design L32 read-space experiment; importing numerics never runs Modal."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llever_app import (MODEL, BOS, make_items, one_token, start, harness,
                        centered_error, patch_norms, patch_precision, base_state,
                        readout_grads, chain, check_deadline)


def families(task):
    import numpy as np
    return np.array([0] * (10 if task == 'A' else 1) + [1]*8 + [2]*8 + [3]*8 + [4]*4 + [5]*12 + [6]*8)


def coefficients(tok, logits, tokens, task, pid):
    """Contrast coefficients; no differentiation through a vocabulary-sized loop."""
    import numpy as np
    import torch
    V = len(logits); dev = logits.device
    excluded = set(tok.all_special_ids) | set(tok.get_added_vocab().values())
    ordinary = np.array([i for i in range(min(tok.vocab_size, V)) if i not in excluded])
    allowed = np.array([i for i in ordinary if i not in tokens])
    assert len(allowed) >= 64
    rng = np.random.default_rng(9_000_000 + pid)
    c1 = np.array([rng.choice(allowed, 2, replace=False) for _ in range(8)])
    top = allowed[torch.topk(logits[torch.as_tensor(allowed, device=dev)], 64).indices.cpu().numpy()]
    c2 = np.array([top[rng.choice(64, 2, replace=False)] for _ in range(8)])
    nt = 10 if task == 'A' else 1
    C = logits.new_zeros((nt+24, V))
    if task == 'A':
        C[:nt, tokens] = -1/nt
        C[torch.arange(nt, device=dev), torch.tensor(tokens, device=dev)] += 1
    else:
        C[0, tokens[0]], C[0, tokens[1]] = 1, -1
    pairs = np.concatenate((c1, c2))
    for k, (a, b) in enumerate(pairs):
        C[nt+k, int(a)], C[nt+k, int(b)] = 1, -1
    gen = torch.Generator(device=dev).manual_seed(9_000_000 + pid)
    xi = torch.randn((8, V), dtype=logits.dtype, device=dev, generator=gen)
    mask = logits.new_zeros(V); mask[ordinary] = 1
    xi *= mask
    xi -= mask * (xi.sum(-1, keepdim=True)/len(ordinary))
    C[nt+16:] = xi
    return C, pairs


def directions(M, tok, read, x, fc, tokens, task, pid, rows=64, deadline=float('inf')):
    import torch
    with torch.no_grad():
        logits = read(x, fc)[1][0]
        C, pairs = coefficients(tok, logits, tokens, task, pid)
        contracted = C @ M.lm_head.weight
    # One coda graph, differentiated in batched chunks; unembedding contracted first.
    with torch.enable_grad():
        q = x[:, -1].detach().clone().requires_grad_(True)
        hidden = read(torch.cat((x[:, :-1].detach(), q[:, None]), 1), fc)[0]
        features = M.transformer.ln_f(hidden[:, -1])
        gs = []
        for a in range(0, len(C), rows):
            check_deadline(deadline)
            gs.append(torch.autograd.grad(features, q, contracted[a:a+rows, None],
                       is_grads_batched=True, retain_graph=a+rows < len(C))[0][:, 0].detach())
    G = torch.cat(gs); gn = G.norm(dim=-1)
    assert bool((gn > 0).all()), 'zero read gradient'
    U = G/gn[:, None]
    gen = torch.Generator(device=x.device).manual_seed(7_000_000 + pid)
    probes = torch.randn((16, x.shape[-1]), device=x.device, dtype=x.dtype, generator=gen)
    probes /= probes.norm(dim=-1, keepdim=True)
    signs = torch.randint(0, 2, (8, x.shape[-1]), device=x.device, generator=gen)*2-1
    U = torch.cat((U, probes, U[-8:]*signs))
    # Norms have meaning only for gradient families; use finite unit placeholders elsewhere.
    norms = torch.cat((gn, gn.new_ones(24)))
    return U, norms, G[:10 if task == 'A' else 1], pairs


def selfgain(step, x, e, fc, U, rows=64, deadline=float('inf')):
    import torch
    w1 = chain(step, x, e, fc, U[None], 1, 'vjp', rows, deadline)[0][0]
    w2 = chain(step, x, e, fc, w1[None], 1, 'vjp', rows, deadline)[0][0]
    gen = torch.Generator(device=x.device).manual_seed(9)
    r = torch.randn(U.shape, dtype=x.dtype, device=x.device, generator=gen)
    r /= r.norm(dim=-1, keepdim=True)
    jr = chain(step, x, e, fc, r[None], 1, 'jvp', rows, deadline)[0][0]
    lhs, rhs = (w1*r).sum(-1), (U*jr).sum(-1)
    cv = torch.stack((U, w1, w2), 1)
    return dict(gram3=cv@cv.transpose(1, 2), ugram=U@U.T, x1=U@w1.T, x2=U@w2.T,
                adjoint_rel_err=(lhs-rhs).abs()/(lhs.abs()+rhs.abs()).clamp_min(1e-12))


def run_set(M, tok, items, task, cfg, deadline=float('inf'), began=None):
    import numpy as np
    import torch
    began = time.monotonic() if began is None else began
    dev = next(M.parameters()).device; prep, step, read = harness(M, dev)
    tokens = [one_token(tok, str(70+k)) for k in range(10)] if task == 'A' else [one_token(tok, w) for w in (' True', ' False')]
    fam = families(task); n, S, d = len(items), len(fam), M.config.n_embd
    shape = dict(gram3=(S,3,3), ugram=(S,S), x1=(S,S), x2=(S,S), adjoint_rel_err=(S,),
                 gnorm=(S,), cos_x=(S,), drift=(), xnorm=(), G_task=(10 if task=='A' else 1,d), xq=(d,))
    out = {k:np.full((n,)+v, np.nan) for k,v in shape.items()}
    out.update(fam=fam, done=np.zeros(n,bool), correct=np.zeros(n,bool), argmax=np.full(n,-1),
               uid=np.array([p.get('uid',p['pid']) for p in items]), pid=np.array([p['pid'] for p in items]),
               C_tokens=np.full((n,16,2),-1), task_tokens=np.array(tokens))
    gates, timings, error = [], [], None
    if task == 'A': assert len({len(tok.encode(p['prompt'],add_special_tokens=False)) for p in items}) <= 1
    for i, it in enumerate(items):
        t = time.monotonic()
        try:
            check_deadline(deadline)
            ids = torch.tensor([[BOS]+tok.encode(it['prompt'] if task=='A' else it['prompt_B'], add_special_tokens=False)], device=dev)
            x0 = start(M, [it], ids.shape[1], 11000000, dev).double()
            with torch.no_grad(): e, fc = prep(ids)
            def checkpoint(k, logits):
                check_deadline(deadline)
                with torch.no_grad(): ref = M(ids, input_states=x0.clone(), num_steps=k).logits[:, -1]
                err = centered_error(logits, ref); gates.append(err)
                assert err <= 1e-6, f'G1 r={k}: {err}'
            x, b = base_state(step, read, x0, e, fc, tokens, deadline, checkpoint if (i == 0 or task == 'B') else None)   # B: lengths differ, so G1 on every context
            U, gn, G, pairs = directions(M,tok,read,x,fc,tokens,task,it['pid'],cfg.get('rows',64),deadline)
            result = selfgain(step,x,e,fc,U,cfg.get('rows',64),deadline)
            with torch.no_grad():
                lg = read(x, fc)[1][0]
                result.update(gnorm=gn, cos_x=(U@x[0,-1])/x[0,-1].norm(), G_task=G,
                              drift=b['base_drift'][0], xnorm=x[0,-1].norm(), xq=x[0,-1])
            check_deadline(deadline)
            for k,v in result.items(): out[k][i] = v.detach().cpu().numpy()
            out['C_tokens'][i] = pairs; out['argmax'][i] = int(lg.argmax())
            out['correct'][i] = (int(lg.argmax()) == one_token(tok,str(it['ans']))) if task=='A' else (bool(lg[tokens[0]] > lg[tokens[1]]) == bool(it['truth']))
            out['done'][i] = True
            if dev.type == 'cuda': torch.cuda.synchronize()
            timings.append(time.monotonic()-t)
            peak = torch.cuda.max_memory_allocated()/2**30 if dev.type == 'cuda' else 0
            print(f'{task} {i+1}/{n}: elapsed {time.monotonic()-began:.2f}s, {timings[-1]:.3f}s/context, peak {peak:.2f} GiB',flush=True)
        except TimeoutError:
            error = f'internal deadline before context {i} was complete'; break
        except Exception as exc:   # keep the completed contexts; an incomplete set gets no reading
            error = f'context {i}: {type(exc).__name__}: {exc}'[:500]; print(task, 'STOPPED', error, flush=True); break
    return out, dict(G1=gates, context_s=timings, elapsed_s=time.monotonic()-began, error=error)


def gpu_run(mode, cfg, run_id):
    began = time.monotonic(); deadline = began+COMPUTE_S
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    M = AutoModelForCausalLM.from_pretrained(MODEL[0], revision=MODEL[1], trust_remote_code=True,
             torch_dtype=torch.float64).to('cuda').double().eval().requires_grad_(False)
    precision = patch_precision(M)
    tok = AutoTokenizer.from_pretrained(MODEL[0], revision=MODEL[1])
    setup = time.monotonic()-began
    A = [p for p in make_items() if p['split']=='eval'][:cfg['n_A']]
    B = [dict(p, pid=900000+p['uid']) for p in cfg['items_B']][:cfg['n_B']]
    out = {}; timings = {}; gates = []
    for name, items in [('A',A),('B',B)]:
        z, tm = run_set(M,tok,items,name,cfg,deadline,began)
        out.update({name+'_'+k:v for k,v in z.items()}); timings[name] = tm; gates += tm['G1']
    meta = dict(cfg=cfg, run_id=run_id, model_pin=MODEL, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                gates=dict(G1=gates), timings=timings, setup_s=setup, runtime_s=time.monotonic()-began,
                precision=precision, versions=dict(torch=str(torch.__version__), transformers=transformers.__version__, numpy=np.__version__),
                status='complete' if out['A_done'].all() and out['B_done'].all() else 'partial',
                errors={k:timings[k]['error'] for k in timings})
    meta=json.loads(json.dumps(meta))   # plain types only: torch's version object cannot be unpickled without torch
    buf=io.BytesIO(); np.savez_compressed(buf,**out)
    try:   # the volume copy is a backup; the payload is returned even if it cannot be written
        root=Path('/results')/run_id; root.mkdir(parents=True,exist_ok=True)
        (root/'lread.npz').write_bytes(buf.getvalue()); (root/'meta.json').write_text(json.dumps(meta,indent=2))
        results_vol.commit()
    except Exception as exc:
        meta['volume_error']=f'{type(exc).__name__}: {exc}'[:300]
    return dict(npz=buf.getvalue(),meta=meta)


COMPUTE_S = 235    # computation stops here; the function is killed at TIMEOUT_S = 300 (five GPU-minutes)
TIMEOUT_S = 300


def main(mode: str = 'smoke'):
    assert mode in ('smoke','run')
    root=Path(__file__).resolve().parent.parent/'results'
    cfg=dict(smoke=mode=='smoke',rows=64,n_A=3 if mode=='smoke' else 40,n_B=3 if mode=='smoke' else 40,K0=64)
    if mode=='run':
        candidates=sorted(root.glob('lread-smoke-*/meta.json'))
        assert candidates, 'Run and save a smoke experiment first'
        path=candidates[-1]; sm=json.loads(path.read_text())
        assert sm['status']=='complete' and sm['cfg']['smoke']
        here=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        assert sm['code_sha256']==here and tuple(sm['model_pin'])==tuple(MODEL), 'smoke run used other code or another model pin'
        ssum=json.loads((path.parent/'lread_summary.json').read_text())   # written by analyze_lread.py
        assert ssum['A']['gates']['PASS'] and ssum['B']['gates']['PASS'], 'smoke analysis gates did not pass'
        import numpy as np
        # Projection with headroom: the slowest smoke context of each set, times 1.25, against the compute deadline.
        a,b=[1.25*float(np.max(sm['timings'][s]['context_s'][1:] or sm['timings'][s]['context_s'])) for s in ('A','B')]
        first=float(sm['timings']['A']['context_s'][0])
        budget=COMPUTE_S-1.25*sm['setup_s']-first-40*a
        assert budget>=b, 'Smoke projection leaves no room for Set B'
        cfg['n_B']=min(40,int(budget/b)); cfg['smoke_source']=str(path)
        cfg['projected_s']=1.25*sm['setup_s']+first+40*a+cfg['n_B']*b
    cfg['items_B']=json.loads((root/'freeprobe-20261107-130520/items.json').read_text())
    run_id='lread-'+mode+'-'+time.strftime('%Y%m%d-%H%M%S',time.gmtime())
    result=worker.remote(mode,cfg,run_id)
    dst=root/run_id; dst.mkdir(parents=True,exist_ok=True)
    (dst/'lread.npz').write_bytes(result['npz']); (dst/'meta.json').write_text(json.dumps(result['meta'],indent=2))
    print(dst, json.dumps(result['meta'],indent=2))


try:
    import modal
except ImportError:
    modal=None
if modal is not None:
    from llever_app import image, hf_cache, results_vol
    app=modal.App('lread-readspace')
    image=image.add_local_python_source('llever_app')
    worker=app.function(image=image,gpu='A100-40GB',volumes={'/cache':hf_cache,'/results':results_vol},timeout=TIMEOUT_S)(gpu_run)
    main=app.local_entrypoint()(main)
