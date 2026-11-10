"""Fixed-design Recurrent-Llama steering experiment. CPU numerics import without Modal.
See docs/llever_design.md. No job runs on import.
"""
from __future__ import annotations
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import random
import time

MODEL = ("smcleish/Recurrent-Llama-3.2-train-recurrence-32", "a5f6f126e9d9302445346844f1d7d29d111a06eb")
BOS = 128000
# NOTE(design): disjoint means no demo operand occurs in ANY query, in either context.
DEMOS = (((11, 12), (13, 14), (15, 16), (17, 18)),
         ((10, 19), (11, 17), (12, 16), (13, 15)))


def make_items(seed=20261110):
    rng = random.Random(seed); cells = {}
    for u in range(10):
        for ctx in range(2):
            a = list(range(1, 10)); rng.shuffle(a); cells[u, ctx] = a
    out = []
    for si, (split, offset, reps) in enumerate((("eval", 0, 2), ("train", 2, 2), ("fit", 4, 5))):
        for rep in range(reps):
            for u in range(10):
                for ctx in range(2):
                    j = rep * 20 + u * 2 + ctx; a = cells[u, ctx][offset + rep]
                    x = (20 if ctx == 0 else 40) + a; y = 70 + u - x
                    prompt = "".join(f"{v} + {w} = {v+w}\n" for v, w in DEMOS[ctx]) + f"{x} + {y} = "
                    out.append(dict(pid=9000000 + 1000*si+j, split=split, j=j, ctx=ctx, a=a,
                                    u=u, x=x, y=y, prompt=prompt, ans=70+u))
    return out

# Copied unchanged from mon_app.py; checked by llever_local_test.
def one_token(tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    assert len(ids) == 1, (text, ids)
    return ids[0]

# Copied unchanged from mon_app.py; checked by llever_local_test.
def start(M, items, L, seed, dev):
    import torch
    std = float(M.config.init_values["std"]); xs = []
    for it in items:
        g = torch.Generator("cpu").manual_seed(seed + it["pid"])
        x = torch.empty((L, M.config.n_embd), dtype=torch.float32)
        torch.nn.init.trunc_normal_(x, std=std, a=-3 * std, b=3 * std, generator=g)
        xs.append(x)
    return torch.stack(xs).to(dev)

# Copied unchanged from mon_app.py; checked by llever_local_test.
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

    def read(x, fc):
        for blk in M.transformer.coda:
            x = blk(x, fc, dum, None, None)
        return x, M.lm_head(M.transformer.ln_f(x[:, -1])).double()
    return prep, step, read

# Copied unchanged from mon_app.py; checked by llever_local_test.
def centered_error(a, b):
    a, b = a.double(), b.double()
    return float(((a - a.mean(-1, keepdim=True)) - (b - b.mean(-1, keepdim=True))).abs().max())

# Copied unchanged from obs_app.py; checked by llever_local_test.
def patch_norms(M):
    """Both RMSNorm classes (Raven's and transformers' Llama) cast to float32 internally. For float64 inputs compute the
    same formula, w * x / sqrt(mean(x^2) + eps), in float64; other dtypes keep the original code. Returns the count."""
    import torch
    n = 0
    for mod in M.modules():
        cls = type(mod)
        if cls.__name__ not in ("RMSNorm", "LlamaRMSNorm"):
            continue
        n += 1
        if getattr(cls, "_orig_forward", None) is None:
            cls._orig_forward = cls.forward

            def fwd(self, x):
                if x.dtype != torch.float64:
                    return type(self)._orig_forward(self, x)
                eps = self.eps if hasattr(self, "eps") else self.variance_epsilon
                return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))
            cls.forward = fwd
    return n


def patch_precision(M):
    """Fix RMSNorm and rotary's unconditional float32 phase calculation."""
    import torch
    from types import MethodType
    from transformers.modeling_rope_utils import dynamic_rope_update
    count = patch_norms(M)
    rotary = M.rotary_emb
    assert type(rotary).__name__ == 'LlamaRotaryEmbedding' and getattr(rotary, 'rope_type', 'llama3') in ('default', 'llama3'), type(rotary)
    original = inspect.getsource(type(rotary).forward)
    # NOTE(design): casting the existing inv_freq buffer preserves the pinned model's
    # initialized frequencies, but computes phase matmul/trig in float64. Both G1
    # paths use this patch. Dynamic update is retained for non-default rope types.
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv = self.inv_freq[None, :, None].to(device=x.device, dtype=x.dtype)
        freqs = (inv @ position_ids[:, None, :].to(x.dtype)).transpose(1, 2)
        emb = torch.cat((freqs, freqs), -1)
        return emb.cos()*self.attention_scaling, emb.sin()*self.attention_scaling
    rotary.forward = MethodType(forward, rotary)
    source = inspect.getsource(type(M))
    module = inspect.getsource(inspect.getmodule(type(M)))
    return dict(norms=count, rotary_original=original,
                float_cast_lines=[s.strip() for s in module.splitlines()
                                  if '.float()' in s or 'float32' in s],
                model_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                attention='SDPA math backend; float64 Q/K/V; no eager float32 softmax',
                unused_casts='label loss and generation helpers are not called')


# Internal deadlines per container (seconds from function start; the function timeout is 450 s). Production total at most
# 300 + 340 + 4 * 250 = 1,640 s (27.3 min) by deadline.
DEADLINE = dict(smoke=390, train=300, pred=340, eval=250)


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('internal deadline (reserved time for meta and volume commit)')


def zread(read, x, fc, tokens):
    logits = read(x, fc)[1]
    z = logits[:, tokens]
    return z - z.mean(-1, keepdim=True)


def base_state(step, read, x, e, fc, tokens, deadline=float('inf'), checkpoint=None):
    import torch
    answers = {}; zs = {}
    with torch.no_grad():
        for t in range(1, 65):
            check_deadline(deadline); prev = x; x = step(x, e, fc)
            if checkpoint is not None and t in (4, 32):
                checkpoint(t, read(x, fc)[1])
            if t in (32, 64):
                logits = read(x, fc)[1]; answers[t] = logits.argmax(-1)
                z = logits[:, tokens]; zs[t] = z-z.mean(-1, keepdim=True)
        drift = (x[:, -1]-prev[:, -1]).norm(dim=-1)/x[:, -1].norm(dim=-1).clamp_min(1e-300)
    return x.detach(), dict(base_drift=drift, xq=x[:, -1], norms=x[:, -1].norm(dim=-1),
                           answer32=answers[32], answer64=answers[64], z32=zs[32], z=zs[64])


def readout_grads(read, x, fc, tokens):
    import torch
    out = []
    for k in range(len(tokens)):
        with torch.enable_grad():
            q = x[:, -1].detach().clone().requires_grad_(True)
            st = torch.cat((x[:, :-1].detach(), q[:, None]), 1)
            out.append(torch.autograd.grad(zread(read, st, fc, tokens)[:, k].sum(), q)[0].detach())
    return torch.stack(out, 1)


def chain(step, x, e, fc, vectors, T, kind, chunk=8, deadline=float('inf'), keep=None):
    """Fixed-state VJP or central-FD JVP; vectors [problem, direction, d]. `keep` (dict, optional) receives the partial
    sums sum_{s=1..t} for every t it has as a key (design section 8)."""
    import torch
    n, c, d = vectors.shape
    ri = torch.arange(n, device=x.device).repeat_interleave(c)
    v = vectors.reshape(-1, d).clone(); acc = torch.zeros_like(v)
    for t in range(T):
        parts = []
        for a in range(0, len(ri), chunk):
            check_deadline(deadline); ii = ri[a:a+chunk]; w = v[a:a+chunk]
            # Rotary frequencies broadcast over batch in the pinned harness.
            f = tuple(z if z.shape[0] == 1 else z[ii] for z in fc)
            if kind == 'vjp':
                with torch.enable_grad():
                    q = x[ii, -1].detach().clone().requires_grad_(True)
                    st = torch.cat((x[ii, :-1], q[:, None]), 1)
                    y = step(st, e[ii], f)[:, -1]
                    parts.append(torch.autograd.grad(y, q, w)[0].detach())
            else:
                assert kind == 'jvp'
                nv = w.norm(dim=-1).clamp_min(1e-300)
                eps = 1e-6*x[ii, -1].norm(dim=-1).clamp_min(1e-300)
                delta = w/nv[:, None]*eps[:, None]
                sp, sm = x[ii].clone(), x[ii].clone()
                sp[:, -1] += delta; sm[:, -1] -= delta
                with torch.no_grad():
                    y = (step(sp, e[ii], f)[:, -1]-step(sm, e[ii], f)[:, -1])/(2*eps[:, None])
                parts.append(y*nv[:, None])
        v = torch.cat(parts); acc += v
        if keep is not None and t+1 in keep: keep[t+1] = acc.reshape(n, c, d).clone()
    return acc.reshape(n, c, d), (v.norm(dim=-1)/acc.norm(dim=-1).clamp_min(1e-300)).reshape(n, c)


def forced(step, read, x, e, fc, tokens, pushes, T=48, chunk=8, deadline=float('inf')):
    import torch
    # NOTE(design): for T=96 retain the specified time course and also record the
    # endpoint at 96 for tail/sham checks and endpoint prediction diagnostics.
    times = sorted(set([1, 2, 4, 8, 16, 48, T]))
    times = [t for t in times if t <= T]
    n, c = len(x), len(pushes); zz, dr, ds, tops = [], [], [], []
    for a in range(0, n*c, chunk):
        ids = torch.arange(a, min(n*c, a+chunk), device=x.device)
        ii, cc = ids//c, ids % c; st = x[ii].clone(); reads = []
        f = tuple(z if z.shape[0] == 1 else z[ii] for z in fc)
        with torch.no_grad():
            for t in range(1, T+1):
                check_deadline(deadline); prev = st
                inp = st.clone(); inp[:, -1] += pushes[cc]
                # NOTE(design): input forcing gives J P on the first loop, hence
                # A=sum(J**t,t=1..T), NOT a sum beginning at the identity.
                st = step(inp, e[ii], f)
                if t in times:
                    reads.append(zread(read, st, f, tokens))
            zz.append(torch.stack(reads, 1)); tops.append(read(st, f)[1].argmax(-1))
            dr.append((st[:, -1]-prev[:, -1]).norm(dim=-1)/st[:, -1].norm(dim=-1).clamp_min(1e-300))
            ds.append((st[:, -1]-x[ii, -1]).norm(dim=-1)/x[ii, -1].norm(dim=-1).clamp_min(1e-300))
    return dict(Z=torch.cat(zz).reshape(n, c, len(times), len(tokens)),
                last_drift=torch.cat(dr).reshape(n, c), displacement=torch.cat(ds).reshape(n, c),
                top=torch.cat(tops).reshape(n, c), times=torch.tensor(times, device=x.device))


def row_spec(names, restricted=False):
    # NOTE(design): C/R secondary doses may be omitted by the prespecified smoke rule.
    return [(-1, 0.)]+[(k, rho) for k, name in enumerate(names) for rho in (.01, .03, .09)
                        if not restricted or name[0] in 'LG' or rho == .03]


def run_stage(M, tok, items, mode, cfg, push=None, deadline=float('inf')):
    """Shared real/fake-model path; all exported floating arrays stay float64."""
    import numpy as np
    import torch
    dev = next(M.parameters()).device
    ids0 = [[BOS]+tok.encode(it['prompt'], add_special_tokens=False) for it in items]
    assert len({len(v) for v in ids0}) == 1
    assert all(tok.decode(v[-1:]) == ' ' for v in ids0)
    tokens = [one_token(tok, str(70+k)) for k in range(10)]
    ids = torch.tensor(ids0, device=dev); prep, step, read = harness(M, dev)
    x0 = start(M, items, ids.shape[1], 11000000, dev).double()
    gates = []; t0 = time.monotonic()
    with torch.no_grad():
        states, bases, embeddings = [], [], []
        for a in range(0, len(ids), cfg.get('base_chunk', 32)):
            check_deadline(deadline)
            ib = ids[a:a+cfg.get('base_chunk', 32)]; xb = x0[a:a+len(ib)]
            eb, fc = prep(ib)
            assert eb.dtype == torch.float64 and all(z.dtype == torch.float64 for z in fc)
            def checkpoint(t, logits):
                check_deadline(deadline)
                ref = M(ib, input_states=xb.clone(), num_steps=t).logits[:, -1]
                err = centered_error(logits, ref); gates.append(err)
                assert err <= 1e-6, f'G1 r={t}: {err}'
            xx, bb = base_state(step, read, xb, eb, fc, tokens, deadline, checkpoint if a == 0 else None)
            states.append(xx); embeddings.append(eb); bases.append(bb)
        x = torch.cat(states); e = torch.cat(embeddings); assert all(z.shape[0] == 1 for z in fc), 'shared rotary table'
        base = {k:torch.cat([b[k] for b in bases]) for k in bases[0]}
    base_s = time.monotonic()-t0; t0 = time.monotonic()
    out = dict(base, pid=np.array([p['pid'] for p in items]),
               split=np.array([p['split'] for p in items]), ans=np.array([p['ans'] for p in items]),
               ans_token=np.array([one_token(tok, str(p['ans'])) for p in items]))
    if mode in ('train', 'pred'):
        # Fit states are used only for centroids, never gradients or VJP training.
        take = [i for i,p in enumerate(items) if p['split'] != 'fit']
        ii = torch.tensor(take, device=dev)
        f = tuple(z if z.shape[0] == 1 else z[ii] for z in fc)
        g = readout_grads(read, x[ii], f, tokens); out['G'] = g
        v = g if mode == 'train' else torch.as_tensor(push['P'][:10], device=dev).double()[None].expand(len(ii), -1, -1)
        keep = {int(t): None for t in cfg.get('partial', [])} if mode == 'pred' else None
        out['AtG' if mode == 'train' else 'AP'], out['tail'] = chain(
            step, x[ii], e[ii], f, v, cfg['T'], 'vjp' if mode == 'train' else 'jvp', cfg['chunk'], deadline, keep or None)
        if keep:  # partial operator sums A_t L_k = (J + ... + J^t) L_k, [problem, t, k, d]
            out['AP_partial'] = torch.stack([keep[t] for t in sorted(keep)], 1); out['partial_t'] = np.array(sorted(keep))
        out['gradient_pid'] = out['pid'][take]
    else:
        rows = row_spec(push['names'], cfg.get('restrict_cr', False))
        P = torch.as_tensor(push['P'], device=dev).double()
        doses = torch.stack([torch.zeros_like(P[0]) if k < 0 else P[k]*rho*float(push['m_ref']) for k,rho in rows])
        out.update(forced(step, read, x, e, fc, tokens, doses, cfg['T'], cfg['chunk'], deadline))
        out.update(row_push=np.array([k for k,r in rows]), row_rho=np.array([r for k,r in rows]))
    stage_s = time.monotonic()-t0
    out = {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k,v in out.items()}
    return out, dict(G1=gates, base_s=base_s, stage_s=stage_s, tokens=ids.shape[1],
                     base_rows=len(items), stage_rows=(len(items)*len(rows) if mode=='eval' else len(take)*(10 if mode=='train' else len(push['P'][:10]))))


def smoke(M, tok, cfg, deadline):
    import numpy as np
    all_items = make_items(); tr = [p for p in all_items if p['split']=='train'][:2]
    ev = [p for p in all_items if p['split']=='eval'][:2]
    train, tm = run_stage(M, tok, tr, 'train', cfg, deadline=deadline)
    def nz(v): return v/np.linalg.norm(v, axis=-1, keepdims=True)
    # NOTE(design): four smoke pushes are L0,L1,G0,G1; two train items cannot
    # supply all ten fit centroids. Smoke is diagnostic, never a production gate.
    P = np.vstack((nz(nz(train['AtG'][:, :2]).mean(0)), nz(nz(train['G'][:, :2]).mean(0))))
    push = dict(P=P, names=np.array(['L0','L1','G0','G1']), m_ref=np.median(train['norms']))
    pred, pm = run_stage(M, tok, ev+tr, 'pred', cfg, push, deadline)
    evaluation, em = run_stage(M, tok, ev, 'eval', cfg, push, deadline)
    lhs = np.einsum('nkd,md->nkm', train['AtG'], P)
    rhs = np.einsum('nkd,nmd->nkm', pred['G'][2:], pred['AP'][2:])
    err = float(np.median(abs(lhs-rhs)/(abs(lhs)+abs(rhs)+1e-9)))
    tail = float(np.median(np.concatenate([train['tail'].ravel(), pred['tail'].ravel()])))
    chosen_T = 96 if tail > 1e-4 else 48
    bench = benchmark(M, tok, deadline)
    return {**{'train_'+k:v for k,v in train.items()}, **{'pred_'+k:v for k,v in pred.items()},
            **{'eval_'+k:v for k,v in evaluation.items()}, 'P':P}, dict(
            G1=tm['G1']+pm['G1']+em['G1'], tail_median=tail, adjoint_error=err, benchmark=bench,
            recommended_cfg=dict(T=chosen_T), stages=dict(train=tm,pred=pm,eval=em))


def benchmark(M, tok, deadline):
    """Production-sized timing (Astra pre-run finding 2): base states for one chunk of 32 problems, readout gradients,
    and two loops of 130 rows (chunk 128) for the VJP chain, the two-pass FD-JVP chain and the forced evaluation;
    CUDA synchronised, peak memory recorded. Forced timing reads the output every loop, so it overestimates."""
    import torch
    dev = next(M.parameters()).device; cuda = dev.type == 'cuda'
    def clock():
        if cuda: torch.cuda.synchronize()
        return time.monotonic()
    items = [p for p in make_items() if p['split'] == 'fit'][:32]
    ids = torch.tensor([[BOS]+tok.encode(p['prompt'], add_special_tokens=False) for p in items], device=dev)
    tokens = [one_token(tok, str(70+k)) for k in range(10)]
    prep, step, read = harness(M, dev); x0 = start(M, items, ids.shape[1], 11000000, dev).double()
    if cuda: torch.cuda.reset_peak_memory_stats()
    t = clock()
    with torch.no_grad():
        e, fc = prep(ids)
    x, _ = base_state(step, read, x0, e, fc, tokens, deadline)
    base = (clock()-t)/len(items)
    t = clock(); g = readout_grads(read, x, fc, tokens); grads = (clock()-t)/len(items)
    n = 13; rl = {}
    for kind in ('vjp', 'jvp'):
        t = clock(); chain(step, x[:n], e[:n], fc, g[:n], 2, kind, 128, deadline); rl[kind] = (clock()-t)/(n*10*2)
    P = g[0]/g[0].norm(dim=-1, keepdim=True)
    t = clock(); forced(step, read, x[:n], e[:n], fc, tokens, P, 2, 128, deadline); rl['forced'] = (clock()-t)/(n*10*2)
    return dict(base_s_per_problem=base, grads_s_per_problem=grads, row_loop_s=rl, rows=n*10, chunk=128,
                peak_gb=torch.cuda.max_memory_allocated()/2**30 if cuda else None)


def project(bench, setup_s, T):
    """Projected seconds per production container from the benchmark; `fits` compares with the internal deadlines."""
    rl = bench['row_loop_s']; b = bench['base_s_per_problem']; g = bench['grads_s_per_problem']
    est = dict(train=setup_s+140*b+40*g+400*T*rl['vjp'], pred=setup_s+80*b+80*g+800*T*rl['jvp'],
               eval_shard=setup_s+10*b+970*T*rl['forced'], eval_shard_restricted=setup_s+10*b+730*T*rl['forced'])
    total = est['train']+est['pred']+4*est['eval_shard']; restricted = est['train']+est['pred']+4*est['eval_shard_restricted']
    return dict(seconds=est, total_seconds=total, restricted_total_seconds=restricted, T=T,
                restrict_cr=bool(total > 26*60 or est['eval_shard'] > DEADLINE['eval']-40),
                fits=dict(train=est['train'] < DEADLINE['train']-40, pred=est['pred'] < DEADLINE['pred']-40,
                          eval_shard=est['eval_shard'] < DEADLINE['eval']-40,
                          eval_shard_restricted=est['eval_shard_restricted'] < DEADLINE['eval']-40))


def gpu_run(mode, cfg, run_id, pushes=b'', shard=0):
    """A failure or internal deadline writes metadata (status failed) and an empty NPZ; no partial arrays are kept."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    began = time.monotonic(); deadline = began+DEADLINE[mode]
    out = {}; items = []; meta = dict(cfg=cfg, mode=mode, shard=shard, run_id=run_id,
        model_revision=MODEL[1], code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        push_file_sha256=hashlib.sha256(pushes).hexdigest() if pushes else None,
        status='running', gates={}, timings={})
    # Modal requires absolute mount points; all local artifact paths are relative.
    root = Path(os.sep)/'results'/run_id; root.mkdir(parents=True, exist_ok=True)
    stem = f'shard{shard}' if mode=='eval' else 'llever'
    try:
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
        M = AutoModelForCausalLM.from_pretrained(MODEL[0], revision=MODEL[1], trust_remote_code=True,
                                                torch_dtype=torch.float64).to('cuda').double().eval().requires_grad_(False)
        meta['precision'] = patch_precision(M)
        tok = AutoTokenizer.from_pretrained(MODEL[0], revision=MODEL[1])
        if mode=='smoke':
            setup_s = time.monotonic()-began
            out, timing = smoke(M, tok, cfg, deadline)
            timing['projection'] = project(timing['benchmark'], setup_s, timing['recommended_cfg']['T'])
            timing['recommended_cfg']['restrict_cr'] = timing['projection']['restrict_cr']
        else:
            items = make_items()
            wanted = {'train':('train','fit'), 'pred':('eval','train'), 'eval':('eval',)}[mode]
            items = [p for p in items if p['split'] in wanted]
            if mode=='eval': items = items[10*shard:10*(shard+1)]  # Draft 1.3: four shards of ten problems
            push = dict(np.load(io.BytesIO(pushes), allow_pickle=False)) if pushes else None
            if push is not None:
                assert push['P'].shape == (32, M.config.n_embd)
                assert np.allclose(np.linalg.norm(push['P'], axis=1), 1)
                assert int(push['T']) == cfg['T'], 'push training T mismatch'
            out, timing = run_stage(M, tok, items, mode, cfg, push, deadline)
        meta.update(timings=timing, gates=dict(G1=timing['G1']), status='complete')
    except Exception as ex:
        meta.update(status='failed', error=repr(ex))
    finally:
        meta['runtime_s'] = time.monotonic()-began
        buf = io.BytesIO(); np.savez_compressed(buf, **out)
        (root/(stem+'.npz')).write_bytes(buf.getvalue())
        (root/(stem+'_meta.json')).write_text(json.dumps(meta, indent=2))
        if mode!='eval': (root/'meta.json').write_text(json.dumps(meta, indent=2))  # eval shards keep shard-specific metadata
        results_vol.commit()
    return dict(npz=buf.getvalue(), meta=meta, items=items)


try:
    import modal
except ImportError:  # numerical routines remain importable in a CPU-only environment
    modal = None
if modal is not None:
    app = modal.App('llever')
    hf_cache = modal.Volume.from_name('huginn-hf-cache', create_if_missing=True)
    results_vol = modal.Volume.from_name('huginn-results', create_if_missing=True)
    image = (modal.Image.debian_slim(python_version='3.11')
             .pip_install('torch==2.6.0', 'transformers==4.53.1', 'safetensors', 'huggingface_hub[hf_transfer]',
                          'numpy', 'pandas', 'pyarrow')
             .env({'HF_HOME':'/cache/hf', 'HF_HUB_ENABLE_HF_TRANSFER':'1'}))
    worker = app.function(image=image, gpu='A100-40GB', volumes={'/cache':hf_cache, '/results':results_vol},
                          timeout=450)(gpu_run)

    @app.local_entrypoint()
    def main(mode: str, train: str='', pushes: str='', pred: str='', t: int=48, restrict_cr: bool=False, chunk: int=128, partial: str=''):
        import numpy as np
        assert mode in ('smoke','train','pred','eval') and t in (48,96)
        # NOTE(design): train/pred arguments validate upstream settings/provenance;
        # no training arrays or eval data are loaded into the GPU to fit directions.
        cfg = dict(T=t, restrict_cr=restrict_cr, chunk=chunk, K0=64)
        if partial:
            assert mode=='pred' and all(1 <= int(v) <= t for v in partial.split(','))
            cfg['partial'] = sorted({int(v) for v in partial.split(',')})
        if mode in ('pred','eval'): assert pushes and train, '--pushes and --train are required'
        if mode=='eval': assert pred, '--pred is required'
        for name, path in [('train',train),('pred',pred)]:
            if not path: continue
            upstream = json.loads((Path(path)/'meta.json').read_text())
            assert upstream['status']=='complete' and upstream['cfg']['T']==t, name
            cfg[name+'_run'] = path
            if name=='pred':
                assert upstream['push_file_sha256']==hashlib.sha256(Path(pushes).read_bytes()).hexdigest(), 'pred used another push file'
        raw = Path(pushes).read_bytes() if pushes else b''
        run_id = 'llever_'+mode+'-'+time.strftime('%Y%m%d-%H%M%S', time.gmtime())
        dst = Path(__file__).resolve().parent.parent/'results'/run_id; dst.mkdir(parents=True)
        calls = [worker.spawn(mode,cfg,run_id,raw,s) for s in range(4 if mode=='eval' else 1)]
        results = []
        for i,c in enumerate(calls):  # every returned shard is saved at once; a dead container does not lose the other
            try: r = c.get()
            except Exception as ex:
                r = dict(npz=b'', meta=dict(status='failed', error=repr(ex), cfg=cfg, mode=mode, shard=i, gates={}), items=[])
            results.append(r)
            if mode=='eval':
                (dst/f'shard{i}.npz').write_bytes(r['npz'])
                (dst/f'shard{i}_meta.json').write_text(json.dumps(r['meta'], indent=2))
        meta = json.loads(json.dumps(results[0]['meta']))
        if mode=='eval':
            if all(r['meta']['status']=='complete' for r in results):
                arrays = [dict(np.load(io.BytesIO(r['npz']))) for r in results]
                merged = {k: arrays[0][k] if k in ('times','row_push','row_rho') else
                          np.concatenate([a[k] for a in arrays], axis=0) for k in arrays[0]}
                np.savez_compressed(dst/'llever.npz', **merged)
            meta['shards'] = [r['meta'] for r in results]
            meta['status'] = 'complete' if all(r['meta']['status']=='complete' for r in results) else 'failed'
            meta.setdefault('gates',{})['G1'] = [g for r in results for g in r['meta']['gates'].get('G1',[])] if meta['status']=='complete' else []
        elif results[0]['npz']: (dst/'llever.npz').write_bytes(results[0]['npz'])
        (dst/'meta.json').write_text(json.dumps(meta, indent=2))
        (dst/'items.json').write_text(json.dumps([p for r in results for p in r['items']], indent=2))
        if raw: (dst/'pushes.npz').write_bytes(raw)
        print(dst, json.dumps(meta, indent=2))
        assert meta['status']=='complete', meta
