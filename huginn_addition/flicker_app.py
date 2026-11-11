"""Fixed-design Huginn flicker scan and confirmation.

Loop k is zero based: after h.step(e,s,k), budget b=k+1. Tail entry j
is budget K-tail+1+j (449..512 for K=512, tail=64); parity refers to b.
Lead = top log probability minus second log probability (nonnegative).
No result may be reported for a prompt that did not complete all K loops.
"""
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))


SCAN_TIMEOUT = 480
CONFIRM_TIMEOUT = 420
CONFIRM64_TIMEOUT = 420
RESERVE = 45
START_POLICY = 'census draw (uid, seed 0), full-length, sliced'
CENSUS_RUNS = ['census-20261105-162542', 'census-20261105-163100',
               'census-20261105-163615', 'census-20261105-164134']


from analyze_flicker import canonical_hits, hits_sha


def deadline_at(entry, loaded, timeout, compute_s):
    return min(entry + timeout - RESERVE, loaded + compute_s)


def production_allowed(kind, smoke_meta, local_sha, ordinary_loops=448, tail_loops=64, batch_scale=1):
    if not smoke_meta: return False, 'Missing matching smoke run; run '+kind+'_smoke first'
    if smoke_meta.get('code_sha256') != local_sha: return False, 'Smoke code sha256 differs'
    if smoke_meta.get('ok') is not True: return False, 'Smoke run is not ok'
    if kind in ('scan', 'screen2', 'scan2'):
        try:
            times = [float(smoke_meta[k]) for k in ('load_s', 'loop_s', 'tail_loop_s')]
            if not all(__import__('math').isfinite(t) and t >= 0 for t in times): raise ValueError()
            loops = batch_scale*(ordinary_loops*times[1] + tail_loops*times[2])
            projection = times[0] + loops
            compute = 200 if kind == 'screen2' else 340
        except (KeyError, ValueError, TypeError): return False, 'Missing/invalid smoke timing'
        message = (f'Loop projection: {loops:.2f}s (limit {0.8*compute:.2f}s); '
                   f'load + loops: {projection:.2f}s (limit {0.8*(SCAN_TIMEOUT-RESERVE):.2f}s)')
        return loops <= 0.8*compute and projection <= 0.8*(SCAN_TIMEOUT-RESERVE), message
    return True, 'Matching smoke passed'


def configure_precision():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def precision_meta(h):
    import torch
    return dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                parameter_dtype=str(next(h.model.parameters()).dtype) if h is not None else None,
                peak_memory_allocated=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)


def sync():
    import torch
    if torch.cuda.is_available(): torch.cuda.synchronize()


def patch_double(h):
    """Audit and replace the two pinned float32 casts, locally to this worker."""
    import inspect
    import torch
    norms = [m for m in h.model.modules() if type(m).__name__ == 'RMSNorm']
    classes = {type(m) for m in norms}
    assert len(classes) == 1, 'Expected a single RMSNorm class'
    cls = classes.pop(); original_norm = cls.forward
    norm_source = inspect.getsource(original_norm)
    assert 'x.float()' in norm_source
    module = sys.modules[type(h.model).__module__]
    original_rotary = module.apply_rotary_emb_complex_like
    rotary_source = inspect.getsource(original_rotary)
    assert '.float()' in rotary_source
    evidence = dict(norm_modules=len(norms), norm_source_sha256=hashlib.sha256(norm_source.encode()).hexdigest(),
                    rotary_source_sha256=hashlib.sha256(rotary_source.encode()).hexdigest(), dtypes={})
    def forward(self, x):
        evidence['dtypes']['norm_input'] = str(x.dtype)
        if x.dtype != torch.float64: return original_norm(self, x)
        out = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)) * self.weight
        evidence['dtypes']['norm_output'] = str(out.dtype)
        return out
    def rotary(q, k, freqs_cis):
        evidence['dtypes']['rotary_input'] = str(q.dtype)
        if q.dtype != torch.float64: return original_rotary(q, k, freqs_cis)
        x = torch.cat([q, k], dim=2).unflatten(-1, (-1, 2))
        f = freqs_cis.to(q.dtype)
        y = torch.stack([x[..., 0]*f[..., 0]-x[..., 1]*f[..., 1],
                         x[..., 1]*f[..., 0]+x[..., 0]*f[..., 1]], -1).flatten(3)
        evidence['dtypes']['rotary_constants'] = str(freqs_cis.dtype)
        evidence['dtypes']['rotary_output'] = str(y.dtype)
        return torch.split(y, q.shape[2], dim=2)
    def restore():
        cls.forward = original_norm
        module.apply_rotary_emb_complex_like = original_rotary
    h._flicker_restore_double = restore
    cls.forward = forward
    module.apply_rotary_emb_complex_like = rotary
    return evidence


def coda_logits64(h, s, gidx):
    import torch
    x = h.ln_f(s)
    freqs = h.model.freqs_cis[:, :s.shape[1]]
    bi = torch.tensor(0, dtype=torch.long)
    for block in h.model.transformer.coda:
        bi -= 1
        x = block(x, freqs, bi, None, None)
    x = h.ln_f(x)
    hidden = torch.gather(x, 1, gidx[..., None].expand(-1, -1, x.shape[-1]))
    logits = h.model.lm_head(hidden)
    assert logits.dtype == torch.float64
    return logits


def load_references(problems, root=Path('/results')):
    import numpy as np
    refs, hashes, errors = {}, {}, []
    for run in CENSUS_RUNS:
        path = root/run/'census.npz'
        try:
            raw = path.read_bytes(); hashes[run] = hashlib.sha256(raw).hexdigest()
            with np.load(io.BytesIO(raw), allow_pickle=True) as z:
                for i, uid in enumerate(z['uid']):
                    uid = int(uid)
                    if uid in refs: raise ValueError(f'duplicate census uid {uid}')
                    refs[uid] = {k: z[k][i] for k in ('ids', 'pred_pos', 'steps', 'ro_argmax')}
        except Exception as exc: errors.append(f'{path}: {type(exc).__name__}: {exc}')
    checks = {}
    for p in problems:
        if p['source'] != 'census': continue
        r = refs.get(p['uid'])
        checks[p['uid']] = bool(r is not None and np.array_equal(r['ids'], p['ids']) and
            np.array_equal(r['pred_pos'], p['pred_pos']) and int(r['steps']) == 512 and
            np.shape(r['ro_argmax']) == (64, len(p['pred_pos'])))
        if not checks[p['uid']]: errors.append(f'reference preflight failed uid={p["uid"]}')
    return refs, hashes, checks, errors

def metadata(cfg, began, compute_start, validation=None, **extra):
    import numpy as np
    import torch
    import transformers
    return json.loads(json.dumps(dict(cfg=cfg, validation=validation,
        runtime_s=time.monotonic()-began, compute_s=time.monotonic()-compute_start,
        revision=REVISION, model_id=MODEL_ID,
        versions=dict(numpy=str(np.__version__), torch=str(torch.__version__), transformers=str(transformers.__version__)),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **extra)))


def save_result(result, run_id, scan=False):
    """A volume failure must never discard the return payload."""
    try:
        root = Path('/results')/run_id
        root.mkdir(parents=True, exist_ok=True)
        if scan: (root/'scan.npz').write_bytes(result['npz'])
        else: (root/'confirm.json').write_text(json.dumps(result, indent=2))
        (root/'meta.json').write_text(json.dumps(result['meta'], indent=2))
        results_vol.commit()
    except Exception as exc:
        result['meta']['volume_error'] = f'{type(exc).__name__}: {exc}'
        result['meta']['ok'] = False
        result['meta'].setdefault('flags', []).append('volume write failed')
    return result


def scan(problems_json, cfg, run_id):
    began = time.monotonic()
    import numpy as np
    import torch
    configure_precision()
    h = None; val = []; flags = []; refs = {}; hashes = {}; checks = {}
    problems = json.loads(problems_json)
    K, tail = cfg['K'], cfg['tail']
    dr, lag, ti, lp, digits, full_agrees = [], [], [], [], {}, {}
    normal_times, tail_times = [], []
    ref_errors = []
    tc = began; load_s = 0
    try:
        h = H('float32'); tc = time.monotonic()
        deadline = deadline_at(began, tc, SCAN_TIMEOUT, cfg['compute_s'])
        h.prep(problems); val = h.validate(problems)
        load_s = time.monotonic()-began
        if cfg.get('preflight', True):
            refs, hashes, checks, ref_errors = load_references(problems)
            flags.extend(ref_errors)
        with torch.no_grad():
            ids, e, s = h.batch(problems, [0]*len(problems))
            g = torch.arange(s.shape[1], device=h.dev)[None].expand(len(problems), -1)
            previous = None
            for k in range(K):
                if time.monotonic() >= deadline: break
                sync(); started = time.monotonic()
                s2 = h.step(e, s, k); d = s2-s
                drift = (d.norm(dim=-1)/s2.norm(dim=-1)).cpu().numpy()
                lagrow = ((d+previous).norm(dim=-1)/d.norm(dim=-1).clamp_min(1e-30)).cpu().numpy() if k >= K-tail and previous is not None else None
                if k >= K-tail:
                    logits = h.coda_logits(s2, g)
                    values, indices = torch.log_softmax(logits, -1).topk(3, dim=-1)
                    top, logs = indices.cpu().numpy().astype(np.int32), values.cpu().numpy().astype(np.float32)
                    if k == K-1:
                        for r, p in enumerate(problems):
                            ds, agree = [], []
                            for j, pos in enumerate(p['pred_pos']):
                                tokens = h.SPD if j == 0 else h.DIG
                                digit = int(logits[r, pos, tokens].argmax())
                                ds.append(digit); agree.append(int(top[r, pos, 0]) == tokens[digit])
                            digits[r], full_agrees[r] = ds, agree
                    ti.append(top); lp.append(logs)
                dr.append(drift)
                if lagrow is not None: lag.append(lagrow)
                s, previous = s2, d
                sync(); (tail_times if k >= K-tail else normal_times).append(time.monotonic()-started)
    except Exception as exc: flags.append(f'scan: {type(exc).__name__}: {exc}')
    done = len(dr)
    if done != K: flags.append('incomplete')
    if not val or not all(v.get('argmax_agree') == 1 and np.isfinite(v.get('max_abs', np.nan)) and v['max_abs'] <= 1e-3 for v in val): flags.append('validation failed')
    keys = ('uid', 'source', 'ids', 'pred_pos', 'steps', 'drift', 'lag2_tail', 'top_ids', 'top_lp',
            'repro_ok', 'repro_mismatches', 'ref_preflight_ok', 'digit_argmax_last', 'full_vocab_agrees', 'tok_str', 'top_tok_str')
    out = {key: [] for key in keys}
    for r, p in enumerate(problems):
        n = len(p.get('ids', [])); pp = p.get('pred_pos', [])
        top = np.stack([a[r, :n] for a in ti]) if ti else np.empty((0, n, 3), np.int32)
        logs = np.stack([a[r, :n] for a in lp]) if lp else np.empty((0, n, 3), np.float32)
        ok, mismatch = None, None
        if cfg['reproduce']:
            if p['source'] == 'census':
                candidate = top[:, pp, 0]; ref = refs.get(p['uid'], {}).get('ro_argmax')
            else: candidate, ref = np.array(digits.get(r, [])), np.array(p['ref_answer_argmax_last'])
            ok = bool(done == K and K == 512 and tail == 64 and ref is not None and np.array_equal(candidate, ref))
            mismatch = 0 if ok else 1
            if not ok: flags.append(f'reproduction failed uid={p["uid"]}')
        row = dict(uid=p['uid'], source=p['source'], ids=np.array(p.get('ids', []), np.int32),
            pred_pos=np.array(pp, np.int32), steps=done,
            drift=np.stack([a[r, :n] for a in dr]) if dr else np.empty((0, n)),
            lag2_tail=np.stack([a[r, :n] for a in lag]) if lag else np.empty((0, n)),
            top_ids=top, top_lp=logs, repro_ok=ok, repro_mismatches=mismatch,
            ref_preflight_ok=checks.get(p['uid']), digit_argmax_last=digits.get(r, []), full_vocab_agrees=full_agrees.get(r, []),
            tok_str=[h.tok.decode([t]) for t in p.get('ids', [])] if h else [],
            top_tok_str={str(int(t)): h.tok.decode([int(t)]) for t in np.unique(top)} if h else {})
        for key in keys: out[key].append(row[key])
    flat = {}
    for key in keys:
        arr = np.empty(len(problems), dtype=object); arr[:] = out[key]; flat[key] = arr
    flat.update(K=K, tail=tail)
    buf = io.BytesIO(); np.savez_compressed(buf, **flat)
    meta = metadata(cfg, began, tc, val, run_id=run_id, flags=flags, ok=not flags,
                    load_s=load_s, loop_s=float(np.mean(normal_times)) if normal_times else None,
                    tail_loop_s=float(np.mean(tail_times)) if tail_times else None,
                    reference_sha256=hashes, reference_errors=ref_errors, precision='float32', **precision_meta(h))
    return save_result(dict(npz=buf.getvalue(), meta=meta), run_id, scan=True)


def initial(h, hit, seed=0):
    """Draw at FULL original length before truncating; verify embedding independence."""
    import torch
    ids = torch.tensor([hit['ids']], dtype=torch.long, device=h.dev)
    assert ids.shape[1] == hit['n_full']
    e, _ = h.model.embed_inputs(ids)
    full = h.init_state(torch.zeros((hit['n_full'], h.d), device=h.dev), hit['uid'], seed).to(e.dtype)
    assert torch.equal(full, h.init_state(e[0], hit['uid'], seed)), 'init depends on embedding values'
    return full


def read_record(h, logits, budget, tokens=()):
    import torch
    all_lp = torch.log_softmax(logits, -1)
    lp, ids = all_lp.topk(3)
    return dict(budget=budget, top_ids=ids.tolist(), top_lp=lp.tolist(), pair_lp=all_lp[list(tokens)].tolist())


def prefix_batch(h, hit, starts):
    import torch
    ids = torch.tensor([hit['ids'][:hit['p']+1]]*len(starts), dtype=torch.long, device=h.dev)
    e, _ = h.model.embed_inputs(ids)
    s = torch.stack([a[:hit['p']+1] for a in starts])
    g = torch.full((len(starts), 1), hit['p'], dtype=torch.long, device=h.dev)
    return ids, e, s, g


def confirm(hits_json, cfg, run_id):
    began = time.monotonic()
    import torch
    configure_precision()
    h = None; tc = began; rows = []; flags = []
    try:
        h = H('float32'); tc = time.monotonic()
        deadline = deadline_at(began, tc, CONFIRM_TIMEOUT, cfg['compute_s'])
        rows = [dict(hit, C1=[], C2=[], C3=[]) for hit in json.loads(prepare_hits(h, hits_json))]
        K = cfg['K']
        parts = cfg.get('parts', ['C1', 'C2', 'C3'])  # 'continue' mode (added after the results) runs C3 only
        with torch.no_grad():
            try:
                estimate = float('inf'); longest = 0
                if 'C1' not in parts: rows_c1 = []
                else: rows_c1 = rows
                if rows_c1 and time.monotonic() < deadline:
                    start = initial(h, rows[0]); ids, _, s, _ = prefix_batch(h, rows[0], [start])
                    sync(); t = time.monotonic()
                    h.model(ids, input_states=s.clone(), num_steps=32)
                    sync(); estimate = (time.monotonic()-t)*max(cfg['budgets'])/32*1.5
                for row in rows_c1:
                    start = initial(h, row); ids, _, s, _ = prefix_batch(h, row, [start])
                    for b in cfg['budgets']:
                        if time.monotonic()+estimate >= deadline:
                            flags.append('C1 deadline admission'); break
                        sync(); t = time.monotonic()
                        logits = h.model(ids, input_states=s.clone(), num_steps=b).logits[0, -1]
                        row['C1'].append(read_record(h, logits, b, row['tokens']))
                        sync(); longest = max(longest, time.monotonic()-t); estimate = 1.5*longest
            except Exception as exc: flags.append(f'C1: {type(exc).__name__}: {exc}')
            try:
                for row in (rows if 'C2' in parts else []):
                    if time.monotonic() >= deadline: break
                    starts = [initial(h, row, seed) for seed in cfg['seeds']]
                    _, e, s, g = prefix_batch(h, row, starts)
                    records = [dict(seed=seed, budgets=[], top_ids=[], top_lp=[], pair_lp=[]) for seed in cfg['seeds']]
                    row['C2'] = records
                    for k in range(K):
                        if time.monotonic() >= deadline: break
                        s = h.step(e, s, k)
                        if k >= K-cfg['tail_c2']:
                            all_lp = torch.log_softmax(h.coda_logits(s, g)[:, 0], -1)
                            vals, inds = all_lp.topk(2)
                            for r, rec in enumerate(records):
                                rec['budgets'].append(k+1); rec['top_ids'].append(int(inds[r, 0]))
                                rec['top_lp'].append(vals[r].tolist()); rec['pair_lp'].append(all_lp[r, row['tokens']].tolist())
            except Exception as exc: flags.append(f'C2: {type(exc).__name__}: {exc}')
            try:
                branches = []
                if 'C3' in parts and time.monotonic() < deadline:
                    for row in rows:
                        if time.monotonic() >= deadline: break
                        full = initial(h, row)
                        extension = h.init_state(torch.zeros((cfg['n_new']+1, h.d), device=h.dev), row['uid'], 1000).to(full.dtype)
                        start = torch.cat((full, extension))
                        for b in (K-1, K):
                            rec = dict(budget=b, token_ids=[], top_lp=[], text='', complete=False)
                            row['C3'].append(rec)
                            branches.append(dict(ids=row['ids'][:row['p']+1], start=start, rec=rec))
                for j in range(cfg['n_new']):
                    if time.monotonic() >= deadline or not branches: break
                    L = max(len(b['ids']) for b in branches)
                    ids = torch.full((len(branches), L), h.NL, dtype=torch.long, device=h.dev)
                    for r, branch in enumerate(branches): ids[r, :len(branch['ids'])] = torch.tensor(branch['ids'], device=h.dev)
                    e, _ = h.model.embed_inputs(ids); s = h.model.initialize_state(e)
                    for r, branch in enumerate(branches): s[r, :len(branch['ids'])] = branch['start'][:len(branch['ids'])]
                    g = torch.tensor([[len(b['ids'])-1] for b in branches], device=h.dev)
                    sampled = {}
                    for k in range(K):
                        if time.monotonic() >= deadline: break
                        s = h.step(e, s, k)
                        if k+1 in (K-1, K):
                            v, ix = torch.log_softmax(h.coda_logits(s, g)[:, 0], -1).topk(2)
                            sampled[k+1] = (v.tolist(), ix[:, 0].tolist())
                    if K not in sampled: flags.append(f'C3 incomplete round={j}'); break
                    for r, branch in enumerate(branches):
                        rec = branch['rec']; v, ix = sampled[rec['budget']]
                        rec['token_ids'].append(ix[r]); rec['top_lp'].append(v[r]); branch['ids'].append(ix[r])
                        rec['text'] = h.tok.decode(rec['token_ids']); rec['complete'] = j+1 == cfg['n_new']
            except Exception as exc: flags.append(f'C3: {type(exc).__name__}: {exc}')
    except Exception as exc: flags.append(f'setup: {type(exc).__name__}: {exc}')
    parts = cfg.get('parts', ['C1', 'C2', 'C3'])
    if not rows or any(('C1' in parts and len(r['C1']) != len(cfg['budgets'])) or
        ('C2' in parts and (len(r['C2']) != len(cfg['seeds']) or any(len(c['budgets']) != cfg['tail_c2'] for c in r['C2']))) or
        ('C3' in parts and (len(r['C3']) != 2 or not all(c['complete'] for c in r['C3']))) for r in rows): flags.append('incomplete')
    meta = metadata(cfg, began, tc, run_id=run_id, flags=flags, ok=not flags,
                    precision='float32', start_policy=START_POLICY, hits_sha256=hits_sha(hits_json), **precision_meta(h))
    return save_result(dict(hits=rows, meta=meta), run_id)


def confirm64(hits_json, cfg, run_id):
    began = time.monotonic()
    import torch
    configure_precision()
    h = None; tc = began; rows = []; flags = []; patch = {}; activation = {}
    try:
        h = H('float64', init_fp32=True); tc = time.monotonic()
        deadline = deadline_at(began, tc, CONFIRM64_TIMEOUT, cfg['compute_s'])
        hits = json.loads(prepare_hits(h, hits_json))
        with torch.no_grad():
            if hits:
                _, e0, s0, _ = prefix_batch(h, hits[0], [initial(h, hits[0])])
                before = h.step(e0, s0.clone(), 0)
                patch = patch_double(h)
                after = h.step(e0, s0.clone(), 0)
                diff = float((after-before).abs().max()); scale = float(after.abs().max())
                activation = dict(max_abs_difference=diff, max_abs_value=scale)
                if not (0 < diff <= 1e-3*scale): flags.append('double patch activation failed')
            for hit in hits:
                row = dict(hit, C4=[]); rows.append(row)
                try:
                    _, e, s, g = prefix_batch(h, hit, [initial(h, hit)])
                    for k in range(cfg['K']):
                        if time.monotonic() >= deadline: break
                        s = h.step(e, s, k)
                        if k >= cfg['K']-8:
                            logits = coda_logits64(h, s, g)[0, 0]
                            patch['dtypes']['logits'] = str(logits.dtype)
                            row['C4'].append(read_record(h, logits, k+1, hit['tokens']))
                except Exception as exc: flags.append(f'C4: {type(exc).__name__}: {exc}')
    except Exception as exc: flags.append(f'setup/patch: {type(exc).__name__}: {exc}')
    finally:
        # Modal may reuse a warm process for a later C4 call.
        if h is not None and hasattr(h, '_flicker_restore_double'): h._flicker_restore_double()
    if not rows or any(len(r['C4']) != 8 for r in rows): flags.append('incomplete')
    return save_result(dict(hits=rows, meta=metadata(cfg, began, tc, run_id=run_id, flags=flags, ok=not flags,
        precision='float64 patched', start_policy=START_POLICY, hits_sha256=hits_sha(hits_json),
        patch=patch, activation=activation, **precision_meta(h))), run_id)


def main(mode: str = 'scan_smoke', hits: str = '', prompts: str = '',
         parts: str = 'C1,C2,C3', n_new: int = 9):
    root = Path(__file__).resolve().parent.parent/'results'
    if mode in ('screen2', 'scan2'):
        return run_scan2(mode, prompts, root)
    requested_parts = parts.split(',')
    if not requested_parts or any(p not in ('C1', 'C2', 'C3') for p in requested_parts):
        raise ValueError('--parts must be a comma-separated subset of C1,C2,C3')
    assert mode in ('scan_smoke', 'scan', 'confirm_smoke', 'confirm', 'confirm64_smoke', 'confirm64', 'continue')
    smoke = mode.endswith('_smoke')
    K = 64 if smoke else 512
    run_id = f'flicker_{mode}-{datetime.now():%Y%m%d-%H%M%S}'
    if not smoke:
        kind = 'confirm' if mode == 'continue' else mode
        candidates = sorted(root.glob(f'flicker_{kind}_smoke-*/meta.json'))
        sm = json.loads(candidates[-1].read_text()) if candidates else None
        allowed, message = production_allowed(kind, sm, hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        print(message, flush=True)
        if not allowed: raise SystemExit(message)
    if mode.startswith('scan'):
        prompts = json.loads((root/'flicker_prompts.json').read_text())
        cfg = dict(K=K, tail=16 if smoke else 64, compute_s=150 if smoke else 340, reproduce=not smoke, preflight=True)
        result = scan.remote(json.dumps(prompts), cfg, run_id)
    else:
        if smoke:
            # Token ids are deliberately generated inside the container, where the tokenizer exists.
            cfg = dict(K=K, budgets=[61,62,63,64], seeds=[0,1,2,3,4], tail_c2=8, n_new=2, compute_s=150 if mode.startswith('confirm64') else 120)
            prompts = json.loads((root/'flicker_prompts.json').read_text())
            hitlist = [dict(smoke_prompt=prompts[0])]
        else:
            assert hits, '--hits is required'
            hitlist = json.loads(Path(hits).read_text())
            cfg = dict(K=K, budgets=[509,510,511,512], seeds=[0,1,2,3,4], tail_c2=8, n_new=6, compute_s=270 if mode.startswith('confirm64') else 290)
            if mode == 'continue':  # descriptive continuation only, 9 new tokens
                cfg.update(parts=['C3'], n_new=n_new, budgets=[], seeds=[])
        if mode != 'continue': cfg['parts'] = requested_parts
        result = (confirm64 if mode.startswith('confirm64') else confirm).remote(canonical_hits(hitlist), cfg, run_id)
    dst = root/run_id; dst.mkdir(parents=True, exist_ok=True)
    if mode.startswith('scan'): (dst/'scan.npz').write_bytes(result['npz'])
    else: (dst/'confirm.json').write_text(json.dumps(result, indent=2))
    (dst/'meta.json').write_text(json.dumps(result['meta'], indent=2))
    print(dst, json.dumps(result['meta'], indent=2), flush=True)


def run_scan2(mode, prompts, root):
    """Sequential bounded batches; each worker payload is also saved locally."""
    if not prompts: raise ValueError('--prompts is required')
    raw = Path(prompts).read_bytes()
    rows = json.loads(raw)
    K, tail, size, compute = (32, 8, 200, 200) if mode == 'screen2' else (512, 64, 44, 340)
    if len(rows) != (400 if mode == 'screen2' else 88):
        raise ValueError('Unexpected prompt count')
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    candidates = sorted(root.glob('flicker_scan_smoke-*/meta.json'), reverse=True)
    sm = next((m for path in candidates if (m := json.loads(path.read_text())).get('ok') is True
               and m.get('code_sha256') == sha), None)
    allowed, message = production_allowed(mode, sm, sha, K-tail, tail, size/44)
    print(message, flush=True)
    if not allowed: raise SystemExit(message)
    run_id = f'flicker2_{"screen" if mode == "screen2" else "scan"}-{datetime.now():%Y%m%d-%H%M%S}'
    dst = root/run_id
    dst.mkdir(parents=True, exist_ok=False)
    meta = dict(ok=False, chunk_run_ids=[], prompts_sha256=hashlib.sha256(raw).hexdigest(), code_sha256=sha)
    (dst/'meta.json').write_text(json.dumps(meta, indent=2))
    oks = []
    for i, start in enumerate(range(0, len(rows), size)):
        chunk_id = f'{run_id}_{i}'
        cfg = dict(K=K, tail=tail, compute_s=compute, reproduce=False, preflight=False)
        try:
            result = scan.remote(json.dumps(rows[start:start+size]), cfg, chunk_id)
        except Exception as exc:
            result = dict(meta=dict(ok=False, run_id=chunk_id,
                error=f'{type(exc).__name__}: {exc}'))
        if 'npz' in result:
            (dst/f'scan_{i}.npz').write_bytes(result['npz'])
        (dst/f'meta_{i}.json').write_text(json.dumps(result['meta'], indent=2))
        meta['chunk_run_ids'].append(chunk_id)
        oks.append(result['meta'].get('ok') is True)
        meta['ok'] = len(oks) == (len(rows)+size-1)//size and all(oks)
        if not oks[-1]:
            meta.update(failed_chunk=i, failure_reason=result['meta'].get('error') or
                result['meta'].get('flags') or 'chunk meta.ok is not true')
        (dst/'meta.json').write_text(json.dumps(meta, indent=2))
        if not oks[-1]:
            raise SystemExit(f'Failed chunk {i}: {meta["failure_reason"]}; saved to {dst}')
    print(dst, json.dumps(meta, indent=2), flush=True)


def prepare_hits(h, hits_json):
    hits = json.loads(hits_json)
    if hits and 'smoke_prompt' in hits[0]:
        prompts = [hits[0]['smoke_prompt']]; h.prep(prompts); p = prompts[0]
        return json.dumps([dict(uid=p['uid'], p=p['pred_pos'][0], ids=p['ids'], n_full=len(p['ids']),
                               scan_top1={}, tokens=[])])
    return hits_json


# Keep import of the app optional: local numerics/tests never need Modal.
try:
    import modal
except ImportError:
    modal = None
if modal is not None:
    from intervene_app import H, image, hf_cache, results_vol, REVISION, MODEL_ID
    image = image.add_local_python_source('intervene_app', 'analyze_flicker')
    app = modal.App('huginn-flicker')
    scan = app.function(image=image, gpu='A100-40GB', volumes={'/cache': hf_cache, '/results': results_vol}, timeout=SCAN_TIMEOUT)(scan)
    confirm = app.function(image=image, gpu='A100-40GB', volumes={'/cache': hf_cache, '/results': results_vol}, timeout=CONFIRM_TIMEOUT)(confirm)
    confirm64 = app.function(image=image, gpu='A100-40GB', volumes={'/cache': hf_cache, '/results': results_vol}, timeout=CONFIRM64_TIMEOUT)(confirm64)
    main = app.local_entrypoint()(main)
