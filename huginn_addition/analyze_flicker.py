"""Offline fixed-design flicker analysis (NumPy and stdlib only).

Loop k is zero based: after step k, budget b=k+1. Tail entry j is
K-tail+1+j (449..512 for K=512, tail=64); parity refers to b.
Lead is top log probability minus second log probability, hence nonnegative.
No result may be reported for a prompt that did not complete all K loops.
"""
import argparse
import json
import hashlib
import re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def build_prompts(root=ROOT):
    root = Path(root)
    census = json.loads((root/'results/relax_selection.json').read_text())
    screen = {p['uid']: p for p in json.loads((root/'results/screen_prompts.json').read_text())}
    rows = json.loads((root/'results/screen-20261105-114648/analysis/summary.json').read_text())['rows']
    out = [dict(p, source='census') for p in census]
    out += [dict(screen[r['uid']], source='screen', ref_answer_argmax_last=r['answer_argmax_last'])
            for r in rows if r['blinking']]
    assert len(census) == 26 and len(out) == 44 and len({p['uid'] for p in out}) == 44
    return out


def make_prompts(root=ROOT):
    out = build_prompts(root)
    (Path(root)/'results/flicker_prompts.json').write_text(json.dumps(out, indent=2)+'\n')
    return out


def settled_period_two(drift, lag2_tail):
    """Screen thresholds: >1e-4 motion, <.01 lag2, <2% relative half change."""
    d, l = np.asarray(drift), np.asarray(lag2_tail)
    if len(d) < 64 or len(l) < 32:
        return np.zeros(d.shape[1], dtype=bool)
    a, b = d[-64:-32].mean(0), d[-32:].mean(0)
    ratio = np.divide(b, a, out=np.full_like(b, np.inf), where=a != 0)
    return ((d[-64:].mean(0) > 1e-4) & (np.median(l[-32:], axis=0) < .01)
            & (np.abs(ratio-1) < .02))


def classify(top_ids, drift, lag2_tail):
    a = np.asarray(top_ids)
    if a.ndim == 3:
        a = a[:, :, 0]
    labels = np.full(a.shape[1], 'NONE', dtype='<U10')
    varying = np.any(a != a[:1], axis=0)
    labels[varying] = 'UNRESOLVED'
    two = np.array([len(np.unique(a[:, p])) == 2 for p in range(a.shape[1])])
    alt = np.all(a[1:] != a[:-1], axis=0)
    labels[two & alt & settled_period_two(drift, lag2_tail)] = 'FLICKER'
    return labels


def parsed_answers(tokens):
    """Return predicting positions of digit tokens after '=' on each line."""
    line, answer, targets = 0, False, []
    for i, token in enumerate(tokens):
        token = str(token)
        if answer and token.strip().isdigit() and i > 0: targets.append((i-1, line))
        for c in token:
            if c == '\n': line += 1; answer = False
            elif c == '=': answer = True
    return [p for p, l in targets if l == line], [p for p, l in targets if l < line]


def roles(prompt_text_or_tokens=None, pred_pos=(), demo_answer_pos=(), tokens=None):
    if isinstance(prompt_text_or_tokens, dict):
        row = prompt_text_or_tokens
        tokens = row.get('tok_str', tokens)
        pred_pos = row.get('pred_pos', pred_pos)
        demo_answer_pos = row.get('demo_answer_pos', demo_answer_pos)
        n = len(row.get('ids', tokens if tokens is not None else []))
    else:
        if tokens is None and isinstance(prompt_text_or_tokens, (list, tuple, np.ndarray)): tokens = prompt_text_or_tokens
        n = len(tokens) if tokens is not None else max([*pred_pos, *demo_answer_pos, -1])+2
    out = ['other']*n
    if tokens is not None: _, demo_answer_pos = parsed_answers(tokens)
    for p in demo_answer_pos:
        if 0 <= p < n-1 and (tokens is None or str(tokens[p+1]).strip().isdigit()): out[p] = 'demo answer'
    for p in pred_pos:
        if 0 <= p < n-1: out[p] = 'final answer'
    return out


def select_hits(positions):
    """Tier A first, decreasing smallest lead, then uid, then p; distinct uids."""
    selected, seen = [], set()
    for p in sorted(positions, key=lambda x: (x['tier'], -x['smallest_lead'], x['uid'], x['p'])):
        if p['uid'] not in seen:
            selected.append(p); seen.add(p['uid'])
            if len(selected) == 4: break
    return selected


def scan_gates(z, meta):
    expected = json.loads((ROOT/'results/flicker_prompts.json').read_text())
    g = dict(validation=False, meta_ok=meta.get('ok') is True, configuration=False,
             manifest=False, complete=False, shapes=False, finite=False,
             reference_preflight=False, reproduction=False, roles_consistent=False)
    try:
        K, tail = int(z['K']), int(z['tail'])
        val = meta.get('validation', [])
        g['validation'] = bool(val) and all(r.get('argmax_agree') == 1 and np.isfinite(r.get('max_abs', np.nan)) and r['max_abs'] <= 1e-3 for r in val)
        g['configuration'] = K == 512 and tail == 64 and meta.get('cfg', {}).get('reproduce') is True
        count = len(z['uid'])
        row_keys = ('uid', 'source', 'ids', 'pred_pos', 'steps', 'drift', 'lag2_tail', 'top_ids', 'top_lp',
                    'ref_preflight_ok', 'repro_ok', 'repro_mismatches', 'tok_str')
        lengths = all(len(z[k]) == len(expected) for k in row_keys)
        g['manifest'] = lengths and all(int(z['uid'][i]) == p['uid'] and str(z['source'][i]) == p['source'] and
            len(z['ids'][i]) > 0 and len(z['pred_pos'][i]) == len(p['digit_info']) > 0 and
            all(0 <= int(q) < len(z['ids'][i])-1 for q in z['pred_pos'][i]) and
            ('ids' not in p or np.array_equal(z['ids'][i], p['ids'])) and
            ('pred_pos' not in p or np.array_equal(z['pred_pos'][i], p['pred_pos'])) for i, p in enumerate(expected))
        g['complete'] = count == 44 and all(int(v) == K for v in z['steps'])
        g['shapes'] = count == 44 and all(np.shape(z['top_ids'][i]) == (64, len(z['ids'][i]), 3) and
            np.shape(z['top_lp'][i]) == (64, len(z['ids'][i]), 3) and np.shape(z['drift'][i]) == (512, len(z['ids'][i])) and
            np.shape(z['lag2_tail'][i]) == (64, len(z['ids'][i])) for i in range(count))
        g['finite'] = count == 44 and all(np.isfinite(np.asarray(a, dtype=float)).all() for key in
            ('ids', 'pred_pos', 'drift', 'lag2_tail', 'top_ids', 'top_lp') for a in z[key])
        g['reference_preflight'] = count == 44 and all(bool(z['ref_preflight_ok'][i]) for i in range(count) if z['source'][i] == 'census') and not meta.get('reference_errors')
        g['reproduction'] = count == 44 and all(v is not None and bool(v) for v in z['repro_ok']) and all(v is not None and int(v) == 0 for v in z['repro_mismatches'])
        g['roles_consistent'] = count == 44 and all(len(z['tok_str'][i]) == len(z['ids'][i]) and
            parsed_answers(z['tok_str'][i])[0] == list(z['pred_pos'][i]) for i in range(count))
    except (KeyError, TypeError, ValueError, IndexError): pass
    g['PASS'] = all(g.values())
    return g


def analyze_scan(z, meta, scan_run=None, *, gates_override=None):
    K, tail = int(z['K']), int(z['tail'])
    gates = scan_gates(z, meta) if gates_override is None else gates_override
    out = dict(gates=gates, K=K, tail=tail, prompts=len(z['uid']),
               completed=int(np.sum(np.asarray(z['steps'], int) == K)), flicker=[], unresolved=[],
               counts_by_tier={'A': 0, 'B': 0}, hits=[])
    if K != 512:
        out['status'] = 'SMOKE: plumbing statistics only; no classification is licensed'
        return out
    if not gates['PASS']:
        out['status'] = 'GATES FAILED: no hits'; return out
    budgets = np.arange(K-tail+1, K+1)
    for i, uid in enumerate(z['uid']):
        ids, top, lp = np.asarray(z['ids'][i]), np.asarray(z['top_ids'][i]), np.asarray(z['top_lp'][i])
        strings = list(z['tok_str'][i])
        rr = roles(dict(ids=ids, tok_str=strings, pred_pos=z['pred_pos'][i]))
        labels = classify(top, z['drift'][i], z['lag2_tail'][i])
        for p, label in enumerate(labels):
            if label == 'NONE': continue
            rec = dict(uid=int(uid), source=str(z['source'][i]), p=p, role=rr[p])
            if label == 'UNRESOLVED': out['unresolved'].append(rec); continue
            lead = lp[:, p, 0]-lp[:, p, 1]
            phases = {}
            for phase, parity in [('even', 0), ('odd', 1)]:
                m = budgets % 2 == parity
                phases[phase] = dict(token=int(top[m, p, 0][0]),
                    median_lead=float(np.median(lead[m])), median_top_probability=float(np.median(np.exp(lp[m, p, 0]))))
            toks = [phases[x]['token'] for x in ('even', 'odd')]
            # Top-three token strings are stored separately from prompt strings.
            token_strings = z['top_tok_str'][i] if 'top_tok_str' in z else {}
            truth = int(ids[p+1]) if p+1 < len(ids) else None
            rec.update(tier='B' if rr[p] == 'other' else 'A', tokens=toks,
                token_strings=[token_strings.get(str(t), '<unavailable>') for t in toks],
                phases=phases, smallest_lead=float(lead.min()), true_next_token=truth,
                true_next_token_string=str(strings[p+1]) if truth is not None else None,
                phase_matches_true={ph: phases[ph]['token'] == truth for ph in phases},
                ids=ids.tolist(), n_full=len(ids),
                scan_top1={str(b): int(top[b-(K-tail+1), p, 0]) for b in range(509, 513)},
                scan_run=scan_run or meta.get('run_id'), model_id=meta.get('model_id'), revision=meta.get('revision'),
                ids_sha256=hashlib.sha256(json.dumps(ids.tolist(), separators=(',', ':')).encode()).hexdigest(),
                scan_pair_lp={str(b): [float(lp[b-(K-tail+1), p, list(top[b-(K-tail+1), p]).index(t)])
                    if t in top[b-(K-tail+1), p] else None for t in toks] for b in range(505, 513)})
            out['flicker'].append(rec); out['counts_by_tier'][rec['tier']] += 1
    out['hits'] = [{k: p[k] for k in ('uid', 'p', 'ids', 'n_full', 'scan_top1', 'tokens', 'scan_run', 'ids_sha256', 'smallest_lead', 'tier', 'role', 'scan_pair_lp', 'model_id', 'revision')}
                   for p in select_hits(out['flicker'])]
    out['outcome'] = 'CANDIDATES, NONE CONFIRMED OR CONFIRMATION INCOMPLETE' if out['flicker'] else 'NO FLICKER POSITION'
    out['status'] = 'PASS'
    return out


def strict_two(a):
    a = list(a)
    return len(a) >= 2 and len(set(a)) == 2 and all(x != y for x, y in zip(a, a[1:]))


def canonical_hits(value):
    # Single wire representation, imported by the app as well.
    if isinstance(value, str): value = json.loads(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def hits_sha(value):
    return hashlib.sha256(canonical_hits(value).encode()).hexdigest()


def confirmation_gates(hits, data, c64=None):
    meta = data.get('meta', {}); cfg = meta.get('cfg', {})
    def bound(rows):
        return len(rows) == len(hits) and all(all(row.get(k) == hit.get(k) for k in ('uid', 'p', 'ids', 'n_full')) for row, hit in zip(rows, hits))
    def model(m):
        return bool(hits) and all(h.get('model_id') is not None and h.get('revision') is not None and
            h['model_id'] == m.get('model_id') and h['revision'] == m.get('revision') for h in hits)
    def start(m): return m.get('start_policy') == 'census draw (uid, seed 0), full-length, sliced'
    g = dict(confirm_rows=bound(data.get('hits', [])), confirm_hits_sha=meta.get('hits_sha256') == hits_sha(hits),
             confirm_model=model(meta), confirm_start=start(meta), confirm_precision=meta.get('precision') == 'float32',
             confirm_config=cfg.get('K') == 512 and cfg.get('budgets') == [509,510,511,512] and
             cfg.get('tail_c2') == 8 and cfg.get('seeds', [])[:1] == [0],
             hit_provenance=bool(hits) and all(h.get('scan_run') and h.get('ids_sha256') ==
                hashlib.sha256(json.dumps(h['ids'], separators=(',', ':')).encode()).hexdigest() for h in hits))
    if c64 is not None:
        m = c64.get('meta', {}); patch = m.get('patch', {}); act = m.get('activation', {})
        g.update(c64_rows=bound(c64.get('hits', [])), c64_hits_sha=m.get('hits_sha256') == hits_sha(hits),
            c64_model=model(m), c64_start=start(m), c64_precision=m.get('precision') == 'float64 patched',
            c64_config=m.get('cfg', {}).get('K') == 512,
            c64_patch=bool(patch.get('norm_modules', 0) > 0 and patch.get('norm_source_sha256') and
                patch.get('rotary_source_sha256') and patch.get('dtypes') and
                0 < act.get('max_abs_difference', 0) <= 1e-3*act.get('max_abs_value', 0)),
            c64_budgets=all([r.get('budget') for r in row.get('C4', [])] == list(range(505,513)) for row in c64.get('hits', [])))
    g['PASS'] = all(g.values())
    return g


def valid_readouts(records, budgets, pair):
    try:
        seq = [r['top_ids'][0] for r in records]
        return ([r['budget'] for r in records] == budgets and strict_two(seq) and set(seq) == set(pair) and
            all(len(r['top_ids']) == len(r['top_lp']) == 3 and len(set(r['top_ids'])) == 3 and
                np.isfinite(r['top_lp']).all() and r['top_lp'][0] > r['top_lp'][1] >= r['top_lp'][2] for r in records))
    except (KeyError, TypeError, ValueError, IndexError): return False


def pair_delta(hit, row):
    try:
        c1 = {r['budget']: r['pair_lp'] for r in row['C1']}
        zero_rows = [r for r in row['C2'] if r['seed'] == 0]
        if len(zero_rows) != 1: return None
        seed0 = zero_rows[0]
        top_lp = np.asarray(seed0['top_lp'])
        if (seed0['budgets'] != list(range(505,513)) or len(seed0['top_ids']) != 8 or
            top_lp.shape != (8,2) or not np.isfinite(top_lp).all() or
            not (top_lp[:,0] >= top_lp[:,1]).all()): return None
        c2 = dict(zip(seed0['budgets'], seed0['pair_lp']))
        differences = []
        for b in range(509, 513):
            values = [hit['scan_pair_lp'][str(b)], c1[b], c2[b]]
            if any(len(v) != 2 or any(x is None for x in v) or not np.isfinite(v).all() for v in values): return None
            gaps = [v[0]-v[1] for v in values]
            differences.append(max(gaps)-min(gaps))
        return float(max(differences))
    except (KeyError, TypeError, ValueError, IndexError, StopIteration): return None


def analyze_confirmation(data, hits, c64=None):
    gates = confirmation_gates(hits, data, c64)
    base_ok = all(v for k, v in gates.items() if k != 'PASS' and not k.startswith('c64_'))
    double_ok = c64 is not None and all(v for k, v in gates.items() if k.startswith('c64_'))
    rows = []
    for i, hit in enumerate(hits):
        row = data.get('hits', [])[i] if i < len(data.get('hits', [])) else {}
        c1 = row.get('C1', [])
        ok1 = base_ok and valid_readouts(c1, [509,510,511,512], hit['tokens']) and all(
            r['top_ids'][0] == hit['scan_top1'].get(str(r['budget'])) for r in c1)
        rec64 = c64.get('hits', [])[i].get('C4', []) if c64 and i < len(c64.get('hits', [])) else []
        ok4 = double_ok and valid_readouts(rec64, list(range(505,513)), hit['tokens'])
        delta = pair_delta(hit, row)
        guard = delta is not None and np.isfinite(hit['smallest_lead']) and hit['smallest_lead'] > max(1e-4, 10*delta)
        label = ('CONFIRMED' if guard else 'NUMERICALLY UNRESOLVED') if ok1 and ok4 else (
            'CONFIRMATION INCOMPLETE' if ok1 and not rec64 else 'NOT CONFIRMED')
        seeds = []
        for r in row.get('C2', []):
            if r.get('seed') == 0: continue
            seq, bs = r.get('top_ids', []), r.get('budgets', [])
            seeds.append(dict(seed=r['seed'], alternates=bs == list(range(505,513)) and len(seq) == 8 and strict_two(seq),
                              token_at_512=seq[bs.index(512)] if 512 in bs and len(seq) == len(bs) else None,
                              same_pair=set(seq) == set(hit['tokens'])))
        rows.append(dict(uid=hit['uid'], p=hit['p'], C1=bool(ok1), C4=bool(ok4), delta=delta, guard=bool(guard),
                         C2=seeds, C3=row.get('C3', []), label=label, confirmed=label == 'CONFIRMED'))
    return dict(gates=gates, hits=rows, outcome='AT LEAST ONE CONFIRMED' if any(r['confirmed'] for r in rows)
                else 'CANDIDATES, NONE CONFIRMED OR CONFIRMATION INCOMPLETE',
                evidence_windows='float32 scan: 449..512; prefix forward: 509..512; float64: 505..512 (pair agrees, phase free)')


def stage_provenance(root, manifest, z, chunks, metas, parent, chunk_size, prompts_path):
    """Independent named gates binding bytes, metadata, row order and decoded text."""
    g = dict(prompts_sha256=False, chunk_consistency=False, chunk_uids=False, text_spelling=False)
    try:
        g['prompts_sha256'] = (prompts_path is not None and
            json.loads(Path(prompts_path).read_bytes()) == manifest and
            hashlib.sha256(Path(prompts_path).read_bytes()).hexdigest() == parent.get('prompts_sha256'))
        g['chunk_consistency'] = bool(metas) and all(m.get('ok') is True and
            all(key in m and m[key] == metas[0].get(key) for key in ('model_id','revision','code_sha256','cfg'))
            for m in metas)
        g['chunk_uids'] = len(chunks) == (len(manifest)+chunk_size-1)//chunk_size and all(
            list(map(int,c['uid'])) == [p['uid'] for p in manifest[i*chunk_size:(i+1)*chunk_size]]
            for i,c in enumerate(chunks))
        g['text_spelling'] = len(z['tok_str']) == len(manifest) and all(
            re.sub(r'^<\|[^>]*\|>', '', ''.join(map(str,strings)), count=1) == p['prompt']+p['answer']
            for strings,p in zip(z['tok_str'],manifest))
    except (KeyError, TypeError, ValueError, OSError): pass
    return g


def load_chunks(root, manifest, K, tail, chunk_size, prompts_path=None):
    """Read only the expected numbered chunks; return closed, independent arrays."""
    root = Path(root)
    count = (len(manifest)+chunk_size-1)//chunk_size
    chunks, metas, errors = [], [], []
    try:
        parent = json.loads((root/'meta.json').read_text())
    except (OSError, ValueError):
        parent = {}; errors.append('missing/invalid meta.json')
    expected = {f'scan_{i}.npz' for i in range(count)}
    if {p.name for p in root.glob('scan_*.npz')} != expected: errors.append('chunk file set')
    for i in range(count):
        try:
            with np.load(root/f'scan_{i}.npz', allow_pickle=True) as f:
                chunks.append({k: f[k].copy() for k in f.files})
            metas.append(json.loads((root/f'meta_{i}.json').read_text()))
        except (OSError, ValueError, KeyError): errors.append(f'missing/invalid chunk {i}')
    z = {}
    if chunks:
        try:
            z = {k: [v for c in chunks for v in c[k]] for k in chunks[0] if k not in ('K', 'tail')}
            z.update(K=K, tail=tail)
        except (KeyError, TypeError): errors.append('inconsistent chunk keys')
    configuration = False
    try:
        configuration = (len(chunks) == count and all(int(c['K']) == K and int(c['tail']) == tail for c in chunks)
            and len(metas) == count and all(m.get('cfg', {}).get('K') == K and
                m.get('cfg', {}).get('tail') == tail for m in metas))
    except (KeyError, TypeError, ValueError): pass
    g = dict(chunks=not errors and len(chunks) == count,
             meta_ok=parent.get('ok') is True and len(metas) == count and all(m.get('ok') is True for m in metas),
             configuration=configuration,
             validation=len(metas) == count and all(bool(m.get('validation')) and all(
                 v.get('argmax_agree') == 1 and np.isfinite(v.get('max_abs', np.nan)) and v['max_abs'] <= 1e-3
                 for v in m['validation']) for m in metas),
             manifest=False, complete=False, shapes=False, finite=False, roles_consistent=False,
             chunk_binding=False)
    try:
        g['chunk_binding'] = (parent.get('chunk_run_ids') == [m.get('run_id') for m in metas]
            and all(m.get('code_sha256') == parent.get('code_sha256') for m in metas)
            and all(len(c['uid']) == min(chunk_size, len(manifest)-i*chunk_size) for i,c in enumerate(chunks)))
        n = len(manifest)
        keys = ('uid','source','ids','pred_pos','steps','top_ids','top_lp','drift','lag2_tail','tok_str','top_tok_str')
        g['manifest'] = all(len(z[k]) == n for k in keys) and all(
            int(z['uid'][i]) == p['uid'] and str(z['source'][i]) == p['source'] and
            len(z['pred_pos'][i]) == len(p['digit_info']) > 0 and
            all(0 <= int(q) < len(z['ids'][i])-1 for q in z['pred_pos'][i]) and
            ('ids' not in p or np.array_equal(z['ids'][i], p['ids'])) and
            ('pred_pos' not in p or np.array_equal(z['pred_pos'][i], p['pred_pos']))
            for i,p in enumerate(manifest))
        g['complete'] = len(z['steps']) == n and all(int(v) == K for v in z['steps'])
        g['shapes'] = len(z['ids']) == n and all(
            np.shape(z['top_ids'][i]) == np.shape(z['top_lp'][i]) == (tail,len(ids),3) and
            np.shape(z['drift'][i]) == (K,len(ids)) and np.shape(z['lag2_tail'][i]) == (tail,len(ids))
            for i,ids in enumerate(z['ids']))
        g['finite'] = all(np.isfinite(np.asarray(v,dtype=float)).all() for k in
            ('ids','pred_pos','top_ids','top_lp','drift','lag2_tail') for v in z[k])
        g['roles_consistent'] = all(len(z['tok_str'][i]) == len(ids) and
            parsed_answers(z['tok_str'][i])[0] == list(z['pred_pos'][i]) for i,ids in enumerate(z['ids']))
    except (KeyError, ValueError, TypeError, IndexError): pass
    g.update(stage_provenance(root, manifest, z, chunks, metas, parent, chunk_size, prompts_path))
    g['PASS'] = all(g.values())
    return z, metas, g


def analyze_screen2(root, prompts, controls=None, prompts_path=None):
    from make_flicker2_prompts import build
    z, _, gates = load_chunks(root, prompts, 32, 8, 200, prompts_path)
    gates['builder_manifest'] = prompts == build()
    gates['fixed_manifest'] = len(prompts) == 400 and [p['uid'] for p in prompts] == list(range(30000,30400))
    gates['PASS'] = all(v for k,v in gates.items() if k != 'PASS')
    out = dict(gates=gates, selected=[], distribution={}, per_context={})
    if not gates['PASS']:
        out['outcome'] = 'SEARCH INVALID OR INCOMPLETE'
        out['failed_gates'] = [k for k,v in gates.items() if k != 'PASS' and not v]
        return out, []
    stats = []
    for i,p in enumerate(prompts):
        lp = np.asarray(z['top_lp'][i])
        value = float((lp[:,:,0]-lp[:,:,1]).mean(0)[list(z['pred_pos'][i])].min())
        stats.append((value,p['uid'],p))
    stats.sort(key=lambda r:(r[0],r[1]))
    values = np.array([s[0] for s in stats])
    out['distribution'] = dict(min=float(values.min()), quantiles={str(q):float(np.quantile(values,q)) for q in (0,.1,.25,.5,.75,.9,1)},
                               **{f'below_{x}':int((values<x).sum()) for x in (.05,.1,.3)})
    selected = [dict(p,screen_stat=value,screen_rank=rank) for rank,(value,uid,p) in enumerate(stats[:85],1)]
    out['selected'] = [p['uid'] for p in selected]
    out['per_context'] = {str(c):sum(p['context_uid']==c for p in selected) for c in (20059,20145,20144,20178)}
    if controls is None: controls = json.loads((ROOT/'results/flicker_prompts.json').read_text())
    lookup = {p['uid']:p for p in controls}
    return out, [lookup[u] for u in (20178,20147,20146)] + selected


def selection_gates(manifest, screen):
    gates = dict(screen_selection=False)
    controls = {p['uid']:p for p in json.loads((ROOT/'results/flicker_prompts.json').read_text())}
    gates['control_manifest'] = manifest[:3] == [controls[u] for u in (20178,20147,20146)]
    try:
        from make_flicker2_prompts import build
        screen_prompts = build()
        # The deterministic rows have fixed bytes in the checked-in manifest.
        screen_path = ROOT/'results/flicker2_screen_prompts.json'
        ss, selected = analyze_screen2(screen, screen_prompts, prompts_path=screen_path)
        gates['screen_selection'] = ss['gates']['PASS'] and manifest[3:] == selected[3:]
    except (OSError, TypeError, ValueError, KeyError): pass
    return gates


def analyze_scan2(root, manifest, control_summary=None, screen=None, prompts_path=None):
    z, metas, gates = load_chunks(root, manifest, 512, 64, 44, prompts_path)
    gates.update(selection_gates(manifest, screen))
    gates['fixed_manifest'] = (len(manifest)==88 and [p['uid'] for p in manifest[:3]]==[20178,20147,20146]
        and len({p['uid'] for p in manifest})==88 and all(p.get('source')=='flicker2' for p in manifest[3:]))
    gates['positive_controls'] = False
    try:
        if control_summary is None:
            control_summary = json.loads((ROOT/'results/flicker_scan-20261111-022450/flicker_summary.json').read_text())
        reference = {(r['uid'],r['p']):r for r in control_summary['flicker']}
        gates['positive_controls'] = all(classify(z['top_ids'][i],z['drift'][i],z['lag2_tail'][i])[p]=='FLICKER'
            and set(np.asarray(z['top_ids'][i])[:,p,0])==set(reference[(uid,p)]['tokens'])
            for i,(uid,p) in enumerate(((20178,40),(20147,41),(20146,44))))
    except (KeyError, ValueError, TypeError, IndexError): pass
    gates['PASS'] = all(v for k,v in gates.items() if k != 'PASS')
    out = dict(gates=gates, answer_hits=[], demo_answer_hits=[], other_flicker_count=0, carry_over={}, near_misses=[], hits=[])
    if not gates['PASS']:
        out['outcome'] = 'SEARCH INVALID OR INCOMPLETE'
        out['failed_gates'] = [k for k,v in gates.items() if k != 'PASS' and not v]
        return out
    # The numerical classifier and original hit wire representation are shared.
    meta = dict(metas[0], run_id=Path(root).name)
    base = analyze_scan(z,meta,Path(root).name,gates_override=gates)
    index = {int(u):i for i,u in enumerate(z['uid'])}
    demos = {}
    for r in base['flicker']:
        i = index[r['uid']]
        if i < 3: continue
        context = manifest[i]['context_uid']
        if r['role']=='final answer':
            right = [ph for ph in ('even','odd') if r['phase_matches_true'][ph]]
            earlier = [int(p) for p in z['pred_pos'][i] if p < r['p']]
            digit_flags = [is_digit_token(t) for t in r['token_strings']]
            label = answer_sublabel(bool(right), digit_flags)
            r.update(sub_label=label, sublabel=label, token_is_digit=digit_flags,
                first_answer_pos=int(z['pred_pos'][i][0]), chunk=i//44,
                a=manifest[i]['a'], b=manifest[i]['b'], c=manifest[i]['c'],
                right_parity=right[0] if right else None,
                earlier_digits_right=all(np.all(np.asarray(z['top_ids'][i])[:,p,0] == z['ids'][i][p+1]) for p in earlier))
            out['answer_hits'].append(r)
        elif r['role']=='demo answer':
            key=(context,r['p'])
            if key not in demos: demos[key]=dict(context_uid=context,p=r['p'],prompts=[],pairs={})
            demos[key]['prompts'].append(r['uid'])
            pair = tuple(sorted(r['tokens']))
            demos[key]['pairs'][pair] = demos[key]['pairs'].get(pair, 0) + 1
        else: out['other_flicker_count'] += 1
    out['demo_answer_hits'] = [dict(r, pairs=[dict(tokens=list(pair), count=count)
        for pair,count in sorted(r['pairs'].items())]) for r in demos.values()]
    carry = {c:dict(prompts=0,settled_prompts=0,swings=[]) for c in (20059,20145,20144,20178)}
    near = []
    for i,prompt in enumerate(manifest[3:],3):
        context = prompt['context_uid']
        c = carry.setdefault(context,dict(prompts=0,settled_prompts=0,swings=[]))
        c['prompts'] += 1
        orbit = settled_period_two(z['drift'][i],z['lag2_tail'][i])
        pp = list(z['pred_pos'][i]); c['settled_prompts'] += int(all(orbit[p] for p in pp))
        hit_positions = {r['p'] for r in out['answer_hits'] if r['uid'] == prompt['uid']}
        for p in pp:
            if not orbit[p]: continue
            pair, even, odd, swing = pair_statistics(z['top_ids'][i], z['top_lp'][i], p, 449)
            if np.isfinite(swing): c['swings'].append(swing)
            if p not in hit_positions and np.isfinite(swing) and swing > 0:
                near.append(dict(uid=prompt['uid'],p=int(p),tokens=pair,
                    mean_even_gap=even,mean_odd_gap=odd,swing=swing,
                    ratio=min(abs(even),abs(odd))/swing))
    for context,c in carry.items():
        out['carry_over'][str(context)] = dict(prompts=c['prompts'],settled_prompts=c['settled_prompts'],
            share=c['settled_prompts']/c['prompts'] if c['prompts'] else None,
            swing=dict(min=min(c['swings']),median=float(np.median(c['swings'])),max=max(c['swings']))
            if c['swings'] else dict(min=None,median=None,max=None))
    out['near_misses'] = sorted(near,key=lambda r:(r['ratio'] if r['ratio'] is not None else float('inf'),r['uid'],r['p']))[:10]
    selected=[];seen=set()
    for r in sorted(out['answer_hits'],key=lambda r:(('RIGHT/WRONG DIGIT','RIGHT/OTHER','WRONG/WRONG').index(r['sublabel']),-r['smallest_lead'],r['uid'],r['p'])):
        if r['uid'] not in seen:
            selected.append(r);seen.add(r['uid'])
            if len(selected)==4: break
    fields=('uid','p','ids','n_full','scan_top1','tokens','scan_run','ids_sha256','smallest_lead','tier','role','scan_pair_lp','model_id','revision','first_answer_pos','chunk','a','b','c','sublabel','token_is_digit')
    out['hits']=[{k:r[k] for k in fields} for r in selected]
    out['outcome'] = ('ANSWER HITS FOUND (RIGHT/WRONG DIGIT)' if any(r['sublabel']=='RIGHT/WRONG DIGIT' for r in out['answer_hits']) else
        'ANSWER HITS FOUND (RIGHT/OTHER, NO RIGHT/WRONG DIGIT)' if any(r['sublabel']=='RIGHT/OTHER' for r in out['answer_hits']) else
        'ANSWER HITS FOUND (WRONG/WRONG ONLY)' if out['answer_hits'] else 'NO ANSWER-POSITION FLICKER')
    return out


def is_digit_token(text):
    return len(text.strip(' ')) == 1 and text.strip(' ') in '0123456789'


def answer_sublabel(right, digit_flags):
    return ('RIGHT/WRONG DIGIT' if all(digit_flags) else 'RIGHT/OTHER') if right else 'WRONG/WRONG'


def pair_statistics(top_ids, top_lp, position, first_budget):
    ids, lp = np.asarray(top_ids)[:,position], np.asarray(top_lp)[:,position]
    pair = list(map(int,ids[-1,:2]))
    gaps = []
    for tokens,values in zip(ids,lp):
        gaps.append(float(values[list(tokens).index(pair[0])]-values[list(tokens).index(pair[1])])
                    if all(t in tokens for t in pair) else float('nan'))
    gaps = np.asarray(gaps)
    parity = np.arange(first_budget,first_budget+len(gaps)) % 2
    means = [float(gaps[(parity==p)&np.isfinite(gaps)].mean())
             if np.any((parity==p)&np.isfinite(gaps)) else float('nan') for p in (0,1)]
    return pair, *means, abs(means[0]-means[1])


def continuation_hits(hits, summary):
    return [dict(h, p=h['first_answer_pos'], tokens=[], scan_top1={}, scan_pair_lp={})
            for h,r in zip(hits, summary['hits'])
            if 'first_answer_pos' in h and r['label']=='CONFIRMED'][:2]


def write_continuation_hits(dest, hits, summary):
    # Old experiment has no first_answer_pos and retains its original artifacts.
    if any('first_answer_pos' in h for h in hits):
        (Path(dest)/'flicker_hits_continue.json').write_text(
            json.dumps(continuation_hits(hits, summary), indent=2, allow_nan=False)+'\n')


def analyze_continue2(data, hits):
    meta = data.get('meta', {}); cfg = meta.get('cfg', {})
    rows = data.get('hits', [])
    gates = dict(meta_ok=meta.get('ok') is True,
        hits_sha=meta.get('hits_sha256') == hits_sha(hits),
        configuration=cfg.get('K')==512 and cfg.get('n_new')==7 and cfg.get('parts')==['C3'],
        rows=len(rows)==len(hits) and all(all(r.get(k)==h.get(k) for k in h) for r,h in zip(rows,hits)),
        branches=all(sorted(r.get('budget',0) for r in row.get('C3',[]))==[511,512] and
            all(isinstance(r.get('text'),str) and isinstance(r.get('token_ids'),list) and len(r['token_ids'])==7
                for r in row.get('C3',[])) for row in rows))
    gates['PASS'] = all(gates.values())
    out = dict(gates=gates, hits=[])
    if not gates['PASS']: return out
    for h,row in zip(hits,rows):
        branches=[]
        for r in sorted(row['C3'],key=lambda r:r['budget']):
            text=r['text']; line=text.split('\n',1)[0]; stripped=line.strip(' ')
            status=('TRUNCATED' if '\n' not in text else 'MALFORMED' if not stripped or
                    any(c not in '0123456789' for c in stripped) else
                    'CORRECT' if int(stripped)==h['a']+h['b'] else 'INCORRECT')
            branches.append(dict(budget=r['budget'],answer_line=line,status=status,
                                 text=text,token_ids=r['token_ids']))
        out['hits'].append(dict(uid=h['uid'],a=h['a'],b=h['b'],c=h['c'],branches=branches,
            right_at_one_parity_wrong_at_other={r['status'] for r in branches}=={'CORRECT','INCORRECT'}))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', nargs='?'); ap.add_argument('--make-prompts', action='store_true'); ap.add_argument('--c64'); ap.add_argument('--hits')
    ap.add_argument('--screen2'); ap.add_argument('--scan2'); ap.add_argument('--prompts')
    ap.add_argument('--screen'); ap.add_argument('--continue2')
    ap.add_argument('--manifest'); ap.add_argument('--out'); ap.add_argument('--output-dir')
    args = ap.parse_args()
    if args.continue2:
        if not args.hits: ap.error('--continue2 requires --hits')
        root = Path(args.continue2)
        out = analyze_continue2(json.loads((root/'confirm.json').read_text()), json.loads(Path(args.hits).read_text()))
        (root/'continue_summary.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
        print(json.dumps(out, indent=2)); return
    if args.screen2:
        if not args.prompts or not args.out: ap.error('--screen2 requires --prompts and --out')
        summary, selected = analyze_screen2(args.screen2, json.loads(Path(args.prompts).read_text()), prompts_path=args.prompts)
        Path(args.out).write_text(json.dumps(selected, indent=2, allow_nan=False)+'\n')
        (Path(args.screen2)/'screen_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        print(json.dumps(summary, indent=2)); return
    if args.scan2:
        if not args.manifest or not args.screen: ap.error('--scan2 requires --manifest and --screen')
        summary = analyze_scan2(args.scan2, json.loads(Path(args.manifest).read_text()), screen=args.screen, prompts_path=args.manifest)
        for name, value in [('flicker2_summary',summary),('flicker_hits',summary['hits'])]:
            (Path(args.scan2)/(name+'.json')).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        print(json.dumps(summary, indent=2)); return
    if args.make_prompts:
        print(f'Wrote {len(make_prompts())} prompts to results/flicker_prompts.json')
    if args.run:
        root = Path(args.run)
        dest = Path(args.output_dir) if args.output_dir else root
        dest.mkdir(parents=True, exist_ok=True)
        if (root/'scan.npz').exists():
            with np.load(root/'scan.npz', allow_pickle=True) as z:
                out = analyze_scan(z, json.loads((root/'meta.json').read_text()), root.name)
            (dest/'flicker_hits.json').write_text(json.dumps(out['hits'], indent=2)+'\n')
            name = 'flicker_summary.json'
        else:
            c64 = json.loads((Path(args.c64)/'confirm.json').read_text()) if args.c64 else None
            if not args.hits: ap.error('--hits <scan_dir>/flicker_hits.json is required')
            hits = json.loads(Path(args.hits).read_text())
            out = analyze_confirmation(json.loads((root/'confirm.json').read_text()), hits, c64)
            if args.c64: write_continuation_hits(dest, hits, out)
            name = 'confirm_summary.json'
        (dest/name).write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
        print(json.dumps(out, indent=2, allow_nan=False))

if __name__ == '__main__': main()
