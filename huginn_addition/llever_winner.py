"""Prospective answer-margin screen and conditional steering follow-up."""
import hashlib
import io
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import llever_app as old

SEED = 20261117
SHA = "f021c80a02d3b0d5b8f21bb9fdb7a42fe65eb7d59ea71109ae5526d44cb8ea3f"
DOSES = (0.01, 0.03, 0.09)
ROOT = Path(__file__).resolve().parents[1] / 'results'
RUN = ROOT / 'llever_winner-20261117'


def prompts():
    rng = random.Random(SEED)
    items = []
    for family in ('add', 'add3', 'sub', 'muladd'):
        seen = set()
        demos = {'add': '18 + 36 = 54\n21 + 41 = 62\n',
                 'add3': '12 + 31 + 14 = 57\n24 + 16 + 21 = 61\n',
                 'sub': '131 - 76 = 55\n108 - 46 = 62\n',
                 'muladd': '7 * 6 + 15 = 57\n8 * 5 + 23 = 63\n'}[family]
        while len(seen) < 192:
            answer = rng.randrange(70, 80)
            a, b = rng.randrange(10, 42), rng.randrange(10, 26)
            if family == 'add':
                expr = f'{a} + {answer-a}'
            elif family == 'add3':
                expr = f'{a} + {b} + {answer-a-b}'
            elif family == 'sub':
                b = rng.randrange(30, 330)
                expr = f'{answer+b} - {b}'
            else:
                a, b = rng.randrange(3, 10), rng.randrange(3, 8)
                expr = f'{a} * {b} + {answer-a*b}'
            if expr in seen:
                continue
            seen.add(expr)
            items.append(dict(pid=17110000+len(items), family=family, answer=answer,
                              prompt=demos + expr + ' = '))
    rng.shuffle(items)
    return items


def select(rows):
    selected = []
    for group in ('near', 'far'):
        selected.extend([r for r in rows if r['group'] == group][:24])
    return selected


def setup():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    opts = dict(revision=old.MODEL[1], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(old.MODEL[0], trust_remote_code=True,
             torch_dtype=torch.float64, **opts).to('cuda').double().eval().requires_grad_(False)
    old.patch_precision(model)
    return model, AutoTokenizer.from_pretrained(old.MODEL[0], **opts)


def run(stage, items, push_file=b''):
    import numpy as np
    import torch
    began = time.monotonic()
    deadline = began + (530 if stage == 'screen' else 1100)
    model, tok = setup()
    prep, step, read = old.harness(model, 'cuda')
    tokens = [old.one_token(tok, str(k)) for k in range(70, 80)]
    if stage != 'screen':
        assert hashlib.sha256(push_file).hexdigest() == SHA
        with np.load(io.BytesIO(push_file), allow_pickle=False) as data:
            assert list(data['names'][:10]) == [f'L{k}' for k in range(10)]
            vectors = torch.as_tensor(data['P'][:10], device='cuda').double()
            norm = float(data['m_ref'])
    buckets = defaultdict(list)
    for item in items:
        ids = [old.BOS] + tok.encode(item['prompt'], add_special_tokens=False)
        buckets[len(ids)].append((item, ids))
    rows = {}
    gates = []
    batch = 16 if stage == 'screen' else 8
    with torch.no_grad():
        for length, entries in buckets.items():
            for j in range(0, len(entries), batch):
                subset = entries[j:j+batch]
                batch_items = [p for p, _ in subset]
                ids = torch.tensor([s for _, s in subset], device='cuda')
                x0 = old.start(model, batch_items, length, 11000000, 'cuda').double()
                e, fc = prep(ids)
                def check(t, logits):
                    ref = model(ids[:1], input_states=x0[:1].clone(), num_steps=t).logits[:, -1]
                    error = old.centered_error(logits[:1], ref)
                    assert error < 1e-6, error
                    gates.append(error)
                x, base = old.base_state(step, read, x0, e, fc, tokens, deadline, check if j == 0 else None)
                logits = read(x, fc)[1]
                top = logits.topk(2)
                batch_rows = []
                for i, item in enumerate(batch_items):
                    w, r = top.indices[i].tolist()
                    gap = float(top.values[i, 0] - top.values[i, 1])
                    drift = float(base['base_drift'][i])
                    stable = drift < 1e-4 and w == int(base['answer32'][i])
                    eligible = stable and r in tokens and tok.decode([w]).strip().isdigit()
                    group = 'near' if eligible and gap <= .5 else 'far' if eligible and gap >= 2 else None
                    record = dict(item, winner=w, runner=r, gap=gap, drift=drift, group=group)
                    if stage != 'screen':
                        assert (w, r) == (item['winner'], item['runner']) and abs(gap-item['gap']) < 1e-6
                        record['doses'] = {}
                    batch_rows.append(record)
                    rows[item['pid']] = record
                if stage == 'screen':
                    continue
                P = torch.stack([vectors[tokens.index(p['runner'])] for p in batch_items])
                if stage == 'predict':
                    keep = {1: None}
                    total, _ = old.chain(step, x, e, fc, P[:, None], 48, 'jvp', batch, deadline, keep)
                    for label, response in [('one', keep[1]), ('full', total)]:
                        v = response[:, 0]
                        # Central directional derivative through the coda for all vocabulary tokens.
                        eps = 1e-6*x[:, -1].norm(dim=-1)/v.norm(dim=-1).clamp_min(1e-300)
                        plus, minus = x.clone(), x.clone()
                        plus[:, -1] += eps[:, None]*v
                        minus[:, -1] -= eps[:, None]*v
                        derivative = (read(plus, fc)[1]-read(minus, fc)[1])/(2*eps[:, None])
                        for dose in DOSES:
                            change = derivative*(dose*norm)
                            winners = (logits+change).argmax(-1).tolist()
                            for i, row in enumerate(batch_rows):
                                result = row['doses'].setdefault(str(dose), {})
                                result[label] = dict(winner=winners[i], shift=float(
                                    change[i, row['runner']]-change[i, row['winner']]))
                elif stage == 'evaluate':
                    for dose in (0., *DOSES):
                        state = x.clone()
                        for _ in range(48):
                            old.check_deadline(deadline)
                            inp = state.clone()
                            inp[:, -1] += P*(dose*norm)
                            state = step(inp, e, fc)
                        actual = read(state, fc)[1]
                        for i, row in enumerate(batch_rows):
                            row['doses'][str(dose)] = dict(winner=int(actual[i].argmax()),
                                margin=float(actual[i,row['runner']]-actual[i,row['winner']]))
                else:
                    raise ValueError(stage)
    return dict(rows=[rows[p['pid']] for p in items], runtime_s=time.monotonic()-began, gates=gates,
                stage=stage, model=old.MODEL, sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def summarize(pred, measured):
    from statistics import mean
    out = {}
    for group in ('near', 'far'):
        for dose in DOSES:
            errors = {k: [] for k in ('full', 'one', 'unchanged')}
            correct = dict.fromkeys(errors, 0)
            targets = other = sham_changes = 0
            for p, m in zip(pred, measured, strict=True):
                assert p['pid'] == m['pid']
                if p['group'] != group:
                    continue
                sham = m['doses']['0.0']
                obs = m['doses'][str(dose)]
                shift = obs['margin'] - sham['margin']
                guesses = dict(p['doses'][str(dose)], unchanged=dict(winner=p['winner'], shift=0))
                for k, g in guesses.items():
                    correct[k] += int(g['winner'] == obs['winner'])
                    errors[k].append(abs(g['shift']-shift))
                targets += int(obs['winner'] == p['runner'])
                other += int(obs['winner'] not in (p['runner'], p['winner']))
                sham_changes += int(sham['winner'] != p['winner'])
            out[f'{group}_{dose}'] = dict(n=len(errors['full']), winner_correct=correct,
                mean_absolute_error={k:mean(vals) for k,vals in errors.items()},
                target_wins=targets, third_token_wins=other, sham_winner_changes=sham_changes)
    return out


def main(stage: str = 'screen'):
    assert stage in ('screen', 'predict', 'evaluate')
    RUN.mkdir(exist_ok=True)
    path = RUN / f'{stage}.json'
    if path.exists():
        raise FileExistsError(path)
    inputs = prompts()
    raw = b''
    if stage != 'screen':
        screen = json.loads((RUN/'screen.json').read_text())
        inputs = select(screen['rows'])
        assert all(sum(p['group']==g for p in inputs) >= 12 for g in ('near', 'far'))
        raw = (ROOT/'llever_pushes.npz').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == SHA
        if stage == 'evaluate':
            pred = json.loads((RUN/'predict.json').read_text())
            assert pred['stage'] == 'predict' and [r['pid'] for r in pred['rows']] == [p['pid'] for p in inputs]
    fn = screen_job if stage == 'screen' else followup_job
    report = fn.remote(stage, inputs, raw)
    path.write_text(json.dumps(report, indent=2))
    if stage == 'screen':
        print({g:sum(p['group']==g for p in report['rows']) for g in ('near', 'far', None)})
    if stage == 'evaluate':
        result = summarize(pred['rows'], report['rows'])
        (RUN/'summary.json').write_text(json.dumps(result, indent=2))
        print(result)
    print(stage, report['runtime_s'])


if old.modal:
    app = old.modal.App('llever-winner')
    options = dict(image=old.image.add_local_python_source('llever_app'), gpu='A100-40GB',
                   volumes={'/cache':old.hf_cache}, cpu=2, memory=16384, max_containers=1,
                   retries=0, scaledown_window=2)
    screen_job = app.function(**options, timeout=600, name='screen')(run)
    followup_job = app.function(**options, timeout=1200, name='followup')(run)
    main = app.local_entrypoint()(main)
