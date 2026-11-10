"""Compact text view of the registered monitor analysis (reads results.json of each seed; changes nothing).

A cell is `m [lo, hi]`: m = median over seeds of the point estimate, [lo, hi] = the widest envelope of the seeds' 95%
intervals (smallest lower end, largest upper end). Marks: `!` an outer fit failed the solver test in some seed,
`?` an interval is unavailable in some seed, `NA` the quantity is missing in some seed. The full per-seed numbers are in
report_mon.py's report; this view is for reading.
"""
import argparse, json, statistics, sys
from pathlib import Path

P = argparse.ArgumentParser(); P.add_argument('--inputs', nargs='+', required=True); P.add_argument('--sections', default='')
args = P.parse_args()
RES = {}
for d in args.inputs:
    p = Path(d); p = p / 'results.json' if p.is_dir() else p
    r = json.loads(p.read_text())
    for name, by in r['analyses'].items():
        for seed, fams in by.items(): RES.setdefault(seed, {'meta': r})[name] = fams
SEEDS = sorted(RES, key=int)
meta = RES[SEEDS[0]]['meta']
FAMS = list(meta['settings']['fams']) + ['family_mean']
WANT = set(args.sections.split(',')) if args.sections else None


def get(name, seed, fam, key):
    try: return RES[seed][name][fam]['estimates'][key]
    except KeyError: return None


def sub(d, path):
    for k in path.split('.'):
        if not isinstance(d, dict) or k not in d: return None
        d = d[k]
    return d


def cell(name, fam, key, field, ivl='interval', digits=3):
    vals, los, his, marks = [], [], [], ''
    for s in SEEDS:
        e = get(name, s, fam, key)
        v = sub(e, field) if e is not None else None
        if not isinstance(v, (int, float)) or isinstance(v, bool): return 'NA'
        vals.append(v)
        if ivl:
            iv = sub(e, ivl)
            if isinstance(iv, list) and len(iv) == 2 and all(isinstance(x, (int, float)) for x in iv): los.append(iv[0]); his.append(iv[1])
            elif '?' not in marks: marks += '?'
        bad = (e.get('nonconverged') or 0) + (e.get('nonconverged_outer_fits') or 0) + (e.get('source_nonconverged') or 0)
        if bad and '!' not in marks: marks += '!'
    out = f'{statistics.median(vals):.{digits}f}'
    if ivl and len(los) == len(SEEDS): out += f' [{min(los):.{digits}f},{max(his):.{digits}f}]'
    return out + marks


def table(title, name, rows, fams=None, width=24):
    """rows: (label, key, field, interval-field or None)."""
    fams = fams or FAMS
    print(f'\n-- {title}')
    print(f'{"":34}' + ''.join(f'{f:>{width}}' for f in fams))
    for label, key, field, ivl in rows:
        print(f'{label:34}' + ''.join(f'{cell(name, f, key, field, ivl):>{width}}' for f in fams))


def section(tag, title):
    if WANT and tag not in WANT: return False
    print(f'\n\n======== {tag}: {title}'); return True


A = lambda k: (k, k, 'auroc', 'interval')
D = lambda k: (k, k, 'difference', 'interval')
LOOPED = [m for m in ('L4', 'L8', 'L16', 'L32')]

if section('audit', 'provenance and caveats'):
    for s in SEEDS:
        r = RES[s]['meta']; p = r['provenance']
        print(f'seed {s}: matches_manifest={p.get("analysis_matches_manifest")} smoke={r["settings"]["smoke"]} outer={r["settings"]["outer"]} inner={r["settings"]["inner"]} boot={r["settings"]["boot"]} seconds={r["seconds"]:.0f}')
    print('item counts:', json.dumps(meta['provenance'].get('item_counts')))
    print('output identity tokens:', meta['settings'].get('output_identity_tokens'))
    print(f'{"analysis":8}{"seed":>5}{"cells":>7}{"outer_fail":>11}{"cells_with":>11}{"inner_fail":>11}{"ivl_unavail":>12}{"undef_draws":>12}  skipped')
    listing = []
    for name in meta['analyses']:
        for s in SEEDS:
            n = outer = cells_outer = inner = unavail = undef = 0; skipped = []
            for fam in FAMS[:-1]:
                row = RES[s].get(name, {}).get(fam)
                if row is None: skipped.append(f'{fam}: MISSING ROW'); continue
                for sk in row.get('skipped') or []: skipped.append(f'{fam}: {sk}')
                for key, e in row['estimates'].items():
                    if not isinstance(e, dict): continue
                    n += 1
                    o = sum(int(e.get(k) or 0) for k in ('nonconverged', 'nonconverged_outer_fits', 'source_nonconverged'))
                    outer += o; cells_outer += o > 0; inner += int(e.get('inner_nonconverged') or 0)
                    if o: listing.append(f'  outer failure: {name} seed {s} {fam} {key} ({o})')
                    for k, v in e.items():
                        if k.endswith('interval') and (v is None or (isinstance(v, list) and any(x is None for x in v))):
                            unavail += 1; listing.append(f'  interval unavailable: {name} seed {s} {fam} {key}.{k}')
                        if k.endswith('undefined_bootstrap_draws') and v: undef += int(v)
            print(f'{name:8}{s:>5}{n:>7}{outer:>11}{cells_outer:>11}{inner:>11}{unavail:>12}{undef:>12}  {skipped[:6]}{" …" if len(skipped) > 6 else ""}')
    print(f'{len(listing)} flagged cells' + (' (first 60):' if len(listing) > 60 else ':')); print('\n'.join(listing[:60]))

if section('m1a', '2x2 table. table/R/C = probe on the states of model C predicting the errors of model R'):
    table('AUROC', 'm1a', [A(f'table/{r}/{c}') for r in ('L32', 'D') for c in ('L32', 'D')] + [('n common', 'common/L32,D', 'n', None)])
    table('contrasts: diagonal = LL-DD; row = whose errors (L32 minus D); column = whose states (L32 minus D); interaction = LL-LD-DL+DD', 'm1a', [D(k) for k in ('diagonal', 'row', 'column', 'interaction')])

if section('m1b', 'best site chosen inside the cross-validation'):
    table('AUROC', 'm1b', [A('best/L32'), A('best/D'), D('best_difference')])

if section('m1c', 'added value over confidence (fixed = registered site, best = nested best site)'):
    for mode in ('fixed', 'best'):
        table(f'{mode}', 'm1c', [x for m in ('L32', 'D') for x in (A(f'{mode}/{m}/confidence'), A(f'{mode}/{m}/flex'), A(f'{mode}/{m}/probe'), A(f'{mode}/{m}/confidence+probe'),
                                                               A(f'{mode}/{m}/flex+probe'), D(f'{mode}/{m}/confidence/added'), D(f'{mode}/{m}/flex/added'))])
    table('L32 added minus D added; key = best/flex', 'm1c', [D(f'{b}/{f}/added_difference') for b in ('False', 'True') for f in ('False', 'True')])
    table('controls', 'm1c', [A(f'{m}/{k}') for m in ('L32', 'D') for k in ('output_identity', 'privileged_ground_truth', 'early')])

if section('m1s', 'secondary: every loop count, earlier sites, concordant items, simple probes'):
    rows = []
    for m in LOOPED:
        for r in dict.fromkeys((m[1:], '32')): rows += [A(f'{m}/{r}/probe'), A(f'{m}/{r}/confidence'), A(f'{m}/{r}/joint'), D(f'{m}/{r}/added')]
        rows += [A(f'{m}/prev8'), A(f'{m}/prev32')]
    rows += [A('D/prev8'), A('D/prev12'), A('D/prev16')]
    table('per model', 'm1s', rows)
    table('robustness', 'm1s', [('n concordant', 'concordant_population', 'n', None), A('concordant/L32'), A('concordant/D'), D('concordant/diagonal_difference')]
          + [A(f'{m}/{k}') for k in ('meandiff', 'pca_logistic', 'permutation') for m in ('L32', 'D')] + [D('meandiff/diagonal_difference'), D('pca_logistic/diagonal_difference')])

if section('m1x', 'cross-family transfer; column = SOURCE family (probe trained there)'):
    fams = FAMS[:-1]
    table('own-family AUROC of the source probe', 'm1x', [A('L32/source'), A('D/source')], fams)
    for t in fams:
        table(f'target {t}', 'm1x', [('L32', f'to/{t}', 'L32_auroc', 'L32_interval'), ('D', f'to/{t}', 'D_auroc', 'D_interval'), ('L32-D', f'to/{t}', 'difference', 'difference_interval'),
                                     ('L32 single probe', f'to/{t}', 'L32_auroc_single_probe', None), ('D single probe', f'to/{t}', 'D_auroc_single_probe', None)], fams)
    table('mean over targets', 'm1x', [(k, 'mean_over_targets', k, None) for k in ('targets', 'L32_auroc', 'D_auroc', 'difference', 'L32_auroc_single_probe', 'D_auroc_single_probe',
                                                                              'L32_minus_own_family', 'D_minus_own_family', 'drop_difference')])

if section('m3a', 'budget transfer: probe trained at loop count a, applied at b'):
    TG = ('1', '2', '4', '8', '16', '32', '48', '64')
    for m in LOOPED:
        for src in dict.fromkeys((m[1:], '32')):
            rows = [A(f'{m}/source{src}')]
            for b in TG:
                rows += [(f'{src}->{b} transfer', f'{m}/{src}->{b}', 'transfer_auroc', 'transfer_interval'), (f'  native at {b}', f'{m}/native{b}', 'auroc', 'interval'),
                         (f'  native-transfer', f'{m}/{src}->{b}/native_minus_transfer', 'difference', 'interval'), (f'  labels changed', f'{m}/{src}->{b}', 'label_changed', None)]
            table(f'{m}, source {src}', 'm3a', rows)
    keys = sorted({k for s in SEEDS for k in RES[s]['m3a'][FAMS[0]]['estimates'] if k.startswith('matrix/')})
    for m in ('L32', 'D'):
        ks = [k for k in keys if k.startswith(f'matrix/{m}/') and '->' in k]
        table(f'matrix {m} (fit a -> apply b), plain AUROC', 'm3a', [(k[7:], k, 'auroc', None) for k in ks])

if section('t1', 'stability forecast'):
    F = ['forecast_fraction', 'path_violation', 'last_position_violation', 'exact_violation', 'bound_over_realized.median', 'score_violation', 'score_violation_beyond_quantization',
         'fraction_forecast_stable', 'false_stability', 'score_bound_over_realized.median', 'classification_transfer.error_32', 'classification_transfer.error_r',
         'classification_transfer.flag_changed', 'classification_transfer.label_changed', 'classification_transfer.slack', 'allowance_over_movement.median',
         'rounding_subset.max_score_error_over_q_r', 'rounding_subset.raw_rule_stability_changed', 'd_pos_full_norm_relative_error.p90']
    CF = ['coverage', 'forecast_fraction', 'bound_over_realized.median', 'fraction_forecast_stable', 'false_stability', 'decision_changed', 'decision_change_bound', 'coverage_with_allowance', 'auroc_change_fixed_labels']
    for m in LOOPED:
        for r in ('4', '8', '16'):
            key = f'{m}/{r}'
            if get('t1', SEEDS[0], FAMS[0], key) is None: continue
            rows = [(f, key, f, None) for f in F]
            for rule in ('scaled_rule', 'constant', 'scaled_step', 'scaled_two_step'):
                rows += [(f'state {rule} {f.split(".")[0]}', key, f'state_conformal.{rule}.{f}', None) for f in CF[:3]]
                rows += [(f'score {rule} {f.split(".")[0]}', key, f'score_conformal.{rule}.{f}', None) for f in CF]
            table(f'{m}, forecast made at loop {r} for loop 32', 't1', rows, width=12)

if section('m4', 'dynamics features'):
    table('AUROC', 'm4', [x for m, r in (('L4', '4'), ('L8', '8'), ('L16', '16'), ('L32', '32'), ('D', '16')) for x in (A(f'{m}/{r}/confidence'), A(f'{m}/{r}/dynamics'), A(f'{m}/{r}/joint'), D(f'{m}/{r}/added'))])

if section('f16', 'float16 storage checks'):
    table('', 'f16', [x for m in ('L32', 'D') for x in (A(f'{m}/diagonal'), ('max score change / sd', f'{m}/rounding', 'max_score_change_over_sd', None), ('median score change / sd', f'{m}/rounding', 'median_score_change_over_sd', None))])

if section('dyn', 'descriptive dynamics'):
    for m in LOOPED:
        e0 = get('dyn', SEEDS[0], FAMS[0], m)
        if e0 is None: continue
        rows = [(f'accuracy at {r}', m, f'accuracy.{r}.accuracy', None) for r in e0['accuracy']]
        rows += [(f'token changes {k}', m, f'token_changes.{k}.fraction', None) for k in e0['token_changes']]
        rows += [('two-cycle fraction', m, 'two_cycle_fraction', None), ('ratio increases 3-16', m, 'fraction_ratio_increases_3_16', None)]
        for w, pr in (e0.get('position_rate') or {}).items():
            for k, v in pr.items():
                if isinstance(v, dict) and 'median' in v: rows.append((f'rate {w} {k}'[:33], m, f'position_rate.{w}.{k}.median', None))
                elif isinstance(v, (int, float)): rows.append((f'rate {w} {k}'[:33], m, f'position_rate.{w}.{k}', None))
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float)): rows.append((f'rate {w} {k}.{kk}'[:33], m, f'position_rate.{w}.{k}.{kk}', None))
                        elif isinstance(vv, dict) and 'median' in vv: rows.append((f'rate {w} {k}.{kk}'[:33], m, f'position_rate.{w}.{k}.{kk}.median', None))
        table(m, 'dyn', rows, width=12)
    e0 = get('dyn', SEEDS[0], FAMS[0], 'D')
    if e0 is not None: print('\nD keys:', list(e0))
