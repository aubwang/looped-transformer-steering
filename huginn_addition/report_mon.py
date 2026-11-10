#!/usr/bin/env python3
"""Report registered monitor analyses using only results.json and the standard library.

No fitting, array loading, or inference across split seeds is performed. Paths in
headline quantities are JSON component lists (estimate keys can contain '/').
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys

ORDER = ('m1a', 'm1b', 'm1c', 'm3a', 't1', 'm4', 'm1s', 'm1x', 'f16', 'dyn')
DESCRIPTIONS = {
    'm1a': ('M1-a: cross-label 2×2 table (§6)', 'Rows name whose correctness is predicted; columns name whose state is read (L32 x32 and D h12). Row and column effects are half-sum contrasts; interaction is the difference of differences.'),
    'm1b': ('M1-b: nested best site (§6)', 'Inner folds jointly select site and penalty; held-out AUROC compares each model predicting its own errors.'),
    'm1c': ('M1-c: added value over confidence (§6)', 'Added value is AUROC(confidence + probe) minus AUROC(confidence), also using flexible confidence; added_difference compares L32 with D. False/True key components denote fixed/best sites and linear/flexible confidence, respectively.'),
    'm1s': ('M1 secondary checks (§6)', 'Own-budget and budget-32 probes, position −2, accepted-answer error at 80% coverage, sensitivity fits and permutation diagnostics use held-out scores. Concordant rows condition on both models being right or both wrong: this is a selected population.'),
    'm1x': ('M1 cross-family transfer, secondary (§6)', 'Frozen source-family diagonal probes score each target with their mean score; single_probe reports the mean of individual-probe target AUROCs. minus_own_family uses like-for-like single-probe AUROCs; drop_difference compares those drops. These compare a held-out fifth with a whole target family, not an isolated transfer loss.'),
    'm3a': ('M3-a: budget portability (§6)', 'Frozen source probes and source held-out 80th-percentile thresholds transfer to each target budget; native_minus_transfer subtracts transfer AUROC from native AUROC. Strict rejection above threshold allows ties to change coverage. Matrix rows describe coordinate alignment across depths.'),
    't1': ('T1 / M3-b: stability forecast (§8)', 'Forecasts bound movement to budget 32. Primary calibration compares scaled_rule with constant; other predictors are secondary. Coverage excludes absent forecasts; stability counts them as not stable. Nominal marginal coverage is 95%, not coverage conditional on stability; thresholds are transductive.'),
    'm4': ('M4-a: dynamical signals (§6)', 'Added value is AUROC(confidence + dynamics) minus AUROC(confidence), using only history up to the budget. D uses layer-step and readout-trajectory features and is descriptive.'),
    'f16': ('Float16 checks (§4, §8)', 'Frozen probes compare float32 copies with stored float16 states; score changes are scaled by score standard deviation. T1 rounding_subset reports movement and decision comparisons.'),
    'dyn': ('Dynamics and positional rates (§8)', 'Relative steps, step ratios, cosines and token changes describe convergence and two-cycles. Position and prefix log rates use finite windows; prefix rates refer to positions taken together. Floor counts distinguish saved per-position norms from whole-state norm divided by square root of prompt length.'),
}
AUDIT_KEYS = {'nonconverged', 'nonconverged_outer_fits', 'source_nonconverged', 'inner_nonconverged', 'undefined_bootstrap_draws', 'transfer_undefined_bootstrap_draws', 'families', 'transfer_families'}
OMIT = AUDIT_KEYS | {'interval', 'transfer_interval', 'L32_interval', 'D_interval', 'difference_interval', 'incomplete', 'transfer_incomplete', 'chosen', 'solver_messages', 'interval_unavailable_reason', 'transfer_interval_unavailable_reason'}


# Field names from the registered analysis schema, including bookkeeping.
KNOWN_FIELDS = frozenset(['D_auroc', 'D_auroc_single_probe', 'D_classes', 'D_interval', 'D_minus_own_family', 'L32_auroc', 'L32_auroc_single_probe', 'L32_classes', 'L32_interval', 'L32_minus_own_family', 'accepted_error', 'accuracy', 'allowance_over_movement', 'auroc', 'auroc_change_fixed_labels', 'bound_over_realized', 'calibration_sizes', 'chosen', 'classes', 'classification_transfer', 'close_pair_mass', 'constant', 'cosine', 'coverage', 'coverage_with_allowance', 'd_pos_full_norm_relative_error', 'decision_change_bound', 'decision_changed', 'difference', 'difference_interval', 'drop_difference', 'error_32', 'error_r', 'exact_n', 'exact_violation', 'false_stability', 'families', 'flag_changed', 'flag_changed_32', 'flag_changed_r', 'floor', 'forecast_fraction', 'fraction', 'fraction_adjacent_drops', 'fraction_forecast_stable', 'fraction_forecast_stable_with_allowance', 'fraction_ratio_increases_3_16', 'fraction_rejected', 'inner_nonconverged', 'interval', 'items_used', 'kappa', 'label_changed', 'largest_shortfall_below_running_max', 'last_position_displacement', 'last_position_minus_median', 'last_position_minus_whole_state', 'last_position_storage', 'last_position_violation', 'log_rate', 'max_score_change_over_sd', 'max_score_error_over_q_32', 'max_score_error_over_q_r', 'median', 'median_score_change_over_sd', 'missing_targets', 'n', 'n_evaluation', 'native_auroc', 'nonconverged', 'nonconverged_outer_fits', 'p10', 'p90', 'path_length', 'path_violation', 'per_position_norm_items', 'position_rate', 'positions_used', 'prefix_fraction_adjacent_drops', 'prefix_items_used', 'prefix_largest_shortfall_below_running_max', 'prefix_log_rate', 'prefix_spearman_position_vs_rate', 'q', 'quantization_allowance', 'ranking_folds_used', 'ratio', 'ratio_times', 'raw_rule_stability_changed', 'realized_score_movement', 'relative_movement_change', 'relative_step', 'rounding_subset', 'scaled_rule', 'scaled_rule_last', 'scaled_rule_two_step', 'scaled_rule_two_step_last', 'scaled_step', 'scaled_step_last', 'scaled_two_step', 'scaled_two_step_last', 'score_bound', 'score_bound_over_realized', 'score_conformal', 'score_violation', 'score_violation_beyond_quantization', 'skipped_optional', 'slack', 'solver_messages', 'source_nonconverged', 'spearman_position_vs_rate', 'stable_and_violated', 'state_bound', 'state_conformal', 'state_conformal_target', 'targets', 'token_changes', 'transfer_auroc', 'transfer_families', 'transfer_interval', 'transfer_undefined_bootstrap_draws', 'two_cycle_fraction', 'undefined_bootstrap_draws', 'violated_pair_mass', 'whole_state_norm_items'])

def number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def sanitize(x):
    if isinstance(x, float) and not math.isfinite(x): return None
    if isinstance(x, dict): return {k: sanitize(v) for k, v in x.items()}
    if isinstance(x, list): return [sanitize(v) for v in x]
    return x


def leaves(x, path=(), numeric_only=False, filtered=False):
    if isinstance(x, dict):
        for k, v in x.items():
            if not filtered or k not in OMIT:
                yield from leaves(v, path+(k,), numeric_only, filtered)
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from leaves(v, path+(i,), numeric_only, filtered)
    elif (number(x) if numeric_only else x is None or isinstance(x, (int, float, bool))):
        yield path, x


def get(x, path, default=None):
    try:
        for k in path: x = x[k]
        return x
    except (KeyError, IndexError, TypeError): return default


def label(path):
    return '/'.join(map(str, path))


def fmt(x):
    if x is None: return 'unavailable'
    if isinstance(x, bool): return str(x).lower()
    return f'{x:.3f}' if number(x) else str(x)


def md(x):
    return str(x).replace('|', '&#124;').replace('\n', ' ')


def table(headers, rows):
    return ['| ' + ' | '.join(map(md, headers)) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] + ['| ' + ' | '.join(map(md, row)) + ' |' for row in rows] + ['']


def tier(name, path):
    key = str(path[0]) if path else ''
    if name in ('m1a', 'm1b'): return 'secondary' if path[-1:] == ('accepted_error',) else 'primary'
    if name == 'm1c': return 'secondary' if any(t in key for t in ('early', 'output_identity', 'privileged')) or path[-1:] == ('accepted_error',) else 'primary'
    if name == 'm3a':
        if key.startswith('matrix/'): return 'secondary'
        if '->' in key:
            target = key.split('->')[1].split('/')[0]
            if target not in ('2', '4', '8', '16', '64'): return 'secondary'
        return 'primary'
    if name == 'm4': return 'secondary' if key.startswith('D/') or path[-1:] == ('accepted_error',) else 'primary'
    if name == 't1':
        if len(path)>2 and path[1] in ('state_conformal', 'score_conformal'):
            return 'primary' if path[2] in ('constant', 'scaled_rule') else 'secondary'
        return 'primary'
    return 'secondary'


def interval_for(row, path):
    """Return interval plus whether this leaf is an interval-bearing estimand."""
    # Native target intervals live on the separately fitted native-budget row.
    if path[-1] == 'native_auroc' and '->' in str(path[0]):
        model = str(path[0]).split('/')[0]
        target = str(path[0]).split('->')[1].split('/')[0]
        return interval_for(row, (model+'/native'+target, 'auroc'))
    parent = get(row, path[:-1], {})
    if not isinstance(parent, dict): return None, False
    key = path[-1]
    ik = {'auroc': 'interval', 'difference': 'difference_interval' if 'difference_interval' in parent else 'interval',
          'transfer_auroc': 'transfer_interval', 'L32_auroc': 'L32_interval', 'D_auroc': 'D_interval'}.get(key)
    if ik is None: return None, False
    value = parent.get(ik)
    valid = isinstance(value, list) and len(value)==2 and all(number(v) for v in value)
    return value if valid else None, ik in parent or 'interval_unavailable_reason' in parent or 'transfer_interval_unavailable_reason' in parent


def flags(row, path, missing=False, skips=()):
    estimate = row.get('estimates', {})
    parents = [get(estimate, path[:i], {}) for i in range(1, len(path))]
    fail = any(any(p.get(k, 0) for k in ('nonconverged', 'nonconverged_outer_fits', 'source_nonconverged')) for p in parents if isinstance(p, dict))
    inc = estimate.get('incomplete', {}) or {}
    incomplete = any(label(path).startswith(str(k)) for k in inc) if isinstance(inc, dict) else bool(inc)
    incomplete |= any(bool(p.get('incomplete') or p.get('transfer_incomplete') or p.get('missing_targets')) for p in parents if isinstance(p, dict))
    _, expected = interval_for(estimate, path)
    iv, _ = interval_for(estimate, path)
    relevant = [s for s in skips if not s.get('name') or str(path[0]).startswith(s['name']) or s['name'].startswith(str(path[0]))]
    return dict(nonconverged=bool(fail), unavailable_interval=bool(expected and iv is None), incomplete=bool(incomplete), skipped=bool(missing or relevant))


def load(inputs):
    documents=[]; seen=set(); merged={}; baseline=None
    for input_path in inputs:
        path=Path(input_path); path=path/'results.json' if path.is_dir() else path
        doc=sanitize(json.loads(path.read_text()))
        p,s=doc['provenance'],doc['settings']
        signature={'provenance':{k:v for k,v in p.items() if 'sha256' in k}, 'settings':{k:s.get(k) for k in ('fams','outer','inner','boot','smoke')}}
        if baseline is not None and signature!=baseline: raise ValueError(f'mismatched provenance or fold/bootstrap settings: {path}')
        baseline=signature
        actual={str(seed) for rows in doc['analyses'].values() for seed in rows}
        if seen & actual: raise ValueError(f'duplicated seed: {sorted(seen & actual)} in {path}')
        seen |= actual
        for name, rows in doc['analyses'].items(): merged.setdefault(name, {}).update(rows)
        documents.append((path,doc,actual))
    seeds=sorted(seen | {str(v) for _,d,_ in documents for v in d['settings'].get('seeds',[])}, key=lambda v:(not v.isdigit(),int(v) if v.isdigit() else v))
    return documents, merged, seeds


def audit(analyses, seeds, families):
    summaries=[]; events=[]
    for name in ORDER:
        for family in families:
            for seed in seeds:
                row=analyses.get(name,{}).get(seed,{}).get(family)
                counts=dict(outer=0, inner=0, intervals=0, skips=0, incomplete=0)
                def event(kind, path, detail):
                    events.append(dict(analysis=name,family=family,seed=seed,key=label(path),kind=kind,detail=detail,tier=tier(name,path)))
                if row is None:
                    counts['skips']=1; event('skipped', ('*',), 'missing seed-family row')
                else:
                    def walk(x, path=()):
                        if not isinstance(x,dict): return
                        for k,v in x.items():
                            if k in ('nonconverged','nonconverged_outer_fits','source_nonconverged') and v:
                                counts['outer']+=v; event('nonconverged',path+(k,),str(v))
                            elif k=='inner_nonconverged' and v: counts['inner']+=v
                            elif k.endswith('interval') and (v is None or not isinstance(v,list) or any(q is None for q in v)):
                                counts['intervals']+=1; event('unavailable_interval',path+(k,), x.get(k+'_unavailable_reason',x.get('interval_unavailable_reason','interval absent or nonfinite')))
                            elif k in ('incomplete','transfer_incomplete') and v:
                                counts['incomplete']+=len(v) if isinstance(v,dict) else 1
                                event('incomplete',path+(k,),json.dumps(v,sort_keys=True))
                            elif isinstance(v,dict): walk(v,path+(k,))
                    walk(row.get('estimates',{}))
                    for skip in row.get('skipped',[]):
                        counts['skips']+=1; event('skipped',(skip.get('name','*'),),json.dumps(skip,sort_keys=True))
                summaries.append([name,family,seed,*[fmt(counts[k]) for k in counts]])
    return summaries,events


def make_report(inputs, out):
    docs, analyses, seeds=load(inputs)
    families=list(docs[0][1]['settings']['fams'])+['family_mean']
    lines=['# Registered monitor analysis report','', '## Provenance and audit','']
    for path,doc,actual in docs:
        p,s=doc['provenance'],doc['settings']
        lines += [f'### {path}', '']
        if s.get('smoke'): lines += ['**WARNING: SMOKE CRASH CHECK — THESE NUMBERS ARE NOT RESULTS.**','']
        lines += [f'analysis_matches_manifest: `{json.dumps(p.get("analysis_matches_manifest"))}`; manifest: `{p.get("manifest", "unavailable")}`; manifest SHA256: `{p.get("manifest_sha256", "unavailable")}`.', '',f'Seeds present: {", ".join(sorted(actual))}. Wall seconds: {fmt(doc.get("seconds"))}; analysis worker seconds: {json.dumps(doc.get("analysis_worker_seconds"))}.', '', 'Settings: `'+json.dumps(s,sort_keys=True)+'`.', '', 'Hashes: `'+json.dumps({k:v for k,v in p.items() if 'sha256' in k},sort_keys=True)+'`.', '', 'Collection completeness: `'+json.dumps(p.get('completeness'))+'`.', '']
        lines += table(['Model','Family','Items'], [[m,f,counts.get(f,'unavailable')] for m,counts in p.get('item_counts',{}).items() for f in families[:-1]])
    summaries,events=audit(analyses,seeds,families)
    lines += ['## Audit of caveats','', 'Counts are reported field occurrences (contrast failure counts can repeat component fits); family-mean counts are the analysis aggregates, not additional fits. Missing rows are skips. Exact primary cells follow; secondary cells are counted above.','']
    lines += table(['Analysis','Family','Seed','Outer failures','Inner failures','Unavailable intervals','Skipped cells','Incomplete'],summaries)
    lines += table(['Analysis','Family','Seed','Key','Caveat','Reason'],[[e[k] for k in ('analysis','family','seed','key','kind','detail')] for e in events if e['tier']=='primary'])
    lines += ['Intervals condition on frozen out-of-fold fits and omit refitting uncertainty. Family means weight all selected families equally. Seed medians and min–max ranges use available estimates and explicitly identify missing seeds; they are not confidence intervals. “not reported” means the analysis supplies no interval for that quantity.','']
    headline={'schema_version':1,'seeds':seeds,'families':families,'quantities':[], 'audit':events, 'fields_not_interpreted':[]}
    unknown=[]
    for name in ORDER:
        title,description=DESCRIPTIONS[name]
        lines += ['## '+title,'',description,'']
        rows=analyses.get(name,{})
        if name == 'm1a':
            models = sorted({k.split('/')[1] for fs in rows.values() for row in fs.values()
                             for k in row.get('estimates', {}) if k.startswith('table/')})
            cells = []
            for family in families:
                for seed in seeds:
                    for target in models:
                        values=[]
                        for source in models:
                            row=rows.get(seed,{}).get(family,{})
                            p=('table/'+target+'/'+source,'auroc')
                            e=get(row.get('estimates',{}),p)
                            iv,_=interval_for(row.get('estimates',{}),p)
                            fl=flags(row,p,missing=e is None,skips=row.get('skipped',[]))
                            values.append(fmt(e)+' '+('['+', '.join(map(fmt,iv))+']' if iv else '[unavailable]')+
                                          (' ('+', '.join(k for k,v in fl.items() if v)+')' if any(fl.values()) else ''))
                        cells.append([family,seed,target,*values])
            lines += ['The full 2×2 table predicts row-model correctness from column-model state; entries are AUROC and 95% interval.', '']
            lines += table(['Family','Seed','Label model',*[m+' state' for m in models]],cells)
        if name == 't1':
            checks=[]
            for seed,fs in rows.items():
                for family,row in fs.items():
                    for key,est in row.get('estimates',{}).items():
                        if not isinstance(est,dict): continue
                        c=est.get('classification_transfer',{}) or {}
                        if c:
                            vals=[c.get(k) for k in ('error_r','error_32','flag_changed','label_changed')]
                            checks.append([family,seed,key,'classification: error_r ≤ error_32 + flag_changed + label_changed',
                                           str(vals[0] <= sum(vals[1:])+1e-12) if all(number(v) for v in vals) else 'unavailable'])
                        for block in ('state_conformal','score_conformal'):
                            for candidate,v in (est.get(block,{}) or {}).items():
                                if not isinstance(v,dict): continue
                                for title,left,right in (
                                    ('decision_changed ≤ decision_change_bound','decision_changed',('decision_change_bound',)),
                                    ('fixed-label AUROC change ≤ close-pair + violated-pair mass','auroc_change_fixed_labels',('close_pair_mass','violated_pair_mass'))):
                                    values=[v.get(left),*[v.get(k) for k in right]]
                                    if any(k in v for k in (left,*right)):
                                        checks.append([family,seed,key+'/'+block+'/'+candidate,title,
                                                       str(values[0]<=sum(values[1:])+1e-12) if all(number(a) for a in values) else 'unavailable'])
            lines += ['Classification, decision and fixed-label ranking inequalities are evaluated from the reported sides; coverage quantities follow in the forecast tables.', '']
            lines += table(['Family','Seed','Forecast','Inequality','Holds'],checks)
        for fs in rows.values():
            for row in fs.values():
                def unfamiliar(value, path):
                    if isinstance(value, dict):
                        for k,v in value.items():
                            if k.isidentifier() and k not in KNOWN_FIELDS:
                                unknown.append(name+'/'+label(path+(k,)))
                            unfamiliar(v,path+(k,))
                    elif isinstance(value,list):
                        for i,v in enumerate(value): unfamiliar(v,path+(i,))
                for k,v in row.get('estimates',{}).items():
                    if k != 'incomplete': unfamiliar(v,(k,))
        paths=set()
        for fs in rows.values():
            for row in fs.values(): paths.update(p for p,v in leaves(row.get('estimates',{}),filtered=True))
        # Deliberate grouping by the analysis's named fits, transfer targets or model.
        # A null parent is represented by its available descendants, not as a
        # second scalar quantity alongside those descendants.
        paths = {p for p in paths if not any(q != p and q[:len(p)] == p for q in paths)}
        for path in paths:
            for field in path[1:]:
                if isinstance(field, str) and field.isidentifier() and field not in KNOWN_FIELDS:
                    unknown.append(name+'/'+label(path))
        if not paths:
            lines += ['unavailable: no estimates were supplied for this analysis.', '']
        keys=sorted({p[0] for p in paths},key=str)
        for key in keys:
            lines += ['### '+str(key),'']
            if name=='m1a' and str(key).startswith('table/'):
                _,target,source=str(key).split('/')
                lines += [f'2×2 cell: predict {target} correctness from {source} state.','']
            else: lines += [description,'']
            display=[]
            for path in sorted((p for p in paths if p[0]==key),key=lambda p:tuple(map(str,p))):
                quantity={'analysis':name,'key':label(path),'path':list(path),'tier':tier(name,path),'families':{}}
                for family in families:
                    byseed={}; available=[]; missing=[]; cells=[]
                    for seed in seeds:
                        row=rows.get(seed,{}).get(family,{})
                        value=get(row.get('estimates',{}),path)
                        iv,expected=interval_for(row.get('estimates',{}),path)
                        fl=flags(row,path,missing=not row or value is None,skips=row.get('skipped',[]))
                        # Cross-family target scores inherit failures from both source fits.
                        if name=='m1x' and (str(key).startswith('to/') or key=='mean_over_targets'):
                            fl['nonconverged'] |= any(v.get('nonconverged',0)>0 for k,v in row.get('estimates',{}).items() if k.endswith('/source') and isinstance(v,dict))
                        byseed[seed]={'estimate':value,'interval':iv,'interval_status':'available' if iv else 'unavailable' if expected else 'not_reported','flags':fl}
                        if number(value): available.append(value)
                        else: missing.append(seed)
                        flagtext=', '.join(k for k,v in fl.items() if v)
                        cells.append(fmt(value)+'; '+('['+', '.join(map(fmt,iv))+']' if iv else 'interval unavailable' if expected else 'interval not reported')+(f' ({flagtext})' if flagtext else ''))
                    summary={'median':statistics.median(available) if available else None,'min':min(available) if available else None,'max':max(available) if available else None,'available_seeds':len(available),'missing_seeds':missing}
                    quantity['families'][family]={'per_seed':byseed,'across_seeds':summary,'flags':{k:any(v['flags'][k] for v in byseed.values()) for k in ('nonconverged','unavailable_interval','incomplete','skipped')}}
                    display.append([label(path[1:]),family,*cells,fmt(summary['median']),f'[{fmt(summary["min"])}, {fmt(summary["max"])}]'+(f'; unavailable seeds: {", ".join(missing)}' if missing else '')])
                headline['quantities'].append(quantity)
            lines += table(['Quantity','Family',*[f'Seed {s}: estimate; 95% interval' for s in seeds],'Seed median','Seed min–max'],display)
        # Site selection and string-valued bookkeeping cannot be summarized numerically.
        bookkeeping=[]
        for seed,fs in rows.items():
            for family,row in fs.items():
                for key,val in row.get('estimates',{}).items():
                    if isinstance(val,dict):
                        for field in ('chosen','missing_targets','state_conformal_target','last_position_storage','skipped_optional'):
                            if val.get(field):
                                bookkeeping.append([family,seed,key,field,json.dumps(val[field])])
                                if field == 'chosen':
                                    counts=Counter(str(c[0]) for c in val[field] if isinstance(c,list) and c)
                                    bookkeeping.append([family,seed,key,'site_selection_counts',json.dumps(dict(counts),sort_keys=True)])
        if bookkeeping: lines += ['Site selections and analysis bookkeeping (chosen entries are site / penalty per outer fold):','']+table(['Family','Seed','Fit','Field','Value'],bookkeeping)
    headline['fields_not_interpreted'] = sorted(set(unknown))
    lines += ['## Fields not interpreted', '']
    lines += [f'- `{p}` (retained without interpretation)' for p in sorted(set(unknown))] if unknown else ['None in the registered analysis schema.']
    if unknown:
        for name in ORDER:
            for seed,fs in analyses.get(name,{}).items():
                for family,row in fs.items():
                    for key,value in row.get('estimates',{}).items():
                        if any(p.startswith(name+'/'+key+'/') for p in unknown):
                            lines += [f'Uninterpreted-field context: {name} / {family} / seed {seed} / {key}', '', '```json', json.dumps(value,sort_keys=True,allow_nan=False), '```', '']
    lines += ['', 'Numeric fields retain their analysis names; array indices are zero-based (dyn ratio index i corresponds to loop i+2).', '', '## Appendix: every family_mean.estimates numeric leaf', '']
    for name in ORDER:
        for seed in seeds:
            lines += [f'### {name} / seed {seed}', '', '```text']
            for path,value in leaves(get(analyses,(name,seed,'family_mean','estimates'),{}),numeric_only=True):
                lines.append(f'{label(path)} = {json.dumps(value,allow_nan=False)}')
            lines += ['```','']
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    (out/'headline.json').write_text(json.dumps(sanitize(headline),allow_nan=False,separators=(',',':'))+'\n')
    return out


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',nargs='+',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    try: out=make_report(args.inputs,args.out)
    except (ValueError,OSError,KeyError,TypeError) as exc: parser.exit(1,f'report_mon: {exc}\n')
    print(f'Wrote {out / "report.md"} and {out / "headline.json"}')


if __name__=='__main__': main()
