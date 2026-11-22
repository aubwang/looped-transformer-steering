"""Generate auditable Typst tables from finished monitor analyses; never writes inputs.

summarize_mon is a CLI-only module. Load it with its section filter disabled,
then reuse its loader, get/sub accessors, and cell median/envelope aggregation.
"""
import argparse
import contextlib
import importlib.util
import io
import json
import re
import sys
from pathlib import Path

LOOPED = ('L4', 'L8', 'L16', 'L32')


def load_summary(inputs):
    spec = importlib.util.spec_from_file_location('monitor_summary', Path(__file__).with_name('summarize_mon.py'))
    module = importlib.util.module_from_spec(spec)
    argv = sys.argv
    try:
        sys.argv = ['summarize_mon', '--inputs', *map(str, inputs), '--sections', '__none__']
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(module)
    finally:
        sys.argv = argv
    return module


def number(v):
    return f'{v:.3f}'.replace('-', '−')


def format_cell(value, lower=None, upper=None, interval=True, count=False):
    if value is None:
        return '–'
    text = str(int(value)) if count and value == int(value) else number(value)
    if interval:
        text += f' [{number(lower)}, {number(upper)}]' if lower is not None and upper is not None else ' [n/a]'
    return text


def quote(s):
    return json.dumps(s, ensure_ascii=False)


class Generator:
    def __init__(self, summary):
        self.s = summary
        self.cells = []
        self.tables = []

    def estimate(self, table, row, col, a, fam, key, field='auroc', iv='interval', italic=False, transform=None):
        s = self.s
        # The summary's aggregation is the single source of truth, at round-trip precision.
        raw = s.cell(a, fam, key, field, iv, digits=340)
        nums = re.findall(r'-?\d+\.\d+', raw)
        value = float(nums[0]) if nums else None
        lower, upper = (map(float, nums[1:3]) if len(nums) == 3 else (None, None))
        n = sum(isinstance(s.sub(s.get(a, seed, fam, key), field), (int, float)) for seed in s.SEEDS)
        paths = [f'analyses.{a}.{{seed}}.{fam}.estimates.{key}.{field}']
        if transform:
            # Accuracy from a class-count vector is a derived fraction, aggregated
            # with the same summary function via an in-memory synthetic estimate.
            for seed in s.SEEDS:
                e = s.get(a, seed, fam, key)
                v = transform(e) if e else None
                s.RES[seed].setdefault('_derived', {}).setdefault(fam, {'estimates': {}})['estimates'][key] = {'value': v}
            raw = s.cell('_derived', fam, key, 'value', None, digits=340)
            value = float(raw) if raw != 'NA' else None
            n = sum(s.get('_derived', seed, fam, key)['value'] is not None for seed in s.SEEDS)
        text = format_cell(value, lower, upper, bool(iv), field in ('n', 'targets'))
        e = dict(table=table, row_label=row, column_label=col, value=value, lower=lower, upper=upper,
                 number_of_seeds=n, results_json_key=paths[0], analysis=a, family=fam, key=key,
                 field=field, interval_field=iv, formatted=text, italic=italic)
        if transform:
            e['derivation'] = 'classes[0] / sum(classes): fraction correct; mean is unweighted across families per seed'
        self.cells.append(e)
        return ('emph(' if italic else '') + 'text(' + quote(text) + ')' + (')' if italic else '')

    def table(self, name, columns, rows, wide=False, groups=None):
        # rows are (label, [(analysis, family, key, field, interval, ...), ...]).
        lines = [f'#let tab_{name} = table(', f'  columns: {len(columns)+1},',
                 '  align: (left, '+', '.join(['right']*len(columns))+'),', '  inset: 3pt,']
        header = ['text("Quantity")'] + ['text('+quote(c)+')' for c in columns]
        if groups:
            lines.append('  table.header(table.cell(rowspan: 2, text("Target loops")), '+', '.join('table.cell(colspan: 4, align: center, text('+quote(g)+'))' for g in groups)+', '+', '.join(header[1:])+'),')
        else:
            lines.append('  table.header('+', '.join(header)+'),')
        for label, specs in rows:
            vals = [self.estimate(name, label, col, *spec) for col, spec in zip(columns, specs)]
            lines.append('  text('+quote(label)+'), '+', '.join(vals)+',')
        lines.append(')')
        self.tables.append(dict(name=name, source=lines, wide=wide))

    def standard(self, name, a, rows):
        self.table(name, self.s.FAMS, [(label, [(a, f, key, field, iv) for f in self.s.FAMS]) for label, key, field, iv in rows])


def build(s):
    g = Generator(s)
    A = lambda k, label=None: (label or k, k, 'auroc', 'interval')
    D = lambda k, label=None: (label or k, k, 'difference', 'interval')
    # Use fit population counts for all models, including D; no bootstrap CI for accuracy.
    rows = []
    for m in (*LOOPED, 'D'):
        a, k = ('m1a', 'table/D/D') if m == 'D' else ('m1s', f'{m}/{m[1:]}/probe')
        specs = []
        for f in s.FAMS:
            if f == 'family_mean':
                # Insert derived per-seed mean alongside other derived values.
                for seed in s.SEEDS:
                    es = [s.get(a, seed, fam, k) for fam in s.FAMS[:-1]]
                    v = sum(e['classes'][0]/sum(e['classes']) for e in es)/len(es) if all(e and 'classes' in e for e in es) else None
                    s.RES[seed].setdefault('_accuracy', {}).setdefault(f, {'estimates': {}})['estimates'][m] = {'accuracy': v}
                specs.append(('_accuracy', f, m, 'accuracy', None))
            else:
                specs.append((a, f, k, 'classes', None, False, lambda e: e['classes'][0]/sum(e['classes']) if 'classes' in e else None))
        rows.append((m+' accuracy (fraction correct)', specs))
    g.table('accuracy', s.FAMS, rows)
    # Replace synthetic provenance with explicit input paths.
    for c in g.cells:
        if c['analysis'] == '_accuracy':
            m = c['key']; a,k = ('m1a','table/D/D') if m=='D' else ('m1s',f'{m}/{m[1:]}/probe')
            c['results_json_key'] = [f'analyses.{a}.{{seed}}.{f}.estimates.{k}.classes' for f in s.FAMS[:-1]]
            c['derivation'] = 'mean across four families of classes[0] / sum(classes), then median across seeds'
    g.standard('table','m1a',[A(f'table/{r}/{c}', f"errors of {r}, read from {c}'s state") for r in ('L32','D') for c in ('L32','D')] + [D(k,k+' contrast') for k in ('diagonal','row','column','interaction')])
    g.standard('best','m1b',[A('best/L32'), A('best/D'), D('best_difference')])
    for mode, name in [('fixed','added'),('best','added_best')]:
        rows = [x for m in ('L32','D') for x in [A(f'{mode}/{m}/{k}') for k in ('confidence','flex','probe','confidence+probe','flex+probe')] + [D(f'{mode}/{m}/{k}/added') for k in ('confidence','flex')]]
        rows += [D(f'{mode=="best"}/{flex}/added_difference', f'L32 minus D added ({"flexible confidence" if flex else "confidence"})') for flex in (False,True)]
        if mode=='fixed': rows += [A(f'{m}/{k}') for m in ('L32','D') for k in ('output_identity','privileged_ground_truth','early')]
        g.standard(name,'m1c',rows)
    looprows = [x for m in LOOPED for x in [A(f'{m}/{m[1:]}/{k}') if k != 'added' else D(f'{m}/{m[1:]}/{k}') for k in ('probe','confidence','joint','added')]]
    looprows += [A(k) for k in s.RES[s.SEEDS[0]]['m1s'][s.FAMS[0]]['estimates'] if '/prev' in k]
    g.standard('loopcounts', 'm1s', looprows)
    g.standard('robust','m1s',[('concordant items','concordant_population','n',None),A('concordant/L32'),A('concordant/D'),D('concordant/diagonal_difference')] + [x for k in ('meandiff','pca_logistic') for x in (A(f'L32/{k}'),A(f'D/{k}'),D(f'{k}/diagonal_difference'))]+[A(f'{m}/permutation') for m in ('L32','D')])
    rows=[]
    for m in ('L32','D'):
        for source in s.FAMS[:-1]:
            rows.append((f'{m}: source {source}', [('m1x',source,f'{m}/source','auroc','interval',True) if target==source else ('m1x',source,f'to/{target}',f'{m}_auroc',f'{m}_interval') for target in s.FAMS[:-1]]))
    g.table('cross',s.FAMS[:-1],rows)
    g.standard('cross_mean', 'm1x', [(k, 'mean_over_targets', k, None) for k in ('L32_auroc','D_auroc','difference','L32_minus_own_family','D_minus_own_family','drop_difference')])
    targets = sorted({int(k.split('->')[1]) for k in s.RES[s.SEEDS[0]]['m3a']['family_mean']['estimates'] if '->' in k and k.startswith('L') and k.split('->')[1].isdigit()})
    for name, own in [('budget',True),('budget32',False)]:
        rows=[]
        for b in targets:
            specs=[]
            for m in LOOPED:
                src=m[1:] if own else '32'; k=f'{m}/{src}->{b}'
                specs += [('m3a','family_mean',k,'transfer_auroc','transfer_interval'),('m3a','family_mean',f'{m}/native{b}','auroc','interval'),('m3a','family_mean',k+'/native_minus_transfer','difference','interval'),('m3a','family_mean',k,'label_changed',None)]
            rows.append((str(b)+(' †' if b in (48,63,64) else ''),specs))
        g.table(name, [k for m in LOOPED for k in ('Transfer','Native','Native − transfer','Labels changed')],rows,True,LOOPED)
    g.standard('budget_L32_families','m3a',[D(f'L32/32->{b}/native_minus_transfer',str(b)+(' †' if b in (48,63,64) else '')) for b in targets])
    forecast = ('path_violation','exact_violation','last_position_violation','score_violation','score_bound_over_realized.median','fraction_forecast_stable','false_stability')
    pairs=[(m,r) for m in LOOPED for r in (4,8,16) if s.get('t1',s.SEEDS[0],'family_mean',f'{m}/{r}')]
    g.table('forecast',list(forecast),[(f'{m}/{r}',[('t1','family_mean',f'{m}/{r}',f,None) for f in forecast]) for m,r in pairs],True)
    rows=[]
    cf=('coverage','bound_over_realized.median','fraction_forecast_stable','false_stability','decision_changed')
    for m,r in pairs:
        e=s.get('t1',s.SEEDS[0],'family_mean',f'{m}/{r}')
        for kind in ('state_conformal','score_conformal'):
            for rule in e.get(kind,{}):
                rows.append((f'{m}/{r} {kind.removesuffix("_conformal")} {rule}',[('t1','family_mean',f'{m}/{r}',f'{kind}.{rule}.{f}',None) for f in cf]))
    g.table('conformal',list(cf),rows,True)
    g.standard('dynamics','m4',[x for m in (*LOOPED,'D') for x in [A(f'{m}/{m[1:] if m!="D" else "16"}/{k}') if k!='added' else D(f'{m}/{m[1:] if m!="D" else "16"}/{k}') for k in ('dynamics','confidence','joint','added')]])
    columns=[str(r) for r in (1,2,4,8,16,32,64) if any(s.sub(s.get('dyn',s.SEEDS[0],'family_mean',m),f'accuracy.{r}.accuracy') is not None for m in LOOPED)]+['two_cycle_fraction']
    g.table('dyn',columns,[(m,[('dyn','family_mean',m,f'accuracy.{r}.accuracy' if r!='two_cycle_fraction' else r,None) for r in columns]) for m in LOOPED])
    extend(g)
    return g


def extend(g):
    """E2: additions only; the original table definitions above are unchanged."""
    from collections import Counter
    s = g.s
    specs = [('m1a', k, 'auroc' if k.startswith('table/') else 'difference')
             for k in [*(f'table/{r}/{c}' for r in ('L32','D') for c in ('L32','D')), 'diagonal','row','column','interaction']]
    specs += [('m1b','best_difference','difference')]
    specs += [('m1c',f'fixed/{m}/{b}/added','difference') for m in ('L32','D') for b in ('confidence','flex')]
    specs += [('m1c',f'False/{b}/added_difference','difference') for b in (False,True)]
    specs += [('m4','L32/32/added','difference')] + [('m1s',f'{m}/permutation','auroc') for m in ('L32','D')]
    lines = ['#let tab_seeds = table(columns: 5, inset: 3pt, table.header([Quantity], [Seed 0], [Seed 1], [Seed 2], [Min–max]),']
    for a,k,f in specs:
        texts=[]; values=[]
        for seed in s.SEEDS:
            e=s.get(a,seed,'family_mean',k); v=e.get(f); iv=e.get('interval') or [None,None]
            texts.append(format_cell(v,*iv)); values.append(v)
            g.cells.append(dict(table='seeds',row_label=a+' '+k,column_label='seed '+seed,value=v,lower=iv[0],upper=iv[1],number_of_seeds=1,analysis=a,family='family_mean',key=k,field=f,interval_field='interval',results_json_key=f'analyses.{a}.{seed}.family_mean.estimates.{k}.{f}',formatted=texts[-1]))
        span=f'{number(min(values))}–{number(max(values))}'
        g.cells.append(dict(table='seeds',row_label=a+' '+k,column_label='min–max',value=None,minimum=min(values),maximum=max(values),results_json_key=f'analyses.{a}.{{seed}}.family_mean.estimates.{k}.{f}',formatted=span))
        lines.append('text('+quote(a+' '+k)+'), '+', '.join('text('+quote(t)+')' for t in texts+[span])+',')
    lines.append(')'); g.tables.append(dict(name='seeds',source=lines,wide=True))
    counts={}
    for m in ('L32','D'):
        counts[m]={f:Counter(site for seed in s.SEEDS for site,_ in s.get('m1b',seed,f,'best/'+m)['chosen']) for f in s.FAMS[:-1]}
    lines=['#let tab_sites = table(columns: 6, inset: 3pt, table.header([Model/site], '+', '.join('text('+quote(f)+')' for f in s.FAMS[:-1])+', [Total]),']
    for m in counts:
        for site in (['x32','post32'] if m=='L32' else [f'h{i}' for i in range(17)]):
            vals=[counts[m][f][site] for f in s.FAMS[:-1]]
            lines.append('text('+quote(m+'/'+site)+'), '+', '.join('text('+quote(str(v))+')' for v in vals+[sum(vals)])+',')
            for fam,v in zip([*s.FAMS[:-1],'total'],vals+[sum(vals)]):
                g.cells.append(dict(table='sites',row_label=m+'/'+site,column_label=fam,value=v,formatted=str(v),results_json_key=f'analyses.m1b.{{seed}}.{{family}}.estimates.best/{m}.chosen',derivation='count site choices over seeds and outer folds; total sums families'))
    lines.append(')'); g.tables.append(dict(name='sites',source=lines,wide=False))
    # Derive overall error from saved class counts, never from rounded accuracies.
    for a in ('m1a','m3a'):
        for seed in s.SEEDS:
            for fam in s.FAMS:
                for k,e in s.RES[seed][a][fam]['estimates'].items():
                    if 'classes' in e: e['overall_error']=e['classes'][1]/sum(e['classes'])
                if a=='m1a' and fam=='family_mean':
                    for m in ('L32','D'):
                        k=f'table/{m}/{m}'
                        s.get(a,seed,fam,k)['overall_error']=sum(s.get(a,seed,f,k)['overall_error'] for f in s.FAMS[:-1])/4
                if a=='m3a':
                    for k,e in s.RES[seed][a][fam]['estimates'].items():
                        if '->' in k and k.count('/')==1:
                            model, budgets=k.split('/'); target=budgets.split('->')[1]
                            native=s.get(a,seed,fam,model+'/native'+target)
                            if native and 'overall_error' in native: e['overall_error']=native['overall_error']
    for seed in s.SEEDS:
        for k,e in s.RES[seed]['m3a']['family_mean']['estimates'].items():
            vals=[s.get('m3a',seed,f,k) for f in s.FAMS[:-1]]
            if all(v and 'overall_error' in v for v in vals): e['overall_error']=sum(v['overall_error'] for v in vals)/4
    g.standard('coverage','m1a',[(m+' '+f,f'table/{m}/{m}',f,None) for m in ('L32','D') for f in ('overall_error','accepted_error')])
    rows=[]
    fields=('fraction_rejected','accepted_error','overall_error','label_changed','native − transfer')
    for m,src in [('L32','32'),('L4','4'),('L8','8'),('L16','16')]:
        keys=[k for k in s.RES[s.SEEDS[0]]['m3a']['family_mean']['estimates'] if k.startswith(f'{m}/{src}->') and k.split('->')[1].isdigit()]
        for fam in (s.FAMS if m=='L32' else ['family_mean']):
            for k in sorted(keys,key=lambda k:int(k.split('->')[1])):
                rows.append((fam+' '+k,[('m3a',fam,k+'/native_minus_transfer','difference','interval') if f=='native − transfer' else ('m3a',fam,k,f,None) for f in fields]))
    g.table('threshold',fields,rows,True)
    fields=('coverage','coverage_with_allowance','bound_over_realized.median','fraction_forecast_stable','fraction_forecast_stable_with_allowance','false_stability')
    # Metric rows keep the two SCORE candidates side by side without an unreadably wide table.
    rows=[]
    for m in LOOPED:
        for r in (4,8,16):
            k=f'{m}/{r}'
            for f in ('raw forecast_fraction','forecast_fraction',*fields,'allowance_over_movement.median','rounding_subset.relative_movement_change.median'):
                specs=[]
                for rule in ('scaled_rule','constant'):
                    field='forecast_fraction' if f=='raw forecast_fraction' else (f'score_conformal.{rule}.{f}' if f in fields or f=='forecast_fraction' else f)
                    specs.append(('t1','family_mean',k,field,None))
                rows.append((k+' '+f,specs))
    g.table('forecast_main',('SCORE scaled_rule','SCORE constant'),rows,True)
    for c in g.cells:
        if c.get('field')=='overall_error':
            c['results_json_key']=c['results_json_key'].replace('.overall_error','.classes')
            c['derivation']='classes[1]/sum(classes), mean over families; for transfer use target native budget classes'
            if c['analysis']=='m3a':
                model,budgets=c['key'].split('/')
                c['results_json_key']=f'analyses.m3a.{{seed}}.{c["family"]}.estimates.{model}/native{budgets.split("->")[1]}.classes'
            if c['family']=='family_mean':
                c['results_json_key']=[c['results_json_key'].replace('.family_mean.', '.'+f+'.') for f in s.FAMS[:-1]]


def write(g, out):
    out.mkdir(parents=True,exist_ok=True)
    (out/'tables.typ').write_text('// Generated; median over seeds, envelope of seed 95% intervals.\n// Accuracy = fraction correct. † = deep subset. [n/a] = unavailable interval; – = missing.\n'+'\n\n'.join('\n'.join(t['source']) for t in g.tables)+'\n')
    (out/'tables.json').write_text(json.dumps(g.cells,indent=2,ensure_ascii=False)+'\n')
    headline=[c for c in g.cells if (c['table'] in ('table','best','added','added_best','coverage') and c['column_label']=='family_mean') or (c['table']=='seeds' and c['column_label'].startswith('seed ')) or (c['table']=='threshold' and c['family']=='family_mean' and c['key'].startswith('L32/') and c['field'] in ('fraction_rejected','accepted_error'))]
    lines=['// Headline strings: num_<table>_<key with punctuation replaced by underscores>.', '// Names and full results.json paths are listed below.']
    for c in headline:
        name='num_'+c['table']+'_'+re.sub('[^a-zA-Z0-9]+','_',c['key']).strip('_')
        if c['table']=='seeds': name += '_'+c['column_label'].replace(' ','_')
        if c['table'] in ('coverage','threshold'): name += '_'+c['field']
        lines += ['// '+name+': '+str(c['results_json_key']), '#let '+name+' = '+quote(c['formatted'])]
    for c in g.cells:
        if c['table'] in ('table','best') and c.get('key') in ('diagonal','best_difference') and c['column_label']=='family_mean':
            for end in ('lower','upper'):
                lines.append('#let num_'+c['table']+'_'+c['key']+'_'+end+'_6 = '+quote(f'{c[end]:.6f}'))
    (out/'numbers.typ').write_text('\n'.join(lines)+'\n')
    preview=['#import "tables.typ": *','#import "numbers.typ": *','#set text(size: 8pt)']
    for i,t in enumerate(g.tables):
        if i: preview.append('#pagebreak()')
        preview += ['#set page(paper: "a4", flipped: '+str(t['wide']).lower()+', margin: 10mm)', '= tab_'+t['name']]
        if t['name'] in ('budget','budget32','seeds','threshold','forecast_main'):
            preview += ['#text(size: 5pt)[#tab_'+t['name']+']']
        else: preview += ['#tab_'+t['name']]
    if (out/'explore'/'explore.typ').exists():
        preview.append('#import \"explore/explore.typ\": *')
        for name in ('joint','scorecorr','stratified','deep'):
            preview += ['#pagebreak()', '#set page(paper: \"a4\", flipped: true, margin: 10mm)', '= tab_'+name]
            if name=='stratified': preview.append('Conditioning on privileged correctness of the other model; descriptive only. Subgroup AUROCs pool items.')
            if name=='deep': preview.append('Same 256 items at source and target. AUROCs average frozen folds. Δ = target minus source. Class-stratified intervals deviate from the registered rule.')
            preview.append('#text(size: 7pt)[#tab_'+name+']')
        preview += ['Privileged correctness conditioning is descriptive only. Deep Δ = target minus source on the same 256 items; AUROC averages frozen folds. Stratified intervals deviate from the registered rule; see explore/explore.json.']
    (out/'preview.typ').write_text('\n'.join(preview)+'\n')
    missing=sorted({c['table'] + ': ' + c['row_label'] + ' / ' + c['field'] for c in g.cells if c['value'] is None and c['table']!='seeds'})
    (out/'README.md').write_text('Regenerate: `.venv/bin/python -I huginn_addition/tables_mon.py --inputs results/mon_analysis-20261110/seed{0,1,2} --out docs/monitor_results_gen`\n\nGenerated with tables_mon.py. All means are unweighted family means within seed, then medians across seeds.\n\nMissing quantities (shown as –; state calibration does not define score decisions; some raw false_stability aggregates are null):\n'+ '\n'.join('- '+f for f in missing)+'\n\nBudget tables use landscape A4 at 5pt; forecast/conformal use landscape at 8pt. New seeds/threshold/forecast_main previews use landscape 5pt; E3 tables use landscape 7pt. Other previews use portrait 8pt. Tables inherit the including document text size.\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    write(build(load_summary(args.inputs)),args.out)


if __name__=='__main__':
    main()
