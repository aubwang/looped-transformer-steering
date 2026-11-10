"""Small end-to-end regression check; no experimental computation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

spec = importlib.util.spec_from_file_location('plot_mon', Path(__file__).with_name('plot_mon.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    inputs = []
    for seed in range(2):
        analyses = {k: {str(seed): {}} for k in ['m1a', 'm1c', 'm1s', 'm3a', 'dyn']}
        for family in ['add3', 'add4', 'add3w', 'arc_easy', 'family_mean']:
            es = {section: {} for section in analyses}
            def estimate():
                return dict(auroc=.65 + seed*.02, interval=[.60-seed*.01, .72+seed*.01])
            for r in ['L32', 'D']:
                for c in ['L32', 'D']:
                    es['m1a'][f'table/{r}/{c}'] = estimate()
                for t in ['confidence', 'probe', 'confidence+probe']:
                    es['m1c'][f'fixed/{r}/{t}'] = estimate()
                for t in ['output_identity', 'privileged_ground_truth', 'early']:
                    es['m1c'][f'{r}/{t}'] = estimate()
            # Incomplete interval must suppress the entire aggregate error bar.
            es['m1a']['table/L32/D']['interval'] = None if seed == 0 else [None, None]
            es['m1c']['fixed/D/probe']['interval'] = None if seed == 0 else [.6, .75]
            for m in p.MODELS[:-1]:
                for t in ['probe', 'confidence']:
                    es['m1s'][f'{m}/{m[1:]}/{t}'] = estimate()
                es['dyn'][m] = {'accuracy': {str(r): {'accuracy': .7+seed*.01} for r in [1,4,64]}}
            for m, src in [('L32',32), ('L4',4)]:
                for r in [1,4,64]:
                    es['m3a'][f'{m}/native{r}'] = estimate()
                    es['m3a'][f'{m}/{src}->{r}'] = dict(transfer_auroc=.66+seed*.01, transfer_interval=[.6,.75])
            for section in es:
                analyses[section][str(seed)][family] = {'estimates': es[section]}
        d = root / f'seed{seed}'
        d.mkdir()
        (d/'results.json').write_text(json.dumps({'settings': {'fams': ['add3','add4','add3w','arc_easy']}, 'analyses': analyses}))
        inputs.append(str(d))
    out = root/'figs'
    subprocess.run([sys.executable, '-I', str(Path(p.__file__)), '--inputs', *inputs, '--out', str(out)], check=True)
    for name in ['fig_table','fig_confidence','fig_loopcounts','fig_budget','fig_budget_L4','fig_accuracy']:
        for ext in ['pdf', 'png']:
            assert (out/f'{name}.{ext}').stat().st_size > 100
    data = json.loads((out/'figures.json').read_text())
    summary = p.load_summary(inputs)
    checked = 0
    for row in data['points']:
        if row['figure'] not in ['fig_table','fig_confidence']:
            continue
        text = summary.cell(row['section'], row['panel'], row['key'], row['field'], row['interval'], digits=12)
        assert text.split()[0].rstrip('?!') == f"{row['value']:.12f}", (text, row)
        if row['lower'] is not None:
            assert f"[{row['lower']:.12f},{row['upper']:.12f}]" in text
        else:
            assert '?' in text and row['upper'] is None
        assert row['n_seeds'] == 2
        checked += 1
    assert checked == 80, checked
    assert any(row['lower'] is None for row in data['points'] if row['figure'] == 'fig_table')
print('ALL CHECKS PASSED')
