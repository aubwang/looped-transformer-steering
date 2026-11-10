"""Integration checks, using raw JSON independently of summarize_mon."""
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

ROOT = Path.cwd()
PYTHON = '/home/owner/work/.t3/projects/analysis/.venv/bin/python'
TYPST_PYTHON = '/tmp/claude-1000/-home-owner-work--t3-worktrees-analysis-t3-5b527f9a/d3850ad3-cf62-4473-a445-987ff1bdf740/scratchpad/typstenv/bin/python'
INPUTS = [ROOT / f'results/mon_analysis-20261110/seed{i}' for i in range(3)]


def main():
    raw = [json.loads((p/'results.json').read_text()) for p in INPUTS]
    with tempfile.TemporaryDirectory() as temp:
        out = Path(temp)
        subprocess.run([PYTHON, '-I', 'huginn_addition/tables_mon.py', '--inputs', *map(str,INPUTS), '--out', str(out)], check=True)
        cells = json.loads((out/'tables.json').read_text())
        checked = 0
        for c in cells:
            if c['table'] not in ('table','best','added','added_best'):
                continue
            values, intervals = [], []
            for i, r in enumerate(raw):
                e = r['analyses'][c['analysis']][str(i)][c['family']]['estimates'][c['key']]
                values.append(e[c['field']])
                intervals.append(e[c['interval_field']])
            assert c['value'] == statistics.median(values), c
            if all(iv is not None and all(x is not None for x in iv) for iv in intervals):
                assert c['lower'] == min(iv[0] for iv in intervals), c
                assert c['upper'] == max(iv[1] for iv in intervals), c
            else:
                assert c['lower'] is None and c['upper'] is None
                assert c['formatted'].endswith('[n/a]')
            assert c['number_of_seeds'] == 3
            checked += 1
        assert checked >= 200
        def find(table,row,col):
            return next(c for c in cells if c['table']==table and c['row_label']==row and c['column_label']==col)
        for m,expected in [('L32',{'x32':46,'post32':14}),('D',{'h7':1,'h12':10,'h13':21,'h14':14,'h15':11,'h16':3})]:
            for site,n in expected.items(): assert find('sites',m+'/'+site,'total')['value']==n
        for c in cells:
            if c['table']=='seeds' and c['column_label'].startswith('seed '):
                i=c['column_label'].split()[-1]
                e=raw[int(i)]['analyses'][c['analysis']][i]['family_mean']['estimates'][c['key']]
                assert c['value']==e[c['field']] and [c['lower'],c['upper']]==e['interval']
            if c['table'] in ('coverage','threshold','forecast_main'):
                values=[]
                for i,r in enumerate(raw):
                    e=r['analyses'][c['analysis']][str(i)][c['family']]['estimates'][c['key']]
                    if c['field']=='overall_error':
                        k=c['key']
                        if c['analysis']=='m3a': k=k.split('/')[0]+'/native'+k.split('->')[1]
                        fs=raw[i]['settings']['fams'] if c['family']=='family_mean' else [c['family']]
                        es=[r['analyses'][c['analysis']][str(i)][f]['estimates'][k] for f in fs]
                        values.append(sum(e['classes'][1]/sum(e['classes']) for e in es)/len(es))
                    else:
                        for part in c['field'].split('.'): e=e.get(part) if isinstance(e,dict) else None
                        values.append(e)
                assert c['value']==(None if None in values else statistics.median(values)),c
        for f,err,base in zip(('add3','add4','add3w','arc_easy'),(.162,.525,.428,.302),(.208,.547,.487,.364)):
            assert round(find('coverage','L32 accepted_error',f)['value'],3)==err
            assert round(find('coverage','L32 overall_error',f)['value'],3)==base
        for f,err,base in zip(('add3','add4','add3w','arc_easy'),(.206,.553,.460,.293),(.257,.589,.513,.346)):
            assert abs(find('coverage','D accepted_error',f)['value']-err)<=.0005000001
            assert abs(find('coverage','D overall_error',f)['value']-base)<=.0005000001
        for b,v in [(1,.120),(2,.311),(4,.196),(8,.207),(32,.200)]:
            c=next(c for c in cells if c['table']=='threshold' and c['row_label']==f'family_mean L32/32->{b}' and c['field']=='fraction_rejected')
            assert abs(c['value']-v)<=0.0005000001
        assert find('seeds','m1s D/permutation','seed 0')['formatted']=='0.472 [0.458, 0.487]'

        spec = importlib.util.spec_from_file_location('tables_mon',ROOT/'huginn_addition/tables_mon.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        expanded=module.build(module.load_summary(INPUTS))
        original_extend=module.extend
        module.extend=lambda g: None
        baseline=module.build(module.load_summary(INPUTS))
        module.extend=original_extend
        assert expanded.tables[:len(baseline.tables)]==baseline.tables
        assert expanded.cells[:len(baseline.cells)]==baseline.cells
        assert module.format_cell(0.5, interval=True) == '0.500 [n/a]'
        assert module.format_cell(-0.125, -0.2, -0.01) == '−0.125 [−0.200, −0.010]'
        assert module.format_cell(None) == '–'
        assert any(c['formatted'].endswith('[n/a]') for c in cells)
        # Actual synthetic null interval through the existing aggregation function.
        s = module.load_summary(INPUTS)
        for seed in s.SEEDS:
            s.RES[seed]['m1b']['add3']['estimates']['best/L32']['interval'] = None
        g = module.Generator(s)
        g.standard('null_check','m1b',[('null interval','best/L32','auroc','interval')])
        assert all(c['formatted'].endswith('[n/a]') for c in g.cells if c['family']=='add3')
        # Tiny import/placement document, one placement of every exported table.
        names = list(dict.fromkeys(c['table'] for c in cells))
        tiny = '#import "tables.typ": *\n#import "numbers.typ": *\n#set page(paper: "a3", flipped: true)\n#set text(size: 6pt)\n'
        tiny += '\n#pagebreak()\n'.join('#tab_'+n for n in names)
        (out/'check.typ').write_text(tiny)
        code = 'import typst; typst.compile('+repr(str(out/'check.typ'))+', output='+repr(str(out/'check.pdf'))+', root='+repr(str(out))+')'
        subprocess.run([TYPST_PYTHON,'-c',code],check=True)
        assert (out/'check.pdf').read_bytes().startswith(b'%PDF')
    print(f'Independent raw-JSON cells checked: {checked}')
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
