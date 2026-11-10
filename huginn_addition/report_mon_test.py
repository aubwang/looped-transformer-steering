#!/usr/bin/env python3
"""Structural and fault-injection tests; never interprets smoke estimates."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'huginn_addition/report_mon.py'
spec=importlib.util.spec_from_file_location('report_mon',SCRIPT)
report=importlib.util.module_from_spec(spec); spec.loader.exec_module(report)


def main():
    smoke=json.loads((ROOT/'results/mon_analysis-smoke-20261110/results.json').read_text())
    with tempfile.TemporaryDirectory(prefix='report-mon-test-') as tmp:
        tmp=Path(tmp)
        copies=[]
        for seed in ('0','1','2'):
            doc=copy.deepcopy(smoke)
            for name,rows in doc['analyses'].items():
                assert len(rows)==1
                doc['analyses'][name]={seed:next(iter(rows.values()))}
            copies.append(doc)
        def run(documents, dest, ok=True):
            inputs=[]
            for i,doc in enumerate(documents):
                folder=tmp/dest/str(i); folder.mkdir(parents=True,exist_ok=True)
                path=folder/'results.json';path.write_text(json.dumps(doc));inputs.append(str(folder) if i%2 else str(path))
            out=tmp/dest/'out'
            proc=subprocess.run([sys.executable,'-I',str(SCRIPT),'--inputs',*inputs,'--out',str(out)],capture_output=True,text=True,cwd=ROOT)
            assert (proc.returncode==0)==ok,proc.stderr
            if not ok: return proc
            text=(out/'report.md').read_text(); head=json.loads((out/'headline.json').read_text(),parse_constant=lambda s: (_ for _ in ()).throw(AssertionError(s)))
            assert text and head['quantities']
            return text,head
        text,head=run(copies,'normal')
        appendix=text.split('## Appendix:')[1]
        for doc in copies:
            for name,ss in doc['analyses'].items():
                for seed,rows in ss.items():
                    block=appendix.split(f'### {name} / seed {seed}\n')[1].split('```text\n')[1].split('```')[0]
                    for path,value in report.leaves(rows['family_mean']['estimates'],numeric_only=True):
                        assert f'{report.label(path)} = {json.dumps(value)}\n' in block,(name,seed,path)
        assert 'SMOKE CRASH CHECK' in text
        assert head['fields_not_interpreted']==[]
        bad=copy.deepcopy(copies)
        family=bad[0]['settings']['fams'][0]
        estimates=bad[0]['analyses']['m1a']['0'][family]['estimates']
        cells=[k for k in estimates if k.startswith('table/')]
        estimates[cells[0]]['interval']=[None,None]
        estimates[cells[0]]['interval_unavailable_reason']='injected undefined interval'
        estimates[cells[1]]['nonconverged']=2
        estimates.pop(cells[2])
        bad[0]['analyses']['m1a']['0'][family]['skipped'].append({'name':cells[2],'reason':'injected skip'})
        del bad[2]['analyses']['m1b']['2'][family]
        bad[0]['analyses']['m1a']['0']['family_mean']['estimates']['incomplete'][cells[0]+'/auroc']=1
        text,head=run(bad,'caveats')
        audit=text.split('## Audit of caveats')[1].split('## M1-a')[0]
        for token in ('injected undefined interval','injected skip','nonconverged','missing seed-family row','incomplete'):
            assert token in audit,token
        def find(name,key,fam,seed):
            return next(q for q in head['quantities'] if q['analysis']==name and q['key']==key)['families'][fam]['per_seed'][seed]
        assert find('m1a',cells[0]+'/auroc',family,'0')['flags']['unavailable_interval']
        assert find('m1a',cells[1]+'/auroc',family,'0')['flags']['nonconverged']
        assert find('m1a',cells[2]+'/auroc',family,'0')['flags']['skipped']
        assert find('m1a',cells[0]+'/auroc','family_mean','0')['flags']['incomplete']
        best=next(q for q in head['quantities'] if q['analysis']=='m1b' and q['path'][-1]=='auroc')
        assert best['families'][family]['per_seed']['2']['flags']['skipped']
        assert best['families'][family]['across_seeds']['missing_seeds']==['2']
        assert 'unavailable seeds: 2' in text
        mismatch=copy.deepcopy(copies)
        mismatch[1]['provenance']['items_sha256']='mismatch'
        assert 'mismatched provenance' in run(mismatch,'mismatch',False).stderr
        mismatch=copy.deepcopy(copies)
        mismatch[1]['provenance']['manifest_sha256']='mismatch'
        assert 'mismatched provenance' in run(mismatch,'manifest-mismatch',False).stderr
        mismatch=copy.deepcopy(copies)
        mismatch[1]['settings']['outer']+=1
        assert 'fold/bootstrap' in run(mismatch,'fold-mismatch',False).stderr
        assert 'duplicated seed' in run([copies[0],copies[0]],'duplicate',False).stderr
        # One file containing all seeds is supported too.
        combined=copy.deepcopy(copies[0])
        for d in copies[1:]:
            for name,rows in d['analyses'].items(): combined['analyses'][name].update(rows)
        run([combined],'combined')
        unknown=copy.deepcopy(copies[0])
        fam=unknown['settings']['fams'][0]
        cell=next(iter(unknown['analyses']['m1a']['0'][fam]['estimates']))
        unknown['analyses']['m1a']['0'][fam]['estimates'][cell]['unfamiliar_text']='opaque field'
        txt,h=run([unknown],'unknown')
        assert any('unfamiliar_text' in p for p in h['fields_not_interpreted'])
        assert 'opaque field' in txt.split('## Fields not interpreted')[1]
    print('ALL CHECKS PASSED')


if __name__=='__main__': main()
