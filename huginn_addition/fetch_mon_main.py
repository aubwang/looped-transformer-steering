"""Fetch completed mon_main launches, sequentially and safely."""
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

MODAL = '/home/owner/work/.t3/projects/analysis/.venv/bin/modal'
started = time.monotonic()
retries = 0


def call(*args):
    global retries
    for attempt in range(5):
        try:
            return subprocess.check_output([MODAL, 'volume', *args], stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            if attempt == 4:
                raise
            retries += 1
            time.sleep(5 * (attempt + 1))


def listing(path):
    return json.loads(call('ls', 'huginn-results', path, '--json'))


def fetch(entry, dest):
    global retries
    if dest.is_file() and dest.stat().st_size == entry['size'] > 0:
        return
    temp = dest.with_name(dest.name + '.download')
    try:
        for attempt in range(5):
            try:
                subprocess.run([MODAL, 'volume', 'get', '--force', 'huginn-results',
                                entry['filename'], str(temp)], check=True, capture_output=True)
                if temp.stat().st_size != entry['size'] or not temp.stat().st_size:
                    raise ValueError('download size mismatch: ' + entry['filename'])
                temp.replace(dest)
                return
            except (subprocess.CalledProcessError, OSError, ValueError):
                if attempt == 4:
                    raise
                retries += 1
                time.sleep(5 * (attempt + 1))
    finally:
        temp.unlink(missing_ok=True)


count = total = problems = 0
for run in sorted(Path('results').glob('mon_main-*')):
    if not (run / 'containers.json').is_file():
        continue
    try:
        shards = [(m, s) for m in listing(run.name) if m['type'] == 'dir'
                  for s in listing(m['filename']) if s['type'] == 'dir']
    except Exception as exc:
        print(f'{run.name}: {exc}', flush=True)
        problems += 1
        continue
    for model, shard in shards:
        dest = Path('results') / shard['filename']
        status, npzs, size = '?', [], 0
        try:
            entries = listing(shard['filename'])
            dest.mkdir(parents=True, exist_ok=True)
            for entry in entries:
                fetch(entry, dest / Path(entry['filename']).name)
            files = list(dest.iterdir())
            size = sum(p.stat().st_size for p in files)
            npzs = sorted(dest.glob('*.npz'))
            meta = json.loads((dest / 'meta.json').read_text())
            status = meta['status']
            assert all(p.is_file() and p.stat().st_size > 0 for p in files), 'empty/non-file entry'
            assert len(files) == len(entries), 'file count mismatch'
            if status == 'partial':
                assert all((dest / n).is_file() for n in meta['chunks']), 'missing chunk'
            for path in npzs:
                with np.load(path, allow_pickle=False) as data:
                    data['pid']
            result = 'OK'
        except Exception as exc:
            result = f'ERROR: {exc}'
            problems += 1
        count += 1
        total += size
        print(f"{run.name} {Path(model['filename']).name} {dest.name} {status} {len(npzs)} npz {size / 1e6:.2f} MB {result}", flush=True)
if not problems:
    print(f'ALL SHARDS OK ({count} shards, {total / 1e6:.2f} MB)', flush=True)
print(f'Elapsed: {time.monotonic() - started:.1f}s; retries: {retries}', flush=True)
sys.exit(bool(problems))
