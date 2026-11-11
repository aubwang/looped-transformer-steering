"""Deterministic, unselected final sums for the fixed flicker2 screen."""
import json
import random
from dataclasses import asdict
from pathlib import Path
from data import make_problem

ROOT = Path(__file__).resolve().parent.parent
CONTEXTS = (20059, 20145, 20144, 20178)


def build(root=ROOT):
    root = Path(root)
    old = json.loads((root/'results/flicker_prompts.json').read_text())
    contexts = {p['uid']: p for p in old}
    used = {(p['a'], p['b']) for name in ('census_prompts_all.json', 'screen_prompts.json')
            for p in json.loads((root/'results'/name).read_text())}
    out = []
    for context in CONTEXTS:
        text = contexts[context]['prompt']
        demo = text[:text.rfind('\n')+1]
        rng = random.Random(20261111 + context)
        for n in [3]*50 + [4]*50:
            while True:
                a, b = rng.randint(10**(n-1), 10**n-1), rng.randint(10**(n-1), 10**n-1)
                if (a, b) not in used:
                    used.add((a, b)); break
            i = len(out)
            p = asdict(make_problem(3000+i, 30000+i, 'flicker2', 'A', a, b))
            p.update(demo=contexts[context]['demo'], ipos=1, source='flicker2', context_uid=context,
                     prompt=demo+f'{a} + {b} =')
            out.append(p)
    assert len(out) == len({p['uid'] for p in out}) == 400
    assert len({(p['context_uid'], p['a'], p['b']) for p in out}) == 400
    return out


def main():
    out = build()
    path = ROOT/'results/flicker2_screen_prompts.json'
    path.write_text(json.dumps(out, indent=2)+'\n')
    print(f'Wrote {len(out)} prompts to {path}')


if __name__ == '__main__': main()
