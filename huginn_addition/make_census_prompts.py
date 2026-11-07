"""Fresh, unselected addition prompts for the prevalence + convergence-mode census (declared sampling distribution).
100 problems: n_digits in {2, 3, 4} (34/33/33), operands uniform over n-digit numbers, excluding every (a, b) pair used
in earlier runs (fit set seed 0 per_cell 6; held-out sets seed 1/2 per_cell 4). Each problem gets TWO independently drawn
few-shot demo sets with the same structure as demo sets A/B (one 2-, 3-, 4-digit example, each with a same-length sum).
No selection on uncertainty, correctness or motion. Split into shards for < 10 GPU-min runs.
Usage: python -I huginn_addition/make_census_prompts.py results/census_prompts n_shards
"""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(__file__))
from data import make_problem, sample_stratified  # noqa: E402
from make_demo_variants import demo_line  # noqa: E402


def main(out_prefix, n_shards):
    used = set()
    for seed, pc in ((0, 6), (1, 4), (2, 4)):
        used |= {(p.a, p.b) for p in sample_stratified(n_digits_list=(2, 3, 4), per_cell=pc, seed=seed)}
    rng = random.Random(20261105)
    nd_list = [2] * 34 + [3] * 33 + [4] * 33
    rng.shuffle(nd_list)
    res, uid = [], 20000
    for pid, n in enumerate(nd_list, start=1000):
        lo, hi = 10 ** (n - 1), 10 ** n - 1
        while True:
            a, b = rng.randint(lo, hi), rng.randint(lo, hi)
            if (a, b) not in used:
                used.add((a, b)); break
        for dv in ("D1", "D2"):
            demo = "".join(demo_line(rng, k) for k in (2, 3, 4))
            q = asdict(make_problem(pid, uid, "census", "A", a, b))
            q.update(demo=dv, prompt=f"{demo}{a} + {b} =", uid=uid, ipos=1)
            res.append(q); uid += 1
    # shards keep both demo sets of a problem together
    pids = sorted({q["pid"] for q in res})
    for s in range(n_shards):
        keep = set(pids[s::n_shards])
        sh = [q for q in res if q["pid"] in keep]
        json.dump(sh, open(f"{out_prefix}_{s}.json", "w"), indent=1)
        print(f"shard {s}: {len(sh)} prompts")
    json.dump(res, open(f"{out_prefix}_all.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
