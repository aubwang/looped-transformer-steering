"""Fresh prompts for the settling-speed experiment: 60 problems (20 each of 2-, 3-, 4-digit operands, uniform), one
randomly drawn few-shot demo block each (same structure as the census), excluding every (a, b) pair used before
(fit/held-out sets seeds 0/1/2 and all census problems). No selection on correctness, uncertainty or motion.
Usage: python -I huginn_addition/make_settle_prompts.py results/settle_prompts.json
"""
import json
import os
import random
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import make_problem, sample_stratified  # noqa: E402
from make_demo_variants import demo_line  # noqa: E402


def main(out):
    used = set()
    for seed, pc in ((0, 6), (1, 4), (2, 4)):
        used |= {(p.a, p.b) for p in sample_stratified(n_digits_list=(2, 3, 4), per_cell=pc, seed=seed)}
    used |= {(p["a"], p["b"]) for p in json.load(open("results/census_prompts_all.json"))}
    rng = random.Random(20261106)
    nd = [2] * 20 + [3] * 20 + [4] * 20
    rng.shuffle(nd)
    res = []
    for i, n in enumerate(nd):
        lo, hi = 10 ** (n - 1), 10 ** n - 1
        while True:
            a, b = rng.randint(lo, hi), rng.randint(lo, hi)
            if (a, b) not in used:
                used.add((a, b)); break
        demo = "".join(demo_line(rng, k) for k in (2, 3, 4))
        q = asdict(make_problem(5000 + i, 50000 + i, "settle", "A", a, b))
        q.update(demo="S", prompt=f"{demo}{a} + {b} =", ipos=1)
        res.append(q)
    json.dump(res, open(out, "w"), indent=1)
    print(len(res), "prompts; answer positions", sum(len(p["digit_info"]) for p in res))


if __name__ == "__main__":
    main(sys.argv[1])
