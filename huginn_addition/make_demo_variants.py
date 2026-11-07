"""Demo-set variants for selected problems (same few-shot structure as demo sets A/B: one 2-digit, one 3-digit and one
4-digit example, each with a same-length sum), so all variants of a problem are token-aligned with A/B.
Usage: python -I huginn_addition/make_demo_variants.py <out.json> <n_variants> <pid,pid,...> [tokenizer.json]
"""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(__file__))
from data import DEMO_SETS, make_problem, sample_stratified  # noqa: E402


def demo_line(rng, n):
    lo, hi = 10 ** (n - 1), 10 ** n - 1
    while True:
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if a + b <= hi:
            return f"{a} + {b} = {a + b}\n"


def main(out, nv, pids, tok=None):
    rng = random.Random(123)
    base = {p.pid: p for p in sample_stratified(n_digits_list=(2, 3, 4), per_cell=4, seed=1) if p.demo == "A"}
    fit = {(p.a, p.b) for p in sample_stratified(n_digits_list=(2, 3, 4), per_cell=6, seed=0)}
    res, uid = [], 10000
    tk = None
    if tok:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(tok)
    demos = [("A", DEMO_SETS["A"]), ("B", DEMO_SETS["B"])] + [(f"V{v}", "".join(demo_line(rng, n) for n in (2, 3, 4))) for v in range(nv)]
    for pid in pids:
        p0 = base[pid]
        assert (p0.a, p0.b) not in fit
        lens = set()
        for name, dtext in demos:
            q = asdict(make_problem(pid, uid, p0.family, "A", p0.a, p0.b))
            q["demo"], q["prompt"], q["uid"] = name, f"{dtext}{p0.a} + {p0.b} =", uid
            uid += 1
            if tk:
                lens.add(len(tk.encode(q["prompt"] + q["answer"]).ids))
            res.append(q)
        if tk:
            assert len(lens) == 1, (pid, lens)
    json.dump(res, open(out, "w"), indent=1)
    print(len(res), "prompts;", "token-aligned per problem" if tk else "")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")], sys.argv[4] if len(sys.argv) > 4 else None)
