"""Controlled carry-chain prompts (procedure experiment, stage 1 behaviour).
7-digit + 7-digit addition, no overflow (top column sum <= 8). Target = answer digit of column 5 (answer index 1), read
at the teacher-forced position after the column-6 answer digit. Column 5 itself is K (sum <= 8) or G (sum >= 10), so
its carry-OUT does not depend on its carry-in (the teacher-forced column-6 digit cannot leak the carry-in). Below it,
columns 4 .. 5-k are P (sum = 9) for chain length k in 0..4, and column 4-k (the decider) is G or K (balanced), so the
carry into column 5 = [decider is G], reachable only by tracing k propagate columns. Lower columns random. 30 prompts per
k (decider G/K 15/15; column-5 type alternating K/G; demo set A/B alternating). dep_len(col 5) = k + 1 (data.py).
Usage: python -I huginn_addition/make_carry_chain_prompts.py results/carry_chain_prompts.json
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import DEMO_SETS, make_problem  # noqa: E402


def pair(rng, kind):
    while True:
        x, y = rng.randint(0, 9), rng.randint(0, 9)
        s = x + y
        if (kind == "P" and s == 9) or (kind == "G" and s >= 10) or (kind == "K" and s <= 8) or kind == "R":
            return x, y


def main(out):
    rng = random.Random(20261107)
    rows, uid = [], 80000
    for k in range(5):
        for i in range(30):
            decider = "G" if i % 2 == 0 else "K"
            c5 = "K" if (i // 2) % 2 == 0 else "G"
            cols = {}
            while True:
                x6, y6 = rng.randint(1, 8), rng.randint(1, 8)
                if x6 + y6 <= 8:
                    break
            cols[6] = (x6, y6)
            cols[5] = pair(rng, c5)
            for c in range(4, 4 - k, -1):
                cols[c] = pair(rng, "P")
            cols[4 - k] = pair(rng, decider)
            for c in range(3 - k, -1, -1):
                cols[c] = pair(rng, "R")
            a = sum(cols[c][0] * 10 ** c for c in range(7)); b = sum(cols[c][1] * 10 ** c for c in range(7))
            demo = "A" if i % 4 < 2 else "B"
            p = make_problem(uid, uid, "carry_chain", demo, a, b).__dict__
            assert p["c"] < 10 ** 7 and len(p["digit_info"]) == 7
            di = p["digit_info"][1]
            assert di["col"] == 5 and di["carry_in"] == (decider == "G") and di["dep_len"] == k + 1, (k, di)
            ans = str(p["c"])
            prompt = p["prompt"] + " " + ans[0]  # teacher-forced column-6 answer digit; next token = column-5 digit
            s5 = cols[5][0] + cols[5][1]
            rows.append(dict(uid=uid, k=k, decider=decider, col5_type=c5, a=a, b=b, c=p["c"], demo=demo, prompt=prompt,
                             gold=int(ans[1]), digit_no_carry=s5 % 10, digit_with_carry=(s5 + 1) % 10, carry_in=int(decider == "G")))
            uid += 1
    json.dump(rows, open(out, "w"), indent=1)
    r = rows[100]
    print(len(rows), "prompts;", repr(r["prompt"][-40:]), "-> gold", r["gold"], "k", r["k"], r["decider"])


if __name__ == "__main__":
    main(sys.argv[1])
