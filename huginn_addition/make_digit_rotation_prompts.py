"""Digit-circle rotation pilot prompts: units digits a, b in 0..9 (full 10 x 10 factorial) x 2 contexts = 200 prompts.
Context A: demo set A, operands 2a + 5b (i.e. 20+a, 50+b); context B: demo set B, operands 4a + 3b. Sums <= 88, so no
overflow: the answer is " T" + "U" with tens digit T = 7 or 8 (T - 7 = carry from the units column) and units digit
U = (a + b) mod 10. The state of interest is at the answer position predicting U (the tens-digit answer token).
Usage: python -I huginn_addition/make_digit_rotation_prompts.py results/digit_rotation_prompts.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import make_problem  # noqa: E402

CTX = {"A": (20, 50), "B": (40, 30)}


def main(out):
    rows = []
    for ci, (demo, (ta, tb)) in enumerate(CTX.items()):
        for a in range(10):
            for b in range(10):
                uid = 40000 + 100 * ci + 10 * a + b
                p = make_problem(pid=uid, uid=uid, family="digit_rotation", demo=demo, a=ta + a, b=tb + b).__dict__
                assert p["c"] < 100 and p["c"] % 10 == (a + b) % 10
                p.update(template=demo, start=a, offset=b, ans=(a + b) % 10, carry=int(a + b >= 10), ipos=1)
                rows.append(p)
    json.dump(rows, open(out, "w"), indent=1)
    print(len(rows), "prompts;", rows[0]["prompt"] + rows[0]["answer"])


if __name__ == "__main__":
    main(sys.argv[1])
