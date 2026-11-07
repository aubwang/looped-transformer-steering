"""Weekday pilot prompts: 7 start days x 7 offsets (1..7) x 2 templates = 98 prompts, full factorial.
The answer day = (start + offset) mod 7 forms a Latin square over (start, offset), so start and offset main effects are
orthogonal to the answer. Each template has its own fixed 3-example demo block (identical for all prompts of that
template; no ordered weekday list is shown). Answers are the space-prefixed day tokens (all single tokens in Huginn's
tokenizer; checked in the run's prep). Number words appear mid-sentence (template A) or as digits (template B).
Usage: python -I huginn_addition/make_weekday_prompts.py results/weekday_prompts.json
"""
from __future__ import annotations

import json
import sys

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven"]


def unit(n):
    return "day" if n == 1 else "days"


def qa(start, n, ans=None):  # template A
    s = f"Q: What day is {WORDS[n]} {unit(n)} after {DAYS[start]}?\nA:"
    return s if ans is None else s + f" {DAYS[ans]}\n"


def stmt(start, n, ans=None):  # template B
    s = f"If today is {DAYS[start]}, then in {n} {unit(n)} it will be"
    return s if ans is None else s + f" {DAYS[ans]}.\n"


TEMPLATES = {
    "A": (qa, [(4, 2), (0, 4), (2, 6)]),     # Friday+2=Sunday, Monday+4=Friday, Wednesday+6=Tuesday
    "B": (stmt, [(3, 3), (6, 5), (1, 1)]),   # Thursday+3=Sunday, Sunday+5=Friday, Tuesday+1=Wednesday
}


def main(out):
    rows = []
    for ti, (tname, (fn, demos)) in enumerate(TEMPLATES.items()):
        demo = "".join(fn(s, n, (s + n) % 7) for s, n in demos)
        for start in range(7):
            for n in range(1, 8):
                ans = (start + n) % 7
                rows.append(dict(uid=30000 + 100 * ti + 7 * start + (n - 1), template=tname, start=start, offset=n, ans=ans,
                                 start_day=DAYS[start], ans_day=DAYS[ans], prompt=demo + fn(start, n), answer=f" {DAYS[ans]}",
                                 cell=f"{start}_{n}"))
    json.dump(rows, open(out, "w"), indent=1)
    print(len(rows), "prompts")
    for t in TEMPLATES:
        print(repr(next(r["prompt"] + r["answer"] for r in rows if r["template"] == t)))


if __name__ == "__main__":
    main(sys.argv[1])
