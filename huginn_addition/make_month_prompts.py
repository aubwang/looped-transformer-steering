"""Month shape / context-blend experiment prompts.
Two few-shot templates (different demo months), query "Q: What month comes after {M}?\nA:" (answer " {M+1}"); all
months are single space-prefixed tokens, so all 24 pure prompts within a template have identical length and differ only
at the query-month token. Blend items: pairs (m0, m1) — 6 OPPOSITE pairs (m, m+6), m = 0..5, and 6 QUARTER pairs
(m, m+3), m = 0, 2, 4, 6, 8, 10 — x 2 templates x a in {0.1, ..., 0.9}; E_a = (1 - a) E(m0) + a E(m1) (whole prelude
output tensor). Pure prompts are the a = 0 / 1 endpoints.
Usage: python -I huginn_addition/make_month_prompts.py results/month_prompts.json
"""
import json
import sys

M = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
DEMOS = {"A": [(2, 3), (9, 10), (5, 6)], "B": [(7, 8), (0, 1), (10, 11)]}


def prompt(t, m):
    d = "".join(f"Q: What month comes after {M[x]}?\nA: {M[y]}\n" for x, y in DEMOS[t])
    return d + f"Q: What month comes after {M[m]}?\nA:"


def main(out):
    pure = [dict(uid=90000 + 100 * ti + m, template=t, month=m, answer=(m + 1) % 12, prompt=prompt(t, m))
            for ti, t in enumerate(DEMOS) for m in range(12)]
    pairs = [(m, m + 6, "opposite") for m in range(6)] + [(m, (m + 3) % 12, "quarter") for m in range(0, 12, 2)]
    alphas = [round(0.1 * i, 1) for i in range(1, 10)]
    items = [dict(template=t, m0=m0, m1=m1, kind=k, a=a) for t in DEMOS for (m0, m1, k) in pairs for a in alphas]
    json.dump(dict(pure=pure, items=items, months=M), open(out, "w"), indent=1)
    print(len(pure), "pure prompts;", len(items), "blend items"); print(pure[3]["prompt"])


if __name__ == "__main__":
    main(sys.argv[1])
