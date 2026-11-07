"""Control experiment for the d* steering direction: answer-token lever vs semantic truth lever (prespecified).
30 held-out statements (15 units: 8 cities + 7 compare, seeded) from results/dstar_eval_items.json, in four formats:
  F1 'tf'   : original few-shot "True or false?" (labels " True"/" False"; semantic-TRUE label = " True")
  F2 'yn'   : "Is this statement correct?" (labels " Yes"/" No"; semantic-TRUE = " Yes")
  F3 'rev'  : "write False if the statement is true and True if it is false" with matching shots (labels " True"/" False";
              semantic-TRUE label = " False")
  F4 'ab'   : "A) true B) false\nAnswer:" (labels " A"/" B"; semantic-TRUE = " A")
  F5 'fq'   : counterbalanced "Is this statement false?" (semantic-TRUE label = " No")
  F6 'ba'   : counterbalanced "A) false B) true" (semantic-TRUE label = " B")
Usage: python -I huginn_addition/make_ctrl_prompts.py
"""
import json

import numpy as np

SHOTS = [("The Pacific is an ocean.", 1), ("Spiders have six legs.", 0), ("Ice is frozen water.", 1)]


def fmt(kind, s):
    if kind == "tf":
        return "".join(f"Statement: {x}\nTrue or false? {'True' if t else 'False'}\n" for x, t in SHOTS) + f"Statement: {s}\nTrue or false?"
    if kind == "yn":
        return "".join(f"Statement: {x}\nIs this statement correct? {'Yes' if t else 'No'}\n" for x, t in SHOTS) + f"Statement: {s}\nIs this statement correct?"
    if kind == "rev":
        head = "Write False if the statement is true and True if it is false.\n"
        return head + "".join(f"Statement: {x}\nAnswer: {'False' if t else 'True'}\n" for x, t in SHOTS) + f"Statement: {s}\nAnswer:"
    if kind == "fq":  # counterbalanced Yes/No: a TRUE statement should get " No"
        return "".join(f"Statement: {x}\nIs this statement false? {'No' if t else 'Yes'}\n" for x, t in SHOTS) + f"Statement: {s}\nIs this statement false?"
    if kind == "ba":  # counterbalanced A/B: a TRUE statement should get " B"
        return "".join(f"Statement: {x}\nA) false B) true\nAnswer: {'B' if t else 'A'}\n" for x, t in SHOTS) + f"Statement: {s}\nA) false B) true\nAnswer:"
    if kind == "ab":
        return "".join(f"Statement: {x}\nA) true B) false\nAnswer: {'A' if t else 'B'}\n" for x, t in SHOTS) + f"Statement: {s}\nA) true B) false\nAnswer:"


def main():
    ev = json.load(open("results/dstar_eval_items.json"))
    rng = np.random.default_rng(21)
    units = {}
    for it in ev:
        units.setdefault((it["dataset"], it["unit"]), []).append(it)
    cu = sorted(k for k in units if k[0] == "cities"); pu = sorted(k for k in units if k[0] == "compare")
    pick = [cu[i] for i in rng.choice(len(cu), 8, replace=False)] + [pu[i] for i in rng.choice(len(pu), 7, replace=False)]
    items = [it for k in pick for it in units[k]]
    rows = []
    for it in items:
        for kind in ("tf", "yn", "rev", "ab", "fq", "ba"):
            rows.append(dict(uid=it["uid"], unit=it["unit"], dataset=it["dataset"], truth=it["truth"], fmt=kind, prompt=fmt(kind, it["statement"])))
    json.dump(rows, open("results/ctrl_rows.json", "w"), indent=1)
    print(len(items), "statements;", len(rows), "rows"); print(rows[2]["prompt"])


if __name__ == "__main__":
    main()
