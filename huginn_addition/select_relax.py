"""Select prompts for the predictive source under-relaxation test (intervene_app.py `relax`).
All 26 census settled-period-2 prompts (fresh: never used in the cycle-source / bifurcation work). Source p0 = earliest
period-2 position (census classification); downstream control position r = the LAST period-2 position (> p0), else None.
Usage: python -I huginn_addition/select_relax.py results/relax_selection.json
"""
import json
import sys


def main(out):
    s = json.load(open("results/census_analysis/summary.json"))["prompts"]
    probs = {p["uid"]: p for p in json.load(open("results/census_prompts_all.json"))}
    rows = []
    for q in s:
        if q["cls"] != "period2":
            continue
        p = dict(probs[q["uid"]])
        pos = q["p2_positions"]
        p.update(p0=int(pos[0]), r=int(pos[-1]) if pos[-1] > pos[0] else None, n_p2_positions=len(pos),
                 earliest_p2_role=q["earliest_p2_role"])
        rows.append(p)
    json.dump(rows, open(out, "w"), indent=1)
    print(len(rows), "prompts;", sum(r["r"] is None for r in rows), "without a downstream position")
    for r in rows:
        print(r["uid"], r["p0"], r["r"], r["n_p2_positions"], r["earliest_p2_role"])


if __name__ == "__main__":
    main(sys.argv[1])
