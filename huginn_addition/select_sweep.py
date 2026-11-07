"""Threshold sweep selection: the 26 prompts of results/relax-20261105-200208 with their SAVED predictions (alpha*, lam)
from that run (computed before any relaxation there; reused unchanged, so the sweep's predictions are fixed in advance).
Usage: python -I huginn_addition/select_sweep.py results/sweep_selection.json
"""
import json
import sys

import numpy as np

RUN = "results/relax-20261105-200208"


def main(out):
    z = np.load(f"{RUN}/relax.npz", allow_pickle=True)
    P = json.load(open(f"{RUN}/problems.json"))
    summ = {r["uid"]: r for r in json.load(open(f"{RUN}/analysis_summary.json"))["rows"]}
    rows = []
    for j, p in enumerate(P):
        assert int(z["uid"][j]) == p["uid"] and bool(z["eligible"][j])
        q = dict(p)
        q.update(alpha_star_pred=float(z["alpha_star"][j]), lam_pred=float(z["flip_lam"][j]), relax_status=summ[p["uid"]].get("status"),
                 relax_all_ok=bool(summ[p["uid"]].get("all_source_predictions_ok")))
        rows.append(q)
    json.dump(rows, open(out, "w"), indent=1)
    print(len(rows), "prompts; relax all-ok:", sum(r["relax_all_ok"] for r in rows))


if __name__ == "__main__":
    main(sys.argv[1])
