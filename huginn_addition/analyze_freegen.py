"""Free-running generation at adjacent loop budgets (intervene_app.py `freegen`).
Usage: python -I huginn_addition/analyze_freegen.py results/<freegen_run>
Per prompt: census reproduction gate; whether budgets 509-512 ever disagree (branch split) at any generated position;
the generated continuation(s) per budget; per generated position, the number of argmax changes over the last 64 loops
(flicker) and the top-1 minus top-2 log-prob gap at loops 511/512. Blinking vs stationary controls side by side;
answer-correct blinking prompts contribute the termination position only.
"""
import json
import sys
from collections import defaultdict

import numpy as np


def main(run):
    d = json.load(open(f"{run}/freegen.json"))
    meta = json.load(open(f"{run}/meta.json"))
    by = defaultdict(list)
    for b in d["branches"]:
        by[b["uid"]].append(b)
    rows = []
    for uid, bs in by.items():
        b0 = bs[0]
        hist = b0["hist"]
        rows.append(dict(uid=uid, role=b0["fg_role"], n_branches=len(bs), done=all(b["done"] for b in bs),
                         outputs={",".join(map(str, b["budgets"])): b["answer_text"] for b in bs}, truth=b0["true_answer"],
                         gen_from=b0["gen_from"], tail_changes=[h["n_changes_tail"] for h in hist],
                         gap_last=[round(h["top2_tail"][-1][0] - h["top2_tail"][-1][1], 3) for h in hist],
                         repro=d["census_repro"].get(str(uid))))
    rows.sort(key=lambda r: (r["role"], r["uid"]))
    for r in rows:
        print(r)
    roles = sorted({r["role"] for r in rows})
    summ = {"meta_skipped": meta["skipped"], "n_splits": len(d["splits"]), "splits": d["splits"],
            "repro_all_match": all(v["match_all"] for v in d["census_repro"].values()), "n_repro": len(d["census_repro"])}
    for ro in roles:
        rr = [r for r in rows if r["role"] == ro]
        summ[ro] = dict(n=len(rr), any_split=sum(r["n_branches"] > 1 for r in rr),
                        any_tail_flicker=sum(any(c > 0 for c in r["tail_changes"]) for r in rr),
                        all_done=sum(r["done"] for r in rr), min_gap=min(min(r["gap_last"]) for r in rr))
    print(json.dumps(summ, indent=1))
    json.dump(dict(summary=summ, rows=rows), open(f"{run}/analysis_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
