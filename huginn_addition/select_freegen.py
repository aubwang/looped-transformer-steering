"""Select prompts for the free-running generation check (intervene_app.py `freegen`).
Logic: the model is causal, so greedy free-running generation is token-for-token identical to the teacher-forced census
readout up to and including the first answer position whose full-vocabulary argmax (at loop 512) differs from the true
token. The census found no tail flicker at any answer position, so free-running can only differ AFTER a wrong token.
Hence: all settled-period-2 ("blinking") census prompts whose full-vocab argmax is wrong somewhere, plus the same number
of stationary wrong prompts matched on n_digits and (preferably) on the first-wrong answer index (seed 0).
Each row carries gen_from = first wrong answer index; the run generates answer indices gen_from..n_ans (forced prefix =
prompt + true answer tokens before gen_from; index n_ans = reference termination position, fixed horizon). The answer-
correct blinking prompts are added with gen_from = n_ans (termination readout only; the census did not read that
position). census_tail_argmax = the census full-vocab argmax over the last 64 loops at gen_from (reproduction gate; None
when gen_from = n_ans).
Usage: python -I huginn_addition/select_freegen.py results/freegen_selection.json
"""
from __future__ import annotations

import json
import sys

import numpy as np


def main(out):
    probs = {p["uid"]: p for p in json.load(open("results/census_prompts_all.json"))}
    cls = {q["uid"]: q["cls"] for q in json.load(open("results/census_analysis/summary.json"))["prompts"]}
    info = {}
    runs = ["results/census-20261105-162542", "results/census-20261105-163100", "results/census-20261105-163615",
            "results/census-20261105-164134"]
    for run in runs:
        z = np.load(f"{run}/census.npz", allow_pickle=True)
        for i, u in enumerate(z["uid"]):
            ids, pp, am = z["ids"][i], z["pred_pos"][i], z["ro_argmax"][i]
            assert int(z["steps"][i]) == 512 and am.shape[0] == 64 and int(u) not in info
            truth = ids[pp + 1]
            wrong_last = np.where(am[-1] != truth)[0]
            flick = bool((am != am[-1]).any())
            fw = int(wrong_last[0]) if len(wrong_last) else None
            info[int(u)] = dict(first_wrong=fw, tail_flicker=flick, n_ans=len(pp), tail=None if fw is None else am[:, fw].tolist())
    assert set(info) == set(cls) == set(probs)
    assert not any(v["tail_flicker"] for v in info.values())
    blink = [u for u in info if cls[u] == "period2" and info[u]["first_wrong"] is not None]
    stat = [u for u in info if cls[u] == "stationary" and info[u]["first_wrong"] is not None]
    rng = np.random.default_rng(0)
    chosen, used = [], set()
    for u in sorted(blink):
        nd, fw = probs[u]["n_digits"], info[u]["first_wrong"]
        cand = [v for v in stat if v not in used and probs[v]["n_digits"] == nd and probs[v]["pid"] != probs[u]["pid"]]
        best = [v for v in cand if info[v]["first_wrong"] == fw] or cand
        v = int(rng.choice(sorted(best)))
        used.add(v)
        for w, role in ((u, "blinking"), (v, "stationary_control")):
            q = dict(probs[w]); q.pop("ids", None); q.pop("pred_pos", None)
            q.update(gen_from=info[w]["first_wrong"], n_ans=info[w]["n_ans"], fg_role=role, match_uid=v if w == u else u,
                     census_tail_argmax=info[w]["tail"])
            chosen.append(q)
    for u in sorted(x for x in info if cls[x] == "period2" and info[x]["first_wrong"] is None):  # termination readout only
        q = dict(probs[u]); q.pop("ids", None); q.pop("pred_pos", None)
        q.update(gen_from=info[u]["n_ans"], n_ans=info[u]["n_ans"], fg_role="blinking_correct_termination", match_uid=None,
                 census_tail_argmax=None)
        chosen.append(q)
    json.dump(chosen, open(out, "w"), indent=1)
    for q in chosen:
        print(q["uid"], q["pid"], q["demo"], q["fg_role"], q["n_digits"], f"{q['a']}+{q['b']}={q['c']}", "gen_from", q["gen_from"], "of", q["n_ans"])


if __name__ == "__main__":
    main(sys.argv[1])
