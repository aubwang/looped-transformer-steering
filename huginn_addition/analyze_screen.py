"""Screen of few-shot demo variants (orbit function, mode `screen`): which variants blink, where the blinking starts,
and whether any answer flips between odd and even loops.
Usage: HUGINN_TOKENIZER=... python -I huginn_addition/analyze_screen.py results/<screen_run>

Per prompt: late drift (mean of the last 64 loops) at every position up to the last answer-prediction position;
'blinking' = some position > 1e-4 with lag-2 / lag-1 ratio < 0.05 at that position; earliest moving position (token);
readout over the last 64 loops (every loop) at every answer position: digit argmax per loop, number of argmax changes,
whether the argmax alternates with loop parity, top-2 margin (digit-renormalised) in each parity, and the largest
probability difference between parities. Matched pairs = same problem, one stationary and one blinking variant
(all variants of a problem are token-aligned by construction).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

THR = 1e-4


def tok_labels(ids):
    try:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(os.environ["HUGINN_TOKENIZER"])
        return [tk.decode([int(i)]) for i in ids]
    except Exception:
        return None


def main(run):
    out = f"{run}/analysis"
    os.makedirs(out, exist_ok=True)
    z = np.load(f"{run}/orbit.npz", allow_pickle=True)
    probs = {p["uid"]: p for p in json.load(open(f"{run}/problems.json"))}
    rows = []
    for i in range(len(z["uid"])):
        uid = int(z["uid"][i]); p = probs[uid]
        dr = z["drift"][i]; pp = z["pred_pos"][i]
        late = dr[-64:].mean(0)
        l2 = np.nanmedian(z["lag2_full"][i][-64:], 0)
        lim = int(pp[-1])
        mov = np.where(late[: lim + 1] > THR)[0]
        # settled period-2 at the earliest moving position: lag-2 closure < 0.01 and 32-loop drift ratio within 2%
        if len(mov):
            q0 = int(mov[0])
            ratio = float(dr[-32:, q0].mean() / dr[-64:-32, q0].mean())
            settled = bool(np.nanmedian(z["lag2_full"][i][-32:, q0]) < 0.01 and abs(ratio - 1) < 0.02)
        else:
            ratio, settled = np.nan, False
        labs = tok_labels(z["ids"][i])
        ro = z["ro"][i]  # (T, P, 10) digit log-probs, every loop over the tail
        P = np.exp(ro); Pn = P / P.sum(-1, keepdims=True)
        am = ro.argmax(-1)  # (T, P)
        truth = np.array([d["digit"] for d in p["digit_info"]])
        ev, od = Pn[0::2].mean(0), Pn[1::2].mean(0)  # parity-averaged distributions (P, 10)
        srt_e, srt_o = np.sort(ev, -1), np.sort(od, -1)
        changes = (np.diff(am, axis=0) != 0).sum(0)
        alternates = [bool(changes[j] == am.shape[0] - 1) for j in range(am.shape[1])]
        rows.append({
            "uid": uid, "pid": p["pid"], "demo": p["demo"], "problem": f"{p['a']} + {p['b']} = {p['c']}",
            "blinking": bool(len(mov) > 0), "settled_period2": settled, "drift_ratio_last32_over_prev32": ratio,
            "earliest_moving_pos": int(mov[0]) if len(mov) else None,
            "earliest_moving_token": (labs[int(mov[0])] if labs and len(mov) else None),
            "eq_pos": int(pp[0]), "max_late_drift": float(late[: lim + 1].max()),
            "lag2_ratio_at_earliest": float(l2[int(mov[0])]) if len(mov) else None,
            "answer_argmax_last": am[-1].tolist(), "truth": truth.tolist(),
            "correct_all_positions": bool((am[-1] == truth).all()),
            "argmax_changes_tail": changes.tolist(), "argmax_alternates_with_parity": alternates,
            "parity_max_prob_diff": float(np.abs(ev - od).max()),
            "margin_even": (srt_e[:, -1] - srt_e[:, -2]).round(3).tolist(),
            "margin_odd": (srt_o[:, -1] - srt_o[:, -2]).round(3).tolist(),
            "min_margin": float(min((srt_e[:, -1] - srt_e[:, -2]).min(), (srt_o[:, -1] - srt_o[:, -2]).min())),
            "prompt": p["prompt"],
        })
    pairs = []
    for pid in sorted({r["pid"] for r in rows}):
        rs = [r for r in rows if r["pid"] == pid]
        st = [r for r in rs if not r["blinking"]]; bl = [r for r in rs if r["blinking"]]
        if st and bl:
            pairs.append({"pid": pid, "stationary": [r["demo"] for r in st], "blinking": [r["demo"] for r in bl]})
    summ = {"prompts": len(rows), "late_motion_above_1e-4": sum(r["blinking"] for r in rows),
            "settled_period2": sum(r["settled_period2"] for r in rows),
            "unresolved_motion": [(r["pid"], r["demo"], round(r["drift_ratio_last32_over_prev32"], 3)) for r in rows if r["blinking"] and not r["settled_period2"]],
            "flicker_prompts": [(r["pid"], r["demo"], r["argmax_changes_tail"]) for r in rows if any(c > 0 for c in r["argmax_changes_tail"])],
            "matched_pairs": pairs, "rows": rows}
    json.dump(summ, open(f"{out}/summary.json", "w"), indent=1)
    for r in rows:
        print(f"{r['pid']:3d} {r['demo']:3s} blink={int(r['blinking'])} settled={int(r['settled_period2'])} src={r['earliest_moving_pos']}({r['earliest_moving_token']!r}) "
              f"drift={r['max_late_drift']:.1e} l2={r['lag2_ratio_at_earliest']} changes={r['argmax_changes_tail']} "
              f"parity_dp={r['parity_max_prob_diff']:.3f} min_margin={r['min_margin']:.3f} correct={r['correct_all_positions']}")
    print(json.dumps({k: v for k, v in summ.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
