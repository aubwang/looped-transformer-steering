"""Long-run (K=1024) trajectories of prompts that were still moving at step 96-128, plus stationary controls.
Usage: python -I huginn_addition/analyze_orbit.py results/<orbit_run> [results/<jac_run>]

Per prompt: late drift by position; candidate source of motion = earliest position (at or before the last answer
prediction position) whose drift stays > 1e-4 late (attention is causal, so a moving position can only be driven by
itself or earlier positions; this is a candidate, not causal proof); drift time course (decaying / sustained /
growing); recurrence on the last 256 steps of a 64-d random projection (dip in ||x_k - x_{k-tau}|| at a period);
spectrum of the tail; digit readout changes over all 1024 steps (every step).
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_gradients import CAT, INK2  # noqa: E402

THR = 1e-4


def tok_labels(ids):
    try:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(os.environ["HUGINN_TOKENIZER"])
        return [tk.decode([int(i)]) for i in ids]
    except Exception:
        return None


def recurrence(x):  # x: (T, D) tail trajectory -> r(tau) = median ||x_k - x_{k-tau}|| / median ||x_k - x_{k-1}||
    T = len(x)
    d1 = np.median(np.linalg.norm(x[1:] - x[:-1], axis=1))
    spread = np.median(np.linalg.norm(x - x.mean(0), axis=1))
    taus = np.arange(1, T // 2 + 1)
    r = np.array([np.median(np.linalg.norm(x[t:] - x[:-t], axis=1)) for t in taus])
    return taus, r / max(d1, 1e-30), d1, spread, r


def main(run, jac_run=None):
    out = f"{run}/analysis"
    os.makedirs(out, exist_ok=True)
    z = np.load(f"{run}/orbit.npz", allow_pickle=True)
    probs = {p["uid"]: p for p in json.load(open(f"{run}/problems.json"))}
    K = z["drift"][0].shape[0]
    steps_tail = z["steps_tail"]
    jac = None
    if jac_run:
        J = np.load(f"{jac_run}/jacobian.npz")
        jac = {int(u): float(dr) for u, dr in zip(J["uid"], J["drift"])}
    res = []
    for i in range(len(z["uid"])):
        uid, pid = int(z["uid"][i]), int(z["pid"][i])
        p = probs[uid]
        dr = z["drift"][i]  # (K, L)
        pp = z["pred_pos"][i]
        ipos = p["ipos"]
        late = dr[-128:].mean(0)
        early = dr[95:128].mean(0)
        lim = int(pp[-1])
        moving = np.where(late[: lim + 1] > THR)[0]
        src = int(moving[0]) if len(moving) else None
        ip = int(pp[ipos])
        labs = tok_labels(z["ids"][i])
        rec = {"uid": uid, "pid": pid, "demo": p["demo"], "problem": f"{p['a']} + {p['b']} = {p['c']}", "L": int(z["L"][i]),
               "pred_pos": pp.tolist(), "ipos_token": ip,
               "answer_pos_drift_96_128": float(early[ip]), "answer_pos_drift_last128": float(late[ip]),
               "max_drift_last128": float(late.max()), "n_positions_moving_late": int((late > THR).sum()),
               "source_candidate": src, "source_offset_from_first_pred_pos": (src - int(pp[0])) if src is not None else None,
               "source_token": (labs[src] if labs and src is not None else None),
               "jac_run_drift_at_96": jac.get(uid) if jac else None, "this_run_drift_at_96": float(dr[95, ip])}
        # time course at the answer position (and source)
        for nm, pos in (("answer", ip), ("source", src)):
            if pos is None:
                continue
            c = dr[:, pos]
            w = [c[95:128].mean(), c[480:512].mean(), c[-32:].mean()]
            rec[f"{nm}_drift_96_512_1024"] = [float(x) for x in w]
            ratio = w[2] / max(w[0], 1e-30)
            rec[f"{nm}_course"] = "decaying" if ratio < 0.1 else "growing" if ratio > 3 else "sustained"
            # recurrence on the tail of the random projection
            x = z["proj_tail"][i][:, pos].astype(np.float64)
            taus, r, d1, spread, rawr = recurrence(x)
            k2 = taus >= 2
            best = int(taus[k2][np.argmin(r[k2])])
            rec[f"{nm}_tail_step_norm"] = float(d1)
            rec[f"{nm}_tail_spread"] = float(spread)
            rec[f"{nm}_recurrence_best_tau"] = best
            rec[f"{nm}_recurrence_min_over_step"] = float(r[k2].min())
            rec[f"{nm}_recurrence_min_over_spread"] = float(rawr[k2].min() / max(spread, 1e-30))
            # power spectrum of the leading tail PC
            xc = x - x.mean(0)
            _, sv, Vt = np.linalg.svd(xc, full_matrices=False)
            pc1 = xc @ Vt[0]
            f = np.fft.rfftfreq(len(pc1))
            pw = np.abs(np.fft.rfft(pc1)) ** 2
            j = 1 + np.argmax(pw[1:])
            rec[f"{nm}_tail_pc1_var_frac"] = float(sv[0] ** 2 / (sv ** 2).sum())
            rec[f"{nm}_tail_pc1_peak_period"] = float(1 / f[j])
            rec[f"{nm}_tail_pc1_peak_power_frac"] = float(pw[j] / pw[1:].sum())
            if nm == "answer":
                rec["_curve"] = (taus, r)
                rec["_pc"] = xc @ Vt[:2].T
        # readout over all steps at every answer position
        ro = z["ro"][i]  # (K, P, 10)
        am = ro.argmax(-1)
        truth = np.array([d["digit"] for d in p["digit_info"]])
        k96 = int(np.searchsorted(z["steps_ro"], 96))  # index of the readout taken after step 96
        rec["readout_correct_frac_from_96"] = (am[k96:] == truth[None]).mean(0).round(4).tolist()
        rec["readout_n_changes_from_96"] = (np.diff(am[k96:], axis=0) != 0).sum(0).tolist()
        rec["readout_full_argmax_is_digit_frac_from_96"] = z["ro_full_is_digit"][i][k96:].mean(0).round(4).tolist()
        res.append(rec)

    moving_pids = sorted({r["pid"] for r in res if r["answer_pos_drift_96_128"] > THR})
    moving_prompts = [f"{r['pid']}{r['demo']}" for r in res if r["answer_pos_drift_96_128"] > THR]
    summ = {"K": int(K), "prompts": len(res), "moving_at_96_128_pids": moving_pids, "moving_at_96_128_prompts": moving_prompts,
            "rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in res]}
    json.dump(summ, open(f"{out}/summary.json", "w"), indent=1, default=float)
    for r in summ["rows"]:
        print(json.dumps(r, default=float))

    # figures
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.3))
    ax = axs[0]
    for r, i in zip(res, range(len(res))):
        mv = r["answer_pos_drift_96_128"] > THR
        c = z["drift"][i][:, r["ipos_token"]]
        ax.plot(np.arange(1, K + 1), np.maximum(c, 1e-9), color=CAT[1] if mv else CAT[0], lw=1, alpha=0.8)
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.plot([], [], color=CAT[1], label="still moving at step 96–128"); ax.plot([], [], color=CAT[0], label="settled (incl. controls)")
    ax.set_xlabel("loop step"); ax.set_ylabel("relative change per step (answer position)")
    ax.set_title("Do the moving trajectories ever settle?"); ax.legend(fontsize=8)
    ax = axs[1]
    for r in res:
        if r["answer_pos_drift_96_128"] > THR and "_curve" in r:
            ax.plot(r["_curve"][0], r["_curve"][1], lw=1, label=f"pid {r['pid']}{r['demo']}")
    ax.set_xlabel("lag τ (steps)"); ax.set_ylabel("median ‖x_k − x_{k−τ}‖ / one-step change")
    ax.set_title("Recurrence over last 256 steps\n(dip to ~0 at τ = period)"); ax.legend(fontsize=6.5, ncol=2)
    ax.set_yscale("log")
    ax = axs[2]
    mv = [r for r in res if r["answer_pos_drift_96_128"] > THR and "_pc" in r]
    if mv:
        top = max(mv, key=lambda r: r["answer_pos_drift_last128"])
        pc = top["_pc"]
        sc = ax.scatter(pc[:, 0], pc[:, 1], c=steps_tail, s=6, cmap="viridis")
        ax.plot(pc[:, 0], pc[:, 1], color=INK2, lw=0.4, alpha=0.6)
        fig.colorbar(sc, ax=ax, label="step")
        ax.set_title(f"Tail trajectory, answer position\n(pid {top['pid']}{top['demo']}, top-2 PCs of projection)")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout(); fig.savefig(f"{out}/orbit.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/") if len(sys.argv) > 2 else None)
