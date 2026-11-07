"""Where the period-2 cycles come from (intervene_app.py `cycsrc`).
Usage: python -I huginn_addition/analyze_cycsrc.py results/<cycsrc_run>

Per cycling prompt: hidden fixed point x* (damped iteration; residual, distance from the cycle midpoint, readout), the
validated flip eigenvalue of the source token's one-step Jacobian at x* (and its alignment with the cycle axis), the
growth factor of a small push along the cycle axis from x*, the answer position's spectrum at x*, the leading two-step
(Floquet) multiplier on the cycle, and the freeze test. Controls: same token types in stationary prompts.
Normal-form check for a supercritical flip (period-doubling) bifurcation with lambda = -(1 + mu): the 2-cycle's leading
multiplier ~ 1 - 4 mu and the amplitude ~ sqrt(mu) for small mu (exact constants depend on the cubic term).
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
from analyze_gradients import CAT, GRID, INK2  # noqa: E402

THR = 1e-4
RES_OK = 1e-2  # FD eigen-residual threshold for "validated"


def main(run):
    out = f"{run}/analysis"
    os.makedirs(out, exist_ok=True)
    z = np.load(f"{run}/cycsrc.npz", allow_pickle=True)
    info = [json.loads(s) for s in z["info"]]
    probs = {p["uid"]: p for p in json.load(open(f"{run}/problems.json"))}
    jr = [tuple(x) for x in z["j_rows"]]
    lab = lambda i, q: ("src" if q == i["src"] else "ans" if q == i["ipos_t"] else "op2" if q == i["op2"] else "eq" if q == i["eq"] else str(q))

    def spec(nm, n):
        lam = z[f"{nm}__ritz"][n]
        sl, v = z[f"{nm}__sel_lam"][n], z[f"{nm}__sel_valid"][n]
        fr, fd, ov = z[f"{nm}__fd_resid"][n], z[f"{nm}__fd_eps_rel_diff"][n], z[f"{nm}__axis_span_overlap"][n]
        H = z[f"{nm}__H"][n]
        m = H.shape[1]
        lh = np.linalg.eigvals(H[: m // 2, : m // 2])
        return {"top_abs": float(np.abs(lam[0])), "top": [float(lam[0].real), float(lam[0].imag)],
                "top_abs_half_krylov": float(np.abs(lh).max()),
                "top_fd_resid_eps_small": float(fr[1, 0]), "top_axis_overlap": float(ov[0]),
                "neg_real_lead": float(sl[1].real) if v[1] else None, "neg_real_fd_resid_eps_small": float(fr[1, 1]) if v[1] else None,
                "neg_real_axis_overlap": float(ov[1]) if v[1] else None,
                "neg_real_validated": bool(v[1] and fr[1, 1] < RES_OK and fd[1] < 1e-2)}

    rows = []
    for r, i in enumerate(info):
        p = probs[i["uid"]]
        truth = [d["digit"] for d in p["digit_info"]]
        nP = len(truth)
        rec = {"prompt": f"{i['pid']}{i['demo']}", "uid": i["uid"], "problem": f"{p['a']} + {p['b']} = {p['c']}",
               "cycling": i["src"] is not None, "src": i["src"], "src_token": ("op2_first" if i["src"] == i["op2"] else "eq" if i["src"] == i["eq"] else i["src"]),
               "ipos": i["ipos_t"], "moving": i["moving"],
               "readout_A_B_xstar_truth": [z["lp_A"][r].argmax(-1)[:nP].tolist(), z["lp_B"][r].argmax(-1)[:nP].tolist(),
                                           z["lp_X"][r].argmax(-1)[:nP].tolist(), truth],
               "xstar_resid_max": float(z["fp_resid"][r, : i["L"]].max())}
        if i["src"] is not None:
            s, ip = i["src"], i["ipos_t"]
            rec.update(cycle_amp_rel_src=float(z["axis_rel"][r][s]), cycle_amp_rel_ans=float(z["axis_rel"][r][ip]),
                       closure_rel_src=float(z["closure_rel"][r][s]),
                       xstar_offset_from_midpoint_over_axis_src=float(z["fp_mid_off"][r, s]),
                       xstar_offset_from_midpoint_over_axis_ans=float(z["fp_mid_off"][r, ip]))
            pr = z["und_push_proj"][r][:, s]
            k = np.arange(len(pr))
            rec["push_growth_per_step_src"] = float(-np.exp(np.polyfit(k[:24], np.log(np.abs(pr[:24])), 1)[0]))
            rec["push_sign_alternates"] = bool(np.all(np.sign(pr[1:]) != np.sign(pr[:-1])))
            rec["plain_rollout_from_xstar_max_dist_src"] = float(z["und_plain_dist"][r][:, s].max())
        for n, (rr, q) in enumerate(jr):
            if rr == r:
                rec[f"J1_xstar_{lab(i, q)}"] = spec("J1_xstar", n)
                rec[f"J2_two_step_{lab(i, q)}"] = spec("J2_two_step_sA", n)
        # freeze
        fz = {}
        for j in np.where(z["f_row"] == r)[0]:
            q, mode, sn = int(z["f_q"][j]), str(z["f_mode"][j]), str(z["f_snap"][j])
            late = z["f_drift"][j][-16:].mean(0)
            key = f"{mode}<= {lab(i, q)}" if mode == "prefix" else f"single {lab(i, q)}"
            if q == -1:
                key = "none"
            elif i["src"] is not None and q == i["src"] - 1:
                key = "prefix<= src-1 (static only)"
            elif q == i["ipos_t"] - 1:
                key = f"{mode}<= ans-1" if mode == "prefix" else "single ans-1"
            fz[f"{key} [{sn}]"] = {"ans_drift": float(late[i["ipos_t"]]), "n_moving": int((late > THR).sum()),
                                   "moving": [int(x) for x in np.where(late > THR)[0]],
                                   "readout_unchanged": bool((z["f_lp_end"][j].argmax(-1) == z["lp_A"][r].argmax(-1)[:nP]).all())}
        rec["freeze"] = fz
        rows.append(rec)
    cyc = [r for r in rows if r["cycling"]]
    mu = np.array([-(r["J1_xstar_src"]["neg_real_lead"]) - 1 for r in cyc])
    flo = np.array([r["J2_two_step_src"]["top_abs"] for r in cyc])
    amp = np.array([r["cycle_amp_rel_src"] for r in cyc])
    summ = {"rows": rows,
            "normal_form": {"mu": mu.round(4).tolist(), "floquet_top": flo.round(3).tolist(), "pred_1_minus_4mu": (1 - 4 * mu).round(3).tolist(),
                            "amp_rel_src": amp.round(4).tolist(), "amp_over_sqrt_mu": (amp / np.sqrt(mu)).round(3).tolist(),
                            "push_growth": [round(r["push_growth_per_step_src"], 4) for r in cyc]},
            "controls_neg_real_lead": {f"{r['prompt']} {k[9:]}": r[k]["neg_real_lead"] for r in rows if not r["cycling"]
                                       for k in r if k.startswith("J1_xstar_") and r[k]["neg_real_lead"] is not None}}
    json.dump(summ, open(f"{out}/summary.json", "w"), indent=1, default=float)
    print(json.dumps(summ["normal_form"], indent=1)); print(json.dumps(summ["controls_neg_real_lead"], indent=1))

    # ---- figure
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.4))
    ax = axs[0]
    yl = []
    for k, r in enumerate(cyc):
        ax.scatter(r["J1_xstar_src"]["neg_real_lead"], k, color=CAT[1], s=40, zorder=3)
        ax.scatter(r["J1_xstar_ans"]["top_abs"] * -1, k, color=CAT[0], s=28, marker="s", zorder=3)
        yl.append(f"{r['prompt']} (cycling; source = {'“=”' if r['src_token'] == 'eq' else '1st digit of 2nd number'})")
    off = len(cyc)
    ctrl = [(r, t) for r in rows if not r["cycling"] for t in ("op2", "eq") if f"J1_xstar_{t}" in r]
    for k, (r, t) in enumerate(ctrl):
        v = r[f"J1_xstar_{t}"]["neg_real_lead"]
        if v is not None:  # filled = FD-validated eigenpair; hollow = unvalidated Ritz candidate
            ok = r[f"J1_xstar_{t}"]["neg_real_validated"]
            ax.scatter(v, off + k, s=30, marker="D", zorder=3, facecolor=INK2 if ok else "none", edgecolor=INK2)
        yl.append(f"{r['prompt']} control, {'“=”' if t == 'eq' else '1st digit of 2nd number'}")
    ax.axvline(-1, color=CAT[1], lw=1, ls=":")
    ax.set_yticks(range(len(yl))); ax.set_yticklabels(yl, fontsize=6.5)
    ax.scatter([], [], color=CAT[1], label="source token: flip eigenvalue at hidden fixed point")
    ax.scatter([], [], color=CAT[0], marker="s", label="answer position: −|largest eigenvalue|")
    ax.scatter([], [], color=INK2, marker="D", label="stationary control: most negative real eigenvalue (hollow = not validated)")
    ax.set_xlim(-1.15, -0.4); ax.invert_yaxis()
    ax.set_xlabel("eigenvalue of one loop step (left of −1 = flip-unstable)")
    ax.set_title("Source token: a sign-flipping mode past −1\n(50A only marginally: μ ≈ 0.0005)")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=1, frameon=False)

    ax = axs[1]
    keys = [("none", "no pinning"), ("prefix<= src-1 (static only)", "pin tokens before source"),
            ("prefix<= src [A]", "pin up to source"), ("prefix<= ans-1 [A]", "pin everything before answer")]
    w = 0.2
    for kk, (key, labk) in enumerate(keys):
        vals = []
        for r in cyc:
            f = r["freeze"]
            hit = [v for k_, v in f.items() if k_.startswith(key)]
            if not hit and key == "prefix<= src [A]" and r["src"] == r["ipos"] - 1:  # source is the token right before the answer
                hit = [v for k_, v in f.items() if k_.startswith("prefix<= ans-1 [A]")]
            vals.append(hit[0]["ans_drift"] if hit else np.nan)
        ax.bar(np.arange(len(cyc)) + (kk - 1.5) * w, vals, width=w, color=[GRID, CAT[2], CAT[1], CAT[0]][kk], label=labk)
    ax.axhline(THR, color=INK2, lw=0.8, ls=":")
    ax.set_yscale("log"); ax.set_xticks(range(len(cyc))); ax.set_xticklabels([r["prompt"] for r in cyc])
    ax.set_ylabel("selected answer position: change per loop (late)"); ax.set_title("Freeze test: pin earlier tokens,\ndoes the selected answer position still cycle?")
    ax.legend(fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    i50 = [k for k, r in enumerate(cyc) if r["prompt"].startswith("50")]
    if i50:
        ax.annotate("problem 50 has a 2nd\nsource (“=”)", (np.mean(i50), 3.5e-2), ha="center", fontsize=6.8, color=CAT[1])
    ax.set_ylim(top=1.2e-1)

    ax = axs[2]
    mm = np.linspace(0, mu.max() * 1.15, 50)
    ax.plot(mm, 1 - 4 * mm, color=INK2, ls=":", lw=1, label="flip normal form, small μ: 1 − 4μ")
    ax.scatter(mu, flo, color=CAT[1], s=40, zorder=3, label="two-step map, source block: largest Ritz value")
    for r, x_, y_ in zip(cyc, mu, flo):
        ax.annotate(r["prompt"], (x_, y_), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("μ  (flip eigenvalue = −(1 + μ))"); ax.set_ylabel("largest two-step multiplier, source block")
    ax.set_title("Consistency with a flip normal form\n(stability vs how far past −1)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.subplots_adjust(bottom=0.27); fig.savefig(f"{out}/cycle_source.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
