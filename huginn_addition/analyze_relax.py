"""Predictive source under-relaxation (intervene_app.py `relax`). Prespecified before the run.
Usage: python -I huginn_addition/analyze_relax.py results/<relax_run>

Eligibility (decided in-run, before any relaxation): valid fixed point and a validated real flip eigenvalue lam < -1 of
the source's own block J_{p0 p0}. Prediction: threshold alpha* = 2 / (1 - lam); conditions target relaxed multipliers.
Outcome per condition, at the source p0, over the last 32 of T loops (D = mean relative drift at p0):
  STOPPED  if D < 1e-5 and D < 0.01 x D(sham)
  PERSISTS if D > 0.25 x D(sham) and median lag-2 closure at p0 < 0.05
  else UNCLEAR.
Predicted: sham PERSISTS; src_m0.6 STOPPED; src_m0.85 STOPPED; src_half PERSISTS (relaxed, but alpha > alpha*);
down_m0.6 (same alpha at the downstream period-2 position r) PERSISTS at p0 (causality sanity check; r reported).
Primary: per eligible prompt, all source predictions correct (sham, m0.6, m0.85, and half where run); count and fraction.
Secondary: (a) whole-sequence stationarity under src_m0.6 (max-over-positions drift last 32 < 1e-5; secondary sources
otherwise); (b) quantitative rate: per-loop decay factor of the source drift under src_m0.6 / src_m0.85, fitted
log-linearly over loops 8..56 (only above 1e-9), vs the predicted |multiplier| (0.6 / 0.85); (c) for src_half, the
relaxed cycle's drift relative to sham vs the supercritical-flip expectation sqrt(mu_relaxed / mu) = sqrt(1/2)
(amplitude ratio; drift ~ 2 x amplitude), descriptive only.
Ineligible prompts are listed with their reason (no valid fixed point / no validated lam < -1, with the candidate value).
Revisions after Astra's code review (before data): (i) COMPLETENESS: a prompt counts only if every scheduled condition
completed; otherwise INCOMPLETE (reported separately). (ii) SHAM ESTABLISHED: sham D > 1e-4 and lag-2 < 0.05, else the
prompt is SHAM-NOT-ESTABLISHED and excluded from the primary count. (iii) OTHER MODES: from the validated top-modulus Ritz
values of start 0 (FD residual < 0.02 at both eps), the predicted relaxed multipliers |1 - a + a mu| under each source
condition's alpha are listed; a STOPPED prediction is 'unconditional' only if all are < 1 (with a < 1 this holds whenever
|mu| < 1, by convexity), else 'conditional'. (iv) LOCALISATION: under src_m0.6, positions with tail drift > 1e-5 are
listed (earliest first) instead of inferring secondary sources from whole-sequence drift.
"""
from __future__ import annotations

import json
import sys

import numpy as np


def main(run):
    z = np.load(f"{run}/relax.npz", allow_pickle=True)
    P = json.load(open(f"{run}/problems.json"))
    meta = json.load(open(f"{run}/meta.json"))
    n = len(P)
    lam = z["flip_lam"]; el = z["eligible"]
    rows = []
    cr, cn, ca, cp, cd = z["cond_row"], z["cond_name"], z["cond_alpha"], z["cond_pred_mult"], z["cond_done"]
    D = z["drift_p0"][:, -32:].mean(1); Dr = z["drift_r"][:, -32:].mean(1); Dm = z["drift_max"][:, -32:].mean(1)
    l2 = z["lag2_p0"]
    pred = {"sham": "PERSISTS", "src_m0.6": "STOPPED", "src_m0.85": "STOPPED", "src_half": "PERSISTS", "down_m0.6": "PERSISTS"}
    for j, p in enumerate(P):
        base = dict(uid=p["uid"], p0=int(z["p0"][j]), r=int(z["r"][j]), role=p["earliest_p2_role"], fp_ok=bool(z["fp_ok"][j]),
                    lam=None if not np.isfinite(lam[j]) else round(float(lam[j]), 4),
                    flip_fd_resid_by_start=np.round(z["val_fd_resid"][:, j, 0].max(-1), 4).tolist(),
                    eligible=bool(el[j]), warm_drift_p0=float(z["warm_drift_p0"][j]))
        if not el[j]:
            rows.append(base); continue
        sched = np.where(cr == j)[0]
        if not cd[sched].all():
            rows.append(dict(**base, status="INCOMPLETE")); continue
        ks = sched
        sham = [k for k in ks if cn[k] == "sham"]
        Ds = D[sham[0]]
        sham_ok = bool(Ds > 1e-4 and l2[sham[0]] < 0.05)
        vl = z["val_lam"][0, j, 1:]; vr = z["val_fd_resid"][0, j, 1:]
        top_valid = [complex(v) for v, rr_ in zip(vl, vr) if np.isfinite(v.real) and np.all(rr_ < 0.02)]
        conds = {}
        for k in ks:
            if D[k] < 1e-5 and D[k] < 0.01 * Ds:
                out = "STOPPED"
            elif D[k] > 0.25 * Ds and l2[k] < 0.05:
                out = "PERSISTS"
            else:
                out = "UNCLEAR"
            c = dict(alpha=round(float(ca[k]), 4), pred_mult=None if not np.isfinite(cp[k]) else round(float(cp[k]), 3), outcome=out,
                     predicted=pred[str(cn[k])], ok=out == pred[str(cn[k])], D_p0=float(D[k]), D_r=float(Dr[k]), D_max=float(Dm[k]),
                     lag2=float(l2[k]))
            if str(cn[k]).startswith("src"):
                a_ = float(ca[k])
                c["top_mode_relaxed_moduli"] = [round(abs(1 - a_ + a_ * mu_), 4) for mu_ in top_valid]
                c["prediction_unconditional"] = bool(all(abs(1 - a_ + a_ * mu_) < 1 for mu_ in top_valid))
            if str(cn[k]) == "src_m0.6" and not Dm[k] < 1e-5:
                tp = z["tail_pos"][k]
                c["active_positions"] = [int(x) for x in np.where(tp > 1e-5)[0]][:12]
            if str(cn[k]) in ("src_m0.6", "src_m0.85"):
                y = z["drift_p0"][k, 8:57]; t = np.arange(8, 8 + len(y)); m = y > 1e-9
                c["fitted_rate"] = float(np.exp(np.polyfit(t[m], np.log(y[m]), 1)[0])) if m.sum() >= 5 else None
                c["whole_sequence_stationary"] = bool(Dm[k] < 1e-5)
            if str(cn[k]) == "src_half":
                c["drift_ratio_vs_sham"] = float(D[k] / Ds) if Ds > 0 else None
            conds[str(cn[k])] = c
        src_ok = all(conds[c]["ok"] for c in ("sham", "src_m0.6", "src_m0.85", "src_half") if c in conds)
        rows.append(dict(**base, status="OK" if sham_ok else "SHAM-NOT-ESTABLISHED", alpha_star=round(float(z["alpha_star"][j]), 4),
                         mu=round(float(z["mu"][j]), 4), flip_spread=round(float(z["flip_spread"][j]), 4),
                         flip_unc=round(float(z["flip_unc"][j]), 4), top_valid_modes=[str(np.round(v, 3)) for v in top_valid],
                         all_source_predictions_ok=src_ok, conds=conds))
    E = [r for r in rows if r["eligible"] and r.get("status") == "OK"]
    summ = dict(
        n_prompts=n, n_fp_ok=int(z["fp_ok"].sum()), n_eligible=int(el.sum()), n_primary=len(E), skipped=meta["skipped"],
        n_incomplete=sum(r.get("status") == "INCOMPLETE" for r in rows),
        n_sham_not_established=sum(r.get("status") == "SHAM-NOT-ESTABLISHED" for r in rows),
        n_conditional_predictions=sum(not r["conds"][c].get("prediction_unconditional", True) for r in E for c in r["conds"]),
        all_source_predictions_ok=sum(r["all_source_predictions_ok"] for r in E),
        by_condition={c: dict(n=sum(c in r["conds"] for r in E), ok=sum(r["conds"][c]["ok"] for r in E if c in r["conds"]),
                              outcomes={o: sum(r["conds"][c]["outcome"] == o for r in E if c in r["conds"]) for o in ("STOPPED", "PERSISTS", "UNCLEAR")})
                      for c in pred},
        whole_sequence_stationary_m06=sum(r["conds"].get("src_m0.6", {}).get("whole_sequence_stationary", False) for r in E),
        rate_m06=[r["conds"]["src_m0.6"].get("fitted_rate") for r in E if "src_m0.6" in r["conds"]],
        rate_m085=[r["conds"]["src_m0.85"].get("fitted_rate") for r in E if "src_m0.85" in r["conds"]],
        half_drift_ratio=[r["conds"]["src_half"].get("drift_ratio_vs_sham") for r in E if "src_half" in r["conds"]],
        lam_eligible=[r["lam"] for r in E], alpha_star=[r["alpha_star"] for r in E],
        ineligible=[(r["uid"], r["fp_ok"], r["lam"]) for r in rows if not r["eligible"]])
    json.dump(dict(summary=summ, rows=rows), open(f"{run}/analysis_summary.json", "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
