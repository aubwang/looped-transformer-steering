"""Convergence check of the own-block spectrum on the 40 random stationary positions (intervene_app.py `ownspec`).
Usage: python -I huginn_addition/analyze_ownspec.py results/<ownspec_run> [results/<ownblock_random_run>]

Prespecified rules (written before the run; revised after Astra's code review, still before any data):
- Ritz values at Krylov size m' are eigenvalues of the leading m' x m' block of the saved Hessenberg H, for m' in SIZES.
  Arnoldi residual of a Ritz pair = |h_{m'+1,m'} * y_last| (unit y). Conjugate pairs are represented by Im >= 0 and
  compared with the conjugacy-aware distance cd(a, b) = min(|a - b|, |a - conj b|).
- The verdicts require the full Krylov size (arn_dim == 64) and FD validation at that size; otherwise the run is
  reported INCOMPLETE (actual sizes listed, no verdict).
- A (row, start) leading value is CONVERGED if (i) cd(lam_lead(64), lam_lead(48)) < 0.01, (ii) its explicit FD residual
  at size 64 is < 0.02 at both eps, (iii) eps-consistency: relative ||J_eps v - J_eps/2 v|| / ||J_eps v|| < 0.02.
- A row is START-STABLE if every start's leading value is converged and the leading values agree pairwise within
  cd < 0.01 (eigenvalue agreement, not just modulus).
- A validated NEGATIVE mode = a validated Ritz value (FD residual < 0.02 at both eps, eps-consistency < 0.02) with
  Re < 0 and |Im| <= 0.05 |lam|. Validation slots: top-4 by modulus + 3 most-negative near-real candidates per start, so
  "none" means none among those candidates. A row's negative mode is REPLICATED if some validated negative value in one
  start lies within cd < 0.02 of a validated negative value in another start.
- Per token type: CONVERGED summaries (start-stable rows only) reported separately from PROVISIONAL ones (all rows with a
  valid fixed point): median leading modulus, fraction with a complex leading pair (|Im| > 0.01), rows with a validated /
  replicated negative mode and their values. Comparison with ownblock_random (m=24) after asserting matching (uid, pos).
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

SIZES = [16, 24, 32, 48, 64]
M_FULL = 64


def cd(a, b):
    return min(abs(a - b), abs(a - np.conj(b)))


def canon(c):
    return complex(c.real, abs(c.imag))


def ritz(H, mm):
    lam, Y = np.linalg.eig(H[:mm, :mm])
    res = np.abs(H[mm, mm - 1] * Y[-1, :]) / np.linalg.norm(Y, axis=0)
    o = np.argsort(-np.abs(lam))
    return lam[o], res[o]


def ttype(p):
    tok = p["val_tok"].strip()
    q = p["val_role"].startswith("query") or p["val_role"] == "answer_prefix"
    return {"=": "eq", "+": "plus"}.get(tok, "digit" if tok.isdigit() else "other") + ("_query" if q else "_demo")


def valid(fd, ed):
    return bool(np.all(fd < 0.02) and ed < 0.02)


def summarize(rs):
    if not rs:
        return dict(n=0)
    return dict(n=len(rs), median_lead_mod=round(float(np.median([np.median([abs(c) for c in o["lead64"]]) for o in rs])), 3),
                frac_complex_lead=round(float(np.mean([abs(o["lead64"][0].imag) > 0.01 for o in rs])), 2),
                n_with_validated_neg=sum(o["neg_best"] is not None for o in rs),
                n_with_replicated_neg=sum(o["neg_replicated"] for o in rs),
                neg_values=sorted(round(o["neg_best"].real, 3) for o in rs if o["neg_best"] is not None))


def main(run, prev=None):
    z = np.load(f"{run}/ownspec.npz", allow_pickle=True)
    probs = json.load(open(f"{run}/problems.json"))
    Hs = z["arn_H"]; dn = int(z["arn_dim"])
    S, n = Hs.shape[:2]
    have_val = "val_lam" in z.files
    if dn != M_FULL or not have_val:
        sizes = [s for s in SIZES if s <= dn]
        print(json.dumps(dict(verdict="INCOMPLETE", arn_dim=dn, sizes_available=sizes, fd_validation=have_val)))
        for r in range(n):
            print(r, {mm: [round(abs(ritz(Hs[si, r], mm)[0][0]), 3) for si in range(S)] for mm in sizes})
        return
    vl, vk, vr, ve = z["val_lam"], z["val_kind"], z["val_fd_resid"], z["val_eps_rel_diff"]
    prevlead = None
    if prev:
        zp = np.load(f"{prev}/ownblock.npz", allow_pickle=True)
        assert (zp["uid"] == z["uid"]).all() and (zp["pos"] == z["pos"]).all(), "row order differs from previous run"
        prevlead = zp["arn_lead"]
    rows = []
    for r, p in enumerate(probs):
        per_start, L, negsets = [], [], []
        for si in range(S):
            seq = {mm: ritz(Hs[si, r], mm) for mm in SIZES}
            lead = {mm: seq[mm][0][0] for mm in SIZES}
            assert cd(lead[M_FULL], vl[si, r, 0]) < 1e-8 and vk[si, r, 0] == 0  # slot 0 = the leading value at m=64
            conv = bool(cd(lead[64], lead[48]) < 0.01 and valid(vr[si, r, 0], ve[si, r, 0]))
            L.append(canon(lead[M_FULL]))
            negs = [canon(vl[si, r, j]) for j in range(vl.shape[2]) if vk[si, r, j] >= 0 and valid(vr[si, r, j], ve[si, r, j])
                    and vl[si, r, j].real < 0 and abs(vl[si, r, j].imag) <= 0.05 * abs(vl[si, r, j])]
            negsets.append(negs)
            per_start.append(dict(lead={mm: complex(np.round(canon(lead[mm]), 4)) for mm in SIZES},
                                  lead_arn_res={mm: round(float(seq[mm][1][0]), 4) for mm in SIZES},
                                  lead_fd=vr[si, r, 0].round(4).tolist(), lead_eps_diff=round(float(ve[si, r, 0]), 4),
                                  converged=conv, negs=[complex(np.round(c, 4)) for c in negs],
                                  slots=[(complex(np.round(canon(vl[si, r, j]), 4)), int(vk[si, r, j]), vr[si, r, j].round(4).tolist(),
                                          round(float(ve[si, r, j]), 4)) for j in range(vl.shape[2]) if vk[si, r, j] >= 0]))
        stable = bool(all(ps["converged"] for ps in per_start) and max(cd(a, b) for a in L for b in L) < 0.01)
        allneg = [c for ns in negsets for c in ns]
        neg_best = min(allneg, key=lambda c: c.real) if allneg else None
        neg_repl = any(cd(a, b) < 0.02 for i in range(S) for j in range(i + 1, S) for a in negsets[i] for b in negsets[j])
        rows.append(dict(i=r, uid=p["uid"], pos=p["val_pos"], tok=p["val_tok"], role=p["val_role"], type=ttype(p),
                         fp_valid=bool(z["fp_valid"][r]), start_stable=stable, lead64=L,
                         prev_m24_lead=None if prevlead is None else canon(prevlead[r]),
                         neg_best=neg_best, neg_replicated=neg_repl, per_start=per_start))
    by = defaultdict(list)
    for o in rows:
        if o["fp_valid"]:
            by[o["type"]].append(o)
    summ = {t: dict(provisional=summarize(rs), converged=summarize([o for o in rs if o["start_stable"]])) for t, rs in sorted(by.items())}
    ok = [o for o in rows if o["fp_valid"]]
    overall = dict(rows=n, starts=S, krylov=dn, fp_valid=len(ok), start_stable=sum(o["start_stable"] for o in ok),
                   converged_row_starts=sum(ps["converged"] for o in ok for ps in o["per_start"]),
                   rows_with_replicated_neg=sum(o["neg_replicated"] for o in ok))
    if prevlead is not None:
        d = [cd(o["prev_m24_lead"], o["lead64"][0]) for o in ok]
        overall["prev_m24_vs_m64_start0_cd"] = dict(median=round(float(np.median(d)), 4), max=round(float(np.max(d)), 4),
                                                    n_over_0p01=int(sum(x > 0.01 for x in d)))
    json.dump(dict(overall=overall, by_type=summ, rows=rows), open(f"{run}/analysis_summary.json", "w"), indent=1, default=str)
    for o in rows:
        print(o["i"], o["type"], repr(o["tok"]), "fp_ok" if o["fp_valid"] else "FP_BAD", "stable" if o["start_stable"] else "UNSTABLE",
              [complex(np.round(c, 3)) for c in o["lead64"]], "neg", None if o["neg_best"] is None else complex(np.round(o["neg_best"], 3)),
              "repl" if o["neg_replicated"] else "", "prev24", None if o["prev_m24_lead"] is None else complex(np.round(o["prev_m24_lead"], 3)),
              "| |lead| by m (start0):", {mm: round(abs(c), 3) for mm, c in o["per_start"][0]["lead"].items()})
    print(json.dumps(overall, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/") if len(sys.argv) > 2 else None)
