"""Counterbalanced control for the frozen d*: truth lever vs affirmation / first-option lever. Prespecified before the run.
Usage: python -I huginn_addition/analyze_ctrl_cb.py results/<ctrl_cb_run> results/<ctrl_formats_run_1>,results/<ctrl_formats_run_2>
Formats (same 30 held-out statements, same push u_dstar, norm 12.5 every loop, offset 48):
  yn: "Is this statement correct?"  TRUE-label ' Yes'   AFFIRM ' Yes'   (from the earlier formats runs)
  fq: "Is this statement false?"    TRUE-label ' No'    AFFIRM ' Yes'
  ab: "A) true B) false"            TRUE-label ' A'     FIRST  ' A'     (earlier runs)
  ba: "A) false B) true"            TRUE-label ' B'     FIRST  ' A'
dR_sem = change in log p(TRUE-label) - log p(other) under the '+' push (the direction that raised ' True' in tf) vs sham;
dR_aff = change in log p(' Yes' or ' A') - log p(' No' or ' B'). In yn/ab the two coincide; in fq/ba they are opposite.
QUALIFICATION: (sham semantic accuracy >= 0.6 or AUROC >= 0.7) AND label mass >= 0.5.
VERDICT on the counterbalanced formats (fq, ba), '+' push, unit-cluster bootstrap 95% CIs (units within dataset):
  TRUTH-DOMINANT if dR_sem CI > 0 in every qualified counterbalanced format (and >= 1 qualified);
  AFFIRM/POSITION-DOMINANT if dR_sem CI < 0 (i.e. dR_aff > 0) in every qualified counterbalanced format;
  MIXED if the qualified counterbalanced formats disagree or any CI includes 0; INCONCLUSIVE if neither qualifies.
Also reported: the '-' push, random, and the yn/ab references.
"""
import json
import sys

import numpy as np
from scipy.stats import rankdata

TOK = {" True": 0, " False": 1, " Yes": 2, " No": 3, " A": 4, " B": 5}
MAP = {"yn": (" Yes", " No", " Yes", " No"), "fq": (" No", " Yes", " Yes", " No"),
       "ab": (" A", " B", " A", " B"), "ba": (" B", " A", " A", " B")}  # (TRUE, other, AFFIRM/FIRST, other)


def load(runs):
    LPs, rows, conds, RO = [], [], None, None
    seen = set()
    for run in runs.split(","):
        z = np.load(f"{run}/ctrl.npz"); rr = json.load(open(f"{run}/rows.json"))
        conds = [json.loads(c)[0] for c in z["conds"]]; RO = list(z["ro"])
        for k, r in enumerate(rr):
            if z["done"][k].all():
                key = (r["uid"], r["fmt"]); assert key not in seen; seen.add(key)
                LPs.append(z["LP"][k]); rows.append(r)
    return np.stack(LPs).astype(np.float64), rows, conds, RO


def main(rcb, rold):
    LPa, ra, ca, ROa = load(rcb); LPb, rb, cb, ROb = load(rold)
    assert ca == cb and ROa == ROb
    LP = np.concatenate([LPa, LPb]); rows = ra + rb; conds, RO = ca, ROa
    t48 = RO.index(48); sh = conds.index("sham")
    fm = np.array([r["fmt"] for r in rows]); y = np.array([r["truth"] for r in rows]); units = np.array([r["unit"] for r in rows])
    ds = np.array([r["dataset"] for r in rows])
    rng = np.random.default_rng(0)

    def boot(v, mask):
        bs = []
        for _ in range(5000):
            idx = []
            for d in ("cities", "compare"):
                uu = np.unique(units[mask & (ds == d)])
                for u in rng.choice(uu, len(uu)):
                    idx += list(np.where(mask & (units == u))[0])
            bs.append(np.median(v[idx]))
        return [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
    out = {"formats": {}}
    for f, (tl, ol, al, al2) in MAP.items():
        m = fm == f
        if not m.any():
            continue
        assert m.sum() == 30, (f, int(m.sum()))
        Rsem = LP[:, :, :, TOK[tl]] - LP[:, :, :, TOK[ol]]; Raff = LP[:, :, :, TOK[al]] - LP[:, :, :, TOK[al2]]
        s_ = Rsem[m, sh, t48]; yy = y[m]
        acc = float(((s_ > 0) == (yy == 1)).mean())
        r_ = rankdata(np.concatenate([s_[yy == 1], s_[yy == 0]])); n1, n0 = (yy == 1).sum(), (yy == 0).sum()
        auc = float((r_[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        mass = float(np.median(np.exp(LP[m, sh, t48, TOK[tl]]) + np.exp(LP[m, sh, t48, TOK[ol]])))
        res = dict(sham_accuracy=acc, sham_auroc=auc, label_mass=mass, qualified=bool((acc >= 0.6 or auc >= 0.7) and mass >= 0.5))
        for cn in ("dstar+", "dstar-", "rand+"):
            c = conds.index(cn)
            dsem = Rsem[:, c, t48] - Rsem[:, sh, t48]; daff = Raff[:, c, t48] - Raff[:, sh, t48]
            res[cn] = dict(dR_sem=float(np.median(dsem[m])), dR_sem_ci=boot(dsem, m), dR_aff=float(np.median(daff[m])))
        out["formats"][f] = res
    F = out["formats"]; cbq = [f for f in ("fq", "ba") if f in F and F[f]["qualified"]]
    if not cbq:
        verdict = "INCONCLUSIVE"
    elif all(F[f]["dstar+"]["dR_sem_ci"][0] > 0 for f in cbq):
        verdict = "TRUTH-DOMINANT"
    elif all(F[f]["dstar+"]["dR_sem_ci"][1] < 0 for f in cbq):
        verdict = "AFFIRM/POSITION-DOMINANT"
    else:
        verdict = "MIXED"
    out["verdict"] = verdict; out["qualified_counterbalanced"] = cbq
    json.dump(out, open(f"{rcb}/analysis_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"), sys.argv[2])
