"""Control experiment for the frozen population d*: answer-token lever vs semantic truth lever. Prespecified before the runs.
Usage: python -I huginn_addition/analyze_ctrl.py results/<ctrl_formats_run> results/<ctrl_dose_run> results/<dstar_vjp_run>

30 held-out statements (15 units) x 4 formats (tf / yn / rev / ab). Push = frozen u_dstar (population d* from training
items, computed in the tf format to raise log p(' True') - log p(' False')), norm 12.5, added to the answer-position state
input every loop ('+' = the direction that raised ' True' in tf). Offsets after the 64-loop warm-up; primary t = 48.
Per format: R_tok = log p(' True') - log p(' False') (literal tokens; measured in every format), R_sem = log p(label meaning
TRUE) - log p(label meaning FALSE) (tf: True/False; yn: Yes/No; rev: False/True; ab: A/B). Effects = (R under push) - (R
under sham) at t = 48; medians with unit-cluster bootstrap 95% CIs (resample units within dataset, 5000).
QUALIFICATION per format (sham at 48): semantic accuracy = sign(R_sem) matches the label; AUROC of sham R_sem for truth;
  label mass (two label tokens). INTERPRETABLE = (accuracy >= 0.6 or AUROC >= 0.7) AND median label mass >= 0.5.
  (Truth discrimination, not proof of instruction-following.)
VERDICT (revised after Astra's review, before data; '+' push; interpretable formats only; requires rev interpretable and
  >= 3 interpretable formats, else INCONCLUSIVE):
  TOKEN EVIDENCE: dR_tok(+) CI > 0 in >= 2 of the interpretable non-tf formats (yn, rev, ab), with dlogp(' True') CIs
    reported (a higher True-vs-False preference can come from False falling).
  SEMANTIC EVIDENCE: dR_sem(+) CI > 0 in the interpretable formats among yn and ab.
  DECISIVE CONTRAST (rev, where they conflict): dR_sem(rev, +) CI < 0 -> TOKEN-DOMINANT; CI > 0 -> SEMANTIC-DOMINANT;
    otherwise NEITHER-DOMINANT. Token and semantic evidence are reported separately (mixtures allowed).
DOSE (tf): per-unit-dose effect dR_sem / norm at norms 1.5625, 3.125, 6.25, 12.5 vs the linear prediction a_i . u_dstar
  (a_i = A^T g from the dstar_vjp run; same items); Spearman across items and median observed/predicted ratio per dose.
PULSE / REMOVAL (tf, norm 12.5): pulse effect at offsets 0, 1, 4, 48; removal: effect at 24 (forced) vs 48 (24 loops after
  removal): RECOVERS if |median effect 48| <= 0.1 median effect 24 (requires median effect 24 >= 0.5).
Random direction: |dR_sem| and |dR_tok| per format (reference).
"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr

SEM = {"tf": (0, 1), "yn": (2, 3), "rev": (1, 0), "ab": (4, 5)}  # (TRUE label index, FALSE label index) into the 6 tokens


def main(rf, rd, rv):
    # formats run(s): comma-separated shards (a deadline-truncated run plus a completion run); keep fully completed rows,
    # merge by (uid, format); every (uid, format) of results/ctrl_rows.json must be present exactly once
    LPs, rows = [], []
    seen = set()
    for run in rf.split(","):
        zf = np.load(f"{run}/ctrl.npz"); rr = json.load(open(f"{run}/rows.json"))
        conds = [json.loads(c)[0] for c in zf["conds"]]; RO = list(zf["ro"])
        for k, r in enumerate(rr):
            if zf["done"][k].all():
                key = (r["uid"], r["fmt"]); assert key not in seen; seen.add(key)
                LPs.append(zf["LP"][k]); rows.append(r)
    expect = {(r["uid"], r["fmt"]) for r in json.load(open("results/ctrl_rows.json"))}
    assert seen == expect, f"missing {len(expect - seen)} rows"
    LP = np.stack(LPs).astype(np.float64); assert np.isfinite(LP).all(); t48 = RO.index(48)
    fm = np.array([r["fmt"] for r in rows]); y = np.array([r["truth"] for r in rows]); units = np.array([r["unit"] for r in rows])
    ds = np.array([r["dataset"] for r in rows]); uid = np.array([r["uid"] for r in rows])
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
    for f, (it, if_) in SEM.items():
        m = fm == f
        Rtok = LP[:, :, :, 0] - LP[:, :, :, 1]; Rsem = LP[:, :, :, it] - LP[:, :, :, if_]
        sh = conds.index("sham")
        acc = float(((Rsem[m, sh, t48] > 0) == (y[m] == 1)).mean())
        from scipy.stats import rankdata
        s_, yy = Rsem[m, sh, t48], y[m]
        r_ = rankdata(np.concatenate([s_[yy == 1], s_[yy == 0]])); n1 = (yy == 1).sum(); n0 = (yy == 0).sum()
        auc = float((r_[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        mass = float(np.median(np.exp(LP[m, sh, t48, it]) + np.exp(LP[m, sh, t48, if_])))
        res = dict(sham_semantic_accuracy=acc, sham_auroc=auc, label_mass=mass, interpretable=bool((acc >= 0.6 or auc >= 0.7) and mass >= 0.5))
        for cn in ("dstar+", "dstar-", "rand+"):
            c = conds.index(cn)
            dtok = Rtok[:, c, t48] - Rtok[:, sh, t48]; dsem = Rsem[:, c, t48] - Rsem[:, sh, t48]
            res[cn] = dict(dR_tok=float(np.median(dtok[m])), dR_tok_ci=boot(dtok, m), dR_sem=float(np.median(dsem[m])), dR_sem_ci=boot(dsem, m),
                           dlogp_True=float(np.median((LP[:, c, t48, 0] - LP[:, sh, t48, 0])[m])),
                           dlogp_True_ci=boot(LP[:, c, t48, 0] - LP[:, sh, t48, 0], m),
                           dlogp_False=float(np.median((LP[:, c, t48, 1] - LP[:, sh, t48, 1])[m])))
        out["formats"][f] = res
    F = out["formats"]; interp = [f for f in F if F[f]["interpretable"]]
    tok_ev = sum(F[f]["dstar+"]["dR_tok_ci"][0] > 0 for f in interp if f != "tf")
    sem_fmts = [f for f in ("yn", "ab") if f in interp]
    sem_ev = bool(sem_fmts) and all(F[f]["dstar+"]["dR_sem_ci"][0] > 0 for f in sem_fmts)
    if "rev" not in interp or len(interp) < 3:
        verdict = "INCONCLUSIVE"
    elif F["rev"]["dstar+"]["dR_sem_ci"][1] < 0:
        verdict = "TOKEN-DOMINANT"
    elif F["rev"]["dstar+"]["dR_sem_ci"][0] > 0:
        verdict = "SEMANTIC-DOMINANT"
    else:
        verdict = "NEITHER-DOMINANT"
    out["verdict"] = verdict
    out["token_evidence"] = dict(n_nontf_formats_tok_pos=int(tok_ev), holds=bool(tok_ev >= 2))
    out["semantic_evidence"] = dict(formats=sem_fmts, holds=bool(sem_ev))
    # dose / pulse / removal
    zd = np.load(f"{rd}/ctrl.npz"); rowsd = json.load(open(f"{rd}/rows.json"))
    assert zd["done"].all() and np.isfinite(zd["LP"]).all(), "incomplete dose run"
    cd = [json.loads(c)[0] for c in zd["conds"]]; ROd = list(zd["ro"]); LPd = zd["LP"].astype(np.float64)
    Rd = LPd[:, :, :, 0] - LPd[:, :, :, 1]; shd = cd.index("sham")
    zv = np.load(f"{rv}/dstar.npz"); vit = json.load(open(f"{rv}/items.json"))
    a_by_uid = {it["uid"]: zv["A"][k].astype(np.float64) for k, it in enumerate(vit)}
    u = np.load("results/ctrl_dirs.npz")["u_dstar"].astype(np.float64)
    pred = np.array([a_by_uid[r["uid"]] @ u for r in rowsd])
    dose = {}
    for cn, nm in (("dstar+_1.5625", 1.5625), ("dstar+_3.125", 3.125), ("dstar+_6.25", 6.25)):
        c = cd.index(cn); obs = (Rd[:, c, ROd.index(48)] - Rd[:, shd, ROd.index(48)]) / nm
        dose[nm] = dict(median_obs_per_unit=float(np.median(obs)), median_pred_per_unit=float(np.median(pred)),
                        median_ratio=float(np.median(obs / pred)), spearman=float(spearmanr(obs, pred)[0]))
    # full dose (12.5) from the formats run, matched by uid (tf rows)
    tf_idx = {int(uid[k]): k for k in range(len(rows)) if fm[k] == "tf"}
    cplus, csh = conds.index("dstar+"), conds.index("sham")
    obs125 = np.array([(LP[tf_idx[r["uid"]], cplus, t48, 0] - LP[tf_idx[r["uid"]], cplus, t48, 1]
                        - (LP[tf_idx[r["uid"]], csh, t48, 0] - LP[tf_idx[r["uid"]], csh, t48, 1])) / 12.5 for r in rowsd])
    dose[12.5] = dict(median_obs_per_unit=float(np.median(obs125)), median_pred_per_unit=float(np.median(pred)),
                      median_ratio=float(np.median(obs125 / pred)), spearman=float(spearmanr(obs125, pred)[0]))
    out["dose_tf"] = dose
    cp = cd.index("pulse+"); cr = cd.index("remove+")
    pe = {int(t): float(np.median(Rd[:, cp, ROd.index(t)] - Rd[:, shd, ROd.index(t)])) for t in (0, 1, 4, 48) if t in ROd}
    e24 = float(np.median(Rd[:, cr, ROd.index(24)] - Rd[:, shd, ROd.index(24)])); e48 = float(np.median(Rd[:, cr, ROd.index(48)] - Rd[:, shd, ROd.index(48)]))
    out["pulse_tf"] = pe
    out["removal_tf"] = dict(E24=e24, E48=e48, by_offset={int(t): float(np.median(Rd[:, cr, ROd.index(t)] - Rd[:, shd, ROd.index(t)])) for t in ROd},
                             median_abs_item_residual_48=float(np.median(np.abs(Rd[:, cr, ROd.index(48)] - Rd[:, shd, ROd.index(48)]))),
                             status=("INCONCLUSIVE" if e24 < 0.5 else ("RECOVERS" if abs(e48) <= 0.1 * e24 else "RESIDUAL")))
    json.dump(out, open(f"{rf.split(',')[-1]}/analysis_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3].rstrip("/"))
