"""Pinned-prefix validation of census modes (intervene_app.py `ownblock`).
Usage: python -I huginn_addition/analyze_ownblock.py results/<ownblock_run>
Per row: own-block mode from the symmetric push response (same Gram estimator as the census; latest window whose mean
|r| > 10x floor 1e-6 relative; per direction x amplitude; agreement requires the 2 directions to give the same class and
|lambda| within 0.02, and the two amplitudes' responses to scale ~10x); Arnoldi leading Ritz value of J_pp at x* with
complex-aware FD residual (eps 0.1, 0.05). Compared with the census trajectory mode of that position.
"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from analyze_census import fit_window  # noqa: E402


def main(run):
    z = np.load(f"{run}/ownblock.npz", allow_pickle=True)
    probs = json.load(open(f"{run}/problems.json"))
    rows, G, N = z["push_rows"], z["push_gram"], z["push_norms"]
    W = z["push_windows"]
    out = []
    for r, p in enumerate(probs):
        fits = []
        for k, (rr, di, ai) in enumerate(rows):
            if rr != r:
                continue
            el = [wi for wi, (a, b) in enumerate(W) if N[k, a:b].mean() > 10 * 1e-6 * (1 if ai == 0 else 0.1) and G[k, wi, 6] >= 3]
            m = fit_window(G[k, el[-1]]) if el else None
            fits.append((di, ai, m, float(N[k, -1] / N[k, 0])))
        main_fits = [f for f in fits if f[1] == 0]  # amplitude 1e-3
        cls = [f[2]["cls"] if f[2] else "none" for f in main_fits]
        mods = [f[2].get("lam", f[2].get("mod")) if f[2] and "mod" in f[2] else None for f in main_fits]
        agree = len(set(cls)) == 1 and None not in mods and abs(abs(mods[0]) - abs(mods[1])) < 0.02
        lin = [f[3] for f in fits]
        H = z["arn_H"][r]; dn = int(z["arn_dim"])
        lam = np.linalg.eigvals(H[:dn, :dn]); lam = lam[np.argsort(-np.abs(lam))]
        lead = z["arn_lead"][r]
        cm = p["census_mode"]
        out.append(dict(i=r, uid=p["uid"], pos=p["val_pos"], tok=p["val_tok"], role=p["val_role"], sample=p["val_sample"],
                        fp_valid=bool(z["fp_valid"][r]), census=(cm.get("cls"), round(cm.get("lam", cm.get("mod", np.nan)), 3)),
                        own_push=[(c, None if mo is None else round(mo, 3)) for c, mo in zip(cls, mods)], own_push_agree=agree,
                        arn_lead=(round(lead.real, 3), round(lead.imag, 3)), arn_lead_abs=round(abs(lead), 3),
                        arn_fd_resid=np.round(z["arn_lead_fd_resid"][r], 4).tolist(),
                        arn_top_real_neg=[round(x.real, 3) for x in lam[:8] if x.real < 0 and abs(x.imag) < 1e-6][:2],
                        response_decay_48=[round(x, 4) for x in lin]))
    for o in out:
        print(o)
    json.dump(out, open(f"{run}/analysis_summary.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
