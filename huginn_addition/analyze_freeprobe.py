"""Random-probe test of the free-loop model (docs/freeloop.pdf, Section 5 of Draft 2 (Section 6 of Draft 1); run intervene_app.py `freeprobe`).
Prespecified before the run.
Usage: python -I huginn_addition/analyze_freeprobe.py results/<freeprobe_run> <dstar_vjp npz> [smoke]
(<freeprobe_run> holds freeprobe.npz, meta.json, items.json; the dstar_vjp npz is
dstar_vjp-20261106-061708/dstar.npz, used only for the cross-run gate.)

Data: 40 truth items (every 3rd of results/dstar_items.json), at x* of each the unit readout gh and K = 4 unit Gaussian
probes z; for each start u the chain w_0 = u, w_t = J^T w_{t-1}, t <= T = 48, saved as the Gram matrix of all chain
vectors. Statistics per start (unit start, lever a = sum_{t=1..T} w_t, i.e. the note's a_T / |g|):
  alpha = <u, w_1>, s_u^2 = |w_1|^2 - alpha^2, Gamma = |a|, x = <u, a>, c0 = x / |a|, c1 = <w_1, a> / |a|,
  kappa = c1 / |w_1|, sigma = 1 / (1 + x), psi = x sigma, E = sigma (<w_1, a> - alpha x), 2C = psi - alpha - E.
Loop estimates per item from the probes: gamma_hat = mean alpha(z), s_hat^2 = mean |J^T z|^2 - gamma_hat^2.
Theorem 5 (Draft 2 numbering; Theorem 2 in Draft 1) predictions (T = infinity) from (gamma, s) and a start's own (lambda = alpha, s_g = s_u):
  F^2 = s_g^2 / ((1 - gamma)^2 - s^2), D = sqrt(lambda^2 + F^2), c0 = sgn(1 - lambda) lambda / D, Gamma = D / |1 - lambda|,
  c1 = sgn(1 - lambda) (lambda^2 + s_g^2 / (1 - gamma)) / D, kappa = c1 / sqrt(lambda^2 + s_g^2), E = s_g^2 / (1 - gamma).
Free norm profile for a probe (beta = 0): |w_t|^2 = gamma^(2t) + sum_{k=1..t} binom(t, k)^2 gamma^(2(t-k)) s_u^2 s^(2(k-1)).

Only items marked done are analysed. GATES (validity; all must hold, else no conclusions): at least 32 of the 40 items
  done (digits: 16 of 20) (any number for the smoke run, which stops after one chunk by design); T >= 8; starts have unit norm (|G_00 - 1| < 1e-5);
  Gram symmetric (relative 1e-10); adjoint check median < 5e-3 over all starts; tail max |w_T| / |a| < 1e-3;
  cross-run identity with dstar_vjp on the same uids: |g - G_dstar| / |G_dstar| < 1e-4 and
  | |g| a_readout - A_dstar | / |A_dstar| < 1e-3 for every item.
P1 LOOP-LEVEL FREENESS (probes; gamma, s from the OTHER K - 1 probes of the item, alpha and s_u from the probe itself):
  P1a cancellation: median |(E + 2C) / E| <= 0.10.
  P1b lever: median |log(Gamma / Gamma_pred)| <= 0.10, median |c0 - c0_pred| <= 0.03, median |kappa - kappa_pred| <= 0.05.
  P1c norm profile: median over probes of max_{t in {2, 4, 8}} |log(|w_t| / |w_t|_pred)| <= 0.10.
  P1d exchangeability: median over items of the within-item sd (across probes) of kappa <= 0.03 and of log Gamma <= 0.05.
  P1e no shared direction: median |cos(a^(z), a^(z'))| over probe pairs of the same item <= 0.10.
  Verdict: loop-level freeness SUPPORTED iff P1a-P1e all pass; otherwise NOT SUPPORTED (reported per criterion).
P2 READOUT GIVEN THE LOOP (gamma, s from all K probes; lambda = alpha, s_g = s_u of the readout): median |log(Gamma /
  Gamma_pred)| <= 0.10 and median |kappa - kappa_pred| <= 0.05 (PASS / FAIL); Spearman and median errors of c0, c1, Gamma,
  kappa reported, next to the gamma = 0 prediction with s from the probes.
Q3 SELF-GAIN SPLIT: gamma_hat and beta_hat = alpha_readout - gamma_hat, medians with item-bootstrap 95% CIs (5000
  resamples, seed 0). Reading: "loop-wide" if the CI of median beta_hat contains 0 and that of median gamma_hat
  excludes 0; "readout-specific" if the CI of median beta_hat excludes 0 and median gamma_hat < alpha_median / 2;
  otherwise "mixed". Descriptive: bulk outer radius |gamma_hat| + s_hat (spectral radius of the free model's bulk; a
  rank-one outlier can exceed it) vs the ownspec values (median 0.77, range 0.66-0.87, other prompts).
STATIONARITY: the Jacobian is that of the map at the 64-step state, as in dstar_vjp (drift = |F(x) - x|_q / |x_q| is
  reported). Sensitivity: P1 and P2 statistics recomputed on items with drift <= 1e-3 (reported, not a gate).
Tail diagnostic (reported): per start, |w_T| / |a_T| and the geometric-tail estimate |w_T| r / (1 - r) / |a_T| with r the
  mean ratio |w_t| / |w_{t-1}| over t = T - 7..T.
DIGITS (cfg task = "digits", specified before that run): the same analysis with NR = len(digits_k) readouts per item
  (starts m < NR) and probes m >= NR; the cross-run gate uses the dglever_train npz (G[:, k], AtG[:, k], k in digits_k);
  P2 and Q3 pool the NR readouts, and the Q3 bootstrap resamples items (clusters).
SIGNATURE (descriptive; added after the truth run, before the digits run): per item, probe means of the trace moments
  m_k = <z, (J^T)^k z> (k = 1..8; m_k -> tr(J^k)/d for random z), the centered second moment kappa_2 = m_2 - m_1^2, the
  per-step norm ratio |w_t| / |w_{t-1}| and lag-one cosine cos(w_t, w_{t+1}) of the probe chains, the random-direction
  steering advantage 1/|c0(z)|, and for each readout the excess alpha - gamma_hat and advantage ratio
  (1/|c0(g)|) / (1/|c0(z)|).
REPLICATION OF READOUT SUPPRESSION (specified 2026-11-07 while the digits production run was in flight, before any
  digits production data were seen; computed in docs/freeloop_check.py Part D): the truth run found alpha_readout <
  gamma_hat for 40/40 items (median excess -0.12). For digits, use all 10 centered-digit readouts per item at the same
  states (alpha = <gh, J^T gh> from the existing adaptcov / dglever_train arrays; the freeprobe gate certifies the same
  x*) and gamma_hat from the digits probes. REPLICATED iff the median of alpha - gamma_hat over the 200 readouts is < 0
  with an item-cluster bootstrap 95% CI (5000 resamples, seed 0) excluding 0. Also reported: the fraction of readouts
  with alpha < gamma_hat, and the advantage ratio (1/|c0(g)|) / (1/|c0(z)|) with the same CI.
Writes <freeprobe_run>/analysis_summary.json.
"""
import json
import sys
from math import comb

import numpy as np
from scipy.stats import spearmanr


def start_stats(Gm, m, T):
    """Statistics of start m from one item's Gram matrix (index v = m (T + 1) + t)."""
    o = m * (T + 1)
    B = Gm[o:o + T + 1, o:o + T + 1]
    alpha = B[0, 1]; n1 = B[1, 1]
    a2 = B[1:, 1:].sum(); x = B[0, 1:].sum(); w1a = B[1, 1:].sum()
    Ga = np.sqrt(a2); sig = 1 / (1 + x); psi = x * sig; E = sig * (w1a - alpha * x)
    return dict(alpha=alpha, n1=n1, su2=n1 - alpha ** 2, Gamma=Ga, c0=x / Ga, c1=w1a / Ga, kappa=w1a / Ga / np.sqrt(n1),
                sigma=sig, psi=psi, E=E, C2=psi - alpha - E, norms=np.sqrt(np.diag(B)[1:]), moments=B[0, 1:])


def thm2(lam, sg2, gam, s2):
    F2 = sg2 / ((1 - gam) ** 2 - s2)
    D = np.sqrt(lam ** 2 + F2); sg = np.sign(1 - lam)
    c1 = sg * (lam ** 2 + sg2 / (1 - gam)) / D
    return dict(F2=F2, D=D, c0=sg * lam / D, Gamma=D / abs(1 - lam), c1=c1, kappa=c1 / np.sqrt(lam ** 2 + sg2),
                E=sg2 / (1 - gam), adv=np.sqrt(1 + F2 / lam ** 2))


def norm_pred(t, gam, s2, su2):
    return gam ** (2 * t) + sum(comb(t, k) ** 2 * gam ** (2 * (t - k)) * su2 * s2 ** (k - 1) for k in range(1, t + 1))


def boot_median(v, B=5000, seed=0):
    """Median with a bootstrap CI; rows (items) are resampled, so a 2-D (items x readouts) array is cluster-resampled."""
    v = np.asarray(v); rng = np.random.default_rng(seed)
    b = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(B)]
    return [float(np.median(v)), float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def q(v):
    v = np.asarray(v, float)
    return [float(np.median(v)), float(np.quantile(v, .1)), float(np.quantile(v, .9))]


def main(run, dstar, min_done=None):
    z = np.load(f"{run}/freeprobe.npz"); meta = json.load(open(f"{run}/meta.json"))
    cfg = meta["cfg"]; T, K = cfg["T"], cfg["K"]
    task = cfg.get("task", "truth"); DK = cfg.get("digits_k", [None]); NR = len(DK); S = NR + K
    if min_done is None:  # 80% of the planned items: truth 32 of 40, digits 16 of 20
        min_done = 32 if task == "truth" else 16
    done = z["done"]
    GR, uid, aT, Gz = z["gram"][done], z["uid"][done], z["aT"][done], z["G"][done]
    if Gz.ndim == 2:  # truth runs before the digits generalization saved G as (n, d)
        Gz = Gz[:, None]
    drift, adj = z["drift"][done], z["adjoint_rel_err"][done]
    out = dict(run=run, n_items=int(len(done)), n_done=int(done.sum()), T=T, K=K, min_done=min_done)
    # ---------------------------------------------------------------- gates
    gates = {}
    gates["n_done"] = int(done.sum())
    gates["T"] = T
    diag0 = np.array([[GR[i, m * (T + 1), m * (T + 1)] for m in range(S)] for i in range(len(uid))])
    gates["unit_starts"] = float(np.abs(diag0 - 1).max())
    gates["gram_sym"] = float(max(np.abs(Gm - Gm.T).max() / np.abs(Gm).max() for Gm in GR))
    gates["adjoint_median"] = float(np.median(adj))
    gates["adjoint_max"] = float(np.max(adj))
    tails, gtails = [], []
    for i in range(len(uid)):
        for m in range(S):
            nm = start_stats(GR[i], m, T)["norms"]; Ga = start_stats(GR[i], m, T)["Gamma"]
            r = float(np.mean(nm[T - 8:T] / nm[T - 9:T - 1]))
            tails.append(nm[T - 1] / Ga); gtails.append(nm[T - 1] * r / (1 - r) / Ga if r < 1 else np.inf)
    gates["tail_max"] = float(np.max(tails))
    out["tail"] = dict(last_over_lever=q(tails), geometric_tail_over_lever=q(gtails), geometric_tail_max=float(np.max(gtails)))
    D = np.load(dstar); idx = {u: k for k, u in enumerate(D["uid"])}
    assert all(u in idx for u in uid)
    kk = np.array([idx[u] for u in uid])
    if task == "truth":
        refG, refA = D["G"][kk][:, None], D["A"][kk][:, None]
    else:
        refG, refA = D["G"][kk][:, DK], D["AtG"][kk][:, DK]
    gd = np.linalg.norm(Gz - refG, axis=-1) / np.linalg.norm(refG, axis=-1)
    aR = aT[:, :NR] * np.linalg.norm(Gz, axis=-1)[..., None]
    ad = np.linalg.norm(aR - refA, axis=-1) / np.linalg.norm(refA, axis=-1)
    gates["g_vs_dstar_max"] = float(gd.max()); gates["a_vs_dstar_max"] = float(ad.max())
    ok = (gates["n_done"] >= max(min_done, 1) and T >= 8 and gates["unit_starts"] < 1e-5 and gates["gram_sym"] < 1e-10 and gates["adjoint_median"] < 5e-3
          and gates["tail_max"] < 1e-3 and gates["g_vs_dstar_max"] < 1e-4 and gates["a_vs_dstar_max"] < 1e-3)
    gates["PASS"] = bool(ok)
    out["gates"] = gates
    print("GATES", json.dumps(gates, indent=1))
    if not ok:
        print("GATES FAILED: no conclusions"); json.dump(out, open(f"{run}/analysis_summary.json", "w"), indent=1); return out
    out["drift"] = dict(q=q(drift), n_above_1e3=int((drift > 1e-3).sum()), uids_above_1e3=[int(u) for u in uid[drift > 1e-3]])
    res = evaluate(GR, aT, T, K, NR)
    out.update(res)
    for k in ("P1", "P2", "Q3", "SIGNATURE"):
        print(k, json.dumps(res[k], indent=1))
    keep = drift <= 1e-3
    if keep.sum() >= 2 and not keep.all():
        sens = evaluate(GR[keep], aT[keep], T, K, NR)
        out["sensitivity_drift_le_1e-3"] = dict(n=int(keep.sum()), P1={k: v for k, v in sens["P1"].items() if k != "descr"},
                                               P2={k: sens["P2"][k] for k in ("c0", "c1", "Gamma", "kappa", "pass")})
        print("SENSITIVITY (drift <= 1e-3)", json.dumps(out["sensitivity_drift_le_1e-3"], indent=1))
    json.dump(out, open(f"{run}/analysis_summary.json", "w"), indent=1)
    return out


def evaluate(GR, aT, T, K, NR=1):
    S = NR + K
    st = [[start_stats(GR[i], m, T) for m in range(S)] for i in range(len(GR))]
    n = len(GR)
    keys = ("alpha", "n1", "su2", "Gamma", "c0", "c1", "kappa", "E", "C2", "psi")
    P = {k: np.array([[st[i][m][k] for m in range(NR, S)] for i in range(n)]) for k in keys}
    R2 = {k: np.array([[st[i][m][k] for m in range(NR)] for i in range(n)]) for k in keys}   # (n, NR)
    R = {k: v.ravel() for k, v in R2.items()}
    gam = P["alpha"].mean(1); s2 = P["n1"].mean(1) - gam ** 2
    # ---------------------------------------------------------------- P1 (leave-one-probe-out)
    pr = {k: np.zeros((n, K)) for k in ("Gamma", "c0", "kappa", "c1", "E")}
    prof = np.zeros((n, K))
    for i in range(n):
        for m in range(K):
            oth = [j for j in range(K) if j != m]
            g_ = P["alpha"][i, oth].mean(); s2_ = P["n1"][i, oth].mean() - g_ ** 2
            p = thm2(P["alpha"][i, m], P["su2"][i, m], g_, s2_)
            for k in pr:
                pr[k][i, m] = p[k]
            nm = st[i][NR + m]["norms"]
            prof[i, m] = max(abs(np.log(nm[t - 1] / np.sqrt(norm_pred(t, g_, s2_, P["su2"][i, m])))) for t in (2, 4, 8))
    canc = np.abs((P["E"] + P["C2"]) / P["E"])
    cosab = []
    for i in range(n):
        A_ = aT[i, NR:].astype(np.float64); A_ /= np.linalg.norm(A_, axis=1, keepdims=True)
        Cm = A_ @ A_.T; cosab += list(np.abs(Cm[np.triu_indices(K, 1)]))
    p1 = dict(
        P1a_cancel=float(np.median(canc)),
        P1b_logGamma=float(np.median(np.abs(np.log(P["Gamma"] / pr["Gamma"])))),
        P1b_c0=float(np.median(np.abs(P["c0"] - pr["c0"]))),
        P1b_kappa=float(np.median(np.abs(P["kappa"] - pr["kappa"]))),
        P1c_profile=float(np.median(prof)),
        P1d_sd_kappa=float(np.median(P["kappa"].std(1, ddof=1))),
        P1d_sd_logGamma=float(np.median(np.log(P["Gamma"]).std(1, ddof=1))),
        P1e_cos=float(np.median(cosab)))
    passes = dict(P1a=p1["P1a_cancel"] <= 0.10,
                  P1b=p1["P1b_logGamma"] <= 0.10 and p1["P1b_c0"] <= 0.03 and p1["P1b_kappa"] <= 0.05,
                  P1c=p1["P1c_profile"] <= 0.10,
                  P1d=p1["P1d_sd_kappa"] <= 0.03 and p1["P1d_sd_logGamma"] <= 0.05,
                  P1e=p1["P1e_cos"] <= 0.10)
    p1["pass"] = {k: bool(v) for k, v in passes.items()}
    p1["verdict"] = "SUPPORTED" if all(passes.values()) else "NOT SUPPORTED"
    p1["descr"] = dict(probe_kappa=q(P["kappa"]), probe_Gamma=q(P["Gamma"]), probe_c0=q(P["c0"]), probe_alpha=q(P["alpha"]),
                       probe_E=q(P["E"]), probe_cancel_signed=q((P["E"] + P["C2"]) / P["E"]),
                       profile_err=q(prof), cos_probe_levers=q(cosab))
    out = dict(P1=p1)
    # ---------------------------------------------------------------- P2 (readout given loop)
    gamR, s2R = np.repeat(gam, NR), np.repeat(s2, NR)
    p = thm2(R["alpha"], R["su2"], gamR, s2R)
    p0 = thm2(R["alpha"], R["su2"], 0 * gamR, s2R)
    p2 = {}
    for k in ("c0", "c1", "Gamma", "kappa"):
        p2[k] = dict(spearman=float(spearmanr(R[k], p[k])[0]),
                     med_err=float(np.median(np.abs(np.log(R[k] / p[k])) if k == "Gamma" else np.abs(R[k] - p[k]))),
                     med_err_gamma0=float(np.median(np.abs(np.log(R[k] / p0[k])) if k == "Gamma" else np.abs(R[k] - p0[k]))))
    p2["pass"] = bool(p2["Gamma"]["med_err"] <= 0.10 and p2["kappa"]["med_err"] <= 0.05)
    p2["readout_cancel_signed"] = q((R["E"] + R["C2"]) / R["E"])
    p2["adv_pred"] = q(p["adv"]); p2["adv_obs"] = q(1 / np.abs(R["c0"]))
    out["P2"] = p2
    # ---------------------------------------------------------------- Q3
    beta = R2["alpha"] - gam[:, None]
    q3 = dict(gamma_hat=boot_median(gam), beta_hat=boot_median(beta), alpha_readout=boot_median(R2["alpha"]),
              s_hat=q(np.sqrt(s2)), bulk_outer_radius=q(np.abs(gam) + np.sqrt(s2)))
    bci, gci = q3["beta_hat"], q3["gamma_hat"]
    if bci[1] <= 0 <= bci[2] and not (gci[1] <= 0 <= gci[2]):
        q3["reading"] = "loop-wide"
    elif not (bci[1] <= 0 <= bci[2]) and gci[0] < q3["alpha_readout"][0] / 2:
        q3["reading"] = "readout-specific"
    else:
        q3["reading"] = "mixed"
    out["Q3"] = q3
    # ---------------------------------------------------------------- SIGNATURE (descriptive)
    mom = np.array([[[GR[i, m * (T + 1), m * (T + 1) + k] for k in range(1, 9)] for m in range(NR, S)] for i in range(n)]).mean(1)
    ratio, lag1 = [], []
    for i in range(n):
        for m in range(NR, S):
            o = m * (T + 1); B = GR[i, o:o + T + 1, o:o + T + 1]; nm = np.sqrt(np.diag(B))
            ratio.append(nm[2:11] / nm[1:10]); lag1.append([B[t, t + 1] / (nm[t] * nm[t + 1]) if t < T else np.nan for t in (1, 2, 3, 5, 10, 20)])
    adv_r = 1 / np.abs(P["c0"]).mean(1)                  # per item: random-direction advantage (probe mean of |c0|)
    adv_g = 1 / np.abs(R2["c0"])                          # (n, NR)
    out["SIGNATURE"] = dict(
        trace_moments_m1_to_m8=[q(mom[:, k]) for k in range(8)], kappa2=q(mom[:, 1] - mom[:, 0] ** 2),
        norm_ratio_t2_to_t10=[float(v) for v in np.median(np.array(ratio), 0)],
        lag1_cos_t_1_2_3_5_10_20=[float(v) for v in np.nanmedian(np.array(lag1, float), 0)],
        random_adv=q(adv_r), readout_adv=q(adv_g), adv_ratio=q(adv_g / adv_r[:, None]),
        alpha_excess=q(R2["alpha"] - gam[:, None]), frac_alpha_below_gamma=float(np.mean(R2["alpha"] < gam[:, None])),
        random_Gamma=q(P["Gamma"]), readout_Gamma=q(R2["Gamma"]), random_c0=q(P["c0"]), readout_c0=q(R2["c0"]))
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], min_done=1 if len(sys.argv) > 3 and sys.argv[3] == "smoke" else None)
