"""Synthetic calibration of the pre-registered random-probe test at Huginn's dimension (d = 5280, CPU only).
Runs analyze_freeprobe.py on fake `freeprobe` outputs built by synth_freeprobe.py and writes
results/synth_calibration/calibration.json, which docs/freeloop_check.py reads to assert the calibration paragraph
of docs/freeloop.typ (Section 5).
Usage: python -I huginn_addition/calibrate_freeprobe.py [--out /tmp/synth_cal] [--only name]
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_freeprobe as af  # noqa: E402
import synth_freeprobe as sf  # noqa: E402

D = 5280
# name: (kind, n contexts, T, param, number of readouts)
CONFIGS = {
    "free": ("free", 40, 48, 0.0, 1),            # the truth design: 40 contexts, 1 readout, K = 4 probes, T = 48
    "symmetric": ("symmetric", 10, 48, 0.0, 1),
    "cyclic": ("cyclic", 10, 48, 0.5, 1),
    "slowmode": ("slowmode", 10, 250, 0.8, 1),   # longer T: the slow mode decays like 0.9^t
    "block": ("block", 10, 48, 0.0, 1),
}
THRESH = dict(P1a_cancel=0.10, P1b_logGamma=0.10, P1b_c0=0.03, P1b_kappa=0.05, P1c_profile=0.10,
              P1d_sd_kappa=0.03, P1d_sd_logGamma=0.05, P1e_cos=0.10)


def run(name, out_root):
    kind, n, T, param, nr = CONFIGS[name]
    out = f"{out_root}/{name}_d{D}_n{n}_T{T}"
    t0 = time.time()
    sf.make(kind, D, n, 4, T, 1, param, out, nr=nr)
    with contextlib.redirect_stdout(io.StringIO()):
        r = af.main(out, f"{out}/dstar.npz", min_done=1)
    p1, p2 = r["P1"], r["P2"]
    stats = {k: p1[k] for k in THRESH}
    return dict(kind=kind, d=D, n=n, T=T, param=param, nr=nr, seed=1, K=4, gates=r["gates"]["PASS"],
                P1_stats=stats, P1_margin={k: THRESH[k] / max(v, 1e-12) for k, v in stats.items()},
                P1_pass=p1["pass"], P1_verdict=p1["verdict"], P2_pass=p2["pass"],
                P2_logGamma_err=p2["Gamma"]["med_err"], P2_kappa_err=p2["kappa"]["med_err"],
                P2_c0_err=p2["c0"]["med_err"], seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/synth_cal"); ap.add_argument("--only", default=None)
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dest = f"{root}/results/synth_calibration/calibration.json"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    res = json.load(open(dest)) if os.path.exists(dest) else {}
    for name in ([a.only] if a.only else CONFIGS):
        res[name] = run(name, a.out)
        print(name, json.dumps(res[name]), flush=True)
        json.dump(res, open(dest, "w"), indent=1)
