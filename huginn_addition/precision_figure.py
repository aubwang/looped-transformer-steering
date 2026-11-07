"""bf16 vs fp32 precision control figure.  Usage: python -I huginn_addition/precision_figure.py <bf16_run> <fp32_run> <out.png>"""
import glob, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SURF, INK2, GRID = "#fcfcfb", "#52514e", "#e4e3df"
C_BF, C_FP = "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
                     "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False})
NORM = 76.36  # raw state norm is pinned by the core block's final RMSNorm


def cat(run, name):
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{run}/{name}_c*.parquet"))], ignore_index=True)


bf, fp, outp = sys.argv[1:4]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for run, c, lab in ((bf, C_BF, "bfloat16"), (fp, C_FP, "float32")):
    st = cat(run, "steps"); st["rel"] = st["delta"] / st["norm"]
    pr = cat(run, "pairs"); pr["rel"] = pr["dist"] / NORM
    for ax, df in ((axes[0], st), (axes[1], pr)):
        q = df.groupby("step")["rel"].quantile([0.1, 0.5, 0.9]).unstack().clip(lower=1e-7)
        ax.plot(q.index, q[0.5], color=c, label=lab)
        ax.fill_between(q.index, q[0.1], q[0.9], color=c, alpha=0.15, lw=0)
for ax, t in ((axes[0], "State change per step, ‖s_k − s_{k−1}‖/‖s_k‖"), (axes[1], "Distance between random starts (relative)")):
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("recurrent iterations (log)"); ax.set_title(t)
axes[0].legend(loc="lower left")
st = cat(fp, "steps"); st["rel"] = st["delta"] / st["norm"]
for uid, c, lab in ((26, C_FP, "1851 + 6224, demo set A (wrong digit)"), (27, "#1baf7a", "same problem, demo set B")):
    s = st[(st["uid"] == uid) & (st["seed"] == 0) & (st["pos"] == 0)].sort_values("step")
    axes[2].plot(s["step"], s["rel"].clip(lower=1e-7), color=c, label=lab)
axes[2].set_yscale("log"); axes[2].set_xlabel("recurrent iterations"); axes[2].legend(loc="lower left", fontsize=8)
axes[2].set_title("fp32 exception (1 of 42 prompts): growing\nperiod-2 component (first answer digit, seed 0)")
fig.suptitle("Precision control: the bf16 ~1.5% floor is precision-sensitive; in fp32 most answer-position states approach stationarity\n(bands are 10–90%; rare non-converging trajectories are hidden by them — see right panel)",
             fontsize=11, fontweight="bold")
fig.tight_layout(); fig.savefig(outp, dpi=150)
