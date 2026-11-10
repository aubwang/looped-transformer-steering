"""Run the exploratory learning curve (mon_learning_curve.py, docs/mon_exploratory.md E1) on one Modal CPU container.

Nothing here computes anything: the container calls the script as a subprocess. The collected arrays are read from the
volume the collection wrote them to; code, design documents, item list, manifest and the registered seed-0 results
(for the script's built-in reproduction check) are copied into the image at their repository paths.

    modal run huginn_addition/mon_learning_curve_app.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
COUNT = "results/mon_count-20261109-203859"
OUT = "mon_learning_curve-20261110"
RUNS = ["mon_main-20261110-" + t for t in ("005610", "010355", "011425", "011555", "012315", "013047", "013810", "014801")]
FILES = ["huginn_addition/mon_learning_curve.py", "huginn_addition/analyze_mon.py", "huginn_addition/mon_probe.py",
         "docs/oversight_monitor.md", "docs/mon_spec.md", "docs/mon_exploratory.md",
         COUNT + "/items_master.json", COUNT + "/main_manifest.json", "results/mon_analysis-20261110/seed0/results.json"]

app = modal.App("oversight-monitor-learning-curve")
results_vol = modal.Volume.from_name("huginn-results")
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.5.4", "scipy==1.18.1")
if modal.is_local():
    for rel in FILES:
        image = image.add_local_file(str(ROOT / rel), "/proj/" + rel)


@app.function(image=image, volumes={"/results": results_vol}, cpu=16, memory=24576, timeout=3600)
def curve(seed: int, jobs: int, out_name: str):
    import os
    import platform
    import resource
    out = f"/results/{OUT}/{out_name}"
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-I", "/proj/huginn_addition/mon_learning_curve.py", "--runs", *[f"/results/{r}" for r in RUNS],
           "--items", f"/proj/{COUNT}/items_master.json", "--manifest", f"/proj/{COUNT}/main_manifest.json",
           "--out", out, "--seed", str(seed), "--jobs", str(jobs)]
    start, last = time.time(), ""
    with open(f"{out}/run.log", "w") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd="/proj")
        for line in p.stdout:
            log.write(line); log.flush()
            last = line
        rc = p.wait()
    info = {"out": out_name, "seed": seed, "jobs": jobs, "returncode": rc, "seconds": time.time() - start,
            "peak_child_rss_mb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, "cpu_count": os.cpu_count(),
            "python": platform.python_version(), "last_line": last.strip(), "command": cmd}
    with open(f"{out}/container.json", "w") as f:
        json.dump(info, f, indent=1)
    results_vol.commit()
    return info


@app.local_entrypoint()
def main(seed: int = 0, jobs: int = 32):
    print(json.dumps(curve.remote(seed, jobs, f"seed{seed}"), indent=1), flush=True)
