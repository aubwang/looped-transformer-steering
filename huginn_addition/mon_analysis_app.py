"""Run the registered monitor analysis (analyze_mon.py, unchanged) on Modal CPU containers, one container per seed.

Nothing here computes anything: each container calls the frozen analysis script as a subprocess with the same arguments
as the local run. The collected arrays are read from the volume the collection wrote them to. The analysis code, the
design documents, the item list and the manifest are copied into the image at the same relative paths as in the
repository, so the script's own hash check against the manifest applies unchanged. numpy and scipy are pinned to the
versions of the local environment (the analysis imports nothing else).

`--only` is used only to start the long tasks first (the tasks are independent, each with its own seeded generator).

    modal run --detach huginn_addition/mon_analysis_app.py --seeds 1,2
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
COUNT = "results/mon_count-20261109-203859"
OUT = "mon_analysis-20261110"
RUNS = ["mon_main-20261110-" + t for t in ("005610", "010355", "011425", "011555", "012315", "013047", "013810", "014801")]
FILES = ["huginn_addition/analyze_mon.py", "huginn_addition/mon_probe.py", "docs/oversight_monitor.md", "docs/mon_spec.md",
         COUNT + "/items_master.json", COUNT + "/main_manifest.json"]
LONG_FIRST = "m3a,m1c,m1s,m1b,m1a,t1,m1x,f16,m4,dyn"      # all ten analyses, longest first
CHECK = "m1a,t1,m4,f16,dyn"                                  # cheap subset, to compare seed 0 with the local run

app = modal.App("oversight-monitor-analysis")
results_vol = modal.Volume.from_name("huginn-results")
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.5.4", "scipy==1.18.1")
if modal.is_local():
    for rel in FILES:
        image = image.add_local_file(str(ROOT / rel), "/proj/" + rel)


def run_analysis(seed, only, jobs, out_name):
    import os
    import platform
    import resource
    import numpy
    import scipy
    out = f"/results/{OUT}/{out_name}"
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-I", "/proj/huginn_addition/analyze_mon.py", "--runs", *[f"/results/{r}" for r in RUNS],
           "--items", f"/proj/{COUNT}/items_master.json", "--manifest", f"/proj/{COUNT}/main_manifest.json",
           "--out", out, "--seeds", str(seed), "--jobs", str(jobs), "--only", only]
    start = time.time()
    with open(f"{out}/run.log", "w") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd="/proj")
        for line in p.stdout:
            print(f"[{out_name} {time.time() - start:6.0f}s] {line}", end="", flush=True)
            log.write(line); log.flush()
        rc = p.wait()
    info = {"out": out_name, "seed": seed, "only": only, "jobs": jobs, "returncode": rc, "seconds": time.time() - start,
            "peak_child_rss_mb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, "cpu_count": os.cpu_count(),
            "python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "processor": platform.processor() or platform.machine(), "command": cmd}
    with open(f"{out}/container.json", "w") as f:
        json.dump(info, f, indent=1)
    results_vol.commit()
    return info


@app.function(image=image, volumes={"/results": results_vol}, cpu=16, memory=32768, timeout=5 * 3600)
def analyze(seed: int, only: str, jobs: int, out_name: str):
    return run_analysis(seed, only, jobs, out_name)


@app.function(image=image, volumes={"/results": results_vol}, cpu=8, memory=16384, timeout=3600)
def analyze_small(seed: int, only: str, jobs: int, out_name: str):
    return run_analysis(seed, only, jobs, out_name)


@app.local_entrypoint()
def main(seeds: str = "1,2", check: bool = True, jobs: int = 16):
    calls = {}
    if check:
        calls["check_seed0"] = analyze_small.spawn(0, CHECK, 8, "check_seed0")
    for s in [int(x) for x in seeds.split(",") if x != ""]:
        calls[f"seed{s}"] = analyze.spawn(s, LONG_FIRST, jobs, f"seed{s}")
    local = ROOT / "results" / OUT
    local.mkdir(parents=True, exist_ok=True)
    (local / "modal_calls.json").write_text(json.dumps({"launched": time.strftime("%Y-%m-%d %H:%M:%S"), "app": app.name,
                                                         "calls": {k: c.object_id for k, c in calls.items()}}, indent=1) + "\n")
    print("spawned", {k: c.object_id for k, c in calls.items()}, flush=True)
    for name, call in calls.items():
        info = call.get()
        print("DONE", name, json.dumps({k: info[k] for k in ("returncode", "seconds", "peak_child_rss_mb", "cpu_count", "processor")}), flush=True)
