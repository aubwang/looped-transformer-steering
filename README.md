<img src="assets/icon.svg" width="64" height="64" alt="">

# analysis

Empirical interpretability experiments on **Huginn-0125**, a recurrent-depth ("looped") transformer
([model card](https://huggingface.co/tomg-group-umd/huginn-0125), pinned revision `bb6621b…`). The study asks how
concepts are represented across loop iterations, and how they can be read and controlled.

## Start here

- **[`huginn_addition/WRITEUP.md`](huginn_addition/WRITEUP.md):** the synthesis. Main findings, methods, limitations,
  negative results and key runs. It was reviewed by GPT-6 Astra and Gemini 4 Argon and fact-checked against the raw
  outputs.
- **[`huginn_addition/README.md`](huginn_addition/README.md):** the experiment log. One section per experiment, in
  chronological order, with the design, prespecified rules, gates, results, reviewer-accepted wording and caveats.

## Layout

| Path | Contents |
|---|---|
| `huginn_addition/intervene_app.py` | Modal app: every GPU experiment (one function per experiment; configs and entrypoint at the bottom; the module docstring has an index) |
| `huginn_addition/modal_app.py` | Modal app for the first pilot (per-iteration dynamics on addition) |
| `huginn_addition/analyze_*.py` | Prespecified analyses; frozen rules and recorded amendments are in each docstring |
| `huginn_addition/make_*.py`, `select_*.py` | Prompt, selection and push/direction builders |
| `huginn_addition/data.py`, `fit_circle.py`, `shape_*.py` | Shared data generation and early offline analyses |
| `results/` | Prompt sets, push/direction files and per-run outputs (`<mode>-<timestamp>/`, with `meta.json` and `analysis_summary.json`) |

## Later work: theory notes and the oversight pilot (2026-11-07 to 11-08)

Math notes are Typst sources with compiled PDFs in `docs/`; each `*_check.py` script re-derives its note's numbers.

| Note | Contents | Code and runs |
|---|---|---|
| `docs/primer.pdf`, `foundations.pdf`, `derivations.pdf` | Background built from definitions (math, no ML assumed), and a workbook deriving the main Huginn results | — |
| `docs/analytic.pdf` | Why the loop-aware lever first points against the readout and turns positive one step later (first-passage decomposition, Theorem 4 bounds) | `docs/analytic_check.py` |
| `docs/freeloop.pdf` (Draft 3) | Random-direction steering geometry, a rejected free-loop null model, readout suppression shared by the read space, opposite sign in Ouro-1.4B | `freeprobe` mode, `ctrlprobe_app.py`, `ouro_app.py`, `weights_app.py`, `docs/freeloop_check.py`; log section in `huginn_addition/README.md` |
| `docs/monitoring_gap.md`, `monitor_protocol.md` | Literature check on correctness monitors for looped models; a capped monitor-transfer protocol (not run) | — |
| `docs/observability.pdf` | Loss-observability of random-depth training (what the training loss can see of the loop's dynamics) | `docs/observability_check.py` |
| `docs/oversight_results.pdf` (Draft 2) | Matched-depth oversight pilot on Recurrent-Llama-3.2 (L4–L32) vs its non-looped partner: state-nudge sensitivity and linear predictability of pushes, on easy (part 1) and screened hard (part 2) prompts, plus a conditional theory section | `obs_app.py`, `analyze_obs.py`, `report_hard.py`, `obs_local_test.py`; designs `docs/oversight_pilot.md`, `oversight_hard.md`; runs `results/obs_*` |

**Status (2026-11-08).** The Huginn write-up is done. The oversight pilot is written up, and GPT-6 Astra recommended
stopping GPU work there. Possible next steps: develop the conditional theory with no GPU; a capped pilot at
non-answer positions; and, once larger evaluations resume, a second matched model family and a monitor experiment.

Large raw arrays (`.npz`/`.parquet` inside run folders, and a few large per-item push files) are not in git. They live
on the Modal volume `huginn-results` and on the local disk; see `.gitignore`.

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python "modal[api-proxy-support]" pandas numpy scipy matplotlib pyarrow
# GPU experiments (Modal, A100-40GB):
cd huginn_addition && ../.venv/bin/modal run intervene_app.py --mode <config> [--probs ...] [--dirs ...]
# Analyses (local CPU):
.venv/bin/python -I huginn_addition/analyze_<name>.py results/<run_id> ...
```

Notes:
- Bulk `modal volume get` returns HTTP 429 behind the proxy. Fetch files one at a time, with retries.
- Modal launches can also hit 429. The shell loops in the log retry after a pause.
