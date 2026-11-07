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
