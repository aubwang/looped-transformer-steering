# Implementation contract for the monitor experiment (2026-11-09)

Design: docs/oversight_monitor.md (Draft 1.3). This file fixes the code interfaces. Part A covers collection;
Part B covers the probe library, precision simulation and analysis CLI/tests. Where this file and the design
disagree, stop and report it.

Conventions: all new code lives in `huginn_addition/`. Reuse patterns from `obs_app.py` (do not import from it and do
not edit it). CPU test interpreter: `/tmp/toyenv/bin/python -I` (torch 2.6.0+cpu, transformers 4.53.1). Analysis
interpreter: `/home/owner/work/.t3/projects/analysis/.venv/bin/python -I` (numpy, scipy, sklearn, pandas; no torch).

---------------------------------------------------------------------------------------------------------------------

## Part A. `mon_app.py` (Modal app `oversight-monitor`) and `mon_local_test.py`

Single file, same volumes, image and pinned `MODELS` as `obs_app.py` (add `pandas`, `pyarrow` to the image for the
multiple-choice parquet files). Forward passes only, `torch.no_grad()`, float32 parameters,
`torch.backends.cuda.matmul.allow_tf32 = False` and `torch.backends.cudnn.allow_tf32 = False`.
All measurement logic is in plain functions that take the model, tokenizer, items, cfg, output dir, device, a commit
callback and a start time (like `run_screen`), so that `mon_local_test.py` can run them on CPU with tiny random models.

### A1. Items

An item: `dict(pid, fam, i, kind, ops|mc, ans, prompt, valid)`.
- `fam` in `FAMS = ["add3", "addsub", "muladd", "add4", "sub3", "add3w", "arc_easy", "arc_chal"]`; `i` = index in the
  family's master list; `pid = 100000 * FAMS.index(fam) + i`.
- Master list per family: `N_SCREEN = 96` screen items (i < 96) followed by up to `N_MAIN_MAX = 1500` main items.
  One seed per family: `random.Random(cfg["item_seed"] * 100 + FAMS.index(fam))`.

Arithmetic (`make_arith(fam, n_total, seed) -> list`, pure Python). Rules:

| fam | query text | draw | accept if | canonical key |
|---|---|---|---|---|
| add3 | `a + b + c = ` | a, b, c in 10..99 | always | sorted (a, b, c) |
| addsub | `a + b - c = ` | a, b in 100..899; c in 10..999 | 1 ≤ a+b−c ≤ 999 | (sorted (a, b), c) |
| muladd | `a * b + c = ` | a in 12..99; b in 3..9; c in 10..99 | a·b+c ≤ 999 | (a, b, c) |
| add4 | `a + b + c + d = ` | each in 10..99 | always | sorted |
| sub3 | `a - b - c = ` | a in 200..999; b, c in 10..99 | always | (a, sorted (b, c)) |
| add3w | `a + b + c = ` | each in 100..799 | sum ≤ 999 | sorted |

- Queries: draw until `n_total` distinct canonical keys.
- Demonstrations: four per prompt, same family, drawn after all queries are fixed, from the same generator; reject a
  demonstration whose canonical key is in the family's query key set (all of it, screen and main) or repeats within
  the prompt.
- Prompt layout exactly as `make_hard_candidates` in `obs_app.py` (one demonstration per line with its answer, then the
  query ending in `= `). `ans` is the integer; the answer token is `tok.encode(str(ans), add_special_tokens=False)`,
  asserted to be one token; assert the last prompt token decodes to `" "`.
- `valid` = "num": the valid-answer set is the token ids of `str(n)`, n = 0..999 (assert each is a single token).

Multiple choice (`make_mcq(rows, tok, fam, n_total, seed, max_tokens=160) -> (items, counts)`; `rows` = list of
`dict(id, question, choices: list[str], labels: list[str], answerKey)` so the test can pass fake rows):
- Source: HF dataset repo `allenai/ai2_arc`, configs `ARC-Easy` / `ARC-Challenge`, all splits concatenated
  (parquet files via `huggingface_hub.hf_hub_download(..., repo_type="dataset")`, read with pandas). Record the dataset
  commit sha.
- Keep rows with exactly four choices and an answer key that maps to one of them; drop duplicate question texts.
  Sort by `id`, then shuffle with the family seed. The first 200 are the demonstration pool, never used as queries.
- For every row (pool and queries) permute its four options with `random.Random(seed * 1000003 + stable_hash(id))`
  (`stable_hash` = first 8 hex digits of sha256), relabel A–D. The gold letters are approximately, not exactly, uniform; store the
  permutation. Option permutation and demonstration selection use separate generators initialized with the same
  per-item seed: coupled draws, not independent randomization. Fix this in a future generator revision; the
  master list is frozen for this experiment.
- Prompt: two demonstrations from the pool (drawn with the query's own generator), then the query, rendered as in
  `huginn_addition/make_mcq_prompts.py` (`Question: … A. … D. … Answer:`), demonstrations followed by ` <letter>`.
  The prompt ends with `Answer:`; the answer token is `tok.encode(" " + letter, add_special_tokens=False)`, asserted
  single. `valid` = "abcd": the four letter tokens.
- Walk the shuffled query rows in order; keep an item if `1 + len(tok.encode(prompt)) ≤ max_tokens`; stop at
  `n_total`. `counts` = rows total, four-option, deduplicated, examined, kept, and the kept length histogram.

`build_items(cfg)`: a Modal CPU function (no GPU) that loads the tokenizer of L32 (pinned revision), builds all eight
master lists and returns them with `counts`. The entrypoint saves `items_master.json` and its sha256.


Count run `results/mon_count-20261109-203859/items_master.json`, dataset commit
`210d026faf9955653af8916fad021475a3f00453`: each arithmetic family and ARC-Easy has 1,596 items; ARC-Challenge
has 477 and is neither screened nor eligible (minimum master size 96 + 800 = 896).
**ARC-Easy is a short-prompt, two-demonstration-compatible subset; its accuracy is not representative ARC-Easy
accuracy.** Counts: 5,197 rows → 5,173 four-option → 5,154 deduplicated → 4,010 examined → 1,596 kept (39.8%).
Kept lengths: 99–160 tokens, median 143; gold A/B/C/D = 395/422/397/382. Selection depends on demonstration
lengths as well as the question.

### A2. Constants

```
T_STD  = [1, 2, 3, 4, 6, 8, 12, 16, 24, 31, 32, 33]      # readouts and saved states, position -1
T_DEEP = T_STD + [48, 63, 64]                            # for "deep" problems (64 loops)
T_PREV = [8, 32]                                         # saved states at position -2
R_POST = [4, 8, 16, 32]                                  # coda output saved at position -1
D_PREV = [8, 12, 16]                                     # layers saved at position -2 for D
R_DISP = [2, 4, 8, 16, 32]                               # displacement source budgets
R_NPOS = [8, 16, 24, 32]                                 # per-position state norm budgets
L_MAX = 160                                              # NaN-padded position dimension
```
- `kind`: "deep" for the first `n_deep` items of the run's set in each family (screen: 16; main: 256), else "std".
- float32 copies for the first `n_f32` items of the run's set in each family (screen: 8; main: 32).
- Both memberships use `i − i0`, with `i0 = 0` in screen and 96 in main, in the whole run set;
  shards share one definition, rather than taking the first items locally.
- Start state per problem: `torch.Generator("cpu").manual_seed(cfg["seed"] + pid)`, shape (L, d), truncated normal as
  in `obs_app.start` (std from the model config). It must not depend on batch composition.
- Batches contain prompts of one exact length and one kind (no padding, no attention mask). Batch size
  `cfg["bs_arith"] = 64`, `cfg["bs_mcq"] = 32`. Multiple-choice prompts are batched only at exactly equal
  token lengths. Right-padding is deferred until after the screen; it is valid for a causal model if readouts use
  each item's own last position, full-sequence norms exclude padding, and starts preserve the unpadded draws.

### A3. Looped model, per batch

Use the `prep / step / read`-style harness of `obs_app.py` (asserts on config included). With X_0 the start and
X_t = step(X_{t−1}):

Per loop t = 1..Tmax (33 or 64), per item, computed from the float32 states cast to float64:
`d_full[t] = ‖X_t − X_{t−1}‖_F` over all positions (BOS included) and coordinates; `d_last[t]` the same at position −1;
`e_full[t], e_last[t]` with X_{t−2} (t ≥ 2); `n_full[t], n_last[t]` = norms of X_t (also t = 0);
`cos_full[t], cos_last[t]` = cosine between (X_t − X_{t−1}) and (X_{t−1} − X_{t−2}) (t ≥ 2). Undefined entries NaN.

At each t in T: save `x_t[:, -1]`; if t in T_PREV save `x_t[:, -2]`; run the coda on X_t; if t in R_POST save the coda
output (before `ln_f`) at −1; logits = `lm_head(ln_f(coda_out[:, -1]))` in float64, `lp = log_softmax(logits)`:
`top_id[5], top_lp[5]` (sorted, largest first), `ent`, `lp_ans`, `valid_lm` = logsumexp of lp over the valid set,
`opt_lp[4]` = lp of the four letters (NaN for arithmetic).
Also save the prelude output at −1 and −2. Save `d_pos` at every loop and `n_pos` at R_NPOS, the per-position
step and state norms, with NaN beyond prompt length (and at t = 0 for `d_pos`). `disp_full` holds exact
‖X_target − X_r‖ for sources R_DISP and targets [32, 64]: target 32 at source 32 is NaN; target 64 is NaN in std
chunks (second-set amendment).

Gate G1, on the first batch of the container (the SAME full batch, with the same start states): the model's own forward
`M(ids, input_states=x0, num_steps=r).logits[:, -1]` against the harness at r = 4 and 32; both in float64 after
subtracting each row's mean; all logits of every row must satisfy `max |diff| ≤ 1e-4`, else the container stops with `gate_fail`. Diagnostic (recorded, not
a gate, no threshold), `batch_single_32`: the model's own forward for the first two items (or fewer if the batch
is smaller), alone versus its own in-batch result at r = 32; this measures batching alone.

### A4. D, per batch

Capture the embedding output and every layer's output with forward hooks (as `obs_app.py` does for its site). Do not
use `output_hidden_states`' last entry: in this transformers version it is after the final norm. Save `h_l[:, -1]`,
l = 0..16; `h_l[:, -2]` for l in D_PREV; per layer l ≥ 1 `dl_full, dl_last = ‖h_l − h_{l−1}‖`, and `nl_full, nl_last`.
Lens readout for every l: `lm_head(norm(h_l[:, -1]))`, same summary keys. True readout from
`M(ids, use_cache=False, logits_to_keep=1)`.
Gate G2: lens at l = 16 against the true logits, centered, float64, `max |diff| ≤ 1e-4`.

### A5. Output files

Volume path `/results/<run_id>/<model>/s<shard>/`. Chunks `c<j>_<kind>.npz` (uncompressed), flushed whenever ≥ 384
items of the current kind are pending, at the end, and at the deadline; `commit()` after each flush.

Looped chunk keys: `pid [n] int64`, `T [nT] int64`, `x_last [n,nT,d] f16`, `x_prev [n,2,d] f16`,
`post [n,4,d] f16`, `pre [n,2,d] f16` (index 0 = position −1), `d_full, d_last, e_full, e_last, n_full, n_last,
cos_full, cos_last [n,Tmax+1] f64`, `top_id [n,nT,5] i32`, `top_lp [n,nT,5] f32`, `ent, lp_ans, valid_lm [n,nT] f32`,
`opt_lp [n,nT,4] f32`, `ans_tok [n] i32`, `plen [n] i32`, `tok_hash [n] int64`, `pid32 [m]`, `x_last32 [m,nT,d] f32`.
Additional looped keys (second-set amendment): `d_pos [n,Tmax+1,L_MAX] f32`,
`disp_full [n,len(R_DISP),2] f64`, `n_pos [n,len(R_NPOS),L_MAX] f32` (thus [n,4,160]).
Float32 copies of `post`, `pre`, `x_prev` are NOT saved: the review recommendation was declined to keep the loader
simple. Looped rounding checks cover `x_last` only; D copies cover `h_last`.
D chunk keys: `pid`, `h_last [n,17,d] f16`, `h_prev [n,3,d] f16`, `dl_full, dl_last, nl_full, nl_last [n,17] f64`,
lens summary with prefix `lens_` and second dimension 17, true readout with prefix `ro_` (no depth dimension),
`ans_tok`, `plen`, `tok_hash [n] int64`, `pid32`, `h_last32 [m,17,d] f32`.

`tok_hash` is the integer from the first 15 hex digits of SHA-256 of `json.dumps(token_ids)` for each item.

Gate G4 at each flush: everything finite (NaN allowed only in the documented undefined entries), `|state| < 6e4`
before the float16 cast, and for the float32 subset `max ‖f16 − f32‖ / ‖f32‖ ≤ 1e-3`. A failing chunk is dropped, not saved;
computed `items_done` can exceed `items_committed`. `G4_checked_states` records the float32-copy item count per
checked chunk; an empty subset returns None (not checked), not 0.0, and adds no value to `G4`.
G4 verifies the NaN pattern of `d_pos`, `disp_full`, `n_pos`, finite real entries, and quadrature sums of position
steps/norms against `d_full`/`n_full`, tolerance 1e-4 × reference + 1e-30 (second-set amendment).

`meta.json` per container: status ("complete" | "partial" | "gate_fail"), items requested/done, `items_committed`, `committed_pids`, `chunks`, `model`, `tokenizer`, hf id, revision,
dtype, torch/transformers versions, code sha256, hash of the model's forward source, gate values (including `G1_items`, `G4_checked_states`), T lists, `R_DISP`, `R_NPOS`, `L_MAX`, `valid_ids` (token ids of the valid answers: `num` = integers 0..999 in order, `abcd` = letters A–D; third-set amendment), cfg
(including `items_sha256`, `run_id`, `mode`), and
`timing`: one row per batch `[fam, kind, L, B, Tmax, seconds]`, plus model-load seconds and total seconds.
G3 additionally asserts that the final multiple-choice token decodes to `":"`; G3/dtype failures are recorded
with status `gate_fail`. `add_accuracy` adds `accuracy`, `accuracy_counts`, and `out_dir` to returned metadata,
which is saved in `containers.json`; these fields are not written back to the on-volume `meta.json`.

### A6. Running

- Problems are processed deep-first, then std, with batches round-robin over families, choosing the length bucket
  whose first item has the smallest `i`. Deadline output is a usable completed subset, not a balanced prefix:
  arithmetic batches hold up to 64 items, while multiple-choice batches hold only exactly equal-length prompts.
  Partial runs are never used for screening.
- A Modal timeout is a decorator constant, so use two functions sharing one implementation: `collect` (timeout 480)
  and `collect_long` (timeout 840), both `gpu="A100-40GB"`. The cooperative deadline is `cfg["limit_s"]` minus a 40 s
  reserve, measured from container start (model loading counts). The entrypoint asserts `limit_s ≤ timeout − 20` and
  that the sum over containers of the *function timeout* is ≤ 28·60 s. Screen uses three `collect` containers: 3 × 480 s = 24 GPU-min worst-case
  allocation, `limit_s` 460, cooperative stop at start + 420 s. Main uses `collect_long`, `limit_s` 820
  (cooperative stop at start + 780 s).
- Shards: main items of a family with `(i − 96) mod S == k`. Identical for every model. Main uses `--shards 3`
  (`SHARDS_MAIN`; any other value is refused in main mode, third-set amendment),
  fixed from first launch: 15 shard-runs for five models in eight launches, at most two containers per launch,
  at most 28 allocated GPU-minutes per launch. Two-shard projections 534–665 s do not survive 25% uncertainty
  against the 780 s stop; three shards project 363–445 s (second-set amendment).
- Entry point `main(mode, items="", spec="", n_main=0, fams="", shards=1)`:
  - `count`: runs `build_items`, saves `results/<run_id>/items_master.json`, prints counts. No GPU.
  - `screen`: needs `--items`; runs i < 96 only for families with at least N_SCREEN + N_MAIN_MIN = 896 master
    items, on L4, L32 and D (three `collect` containers, `limit_s` 460); writes per-family accuracy for every model at every t in T_STD (D: true output), applies the
    eligibility and selection rule of the design (section 3), writes `screen_manifest.json` (selection, hashes) and a
    throughput table (token-loops per second by family and kind, per model).
  - `main`: needs `--items`, `--fams a,b,c,d`, `--n-main`, `--shards S`, `--spec "L32:0,D:0"` (model:shard
    pairs for this launch). Verifies the items sha against the screen manifest and families against its selection. Enforces
    `800 ≤ n_main ≤ 1500`, exact `model:shard` parsing, unique pairs, valid indices, full requested item count and
    non-empty requested shards (second-set amendment). Main results
    in `containers.json` are keyed `model:shard` (screen results by model).
  - The first main launch writes `main_manifest.json` beside the items file: `families`, `n_main`, `shards`,
    `items_sha256`, `cfg` excluding `run_id`, and `code_sha256` of mon_app.py. Later launches must match these
    collection fields or refuse to run. `analysis_sha256` records hashes of analyze_mon.py, mon_probe.py,
    docs/oversight_monitor.md and docs/mon_spec.md at first launch. Later analysis-hash changes only print a warning;
    they do not replace the frozen hashes or block collection (second-set amendment).
  - Retrieval is not part of the app.
- Never retry automatically; print one line per container with status and seconds; exit 1 if any is not complete.

### A7. `mon_local_test.py`

Model on `obs_local_test.py`: tiny random looped model and tiny random D on CPU, the local tokenizer, fake
multiple-choice rows. Must check, with assertions: item rules (key distinctness, demonstration disjointness, letter
uniformity within tolerance, token limit), start independence from batch composition, G1/G2/G4 pass, array shapes and
dtypes as in A5, `d_last` equals differences of consecutive saved float32 states on the f32 subset; `d_full` equals a direct
trajectory recomputation,
readout at t equals the model's own forward at that budget, deadline cut leaves valid chunks and status "partial",
labels reproducible from `top_id[:, :, 0] == ans_tok` (D uses `ro_top_id`). Checks also cover a real two-item
G1 batch and singleton diagnostic, committed ids/counts, token-hash dtype, a mid-batch loop deadline, G1 failure
metadata with no saved chunk, and G2/G4 rejection cases. Checks now include direct recomputation of `d_pos`,
`disp_full`, `n_pos` and shapes/dtypes/padding (second-set amendment). Prints `ALL CHECKS PASSED` on success.

---------------------------------------------------------------------------------------------------------------------

## Part B. `mon_probe.py`, `mon_probe_test.py`, `mon_power_sim.py`

Pure numpy/scipy (sklearn allowed for `roc_auc_score` cross-checks in the test only). No torch.

- `auroc(score, y)`: y = 1 means *wrong answer*; higher score = more suspicious. Ties count half.
- `selective_error(score, y, reject_frac=0.2)`: error rate among the kept items when the top `reject_frac` by score
  are rejected; also returns the threshold.
- `fit_logistic(X, y, lam, U=None)`: minimize mean log loss + lam/2 × ||w||² in standardized X coordinates,
  plus U_RIDGE/2 × ||u||² in whitened U coordinates; free intercept (second-set amendment). X is standardized
  on training data, std floored at 1e-6 of the largest, constants dropped. U is standardized then SVD-whitened
  to unit mean square, dropping singular values ≤ U_RANK_TOL = 1e-8 times the largest; U_RIDGE = 1e-6 is fixed
  identically for baseline and augmented fits. The formerly unpenalized block is thus weakly regularized.
  For d > n the penalized X block uses thin SVD row-space coordinates. L-BFGS runs first, then trust-krylov with
  exact Hessian-vector products if max |gradient| > GRAD_TOL = 1e-6 for the mean objective. This criterion, not
  optimizer success, defines convergence. Returns `score(X,U=None)`, `w_raw`, `u_raw`, `b_raw`, `norm_w_raw`,
  `converged`, `message`, `n_iter`; raw coefficients reproduce the score on unstandardized inputs.
- `fit_path(X, y, lams, U=None)`: one set of training transforms and descending-λ warm starts; input-order models.
  Every fit is retained in selection/estimation even if non-converged. Analysis rows report `nonconverged`,
  `inner_nonconverged`, and contrasts `nonconverged_outer_fits`; any contrast with outer failures must be flagged.
  Both `nested_cv` (fields `inner_nonconverged`, `inner_solver_messages`) and the multi-site `nested_sites`
  count inner failures (second-set amendment).
- `fit_meandiff(X, y)` (score = projection on the difference of class means, standardized coordinates) and
  `fit_pca_logistic(X, y, lam, k=128)` (centre, project on the top k principal directions of the training part, not
  whitened, then `fit_logistic`).
- `spline_features(u, knots)`: natural cubic spline basis with four knots at fixed training quantiles (5, 35, 65, 95%).
- `make_folds(y, n_folds, seed)`: stratified by label. `make_joint_folds(strata,n_folds,seed)` balances joint
  categories; `check_folds(folds,y,n_folds)` requires all fold numbers and both classes in every fold. M1-a uses
  one shared outer assignment per family/seed over joint L32/D correctness for all four cells, skipping the table
  if a shared fold lacks a class. Other fits use their own label (second-set amendment).
- `nested_cv(sites, y, U=None, lams=LAMS, n_outer=5, n_inner=4, seeds=(0, 1, 2), fitter="logistic", folds=None)` where `sites` is
  `dict[name -> array [n, d]]` (one or many). For each seed and outer fold: inner folds on the outer-training part
  choose (site, λ) jointly by mean inner-fold AUROC (ties: first in dict order, then larger λ); refit on the outer
  training part; score the outer test part. Returns per seed: out-of-fold scores, fold index of each item, chosen site
  and λ per fold, per-fold AUROC, their mean, and the fitted outer models (needed later for transfer and for `w_raw`).
  `folds` optionally supplies one outer fold vector per seed, validated by `check_folds`. Inner folds remain
  own-label stratified. `U` can be a callable returning training/test blocks with training-only spline knots.
  `LAMS = 10.0 ** np.arange(-6, 3)`.
- `apply_models(result, sites_other)`: score other feature arrays (another budget, another model's labels) with the
  frozen outer-fold models, each item by the model whose training part excluded it.
- `paired_bootstrap(stat_fn, groups, n_boot=2000, seed=0)`: resample item indices with replacement within each group
  (group = family × outer fold), the same resample for every model; returns the estimate and a 95% percentile
  interval and draws. The analysis wrapper retains fold strata (intersections for multiple fold assignments):
  ANY undefined draw makes the whole interval unavailable with count/reason; draws are never dropped.
- `mon_probe_test.py` (prints `ALL CHECKS PASSED`): solver agrees with a direct full-dimension optimization on a small
  problem (d < n and d > n) to 1e-4 in scores; `w_raw` reproduces scores; whitened ridge-U scores are invariant
  to rescaling within tolerance; separation, nearly collinear splines, invertible basis changes, duplicate-column
  rank handling and shared joint folds are checked; `auroc` agrees with sklearn; planted effect: Gaussian d = 2048,
  n = 1500, known optimal AUROC, nested AUROC within sampling error; null data gives about 0.5; site selection picks
  the informative site; a deliberately leaky pipeline (site chosen on test data) is detectably optimistic while
  `nested_cv` is not; permuted labels give about 0.5; timing printed for one `nested_cv` call with 2 sites,
  n = 1500, d = 2048.
- `mon_power_sim.py`: precision table for the design (section 7). Synthetic "two models on the same problems":
  for each of error rates {0.10, 0.30, 0.50} × n in {800, 1500}: d = 2048 Gaussian features for model 1 and model 2
  with a planted signal giving population AUROC 0.80 for each, the two models' labels agreeing on 70% of problems
  beyond chance (state the construction in the docstring); run the real `nested_cv` (one site, one seed) on both;
  over the selected replications report the standard deviation of one AUROC, of the paired difference, and the mean
  width of the paired bootstrap interval. Writes `results/mon_power_sim.json` and prints a table. The default aims to fit
  within a conservative 25-minute projected compute budget (reserving runtime variability), reducing
  replications if needed, not n or d. `--reps 40` explicitly requests 40 per cell; `--jobs` parallelizes independent
  replications without changing seeds/results. Both models use shared joint outer folds, the final solver and
  paired bootstrap within those folds. As in the analysis, a replication with any undefined bootstrap draw reports
  its interval as unavailable (`intervals_unavailable` per cell); it does not raise and draws are never dropped.
  Cells also report `interval_covers_zero` (coverage of the true difference 0 among available intervals) and
  `nonconverged_outer_fits`. The 40-replication run is complete (`results/mon_power_sim.json`,
  table in section 7 of the design; second-set amendment).

## Amendments 2026-11-09

- A1–A2: recorded counts and ARC-Easy subset limits, approximate letter balance, coupled per-item seeds and
  whole-run subset ranks shared across shards; documented deferred right-padding.
- A3–A7: corrected full-batch G1 and batching diagnostic, G3/dtype failure metadata, token hashes, committed output
  accounting, metadata persistence, deadlines, screen models/eligibility, partial subsets and main result keys;
  aligned the local-test description with its assertions.

Historical requirements from the first review (resolved or qualified by the second set below):
- The entry point must enforce the frozen minimum sample size, exact parsing of the `model:shard` spec,
  non-empty shards, and a frozen manifest holding families, n_main, shard count and hashes. Current selection/hash
  checks do not enforce all of these.
- Record float32-subset coverage per shard and make an unchecked chunk distinguishable from a perfect match.
- Use the float32 subset to quantify the effect of float16 storage on probe scores and on the stability forecast,
  not only on norms.


## Amendments 2026-11-09 (second set, after external review)

These amendments follow review of the frozen package and precede any main data. Stage 1 alone is collected.
The first amendment block is retained as history; its pending implementation statements are superseded here.

- A2–A5: add R_DISP, R_NPOS, L_MAX and `d_pos`, `disp_full`, `n_pos` schemas, padding and G4 checks; metadata
  records all constants. G4 returns None without float32 copies and records copy counts per chunk. Recommended
  float32 `post`, `pre`, `x_prev` copies were declined to keep the loader simple; looped rounding covers `x_last`.
- A6–A7: minimum sample size, exact spec parsing, unique pairs, index/non-empty checks and frozen main manifest
  are implemented. Collection fields must match; changed analysis hashes warn. Three shards/model are fixed from
  first launch for timing margin, replacing adaptive re-planning. Update local-test assertions accordingly.
- B: replace the unpenalized-U solver with training-only standardization/SVD whitening, rank tolerance 1e-8 and
  ridge 1e-6, shared by baseline/augmented fits; use the 1e-6 gradient criterion with trust-krylov fallback.
  Review found spline condition numbers up to about 3.6e7 and old-solver failures. Screen benchmark evidence
  supplied with the review: all nine penalties converged for pooled probe, confidence+probe, flexible+probe and
  baselines; all 1,440 per-family small-sample fits converged, Newton needed twice, worst gradient 5e-7.
  Retain every fit; report failures and flag contrasts with outer failures; inner failure counts are reported
  for every fit. Correct test tolerance and add conditioning/shared-fold tests.
- Inference: shared joint folds for M1-a; bootstrap within retained fold intersections, no dropping undefined
  draws; intervals for each fit and transfer as well as contrasts. Family means require all analysed families.
- T1: primary/secondary forecasts, forecast-only conformal calibration, transductive threshold and marginal
  coverage qualifications; classification/decision/ranking inequalities; quantization allowance and actual
  frozen-probe float32/float16 comparisons replace `t1_relative_B_score_change`. Prefix versus single-position
  rate claims and per-position floors are distinguished below, consistent with the theory note.
- Precision simulation: shared folds, final solver, `--jobs`; the 40-replication run is complete and its table
  is in section 7 of the design. Add the final analysis CLI, outputs and regression-test contract below. Correct M4
  description: D has layer-step/token-change features, not loop-only two-step/cosine features; confidence change
  means top log-probability change. Rounding/copy and solver benchmark values are supplied review evidence.

### Third set (2026-11-09, after the second external review; before any main data)

- `mon_app.py`: `valid_ids` in `meta.json`; main mode asserts `shards == SHARDS_MAIN == 3`. No change to saved arrays.
  `mon_local_test.py` checks the saved table against the tokenizer.
- `analyze_mon.py`:
  - `token_answers(valid_ids)` builds (answer kind, token id) → answer; `problem_features` decodes the predicted
    token with it, asserts that every item's `ans_tok` decodes to its gold answer, and encodes a token outside the
    table as `unknown` = 1 with zeros elsewhere. `load_runs(..., manifest)` requires `valid_ids` in every shard and
    equality across shards. Data without it (the screen) fall back to the old table from observed gold answers,
    flagged in `settings.output_identity_tokens` as LEGACY.
  - `m3a`: rows gain `source_nonconverged`; `native_minus_transfer` counts the frozen source probes' failures in
    `nonconverged_outer_fits`.
  - `Analysis` drops a model with no items of the family (skip reason recorded); `common()` returns None with a
    reason when the models share no problem.
  - `m1x` and `position_rates` as described below.
- `analyze_mon_test.py`: checks for each of the above (decoding independent of gold answers, failure counting,
  per-item floor with mixed data, empty family, disjoint models, missing target).

### Analysis CLI and reporting (second-set amendment)

`analyze_mon.py --runs DIR [DIR ...] --items PATH --out DIR` runs CPU analysis. Optional `--manifest PATH`
reads `main_manifest.json`, checks the items-file hash, every loaded shard's collection-code/items hash, expected
problem ids and completeness. Default families come from the manifest; `--fams`, `--models`, `--seeds`, `--only`
(comma-separated analysis names), `--jobs` (default 6) and `--smoke` restrict or configure the run. Normal settings
are five outer/four inner folds, three seeds (0,1,2), 2,000 bootstrap draws. Smoke uses three/two folds, seed 0,
200 draws. Analyses: m1a, m1b, m1c, m1s, m1x, m3a, t1, m4, f16, dyn. `m1x` is cross-family transfer: for each
source family, the diagonal fits `L32/source` (x_32) and `D/source` (h_12) are applied to every other analysed
family on the items both models have; target score = mean of the five outer-fold models' scores. Rows `to/<family>`
hold `L32_auroc`, `D_auroc`, `difference`, their intervals (2,000 resamples of target items within the four joint
correctness categories, shared by both models, conditional on the fitted probes) and class counts;
`to/<family>` also holds `*_auroc_single_probe` (mean over the five probes of each probe's own target AUROC).
`mean_over_targets` holds the averages over targets, `*_minus_own_family` (single-probe target AUROC minus the
own-family mean of single-probe held-out AUROCs) and `drop_difference`; it needs every other analysed family: a
target with no common items or a missing class is skipped and listed, and then `mean_over_targets` contains only
`targets` and `missing_targets` (third-set amendment). Budget transfer (`m3a`) additionally reports all saved
target budgets as descriptive rows.

`--allow-partial` admits only metadata-listed committed chunks of deadline-stopped shards; completeness is
reported instead of asserted, while hashes and expected-id checks remain. Completeness is checked for loaded
models; entirely absent models produce analysis skips. Provenance records `analysis_matches_manifest` by comparing
current analysis/document hashes with those frozen at first launch. Missing optional arrays are disclosed; mixed
old/new chunks get NaN for missing float arrays. No partial screen is eligible for family selection. Outputs are `results.json` (estimates, settings, skips,
provenance) and `scores.npz` (saved M1/M3 item ids, labels, scores and folds).

Each fit AUROC has `interval`; transfer has `transfer_interval`. Any undefined bootstrap draw (a resampled fold
lost a class) invalidates the entire interval with count/reason. Family means have equal weights over all analysed
families; numeric leaves missing in any family are unavailable and listed under `incomplete`. Intervals require
all families too. Intervals condition on frozen out-of-fold scores, not refitting uncertainty.

Confidence is four features: top log-probability, top-two log-probability gap, entropy and `valid_lm`, the logsumexp
of log-probabilities over valid integer tokens 0..999 or space-prefixed A–D tokens. Labels for analysis are 1 = wrong,
whereas the design's c is 1 = correct. The per-fold source-budget threshold is the held-out 80th percentile,
computed on the same items it is applied to (transductive).

**T1 evaluation** (second-set amendment). The primary calibrated
comparison is `scaled_rule` (rule A) versus `constant`; secondary candidates are `scaled_step`, `scaled_two_step`
and `scaled_rule_two_step`. Score candidates multiply by the frozen raw probe norm and add `_last` versions of
all nonconstant candidates using last-position steps and norms. The two-step rule takes Q, the largest of up to
four ratios e_t/e_{t−2} ending at r, and forecasts e_r Q(1−Q^((32−r)/2))/(1−Q); no forecast if Q ≥ 1 or
e_r/n_r < 10⁻⁶. Rule A similarly requires d_r/n_r ≥ 10⁻⁶.

Split-conformal calibration uses only finite positive predictors in the calibration half of each held-out fold.
The multiplier is the ceil((n_cal+1) × 0.95)-th ordered realized/predictor ratio; it is infinite if that rank exceeds
n_cal. All candidates use the same halves. Items without a finite forecast are excluded from coverage and count as
not stable. Forecast stability is compared at common nominal marginal coverage 95%; this is NOT a guarantee of
95% coverage among items declared stable. The threshold τ is the 80th percentile of the same held-out source-budget
scores to which it is applied: this is transductive (uses test features, though no test labels).

`classification_transfer` checks error_r ≤ error_32 + flag_changed + label_changed and reports `slack` (right side
minus left side). The code also checks decision_changed ≤ `decision_change_bound`, which equals
1 − fraction_forecast_stable + `stable_and_violated`. With labels fixed to those at 32 it checks
`auroc_change_fixed_labels` ≤ `close_pair_mass` + `violated_pair_mass`; the latter counts pairs with a bound
violation outside the close-pair set. Ranking quantities are averaged over evaluation folds with both classes.

Float16 storage introduces score rounding. For stored state x and frozen raw weight w, `quantization_bound` is
Q(x,w) = ε/(1−ε) Σ_j |x_j| |w_j| + 2⁻²⁵ Σ_j |w_j|, ε = 2⁻¹¹. Movement allowance is
Q(x_r,w) + Q(x_32,w), reported as `quantization_allowance`, with `allowance_over_movement` and
`score_violation_beyond_quantization`. Calibrated forecasts report `coverage_with_allowance` and
`fraction_forecast_stable_with_allowance`. `rounding_subset` compares the same frozen probes on float32 copies
and float16 states at both budgets: score errors, changed flags, raw-rule stability and relative movement.
The earlier `t1_relative_B_score_change` field was removed: zero there did not test rounding.

**Positional rates** (second-set amendment). The block-triangular Jacobian implies that the asymptotic rate of the
first p positions TAKEN TOGETHER is non-decreasing in p, not the rate of each single position: diag(0.9, 0.1) is a
counterexample. Inheriting earlier slow rates requires nonzero coupling and excitation, an expectation measured
separately. Windows within 33 or 64 loops give finite-time evidence, not limits. `position_rates` reports single
positions, `prefix_*` statistics and `last_position_minus_whole_state`, in windows 8–16, 16–24, 24–32. Positions
whose end step is below 10⁻⁶ of their own state norm (`n_pos`) are excluded; a prefix is compared with the norm of
its positions together (root sum of squares). The norm source is chosen per item: an item without saved norms at
every real position (older data) uses whole-state norm / sqrt(prompt length) for each position, combined the same
way for prefixes; the output counts the items under each rule. (third-set amendment)


`analyze_mon_test.py` uses synthetic looped/D chunks: planted/null signals, shared M1-a folds, per-fit/transfer
intervals, bootstrap failure without dropping draws, incomplete family means, partial/manifest checks and provenance,
optional-array handling, two-step bounds, forecast-only conformal coverage, classification/decision/ranking
inequalities, quantization/float32 comparisons and position-rate floors. It prints `ALL CHECKS PASSED` on success.
