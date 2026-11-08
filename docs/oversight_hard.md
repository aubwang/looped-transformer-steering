# Oversight pilot, part 2: harder prompts (Draft 1, 2026-11-08)

Follows docs/oversight_pilot.md (Draft 2, frozen analysis rules) and its results, docs/oversight_results.pdf.
Code: huginn_addition/obs_app.py (modes `screen` and `hard`), analyze_obs.py, obs_local_test.py.
Every Modal run stays under 30 GPU-min (the user's instruction); larger evaluations wait for newer models.
Draft 0 was reviewed by GPT-6 Astra (xhigh): screen GO WITH CHANGES, hard run GO WITH CHANGES conditional on a complete
screen and the frozen selection; no separate GPU smoke needed. Draft 1 makes those changes (section "Frozen before the
screen" at the end).

## Why

In runs 1–2 every model answered every prompt correctly, with median answer probability above 0.99. Almost everything
measured was a confident, correct computation. Astra's results review ranked "repeat on harder prompts" as the most
informative next step: does what we saw (L4's output reacts less to state nudges than L32's; looped models are not worse
than the feedforward partner at linear steering prediction) survive when the models are unsure or wrong?

Exploratory and two-sided, like part 1. No theorem predicts an ordering (docs/observability.pdf, Section 7).

## Models

L4, L32 and D, same revisions, float64, RMSNorm patched to float64, D pinned to SDPA (as in part 1).

## Step 1: candidate pool (fixed now)

Four families, 64 candidates each (256 in all), generated with `random.Random(2026_11_08)` by `make_hard_candidates`.
Each prompt has four demonstrations of the same family, one per line, then the query, ending with "= " (the trailing
space is its own token). Every answer is below 1000, so it is one token (asserted).

| Family | Query | Rule |
|---|---|---|
| sub  | a - b = | a, b three-digit, a > b, at least one borrow (some digit of a smaller than b's digit in the same place) |
| add3 | a + b + c = | a, b, c two-digit |
| mul1 | a * b = | a in 12..99, b in 3..9 |
| mul2 | a * b = | a, b in 11..99, a·b < 1000 |

Queries are distinct within the pool. Demonstrations follow the same rules, and a demonstration identical to its own
query is rejected and redrawn (Draft 0's pool had two such prompts).

## Step 2: difficulty screen (base outputs only)

For each candidate and each model, one unperturbed computation:
- Looped models: start drawn from the training start distribution (truncated normal), seeded by the candidate id
  (`seed + 500_000 + cid`); the model's own forward at r ∈ {M, 32, 64} loops. In part 1, gate G1 found this forward
  equal to our harness to 1.2e-15 on the real L4 and L32 (so the inference path does not round its logits to float32).
  The screen records the dtype of the returned logits and a hash of the forward's source code.
- D: one forward pass.
- Recorded at the answer position: probability of the true answer p, greedy token, correctness c (greedy = answer),
  entropy, and the logit gap between the top two tokens.

Nothing about nudges, pushes, gradients or linearity is computed, so selection cannot depend on any measured
intervention quantity.

## Step 3: selection rule (fixed now)

Use r = 64 for L4 and L32 (the depth of the primary Part A summary) and D's single pass. For each candidate, with
m ∈ {L4, L32, D}:
- **W (someone wrong):** c_m = 0 for at least one m.
- **U (unsure but right):** c_m = 1 for all m, and min_m p_m < 0.9.
- **E (easy):** all correct and min_m p_m ≥ 0.9. Not selected unless needed to fill.

Pick 8 from W and 8 from U. Within a stratum, go round-robin through the families in the order sub, add3, mul1, mul2,
taking each family's candidates in pool order (which is random), skipping a family when it has none left.
- If a stratum has fewer than 8, take all of it and fill from the other stratum by the same procedure.
- If W and U together have fewer than 16, fill from E in the same way and report it.
- Final order alternates W, U, W, U, … (then whatever remains), so any prefix is balanced; uids 0..15 in that order.

The screen's per-family accuracy and probability distributions for all three models are reported in full.

## Step 4: the hard run

Exactly the part-1 protocol (Part A at r ∈ {M, 32, 64}; Part C at budgets {M, 32} and D) on the 16 selected prompts,
with the same seeds rule, settings and frozen analysis rules. No prompt is shared with part 1 or with a smoke run, so
the "without smoke prompts" sensitivity analysis is empty (`smoke_uids = []`).

**Primary (exploratory, two-sided):** log(A_L4 / A_L32) at r = 64, relative normalization, low-residual subset, at
least 12 jointly valid prompts (else inconclusive), bootstrap over prompts. Absolute normalization alongside.

**Also reported:**
- Part C looped versus D at all six KL targets, with absolute errors next to the ratios.
- Correctness and answer probability for every model, prompt and depth used.
- Everything split by stratum (W, U), descriptively.
- How many prompts drop out of the low-residual subset. Harder prompts may not settle; that is a result, not a failure.
- Screen-to-run transitions of correctness and of the greedy token (the run uses a different random start). This
  is descriptive: matching correctness can hide different wrong tokens, and two starts cannot establish start
  independence.

## Gates

G1, G2 and G4 as in part 1 (hard gates). Tokenization asserts (single-token answers, trailing space token). Screen:
every candidate gets a finite p in every model. Status rules as in part 1: anything incomplete or with a failed gate is
"not clean".

## Budget

- Screen: three containers (L4, L32, D), 480 s timeout each: worst case 24 GPU-min, expected about 5.
- Hard run: three containers, 480 s timeout each: worst case 24 GPU-min, expected about 9 (part 1 took 8.6).
- Each is a separate run under the 30-minute cap.

## Not in scope

Non-answer positions, Arnoldi spectra, other model families, monitors with correctness labels: later steps.

## Frozen before the screen (after Astra's pre-run review)

**Interpretation.**
- Conclusions concern *this jointly screened, family-balanced prompt set*, not hardness in general. Selection effects
  have no predetermined direction: W can be dominated by one model's weaknesses; U picks whichever model is least sure,
  which includes calibration differences between models. Fresh starts in the run allow regression to the mean,
  especially for the looped models (D's screen output is deterministic and will repeat exactly).
- Reported for every prompt: which models made it W or U, per-model correctness and answer probability in the screen
  and in the run, and the screen-to-run transitions.
- The 0.9 threshold is a declared operational cutoff, not a calibrated definition of uncertainty. It will not be
  changed after the screen, and the full screen probability distributions are published.
- No post-screen threshold tuning, prompt replacement, convergence filtering or outcome-driven reruns. If a gate fails
  or a container is incomplete, the run is "not clean"; difficult prompts are never swapped out to get a clean run.
- The ≥12-pair rule is unchanged. The 8+8 design leaves room for only four jointly invalid prompts before the primary
  comparison becomes inconclusive, and the eight-prompt strata are descriptive only. An inconclusive primary result is
  accepted as a result.
- Bootstrap intervals summarize the selected sample, conditional on the screen and on the frozen normalizations; they
  are not uncertainty over the selection process.

**Comparability with part 1.** Input lengths are 39 tokens for sub, mul1 and mul2 and 54 for add3 (part 1: 39
throughout); operators, demonstrations and operating points also change. A part-1 versus part-2 difference therefore
cannot be attributed to difficulty alone. Part C's per-coordinate scale s pools all non-BOS tokens of the 16 prompts
(longer add3 prompts weigh more) and is model- and sample-specific; subset analyses keep the full-set scale. Both Part A
normalizations are reported, with prompt lengths, family counts and the existing push-size and energy diagnostics.

**Provenance.**
- The screen writes candidates.json, screen_results.json (each model's returned identity, configuration, logit dtype,
  forward-source hash, probabilities and greedy tokens), hard_prompts.json and screen_manifest.json (configuration,
  code hash, model revisions, file hashes, selection rule parameters, counts, selected candidate ids and strata). The
  three screen.npz files on the volume are fetched and kept with them.
- The hard run refuses to start unless `verify_selection` passes: the files match the manifest, the candidate pool
  regenerates exactly, and re-deriving the selection from the saved screen results reproduces the prompts file. The
  run records the screen's run id and the prompts hash.

**Analysis fixes (no measurement change).** Unmeasured prompts are reported as missing and excluded from accuracy
denominators; exclusion counts refer to the requested subset; per-prompt answer probabilities are exported; screen
agreement is reported as correctness agreement and as identical greedy token.

**Code at freeze** (SHA-256 prefixes): obs_app.py f9449bd2190bb93d, analyze_obs.py 6b603e3e0cd0d031; run_obs and obs()
byte-identical to part 1.
