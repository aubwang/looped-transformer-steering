# Read-space self-gain on Recurrent-Llama L32 (Draft 1.2, 2026-11-10)

Status: design fixed before any run. Code `huginn_addition/lread_app.py`, analysis
`huginn_addition/analyze_lread.py`, local test `huginn_addition/lread_local_test.py`.
Budget: one production run of at most 5 GPU-minutes plus a smoke run, A100-40GB.

## 1. Why

`docs/freeloop.typ` (sections "Readout and read-space suppression" and "A second looped model") measured, at a
settled state of a looped model, how much one loop feeds a direction back into itself:

    alpha(u) = <u, J^T u>,   u a unit vector, J the own-position Jacobian of one loop at the state,

against the average over all directions, `tau = tr(J)/d`, estimated from random unit probes. In Huginn the
directions the output reads (family C3 below) have `alpha` far below `tau` (median excess −0.22 and −0.20, all
below every probe). In Ouro-1.4B the excess is positive (+0.022). So "the loop feeds what the output reads back
into itself less than a typical direction" is not a property of looped models in general.

The lever experiments (`docs/llever_design.md`, `docs/steering_results.typ`) found on L32 that a push along the
readout gradient has an immediate effect which is gone after one loop. That is a statement about the task
readouts only, and it does not say whether L32 treats them differently from a typical direction: if `tau` itself
were near zero, every direction would lose its self-component in one loop. This run measures `tau` and the
read-space excess on L32 with the Huginn/Ouro protocol, so that the three models can be put in one table.

## 2. Model, precision, state

Exactly as `docs/llever_design.md` section 2: L32 at the pinned revision, float64 (norm and rotary patches),
harness `prep / step / read` of `llever_app.py`, gate G1 against the model's own forward. Base state: start seeded
`11000000 + pid`, `K0 = 64` loops, `x* = x_64`. `J` is the Jacobian of one loop with respect to the state at the
last position `q`, the other positions held at `x*` (the own-position block; the attention is causal and the
sequence ends at `q`).

## 3. Contexts

- **Set A (primary), addition:** the 40 evaluation problems of the lever experiment (`make_items()` of
  `llever_app.py`, split `eval`, same order). Task readouts: the ten centred class logits
  `z_k = logit("7k") − mean_j logit("7j")`.
- **Set B (secondary), truth:** the 40 truth prompts used for Huginn and Ouro
  (`results/freeprobe-20261107-130520/items.json`, field `prompt_B`, BOS prepended, one sequence at a time, no
  padding, `q` = last token, `pid = 900000 + uid` for the start seed). Task readout: `logit(" True") −
  logit(" False")` (both must be single tokens; asserted). Recorded: whether L32 prefers the labelled answer.

Set B exists only to have the same prompts as the other two models. L32 was not trained to settle on such
prompts; its drift at loop 64 is recorded and contexts with last-step relative drift above 1e-3 are dropped in a
sensitivity analysis, as for Huginn.

## 4. Directions (per context, all normalised to unit length; family codes as in `ouro_app.py`)

With `ell` the logits at `q` as a function of `x_q` (through coda, final norm and unembedding):

| code | family | number | definition |
|---|---|---|---|
| 0 | task | 10 (A), 1 (B) | gradient of the task readout |
| 1 | C1 | 8 | gradient of `ell_a − ell_b`, `a, b` uniform over ordinary tokens, task tokens excluded |
| 2 | C2 | 8 | the same among the 64 highest-logit ordinary non-task tokens at `x*` |
| 3 | C3 | 8 | gradient of `xi · ell`, `xi` Gaussian on the ordinary tokens, centred on them (task tokens included) |
| 4, 5 | probes | 16 | Gaussian unit vectors (4 with code 4, 12 with code 5) |
| 6 | C3flip | 8 | the C3 directions with independent random coordinate signs |

"Ordinary tokens" = ids below the tokenizer's base vocabulary size that are not special or added tokens.
Seeds: `cseed = 9_000_000 + pid` for C1–C3, `seed = 7_000_000 + pid` for the probes and the sign flips.

For each direction: `w_1 = J^T u`, `w_2 = J^T w_1` by vector–Jacobian products (autograd), the 3 × 3 Gram matrix
of `(u, w_1, w_2)`, the start Gram `U U^T`, `x1 = U (J^T U)^T`, `x2`; an adjoint check `(J^T u)·r` against
`u·(J r)` with `J r` by central finite differences (float64, step `1e-6 · ||x*_q||`, `r` a fixed random unit
vector per direction); gradient norms before normalisation; cosine of each direction with `x*_q`.

Cross-run identity (Set A): the ten task gradients must equal the `G` saved for the same problems by the lever
prediction run `results/llever_pred-20261110-172208` (relative difference below 1e-6).

## 5. Analysis (fixed now)

Statistics exactly as `analyze_ctrlprobe.evaluate` (per context: `tau` = mean probe `alpha`; Proposition R
standard deviation; excess `alpha − tau`; `Z`; per-family median excess `Delta_F`; share `rho_F`; paired
task-minus-control difference; fraction below all 16 probes; context bootstrap, 5,000 resamples, 95% intervals)
and the descriptive `rotation` block of `analyze_ouro.py` (one-step norm, self-cosine, retention in the sampled
read span, compression).

Gates (no reading if any fails, per set): run complete for the set; arrays finite; unit starts (1e-9); `x1`
diagonal equals `alpha` from the Gram (1e-9); gradient norms positive; G1 below 1e-6; median adjoint error below
1e-4 in every family; `tau(B^2) > 0`; Set A only: cross-run identity, median last-step drift below 1e-6, and at
least 36 of 40 problems answered correctly at `x*`.

**Primary reading: Set A, family C3**, with the rule of `analyze_ouro.primary` unchanged:

- STRONG DEFICIT: interval of `Delta_C3` entirely below 0, `−median Delta_C3 / median tau ≥ 0.5` and at least 90%
  of C3 directions below all 16 probes of their context (if median `tau ≤ 0.05`, a median `Z ≤ −10` replaces the
  relative threshold; the branch used is reported);
- NOT DETECTED: interval of `Delta_C3` not entirely below 0 (no equivalence margin, so not evidence of absence);
  the sub-label POSITIVE EXCESS is added if the interval is entirely above 0;
- WEAK DEFICIT: otherwise.

Secondary, reported with the same rules and no extra verdict: the reading of `analyze_ctrlprobe` for C1, C2, C3
against the task readouts (SHARED / TASK-SPECIFIC / PARTIAL / UNCLEAR); the coordinate control (C3flip deficit
at least 75% of the C3 deficit: COORDINATE-WISE; at most 25%: ALIGNMENT; otherwise MIXED; meaningful only if a
C3 deficit is detected); all of the above on Set B; `tau` itself with its interval.

What the readings mean for the steering write-up, fixed now:

- STRONG DEFICIT on Set A: L32 is on Huginn's side of the Huginn/Ouro contrast under this protocol; the one-loop
  loss of a readout-gradient push is a read-space property and not a generic one.
- NOT DETECTED with `tau` near zero (interval inside ±0.05): the loop returns almost nothing of *any* direction
  to itself, so the lever result on L32 does not single out the read space; the write-up must say so.
- NOT DETECTED otherwise, or WEAK DEFICIT: reported as measured, no claim of a shared mechanism.

Whatever the reading, the comparison across models remains a comparison of protocols (Huginn and L32: settled
state of a re-injected recurrent core read through a nonlinear coda; Ouro: last loop of a drifting whole-stack
loop with a linear readout).

## 6. Limits

One model revision, 40 + 40 contexts, the own-position block only, sampled read directions (C3 weights
directions by how strongly the output reads them and can miss weakly read ones), `tau` estimated from 16 probes.
Set B contexts may not be settled. No statement about why a deficit is or is not present.

## 7. Cost control

One container, Set A first, then Set B; function timeout 450 s, internal deadline 330 s after which the loop
stops and the partial arrays are returned (an incomplete set gets no reading). The smoke run (3 contexts per
set, all directions) reports seconds per context; the production run is launched only if the projection for
both sets fits 300 s, otherwise Set B is cut to the number of contexts that fits (decided from the smoke timing
alone, before production data exist).

## 8. Changes

Draft 1.1 (after the smoke run `lread-smoke-20261110-220912`, before any production run). The Set A gate "median
last-step drift below 1e-6" was a slip: that threshold is the lever design's gate on the unpushed state after
64 + 48 loops. At loop 64, where this run reads the state, the lever runs themselves have median drift 2.1e-6
(maximum 2.1e-4) on the same 40 problems (`results/llever_eval-20261110-172520`, array `base_drift`), so the gate
as written would have failed for a reason unrelated to this measurement. Replaced by: median drift below 1e-5,
and the length of `x*_q` equal to the one saved by the lever prediction run for the same problem (relative
difference below 1e-9), which together with the gradient identity checks that the state is the lever runs'
state. The drift-sensitivity analysis (contexts with drift above 1e-3 dropped) is reported for both sets.
Smoke timing: 1.4 s per context (A), 1.8 s (B), setup 26 s; the projection for 40 + 40 contexts is about 155 s,
so Set B is not cut. The smoke run's return value failed to unpickle locally (a version object); the saved
outputs were fetched from the volume and the return value now contains plain types only.

Draft 1.2 (after the pre-run review `results/lread_prerun_review.md`, before any production run). The numerical
classifier of section 5 is unchanged; what it licenses is narrowed, and this replaces the three "meanings" there:

- STRONG DEFICIT on Set A: the sampled C3 directions have a strong signed deficit under this rule, so L32 is on
  Huginn's side of the contrast under this protocol. A signed self-gain relative to the trace does not establish
  damping, an explanation of the lever result, or a property of the whole read space.
- NOT DETECTED with the interval of `tau` inside ±0.05: the probe-estimated trace is near zero and no C3 deficit
  was detected. This does not establish uniformly small self-gains (eigenvalues of opposite sign can cancel).
  The write-up must then not present the one-loop loss of a readout-gradient push on L32 as something that
  distinguishes the readout from a typical direction, and must report the absolute self-gains, one-step norms
  and self-cosines of probes, C3 and task directions, with any POSITIVE EXCESS sub-label.
- Otherwise: reported as measured, with the absolute self-gains, norms and the branch used.

The Z branch can call a tiny absolute deficit STRONG when the probe variance is tiny; the absolute values are
therefore always reported next to the label. Descriptive additions to the summary: quantiles of the absolute
self-gain per family; Set B accuracy and drift distribution; adjoint medians recomputed on the drift-filtered
subset. The state identity gate now compares the whole vector `x*_q` with the one saved by the lever prediction
run (relative difference below 1e-9), and Set A must be exactly the 40 lever evaluation problems.

Cost: the function is killed at 300 s (five GPU-minutes), computation stops at 235 s; the projection uses the
slowest smoke context of each set times 1.25 against that compute deadline. A failing context stops its set but
keeps the completed contexts (an incomplete set still gets no reading), and the payload is returned even if the
volume copy fails. The production run requires a smoke run of the same code and model pin whose analysis gates
pass.
