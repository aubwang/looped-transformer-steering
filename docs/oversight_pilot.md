# Oversight-dynamics pilot (Draft 2, 2026-11-07): how do loop dynamics differ across training depths, and are looped models more predictable to steer?

Companion to docs/observability.pdf. **Results: docs/oversight_results.pdf** (runs 1–2, reviewed by Astra). Code: huginn_addition/obs_app.py; CPU dry run: huginn_addition/obs_local_test.py.
- Every Modal run is capped at 30 GPU-min (the user's instruction, 2026-11-07). Larger evaluations wait for the next
  model generation.
- Draft 0 was reviewed by GPT-6 Astra (xhigh): smoke GO WITH CHANGES, run1 NO-GO. Draft 1 made those changes.
- Draft 2 changes the arithmetic to float64 and widens the Part C dose and KL ranges, after the first smoke run
  (validation only; below). Analysis code: huginn_addition/analyze_obs.py.

**What the first smoke run showed (obs_smoke-20261107-210456; L4 and D, 2 prompts, fp32; validation only).**
- G1 passed (1e-6), G2 passed exactly, accuracy 2/2 for both, timing 12.5 s per prompt for L4 and 0.4 s for D.
- **G4 failed on both models**, for a numerical reason:
  - For random directions the linear-prediction error was *flat* in the dose (L4 about 0.1, D about 0.03, from 3e-3
    to 3e-2). That is the signature of roundoff noise in the base finite difference at 1e-3. Real curvature makes the
    error grow with the dose, as it did for the gradient-aligned directions (2e-3 at 3e-3, 1.3e-2 at 1e-2).
  - The outputs are very confident (answer log-probability about -1e-4 for L4), so a random push moves the Fisher
    norm only slightly: about 1e-11 at dose 1e-3, against an fp32 floor of about 2.5e-12.
- **A2 was mostly below the floor.** L4 contracts fast (state response 0.24 after one loop, about 1e-3 after eight).
  At σ = 1e-3 the raw Fisher response fell below 100 × floor after about 4 loops: 14% of the numbers were valid, and
  the primary summary A was undefined for every prompt.
- **Random pushes did not reach the KL targets.** At dose 1, random directions gave KL of 1e-6 to 1e-3 (L4) and 3e-4
  to 4e-3 (D), so the targets 1e-2 and 1e-1 were unreachable and 1e-3 nearly so.
- L4 converges: residual at the fp32 floor (about 1e-6) by about loop 30, and different starts agree to KL about 1e-14
  by loop 16.

**Changes in Draft 2.**
- **Float64 throughout**: a high-precision evaluation of the released checkpoints, with the RMSNorm arithmetic patched
  (both RMSNorm classes cast to float32 internally; the patch computes the same formula in float64; on CPU it differs
  from the float32 original by 9e-8, acceptance threshold 1e-6). This is appropriate for smooth local dynamics; it
  makes no claim about bitwise fp32/bf16 behaviour. D is pinned to SDPA attention (transformers' eager Llama attention
  takes its softmax in float32), and the code asserts it. Rotary cos/sin are still computed in float32: they are fixed
  per-position constants shared by every row, so they add no perturbation-dependent roundoff. On the A100, float64 matrix
  products run at the same nominal rate as strict float32. Expected floors: about 1e-30 in Fisher units. The local
  test now fails if any Fisher floor exceeds 1e-18 (an fp32 cast would give about 1e-12).
- Part C doses gain ε = 3; KL targets become 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1 (all reported).
- The rest of the design is unchanged, including the G4 thresholds and the 100 × floor rule.

**Smoke 2 (obs_smoke-20261107-211159; float64): clean.** G1 1.3e-15; G4 fd median 2.0e-3 (L4), 2.1e-3 (D); Fisher
floors 1e-28 to 1e-30; A2 100% valid; 13.4 s per prompt for L4.

**Frozen analysis rules (amendment of 2026-11-07, after Astra's review of Draft 2: run1 GO WITH CHANGES; all changes
are in the analysis and need no new smoke).** Implemented in huginn_addition/analyze_obs.py and tested on synthetic
inputs in obs_local_test.py.
- Ratios between looped models put the smaller training depth in the numerator: the primary is log(A_L4 / A_L32).
- A: every F_k, k = 1..32, in all four directions must clear 100 × floor. The low-residual thresholds stay as written
  (both residuals < 1e-3, maximum continuation drift < 1e-2, endpoint KL < 1e-3); they are not tightened now that
  float64 residuals are smaller.
- κ: needs F_1 and F_8 positive and above floor in every direction; otherwise NaN.
- Curve crossing is descriptive (median log ratios of both signs across k on the jointly eligible prompts), not a test.
- Part C: a direction is unusable if its base finite difference (dose 1e-3) is below 100 × floor. The bracket is the
  first adjacent increasing-KL pair, scanning upward from 3e-3 to 3. No extrapolation.
- Cross-model Part C comparisons use only directions matched in both models: at least 2 of the 4 random directions and
  both gradient signs. Per prompt: the median of the interpolated errors (not of their logs), then the log ratio.
  Energies are compared on the same directions and prompts. All six KL targets are exploratory; there is no headline
  target.
- Energy is r ‖εu‖² in raw site units (‖u‖ is recovered as the stored ‖u‖/ρ times ρ), plus the adapter-transformed
  version for looped models.
- FLOPs per answer are approximate matmul counts from the recorded parameter counts.
- The two smoke prompts (uids 0, 1) are among the 16, so every comparison is also reported without them.
- Run2 contains no D: its Part C comparisons reuse run1's D (`--ref`), after checking prompt ids, answers and settings.
- Code at freeze (SHA-256 prefixes, before run1): obs_app.py 72912126fe7ff3d3 (also recorded in each run's config),
  analyze_obs.py ccae812e107ded81.

**What changed from Draft 0, and why.**
- Draft 0 derived a one-sided prediction (perturbations fade faster at training depth 4) from Theorem 1. **That
  derivation was wrong.** Theorem 1 bounds the *amplitude* with which a slow mode can show in the output; it says
  nothing about *persistence*.
  - Counterexample (Astra): give one scalar mode a readout gain a with a² = 10⁻⁶ / m_p(λ²).
  - Take λ = 0.95 for the depth-4 model and λ = 0.8 for the depth-32 model.
  - Both have exactly the same Ω_p, yet the persistence ordering is reversed. This can happen at exact fixed points.
- So no outcome of this pilot can confirm or falsify Theorem 1. It is an **empirical association study** of released
  checkpoints.
- The checkpoints differ in operating point, readout, compute, curriculum and gradient horizon, so a difference between
  them is not evidence that training depth *caused* it.

## Models (float64; RMSNorm patched to float64)

The retrofitted family of McLeish et al. (arXiv 2511.07384), all built from Llama-3.2-1B. Same data (Nemotron-CC-Math,
about 50B tokens), same optimizer recipe:

| Name | HF id (revision) | Notes |
|---|---|---|
| L4  | smcleish/Recurrent-Llama-3.2-train-recurrence-4  (c466903f) | prelude 4, core 6, coda 4 layers; linear adapter on [state, prelude output] |
| L8  | smcleish/Recurrent-Llama-3.2-train-recurrence-8  (45f95e3b) | same |
| L16 | smcleish/Recurrent-Llama-3.2-train-recurrence-16 (157b949f) | same |
| L32 | smcleish/Recurrent-Llama-3.2-train-recurrence-32 (a5f6f126) | same |
| D   | smcleish/Llama-3.2-non-recurrent-posttrained (004e96ac) | the 16-layer feedforward partner, same data and recipe |

**Training depth distribution** (checked in `train.py` of github.com/mcleish7/retrofitting-recurrence):
- The number of loops is p = 1 + Poisson(Λ), where Λ is lognormal with σ = 0.5 and E Λ = M, the *mean Poisson rate*.
  The mean loop count is M + 1.
- Gradients flow through the last min(p, b) loops, with b = min(M, 8). So L4's final gradient horizon is 4 and L32's is 8.
- The start is truncated-normal (std 0.0087, cut at 3 std), drawn fresh each step.
- The released runs ("Figure 7") ramp M up from 1 over the first 75% of training (a 1 − sqrt schedule). Shallow depths
  were therefore also trained in the deeper models. The final-depth distribution and the curriculum mixture are reported
  separately, and both only as descriptions.
- Generating function: m_p(x) = x · E exp(Λ(x − 1)). At x = 0.81, m_p = 0.41, 0.22, 0.081, 0.016 for M = 4, 8, 16, 32
  (Monte Carlo; Astra confirmed it by quadrature).

## Prompts

- 16 few-shot addition prompts. Each has four demonstrations "a + b = c", then the query "a + b = " (the trailing space
  is its own token). BOS comes first.
- 8 use 2-digit operands and 8 use 3-digit operands. All sums are below 1000, so every answer is one token (asserted).
- **Order:** 2-digit and 3-digit prompts alternate, so any prefix is balanced. The smoke run uses the first two (one of
  each).
- The answer position q is the last prompt token.
- Every model gets the same prompts and the same seeds.
- Directions are seeded by the experimental identity (prompt, depth, direction), never by a model-dependent index.

## Part A: loop dynamics (looped models only; descriptive and exploratory)

**Notation.**
- x_t is the full-sequence state after t loops; z_t are the logits at q; π_t = softmax(z_t).
- The Fisher norm of a logit change Δ is ‖Δ‖²_H = Σ π Δ² − (Σ π Δ)², where π is the unperturbed distribution at the
  same depth.

**A0, base trajectory.** One fixed start x_0, drawn from the training distribution, for t = 0..64. Recorded:
- answer log-probability, top-1 token, entropy;
- state norms;
- the residual ‖x_{t+1} − x_t‖ / ‖x_t‖, over all positions and at q;
- x_t[q] for t ∈ {1, 2, 4, 8, 16, 32, 64}.

**A1, start robustness (empirical).** 8 further starts from the training distribution, t = 1..64. Recorded:
- KL(π_t^base ‖ π_t^start) and the Fisher norm of the logit change;
- top-1 agreement;
- relative state distances.

This is finite-size, nonlinear and averaged over random directions. It is **not** an estimate of Ω_p, which is a
worst-direction quantity of the linearization; in the linear regime an isotropic average estimates a trace.

**A2, local perturbation response.**
- **Where:** at depths r ∈ {M, 32, 64} (deduplicated).
- **The perturbation:** δ[pos] = σ ‖x_r[pos]‖ ξ[pos] / ‖ξ[pos]‖, with σ ∈ {1e-3, 1e-2}.
- **Directions:** one bank of 4 directions ξ, reused at both σ.
- Run k = 0..32 more loops.
- Estimand: the perturbation covariance is Cov(δ/σ) = blockdiag(‖x_r[pos]‖² I / d). So the response estimates a
  **covariance-weighted trace** at the answer position only, not the isotropic trace of the note. Training supervises
  every position, but only q is read here.
- Recorded:
  - the relative-normalized observable response F_k = ‖z_{r+k}^pert − z_{r+k}‖²_H / σ²;
  - the absolute-normalized response F_k^abs = F_k / ‖x_r‖²_F, since ‖δ‖_F = σ ‖x_r‖_F;
  - the KL;
  - the relative state responses (all positions, and q).
- The base run's drift through the continuation is also recorded: the state distance and KL(π_r ‖ π_{r+k}).

**Low-residual subset.** A prompt and depth qualify when:
- the full-state residual and the q residual are both below 1e-3; and
- the base continuation stays put: state drift below 1e-2 and KL(π_r ‖ π_{r+32}) below 1e-3.

This subset is called "low residual", **not** "fixed point": slow modes can have small residuals far from equilibrium.
Excluded prompts are reported.

**Primary summary (exploratory association; two-sided; the theory makes no directional claim).**
- Observable susceptibility over a common horizon, at the common depth r = 64, in the low-residual subset:
  A = mean over the 4 directions of (1/32) Σ_{k=1..32} F_k, at σ = 1e-3.
- Reported per prompt and model, in both normalizations.
- A prompt's A is valid only if every F_k it averages (k = 1..32, all four directions) clears its floor.
- Comparison L4 versus L32:
  - per-prompt log(A_L4 / A_L32);
  - its median, with a bootstrap CI over prompts (prompts are resampled, never directions);
  - only jointly valid pairs.
- Also reported at r = 32 and at each model's own r = M.

**Secondary summaries.**
- Persistence κ = F_8 / F_1, per prompt as the geometric mean over directions; state persistence uses (Δx_8 / Δx_1)²
  so that both are on the squared scale.
- The full curves F_k.
- The ratio of state persistence to observable persistence: Theorem 2 says unobservable directions are not *forced* to
  persist, not that they do.
- Accuracy against depth.
- Whether L4 still sits in a low-residual state at r = 64. Corollary 1b guarantees nothing beyond its training depths.

**Heuristic (not a prediction of the theorem).** Suppose each model's training kept its *own* depth-weighted start
sensitivity at a similar level. That assumption is untested, and A1 does not establish it. Then L4's response would be
constrained at short horizons and L32's at long ones, so the response curves could cross. We report whether they do.

**Validity per number.**
- A Fisher response counts only if its raw value (before dividing by σ²) is at least 100 × the larger of two floors,
  both recorded for every prompt, depth and k:
  - the duplicate-row discrepancy;
  - the discrepancy between a single-row run and the in-batch reference.
- Otherwise the number is marked "below floor" and not used.
- All ratios are NaN when the denominator is not positive. No silent regularization.

**Sample size.** At most 16 prompts. A claim needs at least 12 jointly valid pairs; with fewer, the comparison is
reported as inconclusive. For scale: with 16 pairs and no ties, a one-sided sign test needs 12 or more favorable signs
for p < 0.05.

## Part C: an interface-level intervention comparison (looped versus feedforward)

**Scope.** This is not an isolated test of recurrence:
- The looped model receives repeated forcing through the adapter, and much more downstream computation.
- Changing the prelude output changes the loop map F(x, e) itself; it is not the additive push u of Proposition 3.
- Whole-trajectory finite differences do not test frozen-Jacobian prediction.

The comparison answers only this: at the same layer of the same parent model, trained on the same data, how linear is
the output response to a push?

**Site.** The output of layer 4 at position q.
- In a looped model this is the prelude output, which re-enters every loop through the adapter (a persistent push).
- In D it is the residual stream after layer 4 (a one-time push through the remaining 12 layers).

**Directions (6 per prompt and budget).**
- 4 random directions: u = s ⊙ g. Here g is standard normal (same seeds for every model), and s is the per-coordinate
  standard deviation of the site over all non-BOS tokens of the 16 prompts, using that model's own statistics.
- 2 gradient-aligned directions: ±s² ⊙ ∇ log π(answer), the steepest ascent and descent in the diagonal Mahalanobis
  metric.
- Every direction is scaled to the **same diagonal-Mahalanobis norm**, ‖u / s‖ = √d. Dose ε = 1 is then a push about
  as large as typical token-to-token variation.
- Random and gradient-aligned directions are analysed separately.

**Doses.** ε ∈ {1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1, 3}.
- The linear prediction uses the finite difference at ε₀ = 1e-3: Δz_lin(ε) = (ε / ε₀)(z(ε₀) − z(0)).

**Budgets.** Each looped model at r = M and at r = 32 (deduplicated). D has no budget.

**Recorded.**
- KL, and the Fisher norm of Δz;
- the relative linear-prediction error ‖Δz − Δz_lin‖_H / ‖Δz‖_H;
- the actual and predicted change of the answer log-probability;
- top-1 flips;
- the Euclidean push size ‖εu‖ / ρ, where ρ is the RMS site norm;
- for looped models, the adapter gain ‖W_e u‖ / ‖u‖, where W_e is the adapter's prelude-output block.

**Reported alongside.**
- Loop count and FLOPs per answer.
- Cumulative raw intervention energy: r ‖εu‖² for a persistent push, ‖εu‖² for D.
- The adapter-transformed energy r ‖W_e εu‖².

**Matching by effect (fixed in advance).**
- KL targets: 1e-6, 1e-5, 1e-4, 1e-3, 1e-2 and 1e-1 (Draft 1 had only the last three; random pushes rarely reach
  them).
- For each direction, log(linear error) is interpolated linearly in log KL between the two bracketing doses. This is
  done only if KL increases between them and the target lies inside that direction's achieved range.
- Brackets use doses 3e-3 and above: at the base dose 1e-3 the linear error is zero by construction.
- A target is compared between models only when both have it inside their ranges. Unmatched targets stay unmatched,
  with no extrapolation.
- Summary: the per-prompt median over directions; then the median of the per-prompt log ratio (looped / D), with a
  bootstrap CI over prompts. Two-sided and exploratory.

## Gates

- **G1 (hard, smoke and every run).** The loop harness (prelude; adapter + core block per loop; coda; final norm;
  head) reproduces `model(ids, input_states=x0, num_steps=r).logits` at q for r ∈ {1, 4, 32}, on the first two prompts.
  Relative error must be ≤ 1e-4, and the latent state must match. Measured on CPU tiny copies: 4e-7.
- **G2 (hard).** For D, the hooked site equals `output_hidden_states[4]` exactly, and a zero push reproduces the logits
  to within 1e-6 relative.
- **G3 (report).** Each answer is one token (asserted). Base greedy accuracy per model and depth.
- **G4 (gate on the run's status).**
  - Two checks on the Part C finite differences:
    - the 1e-3 and 3e-3 derivatives agree, in Fisher norm, to median ≤ 1e-2 and 90th percentile ≤ 5e-2;
    - the directional derivative of log π(answer) from the finite difference matches the autograd gradient, with the
      error scaled by its Cauchy–Schwarz bound, to the same thresholds.
  - 1e-4 was dropped in favour of 3e-3, because fp32 roundoff at 1e-4 is about 1e-2 relative.
  - The paired A2 linearity ratio F_1(σ = 1e-2) / F_1(σ = 1e-3) is reported, not gated: nonlinearity is a property,
    not a bug.
- **Floors (report and use).** Duplicate in-batch reference rows, and single-row versus in-batch reference
  discrepancies in Fisher units, for A1, A2 and C.
- **Status.** A container whose status is not "complete", or which has a failed gate, marks the whole run as not
  clean. The launcher then exits with a nonzero status.

## Budget and stopping

- **Hard limit:** a 480 s Modal timeout per container.
- **Cooperative deadline:** total_s − 45 s of reserve.
  - Smoke: total_s 360.
  - Runs: total_s 400.
- **Time checks:** every 16 loops in A0/A1, before each A2 depth, and before each Part C budget.
- **When to start a prompt:** only if the longest previous prompt time still fits before the deadline.
- **Saving:** results are saved and committed after every prompt; a stop or an error still saves and reports.
- **Smoke:** L4 and D, two prompts, all parts. Worst case 2 × 8 = 16 GPU-min; expected about 5.
- **Run 1:** L4, L32 and D, 16 prompts. Worst case 3 × 8 = 24 GPU-min, plus startup.
  - If smoke timing shows L32 cannot finish 16 prompts in about 350 s, the work is cut **uniformly for every prompt
    and model** before run1: fewer A1 starts, then fewer A2 directions. There is no model-dependent prefix.
- **Run 2:** L8 and L16, same settings, worst case 16 GPU-min.

## Deferred (bigger evaluation)

- P2: Arnoldi on Jᵀ from Fisher read covectors (a worst-direction estimate).
- P3: monitor transfer versus observability cost, with real correctness labels.
- The Huginn blink test of Corollary 1a.
- OLMo, Ouro and LoopUS.
- Several tasks and many prompts.
- Any causal claim about training depth. That needs controlled training runs, not released checkpoints.
