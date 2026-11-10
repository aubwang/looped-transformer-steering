# Matched-pair monitor experiment (Draft 1.3, 2026-11-09)

Follows docs/oversight_results.pdf (the pilot) and docs/monitor_protocol.md (an earlier, narrower plan, never run).
Status: item count and stage-1 screen completed; no main data collected. External review of the frozen package:
"go with changes"; these amendments precede any main data (amended in Draft 1.3). Draft 0 was reviewed by GPT-6 Astra (xhigh): stages 0–1 GO WITH CHANGES,
stage 2 NO-GO until the changes are frozen and the collection gates pass. Draft 1 makes those changes (listed in
section 13) and two of our own, from a look at the pilot data and a literature check (docs/monitor_lit_2.md).
Implementation contract: docs/mon_spec.md.

## 1. The question

The pilot measured how outputs react to pushes on the state. It never tested a *monitor*: a device that reads the
model's internal state and tells an overseer something useful. This experiment does.

> On a looped model and its non-looped partner, trained on the same data from the same starting weights, which is
> easier to monitor from its internal state? And does the loop give a monitor anything a non-looped model cannot?

"Monitor" always means a **linear probe**: a fixed vector m and offset b; the score of a state x is m·x + b.

| | Name | Plain statement |
|---|---|---|
| M1 | Error monitoring | From the state, can a probe tell that the model's answer will be wrong? Does it add to the model's own confidence? |
| M3 | Budget portability | Does a probe fitted at one loop budget keep working when the model is run for fewer or more loops? Can the loop's own dynamics forecast when it will not? |
| M4 | Dynamical signals | The loop produces label-free signals (has the state settled, is it oscillating, did the answer change late). Do they predict errors beyond confidence? |

What the literature check found (docs/monitor_lit_2.md): correctness probes, convergence signals and agreement across
random starts are each known in looped models. Not found: the matched looped/non-looped comparison, transfer to loop
budgets the probe was not fitted on, and a dynamics-based forecast of monitor stability. Those are what this
experiment is for.

**What a result here can mean.** L32 and D are two trained checkpoints. A difference between them is a fact about this
pair, not a proven effect of looping: they also differ in compute per token and in which problems they get wrong.
Section 6 has the controls; section 11 the limits.

## 2. Models

Recurrent-Llama-3.2 (McLeish et al., arXiv 2511.07384), same revisions as the pilot (`obs_app.py`, `MODELS`): looped
L4, L8, L16, L32 (4 prelude layers, a 6-layer core repeated r times, 4 coda layers; trained with a mean of 4, 8, 16,
32 loops) and the non-looped partner D (16 layers). Float32, TF32 disabled.

Notation. X_t: the state of the whole sequence after t passes of the core (X_0 is the random start); x_t: its vector
at one token position. The **budget** r is the number of loops before the coda reads out the answer. M: the model's
training mean. For D, h_l is the output of layer l (h_0 the embedding).

## 3. Tasks

**Format.** Worked examples of the same family, then the query; the answer is one token. Measurements at the last
prompt token (position −1) and, as a robustness check, the one before it (−2).

**Candidate arithmetic families** (four worked examples; answers below 1000):

| Family | Query | Rule |
|---|---|---|
| add3 | a + b + c | each 10..99 (pilot screen: 66–70% correct) |
| addsub | a + b − c | a, b in 100..899, c in 10..999, result in 1..999 |
| muladd | a × b + c | a in 12..99, b in 3..9, c in 10..99, result ≤ 999 |
| add4 | a + b + c + d | each 10..99 |
| sub3 | a − b − c | a in 200..999, b, c in 10..99 |
| add3w | a + b + c | each 100..799, sum ≤ 999 |

**Candidate multiple-choice families** (two worked examples; answer is a letter): ARC-Easy, then ARC-Challenge.
Four-option items only; options reshuffled with a fixed seed so gold letters are approximately, not exactly,
uniform; whole prompt at most 160 tokens (including BOS). The number of usable items is counted before the screen.

**ARC-Easy is a short-prompt, two-demonstration-compatible subset, not representative ARC-Easy accuracy.**
In `results/mon_count-20261109-203859/items_master.json` (dataset commit
`210d026faf9955653af8916fad021475a3f00453`), ARC-Easy has 5,197 rows → 5,173 with four options → 5,154 after
deduplication → 4,010 examined → 1,596 kept: only 39.8% of examined items fit. Kept prompt lengths are 99–160 tokens,
median 143. Selection depends on demonstration lengths as well as the question. Gold A/B/C/D counts are
395/422/397/382. Option permutation and demonstration selection use separate random generators initialized with
the same per-item seed; their draws are coupled, a limitation to fix in a future generator revision. The master
list is frozen for this experiment.

Each of the six arithmetic families also has 1,596 items. ARC-Challenge has only 477, so is not screened and is
not eligible.

**Leakage rules.** Queries within a family are pairwise distinct *up to reordering of interchangeable operands* (so
12 + 34 + 56 and 34 + 12 + 56 cannot both appear). Worked examples are drawn from a pool that shares no problem with
any query, screen or main. A multiple-choice item is used once.

**Family screen (stage 1).** 96 problems per sufficiently large candidate family, L4, L32 and D only (the selection
models), the full collection path. Only families with at least N_SCREEN + N_MAIN_MIN = 96 + 800 = 896 master
items are screened. A family is
*eligible* if accuracy lies in [0.30, 0.90] (multiple choice [0.40, 0.90]) for each of D, L32 at r = 32 and L4 at r = 4,
and it has at least 896 master items. Select the first three eligible arithmetic families in table order and the first
eligible multiple-choice family. Fewer than two eligible in total: stop and redesign. No eligible multiple-choice
family: the claim is narrowed to arithmetic and said so. All screen results are published. Selection uses accuracy
only. Probes were fitted on screen problems for software tests, solver benchmarks and diagnostic checks of the
amended forecast (amended in Draft 1.3). No selection of families, sites or hyperparameters used these fits beyond
the solver settings and the T1 amendment already listed in Draft 1.2. Main problems are disjoint and have not been
collected or used for fitting. Conclusions concern the selected families.

**Main set (stage 2).** Up to 1,500 problems per selected family, with a frozen minimum of 800 enforced by the entry point (amended in Draft 1.3). The final number is set from measured stage-1
throughput so that every launch fits the cap (section 10); it is frozen before stage 2.

## 4. What is collected (forward passes only)

One run of each model per problem. No gradients, no interventions. Full list and array layout: docs/mon_spec.md.

**Looped models**, start seeded per problem, 33 loops:
- x_t at position −1 for t in T = {1, 2, 3, 4, 6, 8, 12, 16, 24, 31, 32, 33}; at position −2 for t in {8, 32}.
- Coda output at position −1 at budgets {4, 8, 16, 32}; prelude output at both positions.
- Readout at every t in T: the five most probable tokens with log-probabilities, entropy, log-probability of the true
  answer (for labels only), probability mass on valid answers (numbers, or the four letters), the four letter
  log-probabilities for multiple choice.
- At every loop t = 1..33, for the whole sequence and for position −1: step size ‖X_t − X_{t−1}‖, two-step size
  ‖X_t − X_{t−2}‖, state norm, cosine between the last two steps. Unnormalized Euclidean norms, backward-looking only
  (nothing at budget r uses loops after r).
- Exact whole-sequence displacement `disp_full` to budgets 32 and 64 from R_DISP = {2, 4, 8, 16, 32};
  target 64 is deep-only and target 32 from 32 is undefined. Per-position steps `d_pos` at every loop, and state
  norms `n_pos` at R_NPOS = {8, 16, 24, 32}, NaN-padded to L_MAX = 160 (amended in Draft 1.3).
- **Deep subset:** the first 256 main problems of each family run to 64 loops, adding t = 48, 63, 64
  (16 per family in the screen). Membership uses rank `i − i0` in the whole run set, with `i0 = 0` for the screen
  and 96 for main, shared across shards.

**D:** h_l at position −1 for all l = 0..16; at −2 for l in {8, 12, 16}; the same readout summary from the true output
and, per layer, from the "logit lens" (h_l pushed through the final normalization and output matrix); layer step sizes
and norms.

States are stored in float16; the first 32 main problems per family (8 in the screen), by the same
whole-run rank, also in float32 for `x_last` (D: `h_last`) only. Float32 copies of `post`, `pre` and `x_prev` are NOT saved:
the review recommendation was declined to keep the loader simple. Looped rounding checks therefore cover `x_last`
only; frozen-probe score and T1 stability comparisons use those copies (amended in Draft 1.3). Step sizes are
computed on the GPU at full precision. Token hashes (`tok_hash`: int64, first 15 hex digits of SHA-256 of the JSON
token-id list), seeds, option
permutations, model revisions and code hashes are saved across the chunks, master list and metadata.

**Labels.** c(model, r, problem) = 1 if the most probable token at budget r is the true answer.

*Change from Draft 0, from the pilot data.* On the pilot's prompts the state has essentially stopped moving by 32
loops (median relative step 10⁻⁵ to 10⁻⁶) and accuracy is the same at 4, 32 and 64 loops. So running everything to 64
would mostly re-measure a settled state. The loop count is cut to 33, with 64 kept for the deep subset, and the
budget-transfer question is turned around: the informative direction is *down* (2, 4, 8 loops), where the state is
still moving.

## 5. The monitors

Fitted offline on CPU, separately within each family.

**Probe (primary).** Standardize each coordinate on the training part. Minimize mean logistic loss + λ‖w‖²/2 (offset
not penalized), λ in {10⁻⁶, 10⁻⁵, …, 10²}.

**Nested fitting.** Five outer folds over problems, stratified by the fit's label except in M1-a: one shared assignment
per family and seed balances the four joint L32/D correctness categories. Supplied folds must contain both classes
for each label; otherwise M1-a is skipped with a reason (amended in Draft 1.3). Inside each outer training part, four inner
folds choose the site (when a site search is part of the estimand) and λ **jointly**, by mean inner AUROC; the chosen
pipeline is refitted on the whole outer training part and scores the outer test part. The AUROC is computed per outer
fold, averaged over folds, then with equal weights over all analysed families. A numeric quantity missing in any family is unavailable
in the family mean and listed under `incomplete`, never averaged over fewer families (amended in Draft 1.3). Repeated for three fixed split seeds; the spread over seeds and how
often each site is chosen are reported. "Best site" means the output of this procedure.

**Probe + other features.** One joint regression. The extra block U (confidence, splines, dynamics) is standardized
on each fit's training data and orthogonalized by singular value decomposition to unit mean square. Directions
with singular value ≤ U_RANK_TOL = 10⁻⁸ times the largest are dropped. The formerly unpenalized block now gets fixed
ridge U_RIDGE = 10⁻⁶, identically for baselines and augmented models: add U_RIDGE‖u‖²/2 to the objective above.
The intercept remains free (amended in Draft 1.3).

**Solver and failure policy** (amended in Draft 1.3). L-BFGS is followed by trust-krylov (a Newton-type method with
exact Hessian-vector products) if max |gradient| > GRAD_TOL = 10⁻⁶ for the mean objective. Our criterion defines
convergence, rather than the optimizer's success flag. Every fit is used in selection and estimation regardless
of convergence. Fit rows report outer failures as `nonconverged` and inner (selection) failures as
`inner_nonconverged`, for every fit. Contrasts report `nonconverged_outer_fits`.
Any contrast with outer failures must be flagged in the write-up.

**Sensitivity probes.** Difference of class means; logistic regression on the first 128 principal components (fitted
on the training part, not whitened).

**Baselines** (same folds):
- *Confidence*: top log-probability, top-two log-probability gap, entropy, and `valid_lm` at the same budget.
  `valid_lm` is logsumexp of token log-probabilities over valid answers: tokens for integers 0..999, or the four
  space-prefixed letters A–D for multiple choice (amended in Draft 1.3).
  Also a flexible version (four-knot spline of each), so that "added value" is not merely fixing miscalibration.
- *Output identity*: the predicted token (its digits, or the letter). The token is decoded with the table of valid
  answer tokens (integers 0–999, letters A–D) that the collection saves in its metadata (`valid_ids`), never from
  the gold answers present in the data; a predicted token outside that table gets an indicator `unknown` and
  zeros elsewhere (amended in Draft 1.3 after the second review: the earlier code inferred the table from gold
  answers, which made the feature depend on held-out ground truth).
- *Privileged problem features*: operand digits, carries, true answer digits; the correct letter. These use the ground
  truth, so no real monitor could have them. Shown to indicate how much of "which problems fail" is a property of the
  problem; it does not bound that confound.
- *Prelude / early layer*: the same probe on the prelude output (looped) or h_4 (D).

**Leak diagnostic.** One fixed label permutation per family through the entire nested pipeline; reported, not a gate,
and never a reason to rerun.

**Metrics.** AUROC (0.5 is chance), and the error rate among accepted answers when the 20% most suspicious are
rejected. Intervals: 2,000 paired bootstrap draws over problem ids within family × shared fold for M1-a, and within
intersections of participating folds for other comparisons, from stored out-of-fold scores. Fold strata are kept.
If ANY draw is undefined because a resampled fold lost a class, the whole interval is unavailable with
`undefined_bootstrap_draws` and `interval_unavailable_reason`; draws are never dropped. Each fit's AUROC has an
`interval`, budget transfer has `transfer_interval`, and contrasts have intervals; family intervals require all
analysed families. These condition on fitted scores and omit refitting uncertainty; seeds indicate split variability
separately (amended in Draft 1.3).

## 6. Estimands

### M1. Error monitoring (descriptive comparison of two checkpoints)

- **M1-a: the 2×2 table.** Rows: whose correctness is predicted (L32 at r = 32; D). Columns: whose state is read
  (x_32 of L32; h_12 of D — both are what the model's last four layers receive). All four cells on the same problems
  and folds. The diagonal is each model monitored from its own state. The off-diagonal cells separate "L32's errors
  are easier to predict" (a row effect, about the labels) from "L32's state is more informative" (a column effect).
- **M1-b: nested best site**, own labels: L32 (x_32 or coda output at 32) against D (any h_l).
- **M1-c: added value over confidence**: AUROC(confidence + probe) − AUROC(confidence), each model, and the
  difference; also against the flexible confidence baseline.
- Secondary: each looped model at its own M and at 32 (L4, L8, L16, L32); position −2; 80%-coverage error; the
  concordant subset (problems where L32 and D are both right or both wrong), with its class counts and the warning
  that it is a selected population; cross-family transfer (analysis `m1x`, specified in Draft 1.3: the diagonal probes of one family, L32 from
  x_32 and D from h_12, are applied unchanged to each other selected family; a target item's score is the mean of
  the five outer-fold models' scores; AUROC for each model and the difference L32 − D, with intervals from 2,000
  resamples of the target items within the four joint correctness categories, the same resample for both models,
  conditional on the fitted probes; the mean over targets requires every other selected family and is otherwise
  unavailable, with the missing targets listed; the drop against the own family compares like with like, the mean
  of the five probes' single-probe target AUROCs against the own-family mean of single-probe held-out AUROCs,
  because averaging five probes' scores changes rankings by itself; it is still a difference between a
  held-out fifth of one family and the whole of another, reported as a description, not as an isolated
  transfer loss).

### M3. Budget portability

- **M3-a.** Fit on x_32 with labels c(32); freeze weights, standardization and the held-out-fold 80th-percentile
  threshold at r = 32. Reject scores strictly above it; ties can change the rejected fraction (amended in Draft 1.3).
  Apply to x_r with labels c(r), r in {2, 4, 8, 16} (and 64 on the deep subset). The code additionally reports all
  saved target budgets,
  including 1, 3, 6, 12, 24, 31, 32, 33 and deep-only 48, 63, as descriptive rows. Targets lacking both classes
  in every frozen source fold are skipped with a reason (amended in Draft 1.3). Report, per target budget: transfer
  AUROC, native AUROC, native − transfer, fraction rejected, accepted-answer error, and the fraction of problems whose
  label changed. The same for each looped model fitted at its own M.
- **M3-b: stability forecast** (section 8, T1): tested item by item on held-out problems.
- Descriptive only, no summary number and no claim: the matrix "fit at depth s, test at depth s′" for L32 (loops) and
  for D (layers). D's layers are different maps, so its matrix measures how well coordinates line up across layers,
  which is not the same question.

### M4. Dynamical signals

Features at budget r, using loops up to r only: log step size and log two-step size (whole sequence and position −1),
cosine between the last two steps, last t in T at which the top token changed, number of such changes, whether the
top tokens at r/2 and r agree, change in top log-probability between r/2 and r (amended in Draft 1.3).

- **M4-a.** AUROC(confidence + dynamics) − AUROC(confidence), L32 at r = 32 and each model at its M.
- D's logit-lens trajectory uses log layer-step sizes (whole sequence and last position) and the token-change,
  agreement and top-log-probability-change features, not two-step or cosine features (amended in Draft 1.3). It is
  shown for description. D's intermediate readouts were never
  trained to be read, so a weaker result for D is not attributed to looping.

All estimands are reported with intervals whatever their sign; "not resolved at this sample size" is an allowed
outcome.

## 7. Precision

Stage 0 simulates the actual nested procedure at 10%, 30% and 50% errors and at the affordable sample
size; that table is frozen with the design. The handful of families is not a sample of tasks: per-family results are
always shown. The simulation now uses shared joint folds, the final solver and `--jobs` parallel workers
(amended in Draft 1.3).

Simulated precision (`results/mon_power_sim.json`; 40 replications per cell, d = 2048, population AUROC 0.80 for
both models and true difference 0; the real nested procedure with shared joint folds, the final solver and 2,000
bootstrap draws):

| Errors | n | SD of one AUROC | SD of the paired difference | Mean width of the 95% interval for the difference | Intervals containing the true difference | Intervals unavailable | Non-converged outer fits |
|---|---|---|---|---|---|---|---|
| 10% | 800 | 0.023 | 0.028 | 0.111 | 36/39 | 1 | 0 |
| 10% | 1,500 | 0.018 | 0.021 | 0.076 | 36/40 | 0 | 0 |
| 30% | 800 | 0.014 | 0.017 | 0.069 | 38/40 | 0 | 0 |
| 30% | 1,500 | 0.011 | 0.012 | 0.050 | 38/40 | 0 | 0 |
| 50% | 800 | 0.018 | 0.012 | 0.062 | 40/40 | 0 | 0 |
| 50% | 1,500 | 0.009 | 0.011 | 0.045 | 38/40 | 0 | 0 |

How to read it. For one family with 1,500 problems and 30% errors, a difference between the two models' AUROCs is
estimated with a standard deviation of about 0.012, and the 95% interval is about 0.05 wide: differences
of about 0.03 or more are distinguishable from zero in one family, smaller ones are not. At 10% errors the interval
is half again as wide. A mean over four families is more precise by up to a factor of two, less if family
effects differ (they are reported separately in any case). With 40 replications each standard deviation carries
about 11% relative Monte Carlo error and each coverage count about ±2 intervals, so the coverage column is
compatible with 95% throughout but cannot establish it. One interval was unavailable in the smallest cell
(80 errors among 800 problems, about 16 per fold: one of 2,000 resamples lost a class in a fold); in the
screen the selected families had error rates between 21% and 67% for L32, L4 and D (96 problems each, so only
roughly known), which puts the main data near the 30% and 50% rows, with some 300 or more items in the rarer
class at n = 1,500. The table is for
a signal of this strength (AUROC 0.80) and Gaussian features; it is a guide to the scale of the uncertainty,
not a promise about the main data.

## 8. Theory: what the loop predicts

Definitions, proofs and numerical checks are in docs/monitor_theory.typ. Statements below include review
corrections (amended in Draft 1.3).

Setup: fix the prompt. One loop is a map F on the whole-sequence state, X_{t+1} = F(X_t). ‖·‖ is the Euclidean
(Frobenius) norm over all positions and coordinates: the last position alone is not a closed system, because
attention feeds it from every earlier position. d_r = ‖X_r − X_{r−1}‖.

**T1 (stability bound).** If F shrinks distances by q < 1 on a region containing the trajectory from loop r − 1 on,
then for every h ≥ 1, ‖X_{r+h} − X_r‖ ≤ q d_r (1 − q^h)/(1 − q). For a linear probe with weight vector m (in
unstandardized coordinates, acting on one position), |s_{r+h} − s_r| is at most ‖m‖ times that. For the answer
itself: if each logit changes by at most L per unit of state displacement, the top token cannot change while the
top-two logit gap exceeds 2L times the displacement bound.
- This is the classical after-the-fact bound for contractions (and residual bounds at approximate equilibria are in
  the literature). The use as a per-item statement about a monitor is what we add.
- **It is a forecast, not a certificate.** Observed ratios of step sizes cannot establish q. Frozen rule: q̂ = the
  largest of the last eight ratios d_t/d_{t−1}, t ≤ r; no forecast if q̂ ≥ 1 or the step size is at the float32
  floor. B_i = ‖m‖ q̂ d_r (1 − q̂^h)/(1 − q̂).
- **Test (falsifiable).** At r in {4, 8, 16}, forecast B_i for h = 32 − r; on held-out problems report how often the
  realized |s_32 − s_r| exceeds B_i (violations), and how often an item forecast "decision stable" changes decision
  (false stability). No retuning. Few forecasts of stability can mean a loose bound or a large ‖m‖, not necessarily an
  unsettled state; the ratio of bound to realized movement is reported.
- **Consequence for M3-a.** For one frozen classifier with threshold τ, with η the violation rate,
  error at r′ ≤ error at r + P(|s_r − τ| ≤ B_i) + P(c(r′) ≠ c(r)) + η. About classification error, not AUROC.
  For ranking: with fixed labels, |ΔAUROC| ≤ P(|s⁺ − s⁻| ≤ B⁺ + B⁻) over pairs of one wrong and one right item,
  when the bounds hold. Both are checked with the *forecast* bounds (using the realized movement would be circular).
- D's layers are different maps: one can measure displacement between layers, but nothing extrapolates.

**T1 evaluation additions** (amended in Draft 1.3; Draft 1.2 calibration retained). The primary calibrated
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

**Positional rates** (amended in Draft 1.3). The block-triangular Jacobian implies that the asymptotic rate of the
first p positions TAKEN TOGETHER is non-decreasing in p, not the rate of each single position: diag(0.9, 0.1) is a
counterexample. Inheriting earlier slow rates requires nonzero coupling and excitation, an expectation measured
separately. Windows within 33 or 64 loops give finite-time evidence, not limits. `position_rates` reports single
positions, `prefix_*` statistics and `last_position_minus_whole_state`, in windows 8–16, 16–24, 24–32. Positions
whose end step is below 10⁻⁶ of their own state norm (`n_pos`) are excluded; a prefix is compared with the norm of
its positions together (root sum of squares). The norm source is chosen per item: an item without saved norms at
every real position (older data) uses whole-state norm / sqrt(prompt length) for each position, combined the same
way for prefixes; the output counts the items under each rule.

**T2 (two-cycles).** Huginn showed period-2 endings in 13% of prompts. A two-cycle has two distinct points, so F∘F
cannot be a contraction on one region containing both. Assume instead that F∘F contracts by q₂ on a region around each
of the two points; then ‖X_{r+2k} − X_r‖ ≤ ‖X_{r+2} − X_r‖ (1 − q₂^k)/(1 − q₂). Probe scores then may alternate
between even and odd budgets (not necessarily: the probe can be blind to the direction of the swing). Budgets 31, 32,
33 and 63, 64 are collected to see this; whether these models show it is unknown.

**T4 (theory only, planned).** Extend docs/observability.pdf from the fixed point to the transient (products of
Jacobians along the trajectory in place of powers of one). Also to try: a sharper version of T1 that uses the
one-way structure of attention (position p depends only on positions up to p) to bound the last position's movement
without paying for the whole sequence.

## 9. Cut from Draft 0

Intermediate-quantity probes (M2), the extra random starts (stage 3) and the start-variance prediction (T3), and the
transfer-efficiency summary. Agreement across starts as an error signal is prior work (Anil et al. 2022), and three
starts could not test T3.

## 10. Stages, gates and budget

| Stage | What | GPU |
|---|---|---|
| 0 | CPU dry run of the collection code on a tiny random model; analysis code on synthetic data with a planted effect and with none; the precision simulation | 0 |
| 1 | Completed family screen, 96 × 7 sufficiently large families × 3 models; also throughput by prompt length | 3 × 480 s = 24 GPU-min worst-case allocation |
| 2 | Main: 3 shards/model, 15 shard-runs in 8 launches; `collect_long` (840 s, `limit_s` 820) | ≤ 2 containers/launch, ≤ 28 GPU-min |

Stage-1 timing sets the projection below; the Draft 0 guess is withdrawn (amended in Draft 1.3).
Shards use the same problem assignment for every model, but deadlines can leave different completed subsets;
analysis uses only problems complete in every model it compares. Screen `collect` has timeout 480 s and
`limit_s` 460, with a cooperative stop at container start + 460 − 40 = 420 s. Main stops at start + 780 s.
A deadline leaves a usable completed subset, not a balanced prefix: batches rotate over families, but arithmetic
batches hold up to 64 items while multiple-choice batches contain only exactly equal-length prompts (up to 32).
Partial runs are never used for screening. Main results are keyed `model:shard`.

Right-padding is deferred until after the screen. It is valid for a causal model if each readout uses the item's
own last position, whole-sequence norms exclude padding, and start states preserve the unpadded draws.

**Gates** (a failed gate makes the run "not clean"; it is not analysed):
- G1: harness against the model's own forward at r = 4 and 32, the same full first batch and start states: all logits of
  every row, after subtracting each row's mean, agree within 10⁻⁴. Separate diagnostic `batch_single_32` compares
  the model's own forward batched versus alone for the first two items at 32 loops (no threshold); batching alone
  is varied.
- G2: D's h_16 through the final normalization and output matrix reproduces the model's logits within 10⁻⁴.
- G3: tokenization (single-token answers, expected final token, prompt length limit); multiple-choice final
  token must decode to `":"`. G3 and dtype failures leave `meta.json` with status `gate_fail`.
- G4: all stored values finite; no float16 overflow; float16 against float32 states within 10⁻³ relative on the
  per-family float32 subset (screen 8, main 32). A chunk failing G4 is dropped, not saved.

Metadata records computed `items_done` separately from `items_committed`, `committed_pids`, and `chunks`;
the first count can exceed the committed count. It includes model/tokenizer identity and gate entries `G1_items`
and `G4_checked_states`; cfg includes `items_sha256`, `run_id`, `mode`. `accuracy_counts` and `out_dir` are added
to the returned metadata (and `containers.json`), after the on-volume `meta.json` is written.

**Collection safeguards now implemented** (amended in Draft 1.3). The entry point enforces 800 ≤ n_main ≤ 1500,
exact `model:shard` parsing, unique pairs, valid indices and non-empty requested shards. Per-chunk float32-copy
counts are recorded; the rounding gate returns None (not checked) without copies, rather than a perfect 0.0.
G4 checks `n_pos` against whole-state norms; metadata includes R_DISP, R_NPOS and L_MAX.

The first main launch writes `main_manifest.json` beside the items file: families, n_main, shards, items hash,
cfg excluding run_id, collection code hash, and `analysis_sha256` for analyze_mon.py, mon_probe.py and these two
documents. Later launches refuse collection-field changes; analysis-hash changes only warn. Analysis with
`--manifest` records `analysis_matches_manifest` and validates items, shard hashes and expected problem ids.
`--allow-partial` admits committed chunks of deadline-stopped shards and reports completeness instead of asserting
it, retaining expected-id and hash checks. Completeness is checked for loaded models; absent models produce skips.

**Frozen before stage 2:** families, n, shards, every rule in sections 5–8, the stage-0 precision table, code hashes.
Later changes are listed as amendments with reasons. The code and frozen package have been externally reviewed;
these amendments precede stage 2 (amended in Draft 1.3).

## 11. What this can and cannot show

- One model family, one size (1B). "Looped versus non-looped" means this pair of checkpoints.
- Matched in data and starting weights only: D has 16 distinct layers, the looped models 14 with six reused; compute
  per token differs.
- Linear probes; two token positions, both before the answer; short prompts; a few selected task families.
- A probe that predicts errors may be reading problem difficulty. The baselines and the 2×2 table address this in
  part; they do not remove it.
- T1 is conditional on contraction, which is not verified for the real model. Its forecast is tested, not proven.
- Monitoring an honest model's mistakes is not monitoring a model that hides something.

## 12. Reading the outcomes

| Outcome | Reading |
|---|---|
| 2×2 table flat, small added value over confidence in both | State probes add little to asking the model, in either architecture |
| Column effect favouring L32's state, or M4 adds over confidence | The loop state or its dynamics carry error information the output does not; worth a second model family |
| Column effect favouring D's state | The looped state is harder to read linearly; check a nonlinear probe before concluding |
| Transfer to small budgets fails and the forecast flags it | Monitors are budget-specific, but the loop tells you when to distrust them |
| Transfer fails and the forecast does not flag it | An oversight risk specific to looped models: silent failure under a budget change |
| Transfer holds everywhere | Budget changes are benign for monitors in these models (the answer settles very early); the forecast adds little |

## 13. Changes from Draft 0 after review

1. M1 reframed as a descriptive checkpoint comparison; 2×2 cross-label table added and made the lead estimand; the
   fixed-site comparison (x_32, h_12) promoted; concordant subset kept with caveats.
2. Complete nested procedure specified (joint site and λ selection, explicit penalty and grid, per-fold AUROC, three
   seeds, joint regression for added features).
3. Leakage: disjoint worked-example pool; no reordered duplicates; problem-feature baseline relabelled as privileged;
   output-identity and flexible confidence baselines added.
4. M4 features are backward-looking only; extra starts removed.
5. Tasks: exact division dropped (only 1,195 distinct queries); CommonsenseQA dropped (five options); ARC-Challenge
   is the second multiple-choice candidate; usable items counted before the screen.
6. Budget: guess withdrawn; shards sized from measured throughput; allocations summed per launch; common paired set.
7. Gates: G1 on all centered logits at 10⁻⁴; G2 likewise; float32 subset; provenance saved; shuffled-label gate
   replaced by one permutation diagnostic.
8. PCA-128 sensitivity probe added. 9. Budget transfer kept with frozen threshold and coverage reporting; depth
   matrix demoted to description; efficiency ratio dropped. 10. D's logit lens is descriptive; more readout features
   and odd budgets saved. 11. Power promise replaced by a stage-0 simulation.
12–15. Section 8 corrected: whole-sequence norm, unstandardized probe norm, backward step size with the extra factor
   q, logit-margin condition for the answer, forecast-versus-certificate wording, violation term η in the transfer
   inequality, parity-specific regions in T2, T3 dropped. 16. Ranking bound added.

Ours: loops cut to 33 with a deep subset at 64; budget transfer pointed at small budgets; literature positioning.

### Changes in Draft 1.1 (2026-11-09, after code review)

- Corrected screen models, family size threshold, timeouts, deadlines and allocation; recorded the count run and
  ARC-Easy subset limits, letter counts and coupled random seeds.
- Corrected whole-run deep/float32 membership, gates, token hashes and metadata, and unbalanced partial output;
  documented exact-length batching, deferred padding and main result keys.
- Listed the entry-point, manifest, float32 coverage and rounding-analysis work required before main collection.

### Changes in Draft 1.2 (2026-11-09, after the screen, before any main data)

**Screen result** (`results/mon_screen-20261109-204929`, 96 problems per family, 5.1 GPU-minutes, all gates passed:
G1 ≤ 5.0 × 10⁻⁵, G2 ≤ 5.9 × 10⁻⁵, float16 against float32 ≤ 3.3 × 10⁻⁴).

| family | D | L4 at 4 | L32 at 32 | eligible |
|---|---|---|---|---|
| add3 | 0.73 | 0.77 | 0.79 | yes |
| addsub | 0.08 | 0.13 | 0.10 | no (too hard) |
| muladd | 0.20 | 0.19 | 0.16 | no (too hard) |
| add4 | 0.39 | 0.38 | 0.33 | yes |
| sub3 | 0.30 | 0.27 | 0.29 | no (L4 and L32 just under 0.30) |
| add3w | 0.57 | 0.47 | 0.51 | yes |
| arc_easy | 0.56 | 0.55 | 0.60 | yes |

Selected by the frozen rule: **add3, add4, add3w, arc_easy**. With 96 problems an accuracy has a standard error of
about 0.05, so sub3's exclusion is a coin flip; the rule is applied as written.

**T1 amended, using screen problems only** (they are disjoint from the main problems). The registered rule
extrapolates with the largest of the last eight step-size ratios. On the screen the ratio d_t/d_{t−1} *rises* during
the run (L32 medians: 0.43 at loop 4, 0.58 at 8, 0.80 at 16, 0.83 at 32; consecutive ratios increase in 86% of steps
between loops 3 and 16). So the past ratio underestimates the future one: the realized path length
Σ_{t>r} d_t exceeds the registered bound for every screen problem at r = 4 (both models) and at r = 8 (L32), and for
39–48% at r = 16. This is what a linear map with several decay rates does (the slow directions take over; for a
normal linear map log d_t is convex in t, proved in the theory note), so the registered rule is not merely loose in
its justification, it is biased in a known direction. Violations were invisible in the only displacement the screen
stored (the last position: 0–1.4%). Amendments:
- The collection now stores the exact whole-sequence displacement ‖X_32 − X_r‖ for r = 2, 4, 8, 16 (and ‖X_64 − X_r‖
  on the deep subset) and the per-position step sizes at every loop, so the forecast is tested in the norm the bound
  is stated in.
- The registered rule is still reported as registered ("rule A"), with its violation rate.
- Added: a **calibrated** forecast. Inside each held-out fold, half the problems calibrate one multiplier κ so that
  κ × (rule A bound) covers 95% of them; the other half test it. Coverage near 95% on the test half then follows from
  exchangeability of the problems (split conformal prediction), not from any property of the model. The scientific
  question becomes how *tight* the calibrated bound is: the fraction of problems forecast "decision will not change",
  against two baselines calibrated the same way: a constant bound that ignores the dynamics, and the bare step size.
- Threshold for M3-a and T1: per outer fold, the 80th percentile of held-out scores at the fitting budget (no labels).

**Other screen observations that shape the analysis** (descriptive, 96 problems per family):
- L4's late steps are aligned (cosine of successive steps ≈ 1.0 at loops 10–16, ratio ≈ 0.65) and reach the float32
  floor (relative step 3 × 10⁻⁷) by loop ~24. L32's late steps point against each other (cosine −0.1 to −0.6 from
  loop 10 on, ratio ≈ 0.83): a slowly decaying oscillation; 23% of problems have a two-step change under half the
  one-step change at loop 33. The two looped models approach their limits in qualitatively different ways.
- Answers still change at small budgets: for L32, 17% of top tokens differ between 4 and 32 loops (8% between 8 and
  32, 2% between 16 and 32) and 7% of correctness labels differ between 4 and 32. For L4: 4% and 1.6%. Budget
  transfer has something to measure at r ≤ 8 and little above.
- Accuracy is nearly the same for D, L4 and L32 on every family: the pair is matched in accuracy as well as in
  architecture budget, which is what the 2×2 table needs.

**Measured cost.** Looped models process about 17,000 token-loops per second on arithmetic and 13,000 on multiple
choice (small batches); D's whole screen took 35 seconds including loading. Projection for 1,500 problems × 4
families: about 19 GPU-minutes per looped model and 3 for D: 4 × 19 + 3 = 79, about 80 GPU-minutes,
assuming screen throughput scales to the full set. This is elapsed GPU time, not timeout allocation. Three shards
per model (`--shards 3`, enforced by the entry point) are fixed from first launch: five models require 15 shard-runs in eight launches, at most
two containers and 28 allocated GPU-minutes per launch (224 GPU-minutes maximum across eight launches). Screen
projections of 534–665 seconds per shard with two shards do not survive 25% uncertainty against the 780-second
stop; three shards project about 363–445 seconds. The manifest makes shard count immutable; it will not be
re-planned after L32 (amended in Draft 1.3).

### Changes in Draft 1.3 (2026-11-09, after external review of the frozen package, before any main data)

- Status/disclosure: screen complete, external verdict "go with changes", no main data. Screen probes were used
  for software tests, solver benchmarks and forecast diagnostics, not for family/site/hyperparameter selection
  beyond solver settings and the T1 amendment already in Draft 1.2.
- Solver: training-only SVD whitening, rank tolerance 10⁻⁸ and ridge 10⁻⁶ on U for all models; gradient criterion
  10⁻⁶ and trust-krylov fallback after L-BFGS. Review found spline condition numbers up to about 3.6 × 10⁷ and
  previous solver failures. Screen benchmark evidence supplied with the review: all nine penalties converged for
  pooled probe, confidence+probe, flexible+probe and baselines; 1,440 per-family small-sample fits all converged,
  Newton needed twice, worst gradient 5 × 10⁻⁷. These are diagnostics, not main results. All fits remain in
  selection/estimation; report inner and outer failure counts with every fit and flag contrasts with outer
  failures.
- Inference: shared joint folds for M1-a, own-label folds elsewhere, bootstrap within fold intersections. Any
  undefined draw invalidates the interval, with count/reason. Per-fit AUROC and transfer intervals added; family
  means require all analysed families and identify missing numeric leaves under `incomplete`.
- T1: primary calibrated rule-versus-constant comparison and secondary one/two-step and last-position candidates;
  calibrate only forecastable items, others not stable. Disclose transductive τ and marginal rather than
  stable-subset coverage. Add classification, decision and fixed-label ranking checks, quantization allowances
  and actual frozen-probe rounding comparisons; remove the uninformative old rounding field.
- Position: distinguish prefix theorem from coupling/excitation expectation and finite-time slopes from limits;
  save `n_pos` for position-specific floors. Collection: document `d_pos`, `disp_full`, `n_pos`, G4 and metadata
  constants; unchecked rounding returns None; manifest freezes collection fields and records analysis hashes.
  Float32 `post`, `pre`, `x_prev` copies were recommended but declined to keep the loader simple; looped rounding
  checks cover `x_last` only. Document enforced launch checks, warning-only analysis changes and partial-data CLI.
- Schedule: freeze three shards/model for timing margin, 15 shard-runs in eight launches; remove adaptive shard
  re-planning. Distinguish about 80 elapsed GPU-minutes from maximum timeout allocation.
- Other corrections: confidence uses log-probability/log valid mass; theory note is written; extra saved-budget
  transfer rows are descriptive; cross-family transfer, registered by name only, is now specified and
  implemented as `m1x` (section 6).
  Clarify D dynamics features and top-log-probability change. Update stale safeguards, interface, test tolerance
  and precision runtime-policy descriptions in the implementation contract.
- Second external review (same day, of the amended package, still before any main data): verdict "go with
  changes", all in the analysis, none in the saved arrays. (i) Output-identity features were decoded through a
  table inferred from gold answers; now from `valid_ids` saved in the collection metadata, required for main data.
  (ii) Budget-transfer contrasts now count solver failures of the frozen source probes
  (`source_nonconverged`, and in `nonconverged_outer_fits`). (iii) A family with no items for a model, or two
  models with no problems in common, is skipped with a reason. (iv) `m1x`: mean over targets
  requires every target; like-for-like single-probe drop. (v) Positional floor chosen per item, also for
  prefixes. (vi) The entry point refuses a main launch with other than three shards.
- Precision: shared joint folds, final solver and `--jobs`; the 40-replication table in section 7 replaces the
  earlier provisional figures.
