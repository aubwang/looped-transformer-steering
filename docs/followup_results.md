# Two follow-up runs on the Recurrent-Llama pair (2026-11-10)

Designs: `docs/lateshift_screen.md` (Draft 1.1) and `docs/llever_design.md` (Draft 1.3). The scientific
decision rules and their amendments were written down before the production runs. Lever Draft 1.3 followed
that experiment's smoke run and changed only how the evaluation is split into containers, in response to the
measured runtime; it kept the full dose design and T = 48 under the rule set before the smoke run.

GPT-6 Astra reviewed design and code of each experiment before launch
(`results/lateshift_items-20261110-170859/astra_prerun_review.md`, `results/llever_prerun_review.md`) and the
results and this note afterwards (`results/followup_results_review.md`: "accept with changes"; the changes are
applied here, and every table number below was reproduced independently from the saved arrays).

GPU time, recorded function seconds on A100-40GB: screen 9.2 minutes including its smoke run and the
confirmation (the screen itself 5.4), lever replication 12.0 minutes including its smoke run, and 2.8 minutes for
its exploratory follow-up (section 2, last part).

Models: L32 = Recurrent-Llama-3.2 trained around 32 loops; D = its ordinary partner.

## 1. Late-shift screen: no task family passed

**Question.** Is there a task family on which L32's answer still changes between loops 8 and 32 on at least
10–15% of problems, with both models in a usable accuracy range? The screen used these thresholds, fixed in
advance, to decide whether to pursue the monitoring comparison on new tasks.

**Result.** Two of eighteen families passed the first screen (64 problems each) and were re-collected on 300
fresh problems. Neither was accepted, so the monitoring comparison stops here under the fixed rule and budget.

`mul2add` passed every confirmation condition except D's accuracy: 22 of 300 correct (7.3%) against the
required 30 of 300 (10%); L32 had 33 of 300 (11.0%). This is a narrow failure of an eligibility threshold (the
95% interval for D's accuracy, 4.9–10.9%, includes 10%), and it does not establish that no suitable task exists.
A monitoring experiment at about 11% accuracy could study how well rare correct answers are told apart, but it
would have few correct cases and would not by itself show useful late computation: 47 of the 52 changes in
`mul2add` go from one wrong answer to another.

Baseline from the earlier monitor data (6,000 problems, `results/lateshift_existing-20261110`): L32's answer
changes between loops 8 and 32 on 3.5% of problems (1.8–5.2% by family); pooled accuracy is 59.88% at 8 loops
and 59.85% at 32.

Screen, run `results/lateshift-20261110-171533` (64 problems per family, all complete, all gates passed;
brackets are 95% Wilson intervals):

| Family | Verdict | Changed 8→32 | Changed 16→32 | Changed 31→32 | L32 correct at 8 / 32 loops | D correct | wrong→right / right→wrong |
|---|---|---:|---:|---:|---:|---:|---:|
| `add3` | Settled | 0% [0, 6] | 2% | 0% | 80% / 80% | 75% | 0 / 0 |
| `add6` | Settled | 6% [2, 15] | 2% | 2% | 12% / 12% | 9% | 0 / 0 |
| `chain6` | Settled | 8% [3, 17] | 0% | 0% | 5% / 5% | 2% | 0 / 0 |
| `mul2add` | Qualifies | 20% [12, 32] | 3% | 0% | 11% / 12% | 11% | 1 / 0 |
| `paren` | Changes but unusable | 17% [10, 28] | 5% | 2% | 3% / 2% | 0% | 0 / 1 |
| `var4` | Changes but unusable | 44% [32, 56] | 2% | 0% | 0% / 0% | 2% | 0 / 0 |
| `var4s` | Changes but unusable | 41% [29, 53] | 6% | 0% | 2% / 0% | 0% | 0 / 1 |
| `ptr3` | Changes but unusable | 14% [8, 25] | 0% | 2% | 14% / 12% | 16% | 0 / 1 |
| `swap` | Settled | 8% [3, 17] | 0% | 0% | 22% / 25% | 28% | 2 / 0 |
| `acc5` | Settled | 8% [3, 17] | 2% | 0% | 0% / 0% | 2% | 0 / 0 |
| `count` | Settled | 9% [4, 19] | 0% | 0% | 47% / 52% | 38% | 4 / 1 |
| `sort2` | Settled | 0% [0, 6] | 0% | 0% | 12% / 12% | 19% | 0 / 0 |
| `bool` | Settled | 2% [0, 8] | 0% | 0% | 66% / 67% | 62% | 1 / 0 |
| `story2` | Borderline | 11% [5, 21] | 2% | 0% | 53% / 52% | 38% | 1 / 2 |
| `story4` | Settled | 9% [4, 19] | 3% | 0% | 0% / 0% | 0% | 0 / 0 |
| `gsm` | Settled | 3% [1, 11] | 0% | 0% | 17% / 17% | 5% | 0 / 0 |
| `svamp` | Settled | 3% [1, 11] | 0% | 0% | 66% / 67% | 47% | 1 / 0 |
| `arc_chal` | Settled | 5% [2, 13] | 0% | 0% | 55% / 53% | 42% | 0 / 1 |

Confirmation on 300 fresh problems each (indices 64–363), run `results/lateshift-20261110-172550`:

| Family | Changed 8→32 | L32 correct at 8 / 32 | D correct | Outcome |
|---|---:|---:|---:|---|
| `mul2add` | 17.3% [13.5, 22.0] | 10.0% / 11.0% | 7.3% | Not accepted: D below the 10% floor (22 of 300 against 30) |
| `story2` | 7.3% [4.9, 10.9] | 45.0% / 46.3% | 36.7% | Not accepted: change rate below 10% |

**What the numbers say beyond the verdict** (descriptive, no rule attached):

- Late changes occur in several families, and both candidates pass the fixed not-flicker condition. In the
  screen, changes between loops 16 and 32 range up to 6% (4 of 64) and between 31 and 32 up to 2% (1 of 64).
  These are comparisons of two loop counts; they do not show that nothing changes in between.
- Most changed answers stay wrong: of the 133 answer changes between loops 8 and 32 in the screen, 116 go from
  one wrong answer to another, 10 from wrong to right and 7 from right to wrong. This is a predominance, not a
  statement that every change happens on an unsolved problem.
- No family shows a resolved gain in accuracy from loop 8 to loop 32 in these samples. The largest observed
  gain in the screen is `count`, 30 of 64 to 33 of 64 (four wrong-to-right against one right-to-wrong, exact
  McNemar p = 0.375); on the fresh problems `mul2add` gains 1 point (p = 0.375) and `story2` 1.3 points
  (p = 0.42). This does not establish that the accuracies are equal or exclude small benefits.
- Changed answers tend to have small margins. In the confirmation, the median gap between the two most likely
  tokens at loop 32 is 0.06 against 0.30 nats for changed against unchanged `mul2add` problems, and 0.04
  against 0.84 for `story2`.

**Limits.** One-token answers without written working; small synthetic or short-prompt families; 64 problems
per family, so a family with a true rate of 10% is missed more than half the time. "Settled" is a stopping
decision under this budget for this pair, not a statement about looped models in general, and not a statement
that the hidden state is idle: family-median relative steps of the last-position state range from 0.013 to
0.096 at loop 8 and from 1.2e-5 to 2.4e-3 at loop 32.

## 2. Steering levers: the Huginn pattern replicates on L32

**Question.** In Huginn, a small push `P` added at the answer position every loop changed the output as the
loop's linear response predicts, `Δz ≈ G·A·P` with `A = J + J² + … + J^T`; directions built from that operator
("levers", `L_k ∝ Aᵀ G_k`) steered, while the readout gradient `G_k` itself did nothing. Does this pattern hold
on a second looped model?

**Result.** Verdict by the pre-set rule: HUGINN PATTERN REPLICATES, at all three doses, with all gates passed.
The label certifies what the rule tests: the lever raises its target and beats the readout gradient (R1), and
the loop-aware prediction matches the shape of the response and beats the loop-free one (R2); the secondary
criterion R3 (the problem's own operator beats an averaged and a swapped one) also holds.

Gates: harness against the patched model's own forward, centred-logit maximum errors 0 to 9.6e-14 (both sides
share the float64 patches, so this checks the harness, not the patches); the two derivative routes agree to
7e-10; all 40 held-out problems answered correctly; sham drift 4e-9. Task: two-digit sums 70..79 as ten answer
classes; 40 held-out problems; primary push norm 0.03 × state norm = 0.75; 48 forced loops after 64 base loops;
float64.

The table puts two experiments side by side. They differ in task (digit of a sum against sums 70–79), sample
(20 against 40 held-out problems), precision (float32 against float64) and dose (absolute norm 3 against 0.75),
so it is not a controlled comparison of the models. Correlations are medians over problems; intervals are
bootstrap intervals of the mean paired per-problem difference.

| Quantity (primary dose) | Huginn (`dglever`) | Recurrent-Llama L32 |
|---|---:|---:|
| Target effect of lever L (centred logits) | +1.78 | +0.165 [0.154, 0.177] |
| Target effect of readout gradient G | −0.05 | −0.001 [−0.002, 0.000] |
| Target effect of centroid direction C | −0.05 | 0.000 [−0.002, 0.001] |
| Prediction, own operator (FULL): correlation / slope / error | 0.978 / 0.95 / 0.22 | 0.997 / 1.01 / 0.08 |
| Prediction, operator averaged over training problems | 0.947 | 0.937 |
| Prediction, operator of another problem (other class) | 0.938 | 0.920 |
| Prediction, operator of another problem (same class) | 0.931 | 0.818 |
| Prediction without the loop (IDENTITY) | −0.59 | 0.11 |
| FULL − IDENTITY | [1.48, 1.59] | [0.79, 0.91] |
| FULL − averaged; FULL − other problem | [0.02, 0.04]; [0.03, 0.06] | [0.06, 0.07]; [0.07, 0.10] |

Runs: `results/llever_train-20261110-171953`, `llever_pred-20261110-172208`, `llever_eval-20261110-172520`
(summary in `analysis_summary.json` there), push file `results/llever_pushes.npz`.

Dose: effects are close to proportional over a ninefold range (0.055, 0.165, 0.480 at 1%, 3%, 9% of the state
norm; ratios 0.334 and 2.90 against 1/3 and 3). The own-operator prediction has correlation 0.9997, 0.997,
0.976 with slope 1.00–1.01 and median error 2.6%, 7.7%, 21.7%: at the small and primary doses the agreement is
quantitative, not only in shape.

**Further observations.** Time courses and predictor comparisons were specified in advance; the immediate
derivative summaries and the comparison of whole response matrices over time were computed after seeing the
results.

- *The mean target effect appears at once, but later loops still change the response.* The mean lever effect
  is 0.171 after one forced loop and 0.165 after 48. The whole 10 × 10 response matrix after one loop differs
  from its final value by a median 25% in relative norm (correlation with the final matrix 0.97); the difference
  falls to 9% after two loops, 1.6% after four and 0.05% after eight, with the same figures at the smallest
  dose. So about a quarter of the final response, by this measure, comes from applications of the block after
  the first, and the forced response has settled by about eight loops. The last terms of the operator sums are
  below 1e-12 of the sum for the sampled directions. This shows fast settling of this local forced response; it
  is not a measurement of the loop's spectrum, and no matched spectral comparison with Huginn was made.
- *A push along the readout gradient has almost no measured target effect after a loop.* The linearised
  immediate effect `ρ·m·G_k·P_k` (a derivative prediction, not a measured readout) averages +0.96 for gradient
  pushes and −0.007 for lever pushes (negative for 55% of problem–class pairs). After one forced loop the
  measured mean target effects are −0.001 and +0.171. The loop-free prediction is weakly positively correlated
  with the final response here (0.11), where it was anti-correlated in Huginn (−0.59). This supports a related
  picture in the two models (what the readout is most sensitive to is not what the loop passes on), without
  establishing the same mechanism.
- *Own-problem operators predict best.* They beat the averaged and the swapped operators; the penalties are
  larger than in the Huginn run, most of all for a problem of the same class in the other context (0.818), but
  the differing protocols prevent attributing that to the model alone.
- *Effect per unit of push.* 0.22 per unit of absolute push norm here against 0.59 in Huginn; state scales,
  tasks and readouts differ, so this is not a comparison of how steerable the models are.

**What repetition adds to the prediction** (exploratory follow-up E1, planned in `docs/llever_design.md`
section 8 after the main results and before its run; run `results/llever_pred-20261110-174655`, 2.8 GPU-minutes,
which reproduces the first prediction run exactly). The prediction `G·(J + … + J^t)·L` with t passes through
the block, against the measured final response (median over the 40 problems):

| Passes t in the prediction | Error, dose 1% | Correlation, dose 1% | Error, dose 3% | Correlation, dose 3% |
|---|---:|---:|---:|---:|
| 1 | 25.3% | 0.972 | 26.7% | 0.968 |
| 2 | 9.8% | 0.996 | 12.9% | 0.993 |
| 4 | 3.3% | 0.9995 | 8.2% | 0.997 |
| 8 | 2.6% | 0.9997 | 7.7% | 0.997 |
| 48 (full sum) | 2.6% | 0.9997 | 7.7% | 0.997 |

One pass through the block already gives the right shape (correlation 0.97) but misses a quarter of the final
response in norm; the excess error over the full sum is 22–27 points at the 1% dose and 18–22 at 3% (95%
bootstrap intervals over problems). Two passes leave 7–9 points, four passes under one point, and from eight
passes on the prediction is that of the full sum. The same prediction with t passes matches the response
*measured after t forced loops* with error 2.4–2.7% at the 1% dose for every t, so the linear response holds
along the way and not only at the end. In short: repetition matters for about the first four loops of a
sustained push, and adds nothing measurable after eight. This concerns these fixed directions and problems.

**Limits.** One task, one position, population directions, a base state after 64 loops (answers at 32 and 64
agree on all problems). This is a local intervention experiment around a settled state; it is not evidence that
recurrence improves reasoning or monitoring. Directions are compared at equal norm, not equal overlap with L.
The time-course statistic written in the design ("negative after one loop, positive at the end", 0.75% of
pairs) does not measure Huginn's reversal, which compares the immediate readout with the response one loop
later.

## 3. Second matched pair: none verified

`docs/pairs_check.md`: the looped OLMo-2 and TinyLlama models are public at recurrences 4/8/16/32, but their
ordinary post-trained partners were not found among the public repositories of the authors (the paper reports
training them). The Llama pair is the only fully verified downloadable matched pair established by this search.
One other released pair was found whose matching in data and training budget is unverified
(`Mihara-bot/LoopLM-Llama3Arch-4x5-Muon` with `…-1B-Nonloop-Muon`).

## 4. Where this leaves the project

- The monitoring comparison for this pair stops under the fixed eligibility rule and budget. On the original
  monitor tasks the answer changes on 3.5% of problems between loops 8 and 32, and no screened family passed
  every confirmation criterion. This does not imply that the hidden state is idle or that no suitable task
  exists.
- The lever-advantage and loop-aware-prediction pattern holds in both looped models tested, each under its own
  protocol. On L32 the prediction at the primary dose has median error 7.7%, correlation 0.997 and slope 1.01.
  This supports local linear response around the settled state; it does not show a benefit of recurrence for
  solving tasks, or an identical mechanism in the two models.
- A second fully verified matched pair has not been established. Release of the ordinary OLMo-2 or TinyLlama
  partner by the authors is one route; verifying the other candidate pair is another.
