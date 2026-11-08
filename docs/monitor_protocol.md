# Pilot protocol (Draft 1): are correctness monitors robust to a change of loop budget, and can local dynamics predict their drift?

Draft 0 was reviewed by GPT-6 Astra (xhigh): **GO WITH CHANGES** for a capped smoke test. This is Draft 1, with those
changes made.
- Context: docs/monitoring_gap.md.
- **No GPU work before the user approves.** Every stage below has a hard cap.

## Question

How robust are hidden-state correctness monitors in Huginn-0125 to changes of the loop budget? Can a local linear
model of the loop predict how their scores drift?

Draft 0's slogan, "trading text transparency for dynamical transparency", asserts a trade-off that was never
measured, so it is dropped. What remains is narrower: weight sharing lets one local linear operator be reused near a
stable fixed point. Ordinary transformers also allow local sensitivity analysis; no claim of general analytical
superiority is made.

## Hypotheses and readings

**H0 (gate, development data only).** At the source budget r = 32, format A:
- A linear monitor on the pre-answer state lowers the error among accepted answers at exactly 80% coverage.
- It is compared with the best confidence baseline, chosen on development data.
- The advantage is the difference of the two errors at 80% coverage; the paired bootstrap CI must exclude 0.
- If H0 fails, stop. Report "insufficient precision" and "no useful effect" as different outcomes.
- No task shopping.

**H2 (primary, sealed test, run once).**
- The monitor is frozen at r = 32; the **primary target is r = 16**. That budget is inside Huginn's training support,
  and accuracy there is near its plateau, but the state is still moving (about 10% per step).
- r = 8 and r = 64 are exploratory.
- Readings:
  - TRANSFERS: the bootstrap lower bound of (target advantage − 0.5 × source advantage) is ≥ 0.
  - BREAKS: the target advantage has an upper bound below 0, **or** the coverage failure is supported by its CI.
  - UNCLEAR: otherwise.
- Threshold transfer is reported separately from ranking: realized coverage and error under the source-frozen
  threshold.

**H3 (exploratory).** A local linear forecast of score drift, for q_r = m·x_r:

  q_{r+k} − q_r ≈ Σ_{j<k} ⟨(J_rᵀ)ʲ M, Δx_r⟩

- Δx_r is the displacement of the **full prompt state** over one loop.
- M is the monitor direction placed at its extraction position, and zero elsewhere.
- J_r is the Jacobian of the full-sequence step. `H.step` runs the whole sequence without caches; differentiate it
  through `core_block_forward`, because `iterate_one_step` turns off gradients.
- Checks before use: forward equivalence, and agreement with finite differences.
- Both baselines, no change and linear extrapolation, get the same extra loop that Δ uses.
- Starts r and horizons k are fixed in advance.
- The "settled" regime is defined by the measured residual, not by r.
- SUPPORTED only if the upper confidence bound of (forecast error / better baseline error) is < 0.8.
- This predicts the score only; it does not explain discrimination.

**H1 (appendix remark only; no GPU).**
- Statement: steering that is a function of everything the later computation can access cannot beat the best answer
  predictor given that same information. Everything here means the input/prelude embeddings, all relevant states and
  caches, the policy's information, and any independent sampling randomness.
- It holds for autoregressive answers and for any steering size.
- Steering can still beat a *particular* (e.g. linear) monitor through nonlinear computation.
- Predicting correctness is not predicting the answer, so Draft 0's "headroom" (correctness-monitor accuracy minus
  answer accuracy) was ill-defined and is removed, together with the MLP monitor.

## Tasks, labels, extraction

**T1, addition.**
- Equal-digit addition, 2–5 digits (`data.py`), in fp32.
- Label: exact match of the greedy answer, with fixed rules for parsing and stopping.
- The 63–65% from earlier was teacher-forced, mostly bf16, with 2–4 digits, so accuracy must be re-measured.

**T2, multiple choice.**
- ARC-Easy and/or PIQA, scored by option log-likelihood.
- Huginn's paper reports **69.91% (ARC-Easy) and 76.22% (PIQA)** at r = 32, using normalized accuracy.
- The harness template and normalization are pinned, and these numbers must be reproduced in the smoke run.
- Label: whether the highest-scoring option is correct.
- Confidence baselines come from the option scores: maximum, margin and entropy.

**Extraction.** The core state at the final prompt token, before any answer or option continuation, at every loop
k ≤ r.

## Leakage controls

- Monitors are trained and calibrated on format A only. Format B (another demo set or template) is reserved for format
  transfer, which is exploratory.
- Splits are by problem. Every budget, seed, format and commuted operand pair (a + b and b + a) of a problem stays in
  one split.
- Preprocessing is fitted on training data only.
- Bootstrap resamples problems, not their renderings.

**Baselines:**
- confidence (as above; for T1, max softmax of the first answer token and the log-probability of the greedy
  sequence);
- surface features: length, digits, carries;
- confidence plus state;
- trajectory features computed **only up to the available budget**, with no peeking at later settling;
- a probe trained on mixed budgets;
- a probe refitted at the target budget (oracle).

## Power

- About 300 test problems per task is too few. Astra's rough figure: 680–1,360 independent test problems for 80% power
  to detect 3 pp at the two-sided 5% level, assuming 5–10% discordance.
- The test size is fixed by **simulating the actual clustered, calibrated procedure** with the development-stage
  estimates.
- Repeated formats do not count as independent samples.
- T1 problems can be generated freely. T2 is limited by the dataset sizes (ARC-Easy test about 2.4k; PIQA validation
  about 1.8k).

## Stages and caps (A100-40GB)

| Stage | What | Cap |
|---|---|---|
| 0 | Smoke: time generation, option scoring, readouts and one full-state JVP; T1 fp32 accuracy at r = 32 (100 problems); reproduce T2 accuracy (200 items); validate against `model.forward` | ≤ 10 min |
| 1 | Development only: format A, r = 32, linear and mass-mean monitors; H0 gate; power simulation fixes test size | ≤ 30 min |
| 2 | Sealed test: r ∈ {8, 16, 32, 64}, formats A and B; H2 | set from stage 0 timing; needs approval |
| 3 | H3 forecast on 100 test items (full-sequence JVPs) | ≤ 20 min, only if H0 passes |

Each stage is reviewed by Astra (code before running, results after).

## Deferred

- The truth "knows more than it says" check, with its own controls.
- Steering that changes later reasoning.
- Ouro, at budgets 2/4/6.
