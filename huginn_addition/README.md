# Huginn-0125 looped-LM experiments: experiment log

This is the chronological log of the study: one section per experiment, each with its design, prespecified rules,
numerical gates, results, the reviewer-accepted wording (GPT-6 Astra), and caveats. For the synthesis of what was found,
see **[`WRITEUP.md`](WRITEUP.md)**. The repository overview and setup are in the [root README](../README.md).

Conventions:
- Sections appear in the order the experiments were run. Later sections can qualify or supersede earlier ones; where
  that matters, the later section says so.
- "Next (Astra)" notes record the follow-up suggested at the time. Each is marked with its status as of 2026-11-07
  (done, with a link; or open).
- Quoted blocks (`>`) are reviewer-accepted wording; bullet details below them give the numbers and gates.

## Contents

- [Pilot setup](#pilot-setup-experiment-1-addition-dynamics)
- [Pilot results](#pilot-results-126-problems--2-demo-sets--4-seeds--256-iterations-bf16-plus-a-separate-21-problem-bf16fp32-control)
- [Shape of the answer-digit representation](#shape-of-the-answer-digit-representation-offline-shape_analysispy-shape_sensitivitypy)
- [Causal test: pushing the state along the circle](#causal-test-pushing-the-state-along-the-circle-fp32-intervene_apppy-fit_circlepy-analyze_interventionspy)
- [Angular sweep](#angular-sweep-readout-only-intervene_apppy-sweep-analyze_sweeppy-results-reviewed-by-gpt-6-astra)
- [Local readout gradients](#local-readout-gradients-intervene_apppy-gradients-analyze_gradientspy-reviewed-by-gpt-6-astra)
  - [Digit-centered reanalysis](#digit-centered-reanalysis-analyze_centeredpy-no-model-run-suggested-and-reviewed-by-gpt-6-astra)
- [Recurrence dynamics around the late state](#recurrence-dynamics-around-the-late-state-intervene_apppy-jacobian-analyze_jacobianpy-reviewed-by-gpt-6-astra)
- [Long runs: some trajectories end in a period-2 cycle](#long-runs-some-trajectories-end-in-a-period-2-cycle-intervene_apppy-orbit-analyze_orbitpy-reviewed-by-gpt-6-astra)
  - [Where the cycle comes from](#where-the-cycle-comes-from-intervene_apppy-cycsrc-analyze_cycsrcpy-code-and-results-reviewed-by-gpt-6-astra)
  - [A period-doubling bifurcation along a contextual-input family](#a-period-doubling-bifurcation-along-a-contextual-input-family-intervene_apppy-bifur-analyze_bifurpy-design-code-and-results-reviewed-by-gpt-6-astra)
  - [Follow-ups: replication, loop-parity flicker, real-text screen](#follow-ups-replication-loop-parity-flicker-real-text-screen-results-reviewed-by-gpt-6-astra)
  - [Prevalence and convergence-mode census on fresh prompts](#prevalence-and-convergence-mode-census-on-fresh-prompts-make_census_promptspy-intervene_apppy-census--ownblock-analyze_censuspy-analyze_ownblockpy-design-code-and-results-reviewed-by-gpt-6-astra)
- [Is uncertainty a weak pull?](#is-uncertainty-a-weak-pull-select_doubtpy-intervene_apppy-doubt-analyze_doubtpy-design-code-and-results-reviewed-by-gpt-6-astra)
- [Carry: is the carry an extra one-notch rotation?](#carry-is-the-carry-an-extra-one-notch-rotation-intervene_apppy-carry_traj-analyze_carrypy-reviewed-by-gpt-6-astra)
  - [Stricter carry audit](#stricter-carry-audit-analyze_carry_auditpy-no-model-run-code-and-results-reviewed-by-gpt-6-astra)
- [Weekday pilot: a circular concept outside arithmetic](#weekday-pilot-a-circular-concept-outside-arithmetic-make_weekday_promptspy-intervene_apppy-wkcollect-analyze_weekdaypy-analyze_weekday_startpy-design-code-and-results-reviewed-by-gpt-6-astra)
- [Predictive source under-relaxation](#predictive-source-under-relaxation-select_relaxpy-intervene_apppy-relax-analyze_relaxpy-design-code-and-results-reviewed-by-gpt-6-astra)
- [Digit-circle rotation pilot: carry-conditioned units prediction](#digit-circle-rotation-pilot-carry-conditioned-units-prediction-make_digit_rotation_promptspy-intervene_apppy-dgcollect-analyze_digit_rotationpy-design-code-and-results-reviewed-by-gpt-6-astra)
  - [Threshold sweep](#threshold-sweep-select_sweeppy-intervene_apppy-relaxsweep-analyze_sweep_relaxpy-code-and-results-reviewed-by-gpt-6-astra)
- [Does settling speed predict how many loops are needed?](#does-settling-speed-predict-how-many-loops-are-needed-make_settle_promptspy-intervene_apppy-settle-analyze_settlepy-design-code-and-results-reviewed-by-gpt-6-astra)
- [Budget robustness on real multiple-choice prompts, and a global relaxation stabilizer](#budget-robustness-on-real-multiple-choice-prompts-and-a-global-relaxation-stabilizer-fetch_data_apppy-make_mcq_promptspy-intervene_apppy-mcq-analyze_mcqpy-design-code-and-results-reviewed-by-gpt-6-astra)
- [Pointer-chasing behavioral screen: extra loops do not add hops](#pointer-chasing-behavioral-screen-extra-loops-do-not-add-hops-make_pointer_promptspy-intervene_apppy-mcq-mode-ptr_screen-analyze_pointer_screenpy-code-reviewed-by-gpt-6-astra-exploratory)
- [Carry-chain procedure experiment, stage 1](#carry-chain-procedure-experiment-stage-1-make_carry_chain_promptspy-intervene_apppy-mcq-mode-carry_chain-analyze_carry_chainpy-code-and-results-reviewed-by-gpt-6-astra)
- [Month shape and opposite-month context blends](#month-shape-and-opposite-month-context-blends-make_month_promptspy-intervene_apppy-mblend-analyze_month_blendpy-design-code-and-results-reviewed-by-gpt-6-astra)
- [Steering a looped model: one-shot state edit vs persistent context edit](#steering-a-looped-model-one-shot-state-edit-vs-persistent-context-edit-make_steer_dirspy-intervene_apppy-steer-analyze_steerpy-design-code-and-results-reviewed-by-gpt-6-astra)
- [Truth in a looped model, stage 1: reading](#truth-in-a-looped-model-stage-1-reading-make_truth_promptspy-intervene_apppy-truth-analyze_truthpy-design-code-and-results-reviewed-by-gpt-6-astra)
- [Truth stage 2: steering at the statement-final token](#truth-stage-2-steering-at-the-statement-final-token-truthsteer-analyze_truthsteerpy-code-and-results-reviewed-by-gpt-6-astra)
  - [Truth stage 2b: steering at the answer position](#truth-stage-2b-steering-at-the-answer-position-truthsteer-siteans-analyze_truthsteerpy-with-prespecified-p5-reviewed-by-gpt-6-astra)
- [Why truth steers weakly: a linear-response (resolvent) decomposition](#why-truth-steers-weakly-a-linear-response-resolvent-decomposition-intervene_apppy-respond-analyze_respondpy-design-code-and-results-reviewed-by-gpt-6-astra)
- [Loop-aware steering: the resolvent-optimal direction d*](#loop-aware-steering-the-resolvent-optimal-direction-d-dstar_vjp-make_dstar_pushespy-dstar_eval-analyze_dstarpy-design-code-and-results-reviewed-by-gpt-6-astra)
  - [d* controls: answer-token lever vs truth, dose, pulse, removal](#d-controls-answer-token-lever-vs-truth-dose-pulse-removal-make_ctrl_promptspy-ctrl_eval-analyze_ctrlpy-design-code-and-results-reviewed-by-gpt-6-astra)
  - [d* counterbalanced control](#d-counterbalanced-control-make_ctrl_promptspy-formats-fqba-ctrl_eval-mode-ctrl_cb-analyze_ctrl_cbpy-reviewed-by-gpt-6-astra)
- [Is there information in the loop path beyond the endpoint?](#is-there-information-in-the-loop-path-beyond-the-endpoint-analyze_trajectory_infopy-offline-on-existing-runs-rules-reviewed-by-gpt-6-astra-before-the-first-run)
- [Digit levers: loop-aware control and the full 10×10 response matrix](#digit-levers-loop-aware-control-and-the-full-1010-response-matrix-dglever-trainpredeval-make_dglever_pushespy-analyze_dgleverpy-design-code-and-results-reviewed-by-gpt-6-astra)
- [Mechanism of the readout reversal](#mechanism-of-the-readout-reversal-intervene_apppy-revmech-analyze_revmechpy-design-code-and-results-reviewed-by-gpt-6-astra)
- [Local readout transfer operator](#local-readout-transfer-operator-intervene_apppy-rtransfer-analyze_rtransferpy-design-code-and-results-reviewed-by-gpt-6-astra)
- [Which part of the block implements the one-step reorientation](#which-part-of-the-block-implements-the-one-step-reorientation-intervene_apppy-blockdecomp-analyze_blockdecomppy-design-code-and-results-reviewed-by-gpt-6-astra)
- [Cheap levers from the adapter and one-step pullbacks](#cheap-levers-from-the-adapter-and-one-step-pullbacks-intervene_apppy-adaptcov--dglever-mode-eval-adlever_eval-analyze_adaptcovpy-make_adlever_pushespy-analyze_adleverpy-design-code-and-results-reviewed-by-gpt-6-astra)
- [Overlap-matched control for the cheap levers](#overlap-matched-control-for-the-cheap-levers-make_overlap_pushespy-dglever-eval-config-ovl_eval-analyze_overlappy-design-code-and-results-reviewed-by-gpt-6-astra)
- [Cheap levers on the truth task](#cheap-levers-on-the-truth-task-intervene_apppy-truthcov--dstar_eval-config-tlever_eval-make_tlever_pushespy-analyze_tleverpy-design-code-and-results-reviewed-by-gpt-6-astra)
- [Label-free sign selection: does steering add accuracy beyond the monitor?](#label-free-sign-selection-does-steering-add-accuracy-beyond-the-monitor-analyze_selectorpy-offline-exact-design-and-results-reviewed-by-gpt-6-astra)

## Pilot setup (Experiment 1: addition dynamics)

How does a looped / recurrent-depth transformer ([Huginn-0125](https://huggingface.co/tomg-group-umd/huginn-0125),
pinned revision `bb6621b…`) evolve its latent state across recurrent iterations while adding numbers?

- `data.py`: equal-digit addition problems (2–4 digits), stratified by number of carries and by carry
  *dependency length* (carry-lookahead generate / propagate / kill), each rendered with two few-shot demo sets.
- `modal_app.py`: runs the core block manually for K iterations from several seeded random initial states.
  Logs per-step state geometry, logit-lens readout, cross-seed distances, lag profiles, phase alignment, PCA,
  compact random-projection trajectories, and a greedy-generation check. Checks itself against `model.forward`
  and fails on mismatch.
- `analyze.py`: figures and `summary.json`, with bootstrap CIs that resample problems.
- `precision_figure.py`: compares the bf16 and fp32 runs.

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python "modal[api-proxy-support]" pandas numpy matplotlib pyarrow
.venv/bin/modal run huginn_addition/modal_app.py --mode pilot        # also: smoke, prec_bf16, prec_fp32
# bulk `modal volume get` gets HTTP 429 behind the proxy; fetch files one at a time (see shell loop in history)
.venv/bin/python -I huginn_addition/analyze.py results/<run_id>
```

## Pilot results (126 problems × 2 demo sets × 4 seeds × 256 iterations, bf16; plus a separate 21-problem bf16/fp32 control)

Wording revised after an independent results review by GPT-6 Astra. All numbers below were re-verified by that review.

1. **Accuracy plateaus at 16–32 iterations, around 63%.** Teacher-forced all-digits accuracy at 16 / 32 / 64 / 128 / 256 iterations
   is 63.9 / 65.3 / 63.3 / 63.0 / 63.2%. Greedy accuracy is statistically compatible with teacher forcing at every tested depth
   (paired bootstrap CIs on the difference include 0). That shows compatibility, not identical predictions.
2. **Dependency length vs. iterations (conditional on ending correct).** Among digit trajectories that are correct at step 256,
   each extra unit of carry dependency length goes with +0.63 log₂ iterations (CI 0.45–0.93) until the last transition into
   uninterrupted correctness through step 256. Covariates: column, overflow, first token, digit count, carry run, carry-in.
   Adding family / demo / column type / answer digit leaves it at 0.63–0.64. No clear extra digit-count association showed up.
   **Caveats:** the selection is strongly differential. Retention by dep_len 0–4 is 100 / 87 / 63 / 56 / 50%, and dep_len 4 is
   a single problem. "Stably correct" is finite-horizon bookkeeping: 11 trajectories first become correct at step 256, and the
   coefficient moves 0.54 → 0.58 → 0.62 → 0.63 for K = 64 / 128 / 192 / 256. Long-dependency strata still contain errors at
   the tested horizon.
3. **The argmax often settles early.** Final-argmax stabilization has a median of 7 iterations for trajectories that end
   correct and 19 for those that end incorrect. The state is still changing ~65% per step at 4 iterations and ~10% at 16. This
   concerns the argmax token only, not the full distribution or the computation.
4. **The bf16 floor is precision-sensitive.** In bf16 the per-step change never drops below ~1.5%, and a weak period-2 zigzag
   appears in the lag profiles. In the fp32 control, 306/316 answer-position trajectories (96.8%) are below 1e-4 relative change
   at step 128 (median ~4e-6). The two tested initializations end up closely aligned (median distance 4.7e-6, max 0.021);
   39/42 prompts have every trajectory below 1e-4. So most measured answer-position trajectories approach nearly stationary
   states in fp32, and the two tested starts converge together. That is consistent with an input-determined attractor, but
   it does **not** establish uniqueness, convergence from other initializations, or convergence of the whole sequence.
   Changing dtype also changes how parameters are represented, not only rounding.
5. **Exception, uid 26 (1851 + 6224, demo A), fp32.** It predicts the correct digit 8 at steps 32–48 and the no-carry digit 7
   at steps 64–128. Meanwhile a period-2 component grows by ~+3.7–4% per step (log-linear fit over steps 64–128, both seeds).
   Demo B on the same problem approaches stationarity. It is not yet established whether this is an unstable mode of the
   exact-arithmetic model or a numerical effect.

### Next

*Status (2026-11-07):* the period-2 behavior seen in uid 26 was followed up generally in [Long runs](#long-runs-some-trajectories-end-in-a-period-2-cycle-intervene_apppy-orbit-analyze_orbitpy-reviewed-by-gpt-6-astra) and its subsections (cycle source, bifurcation, census). The reanalysis items and matched operand designs below remain **open**.

- Reanalysis: retention by dependency, horizon / minimum-dwell sensitivity, modeling final correctness separately from
  correction timing, leave-one-problem-out.
- uid 26: longer runs, matched controls, forced high-precision kernels, full saved states, controlled perturbations /
  Jacobian-vector growth.
- Generalization: more initializations, full-sequence residuals, matched operand designs that separate dependency structure
  from digit patterns, answer digits, and overflow.

## Shape of the answer-digit representation (offline, `shape_analysis.py`, `shape_sensitivity.py`)

Plan and wording reviewed by GPT-6 Astra. Data: seed-averaged late states (steps 192–256) at answer positions after the
first (678 positions, 126 problems), in the 64-d random-projection sketch of the bf16 pilot. Everything is held-out by
problem and adjusted for nuisance (previous token, position, column, lengths, overflow, demo, family).

- For answer positions after the first, the late-state sketches contain a digit-related component consistent with
  numerical cyclic order. Two sinusoidal digit features (cos/sin 2πd/10) achieve held-out prediction gains comparable to
  a regularized categorical lookup table: 105% of its gain (95% bootstrap 94–118%). A straight-line code captures 32%
  (24–41%). Using the model's own predicted digit gives the same picture (circle 99%).
- Numerical cyclic order beat all 200 sampled scrambled orders and all 200 random 2-D codes (Monte Carlo p = 1/201 ≈
  0.005, the minimum reportable). Line order: p = 0.065.
- In an unconstrained decoder, 70% of confusions are ±1 around the loop (22% expected from error totals). The 9↔0
  wraparound accounts for 34 confusions vs 11.6 expected, so 9 and 0 behave like neighbors (descriptive, not a clustered
  test).
- Adding categorical digit identity improves held-out R² by 5.6 percentage points of sketch variance beyond nuisance.
- Robust to stronger controls (previous token × position/column/family, Frisch–Waugh residualization, dropping rows after
  a 9): circle 106–113% of categorical, line 34–39%, true order beats all 100 scrambled orders in each variant.
- Under the chosen normalization, answer directions show no earlier settling than the leading PCA directions (rank-9
  medians 39 vs 35 steps; random rank-9: 42).

Caveats: teacher forcing; 64-d sketch of bf16 states; pre-coda states, so the circle may reflect alignment with
digit-token output geometry rather than the arithmetic itself; the cos/sin image can be an ellipse; no continuous-helix
or carry-rotation claim.

## Causal test: pushing the state along the circle (fp32, `intervene_app.py`, `fit_circle.py`, `analyze_interventions.py`)

Design and results both reviewed by GPT-6 Astra; wording follows its corrections.

**Fit (126 problems, full 5280-d fp32 states):** the circle replicates in full space (cross-validated predictive gain
108% of the categorical lookup; line 29%; scrambled orders p = 1/51). The plane is stable under strong nuisance controls
(principal cosines 0.994 / 0.977). The basis is near-circular (|u_cos| 17.3, |u_sin| 15.5, cosine between them 0.017).
One notch ≈ 10.1 (13% of state norm).

**Interventions (84 held-out problems / 168 prompts; 147 baseline-correct prompts / 77 problems; one answer position
per prompt; sham replay exact):**
- Intervening along fitted circle chords causally changes the immediate readout in the intended direction. +1 notch:
  +2.32 target-vs-source log-odds [2.14, 2.52]. −1 notch: +2.21 [2.02, 2.41]. The opposite neighbor moves the other way
  (−0.76). The effect is roughly linear in push size (1.07 / 2.32 / 4.71 at 0.5 / 1 / 2×). Equal-size random, top-PC and
  radial pushes have much smaller effects (−0.28 to +0.37). Target-matched scrambled-order chords have intermediate
  effects (+0.66 to +1.52); 60–84% of their squared norm lies in the true plane, and true-plane sensitivity predicts
  them with r = 0.84, though not completely.
- Large baseline margins limit target switches: the median source-over-target margin is about 6.3 nats. Switch rates are
  12–13% at one notch, 22% / 16% at 2×, and 61% for a two-notch chord at 2× (no matched controls for that one).
- Under continued recurrence (push at converged step 96, or at step 24), the component along the original push nearly
  vanishes by ~8 iterations (circle: 0.73 / 0.42 / 0.09 at +1 / +2 / +4; random: 0.25 / 0.05). Residual displacement
  keeps decaying (circle ~13% of push size at +8, ~0 by +32). The final argmax matches the unpushed baseline for all 168
  prompts. Circle chords keep their push-direction component longer than the tested random direction (total-norm
  retention after 1 step: 0.84 vs 0.67, similar to radial and scrambled). That is not an identified eigenmode.
- Interpretation: the coda reads digit identity along this plane (directional readout sensitivity). A single-position
  push does not persistently relocate the answer under continued recurrence: the state recovers toward the unperturbed
  trajectory for these α = 1 pushes. Not shown: a uniquely circular computational mechanism, that arithmetic proceeds by
  rotation, or that the input specifically "pins" the answer (other positions could restore it).
- Next (Astra) — **done**, see [Angular sweep](#angular-sweep-readout-only-intervene_apppy-sweep-analyze_sweeppy-results-reviewed-by-gpt-6-astra): a norm-matched angular sweep within the true plane, plus each scrambled chord split into its true-plane
  projection and orthogonal residual, measuring all-digit readout effects.

## Angular sweep (readout-only, `intervene_app.py sweep`, `analyze_sweep.py`; results reviewed by GPT-6 Astra)

Same 168 held-out prompts (147 eligible / 77 problems). Pushes at 24 angles in the fitted plane at a fixed size (1× and 2×
the one-notch chord), read out immediately.
- Immediate-neighbor phase preferences closely match the fitted circle geometry: the sinusoid peak is 107° vs 107°
  predicted for d+1, and 254° vs 251° for d−1. ±2 agree only approximately (errors ~10–13°, outside pointwise CIs). This
  is non-trivial (a linear readout's gradient need not align with the chords), but it uses the same label-fitted plane,
  so it is supporting evidence, not an independent replication.
- Offsets 3–7 have similar preferred angles near 180° instead of the predicted 18°-per-digit progression, with different
  amplitudes. Why is not established.
- The first angular harmonic explains most of the angle dependence at each radius (97% / 91%). Positive angular means
  (0.44–0.77 at 1×) mean the response is not purely linear.
- Holding the fitted plane fixed, the numerical-order prediction correlates r = 0.75 / 0.77 with the observed map
  (0.75 / 0.79 offset-centered), exceeding about 98% of 1000 scrambled-label predictions (p ≈ 0.02). Some scrambled
  orders do as well. ±1/±2 offsets carry ~40–46% of the covariance.
- Scrambled chords: the in-plane component produces most of the target-vs-source effect (54–82%), with components
  approximately additive. The orthogonal component also shifts the contrast (0.23–0.43 nats, above roughly half-size
  random controls). Target specificity of that part is untested.
- Overall: the fitted circle plane causally influences the immediate digit readout, with especially close geometric
  alignment for adjacent digits. Earlier recurrence experiments show recovery toward baseline. Not shown: a complete
  circular decoder.
- Next (Astra) — **done**, see [Local readout gradients](#local-readout-gradients-intervene_apppy-gradients-analyze_gradientspy-reviewed-by-gpt-6-astra): small symmetric ±ε perturbations along two orthonormal plane directions, per source digit/problem, to
  estimate local readout gradients and compare them with the chords (separates local alignment from finite-radius effects).

## Local readout gradients (`intervene_app.py gradients`, `analyze_gradients.py`; reviewed by GPT-6 Astra)

Fresh held-out set (per_cell=4, seed=1): 168 rows, 147 eligible (baseline full-vocab argmax = correct digit) / 77 problems.
At one random answer position (pos≥1) per problem, the step-96 state is differentiated through the coda (autograd,
fp32, teacher-forced). For each offset j, the gradient of the log odds L_j = lp(d+j) − lp(d) is projected into circle
coordinates. The preferred push angle is the unit push in the plane that increases L_j fastest. The prediction applies
the same rule to the fitted chord f(d+j) − f(d), which reduces to 90° + 18°·j. Results: `results/grad-20261104-082200/analysis`.
- **Gradient check.** Central differences along two orthonormal plane axes match autograd: correlation 0.9999999,
  median relative error 5e-4 at ε=0.25. Absolute first-order Taylor residuals are 1.7% of the linear term at ε=0.25 and
  13% at ε=2. Here ε=2 is about 0.2 of one notch, so this only shows near-linearity for small pushes.
- **Far offsets.** Clustering near 180° is already present in the infinitesimal directional sensitivity, so finite push
  size is not needed to produce it. Problem-averaged preferred angles for j=3..7 are 172–183° (predicted 144–216°), with
  low resultant lengths (0.35–0.57): individual rows vary a lot. Weighting each source digit equally gives the same
  picture (182–193°). The problem-averaged local angles are qualitatively close to the 1×/2× sweep phases (within about
  12°). They are not a direct prediction of the sweep, because the two analyses aggregate differently.
- **Near offsets.** Problem-averaged ±1 preferences closely match the circle prediction (108.9° vs 108°; 252.5° vs 252°).
  ±2 is shifted about 18° toward 180° (143.9° vs 126°; 215.3° vs 234°). Individual rows are less tight: median |error|
  is 17–25°.
- **Error by source digit.** Problem-weighted errors are systematic: mostly positive for source digits 0–5 and negative
  for 6–9. This shows departures from the *uniform fitted-chord* model, which could come from non-uniform digit centroids
  or a context-dependent coda Jacobian. It is not direct evidence of non-uniform state geometry. Per-digit samples are
  small (4–17 problems), and opposite-sign errors cancel in the pooled means.
- **Most readout sensitivity is outside the plane.** For individual digit log-prob gradients, the median in-plane
  squared-norm fraction is 0.5–0.7%. That is about 13–19× the isotropic random-2-D-plane expectation (a reference value,
  not a significance test).
- **Target vs source split.** At the source digit, p(d) ≈ 1, so ∇lp(d) is small: the in-plane target term is 30–70×
  stronger. But at saturation ∇lp(t) ≈ ∇z(t) − ∇z(d), so splitting at the log-prob level cannot separate a "toward t"
  signal from an "away from d" signal. (Correction, Astra: digit-centered logit gradients ∇z(q) − mean_k ∇z(k) are
  recoverable from the saved data by subtracting the across-digit mean of the 10 log-prob gradients, because the
  normalizer term is shared. That analysis has not been run yet.)
- **Overall.** Local contrast preferences align with adjacent digits and cluster for far targets. This is inconsistent
  with a simple decoder whose gradients follow all fitted circle chords. It does not rule out a richer circular decoder.
  The results use the same label-fitted plane and teacher forcing.

### Digit-centered reanalysis (`analyze_centered.py`, no model run; suggested and reviewed by GPT-6 Astra)

Subtracting the across-digit mean of the 10 saved log-prob gradients gives exact gradients of digit-centered logits,
z~_q = z_q − mean_k z_k. The contrast gradient then splits exactly into a target part T = ∇z~(d+j) and a shared source
part S = −∇z~(d). Output: `results/grad-20261104-082200/analysis_centered/`.
- **Far offsets.** For j=3..7, the shared source term supplies a median 73–89% of the *contrast-aligned projection*
  (a signed projection share, not a variance fraction or a mechanistic attribution). The source term points near "away
  from the current digit" (178°, R 0.61, row-weighted). Target terms are smaller (41–56% of the source norm) and poorly
  aligned with the target's radius. So the far-target clustering is mostly the shared lowering of the current digit's
  *centered* logit, not similar representations of the far targets.
- **Near offsets.** ±1/±2 have larger target contributions (58–82%).
- **Centroids.** Empirical residual-state centroids, which are not jointly adjusted digit effects, are visibly uneven:
  digit 9 is pulled inward (radius 0.59) and consecutive steps range 17–70°. Their chords show no overall improvement over
  uniform chords (all offsets: +0.5° [−0.6, 1.7] in mean |error|). Descriptively they help +1/+2 and hurt −1/−2; only +2
  and −1 have intervals excluding zero.

## Recurrence dynamics around the late state (`intervene_app.py jacobian`, `analyze_jacobian.py`; reviewed by GPT-6 Astra)

Same 168 held-out prompts (147 baseline-correct rows), one answer position each, fp32, K=96.
- **Why the position can be studied alone.** Attention is causal, so perturbing only position p leaves earlier positions
  unchanged. To first order, δ_p(k+1) = J_pp(k) δ_p(k), where J_pp is position p's own one-step Jacobian, computed by
  central finite differences.
- **What was measured.**
  - The 2×2 circle-plane block of J_pp.
  - An Arnoldi spectrum (m=60). The top Ritz pairs have residual ≈0.02, confirmed by fresh finite differences.
  - Nonlinear propagation of circle ±1 pushes and a norm-matched random push, followed for 32 loops.
- Results: `results/jac-20261104-164552/analysis`. Runtime 12.5 GPU-min.
- **Near-stationary, mostly.** Median per-step relative drift at K is 6.8e-6, near the fp32 floor, and it stays ~4e-6
  over the next 32 steps (not exactly zero). Exception: 8 rows from 6 problems still drift at 1e-4 to 1.8% per step
  through step 128, and their earlier positions are also still moving. Their own J_pp is locally stable (largest Ritz
  modulus 0.76–0.81), so the motion is driven by the moving prefix. Whether those trajectories converge slowly, drift, or
  orbit is untested.
- **Local attraction.** Tested perturbations return toward the reference trajectory, consistent with local attraction.
  The largest observed Ritz modulus has median 0.78 (p10–p90 0.75–0.82). This is asymptotic linear stability, not
  shrinkage in every direction on every loop. A single 60-step Arnoldi run can also miss modes.
- **Not a neutral ring in the fitted plane.** The circle-plane block is ≈0.71·I (moduli 0.72/0.69). Systematic rotation
  is negligible: polar rotation median +0.6°/step in circle coordinates, p10–p90 −4.5…+5.7. About 53% of each basis
  direction's image leaks out of the plane each step. Circle ±1 pushes keep 73% in-plane after 1 loop, 11% after 4 and
  1.8% after 8. That is faster than the plane block alone predicts (28% / 8% at t4 / t8), so recovery is not described by
  the plane block. Pushes of 0.1 and 1 notch give similar normalized responses.
  - Wording (Astra): these measurements argue against a *locally neutral ring lying in the fitted circle plane* under a
    fixed context. Different teacher-forced contexts have different conditional dynamics, so this is not evidence for ten
    attractors within one dynamical system. The pushes are chords, and the plane is not invariant.
- **Leading modes.** The leading observed Ritz modes are often complex (71% of rows; modal phase median 13° over all
  rows). They lie predominantly outside the fitted plane (overlap 0.18 vs RMS random-plane overlap 0.019, about 9×).
- **Circle vs random pushes.** Circle pushes keep somewhat more total displacement than the single tested random
  direction (0.13 vs 0.08 at t8), but still decay strongly: they are not neutral.
- **Carry rotation not tested.** This measures late-state recovery. It does not test whether carry processing produces
  an extra one-notch (+36°) phase transition earlier in the recurrence. Exploratory: carry-in vs no-carry rows differ
  little (plane modulus 0.714 vs 0.727; radius 0.795 vs 0.778; n = 23 vs 124).

## Long runs: some trajectories end in a period-2 cycle (`intervene_app.py orbit`, `analyze_orbit.py`; reviewed by GPT-6 Astra)

20 prompts, fp32, K=1024 loops. These are the 6 problems flagged as still moving in the Jacobian run plus 4 stationary
controls, run from the same initial state. Their early trajectories reproduce the Jacobian run exactly. Results:
`results/orbit-20261104-193804/analysis`.
- **Six cycling prompts.** 6 prompts (4 problems: 23A/B, 24B, 49A, 50A/B) show persistent, approximately period-2 motion
  through step 1024:
  - Over the last 256 steps, one direction carries ≥99.9999% of the variance (random 64-d projection) at the answer
    position.
  - The FFT puts essentially all power at period 2.
  - ||x_k − x_{k−2}|| / ||x_k − x_{k−1}|| is 3–6e-4.
  - The late per-step change is 0.7–1.85% of the state norm and constant from step 512 to 1024.
- **Time course.** The step size first declines to a minimum around steps 50–70, then grows and saturates. Astra's
  wording: "a period-2 signature following an initial decline in step size". The data do not show that the state
  approached a fixed point that then lost stability by flipping.
- **Readout unaffected.** The readout argmax never changes at any answer position from step 96 to 1024 (checked every
  step).
- **Candidate source.** The earliest moving position is the first digit of the second operand for problems 23 and 50
  (both demos), and the "=" token for 24B and 49A. This is a candidate source, not causal proof.
- **Marginal cases and controls.** The two marginal rows from the Jacobian run (26A, 55B) decay to the ~4e-6 floor, and
  all controls stay at that floor.
- **Prevalence.** In the Jacobian sample this is 4/84 problems (4.8%) or 6/168 prompts (3.6%). That comes from a targeted
  follow-up, not an unbiased prevalence estimate.
- **Interpretation.** For a small minority of prompts, the recurrence settles into a sustained alternation rather than a
  fixed point, without affecting the answer. This is concrete evidence that "converges to a fixed point" is not universal
  for this model.
- **fp64 replication** (`orbit64`, pids 23/50 plus control 44, `results/orbit64-20261104-195601`; wording by Astra):
  - With fp32-generated initialization, fp64 closely reproduces the cycling amplitudes. Controls reach zero measured
    answer-position change.
  - Cycling answer-position states approximately repeat every two steps (median lag-2 residual over the last 64 steps:
    2–5×10⁻⁵ of the one-step change).
  - A single random perturbation of relative size 10⁻³ at every position contracts strongly. For three cycling prompts,
    the answer-position distance falls to ~4×10⁻⁷ after 64 updates. 50A contracts more slowly, reaching ~3.8×10⁻⁶ after
    256 updates.
  - These results support numerically robust near-period-2 behavior and attraction along the tested perturbation. They
    argue strongly against an fp32 rounding artifact.

### Where the cycle comes from (`intervene_app.py cycsrc`, `analyze_cycsrc.py`; code and results reviewed by GPT-6 Astra)

fp32, 12 prompts (cycling 23A/B, 24B, 49A, 50A/B; stationary 24A and 49B, the same problems with other demos; 26A/B,
44A/B), K0 = 512 warm-up loops. About 4 GPU-min. Results: `results/cycsrc-20261105-025703/analysis` (`cycle_source.png`,
`summary.json`). The loop map is time-invariant here (the step index only sets noise, which is off).
- **Hidden fixed point.** For every prompt, damped iteration x ← x + ½(F(x) − x) from the cycle midpoint converges to a
  fixed point x* of the full loop map:
  - Relative residual ≤ 2.14×10⁻⁶ at all real positions.
  - In cycling prompts, x* lies within 0.3–7% of the cycle-axis length from the midpoint of the two cycle states at the
    source and selected answer positions (up to 8.7% at another moving position).
  - Its digit-restricted argmax at the saved prediction positions equals the cycle's.
- **Source token: a sign-flipping mode past −1.**
  - At the source token (earliest moving position: first digit of the second operand for 23 and 50; "=" for 24B and
    49A), the one-step Jacobian of that token's own update at x* has a real eigenvalue −1.060, −1.069, −1.009, −1.052,
    −1.0005, −1.086 (23A, 23B, 24B, 49A, 50A, 50B).
  - The eigenvector is aligned with the cycle axis (overlap 0.994–1.000). FD residuals are ~0.001–0.003 at two step
    sizes, and the estimates agree at half and full Krylov size.
  - Independently, a 10⁻³ push along the axis from x* alternates sign each loop and grows by a factor of −1.060, −1.070,
    −1.0095, −1.0525, −1.0016, −1.085.
  - Earlier positions are settled (drift ≈ 5×10⁻⁶), so the source's dynamics are effectively autonomous given its
    prefix.
  - **50A is numerically marginal.** Its excess past −1 (μ ≈ 0.0005) is comparable to the FD diagnostics, and the
    rollout estimate differs from the Ritz value by about twice μ.
- **Two-step map on the cycle (source block).** The largest Ritz values are 0.822, 0.808, 0.963, 0.806, 0.995, 0.745,
  below 1. This is consistent with the earlier perturbation tests. It is not a certificate that no larger mode was missed
  or that the whole sequence is attracting.
- **Flip normal form, as a consistency check.**
  - For small μ (eigenvalue −(1+μ)), the cycle's multiplier should be about 1 − 4μ: predicted 0.762, 0.723, 0.963,
    0.790, 0.998, 0.658.
  - The agreement is close for 24B. 50A is off by ~2.5× in 1 − multiplier.
- **Interpretation (Astra's wording).** "Consistent with a supercritical flip (period-doubling) mechanism at the source
  token." The bifurcation itself is not established: that would require tracking a parameterized family through −1.
- **Freeze test (pin positions to a snapshot after every loop).**
  - Pinning only the settled positions before the source changes nothing.
  - Pinning everything before the selected downstream answer-prediction position, at either cycle phase or at x*,
    eliminates the observed oscillation there in all 6 prompts (drift ≈ 3×10⁻⁶).
  - For 23, 24B and 49A, pinning through the source stops all motion.
  - Single-position pins of intermediate tokens do not stop it.
  - Caveat: "=" itself predicts the first answer digit, so for 24B and 49A one answer-prediction position is the source.
- **Problem 50 has a second cycling position.** Pinning through the first source (the first digit of the second operand)
  leaves "=" and later positions cycling. With everything through position 45 pinned, "=" keeps a relative drift of
  ~0.085–0.10 and a lag-2 ratio of ~4×10⁻⁵. "=" can sustain period-2 motion under a frozen prefix. It is coupled to the
  first source in the free dynamics, and its own flip mode was not spectrally checked.
- **Matched demos (two case studies).** For the same problem with other few-shot demos (24A, 49B), the "=" token's
  validated most-negative real eigenvalue is −0.946 and −0.917, just inside −1, and the prompt is stationary. This is not
  a tracked eigenvalue branch, and it is not evidence of a general few-shot effect.
- **Answer position (selected position only).** At x* the largest |eigenvalue| is 0.75–0.81. Not all of these pass the
  0.01 residual check (23B 0.039, 49A 0.095).

### A period-doubling bifurcation along a contextual-input family (`intervene_app.py bifur`, `analyze_bifur.py`; design, code and results reviewed by GPT-6 Astra)

Runs: coarse `results/bifur-20261105-050715` and zoom `results/bifur_zoom-20261105-051334`. Figure:
`results/bifur_zoom-20261105-051334/analysis/bifurcation.png`. About 4 + 4 GPU-min, fp32.
- **Family.** The loop re-injects the prelude output E at every step. E is contextual, so the "=" token's E differs between
  49B (stationary) and 49A (cycling), the same problem with different few-shot demos. The two prompts are token-aligned.
  The family is F_α(s) = step(E_α, s) with E_α = (1−α)E(49B) + αE(49A), interpolating the whole aligned tensor.
  - α = 0 and α = 1 are the real prompts; intermediate α are not textual inputs.
  - All α share one common initial state.
- **Main result (Astra's wording).** "Along this contextual-input interpolation, we find numerical evidence consistent
  with a supercritical period-doubling bifurcation of the '=' token dynamics near α≈0.71. A consistently tracked real
  mode crosses approximately −1, finite-amplitude period-2 motion exhibits approximately linear amplitude-squared scaling
  with estimated distance beyond the crossing, and the measured cycle-axis two-step multiplier agrees approximately with
  the leading-order flip prediction. Finite precision and slow near-critical transients limit resolution of the onset and
  its asymptotic behavior."
- **Eigenvalue.** The "=" block eigenvalue is tracked by adjacent eigenvector overlap ≈ 1.0. On the coarse grid it runs
  from −0.917 (α=0) to −1.053 (α=1), non-monotonically (it peaks at about −0.88 near α=0.27). Most coarse rows fail the
  fixed-point gate, so the coarse curve is exploratory.
- **Zoom grid (all 11 rows pass the gates).** Gates: prefix fixed-point residual < 1e-5, FD eigen-residual ≈ 0.0017,
  ε-consistency ≈ 0.003.
  - λ = −0.971 (α=0.64) … −0.9978 (0.705), −1.0001 (0.71), −1.0023 (0.715) … −1.0249 (0.78). Local slope dλ/dα ≈ −0.45.
  - The crossing is near α ≈ 0.71. Rayleigh estimates at α=0.71 straddle −1 across FD step sizes, so finer resolution is
    not claimed.
- **Dynamics.**
  - α = 0.64–0.69 settle.
  - 0.70 and 0.705 are unresolved after 1536 loops: the drift plateaus at ~1e-4 at "=" with no clean period 2. No other
    near-unit Ritz value appears in the sampled "=" block. Roundoff amplified near criticality is an untested hypothesis
    (fp64 would test it).
  - α = 0.71 passes the period-2 classifier but is still a slow transient.
  - Clear finite-amplitude cycles sourced at "=" appear at sampled α ≥ 0.715.
- **Amplitude.** Amplitude² (relative, at "=") vs μ = −1 − λ is approximately linear through the origin over
  μ = 0.0023–0.025: slope 0.0125, intercept −6e-6, A²/μ = 0.011–0.013 (5 gated points).
- **Two-step multiplier.** The cycle-axis multiplier of the "=" block is 0.9899, 0.9832, 0.9660 at μ = 0.0023, 0.0048,
  0.0086. The leading-order flip prediction 1 − 4μ gives 0.9907, 0.9808, 0.9656. (At α=0.71: 0.9975, but that state is
  still decaying.)
- **Pushes.** Signed pushes along the tracked eigenvector from x*, with an unpushed control: the antisymmetric response
  alternates sign.
  - The |rate| for 10⁻³ pushes tracks |λ|: 0.9944 at λ = −0.9945, 0.9973 at −0.9978, 0.9995 at −1.0001, 1.0019 at
    −1.0023.
  - A 10⁻⁴ push at α=0.715 gives 0.9993. The cause is unresolved (precision, residual upstream motion, size dependence).
- **Endpoint-start runs** (started from the cycling endpoint's late state). They reach the same amplitudes as the common
  initial state at α = 0.73, 0.75 and 0.78. Below the crossing they shrink but have not finished decaying in 256 loops.
  There is no evidence of bistability, but it is not excluded near the onset.
- **Readout.** The digit-restricted teacher-forced argmax is identical in both cycle phases at every α. The answers do
  change across α, independently of the flip.
- **Separate phenomenon.** At α ≈ 0.33–0.60 there is period-2 motion whose earliest moving position is token 1, the first
  demo token, whose interpolated input blends two different tokens. This is a different upstream source. Calling it an
  off-manifold artifact is plausible but not established.
- **Not shown.** A rigorous bifurcation theorem check: the cubic coefficient was not computed independently, the full
  spectrum and upstream blocks are not certified, and fp32 is a discrete implementation.

### Follow-ups: replication, loop-parity flicker, real-text screen (results reviewed by GPT-6 Astra)

- **Replication on 24A → 24B** (`bifur24`, `results/bifur24-20261105-113450`, ~6 GPU-min).
  - The tracked "=" eigenvalue (position 44) runs from −0.9455 to −1.0091. All zoom rows pass the gates.
  - Wording: "The tracked mode crosses approximately −1 near α≈0.91; finite-amplitude cycles above the crossing and signed
    perturbation responses support the same flip mechanism seen in family 49. Near-critical motion remains unresolved."
    At α≈0.909 the Rayleigh estimates straddle −1. Period-2-like, "="-sourced motion already appears at α=0.905 (λ=−0.999),
    which is near-critical and unresolved.
  - For 10⁻³ pushes, |rate| tracks |λ|: 0.995 at λ = −0.995 and 1.0031 at λ = −1.0031.
  - Runs started from the cycling endpoint settle (to run tolerance) at α = 0.936, 0.968 and 1.0, with A²/μ = 0.044, 0.038
    and 0.042. That is consistent with supercritical scaling, but it is not an exponent estimate. The constant differs from
    family 49, as expected.
  - Common-initial-state runs near onset had not saturated (critical slowing). The token-1 blended-input window
    (α≈0.4–0.67) recurs.
- **Loop-parity flicker (synthetic)** (`bifur49_flicker`, `bifur49_flicker2`). Astra's wording: "Within this
  non-textual contextual-input interpolation, two consecutive late cycle phases select different full-vocabulary argmax
  digits at the third teacher-forced answer position over a narrow sampled window, consistent with loop-parity
  flicker."
  - The window is α ∈ [0.7576, 0.7593] (width ≈ 0.002): "3" in one phase, "4" in the other.
  - The digit-normalized probability swing is ≈0.7 pp. The winning margins are as small as 7e-5 at the window edge and
    about 0.003 at its centre.
  - Only two consecutive phases were saved, so persistent alternation is inferred, not measured. Free-running generation
    was not tested.
- **Real-text screen of few-shot demo sets** (`make_demo_variants.py`, mode `screen`, `analyze_screen.py`,
  `results/screen-20261105-114648`, ~5.5 GPU-min).
  - Design: 6 problems × 10 token-aligned demo sets (A, B and 8 new sets with the same 2/3/4-digit example structure),
    512 loops, readout every loop over the last 64.
  - Late alternating motion (> 1e-4): 18/60 prompts, of which 15 are settled period-2 (lag-2 < 0.01, drift stable within
    2%). The 3 unresolved transients are 23V3 (growing), 44V3 and 50V3 (shrinking).
  - By problem: 23 → 8/10, 24 → 1/10, 26 → 0/10, 44 → 1/10, 49 → 2/10, 50 → 6/10. Motion is demo-dependent within a
    problem. These problems were chosen because they blink, so this is not a prevalence estimate.
  - The earliest position above the motion threshold varies with the demos: the first digit of the second operand, "=",
    or " +" (in all V3 cases). It is not a causal source identification.
  - No teacher-forced argmax changed over the final 64 loops in any of the 60 real-text prompts, including the 18 with
    late motion. The largest parity-averaged probability difference is ≈5%; the smallest parity-averaged margin in a
    blinking prompt is 0.03 (50V1).
  - The screen adds matched stationary/blinking pairs for problems 23, 44 (unresolved), 49 and 50.
- **Literature.** Orbits in Huginn have been reported qualitatively: Geiping et al. 2025 for numerical tokens; Blayney
  et al. 2026 (arXiv 2604.11791) on GSM8K, more often with a long persona system prompt, without period or mechanism. A
  Lyapunov study (ICML 2026) reports that Huginn "converges uniformly in all dimensions". Contraction onto a period-2
  orbit is compatible with negative exponents, so our results challenge only an unqualified fixed-point-convergence
  reading. Period-doubling cascades have been shown in an external soft-token feedback loop of standard LMs (arXiv
  2610.31692), and a fold bifurcation in a small looped transformer (Kim 2026, arXiv 2608.26556). We found no prior
  mechanistic flip-bifurcation analysis of a pretrained recurrent-depth LM's internal loop, or context-controlled
  switching of it. That is a limited search, not a systematic review.

### Prevalence and convergence-mode census on fresh prompts (`make_census_prompts.py`, `intervene_app.py census / ownblock`, `analyze_census.py`, `analyze_ownblock.py`; design, code and results reviewed by GPT-6 Astra)

Runs: `results/census-20261105-{162542,163100,163615,164134}` (4 × ~5 GPU-min); analysis `results/census_analysis`.
Validation: `results/ownblock-20261105-165251` and `results/ownblock_random-20261105-165736`.
- **Prevalence (Astra's wording).** "Under this sampling design — 100 fresh addition problems, each with two randomly
  drawn three-example demo sets — 26/200 prompts met the prespecified settled-period-2 criterion by loop 512 (13%;
  problem-cluster bootstrap 95% interval 8–18%). Ten prompts remained unresolved. No teacher-forced answer argmax
  changed over the final 64 loops."
  - 22% [14, 30] of problems blink with at least one of their two demo sets; 4 problems blink with both.
  - The earliest detected period-2 position is in a demo in 20/26 prompts. Most often it is the "=" of the 4-digit demo
    line (12/26, a plurality, not a majority). The rest start at the query's operand digit (3), "=" (2) or "+" (1).
  - A stable argmax does not mean the logits are unchanged.
- **Trajectory-mode census.** It does not establish an own-spectrum pile-up near −1.
  - All 266 real-negative trajectory modes (from 31 prompts) lie in [−1.05, −0.85]. Positive modes are mostly 0.85–0.95;
    the median complex modulus is 0.82.
  - Adjacent-window consistency is only 18%, so trajectory fits are noisy, and they include upstream forcing.
- **Pinned-prefix validation, diagnostic sample** (16 census own-candidate negatives at "=", "+" and leading digits). The
  own-block leading Ritz value is real-negative and matches the census, −0.836 to −0.981. FD residuals are mostly
  0.002–0.007; one is 0.016 and one 0.074.
- **Pinned-prefix validation, 16 random fitted positions.** The own-block leading Ritz values are complex or positive
  with |λ| 0.67–0.82, below their census trajectory values (0.73–0.94). But the Ritz residuals are 0.03–0.33 (m=24),
  i.e. unconverged.
- **Unbiased by-type sample** (random stationary positions; Ritz convergence again incomplete):
  - "=": real-negative candidates in 5/12, at −0.66 to −0.71, never the leading mode.
  - "+": leading mode real-negative in 3/12 (−0.75, −0.75, −0.87).
  - Leading-space digits: 1/8 (−0.82).
  - Other digits: none detected in 8 (the one-sided 95% upper bound is ~31%).
- **Reading (Astra).** This is a suggestive association of sign-flipping modes with operator and leading-digit tokens,
  not specificity. The strong edge-of-stability hypothesis is not established. "Beyond −1 in blinking prompts" needs
  fixed-point Jacobian evidence per case.
- **Most important remaining check** (**done**: see the own-block spectrum convergence result later in this section): larger or restarted Arnoldi with independent starts and residual validation on the
  same random positions. Free-running generation is still worthwhile before claiming behavioral invisibility. Both are
  done below; they supersede parts of this reading.

#### Own-block spectrum convergence and free-running generation (`intervene_app.py ownspec / freegen`, `analyze_ownspec.py`, `select_freegen.py`, `analyze_freegen.py`; code and results reviewed by GPT-6 Astra)

Runs:
- `results/ownspec-20261105-182425`: same 40 positions as `ownblock_random`; Arnoldi m=64, 3 independent starts, FD
  validation of the top-4 and the 3 most-negative near-real Ritz pairs; ~5 GPU-min.
- `results/freegen-20261105-183834`: ~6.5 GPU-min.

Astra's accepted wording:

> **Own-block spectrum convergence.** We revisited 40 stationary positions sampled by token stratum using Arnoldi
> dimension 64 and three starts. All fixed points passed validation, but only 14/40 positions met the prespecified
> leading-eigenvalue convergence and start-agreement criterion. Leading Ritz **moduli** were substantially more
> reproducible: their across-start spread had median 0.003 and maximum 0.013. Taking the median across starts per
> position, leading moduli ranged from 0.66 to 0.87, with median 0.77. These estimates show no near-unit pile-up in this
> selected stationary sample, although they do not certify the full spectral radius. Multiple near-edge Ritz values
> suggest competition between modes of similar modulus; a spectral "ring" remains an interpretation.
>
> Validated near-real negative modes occurred at 31/40 positions and replicated across starts at 27/40. The
> most-negative validated value per detected position ranged from −0.87 to −0.52, with median −0.70. They occurred in
> every sampled token category, including 4/8 other-digit positions. This supersedes the earlier m=24 non-detection among
> other digits and withdraws the interpretation based on that non-detection: negative modes are not exclusive to
> operators or leading-space digits. These data do not establish equal prevalence across token types, population-wide
> genericity, or a −1 crossing in the census blinking cases.
>
> **Greedy generation across loop budgets.** At budgets 509–512, all six answer-wrong blinking prompts and six matched
> stationary controls produced identical fixed-horizon greedy continuations from the first wrong answer index. All 12
> census reproduction gates matched 64/64, with no tail argmax changes at any generated position. The remaining 20
> answer-correct blinking prompts all emitted newline at the reference termination position. Together with causality and
> the census's unchanged tail argmaxes, this supports identical greedy answers through the first newline for all 26
> census blinking prompts at these four budgets, under the tested initialization and recomputation protocol. It does not
> establish invariance at other budgets or unchanged probabilities.

Details:
- The m=24 leading values were mostly wrong in angle (median conjugacy-aware distance 0.31 from m=64), though their
  moduli were within ~0.03.
- Near the spectral edge: a median of 3 Ritz values (2 after merging conjugate pairs) lie within 0.05 of the leading
  modulus, at widely different angles. That is why the strict "same leading eigenvalue across starts" criterion often
  fails while the modulus agrees.
- By stratum: "+" query 5/5 validated negatives (−0.72 to −0.87); leading-space digits 8/8 (replicated 8/8); other
  digits 4/8 (replicated 1/8). Only 3 positions are negative-dominant ("+" −0.76, "+" −0.87, " 4" −0.82).
- Free-running controls match digit count and first-error index but not overflow, demos or carry structure. One
  blinking prompt shares its problem with another (5 distinct problems).

## Is uncertainty a weak pull? (`select_doubt.py`, `intervene_app.py doubt`, `analyze_doubt.py`; design, code and results reviewed by GPT-6 Astra)

Run `results/doubt-20261105-110426`, about 6.5 GPU-min, fp32. Figure: `analysis/doubt.png`. Summary: `analysis/summary.json`.
- **Design.** 27 teacher-forced answer positions, one per problem, selected from the saved carry run by the top-1 minus
  top-2 digit probability (steps 48–64).
  - Groups: unsure (< 0.4) or confident (> 0.9), crossed with correct or wrong. Group sizes are 8 / 8 / 8 / 3; only 3
    confident-wrong problems exist.
  - Fresh initial seeds: 8 seeds on 15 prompts, 1 seed on the rest.
  - Fixed points by damped iteration (all 27 converged, prefix residual < 1e-5).
  - Pushes of relative size 10⁻³ at the fixed point: random at the answer position, the direction that shrinks the
    top-2 logit gap, and random at every earlier position. All are measured against an unpushed control.
- **Main result (Astra's wording).** "In these selected teacher-forced prompts, low answer margins persist at
  approximate fixed points. We found no clear association between uncertainty and early recovery from random
  answer-position perturbations, and no terminal answer disagreements across eight initializations on 15 prompts. These
  findings argue against a simple account of uncertainty as uniformly weaker local recovery, but do not establish equal
  stability or unique attractors. Two prompts remain nonstationary and show nonrecovering prefix perturbations.
  Readout-aligned perturbations show exploratory directional effects, especially greater transient amplification in
  confident cases. The results are compatible with a readout-geometric account of uncertainty, but do not identify
  distance to a decision boundary as its mechanism."
- **Persistence.** Spearman(selection margin, fixed-point margin) = 0.987.
- **Multi-start.** 0 of 15 prompts gave disagreeing answers across 8 seeds. The answer-position state spread across seeds
  is ≤ 2.6e-5 (excluding moving prompts). With 8 seeds, the one-sided 95% upper bound is ~18% per prompt.
- **Random-nudge recovery** (early finite-time factor per loop, fitted above the late plateau): ρ = −0.15 (p = 0.46).
  The Welch 90% CI for the unsure − confident difference is [−0.006, +0.017] per loop. That excludes only differences
  larger than about ±0.018; it does not show equivalence within ±0.01.
- **Settling loop.** ρ = 0.01; the 90% CI for the difference is [−8, +6] loops.
- **Gap-shrinking nudge, state recovery.** ρ = −0.45 (p = 0.019, Holm 0.094): unsure answers recover slightly slower.
  Window-sensitive.
- **Gap-shrinking nudge, peak transient growth** (max distance / initial): ρ = +0.54 (p = 0.005, Holm 0.028). Medians are
  1.09 for unsure and 1.48 for confident; random nudges never grow.
  - It survives removing the moving prompts and leave-one-out.
  - Exploratory adjustment for carry, digit position, n_digits, length and correctness gives a partial ρ ≈ +0.58.
  - It partly tracks the initial readout response: the partial ρ is ≈ 0.28 controlling for the initial gap change.
  - Nonnormal transient growth is plausible but not distinguished from nonlinear response.
- **Withdrawn and unreliable measures.**
  - The readout-gap decay fit is withdrawn: it is window-sensitive, about half of the fit window lies below the plateau,
    and the signed curves reverse.
  - The answer-block leading Ritz |eigenvalue| estimates are unconverged (FD residual median ≈ 0.08; half/full Krylov
    difference ≈ 0.06), so they are not evidence of equal stability.
  - Prefix-push recovery: ρ = +0.41 (p = 0.04, Holm 0.17).
- **Nonstationary prompts.** #12 (unsure, wrong) and #23 (confident, correct) are still moving at loop 224, and their
  prefix pushes grow (2.3× and 4.6×). #23 looks period-2-like; #12 is unclassified.
- **Caveats.** The groups differ in carry, digit position, n_digits and length, with poor overlap. The sample is
  extreme-margin selected (restricted range). The readout is teacher-forced.

## Carry: is the carry an extra one-notch rotation? (`intervene_app.py carry_traj`, `analyze_carry.py`; reviewed by GPT-6 Astra)

168 fresh held-out prompts, steps 0..64, every answer position. The analysis uses pos≥1 and col≥1 and excludes the
overflow digit: 138 carry rows and 152 no-carry rows from 76 problems. No-carry rows are reweighted to the carry rows'
(n_digits, col) strata. Final accuracy: carry 72%, no-carry 81%. Results: `results/carry-20261104-194505/analysis`.
- **Readout over loops (rows correct at the end).** Both groups transiently put probability on the alternative carry
  outcome around steps 6–12, while P(final) rises:
  - Carry rows, step 8: P(final−1, the "no carry" answer) 0.155 [0.12, 0.19]; P(final+1) 0.128.
  - No-carry rows, step 8: P(final+1, the "carry" answer) 0.21 [0.17, 0.27]; P(final−1) 0.064.
  - This is compatible with uncertainty about the carry bit, but also with generic neighboring-digit confusion.
  - Settling times are similar (median 8 vs 9 steps; mean 10.4 vs 10.5). This does not establish equal computation time
    or rule out a carry-specific intermediate.
  - Mean settling time increases descriptively with dependency length (10.0 / 12.2 / 16.5 steps), but the longer groups
    are small (n = 13 and 2).
- **Decoded phase.**
  - Decoders were trained only on no-carry rows and cross-fitted by problem: a per-step ridge model on a 256-d random
    projection plus nuisance, and the fixed late circle plane. Both show a late negative phase bias on correct carry rows:
    about −19 to −20° from the final digit, toward the pre-carry digit, from step 16 to 64.
  - On no-carry rows the bias is −4° / 0°. The late carry-minus-no-carry contrast is −16° [−21, −10] (projection) and −21°
    [−29, −12] (plane).
  - This is a *decoded bias*, not proof of a physical half-notch displacement. (An earlier version said rows that ended
    wrong show a similar bias and that this weakens the carry reading. The audit below shows that mean was a mixture of
    outputs; that inference is withdrawn.)
  - Alternatives include operand or column-sum information that correlates with the label only in no-carry training data,
    distribution shift, and nuisance interactions.
  - The probes do not reliably resolve an early "pre-carry first, then rotate" sequence: early angular precision is poor
    (held-out no-carry accuracy 0.24–0.30 before step 6).

### Stricter carry audit (`analyze_carry_audit.py`, no model run; code and results reviewed by GPT-6 Astra)

Late features (mean over steps 32–64). Results: `results/carry-20261104-194505/analysis_audit/summary.json`. All intervals
are problem-cluster bootstraps conditional on the fitted decoders, centroids and residualisers (not refitted per
replicate), so they are exploratory.
- **Decoded offset survives matching.** Correct carry rows show an approximately 15° more negative decoded angular offset
  than digit-matched correct no-carry rows:
  - Replication decoder (trained on no-carry rows): −15.7° [−25.3, −5.7]. Strict cells (final digit × previous token ×
    n_digits × col): −17.2° [−38.8, −6.1], but only 8 cells, 27 carry rows and 14 problems.
  - Decoder trained on all correct rows, including correct carry rows: −15.4° [−21.9, −7.7]; strict cells −7.7° [−18.7,
    1.7].
  - So the effect is not solely caused by excluding carry rows from training. That does not rule out decoder bias in
    general.
  - These are contrasts. Absolute mean carry offsets are about −20° (replication) and −9° (all-rows decoder).
- **Decoder-free geometry is direction-consistent but inconclusive under stricter matching.** The measure is the
  projection of each held-out row onto the chords toward the pre-carry and the next digit's no-carry centroids
  (asymmetry = f_pre − f_next), in the 256-d projection:
  - Digit-matched: +0.25 [0.06, 0.44], and +0.29 [0.13, 0.47] after adjusting for operand-digit dummies. Across 10 fold
    seeds the medians are +0.23 and +0.20.
  - Digit × fold matched: +0.22 [−0.20, 0.63]; strict cells: +0.08 [−0.17, 0.31].
  - An inputs-only baseline (separate operand-digit one-hots) gives ≈0. That does not rule out nonlinear column-sum
    information: a state storing only the column sum would give ≈1.0.
  - "Compatible with a trace of the pre-carry sum" is the safe reading. Carry-specific structure and leftover column-sum
    information are not separable here.
  - The "input-blind" decoders (residualised on operand dummies) lose most accuracy and are not identified. Treat them as
    inconclusive.
- **Wrong carry rows (decoder angles, adjacent-error subsets).**
  - 68% of wrong carry rows output the pre-carry digit. These decode +15.4° from their own output, toward the truth
    (19/25 strictly between output and truth). With the all-rows decoder: +20.4°, 23/26 between.
  - Rows outputting final+1 decode −30.7° from their output (6/8 between).
  - So decoded angles usually fall between the model's output and the truth, rather than at the output. This is contrary
    to a pure training artifact pulling toward the pre-carry digit. It does not show that whole hidden states lie
    between them.


## Weekday pilot: a circular concept outside arithmetic (`make_weekday_prompts.py`, `intervene_app.py wkcollect`, `analyze_weekday.py`, `analyze_weekday_start.py`; design, code and results reviewed by GPT-6 Astra)

Run `results/wk-20261105-185928` (~4.6 GPU-min, fp32).
- **Design.** 98 prompts: 7 start days × 7 offsets (1–7) × 2 few-shot templates. Template A is Q/A with number words;
  template B is a statement with digits. The state is read at the answer-prediction position at every loop 0–96.
- **Prespecified question.** Does recurrence rotate the state from the start day by the offset, or relax directly to the
  answer?
- **Analysis rules.** Frozen before data, after synthetic validation (rotation, short-way angular, and cross-fade worlds
  at several noise levels). Every synthetic verdict was correct or INCONCLUSIVE.
  - A per-loop-fraction version of the non-additivity statistic falsely flagged a weak-early-signal cross-fade world.
    It was replaced by a fixed late denominator.
  - Naive class means are anti-biased in this Latin square, so decoding is factor-adjusted.

Astra's accepted wording:

> On this balanced 98-prompt pilot using two few-shot templates, Huginn did not reliably perform weekday-offset
> arithmetic: loop-96 accuracy was 24.5% and 20.4% (uniform baseline 14.3%), with strong template-dependent response
> biases. Full-vocabulary and seven-way predictions agreed. Extending 49 selected prompts to loop 192 changed no endpoint
> answers. The prespecified answer-factor analysis found no positive held-out geometry fit, and all seven phase-validity
> gates failed; rotation versus direct computation therefore remains INCONCLUSIVE.
>
> A post-hoc exploratory analysis found a canonical cyclic component associated with the supplied start day: canonical
> weekday order ranked first among 360 cyclic orders at all four examined readouts. This supports structured input
> representation, but does not establish successful weekday computation, a continuous circle, or rotational dynamics.

Details:
- **Response biases.**
  - Template A answers "start + 1" regardless of offset in 25/49 prompts; offset-1 accuracy is 6/7.
  - Template B answers "Monday" in 32/49.
  - Offset 7 is never answered correctly (0/14).
- **Decoding.** Conditional start-day decoding is 96.9–100% from loop 5. Answer decoding is 4–14%.
- **Exploratory start-day circle** (held-out R²):

  | Readout | Circle | Categorical | Best of 5040 lines | Max of 200 random rank-2 codes |
  |---|---|---|---|---|
  | Answer position, loop 8 | 0.29 | 0.45 | 0.19 | 0.26 |
  | Answer position, loop 32 | 0.20 | 0.38 | 0.12 | 0.20 |
  | Answer position, loop 96 | 0.19 | 0.36 | 0.12 | 0.18 |
  | Start-day token, loop 96 | 0.36 | 0.68 | 0.21 | 0.34 |

  - The margin over random codes is small at loops 32 and 96, and the readouts are correlated.
  - The decline from loop 8 to 96 is not evidence that recurrence erases the circle: each readout refits its nuisance
    effects and PCA.
  - Engels et al. (arXiv 2405.14860) already reported weekday circles in other LLMs. This is a Huginn-specific
    replication.
- **Next (Astra).** Status: the first item is **done** in [Digit-circle rotation pilot: carry-conditioned units prediction](#digit-circle-rotation-pilot-carry-conditioned-units-prediction-make_digit_rotation_promptspy-intervene_apppy-dgcollect-analyze_digit_rotationpy-design-code-and-results-reviewed-by-gpt-6-astra); months are **done** in [Month shape and opposite-month context blends](#month-shape-and-opposite-month-context-blends-make_month_promptspy-intervene_apppy-mblend-analyze_month_blendpy-design-code-and-results-reviewed-by-gpt-6-astra); clocks remain **open**.
  - Ask the rotation-versus-direct question on a task Huginn can do, e.g. the addition digit circle. Require a
    prespecified behavioral qualification first.
  - Offset-1 weekdays are only a positive control: rotation and direct relaxation predict the same path there.
  - Months and clocks would need a behavioral screen first.

## Predictive source under-relaxation (`select_relax.py`, `intervene_app.py relax`, `analyze_relax.py`; design, code and results reviewed by GPT-6 Astra)

Run `results/relax-20261105-200208` (~6.2 GPU-min, fp32). Prompts: all 26 census blinking prompts, none used in the earlier
cycle-source or bifurcation work.

**Method.**
- **Source token:** the census-selected first period-2 position.
- **Fixed point:** damped iteration finds the unstable fixed point between the two cycle states.
- **Eigenvalue:** two independent Arnoldi starts (m=40) on the source token's own-block Jacobian. Eligibility requires
  an exactly real eigenvalue, FD-validated, agreeing across starts, and past −1 by more than its validation margin.
- **Prediction, saved before any intervention:** relaxing only that token, s ← (1−a)s + aF(s), maps the mode to
  1 − a + aλ. The damping strength a was chosen per condition to target multipliers −0.6 and −0.85 (predicted to stop
  the blink), or halfway between −1 and λ (predicted to keep blinking).
- **Controls:** sham (a = 1), and the same a applied at a downstream blinking token.

Astra's accepted wording:

> All 26 census-selected blinking prompts passed the prespecified fixed-point and real flip-mode validation criteria.
> Relaxing only the census-selected token, using its own-block eigenvalue to target multipliers −0.6 or −0.85, met the
> stopping criterion in the same 24/26 prompts. Sham blinking persisted in 26/26, above-threshold source relaxation in
> 25/25 tested prompts, and downstream relaxation preserved blinking at the source in 26/26. The two unresolved cases
> showed substantial source-drift reduction and persistent upstream activity; upstream forcing remains a possible
> explanation. These results support eigenvalue-guided local suppression at the tested settings, without locating the
> exact transition threshold or establishing whole-sequence convergence, observed in 15/26 under the −0.6 condition.

Details:
- **Flip eigenvalues:** λ from −1.007 to −1.226 (median −1.071). Predicted thresholds α* = 2/(1−λ) range 0.899–0.996.
- **Halfway condition** omitted for uid 20059: μ = 0.007 is below the prespecified 0.02 cutoff.
- **Decay rate at the −0.85 setting:** fitted per-loop decay of source drift (loops 8–56), median 0.851 (IQR
  0.849–0.853). This agrees with the targeted flip-mode magnitude for this window and observable. It is not proof of
  the asymptotic rate of the full dynamics.
- **Decay rate at the −0.6 setting:** median 0.918, not 0.6. Slower non-flip modes or upstream forcing are hypotheses,
  not verified: few additional modes passed FD validation (0 for 14 prompts, 1 for 9, 2 for 2, 3 for 1).
- **Halfway condition amplitude:** drift relative to sham, median 0.682 (IQR 0.664–0.694), versus √½ ≈ 0.707.
  Descriptive only; it does not establish square-root scaling.
- **The two unresolved prompts:**
  - Source drift fell 83× (uid 20041) and 491× (uid 20117) under the −0.6 setting.
  - Upstream positions were active in sham, unchanged by relaxation (20041: 42–45; 20117: 27–30).
  - The census-selected source was the first *period-2* position; earlier non-period-2 motion was not selected.

## Digit-circle rotation pilot: carry-conditioned units prediction (`make_digit_rotation_prompts.py`, `intervene_app.py dgcollect`, `analyze_digit_rotation.py`; design, code and results reviewed by GPT-6 Astra)

Run `results/dg-20261105-200850` (~4 GPU-min, fp32).
- **Design.** Units digits a × b (full 10 × 10) × 2 contexts (2a + 5b with demo set A; 4a + 3b with demo set B).
- **State.** Read at the teacher-forced tens answer token, which predicts the units digit U = (a + b) mod 10, at every
  loop 0–96.
- **Analysis rules.** The 10-level generalization of the weekday rules, frozen before data and validated on synthetic
  data. Additions: a behavioral qualification gate and an endpoint gate.

Astra's accepted wording:

> On 200 balanced, carry-conditioned units-prediction prompts, Huginn achieved 99.5% digit-restricted and
> full-vocabulary accuracy at loop 96. The adjusted endpoint representation supported canonical cyclic digit structure
> (held-out R² 0.534; categorical 0.605), and all ten fold planes passed the prespecified validity gates.
> Context-centred projections grew outward during recurrence and were well described by per-loop additive operand and
> answer effects, with little transient excess non-additivity. Exploratory early phases were closer to the answer than
> either operand, but do not establish emergence at the final angle. The prespecified rotation test was INCONCLUSIVE:
> only 3/80 primary contrast rows qualified for start=a, and 4/80 for start=b. Thus this pilot provides no affirmative
> evidence for start-anchored offset rotation; it does not establish direct or cross-fade computation. The
> teacher-forced tens token supplies carry information throughout.

Details:
- **Accuracy by loop:** 6.5% (loop 0), 15.5% (4), 77% (8), 98.5% (12), 100% (16–40), 99.5% after.
- **Geometry comparisons** (held-out R², in 20 training PCs of the nuisance-adjusted states):
  - Canonical line 0.336; ×3 circle −0.01; random rank-2 codes at most 0.356.
  - No sampled cyclic order scored higher than canonical; the reversed canonical order, which spans the same code, tied.
  - Centroid spacing is uneven, including an 83–87° wraparound gap.
- **Radius** (relative to the loop-96 centroid radius): 0.04 at loop 0, 0.56 at 4, 0.86 at 8, 1.27 at 12, ~1.41 from 16.
- **Non-additivity:** the statistic was 0.0004 (transient excess). Late additive-prediction SSE is ~4.6% of late
  in-plane variance.
- **Early phases** (exploratory, first reliable sample at loops 1–3): median distance to U 1.54 digits (32.5% within 1),
  versus 3.29 to a (8.5%) and 3.36 to b (10.5%). Net unwrapped displacement afterwards: median 0.91 digits.

### Threshold sweep (`select_sweep.py`, `intervene_app.py relaxsweep`, `analyze_sweep_relax.py`; code and results reviewed by GPT-6 Astra)

Run `results/rsweep-20261105-203941` (567 s including model load).
- **Predictions** are the α* values saved by the relax run, not recomputed.
- **Conditions:** relaxation with a = r·α* for r ∈ {0.96, 0.98, 0.99, 1.01, 1.02, 1.04}, 192 loops each. The predicted
  flip multiplier is 1 − 2r.

Astra's accepted wording:

> The prespecified primary criterion—all six conditions matching prediction—was met by 5/24 previously resolved
> prompts. All 19 nonconfirmations were UNCLEAR at r=0.96; all other conditions matched prediction. These cases reached
> small late drifts (median 4.13e-6), consistent with a numerical floor but not establishing its cause; we retain the
> prespecified classifications.
>
> Descriptive above-side extrapolation gave median r*=1.00169 (IQR 1.00135–1.00325; all 24 within 2% of prediction).
> This is a model-dependent extrapolated crossing, not direct threshold localization. Below-side extrapolation gave
> median 1.01260 (22/24 within 2%), with slope 1.07 versus predicted 2. An exploratory earlier window recovered the
> predicted r=0.96 decay rate (0.91898 versus 0.92), supporting floor contamination as an explanation for the
> below-side discrepancy.

Details:
- **Decay rates in the prespecified window (loops 32–160):** r=0.98 gave 0.9600 (predicted 0.96); r=0.99 gave 0.9794
  (predicted 0.98).
- **Bracketing:** the observed DECAY/PERSIST labels bracket the transition between r = 0.99 and 1.01, if these
  finite-time labels are read as stability.
- **Unresolved prompts:**
  - uid 20041 persists at all six r, consistent with (not shown to be) upstream forcing.
  - uid 20117 decays at 0.98 and 0.99, is UNCLEAR at 0.96, and persists above the threshold.

## Does settling speed predict how many loops are needed? (`make_settle_prompts.py`, `intervene_app.py settle`, `analyze_settle.py`; design, code and results reviewed by GPT-6 Astra)

Run `results/settle-20261105-204943` (258 s, fp32). 60 fresh problems (20 each of 2-, 3- and 4-digit operands), 217
teacher-forced answer positions.
- **Predictors:** the own-block Jacobian's Ritz modulus at the fixed point, and online drift-ratio estimates.
- **Outcomes:** loop at which the argmax stabilizes, KL-exit loop, and state convergence (drift < 1e-4).
- **Early-exit simulation:** calibration/evaluation split by prompt; rules compared are fixed budgets, Huginn-style KL
  stopping, an online rate-extrapolation rule, and a Jacobian rule.

Astra's accepted wording:

> On 217 teacher-forced answer positions from 60 prompts, the tested settling-rate methods showed no practical advantage
> over KL stopping. The primary Jacobian–convergence correlation was inconclusive (Spearman 0.24, 95% CI −0.15 to 0.64;
> 31 usable, uncensored positions). Both specified convergence-time predictors performed worse than a
> calibration-median baseline on evaluation completers. Calibration-selected RATE stopping required 6.4 more loops than
> KL (95% CI 4.4–8.4), with identical observed final-token agreement of 97.4%; the prespecified practical-usefulness
> criterion failed.
>
> Argmax stabilization preceded threshold-defined state convergence substantially: median 7 versus 74 loops among
> completers. This is consistent with prolonged state relaxation after token stabilization, but does not establish why
> prediction failed. Conclusions are limited by low Jacobian coverage, censoring, finite-horizon definitions, teacher
> forcing, and exclusion of readout overhead.

Details:
- **Coverage:**
  - Jacobian estimates are usable (FD residual < 0.05) for 37/217 positions.
  - 36 positions are censored: drift is still ≥ 1e-4 at loop 128.
  - Final accuracy is 90.3%.
- **Selected early-exit rules on evaluation prompts:**

  | Rule | Mean loops | Agreement with loop 128 |
  |---|---|---|
  | Fixed budget, 48 loops | 48 | 98.2% |
  | KL, τ = 1e-4 | 25.1 | 97.4% |
  | RATE | 31.5 | 97.4% |
  | Jacobian (not online) | 115 | 99.1% |

  - The KL–RATE agreement difference has a 95% CI of [−2.7, +1.9] percentage points, so equivalence is not shown.
  - Neither adaptive rule reached 99% evaluation agreement.

## Budget robustness on real multiple-choice prompts, and a global relaxation stabilizer (`fetch_data_app.py`, `make_mcq_prompts.py`, `intervene_app.py mcq`, `analyze_mcq.py`; design, code and results reviewed by GPT-6 Astra)

Runs:
- Baseline (α = 1): `results/mcq_base-20261106-014652` and `results/mcq_base-20261106-015605`.
- Relaxed (α = 0.85 at every position): `results/mcq_relax-20261106-020236` and `results/mcq_relax-20261106-020942`.
- Each condition ran as two 50-prompt shards because of the end-to-end deadline; prompts and initialization are
  identical across conditions. Merged analysis: `results/mcq_analysis`.

**Design.** 100 random ARC-Challenge test questions (4 options, ≤ 120 tokens), zero-shot letter format, 256 loops.

Astra's accepted wording:

> On 100 length-filtered, four-option ARC-Challenge prompts with one initialization per prompt, 12 met our hidden-state
> period-2 criterion (Wilson 95% interval: 7.0–19.8%). Tail answer choices were already stable, and conditional
> letter-probability oscillations were small. Relaxation at α=0.85 eliminated detected blinking, with two prompts
> remaining unresolved, and satisfied all four prespecified empirical criteria. Endpoint answers were unchanged.
> Practical-budget metrics were mixed, and accuracy remained near chance; these results support dynamical stabilization
> on this sample, without establishing a task-performance benefit or cost-free inference.

Details:
- **Baseline classes:** 76 stationary, 12 unresolved.
- **Baseline oscillation size:**
  - Maximum odd/even swing in the four-letter probabilities: 0.0049. No answer flips in the tail.
  - Median TV between loops 255 and 256: 9.4e-4 for blinking prompts, 3.5e-6 for stationary ones.
- **Under relaxation:**
  - 98 stationary, 2 unresolved, 0 blinking.
  - The 76 prompts stationary in both runs agree at loop 256 (median TV 3.9e-6).
  - Tail parity TV falls from 3.4e-4 to 4.2e-6.
- **Practical budgets:**
  - Median confirmed answer stability is 15 loops in both conditions.
  - Disagreement with the loop-256 answer at budget 16 is 26 (baseline) versus 33 (relaxed).
  - Accuracy is about 26%, against 25% chance.
- **Framing:** "theory-motivated", not guaranteed. Relaxation preserves fixed points and transforms local eigenvalues,
  but not necessarily basins or trajectories.

## Pointer-chasing behavioral screen: extra loops do not add hops (`make_pointer_prompts.py`, `intervene_app.py mcq` mode `ptr_screen`, `analyze_pointer_screen.py`; code reviewed by GPT-6 Astra; exploratory)

Run `results/ptr_screen-20261106-023014` (~6.4 GPU-min). An earlier attempt, `ptr_screen-20261106-022252`, hit the
deadline and saved nothing.
- **Design.** 120 prompts. Each has a random 10-cycle lookup table f over the digits, three worked demos (depths 1–3),
  and a query f^n(x) with n = 1..6 (20 per depth, answers balanced). All intermediate values are distinct.
- **Accuracy at loop 96** (chance 10%): 90% at depth 1 (settling around loop 16); 5–10% at depths 2–6.
- **Answers reflect one hop.**
  - At depth 2, 15/20 final answers equal f(x), a single hop (chance for "is an intermediate" ≈ 20%).
  - At depths 3–6, "one hop" is the most common single outcome (4–7 of 20); the rest look scattered.
  - Answers at loop 32 and loop 96 are essentially identical.
- **Reading:** in this nested-notation format, Huginn performs one lookup, and additional recurrence does not add
  sequential hops.
- **Caveats** (Astra): demo chains can overlap query subchains; there are 20 prompts per depth; results are specific to
  this format.

## Carry-chain procedure experiment, stage 1 (`make_carry_chain_prompts.py`, `intervene_app.py mcq` mode `carry_chain`, `analyze_carry_chain.py`; code and results reviewed by GPT-6 Astra)

Run `results/carry_chain-20261106-024037` (228 s).
- **Design.** 7-digit + 7-digit sums, 54 tokens each.
- **Target:** the column-5 answer digit, read after the teacher-forced column-6 digit.
- **Chain:** k = 0–4 propagate columns (sum 9) lie between the target and a carry-deciding column (generate or kill,
  balanced); 30 prompts per k.

Astra's accepted wording:

> In these tested formats, Huginn showed a behavioral pattern consistent with a short effective dependency horizon. The
> prespecified carry-chain test did not detect increasing attainment time with chain length (ρ=0.084, one-sided
> permutation p=0.160; 87/150 censored). Exploratory carry-choice analysis showed positive decider tracking with no
> intervening propagate column, but no consistent positive tracking with one to four. The pointer screen showed 90%
> accuracy at depth one, 5–10% at greater depths, and one-hop answers on 15/20 depth-two queries. Increasing recurrence
> did not demonstrate reliable extension to longer dependencies, although individual answers changed. These results are
> compatible with refinement toward stable predictions, but do not identify the computation's mechanism or establish a
> general one-step limit.

Details:
- **Accuracy at loop 96 by k:** 0.60, 0.30, 0.33, 0.43, 0.43.
- **Carry-flip share of errors by k:** 4/12, 16/21, 15/20, 13/17, 15/17.
- **Carry tracking (exploratory):**
  - k = 0: answers imply carry = 1 in 7/9 with a generating decider versus 2/13 with a killing one.
  - k ≥ 1 pooled: 23/50 (generate) versus 32/54 (kill).
- **Answers do change with more loops:** 52/150 answers change between loops 16 and 96 (18 become correct, 16 become
  incorrect).
- **Pointer screen correction:** loop-32 and loop-96 answers agree on 115/120 prompts.
- **Open confounds:**
  - 7-digit queries exceed the demos' 2–4-digit range.
  - Decider position and column composition change with k.
  - Demo set is coupled to the column-5 type.
  - Prelude/coda computation is not isolated.

## Month shape and opposite-month context blends (`make_month_prompts.py`, `intervene_app.py mblend`, `analyze_month_blend.py`; design, code and results reviewed by GPT-6 Astra)

Run `results/mblend-20261106-025402` (337 s).
- **Design.**
  - Prompts: "What month comes after M?", 2 few-shot templates × 12 months.
  - Blends: E_a = (1−a)E(m0) + aE(m1) for 6 opposite and 6 quarter month pairs, a = 0.1–0.9.
  - One common init per template; 96 loops.
- **Analysis rules:** frozen before data and validated on synthetic arc, chord and noise worlds.

Astra's accepted wording:

> At loop 96, all 24 pure prompts selected the correct successor among month tokens. The prespecified
> leave-one-month-out canonical cyclic code achieved R²=0.215, exceeding all 5,000 sampled cyclic reorderings and the
> tested line and random rank-2 controls. This supports cyclic structure in finite-time representations. However, no
> pure reference passed the stationarity criterion, and the leading two-PC plane failed canonical ordering; the
> prespecified path verdict is therefore INCONCLUSIVE.
>
> Exploratorily, opposite-month context blends produced predominantly endpoint-successor readouts, with little
> intermediate-month selection. This describes categorical output switching but does not establish winner-take-all
> hidden dynamics or rule out continuous cyclic representations. Projected midpoint radii remained reduced through loop
> 96; these projections do not establish settled paths around or across a ring.

Details:
- **Shape baselines:** random cyclic orders reached at most R² 0.071; the best of 2001 lines 0.090.
- **Answer switching:** switches occur between roughly a = 0.3 and a = 0.7. One exception: on one May→Nov path, August
  takes 29% at a = 0.4.
- **Readout at a = 0.5:** conditional month-token mass is 0.85 on endpoint successors versus 0.06 on the in-between
  months' successors (uniform baselines 1/6 and 0.5); absolute month mass is 0.94.
- **Non-stationarity:**
  - Answer-state change between loops 64 and 96: median 3.5% (2.9% for pure rows).
  - Late-8 max-position drift: about 3e-3.
  - No readout evidence of dominant period-2 behavior.
  - The role of the shared initialization is untested.
- **Phase-derived quantities** are uninterpretable because canonical ordering failed in the top-2 plane.

## Steering a looped model: one-shot state edit vs persistent context edit (`make_steer_dirs.py`, `intervene_app.py steer`, `analyze_steer.py`; design, code and results reviewed by GPT-6 Astra)

Run `results/steer-20261106-032913` (243 s).
- **Prompts:** 50 held-out digit-rotation prompts (5 per answer digit), with the answer position settled after 96 loops.
- **Direction:** the adjusted one-notch chord C[U+1] − C[U], fit on held-out cells, or a norm-matched random direction;
  scales 1 and 2.
- **Context edit:** chosen so that W_e·d_E = W_s·d_s. The adapter is a linear layer on [state, E], so the context edit is
  exactly the state-input push applied every loop. At t = 1 the two methods agree (max |difference| 4e-4).

Astra's accepted wording:

> On 50 held-out addition prompts, the tested one-shot answer-position state edits decayed back to the sham state,
> whereas matched persistent context-injection edits produced a shifted, numerically settled answer-position state under
> continued forcing. At scale 1, the median target-versus-source log-odds gain was 3.72 (cell-stratified bootstrap 95% CI
> [2.46, 5.59]), and median state displacement was 2.51 times the matched input-push norm. This is consistent with
> recurrent accumulation but does not identify a linear response mechanism. Target argmax transitions occurred on 8/50
> prompts, versus 1/50 for norm-matched random edits; 12/50 produced other digits. Doubling the scale yielded only 5/50
> target transitions and 31/50 other-digit outputs. These results demonstrate persistent directional bias under continued
> injection, with limited control over the final answer.

Details:
- **One-shot chord at scale 1:** gain 4.13 immediately, then 1.39, 0.53 and 0.09 after 1, 2 and 4 loops; 0 from
  loop 16. The state returns to the sham state (ratio 2.2e-5).
- **Context chord:** all 50 prompts at both scales are settled (late drift < 1e-4, lag-2 closure > 1.06).
- **Paired comparisons at scale 1:**
  - Context chord minus context random: 2.94 [1.49, 4.82].
  - Context chord minus one-shot chord: 3.72 [2.46, 5.03].
- **Accumulation (locally):** δx∞ ≈ (I − J)⁻¹ J d. This is a plausible reading of the 2.51× norm ratio (1.70× along the
  chord), not a measured mechanism.
- **Off-target outputs at scale 2:** 14/50 are U+2, i.e. apparent overshoot.

## Truth in a looped model, stage 1: reading (`make_truth_prompts.py`, `intervene_app.py truth`, `analyze_truth.py`; design, code and results reviewed by GPT-6 Astra)

Run `results/truth-20261106-035631` (416 s).
- **Datasets** (240 statements):
  - CITIES: "The city of X is in Y.", 60 cities, each true once and false once. Wrong countries come from a derangement
    of the true countries, so country marginals are balanced.
  - COMPARE: "a is larger than b.", 60 two-digit pairs, both orders.
- **Formats:**
  - A: bare statement after an unlabeled prefix.
  - B: few-shot True/False judgment.
- **Measurement:** mass-mean probes on states at the statement's final "." (and at the answer position), held out by unit
  (city or pair).

Astra's accepted wording:

> On these two balanced datasets, mass-mean probes decode truth labels from statement-final recurrent states on held-out
> cities or number pairs (loop-64 AUROC: 0.997 and 0.914; within-unit permutation p=0.00498 each). Separation increases
> from loop 4 to 64, with conditional descriptive bootstrap intervals above zero; COMPARE peaks earlier at loop 16. These
> results show increasing linear accessibility during recurrence, without establishing when truth information first
> arises or distinguishing abstract truth from task-specific relational features. Transfer across prompt formats is
> strong and across datasets asymmetric. Exploratory judgment logits rank cities accurately but favor False at the
> default threshold. Behavioral accuracy falls below the prespecified Q4 interpretation threshold in both datasets.

Details:
- **Held-out d′ by loop** (1, 2, 4, 8, 16, 32, 64):

  | Dataset | Loop 1 | 2 | 4 | 8 | 16 | 32 | 64 |
  |---|---|---|---|---|---|---|---|
  | Cities | −0.67 | −0.33 | 1.43 | 2.69 | 5.55 | 5.69 | 5.73 |
  | Compare | 0.58 | 0.93 | 0.94 | 0.68 | 2.64 | 1.76 | 1.81 |

- **Transfer:**
  - Format A → B: 0.978 (cities) and 0.898 (compare).
  - Cross-dataset: cities → compare 0.63, compare → cities 0.90; direction cosine 0.14.
- **Judgments** (exploratory):
  - Accuracy at the default threshold is 0.63 / 0.55, and all errors are true statements judged False.
  - Cities judgment-logit AUROC: 0.54 at loop 1, 0.82 at loop 8, 0.986 from loop 32.
  - The best-threshold accuracy of 0.98 is an in-sample optimum, not validated.
  - Compare judgment AUROC is 0.78, versus 0.988 for the internal B-dot probe.
- **Caveats:**
  - COMPARE labels are predicted by signed magnitude/order, and numbers are not held out individually.
  - Relation reversal and negation tests are needed before claiming abstract truth.

## Truth stage 2: steering at the statement-final token (`truthsteer`, `analyze_truthsteer.py`; code and results reviewed by GPT-6 Astra)

*Post-hoc note (2026-11-07, from Claude Opus 5.5 and GPT-6 Astra reviews of the write-up):* the doses here (class-mean distances) were about 7.5× larger for CITIES (≈ 44) than for COMPARE (≈ 5.8). Dose-normalized medians still differ (0.012 vs 0.008 logits per unit norm, 1.6×), so part of the CITIES-vs-COMPARE difference in this section reflects dose. Matched-dose equivalence was not tested. Numbers: `results/truthsteer_dose_by_dataset.json`.

Run `results/tsteer-20261106-041051` (407 s).
- **Items:** 60 judgment-format items (15 held-out units per dataset).
- **Direction:** fold-specific B-dot truth direction; dose 1 = training class-mean distance (~1/3 of the state norm).
- **Conditions:** one-shot state edit versus adapter-matched context edit (active 24 loops, then removed for 24), both
  signs, plus random and sham controls.

Astra's accepted wording:

> Held-out truth directions were highly decodable, but intervening at the statement-final token produced limited causal
> control over judgments. Sustained injection passed the prespecified persistence, specificity and dose criteria, with
> substantial dataset heterogeneity. One-shot erasure and post-removal recovery remained inconclusive because their
> minimum-effect gates were not met. Judgment flips were confined to true statements initially judged correctly; no
> initially correct false statements flipped to true. These findings establish sensitivity to the tested intervention but
> do not identify the mechanism or demonstrate a general truth-control variable.

Details (median movement toward the opposite label, logits):
- **One-shot:** 0.02 after 1 loop, 0 from loop 8.
- **Context:** 0.06 at loop 24 (CI [0.051, 0.186]); about 0 after removal.
- **By dataset (context, loop 24):** cities 0.54, compare 0.04.
- **Specificity versus random:** 0.048 [0.042, 0.057]. **Dose (1 versus 0.5):** 0.014 [0.008, 0.017].
- **Flips:**
  - 4/4 sham-correct true cities flipped to False.
  - 0/15 false cities and 0/15 false comparisons flipped to True.

### Truth stage 2b: steering at the answer position (`truthsteer` site=ans, `analyze_truthsteer.py` with prespecified P5; reviewed by GPT-6 Astra)

*Post-hoc note (2026-11-07, from Claude Opus 5.5 and GPT-6 Astra reviews of the write-up):* the doses here (class-mean distances) were about 10× larger for CITIES (22.4–23.6) than for COMPARE (2.3–2.6). Dose-normalized medians are similar (0.025 vs 0.026 logits per unit norm), so the answer-site median difference is largely accounted for by unequal doses. Matched-dose equivalence was not tested. Numbers: `results/truthsteer_dose_by_dataset.json`.

Run `results/tsteer_ans-20261106-042907` (433 s).
- **Setup:** same items, conditions and timing as stage 2, now at the answer position with fold-specific answer-site
  truth directions (dose 1 = class-mean distance, median 12.5).

Astra's accepted wording:

> Across two datasets, truth labels were highly decodable from held-out recurrent states, but steering along the tested
> mean-difference directions produced limited, heterogeneous control over judgments at both intervention sites. At the
> answer site, sustained injection passed the prespecified persistence, specificity, and dose criteria, with pooled
> median opposite-label movement of 0.079 logits (95% CI [0.069, 0.103]). It failed the prespecified
> meaningful-bidirectional criterion. Effects were substantially larger for cities than numerical comparisons. Observed
> one-shot effects decayed toward zero, but minimum-effect gates left erasure and post-removal recovery inconclusive.
> These results separate decodability from reliable behavioral control; they do not establish a general truth-control
> variable or a dynamical stiffness mechanism.

Details:
- **One-shot:** median movement 0.46 at offset 0 (coda only), 0.10 after 1 loop, −0.03 at loop 4, about 0 from loop 8.
- **Two-way test (P5):**
  - True items toward False: 0.29 [0.069, 0.651].
  - False items toward True: 0.071 [0.055, 0.150].
  - The P5 bootstrap pooled datasets instead of stratifying by dataset as documented; the median-based failure stands.
- **Flips:** 2/15 false cities flipped to True; 0/15 false comparisons.
- **Comparison with digit steering** (log-odds 3.72 at a similar input norm): not a controlled comparison. "A stiff truth
  direction" remains a hypothesis. The local prediction for adapter-matched forcing, δx∞ ≈ (I − J)⁻¹ J d, could separate
  recurrent attenuation from weak readout coupling.

## Why truth steers weakly: a linear-response (resolvent) decomposition (`intervene_app.py respond`, `analyze_respond.py`; design, code and results reviewed by GPT-6 Astra)

*Post-hoc note (2026-11-07, from a Claude Opus 5.5 review of the write-up):* truth coupling splits into two dataset clusters because of the dose difference noted under Truth stage 2 (per unit norm: CITIES 0.041, COMPARE 0.038; digits 0.207). The per-norm gap mainly reflects readout sensitivity (‖g‖ 2.76 vs 0.66). Among positive responders, truth retention is somewhat lower than digit retention (geometric mean 0.63 vs 0.82); the all-item retention medians mix in sign reversals. Digit responses at full dose are superlinear (3.72 observed vs 1.44 predicted).

Runs:
- `results/respond_digits-20261106-054706` (172 s; 50 digit items, chord direction).
- `results/respond_truth-20261106-055046` (259 s; 60 truth items, answer-site direction toward the opposite label).

**Method.**
- At the settled answer-position state x* (the last token, so J is exactly the own block), with readout gradient g
  (autograd through the coda), the following are computed:
  - immediate coupling S0 = g·d;
  - the finite-horizon prediction S_T = g·Σ_{k≤48} J^k d (J v by central FD; Neumann series, 64 terms).
- These are compared with the observed response to continued forcing s ← F(s + dose·d), the adapter-matched context
  edit, at doses 0.1 and 1.

Astra's accepted wording:

> At settled answer-position states, a local linear-response model accurately predicted small-dose responses to
> continued forcing over 48 recurrent steps, meeting the prespecified criterion in both tasks. Immediate readout coupling
> was stronger for the tested digit directions than for the truth directions, while the hypothesized stronger recurrent
> attenuation of truth directions was unsupported. These findings favor weak immediate coupling over differential
> recurrent attenuation as an explanation of the observed steering gap in this setting. Recurrence nevertheless reversed
> the predicted response on 42% of valid digit items and 19% of truth items, and digit responses were substantially
> nonlinear at full dose. Among positive responders, the log-response decomposition attributed most of the between-task
> difference to immediate coupling.

Details:
- **Validity:** 50/50 and 59/60 valid items; Neumann tail ≤ 3e-8; FD step sensitivity 0.0035.
- **Linear prediction (L1):** 90% / 100% of items within tolerance; Spearman 0.995 / 0.999.
- **Medians:**

  | | Digits | Truth |
  |---|---|---|
  | S0 | 3.36 | 0.76 |
  | S0 per unit push norm | 0.207 | 0.0385 |
  | S_inf | 1.44 | 0.063 |
  | Retention | 0.41 | 0.66 |
  | State amplification | 2.6× | 2.1× |
  | Observed response, dose 1 | 3.72 | 0.083 |

- **Hypothesis tests:**
  - H_coup ratio 4.43 [3.01, 40.7] holds.
  - H_att ratio 0.62 [−0.22, 1.22] fails.
- **Log decomposition (positive responders only):** difference 3.14 = coupling 2.88 + retention 0.26.
- **Next (Astra) — done**, see [Loop-aware steering: the resolvent-optimal direction d*](#loop-aware-steering-the-resolvent-optimal-direction-d-dstar_vjp-make_dstar_pushespy-dstar_eval-analyze_dstarpy-design-code-and-results-reviewed-by-gpt-6-astra): a resolvent-optimal finite-horizon direction d* ∝ Aᵀg, with A = Σ_{k≤T} J^k, compared with g and
  the mass-mean direction at matched norms. Population directions must be frozen on training items and evaluated on
  held-out pairs.

## Loop-aware steering: the resolvent-optimal direction d* (`dstar_vjp`, `make_dstar_pushes.py`, `dstar_eval`, `analyze_dstar.py`; design, code and results reviewed by GPT-6 Astra)

Runs:
- `results/dstar_vjp-20261106-061708` (253 s): per item, g = dR/dx_q and a = Aᵀg with A = Σ_{k≤48} J^k, computed by 48
  vector-Jacobian products.
  - The pinned model's `iterate_one_step` is decorated `@torch.no_grad`, so the VJPs call `core_block_forward` directly;
    forward equivalence is asserted.
  - Adjoint check (relative error): median 8.4e-4, max 2.2e-2.
- `results/dseval-20261106-062151` (434 s).

**Design.**
- Directions are frozen on 30 training units and evaluated on 60 held-out statements from 30 disjoint units.
- Every push has norm 12.5 and is applied to the answer-position state input before every loop (persistent forcing).
- Directions compared:
  - u_dstar = normalized mean of normalized a_i (label-free);
  - the readout gradient g;
  - the probe (mass-mean truth) direction;
  - an item-specific d*;
  - random.

Astra's accepted wording:

> A population steering direction constructed from training-item readout gradients and a frozen linearization of the
> recurrent dynamics produced strong bidirectional changes in the True/False log-odds readout on 60 held-out statements
> from 30 disjoint units. Under persistent answer-position forcing of L2 norm 12.5 per step, its median opposite-label
> shift after 48 steps was 8.13 log-odds units. Paired comparisons favored it over both the raw readout-gradient and
> mass-mean directions. This demonstrates control of this readout under one prompt format and intervention regime; it
> does not establish manipulation of an internal truth representation or persistence after forcing ends. True-to-false
> flipping evidence is limited to three sham-correct true statements.

Details:
- **Median opposite-label shift at loop 48** (log-odds):

  | Direction | Shift |
  |---|---|
  | d* (population) | 8.13 |
  | d* (item-specific) | 7.66 |
  | Readout gradient g | 0.93 |
  | Probe (mass-mean) | 0.25 |
  | Random (absolute deviation) | 0.14 |

- **Paired comparisons:**
  - d* − probe: 8.03 [7.85, 8.13].
  - d* − g: 7.13 [7.06, 7.22].
  - Item − population: −0.22 [−0.32, −0.16].
- **By truth stratum (d*):** true → false 9.41; false → true 6.74.
- **False statements flipped to True:** 30/30 (d*), 20/30 (g), 12/30 (probe).
- **Geometry:** cos(d*, g) = 0.096; cos(d*, probe) = 0.025.
- **Caveats** (Astra): the ratios are descriptive ratios of medians; stable True+False mass does not exclude collateral
  effects; absolute adjoint error and the Neumann tail along d* were not saved.
- **Controls still needed:**
  - unrelated True/False tasks and other label tokens (Yes/No, A/B, reversed mappings), to separate a literal answer-token
    lever from semantic truth;
  - dose-response against the predicted g·A·d, single-pulse and removal conditions;
  - a cohort with enough sham-correct true items;
  - other formats and seeds.

### d* controls: answer-token lever vs truth, dose, pulse, removal (`make_ctrl_prompts.py`, `ctrl_eval`, `analyze_ctrl.py`; design, code and results reviewed by GPT-6 Astra)

Runs:
- Formats: `results/ctrl_formats-20261106-064804` hit its deadline after 320/480 condition rows; its done mask kept 20
  complete statements × 4 formats. The completion run `results/ctrl_formats-20261106-070550` (same seeds) covered the
  remaining 10 statements.
- The shard-merge step is a completion-handling amendment added after the truncation was seen. Its validity was verified
  by Astra (keys, axes, configs, code hashes).
- Dose / pulse / removal: `results/ctrl_dose-20261106-070217`.

**Design.**
- Frozen population d* (norm 12.5, every loop) on 30 held-out statements in four formats:
  - original True/False;
  - Yes/No;
  - reversed labels ("Write False if the statement is true…");
  - A/B.
- Plus, in the original format: smaller doses against the linear prediction, a single pulse, and removal after
  24 loops.

Astra's accepted wording:

> On 30 held-out statements in four formats, the frozen population d* increased literal True-versus-False log odds even
> when those tokens were not the requested answers, with positive changes in log p(True). It also shifted Yes/No and A/B
> responses toward their true-designated labels. These results demonstrate transfer across answer vocabularies, but do
> not distinguish truth-related steering from affirmation or label preference, or identify separate token and semantic
> components. The reversed-label format failed baseline qualification, leaving the prespecified token-versus-semantic
> verdict INCONCLUSIVE. Small-dose responses closely followed the local linear prediction, with increasing sublinearity
> at larger doses. Pulse and removal effects decayed toward baseline, providing no evidence of lasting readout changes
> after forcing ended.

Details:
- **Qualification:**
  - The reversed-label format: accuracy 0.23, AUROC 0.03. The model ignores the reversal.
  - The other formats: accuracy 0.57–0.77, AUROC 0.88–0.95, label mass ≥ 0.95.
- **'+' push effects** (log-odds):

  | Format | Literal True − False | Format's own true-designated label |
  |---|---|---|
  | True/False | +6.48 | +6.48 |
  | Yes/No | +4.72 | Yes +2.83 |
  | A/B | +9.20 | A +3.19 |

  A random direction moves these by 0.13–0.25.
- **Dose** (observed / predicted, median): 0.93 at norm 1.56, 0.87 at 3.125, 0.77 at 6.25, 0.67 at 12.5. Spearman
  0.997 → 0.908.
- **Pulse:** 0.78 at t0, 3.76 at t1, 0.69 at t4, about 0 at t48.
- **Removal:** 6.49 at loop 24 (forced) → 3.63, 1.89, 0.34 at loops 25, 26, 28 → −0.06 at 32 → about 0 at 48 (median
  |residual| 1.7e-4).

**Steering thread overall (Astra):**

> Truth labels were linearly decodable, but the tested probe directions gave weak, heterogeneous behavioral control.
> Local response analysis implicated weak immediate readout coupling more than unusually strong recurrent attenuation.
> Incorporating recurrent dynamics into a training-frozen d* produced much stronger held-out True/False readout control
> than probe or raw-gradient directions. Controls establish transfer across answer vocabularies and transient
> forcing-dependent responses; they leave abstract truth manipulation unresolved.

### d* counterbalanced control (`make_ctrl_prompts.py` formats fq/ba, `ctrl_eval` mode `ctrl_cb`, `analyze_ctrl_cb.py`; reviewed by GPT-6 Astra)

Run `results/ctrl_cb-20261106-071755` (269 s).
- **Formats:**
  - fq: "Is this statement false?" (a true statement should get No).
  - ba: "A) false B) true" (a true statement should get B).
- **Setup:** same 30 statements and frozen d*; references are the earlier Yes/No and A/B formats.

Astra's accepted wording:

> Neither counterbalanced format passed baseline qualification: accuracy was 0.50 in both, with AUROC 0.49 for the
> false-question format and 0.52 for reversed A/B labels. The prespecified verdict is therefore INCONCLUSIVE.
> Descriptively, +d* shifted responses toward Yes/A, with effects of similar magnitude and the same sign as the tested
> random direction; −d* shifted responses toward No/B. This pattern is consistent with response-polarity or label
> preference, but the failed baseline qualification prevents these controls from distinguishing truth-related steering
> from response preference. It does not establish absence of internal truth information.

Details:

| Format | +d* | Random | −d* |
|---|---|---|---|
| fq | +0.48 toward Yes | +0.36 | +2.77 toward No |
| ba | +0.26 toward A | +0.18 | +0.87 toward B |

- The asymmetry cautions against treating ±d* as a simple bidirectional truth axis.
- **Possible bounded follow-up (Astra):** screen balanced counterbalanced prompts on separate calibration items without
  steering, freeze a mapping only if baseline discrimination succeeds, then test the frozen d* on untouched items with
  several random controls.

## Is there information in the loop path beyond the endpoint? (`analyze_trajectory_info.py`, offline on existing runs; rules reviewed by GPT-6 Astra before the first run)

Output: `results/trajinfo/summary.json`.

**Design.** Matched-capacity comparisons between endpoint features and endpoint plus earlier-loop features:
- truth from states, at loops 4/8/16 versus 64;
- correctness from readout/drift/KL paths, on the carry-chain, census and settle runs.

Controls: ten time-shuffle seeds and order-free path summaries. Folds are unit-grouped; logistic regression with fixed
C = 1.

Astra's accepted wording:

> Across these datasets, the tested intra-loop trajectory features provided no detectable predictive improvement over
> endpoint features. Truth prediction was already near ceiling; correctness prediction showed no reliable incremental
> benefit. These results do not establish additional information in the trajectory beyond the endpoint. They also do not
> rule it out: correctness analyses used scalar readout/drift paths rather than full states, had limited error counts, and
> held C fixed. The results leave the separately supported operational claim—that dynamics affect accessibility and
> control—unchanged.

Details (AUROC):
- **Truth:** endpoint 0.996–1.000 at both sites and in both datasets; adding an earlier snapshot changes it by ≤ 0.005.
- **Correctness:**

  | Dataset | Endpoint | Endpoint + path | Time-shuffled path (10 seeds) | Increment CI |
  |---|---|---|---|---|
  | Carry chain (n = 150) | 0.600 | 0.519 | 0.43–0.59 | — |
  | Census (n = 200) | 0.743 | 0.762 | 0.68–0.75 | [−0.045, 0.082] |
  | Settle (217 positions, 21 errors) | 0.778 | 0.765 | median 0.775 | — |

- Census beating a single shuffle does not establish an increment beyond the endpoint.
- Settle's order-free summaries scored 0.836, which also does not establish a benefit from temporal order.

## Digit levers: loop-aware control and the full 10×10 response matrix (`dglever` train/pred/eval, `make_dglever_pushes.py`, `analyze_dglever.py`; design, code and results reviewed by GPT-6 Astra)

Runs:
- `results/dglever_train-20261106-080920` (190 s): per training item, gradients of the 10 centered digit logits and
  Aᵀg via VJPs.
- `results/dglever_pred-20261106-081508` (355 s): A·P for the 10 loop-aware levers, on eval and train items.
- `results/dglever_eval-20261106-082140` (290 s): persistent forcing, 33 conditions × 20 held-out items.

**Design.**
- Training and held-out items each have one prompt per context per answer digit, all from distinct cells.
- Push families, frozen on training items, all at norm 3:
  - loop-aware levers L_k ∝ mean normalized Aᵀ∇z_k;
  - immediate readout gradients G_k;
  - representation centroids C_k;
  - random directions.
- Gates passed: Neumann tails about 3e-6; adjoint relative error 2e-4; sham drift 5e-6.

Astra's accepted wording:

> On 20 held-out prompts from distinct cells, training-frozen loop-aware digit levers applied at norm 3 before each of 48
> recurrent updates increased target centered digit logits by 1.78 on average, versus approximately −0.046 for
> matched-norm immediate-gradient and centroid directions. For these levers, the local recurrent-response model predicted
> the 10×10 response matrices with median correlation 0.978, slope 0.95, and normalized error 0.22. Shared or swapped
> operators retained substantial predictive power but performed reliably worse; immediate-readout predictions were
> anticorrelated with observed responses. Off-target effects showed a modest positive association with circular
> proximity, without a monotonic distance profile. These findings establish held-out control and quantitative response
> prediction for the tested intervention family.

Details:
- **Target effect, L minus G:** [1.64, 2.03]. **L minus C:** [1.63, 2.04]. Random median |Δz| is 0.046.
- **Prediction of the 10×10 response matrix:**

  | Operator | Median correlation | Normalized error |
  |---|---|---|
  | Own (FULL) | 0.978 | 0.22 |
  | Averaged over training | 0.947 | 0.37 |
  | Swapped (other item) | 0.938 | 0.38 |
  | Swapped (same answer digit) | 0.931 | 0.39 |
  | Identity (readout only) | −0.587 | 1.11 |

  - FULL minus each control: SWAPPED [0.030, 0.063]; SWAPPED_SAME_DIGIT [0.034, 0.056]; AVERAGED [0.022, 0.040].
  - Excluding the target diagonal, FULL still reaches r = 0.972.
- **Off-target structure:**
  - Circular-proximity Spearman: L +0.28, G −0.11, C −0.24.
  - L's mean effect at circular distance 1–5: −0.09, +0.38, −0.74, +0.01, −0.88 (non-monotonic).
- **Caveats** (Astra):
  - The outcome is centered digit logits, not accuracy.
  - The levers were built from the operator family, so this is direction-conditioned prediction (yet genuinely held-out).
  - The negative identity prediction holds for these selected levers; it is not a general "recurrence reverses pushes".
  - A single dose was tested; endpoint displacement is about 17%.

## Mechanism of the readout reversal (`intervene_app.py revmech`, `analyze_revmech.py`; design, code and results reviewed by GPT-6 Astra)

Run `results/revmech-20261106-204036`.
- **Setup:** 20 held-out digit prompts × 21 pushes (levers L0–L9, readout-gradient directions G0–G9, random R0).
- **Method:** Arnoldi on J_qq started from each push (m = 50). The exact impulse response of all 10 digit readouts,
  c_t = g·J^t P, is computed and split into Ritz modes.
- **Primary test:** the full-matrix reversal of the levers, with the counterfactual S_noflip (each mode keeps its
  immediate sign, weighted only by the size of its finite-horizon gain).

Astra's accepted wording:

> All implemented numerical gates pass, and the prespecified classification is REWEIGHTING. Removing modal gain
> signs/phases substantially attenuates the reversal, with an uncertain residual anticorrelation; this counterfactual
> does not identify a causal fraction. Modal cancellation and conditioning limit confidence in individual-mode
> interpretations. The clearest basis-independent finding is rapid target-readout reorientation: 154/157 accumulated L
> target reversals already change sign after one application of the local Jacobian, while G target responses are
> strongly suppressed after one step. These findings do not by themselves establish non-normality or exclude later
> dynamical contributions.

Details:
- **Gates:**
  - Versus the independent prediction run: median relative error 6.9e-4.
  - Modal reconstruction error about 5e-14; orthogonality 4e-15; FD step sensitivity 0.2%.
  - Eigenvector conditioning cond(W): median 75 (90th percentile 216).
  - Cancellation ratios: 19.5 (S) and 35.9 (c0).
- **Primary:**
  - corr(c0, S) = −0.591 [−0.655, −0.525]; corr(S, observed) = 0.978.
  - corr(c0, S_noflip) = −0.080 [−0.265, +0.047].
  - At Krylov size 40: −0.591 / −0.168.
- **Levers (L), target readout:**
  - Median c0 −0.134, c1 +0.936; c1/c0 ≈ −4.
  - Mode-class contributions to S: forward 1.57, rotating 0.23, overshoot 0.01.
- **Readout gradients (G), target readout:**
  - Median |c1/c0| = 0.051, while median ‖JP‖/‖P‖ = 1.17: the target readout is suppressed without overall
    contraction.
  - Mode-class contributions to S: forward +0.15, rotating −0.16, overshoot −0.02.
- **Not reported:** the prespecified dominant-pair Ritz residuals.
- **Next (Astra) — done**, see [Local readout transfer operator](#local-readout-transfer-operator-intervene_apppy-rtransfer-analyze_rtransferpy-design-code-and-results-reviewed-by-gpt-6-astra): characterize the local readout transfer operator QᵀJ (Q = orthonormalized digit-gradient span),
  decomposing JᵀQ = QA + B⊥, to separate within-readout transformation from coupling through outside directions.

## Local readout transfer operator (`intervene_app.py rtransfer`, `analyze_rtransfer.py`; design, code and results reviewed by GPT-6 Astra)

Run `results/rtransfer-20261106-205118` (45 s).
- **Setup:** the same 20 held-out digit prompts, at x* after 96 loops.
- **Method:**
  - Q is the orthonormal span of the 10 centered digit-logit gradients (rank 9).
  - B = QᵀJ is formed from VJPs, and JᵀQ is split into QA_in (within-readout) plus B⊥ (input from outside the span).
  - The frozen dglever pushes (L, G, C) are projected onto this structure.
- **Gate:** adjoint consistency between the FD JVPs and the VJPs.
- **Revision (after the run, before interpretation, flagged by Astra):**
  - The original 1e-6 relative rank cut let float noise through on 2 items (10th singular value 1.1e-6), and both
    failed the adjoint check (errors 76 and 66).
  - With the cut at 1e-4, every item has rank 9 and passes on its own (median 6.2e-4, max 9.1e-4).
  - Headline numbers are unchanged.

Astra's accepted wording (its numerical caveat is now resolved by the revision above):

> At settled states on held-out digit prompts, the local one-step transfer into the digit-gradient span draws
> predominantly on perturbations outside that span: approximately 97% of its squared Frobenius sensitivity lies outside
> it. Within-span transfer is weak. Loop-aware levers overlap more strongly with readout-sensitive input directions than
> immediate-gradient or centroid controls, and their target response changes from negative immediately to positive
> after one step, while immediate-gradient target responses are strongly suppressed. Together with the earlier
> impulse-response results, this supports local readout reorientation as an explanation for these tested
> interventions. It does not establish activation replenishment, globally orthogonal subspaces, or a mechanism for
> truth steering.

Details (medians over items, item-bootstrap CIs):
- **R1:** rho = ‖A_in‖/‖JᵀQ‖ = 0.171 [0.168, 0.176]; outside norm share 0.985. Prespecified FED FROM OUTSIDE.
- **R3 (within-readout map QᵀJQ):** spectral radius 0.17; top singular value 0.23; median eigenvalue real part 0.09;
  22% of eigenvalues have a negative real part.
- **R2 (pushes, norm 3):**

  | Family | In-span fraction | One-step readout gain | Top-9 input-direction energy | Target c0 → c1 |
  |---|---|---|---|---|
  | L (levers) | 0.12 | 0.336 | 0.185 | −0.135 → +0.945 |
  | G (readout gradients) | 0.71 | 0.105 | 0.015 | 2.18 → −0.002 |
  | C (centroids) | 0.16 | 0.092 | 0.015 | 0.18 → −0.016 |

  - Paired L − G: gain [0.203, 0.234]; in-span [−0.632, −0.547]; top-V [0.142, 0.188].
- **Wording limits (Astra):**
  - "Feeder" is shorthand for local Jacobian sensitivity, not measured activation flow.
  - Only 18.5% of L's energy lies in the top input directions, so L is enriched there, not concentrated there.
  - rho is a singular-value-weighted overlap, not a principal-angle test.
  - The sign reversal itself is shown by the signed c0/c1 values, not by the coupling geometry.
  - The truth-steering conclusion (weak immediate coupling) stands and is not explained by this digit analysis.

## Which part of the block implements the one-step reorientation (`intervene_app.py blockdecomp`, `analyze_blockdecomp.py`; design, code and results reviewed by GPT-6 Astra)

Run `results/blockdecomp-20261106-220105` (60 s).
- **Setup:** the same 20 held-out digit prompts at x* (96 loops); the frozen pushes (L, G, C at norm 3; R0, R1).
- **Split:** the one-step target response c1 = g_k·JP is split exactly (linearized) into 9 writes at the answer
  position:
  - the adapter output (state → hidden), carried to the block output through residual/norm derivatives only;
  - the attention and MLP output of each of the 4 core layers.
- **Attention routes:** query path (the answer position re-weights its context) versus its own key/value.
- **Query path by source token:** demo lines, symbols, same-column operand digits, other operand digits, self.
- **Pre-run revisions:** Astra's pre-run changes (gate-suppressed labels, denominator-adequacy rule for signed shares,
  per-item gates) were applied before the full run.

Astra's accepted wording:

> Across 20 held-out prompts, all numerical validation gates passed. The lever-induced one-step response, measured
> using the frozen coda readout gradient, was attributed mainly to the adapter contribution through residual/norm
> derivatives (median signed share 52.9%, 95% item-bootstrap CI 47.7–55.8%) and MLP writes (39.9%, 37.6–43.7%);
> attention writes contributed 7.4% (5.1–8.4%). No individual group demonstrated a majority under the prespecified rule.
> The adapter contribution was positive in aggregate despite a negative immediate response. These are local derivative
> accounting results, not ablation effects or evidence that context is unnecessary.
>
> For gradient pushes, gross target-readout contributions were small relative to the immediate response (median ratio
> 0.202, CI 0.168–0.214), satisfying the prespecified "NOT TRANSMITTED" criterion. Opposite-signed adapter and MLP
> contributions additionally indicate partial cancellation. Normalized group shares were undefined because only 5/20
> items met denominator adequacy.

Details:
- **Gates:**
  - Sum of the 9 writes versus FD response: 2.5e-5.
  - FD versus VJP: 5.0e-4 (worst item 2.3e-3).
  - Q + KV split: 1.3e-3; categorized source sum: 5.9e-4.
  - Replica errors: 0; non-finite terms: 0.
- **Levers (L):**
  - c0 −0.134 → c1 +0.936; median adapter contribution +0.486.
  - Per-write shares: adapter 0.53, mlp4 0.17, mlp1 0.09, mlp3 0.08, mlp2 0.07; attention layers 0.01–0.03 each.
  - Attention:
    - The query path carries 0.96 of the attention contribution.
    - Query-path sources (signed shares of the query path; the per-position split depends on the centering
      convention): demo lines 0.52, same-column operand digits 0.26, symbols 0.11, self 0.08, other operand digits
      0.07.
- **G pushes:**
  - c0 2.27 → c1 0.011; |net|/gross 0.30 [0.24, 0.33].
  - Signed per-item numerators: adapter +1.60, MLP −1.14, attention −0.21. These are separate medians, not an
    additive balance.
- **Caveats (Astra):**
  - The adapter path includes state-dependent normalization derivatives, so it is not "near-linear".
  - The positive adapter versus negative c0 is an aggregate sign contrast, not a per-item flip.
  - MLP write tangents include effects propagated through earlier attention, and the state already contains context.
    Attention's small direct share does not show it is causally dispensable.
- **Next (Astra) — done**, see [Cheap levers from the adapter and one-step pullbacks](#cheap-levers-from-the-adapter-and-one-step-pullbacks-intervene_apppy-adaptcov--dglever-mode-eval-adlever_eval-analyze_adaptcovpy-make_adlever_pushespy-analyze_adleverpy-design-code-and-results-reviewed-by-gpt-6-astra): test whether the pulled-back readout a_k(x) = W_sᵀ R_adapter(x)ᵀ g_k(x) is stable across states. A
  direction estimated on training states would then predict lever responses on held-out states, compared with
  W_sᵀ g_k(x). This separates a stable adapter-map explanation from dependence on normalization geometry.

## Cheap levers from the adapter and one-step pullbacks (`intervene_app.py adaptcov` + `dglever` mode eval `adlever_eval`; `analyze_adaptcov.py`, `make_adlever_pushes.py`, `analyze_adlever.py`; design, code and results reviewed by GPT-6 Astra)

Runs `results/adaptcov-20261106-224906` (stage 1, 60 s) and `results/adlever_eval-20261106-225059` (stage 2, 351 s).
- **Stage 1:** at x* on 20 TRAIN and 20 held-out EVAL digit prompts, compute four covectors:
  - the readout gradient g;
  - the adapter pullback a = W_sᵀR_adapterᵀg;
  - the adapter-weight pullback w = W_sᵀg;
  - the one-step pullback b = Jᵀg.
- **Stage 2:** fixed population levers built from TRAIN items, with the same construction as the L levers and norm 3:
  - A, W and B, alongside the reference L (the 48-step resolvent);
  - each applied as persistent forcing for 48 loops on the held-out prompts.
- **Pre-run revisions:** Astra's pre-run changes were applied (means as in P1; stage-1 gate before building pushes;
  push hash; cell-level L/sham replication; L adequacy gate; H1 primary, H2/H3 secondary).

Astra's accepted wording:

> On the held-out evaluation, the fixed adapter-transpose direction achieved a mean target effect of 0.874,
> approximately 49% of the 48-step reference effect (1.78). The full adapter-path direction performed similarly (0.875);
> its preregistered primary half-reference test was inconclusive. The one-step pullback achieved 1.35, approximately 76%
> of the reference, and passed the secondary half-reference test. These findings identify promising inexpensive
> steering directions in this setting. They do not yet establish adapter-specific causality, dose robustness, or
> transfer to other objectives. The adapter matrix is fixed, but its steering direction depends on the state-dependent
> readout gradient.

Details:
- **Gates:**
  - Stage 1 cross-run identities: a·P versus blockdecomp's adapter term 2.4e-4 (worst item 3.8e-4); b·P versus its VJP
    6e-7.
  - Stage 2: L and sham replicate the original eval exactly (max difference 0.0); L mean 1.78 [1.59, 1.99].
- **Recorded before stage-2 data:** cos(a, w) median 0.9987 (min 0.9937), with |a|/|w| between 2.72 and 2.86. The
  residual/norm chain therefore acts on these covectors as approximately a scalar, so A ≈ W and H2 is uninformative
  (A − W = 0.001 [−0.001, 0.002]).
- **Held-out 48-loop target effects (means):**

  | Direction | Effect | Fraction of L |
  |---|---|---|
  | g (readout gradient, original run) | −0.046 | — |
  | A (adapter pullback) | 0.875 [0.79, 0.97] | 0.49 |
  | W (adapter weights) | 0.874 | 0.49 |
  | B (one-step pullback) | 1.35 [1.20, 1.52] | 0.76 |
  | L (48-step resolvent) | 1.78 | 1.00 |

  - H1 (A − 0.5L): −0.013 [−0.076, 0.051], INCONCLUSIVE.
  - H3 (B − 0.5L): 0.47 [0.40, 0.54], B RETAINS ≥ HALF OF L.
  - Endpoint displacement: L 0.17, B 0.12, A 0.07. Final-step drift is at the sham level.
- **Stage 1 descriptive:**
  - Held-out stability (item-mean cosine CIs): g 0.57–0.63, a 0.54–0.60, b 0.45–0.51.
  - Train LOO: g 0.60, a 0.55, b 0.45, resolvent 0.28.
  - Cosine with L: g −0.04, a 0.40, b 0.66.
- **Caveats (Astra):**
  - This is an observed efficacy ordering, not a "lookahead hierarchy": the adapter transpose is a different
    transformation, not a shallower lookahead.
  - An L-overlap-matched control is missing. A naive linear expectation from overlap alone (cos × L effect) is
    0.40 × 1.78 ≈ 0.71 for A and 0.66 × 1.78 ≈ 1.18 for B, so the effects may largely reflect alignment with L rather
    than a special adapter property.
  - Single dose; cost claims need measured end-to-end compute.
- **Next (Astra) — done**, see [Cheap levers on the truth task](#cheap-levers-on-the-truth-task-intervene_apppy-truthcov--dstar_eval-config-tlever_eval-make_tlever_pushespy-analyze_tleverpy-design-code-and-results-reviewed-by-gpt-6-astra): a preregistered held-out truth-task transfer comparing W, B and L across a small signed dose
  sweep, measuring truth performance, collateral degradation and compute cost.

## Overlap-matched control for the cheap levers (`make_overlap_pushes.py`, `dglever` eval config `ovl_eval`, `analyze_overlap.py`; design, code and results reviewed by GPT-6 Astra)

Runs `results/ovl_eval-20261106-232326` and `results/ovl_eval-20261106-233019` (about 270 s each); reference
`results/adlever_eval-20261106-225059`.
- **Pushes:**
  - For A (adapter path) and B (one-step), control directions have the same cosine with L_k and the same norm (3), with
    a random remainder orthogonal to L_k. There are two independent draws per (family, k).
  - Fperp is F_k's own remainder after its L_k component is removed.
- **Pre-run revisions:** Astra's changes were applied (two draws; push hashes recorded in each run and checked; cross-run
  item/config/revision/sham gates; label wording).

Astra's accepted wording:

> At norm 3 under 48-loop persistent forcing, controls matched to each lever's overlap with the fixed reference
> direction L_k reproduced approximately 85% of A's and 90% of B's mean steering effect. Both levers exceeded their
> matched controls: A by 0.134 logits (97.5% CI 0.070–0.200), B by 0.142 (0.088–0.200). These item-bootstrap intervals
> are conditional on the sampled control directions. Components orthogonal to the corresponding L_k also steered
> independently; a linear additivity calculation approximated the full effects, descriptively. Together with the
> adapter transpose's reorientation of the readout covector toward L, these results support alignment as a substantial
> contributor, with additional steering associated with the levers' orthogonal components. They do not establish
> adapter-specific causality or a unique decomposition of the mechanism.

Details:
- **Gates:** all pass (sham difference 0.0; push hashes; identical items, config and revision).
- **A (cosine with L 0.40):**
  - Observed 0.875; matched controls 0.741 and 0.742.
  - Linear overlap prediction 0.713.
  - Aperp alone 0.146 [0.084, 0.211].
  - Additivity calculation 0.847.
- **B (cosine with L 0.66):**
  - Observed 1.354; matched controls 1.208 and 1.215.
  - Linear overlap prediction 1.183.
  - Bperp alone 0.212 [0.150, 0.278].
  - Additivity calculation 1.343.
- **Caveats (Astra):**
  - "Non-L" means orthogonal to the corresponding L_k only.
  - The roughly 0.03 gap between control and linear prediction is consistent with nonlinearity, though finite
    random-direction alignment can also contribute.
  - "Reorients" rather than "rotates": W_sᵀ need not be orthogonal.
- **Status (Astra):** a natural close for the scoped steering/reversal investigation.

## Cheap levers on the truth task (`intervene_app.py truthcov` + `dstar_eval` config `tlever_eval`; `make_tlever_pushes.py`, `analyze_tlever.py`; design, code and results reviewed by GPT-6 Astra)

Runs:
- `results/truthcov-20261106-234613` (stage 1, 115 s);
- `results/tlever_eval-20261106-235208`, `-20261106-235648` and `-20261107-000141` (about 260 s each).

Setup:
- **Directions:** covectors of R = log p(True) − log p(False) at x* (loop 64). L is the existing 48-step resolvent d*;
  B = Jᵀg (one VJP); W = W_sᵀg; G = g.
  - Population directions come from the 60 TRAIN statements; item-specific L_item and B_item are label-free.
  - Persistent forcing for 24 loops on 60 held-out statements (30 units, cities + compare).
- **Doses:** 1.5625, 3.125, 12.5.
- **Signs:** toward the opposite label (runs 1–2) and toward the own label (run 3, population L/B/W).
- **Pre-run revisions:** Astra's changes were applied (L adequacy gate, paired-contrast wording, both signs,
  corrections/corruptions, provenance gates including itemwise L@12.5 replication, trajectories).
- **Post-run fix:** the H2 label string was fixed after the results review (it was hardcoded to "B"; the value was
  unaffected).

Astra's accepted wording:

> On 60 held-out statements spanning 30 units and two datasets, a population direction constructed using one VJP
> supported persistent True/False readout control. The prespecified primary paired contrast, median(E_B − 0.5E_L), was
> positive: 0.248, with a 95% unit-cluster bootstrap interval of [0.247, 0.249]; the L reference passed its adequacy
> gate. Intervention signs used ground-truth labels. These results establish controllability of the token readout in
> this prompt format; they do not establish improved truth inference or changed internal knowledge. Construction
> timings exclude shared warm-up, estimate L from 48 VJPs, and amortize population directions across evaluation items.
> Collateral measures describe on-prompt output-distribution changes.

Details:
- **Gates:** all pass. Sham differences across runs and versus the old dseval: 0.0; L@12.5 replication: 0.0; adjoint
  median 8e-4; sham off-TF KL 1e-8.
- **Population cosines with L:** B 0.70, W 0.40, G 0.10 (digits: 0.66 / 0.40 / −0.04).
- **E at 24, toward the opposite label (medians):**

  | Direction | Dose 1.5625 | Dose 3.125 | Dose 12.5 | Ratio to L at 1.5625 | Off-TF KL at 1.5625 / 3.125 / 12.5 |
  |---|---|---|---|---|---|
  | L_pop (48-step) | 1.24 | 2.49 | 8.12 | 1.00 | 0.060 / 0.225 / 1.69 |
  | B_pop (one VJP) | 0.85 | 1.71 | 6.67 | 0.68 | 0.025 / 0.100 / 1.19 |
  | W_pop (adapter weights) | 0.49 | 0.97 | 3.89 | 0.39 | 0.005 / 0.019 / 0.29 |
  | G_pop (readout gradient) | 0.12 | 0.23 | 0.93 | 0.09 | 0.0007 / 0.003 / 0.046 |
  | L_item | 1.30 | 2.45 | 7.66 | — | — |
  | B_item | 0.88 | 1.76 | 6.41 | — | — |

  - Random at 12.5: 0.04 [−0.04, 0.07], KL 0.015.
  - E is flat across loops 8, 16 and 24, and roughly linear up to 3.125; ratios compress at 12.5.
  - Own-label magnitudes are similar.
  - True+False mass changes (signed medians) are ≤ 0.02.
- **Hypotheses:**
  - H1 (primary): B_pop − 0.5 L_pop = 0.248 [0.247, 0.249], POSITIVE.
  - H2 (exploratory): W_pop: −0.136, NEGATIVE.
  - H3 (exploratory): B_item vs L_item: 0.251, POSITIVE.
- **Accuracy (label-informed sign):**
  - The sham is False-biased: 0.55 accuracy, with all 27 errors being true statements judged False.
  - Own-label pushes by L, B and W correct 27/27 at every dose, including W@1.5625.
  - Opposite pushes corrupt progressively.
  - This is readout control with an oracle sign, not truth inference.
- **Collateral per effect:** median KL / median E² at 1.5625 is L 0.039, B 0.035, W 0.020, G 0.049. This is descriptive
  and on-prompt only. Matched-effect collateral was not tested directly, so dose compensation is suggested, not shown.
- **Compute per item** (batch 40, cuda-synchronized):
  - Shared 64-loop warm-up: 0.75 s.
  - g: 0.0125 s; W: 0.0125 s; B: 0.037 s.
  - L: estimated 1.19 s (48 VJPs; historical dstar_vjp wall time 252.7 s for 120 items including model load).
  - Population directions are built once.
- **Next (Astra) — partly done**: run retrospectively on the same statements in [Label-free sign selection: does steering add accuracy beyond the monitor?](#label-free-sign-selection-does-steering-add-accuracy-beyond-the-monitor-analyze_selectorpy-offline-exact-design-and-results-reviewed-by-gpt-6-astra); fresh units and prompt formats remain **open**. Original note: a frozen, label-free sign selector (e.g., a probe) on fresh units and prompt formats, with
  selector-only prediction and readout-bias calibration as baselines. Steering must add value beyond simply reporting
  the selector's answer.

## Label-free sign selection: does steering add accuracy beyond the monitor? (`analyze_selector.py`; offline, exact; design and results reviewed by GPT-6 Astra)

- **Retrospective diagnostic:** no new model runs. The tlever runs pushed every held-out statement with population L,
  B and W at three doses in both directions, so any sign policy's steered outcome is exactly the recorded R of the
  matching condition.
- **Selector:** a frozen pooled mass-mean truth probe on the answer-position state at loop 64 (stage-1 `stBa`; the same
  x*, with exact readout equality). It is trained on the 180 statements of the other 90 units.
- **Baselines:**
  - the model's readout;
  - readout-bias calibration;
  - the selector alone;
  - selector-signed bias;
  - a cross-fitted logistic combination of readout and probe score.
- **Inference:** a unit-level sign-flip randomization test with Holm correction. Bootstrap CIs are descriptive.
- **Revisions:** Astra's pre-analysis revisions were applied before computing (cross-fitting, permutation test, new
  baseline, gates, sensitivity check).

Astra's accepted wording:

> In a retrospective diagnostic on 60 statements from 30 held-out units, all nine selector-directed steering policies
> exactly reproduced the selector's binary predictions, yielding 76.7% accuracy. Thus, at the tested doses and readout
> position, steering provided no answer-accuracy benefit over reporting the monitor's verdict. A training-fitted
> logistic combination of model readout and cross-fitted probe scores achieved 90.0% accuracy (descriptive
> unit-bootstrap interval: 83.3–96.7%). This exploratory advantage may reflect probe recalibration, complementary
> readout information, or both. Steering incurred dose-dependent off-target distribution changes. These findings concern
> one prompt format and binary answer readout; they do not establish information loss in the latent state or effects on
> downstream generation.

**Post-hoc resolution (requested by Astra; not prespecified):** a training-fitted, probe-score-only logistic
(cross-fitted scores, `selector_posthoc_probe_only.json`) also reaches 0.900 (cities 1.00, compare 0.80). The combined
policy's advantage is therefore attributable to probe recalibration. The model's own readout added no detected
information here.

Details:
- **Gates:** all differences 0.0. Using R64 instead of R@24 changes 0 predictions for every baseline.
- **Selector:** train accuracy 0.79 (AUROC 0.95); eval accuracy 0.767 (AUROC 0.956); cities 1.00, compare 0.53.
- **Accuracy on 60 held-out statements:**

  | Policy | Accuracy |
  |---|---|
  | Model | 0.55 |
  | Calibrated readout (τ −0.33) | 0.783 |
  | Selector | 0.767 |
  | Selector-signed bias (a* 0.5) | 0.767 |
  | Combined | 0.900 |
  | Probe-only recalibrated (post-hoc) | 0.900 |
  | Every steering policy (L/B/W × 3 doses) | 0.767, with 100% agreement with the selector |

- **Hypotheses:**
  - H1 (versus selector) and H3 (versus selector-bias): NO ADDED ACCURACY DETECTED; no unit has a nonzero net
    difference.
  - H2 is exploratory. It tests steering superiority, so p = 1 does not test the combined policy's superiority. The gap
    is −0.133, with 8 units showing a nonzero net difference.
- **Collateral (off-TF KL median):** W@1.5625 0.005 up to L@12.5 1.14.

## Free-loop null model, read-space control and a second looped model (`intervene_app.py freeprobe`, `ctrlprobe_app.py`, `ouro_app.py`, `weights_app.py`; `analyze_freeprobe.py`, `analyze_ctrlprobe.py`, `analyze_ouro.py`, `analyze_readspace.py`, `analyze_chains.py`; design, code and results reviewed by GPT-6 Astra)

Write-up: `docs/freeloop.pdf` (Draft 3). All numbers are checked by `docs/freeloop_check.py` (ALL CHECKS PASSED,
2026-11-07). The criteria were fixed before each production run unless marked exploratory.

- **Free-loop null model** (`freeprobe`, 16 Gaussian probes and the readouts per context, 48-step pullback chains at
  the fixed point, fp32).
  - Runs: truth `freeprobe-20261107-130520` (40 contexts) and digits `freeprobe_digits-20261107-133750` (20 contexts),
    plus smoke runs.
  - The thresholds were calibrated on synthetic free loops (`synth_freeprobe.py`, `calibrate_freeprobe.py`, output in
    `results/synth_calibration/`).
  - Verdict: NOT SUPPORTED in both contexts.
  - Readout suppression: every readout lies below all 4 probes of its context (240 of 240, Z ≤ −7.9).
- **Read-space control** (`ctrlprobe_app.py`). Families:
  - C1: random-token contrasts.
  - C2: contrasts among the top-64 tokens.
  - C3: gradients of random centred logit combinations.
  - Runs: `ctrlprobe-20261107-144949` (truth) and `ctrlprobe_digits-20261107-144951` (digits), plus smoke runs.
  - Reading: SHARED in both tasks. The C3 excess is −0.224 / −0.200 (Z −28.0 / −24.3), and all 1,440 controls lie below
    all probes.
  - Exploratory (`analyze_readspace.py`): read directions have pullback norms like the probes', but self-cosines
    ≈ 0.03 against 0.38.
- **Ouro-1.4B** (`ouro_app.py`, same 40 truth prompts, 4 loops, includes C3flip).
  - Run: `ouro-20261107-150835` (plus `ouro_smoke-20261107-150630`).
  - PASS; NOT DETECTED. The C3 excess is +0.022 [0.016, 0.025], τ is 0.06, and the coordinate control reads
    COORDINATE-WISE (C3 − C3flip is −0.0004 [−0.006, 0.008]).
- **Weights proxy** (`weights_app.py`, CPU only): run `weights-20261107-145443`. The adapter state block does not
  reproduce the deficit: τ(W_s) is 0.110, and unembedding directions give 0.101.
- **Chain test** (`analyze_chains.py`, exploratory, existing arrays only). Exact start invariance at the fixed point
  makes the readout chains vanish; it implies no ordering relative to probes before that.
  - Result: normalized readout chains exceed the context's median probe chain at every t ≤ 32 (median ratio 1.11–1.33
    truth, 1.22–1.64 digits).
  - Reading: no support for preferential readout attenuation. Approximate start invariance is not ruled out.
