# Local recurrent response predicts and enables readout control in a looped language model

*An empirical study of Huginn-0125 (pinned revision `bb6621b…`). Working write-up, 2026-11-07. Revised after
independent reviews by GPT-6 Astra (two rounds), Gemini 4 Argon (round 1), and a numbers-level fact-check against the
raw saved outputs (Claude Opus 5.5).*

The study combines prospectively specified experiments with explicitly identified exploratory, post-hoc and
retrospective analyses. Designs, analyses and result wording underwent model-assisted review (GPT-6 Astra throughout).
`README.md` records each experiment's timing, gates, amendments and accepted wording; run directories are under
`results/`.

## Summary

Huginn-0125 is a recurrent-depth ("looped") transformer: a prelude embeds the prompt, one core block is applied
repeatedly to a latent state, and a coda reads out tokens. We studied how concepts are represented in the loop, and how
they can be read and controlled.

**Main contribution: control.**
- In these tasks, directions that *decode* a label (linear probes) give weak behavioral control.
- A local linear-response model of the recurrence predicts small-dose steering responses.
- Pulling the readout gradient back through the recurrent dynamics gives much stronger, forcing-dependent control on
  held-out items:
  - True/False log-odds shift 8.1, versus 0.93 for the readout gradient and 0.25 for the probe direction;
  - +1.78 on target digits, versus ≈ −0.05 for controls;
  - a quantitatively predicted 10×10 digit response matrix (median r = 0.978).
- On digits, the response reorients within a single loop update, and the readout span is fed mostly from outside
  itself. Inexpensive one-step pullbacks retain much of the effect: 76% on digits, 68% on truth at low dose.
- In a retrospective test where a probe chose the steering sign, steering added no answer accuracy beyond the probe's
  own verdict.

**Second contribution: dynamics.**
- Late answer-position states are locally attracting and near-stationary in the tested settings.
- A minority of prompts (about 13% of fresh addition prompts) settle into period-2 cycles. These originate at a source
  token, usually upstream of the answer, whose validated own-block eigenvalue lies past −1 (−1.007 to −1.226 across 26
  census sources). That is consistent with a period-doubling (flip) bifurcation along a contextual-input
  family.
- The cycles can be suppressed by eigenvalue-guided under-relaxation.

**Context.**
- Digits (and, in exploratory or partial analyses, weekdays and months) have cyclic representations.
- In the tested formats, extra recurrence did not reliably extend dependency handling.
- No predictive information in the loop path beyond the endpoint was detected.

## 1. Setting and methods

**Model.** Huginn-0125 with test-time noise off and a seeded random initial state. All mechanistic work is in fp32; the
initial addition pilot (and the pilot figures marked as such below) used bf16. The manual loop is validated against
`model.forward`.

**Notation.**
- **Update:** s_{t+1} = F(s_t; E), where F is the core block applied to the adapter output Linear([s, E]) = W_s s + W_e
  E (+ bias). E is the prelude output, re-injected every loop.
- **Fixed point:** x* = F(x*; E) at the answer position q (the last token).
- **Own-position Jacobian:** J = ∂F_q/∂s_q at x*. It is exact for perturbations at q because q is last and attention is
  causal; earlier positions, including their keys and values, are unaffected.
- **Readout:** R(s_q) (a centered digit logit, or log p(True) − log p(False)), with gradient g = ∇R(x*).
- **Persistent forcing** s_{t+1} = F(s_t + P; E) gives, to first order:
  - δs_T ≈ A_T P, with A_T = Σ_{t=1}^{T} Jᵗ (no identity term; the long-time limit is (I − J)⁻¹ J P);
  - readout change ΔR_T ≈ gᵀA_T P = Σ_{t=1}^{T} c_t, with c_t = gᵀJᵗP;
  - immediate (no-loop) readout c₀ = gᵀP.
- **Loop-aware lever:** the unit push maximizing ΔR_T is a_T ∝ A_Tᵀg. The one-step truncation is Jᵀg.
- **Population directions** are built from training items as normalize(mean_i normalize(a_i)), each item's covector
  normalized before averaging, and frozen.
- **Context equivalence:** a final-token context edit with W_e δE = W_s P is equivalent to adding P to that token's
  state input before every loop. W_e δE = W_s P was solved by least squares (residual ≈ 2e-4 fp32, 1e-12 fp64); later
  experiments add P directly.

**Tools.**
- Jv by central finite differences; Jᵀw by autograd through `core_block_forward`. The pinned `iterate_one_step` is
  decorated `@torch.no_grad`, so VJPs call `core_block_forward` directly after an equivalence check.
- Arnoldi spectra and impulse responses.
- Exact linearized decompositions of one block step.

**Tasks.**
- Equal-digit addition with few-shot demos, studied at teacher-forced answer positions.
- Carry-conditioned units prediction.
- Weekday/month successor pilots.
- ARC-Challenge multiple choice (stability only).
- True/False judgments: CITIES (city–country) and COMPARE (two-digit comparisons), with held-out units.

**Validation practice.**
- Prespecified hypotheses, labels and numerical gates, for the confirmatory experiments.
- Synthetic-world checks of analyses where relevant, and smoke runs.
- Model review of design and code before runs, with changes applied before data. Documented exceptions (post-run
  amendments) are recorded in the README: the rtransfer rank cut, the ctrl shard merge, the tlever label fix, and the
  post-hoc selector baseline.
- Result wording matched to the evidence; negative and INCONCLUSIVE verdicts kept as prespecified.
- **Gates** include adjoint consistency (VJP vs FD, typically ≤ 1e-3), exact sham/replication equality across runs,
  replica-versus-model equality, and Neumann-tail checks.
- The early addition, circle and orbit studies were more iterative than the later control experiments. Exploratory and
  post-hoc items are labeled below.

## 2. Representations and dynamics

**2.1 Cyclic codes.**
- In late answer-position states, two sinusoidal digit features (cos/sin 2πd/10) predict held-out states about as well
  as a categorical lookup: 105% of its gain in the pilot (bf16, 64-d sketch), 108% in full fp32 space. A line code
  reaches 32% in the pilot (95% CI 24–41%) and 29% in full space.
- Numerical cyclic order beat all sampled scrambled orders.
- Pushing along fitted circle chords causally shifts the *immediate* digit readout toward the intended neighbor (+2.3
  log-odds; random directions −0.06 to +0.37; radial and top-PC pushes −0.28 and −0.17). Under continued recurrence the
  push decays, and the final answer equals baseline on 168/168 prompts.
- Local readout gradients align with ±1 neighbors. Far-offset sensitivity is dominated by lowering the current digit,
  and less than 1% of readout-gradient energy lies in the circle plane.
- Exploratory/partial evidence of cyclic structure:
  - weekdays: the supplied start day; canonical order ranked 1/360; post-hoc;
  - months: leave-one-month-out R² 0.22, beating 5,000 reorderings; path verdict INCONCLUSIVE.
- Late local dynamics contract the fitted plane without systematic rotation. The separate test of start-anchored
  computational rotation was INCONCLUSIVE; the motion was mostly additive radial growth. No complete circular decoder is
  claimed.

**2.2 Locally attracting, near-stationary states.**
- At the tested answer positions, late states are near-stationary in fp32 (per-step drift ~5e-6). The largest observed
  Ritz modulus has median 0.78 (p10–p90 0.75–0.82).
- A separate token-stratified sample gave leading moduli 0.66–0.87 (median 0.77). Strict leading-eigenvalue convergence
  passed at only 14/40 positions.
- The circle-plane block is ≈ 0.71·I, so the plane is not a neutral ring.
- The answer argmax stabilizes long before the state: median loop 7 vs 74 among convergence completers, with 36/217
  positions censored at loop 128.
- Settling-rate predictors did not beat Huginn's KL stopping rule: the RATE rule needed 6.4 more loops.

**2.3 Period-2 cycles from a flip mode at a source token.**
- **Prevalence:** about 13% of fresh addition prompts (95% CI 8–18%) and 12% of ARC prompts (7.0–19.8%) contain a
  settled period-2 position.
- **Source:** in analyzed cases, the cycle originates at a source token, often a demo "=" or a leading operand digit.
  There, a validated real own-block eigenvalue lies past −1 (−1.007 to −1.226 across 26 census sources; median −1.07),
  and downstream positions, including the answer, inherit the cycle. In two case-study prompts the "=" source is
  itself an answer-prediction position.
- **Bifurcation evidence:** interpolating the prelude output E between a stationary and a cycling context gives
  evidence consistent with a supercritical flip (the tracked eigenvalue crosses −1; amplitude² is approximately linear
  in μ). This replicated on a second problem. Validated negative modes (−0.52 to −0.87) also occur at stationary
  positions in every sampled token category.
- **Behavior:**
  - Real-text census prompts had unchanged tail argmaxes, with identical greedy answers through the first newline at
    budgets 509–512. Unchanged probabilities were not established. In sibling experiments, parity-averaged probability
    differences reached ≈ 5% (demo screen), and ARC letter-probability oscillations were small.
  - A narrow non-textual interpolation window showed phase-dependent argmax differences (loop-parity flicker).
- **Suppression:**
  - Eigenvalue-guided under-relaxation of the source token stopped the source blink in 24/26 prompts. Whole-sequence
    stationarity was reached in only 15/26 under the −0.6 setting.
  - For the 24 previously resolved prompts, finite-time DECAY/PERSIST labels bracketed the predicted transition between
    r = 0.99 and 1.01. This does not directly localize an asymptotic stability threshold. The strict prespecified
    all-six-conditions criterion passed on only 5/24 (the others were UNCLEAR at r = 0.96, near the fp32 floor).
  - A global relaxation α = 0.85 removed detected blinking on ARC, with endpoints unchanged and two prompts unresolved.
    Practical-budget metrics were mixed (budget-16 disagreements 26 → 33).

**2.4 Extra recurrence did not reliably extend dependency handling (tested formats).**
- Pointer chasing (exploratory screen): 90% accuracy at depth 1, 5–10% beyond; one-hop answers on 15/20 depth-2 queries.
- Carry chains (7-digit sums, beyond the 2–4-digit demos): an increase of attainment time with chain length was not
  detected (ρ = 0.08, p = 0.16; 87/150 censored).
  - Exploratory decider tracking was positive with no intervening propagate column, and not consistently positive with
    one to four.
  - Answers did change between loops 16 and 96 on 52/150 prompts.
- These results are compatible with refinement toward stable predictions. They do not identify the computation or
  establish a general one-step limit.
- The early loops clearly do substantial work: on the carry-conditioned units task, digit accuracy rises from 6.5%
  (loop 0) to 100% by loop 16, and truth separation grows from loop 4 to 64.
- In the bf16 pilot, among trajectories ending correct, longer carry dependencies were associated with later
  stabilization (+0.63 log₂ loops per unit), under strong differential-retention caveats.

**2.5 No detectable predictive improvement from the path.** Trajectory features gave no detectable predictive
improvement over endpoint features. Truth prediction was near ceiling; correctness analyses used scalar paths with limited
errors. Other results, briefly:
- An exploratory ~15° decoded carry offset survived several controls but was not separated from column-sum information.
- No clear association was found between low answer confidence and random-perturbation recovery, in a sample selected
  for extreme margins. A gap-shrinking nudge did show associations (exploratory; ρ −0.45 and +0.54).

## 3. Reading versus controlling

*Bridge.* In a frozen local linearization around a settled state (with a settled prefix), a transient state edit evolves
as Jᵗδs, which decays when the local dynamics are attracting. A persistent input change instead moves the fixed point,
to first order by (I − J)⁻¹J P. Empirically, the tested one-shot edits decayed, whereas persistent forcing maintained
shifted readouts.

**3.1 One-shot edits are erased; persistent edits are not (digits).**
- A one-shot answer-position edit decays to sham.
- The same edit applied every loop settles at a shifted state: +3.7 log-odds, displacement 2.5× the push norm.
- Answer control was limited: 8/50 target flips, versus 1/50 for a random edit.

**3.2 Truth is decodable but the probe direction is a weak lever.**
- Mass-mean probes on *statement-final* states decode truth labels at loop 64: AUROC 0.997 cities, 0.914 compare.
  Separation grows from loop 4 (COMPARE peaks at loop 16). This is truth-label decoding, not demonstrated abstract
  truth; cross-dataset transfer was asymmetric.
- Persistent steering along the probe direction at the answer site moved the judgment by a pooled median of 0.08 logits:
  0.57 on CITIES vs 0.06 on COMPARE.
  - Protocol: fold-specific directions, dose equal to the class-mean distance, 24 loops. §3.4's 0.25 uses a frozen
    population mass-mean at norm 12.5 and 48 loops.
  - It passed the prespecified persistence, specificity and dose criteria but failed the bidirectional criterion.

**3.3 Why: weak immediate coupling, not stronger attenuation.**
- The local linear-response model predicted small-dose persistent responses (Spearman 0.995 digits, 0.999 truth).
- In these formats, median immediate readout coupling per unit norm was about 5.4× larger for the tested fitted
  digit chords than for the mass-mean truth directions (0.207 vs 0.0385; the unnormalized contrast is 4.4×). Recurrent
  retention of the truth directions was no weaker (0.66 vs 0.41).
- These results favor weak immediate coupling over stronger recurrent attenuation as the explanation for the weak truth
  steering.
- Recurrence nevertheless reversed the predicted sign of response on 42% of digit items and 19% of truth items, and
  digit responses were substantially nonlinear at full dose.

**3.4 Loop-aware steering of the True/False readout (d*).**
- d* = normalize(mean_i normalize(A_{48,i}ᵀ g_i)) is built on 60 training statements.
- On 60 held-out statements (30 units) at matched norm 12.5, with persistent forcing, it shifted True/False log-odds by
  a median of 8.13, versus 0.93 for the readout gradient and 0.25 for the probe direction.
- It is nearly orthogonal to both (cos 0.096 and 0.025).
- The model is strongly False-biased (sham accuracy 55%), so true-to-false flip evidence is limited to 3 sham-correct
  true statements; log-odds shifts are measured on all 60.
- **Readout lever, not demonstrated truth manipulation:**
  - d* raises the literal True token and shifts Yes/No and A/B formats toward their true-designated labels. Yes and A
    are confounded with affirmative/first-option polarity.
  - The formats that would separate truth from polarity failed baseline qualification, so the verdict is INCONCLUSIVE.
  - Descriptively, in "Is this statement false?", +d* moved toward "Yes" (that is, toward *False*) by 0.48, similar to a
    random direction (0.36). That is consistent with response polarity.
- Small-dose responses closely follow the local linear prediction (observed/predicted 0.93), with increasing
  sublinearity at larger doses. Effects decay after forcing stops.

**3.5 Quantitative control on digits.**
- Frozen loop-aware digit levers (norm 3, 48 loops of forcing) raise the target centered digit logit by +1.78 on 20
  held-out prompts, versus ≈ −0.05 for readout-gradient and centroid directions.
- Each prompt's own linearized operator predicts the full 10×10 response matrix: median r = 0.978, slope 0.95,
  normalized error 0.22.
- Averaged or swapped operators are reliably worse (r 0.93–0.95). The immediate-readout prediction is anticorrelated
  (−0.59).
- The levers were constructed from the same operator family, so this is direction-conditioned prediction, though
  held out.

## 4. Mechanism of the readout reversal (digits)

All analyses in this section use the same 20 held-out digit prompts and perturb only the answer position.

**4.1 The reversal is a first-step reorientation.**
- Levers start slightly negative on the target readout and become strongly positive after one loop: c₀ = −0.13 →
  c₁ = +0.94.
- Because the objective starts at t = 1, it places no requirement on the immediate response c₀. Empirically, the frozen
  levers have slightly negative immediate responses but strongly positive one-step responses. The lower bound makes
  this possible; it does not explain why the sign is negative.
- Basis-independently:
  - 154/157 lever reversals already flip sign after one application of J.
  - Readout-gradient pushes are suppressed after one step (|c₁/c₀| = 0.05) without overall contraction (‖JP‖/‖P‖ =
    1.17).
- A modal (Arnoldi) analysis reproduced the responses (r = 0.978). Removing modal gain signs/phases substantially
  attenuates the immediate-vs-accumulated anticorrelation (−0.59 → −0.08 [−0.27, 0.05]; −0.17 at Krylov size 40). The
  residual anticorrelation is uncertain, and the counterfactual does not identify a causal fraction.
- The prespecified label is REWEIGHTING, meaning the no-flip correlation remained negative in point estimate.
- The estimated modal decomposition is ill-conditioned and shows substantial cancellation (eigenvector-matrix
  condition number median 75; cancellation ratios 20–36×), so individual-mode attributions are unreliable. The
  time-domain result above is the robust finding.

**4.2 The readout span is fed from outside itself.**
- Let Q be an orthonormal basis matrix for the span of the 10 digit-readout gradients (rank 9); norms are Frobenius.
  About 97% of the squared sensitivity of the one-step readout map QᵀJ comes from directions outside Q (ρ = ‖QᵀJQ‖/‖QᵀJ‖
  = 0.17).
- The within-span map is weak (spectral radius 0.17).
- Levers lie largely outside Q: in-span norm ratio ‖QᵀP‖/‖P‖ = 0.12, about 1.4% of squared norm. Readout-gradient
  pushes: 0.71.
- Levers have three times the one-step readout gain of readout-gradient pushes (0.34 vs 0.11).
- This is local Jacobian sensitivity, not measured activation flow.

**4.3 Which paths carry the one-step response.**
- Every perturbation at q enters the block through the adapter (dw₀ = W_s P). An exact linearized decomposition splits
  the lever's one-step target response by the *last write* on each path:
  - the direct path from the adapter output through residual/normalization derivatives only: 52.9% [47.7, 55.8];
  - paths ending in MLP writes: 39.9% [37.6, 43.7];
  - paths ending in attention writes: 7.4% [5.1, 8.4], of which 96% is via the query re-weighting fixed context.
- No group demonstrated a majority under the prespecified rule.
- Because only q is perturbed, context keys and values are fixed. Attention can act only through q's query and its own
  key/value, so cross-position transmission is not measured.
- Readout-gradient pushes are "not transmitted" under this decomposition: their gross target contribution is 0.20 of
  the immediate effect, with partial direct-path (+) versus MLP (−) cancellation.
- This is derivative accounting, not ablation.

**4.4 A fixed adapter transpose approximates the tested adapter pullbacks.**
- On the tested covectors, the full direct-path pullback W_sᵀR_adapterᵀg (R_adapter = the residual/normalization
  chain; not the readout R) and W_sᵀg were nearly collinear: median cosine
  0.9987 (min 0.9937), norm ratios 2.71–2.87 (5th–95th percentile 2.72–2.86).
- This supports a scalar approximation of the residual/normalization chain *for these pullbacks*, not a globally scalar
  normalization map.
- A candidate explanation (untested): each residual RMSNorm acts on covectors nearly orthogonal to the state as
  approximately a scalar 1/rms, provided its gain is near-isotropic on these directions.
- W_sᵀ reorients the readout gradient from cosine −0.04 with the lever to 0.40.

## 5. Cheap levers and the limits of control

**5.1 Cheap levers (digits; frozen population directions; norm 3; 20 held-out prompts; 48 loops; effects are mean
changes in the target centered digit logit).**

| Direction | Effect | Fraction of the 48-step lever |
|---|---|---|
| Readout gradient g | −0.05 | — |
| Adapter pullback W_sᵀR_adapterᵀg | 0.875 | 49%; the prespecified half-of-reference test was INCONCLUSIVE |
| W_sᵀg | 0.874 | — |
| One-step Jᵀg | 1.35 | 76%; passed |
| 48-step lever | 1.78 | 100% |

- Overlap-matched controls, built for the adapter pullback and for Jᵀg, share each lever's cosine with the reference and
  have random remainders. They reproduce about 85% and 90% of the levers' effects.
- Both levers exceed their controls: +0.13 [0.07, 0.20] and +0.14 [0.09, 0.20] (97.5% CIs). These intervals resample
  prompts conditional on two random control draws per target.
- The remainders orthogonal to the reference steer on their own. These are effect ratios, not reconstructions of the
  reference direction.

**5.2 Cheap levers (truth; frozen population directions; 60 held-out statements; dose 1.5625; 24 loops; effects are
median opposite-label True/False log-odds shifts).**

Cosines with d*: Jᵀg 0.70, W_sᵀg 0.40, g 0.10.

| Lever | Effect | Fraction of d* | Off-target KL | Construction time per training statement |
|---|---|---|---|---|
| d* | 1.24 | 100% | 0.060 | 1.19 s (estimated: g + 48 VJPs) |
| Jᵀg | 0.85 | 68% | 0.025 | 0.037 s |
| W_sᵀg | 0.49 | 39% | 0.005 | 0.0125 s |
| g | 0.12 | 9% | 0.0007 | 0.0125 s |

- "Off-target KL" is KL(x* reference ‖ pushed) after renormalizing over all tokens except True/False.
- **Timing:** measured with cuda synchronization, batch 40. It excludes the 0.75 s shared 64-loop warm-up; W_sᵀg costs
  essentially nothing beyond computing g. Population levers are built once and amortized, so they require no
  per-test-prompt direction construction.
- **Hypotheses (effects E):** the prespecified primary paired contrast median(E_{Jᵀg} − ½E_{d*}) was positive (0.248
  [0.247, 0.249]). The exploratory median(E_{W_sᵀg} − ½E_{d*}) was negative (−0.136).
- **Item-specific levers:** d*_item 1.30 and (Jᵀg)_item 0.88; the exploratory paired contrast
  median(E_{Jᵀg,item} − ½E_{d*,item}) = 0.251 was positive.
- **Linearity:** effects are linear to dose 3.125. At 12.5 the relative efficacy gaps narrow (Jᵀg/d* rises from 0.68
  to 0.82).
- **Collateral:** median KL / median E² is lowest for W_sᵀg. This is descriptive and on-prompt only; matched-effect
  collateral was not tested.
- **Oracle-sign correction:** with ground-truth-informed signs, the population d*, Jᵀg and W_sᵀg corrected all 27
  False-biased errors at every tested dose. This is readout control with an oracle sign, not truth inference.

**5.3 Probe-selected steering transmits the probe's verdict (retrospective diagnostic).**
- A frozen probe, trained on 90 other units, chose a binary steering sign for the 60 held-out statements, using
  already-recorded both-direction runs.
- All nine lever × dose policies exactly reproduced the probe's binary verdict: 76.7% accuracy, 0 discordant units. Even
  the weakest tested policy overcame the baseline margins wherever the selector required a changed verdict.
- A training-fitted combination of the model readout and cross-fitted probe scores reached 90.0%. A post-hoc probe-only
  recalibration also reached 90.0%, so the model readout added no detected benefit.
- The 76.7% ceiling reflects binary sign selection with an uncalibrated threshold at these doses. A recalibrated
  selector would also be transmitted if the pushes overcame the margins on its newly selected flips; this was not
  evaluated.
- This shows no accuracy gain from steering in this setting. It does not show that steering can never combine
  information at smaller or continuous doses. Effects on downstream generation were not tested.

## 6. Interpretation

- **Decoding and control directions differ.**
  - In this looped model, the tested mass-mean probe direction and the most effective tested control direction for the
    corresponding readout were nearly orthogonal.
  - Pulling the readout gradient back through the recurrence finds strong control directions.
  - On digits, much of the needed reorientation happens in one loop update, carried by the adapter's direct path and
    MLP-mediated paths.
  - In this model and these tasks, probing results alone were therefore poor guides to intervention design.
- **The loop-aware geometry is cheap to approximate.**
  - One-step pullbacks recover 76% (digits) and 68% (truth, low dose) of the reference effect.
  - Adapter-transpose directions recover 49% and 39%.
  - These remain *readout* levers. The truth levers' behavior is consistent with response polarity, and is not shown to
    manipulate an internal truth representation.
- **Steering transmits the sign-choosing monitor (here).** When a probe chooses the sign at the tested doses,
  answer-level steering delivers the probe's verdict. Its distinctive value, if any, would have to come from effects on
  subsequent behavior, which remain untested.
- **Recurrence and stability.**
  - In the tested settings, late recurrence settles readouts into locally attracting, near-stationary states. The tested
    one-shot edits decayed, whereas persistent forcing maintained shifted readouts.
  - Flip modes at source tokens (about 13% of addition prompts and 12% of ARC prompts) produce period-2 cycles that are
    mechanistically traceable and largely suppressible by damping.

## 7. Limitations

- **Scope.**
  - One model.
  - Small held-out sets: 20 digit prompts shared by all mechanism analyses; 60 truth statements (30 units); 30
    statements for format controls.
  - Few prompt formats; teacher forcing at answer positions.
  - Huginn performs near chance on several formats (weekday arithmetic, counterbalanced truth formats, ARC).
- **Own-position analysis.** The lever and readout-decomposition analyses perturb and differentiate only the last token.
  Full-sequence dynamics are block lower-triangular, and cross-position coupling (for example, upstream flip modes, or
  prefix-push transient growth) is outside the lever and decomposition analyses.
- **Linearizations.** Directions are local linearizations at fixed points, frozen and averaged over training items.
  Truth responses are sublinear at dose 12.5. Digit levers were tested at a single dose (norm 3), so their linearity is
  unverified.
- **Controls.** Random-direction controls use few draws (often one per item or target). Their CIs are conditional on
  those draws.
- **Attribution.** The direct-path / MLP / attention split is a last-write path convention, not an ablation.
- **Readout versus representation.** Truth levers act on a True-versus-False token readout. Truth versus polarity is
  unresolved.
- **Retrospective analyses.** The selector analysis reused statements whose intervention outcomes had been inspected. It
  was prespecified before computation but is a diagnostic.

## 8. Negative and inconclusive results (as prespecified)

- Weekday-offset rotation: INCONCLUSIVE (Huginn did not reliably perform the task: 24.5% / 20.4% vs 14.3% chance). All
  seven phase-validity gates failed.
- Digit rotation: INCONCLUSIVE.
- Month path: INCONCLUSIVE.
- Relaxation threshold sweep, strict criterion: 5/24.
- Settling speed: the primary Jacobian–convergence correlation was INCONCLUSIVE (ρ 0.24 [−0.15, 0.64]), and early exit
  was worse than KL.
- Carry-chain primary (attainment time vs chain length): not detected (ρ 0.084, one-sided p 0.160).
- Truth stage 1: the behavioral-accuracy threshold for interpreting probe-vs-behavior analyses (Q4) was not met.
- Trajectory information beyond the endpoint: no detectable improvement.
- Probe-direction truth steering: failed the bidirectional criterion. One-shot erasure and post-removal recovery:
  INCONCLUSIVE.
- Stronger recurrent attenuation of truth directions (H_att): not supported (ratio 0.62 [−0.22, 1.22]).
- d* truth versus token, and truth versus polarity: INCONCLUSIVE (formats failed qualification).
- Block decomposition majority: NO MAJORITY GROUP.
- Adapter lever ≥ half of reference (digits): INCONCLUSIVE. W_sᵀg ≥ half of d* (truth): NEGATIVE.
- Probe-selected steering beyond the selector: no added accuracy detected.

## 9. Relation to prior work (brief)

- **Probing versus intervention, and representation engineering.** Examples: Marks & Tegmark 2023 (arXiv:2310.06824),
  inference-time intervention (Li et al. 2023), representation engineering (Zou et al. 2023), activation addition
  (Turner et al. 2023).
  - These study linear representations and activation interventions in feed-forward transformers, including differences
    between predictive and causal efficacy. Marks & Tegmark found mass-mean directions more causally implicated than
    other probe directions.
  - Here, the gap is analyzed through a recurrent fixed point: single-step additions are erased by contraction, and
    effective directions are those the loop's Jacobian carries into the readout.
- **Looped / recurrent-depth models.** Orbits in Huginn's latent trajectories were reported qualitatively by Geiping et
  al. 2025 (arXiv:2502.05171, §7) for numerical tokens, and by Blayney et al. 2026 (arXiv:2604.11791) on GSM8K, without
  period or mechanism. A Lyapunov study (ICML 2026) reports uniform convergence; a contracting period-2 orbit is
  compatible with negative exponents. Period-doubling cascades were shown in an external soft-token feedback loop of
  standard LMs (arXiv:2610.31692), and a fold bifurcation in a small looped transformer (Kim 2026, arXiv:2608.26556).
  - Here we show a source-token flip mode with bifurcation evidence inside a pretrained recurrent-depth LM's loop,
    context-controlled switching, and eigenvalue-guided suppression.
  - This is based on a limited literature search, not a systematic review.
- **Jacobian-based steering.** The resolvent pullback is standard linear-response reasoning. The empirical findings here
  are its held-out accuracy, the first-step reorientation, the outside-fed readout span, and the effectiveness of
  one-step and adapter-transpose approximations.

## 10. Reproducibility

- Code: `huginn_addition/intervene_app.py`, a Modal app with all GPU functions and configs; `analyze_*.py` and
  `make_*.py` hold the analyses and builders, with frozen rules and recorded amendments in their docstrings.
- Run directories hold `meta.json` with config and model revision. Code hashes are recorded from the later experiments
  onward; the early pilot through cycle-source runs lack them. Push-file hashes are recorded in the ovl_eval and
  tlever_eval runs.
- Key runs:

| Result | Run directory |
|---|---|
| d* | `dstar_vjp-20261106-061708`, `dseval-20261106-062151` |
| Format controls | `ctrl_formats-20261106-064804` + `ctrl_formats-20261106-070550` (two merged shards), `ctrl_cb-20261106-071755`, `ctrl_dose-20261106-070217` |
| Digit levers | `dglever_*-20261106-08*` |
| Reversal mechanism | `revmech-20261106-204036`, `rtransfer-20261106-205118`, `blockdecomp-20261106-220105` |
| Cheap levers | `adaptcov-20261106-224906`, `adlever_eval-20261106-225059`, `ovl_eval-20261106-23*` |
| Truth cheap levers | `truthcov-20261106-234613`, `tlever_eval-20261106-235208`, `-235648`, `-20261107-000141` |
| Selector | `selector_summary.json` and `selector_posthoc_probe_only.json` in the first tlever run |
