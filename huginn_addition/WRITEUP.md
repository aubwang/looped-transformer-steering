# Local recurrent response predicts and enables readout control in a looped language model

*An empirical study of Huginn-0125 (pinned revision `bb6621b…`). Working write-up, 2026-11-07. Revised after
independent reviews by GPT-6 Astra (two rounds), Gemini 4 Argon (round 1), a numbers-level fact-check against the raw
saved outputs, and a full scientific review (both Claude Opus 5.5).*

The study combines prospectively specified experiments with explicitly identified exploratory, post-hoc and
retrospective analyses. Designs, analyses and result wording underwent model-assisted review (GPT-6 Astra throughout).
`README.md` records each experiment's timing, gates, amendments and accepted wording; run directories are under
`results/`.

## Summary

Huginn-0125 is a recurrent-depth ("looped") transformer: a prelude embeds the prompt, one core block is applied
repeatedly to a latent state, and a coda reads out tokens. We studied how concepts are represented in the loop, and how
they can be read and controlled.

**Main contribution: control.**
- In these tasks, directions that *decode* a label (linear probes) give weak behavioral control. At matched push norm,
  the tested probe direction moved the True/False readout 0.25 log-odds, versus 8.1 for the loop-aware lever.
- A local linear-response model of the recurrence predicts small-dose steering responses.
- Pulling the readout gradient back through the recurrent dynamics gives much stronger, forcing-dependent control on
  held-out items:
  - a median True/False log-odds shift of 8.1 at push norm 12.5, versus 0.93 for the readout gradient and 0.25 for
    the probe direction;
  - a mean gain of +1.78 in the target digit's centered logit (189/200 target effects positive; all 20 prompts),
    versus ≈ −0.05 for readout-gradient and centroid directions, which is at random-direction noise level (|Δz| ≈ 0.05);
  - a quantitatively predicted 10×10 digit response matrix (median r = 0.978).
- On digits, the response reorients within a single loop update, and the readout span is fed mostly from outside
  itself. One-step pullbacks (one backward pass through the block) retain much of the effect: 76% on digits, 68% on
  truth at low dose. On truth this is about what their cosine with the reference lever (0.70) predicts; on digits they
  modestly exceed overlap-matched random controls.
- In a retrospective test where a probe chose the steering sign, steering added no answer accuracy beyond the probe's
  own verdict.

**Second contribution: dynamics.**
- Late answer-position states are locally attracting and near-stationary in most tested prompts.
- A minority of prompts (about 13% of fresh addition prompts) settle into period-2 cycles. In all 26 census cases, the
  earliest period-2 token has a validated own-block eigenvalue past −1 (−1.007 to −1.226). Separately, two
  contextual-input interpolation families gave evidence consistent with a period-doubling (flip) bifurcation.
- Eigenvalue-guided under-relaxation of that source token stopped its own blink in 24/26 prompts, but all motion in
  only 15/26: residual downstream motion often persists.

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
- Pointer-chasing and carry-chain screens.
- Weekday/month successor pilots.
- ARC-Challenge multiple choice (stability only).
- True/False judgments: CITIES (city–country) and COMPARE (two-digit comparisons), with held-out units.

**Validation practice.**
- Prespecified hypotheses, labels and numerical gates, for the confirmatory experiments.
- Synthetic-world checks of analyses where relevant, and smoke runs.
- Model review of design and code before runs, with changes applied before data. Documented exceptions (post-run
  amendments and deviations) are recorded in the README: the rtransfer rank cut, the ctrl shard merge, the tlever label
  fix, the post-hoc selector baseline, the tsteer_ans P5 bootstrap (pooled rather than dataset-stratified), and the
  revmech dominant-pair Ritz residuals (prespecified but not reported).
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
  log-odds; for the primary +1-target controls, random directions −0.06 to +0.37, radial −0.28, and random directions
  within the top-50-PC subspace −0.17 to +0.15). Under continued recurrence the push decays, and the final answer
  equals baseline on 168/168 prompts.
- Local readout gradients align with ±1 neighbors. Far-offset sensitivity is dominated by lowering the current digit,
  and less than 1% of readout-gradient energy lies in the circle plane.
- Exploratory/partial evidence of cyclic structure:
  - weekdays: the supplied start day; canonical order ranked 1/360; post-hoc;
  - months: leave-one-month-out R² 0.22, beating 5,000 reorderings; path verdict INCONCLUSIVE.
- Late local dynamics contract the fitted plane without systematic rotation. The separate test of start-anchored
  computational rotation was INCONCLUSIVE; the motion was mostly additive radial growth. No complete circular decoder is
  claimed.

**2.2 Locally attracting, near-stationary states (most tested prompts).**
- At the tested answer positions, late states are near-stationary in fp32 (per-step drift ~5e-6). The largest observed
  Ritz modulus has median 0.78 (p10–p90 0.75–0.82).
- A separate token-stratified sample gave leading moduli 0.66–0.87 (median 0.77). Strict leading-eigenvalue convergence
  passed at only 14/40 positions.
- The circle-plane block is ≈ 0.71·I, so the plane is not a neutral ring.
- The answer argmax stabilizes long before the state: median loop 7 vs 74 among convergence completers, with 36/217
  positions censored at loop 128.
- Settling-rate predictors did not beat Huginn's KL stopping rule: the RATE rule needed 6.4 more loops.
- Exceptions: 26 period-2 and 10 unresolved prompts among 200 census prompts; 24/100 ARC prompts not stationary at
  baseline (12 blinking, 12 unresolved); 36/217 settle positions still drifting at loop 128; month prompts not
  stationary by loop 96.

**2.3 Period-2 cycles from a flip mode at a source token.**
- **Prevalence:** about 13% of fresh addition prompts (95% CI 8–18%) and 12% of ARC prompts (7.0–19.8%) contain a
  settled period-2 position.
- **Source token:** in all 26 census cases, the earliest period-2 token, often a demo "=" or a leading operand digit,
  has a validated real own-block eigenvalue past −1 (−1.007 to −1.226; median −1.07). The eigenvalue is measured at the
  hidden unstable fixed point between the two cycle states. In two case-study prompts the "=" source is itself an
  answer-prediction position.
- **Inheritance:** in the six cycle-source case studies, freezing everything before the answer-prediction position
  stopped the answer-position cycle. But in problem 50, "=" kept sustaining period-2 motion when the prefix through the
  first source was pinned: a demonstrated second oscillator. In the relaxation run, residual downstream motion is
  common after the selected source is suppressed, consistent with additional oscillators or other unresolved dynamics
  (see Suppression).
- **Robustness:** for the two problems tested in fp64, fp64 closely reproduces the cycling amplitudes. Caveat added
  2026-11-11: those fp64 runs were mixed precision. The pinned model code casts to float32 inside its normalisation
  and rotary step, and the harness cast the logits to float32. A fully patched double path exists only for the
  flicker check below; it does not retrospectively validate these amplitude measurements.
- **Bifurcation evidence:** interpolating the prelude output E between a stationary and a cycling context gives evidence
  consistent with a supercritical flip: the tracked eigenvalue crosses −1, amplitude² is approximately linear in μ = −1
  − λ, and the cycle's two-step multiplier is close to the leading-order prediction 1 − 4μ. This replicated on a second
  problem. Validated negative modes (−0.52 to −0.87) also occur at stationary positions in every sampled token category.
- **Behavior:**
  - Real-text census prompts had unchanged tail argmaxes at the answer-prediction positions (the only positions read
    in that run), with identical greedy answers through the first newline at budgets 509–512. Unchanged probabilities were not established. In sibling experiments, parity-averaged probability
    differences reached ≈ 5% (demo screen), and ARC letter-probability oscillations were small.
  - A narrow non-textual interpolation window showed phase-dependent argmax differences (loop-parity flicker).
  - **Addendum 2026-11-11, real-text flicker at operand-choice positions** (`docs/flicker_design.md`; runs
    `flicker_scan-20261111-022450`, `flicker_confirm-20261111-023055`, `flicker_confirm64-20261111-023903`; reviews
    `results/flicker_prerun_review.md`, `results/flicker_results_review.md`). Reading the output at every position of
    44 prompts selected for oscillating (26 census, 18 screen), three real-token prefixes have a most-probable next
    token that alternates at every budget from 449 to 512, from the census start: after `839 +` (` 8` at even
    budgets, ` 1` at odd; smallest lead 0.040 in log-probability), after `2136 +` (` 1` / ` 2`; 0.010) and after
    `2136 + 406` (` =` / `4`; 0.0014). All three are guesses about an operand, and neither phase matches the token
    that actually follows. None of the 551 final or demonstration answer positions changed its top token. The
    model's own forward on the cut sequence reproduces budgets 509–512; a patched double-precision computation
    reproduces the alternating pairs over 505–512; three float32 computations agree on the gap between the two
    tokens to about 2e-5. Four other start seeds also alternate, and for `2136 +` two of them are in the opposite
    phase. Not established: any answer-position flicker, an effect on accuracy, prevalence in general text,
    persistence beyond 512 loops, or behaviour with the default random start and stopping rule.
    Descriptive follow-up added after the results (each branch uses its own budget for every token, through the
    first newline): after `839 +`, 511 loops per token writes ` 104 = 943` and 512 loops writes ` 839 = 1678`, two
    different correct sums; after `2136 + 406`, 511 loops writes `4 = 6300` (wrong: 2136 + 4064 = 6200) and 512
    loops writes ` = 2542` (correct). The branches choose different operands; they are not different answers to
    one fixed sum.
  - **Addendum 2026-11-11, answer-digit flicker in a targeted search** (`docs/flicker2_design.md`; runs
    `flicker2_screen-20261111-040148`, `flicker2_scan-20261111-040507`, `flicker_confirm-20261111-041415`,
    `flicker_confirm64-20261111-041858`, `flicker_continue-20261111-042010`; reviews
    `results/flicker2_prerun_review.md`, `results/flicker2_results_review.md`). Selection, three times over: the
    worked examples of four prompts whose oscillation starts inside the examples (20059, 20145, 20144, 20178); 400
    new final sums screened at 32 loops for a nearly tied answer digit; the 85 closest scanned at 512 loops
    together with the three earlier flickers as positive controls (re-found). In 84 of the 85 every answer
    position is on a settled period-two orbit. Among 358 settled final-answer positions, four have a most-probable
    token that alternates at every budget from 449 to 512 between the correct digit and a wrong digit: `924 + 675`
    (second digit, 5/6, smallest lead 0.224), `816 + 254` (third digit, 7/6, 0.031), `1279 + 7419` (last digit,
    8/9, 0.030), `769 + 843` (third digit, 1/0, 0.027). All four are confirmed by the model's own forward
    (509–512), by the patched double-precision path (505–512, same pair) and by the numerical guard. The scan also
    has six flickers at other positions of the new prompts and one unresolved position (uid 30381, position 42).
    4/358 is the yield of this selected sample, not a rate in any population and not an accuracy estimate.
    Continuation from the `=` sign, 7 tokens, each branch at its own budget for every token: `816 + 254 =` gives
    ` 1070` at 511 loops (correct) and ` 1060` at 512 (incorrect). `924 + 675 =` gives ` 1609` and ` 1509`, both
    incorrect (1599): its correct/wrong flip holds only with the true earlier digits supplied.
    Start dependence (four other seeds, budgets 505–512): for `816 + 254` one seed gives the correct digit at all
    eight budgets and three alternate; for `924 + 675` all alternate but two seeds have a different gap
    (−0.178/+0.260 against −0.224/+0.318), so the orbit observed is not unique across starts; for the other two
    all alternate, some in the opposite phase. Not established: any general effect on accuracy, prevalence,
    behaviour with the default random start and stopping rule, or anything beyond 512 loops.
- **Suppression:** under-relaxation replaces the update at the source token by s ← (1 − a)s + aF(s). This maps the flip
  eigenvalue λ to 1 − a + aλ, which crosses −1 at a* = 2/(1 − λ). The relaxation strength is r = a/a*, and the setting
  is chosen so the relaxed multiplier hits a target (−0.6 or −0.85).
  - Relaxing only the source token stopped its own blink in 24/26 prompts. Controls: the sham kept blinking in 26/26,
    and insufficient damping (multiplier still past −1) kept it blinking in 25/25. That is what makes the result
    eigenvalue-specific.
  - The fitted decay rate matched the −0.85 target (0.851) but not the −0.6 target (0.918; cause unverified).
  - Whole-sequence stationarity was reached in only 15/26 under the −0.6 setting. In at least 6 of the 24 prompts where
    the source stopped, a downstream position kept moving at 0.4–0.95× its undamped drift.
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
  - Answers did change between loops 16 and 96 on 52/150 prompts (18 became correct, 16 became incorrect).
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
  for extreme margins. A gap-shrinking nudge did show associations (exploratory; ρ −0.45, Holm p 0.094; ρ +0.54, Holm
  p 0.028).

## 3. Reading versus controlling

*Bridge.* In a frozen local linearization around a settled state (with a settled prefix), a transient state edit evolves
as Jᵗδs, which decays when the local dynamics are attracting. A persistent input change instead moves the fixed point,
to first order by (I − J)⁻¹J P. Empirically, the tested one-shot edits decayed, whereas persistent forcing maintained
shifted readouts.

**3.1 One-shot edits are erased; persistent edits are not (digits).**
- A one-shot answer-position edit decays to sham.
- The same edit applied every loop settles at a shifted state: +3.7 log-odds, displacement 2.5× the push norm.
- Much of that displacement does not reach the readout. In the response run on the same 50 prompts, the settled
  persistent readout (3.72) was below the one-shot edit's immediate readout (4.13).
- Answer control was limited: 8/50 target flips (versus 1/50 for a random edit), and 12/50 produced other digits; at
  double scale, 5/50 target flips and 31/50 other digits.

**3.2 Truth is decodable but the probe direction is a weak lever.**
- Mass-mean probes on *statement-final* states decode truth labels at loop 64: AUROC 0.997 cities, 0.914 compare.
  Separation grows from loop 4 (COMPARE peaks at loop 16). This is truth-label decoding, not demonstrated abstract
  truth; cross-dataset transfer was asymmetric.
- Persistent steering along the probe direction at the answer site moved the judgment by a pooled median of 0.08 logits:
  0.57 on CITIES vs 0.06 on COMPARE.
  - The answer-site median difference is **largely accounted for by unequal doses**. The dose was the class-mean
    distance, about 10× larger for CITIES (22.4–23.6) than for COMPARE (2.3–2.6), and dose-normalized medians are
    similar (0.025 vs 0.026 logits per unit norm).
  - At the statement-final site (doses ≈ 44 vs 5.8), the normalized medians still differ (0.012 vs 0.008, 1.6×).
    Matched-dose equivalence of the datasets was not tested.
  - Per-dataset doses and effects are saved in `results/truthsteer_dose_by_dataset.json` (post-hoc).
  - Protocol: fold-specific directions, dose equal to the class-mean distance, 24 loops. §3.4's 0.25 uses a frozen
    population mass-mean at norm 12.5 and 48 loops.
  - It passed the prespecified persistence, specificity and dose criteria but failed the bidirectional criterion.

**3.3 Why: weak immediate coupling, not stronger attenuation.**
- The local linear-response model predicted small-dose persistent responses (Spearman 0.995 digits, 0.999 truth).
- Per unit push norm, immediate readout coupling was about 5× larger for the tested digit chords (0.207) than for the
  mass-mean truth directions (0.041 CITIES, 0.038 COMPARE). This is mainly because the digit readout is more sensitive
  (‖g‖ 2.76 vs 0.66); both kinds of direction are nearly orthogonal to their readout gradients (cos 0.08 vs 0.06).
- The pooled unnormalized ratio (4.4×) is unstable: truth coupling splits into two dataset clusters (because of the
  dose difference in §3.2), and its bootstrap CI is 3.0–40.7.
- Stronger recurrent attenuation of the truth directions was not supported (ratio 0.62, CI −0.22 to 1.22). Among
  positive responders, truth retention was somewhat lower than digit retention (geometric mean 0.63 vs 0.82), but
  coupling accounts for 2.88 of the 3.14 log-unit gap. All-item retention medians (0.66 vs 0.41) mix in sign reversals.
- These results favor weak immediate coupling over stronger recurrent attenuation as the explanation for the weak truth
  steering.
- Recurrence reversed the predicted sign of response on 42% of digit items and 19% of truth items (all in CITIES).
- Digit responses were *superlinear* at full dose: median observed 3.72 versus a linear prediction of 1.44, and 20/21
  small-dose reversals were positive at full dose. The +3.7 in §3.1 is therefore outside the regime where the linear
  model is validated. The small-dose fit holds within each truth dataset.

**3.4 Loop-aware steering of the True/False readout (d*).**
- d* = normalize(mean_i normalize(A_{48,i}ᵀ g_i)) is built on 60 training statements.
- For one item, A_Tᵀg = ∇_P ΔR_T at P = 0, so the item-specific lever is the first-order optimal persistent steering
  vector for the readout being evaluated. The population d* averages normalized item covectors and is not itself
  optimal for the mean readout change.
- The probe and readout-gradient baselines are not optimized for that objective. The comparison therefore shows how
  much the loop changes the effective direction, not that probes are poorly chosen in general.
- The push norm 12.5 came from the earlier probe-steering doses: it is the median class-mean distance, which falls
  between the two dataset clusters (≈ 2.3 and ≈ 23). It is about 16% of the answer-position state norm (median 76).
- On 60 held-out statements (30 units) at matched norm 12.5, with persistent forcing, it shifted True/False log-odds by
  a median of 8.13, versus 0.93 for the readout gradient and 0.25 for the probe direction.
- It is nearly orthogonal to both (cos 0.096 and 0.025).
- The model is False-biased (sham accuracy 55%). True statements sit just below the threshold (median sham log-odds
  −0.10, range −0.39 to +0.08); false statements are further below (median −0.56). True-to-false flip evidence is
  therefore limited to 3 sham-correct true statements; log-odds shifts are measured on all 60.
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
  held-out prompts (189/200 target effects positive; all 20 prompts positive on average). Readout-gradient and
  centroid directions give ≈ −0.05, at random-direction noise level (median |Δz| 0.046).
- Each prompt's own linearized operator predicts the full 10×10 response matrix: median r = 0.978, slope 0.95,
  normalized error 0.22.
- Excluding the target diagonal, the prediction still reaches r = 0.972.
- Averaged or swapped operators are reliably worse, though the training-averaged operator already reaches r = 0.947
  (swapped 0.93–0.94). The immediate-readout prediction is anticorrelated (−0.59).
- The levers were constructed from the same operator family, so this is direction-conditioned prediction, though
  held out.

## 4. Mechanism of the readout reversal (digits)

All analyses in this section use the same 20 held-out digit prompts and perturb only the answer position.

**4.1 The reversal is a first-step reorientation.**
- Levers start slightly negative on the target readout and become strongly positive after one loop: c₀ = −0.13 →
  c₁ = +0.94.
- Because the objective's sum starts at t = 1, it places no requirement on the immediate response c₀.
- Relation to readout-gradient steering: for an item-specific lever a = A_Tᵀg/‖A_Tᵀg‖, c₀ = gᵀA_Tᵀg/‖A_Tᵀg‖, which is
  proportional to the accumulated readout response to a push along g itself. So the two signs are linked: a lever
  starts negative exactly when a readout-gradient push ends negative. This identity does not explain why the common
  sign is negative.
- For the frozen population pushes this holds approximately: across the 200 (prompt, digit) cells, lever c₀ correlates
  with the readout-gradient push's accumulated 48-loop response (r = 0.64, 68% sign agreement), and both medians are
  negative (−0.134 and −0.059).
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
- Reference points: ρ = 1 would require the readout span to be invariant under the adjoint Jᵀ, with no sensitivity to
  inputs outside the span. As an isotropic random-mixing benchmark, a random 9-dimensional subspace would give about
  √(9/5280) ≈ 0.04.
- The within-span map is weak: spectral radius 0.17, versus ≈ 0.78 for J itself.
- Levers lie largely outside Q: in-span norm ratio ‖QᵀP‖/‖P‖ = 0.12, about 1.4% of squared norm. Readout-gradient
  pushes: 0.71.
- Levers have three times the one-step readout gain ‖QᵀJP‖/‖P‖ of readout-gradient pushes (0.34 vs 0.11).
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
| W_sᵀg | 0.874 | 49% (not separately tested; nearly collinear with the adapter pullback) |
| One-step Jᵀg | 1.35 | 76%; passed |
| 48-step lever | 1.78 | 100% |

- Overlap-matched controls, built for the adapter pullback and for Jᵀg, share each lever's cosine with the reference and
  have random remainders. They reproduce about 85% and 90% of the levers' effects.
- Both levers exceed their controls: +0.13 [0.07, 0.20] and +0.14 [0.09, 0.20] (97.5% CIs). These intervals resample
  prompts conditional on two random control draws per target.
- The remainders orthogonal to the reference steer on their own. These are effect ratios, not reconstructions of the
  reference direction.
- For a single item, linear response gives ΔR_i ≈ ‖P‖·‖A_{T,i}ᵀg_i‖·cos(P, A_{T,i}ᵀg_i).
- For the frozen population directions, a descriptive reference-component projection, mean_{i,k}[c_k·Δ_{L,i,k}] with c_k
  the population cosine with the reference, gives 1.18 (Jᵀg) and 0.71 (adapter pullback). The levers' observed effects
  are 1.35 and 0.875. This is not a full linear-response prediction; the overlap controls test against the observed
  matched controls (1.21 and 0.74).

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
- **Fractions ≈ cosines:** the fractions of d* closely match the population directions' cosines with d* (0.68/0.39/0.09
  vs 0.70/0.40/0.10). The substantive finding is that the population one-step direction already has cosine ≈ 0.70 with
  the population 48-step direction.
- **Timing:** measured with cuda synchronization, batch 40. It excludes the 0.75 s shared 64-loop warm-up; W_sᵀg costs
  essentially nothing beyond computing g. Population levers are built once and amortized: building d* from 60 training
  statements takes about 2 GPU-minutes (estimated: 117 s including the warm-up loops, 71 s excluding them), so the
  cost advantage matters mainly for per-prompt (item-specific)
  levers (1.19 s vs 0.037 s).
- **Hypotheses (effects Δ):** the prespecified primary paired contrast median(Δ_{Jᵀg} − ½Δ_{d*}) was positive (0.248),
  and positive for all 60 statements (0.20–0.29). The very narrow bootstrap CI ([0.247, 0.249]) describes uncertainty
  in the median, not the statement-level spread. The exploratory median(Δ_{W_sᵀg} − ½Δ_{d*}) was negative (−0.136).
- **Item-specific levers:** d*_item 1.30 and (Jᵀg)_item 0.88; the exploratory paired contrast
  median(Δ_{Jᵀg,item} − ½Δ_{d*,item}) = 0.251 was positive.
- **Linearity:** effects are linear to dose 3.125. At 12.5 the relative efficacy gaps narrow (Jᵀg/d* rises from 0.68
  to 0.82).
- **Collateral:** median KL / median Δ² is lowest for W_sᵀg. This is descriptive and on-prompt only; matched-effect
  collateral was not tested.
- **Oracle-sign correction:** with ground-truth-informed signs, the population d*, Jᵀg and W_sᵀg corrected all 27
  False-biased errors at every tested dose. This is readout control with an oracle sign, not truth inference.

**5.3 Probe-selected steering transmits the probe's verdict (retrospective diagnostic).**
- A frozen probe, trained on 90 other units, chose a binary steering sign for the 60 held-out statements, using
  already-recorded both-direction runs.
- All nine lever × dose policies exactly reproduced the probe's binary verdict: 76.7% accuracy, 0 discordant units.
- This outcome was nearly guaranteed by the design. Every flip the selector requested (39) had a baseline margin of at
  most 0.40 log-odds, while even the weakest policy (W_sᵀg at dose 1.5625) shifts every statement by 0.46–0.51. The
  diagnostic therefore could not have detected steering that combines its own information with the selector's.
- Baselines: the model's own readout with a recalibrated threshold reached 78.3% (CITIES 96.7%, COMPARE 60%), slightly
  above the probe selector.
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
  - These fractions largely track the levers' cosines with the reference. Overlap-matched random controls reproduce
    85–90% of the digit effects, and the levers exceed them modestly.
  - The cost advantage matters mainly for per-prompt levers; population levers are built once.
  - These remain *readout* levers. The truth levers' behavior is consistent with response polarity, and is not shown to
    manipulate an internal truth representation.
- **Steering transmits the sign-choosing monitor (here).** When a probe chooses the sign at the tested doses,
  answer-level steering delivers the probe's verdict. Its distinctive value, if any, would have to come from effects on
  subsequent behavior, which remain untested.
- **Recurrence and stability.**
  - In most tested prompts, late recurrence settles readouts into locally attracting, near-stationary states. The tested
    one-shot edits decayed, whereas persistent forcing maintained shifted readouts.
  - In addition prompts (about 13%), period-2 cycles trace to source-token flip modes and are largely suppressible by
    eigenvalue-guided damping, though downstream positions can keep oscillating. ARC cycles (12%) were not traced to
    sources, but global damping removed them.

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
- **What "held-out" means.** Held-out digit prompts are new operand cells within the same two prompt templates; truth
  statements are held out by unit.
- **Initialization.** One initialization per prompt in the control and mechanism experiments.
- **Multiplicity.** Headline results were selected from about 40 experiments without study-level multiplicity control.
  Confirmatory per-experiment tests were prespecified.
- **Circle interpretation.** The circle is measured in pre-coda states, so it may partly reflect alignment with the
  digit-token output geometry rather than the arithmetic itself.

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
- **Looped / recurrent-depth models.** Updated 2026-11-11 after a targeted search for earlier work
  (`results/novelty-20261111/orbits.md`; about 50 queries, not a systematic review).
  - *Orbits in Huginn are known.* Geiping et al. 2025 (arXiv:2502.05171, §7) show them qualitatively for numerical
    tokens. Blayney et al. 2026 (arXiv:2604.11791, Appendix C) count them over the GSM8K test set with an FFT-based
    classifier: orbits in 0.23–2.50% of examples (0.00–0.13% of tokens), depending on the system prompt, with
    thresholds set by eye. They report neither the period nor a mechanism and leave the explanation to future work.
    Their counts and the 13% here use different tasks, units and criteria and are not comparable.
  - *Limit cycles in equilibrium models are known* (Anil et al. 2022, arXiv:2211.09961).
  - *Flip bifurcations in recurrent attention are known.* Gao, Yang and Chen 2026 (arXiv:2609.28448) show a flip
    bifurcation to period-two motion in a minimal recurrent self-attention model with a prescribed negative value
    map (not a trained model). Kim 2026 (arXiv:2608.26556) analyses a small trained looped transformer through a
    fold and a Neimark–Sacker-type transition to bounded non-stationary motion, with normal-form scaling predicted
    from local derivatives (not only "a fold", as an earlier version of this section said). Jones 2026
    (arXiv:2610.31692) studies an external soft-token feedback loop of standard LMs, with period-doubling scaling
    and interval-arithmetic proofs of attracting periodic orbits.
  - *The damping rule is known.* Movahedi et al. 2026 (arXiv:2606.18206, §3.3 and Appendix A.3) observe oscillation
    in looped transformers and use the same relaxed update, with the eigenvalue map μ = 1 − η + ηλ and the bound
    η < 2(1 − Re λ)/|λ − 1|². The threshold a* = 2/(1 − λ) of Section 2.3 is the case of a real λ.
  - *Lyapunov exponents.* Kiwitt, Irwin, Ali and Liò (poster, ICML 2026 Mechanistic Interpretability Workshop; only
    the poster was read, by the search agent) report a negative spectrum for Huginn on 20 prompts and a positive
    leading exponent for Ouro. A contracting period-2 orbit is compatible with negative exponents.
  - *Not found:* a flip at one source position inside a pretrained recurrent-depth LM, its inheritance by later
    positions through attention, the two context-interpolation tests, and relaxation applied at that source
    position alone. That narrower combination is what this write-up adds. The relaxation acts on the whole state
    at the source position, not along an eigenvector, and stopped all motion in 15 of 26 prompts (Section 2.3).
  - Could not be opened, possibly relevant to context-controlled bifurcations: Haschke & Steil 2005 (Neurocomputing
    64), Doya 1992 (ISCAS).
- **Cyclic number representations.** Weekday and month circles in other LLMs (Engels et al. 2024, arXiv:2405.14860);
  Fourier/helical number features used for addition (Zhou et al. 2024; Kantamneni & Tegmark 2025); Fourier features in
  modular-arithmetic transformers (Nanda et al. 2023).
- **Fixed-point analysis of recurrent networks.** Sussillo & Barak 2013; Maheswaranathan et al. 2019 (line attractors in
  trained RNNs).
- **Implicit / equilibrium models.** Deep equilibrium models (Bai et al. 2019) differentiate through fixed points with
  (I − J)⁻¹. Jacobian-free and one-step "phantom" gradients (Fung et al. 2022; Geng et al. 2021) are related
  precedents for the resolvent pullback and the Jᵀg shortcut used here. Those works concern training gradients, not
  steering interventions.
- **Jacobian-based steering.** For each item, A_Tᵀg = ∇_P ΔR_T is the first-order optimal persistent steering vector:
  standard linear-response reasoning. The population d* is an average of normalized item covectors. The empirical
  findings here are its held-out accuracy, the first-step reorientation, the outside-fed readout span, and the
  effectiveness of one-step and adapter-transpose approximations.
  - Interventions along directions pulled back through a Jacobian already exist for Huginn and Ouro: Wang & Reid 2026
    (arXiv:2609.01924, §7) use Jᵥᵀu with Jᵥ a corpus-averaged Jacobian between two depths, and report that in Ouro
    later loops partly undo a single-loop write. They give no sum of powers of J and no prediction of effect size.
    In feed-forward models, FishBack (Wang et al. 2026, arXiv:2605.17231) pulls an output direction back through the
    downstream Jacobian to choose the steering direction. Search memo: `results/novelty-20261111/levers.md`.
- **Not yet done:** a baseline that directly optimizes a steering vector for the evaluated readout at norm 12.5 (where
  d* is sublinear) would test how close d* comes to the nonlinear optimum.

## 10. Reproducibility

- Code: `huginn_addition/intervene_app.py`, a Modal app with all GPU functions and configs; `analyze_*.py` and
  `make_*.py` hold the analyses and builders, with frozen rules and recorded amendments in their docstrings.
- Raw run arrays (`.npz`/`.parquet` inside run folders, ~1.4 GB) are not in git. They are on the local disk and the
  Modal volume `huginn-results`; `.gitignore` lists what is excluded.
- Run directories hold `meta.json` with config and model revision. Code hashes are recorded from the later experiments
  onward; the early pilot through cycle-source runs lack them. Push-file hashes are recorded in the ovl_eval and
  tlever_eval runs.
- Key runs:

| Result | Run directory |
|---|---|
| Circle and late-state dynamics | `collect-20261104-071439`, `intervene-20261104-072425`, `jac-20261104-164552` |
| Period-2 cycles | `orbit-20261104-193804`, `orbit64-20261104-195601`, `cycsrc-20261105-025703`, `bifur-20261105-050715`, `bifur24-20261105-113450`, `census-20261105-{162542,163100,163615,164134}`, `ownspec-20261105-182425` |
| Suppression | `relax-20261105-200208`, `rsweep-20261105-203941`, `mcq_base-20261106-01*`, `mcq_relax-20261106-02*` |
| Steering and truth (§3.1–3.3) | `steer-20261106-032913`, `truth-20261106-035631`, `tsteer-20261106-041051`, `tsteer_ans-20261106-042907`, `respond_digits-20261106-054706`, `respond_truth-20261106-055046` |
| d* | `dstar_vjp-20261106-061708`, `dseval-20261106-062151` |
| Format controls | `ctrl_formats-20261106-064804` + `ctrl_formats-20261106-070550` (two merged shards), `ctrl_cb-20261106-071755`, `ctrl_dose-20261106-070217` |
| Digit levers | `dglever_*-20261106-08*` |
| Reversal mechanism | `revmech-20261106-204036`, `rtransfer-20261106-205118`, `blockdecomp-20261106-220105` |
| Cheap levers | `adaptcov-20261106-224906`, `adlever_eval-20261106-225059`, `ovl_eval-20261106-23*` |
| Truth cheap levers | `truthcov-20261106-234613`, `tlever_eval-20261106-235208`, `-235648`, `-20261107-000141` |
| Selector | `selector_summary.json` and `selector_posthoc_probe_only.json` in the first tlever run |
