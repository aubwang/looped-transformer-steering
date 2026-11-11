# Second, independent reading of the project (Gemini 4 Carbon, 2026-11-10)

Reviewer: Gemini 4 Carbon (via OpenCode, effort "high"), given the repository read-only and asked for a critical
second opinion on what was accomplished, whether to stop, and whether the candidate next steps are worth their
cost. It reviewed Draft 2 of `docs/steering_results.typ`. Its answer is reproduced below as returned. The
literature references and the citation `arXiv:2609.01924` are the reviewer's and were not checked. What was done
in response is listed at the end.

---

### 1. What has actually been accomplished?

**Mathematical tautology vs. empirical content.** Propositions 1–2 of `docs/steering_results.typ` are near-tautologies at infinitesimal dose: for any $C^2$ map at a fixed point $x^* = F(x^*; y^*)$, Taylor's theorem forces $\Delta z_T = G A_T P + O_T(\|P\|^2)$ (`r = 0.9997, err = 2.6%` at L32's 1% dose is essentially a smoothness and harness check), and Cauchy–Schwarz forces $A_T^\top G_k^\top$ to maximize that linear form. Furthermore, because $s_{t+1} = R(W_s s_t + W_e e)$, a sustained state push $P$ is equivalent to a static context perturbation ($W_e \delta e = W_s P$; `huginn_addition/WRITEUP.md` §1). Thus $A_T^\top g$ is simply unrolled backpropagation to a shared input—mathematically the same resolvent/`1`-step ("phantom") gradient standard in Deep Equilibrium Models (Bai et al. 2019; Geng et al. 2021), applied to inputs rather than weights. Testing $G A_T P$ *only along the levers* also flatters linearity by maximizing signal-to-curvature; on random pushes at moderate KL, linear predictions fail (`docs/oversight_results.typ` Part C: median errors 0.42–0.88).

**What is genuinely empirical and solid:**
- **Single-step read-space suppression:** In both Huginn and L32, $J^\top$ rotates readout gradients nearly $90^\circ$ in one step (`cos(L, U) = −0.005` on L32; gradient-push target effect `−0.001` vs. lever `+0.165`) without contracting perturbations overall (`||JP||/||P|| = 1.17` in Huginn). A symmetric contractive $J$ cannot do this (`docs/analytic.typ` Prop. 2c, Cor. 4c). In Huginn this extends across 1,440 non-task read directions, whereas Ouro-1.4B shows the *opposite* sign (`+0.022`; `docs/freeloop.typ`).
- **Finite-dose & multi-pass locality:** Linear response holds quantitatively up to 3–9% state norm on L32 despite unfrozen prefix states and population-averaged directions; passes 2–4 contribute ~25% of the response matrix (`E1`), settling by pass 8.
- **Huginn period-2 bifurcations (`huginn_addition/WRITEUP.md` §2.3):** Tracing ~13% addition-prompt 2-cycles to upstream own-block eigenvalues crossing $-1$, with supercritical flip scaling and targeted damping (24/26 source blinks stopped), goes substantively beyond qualitative orbit reports (Geiping 2025; Blayney 2026).
- **Matched-pair monitoring null (`docs/monitor_results.typ`, `docs/followup_results.md`):** Across 6,000 problems, linear probes tie (`0.687` L32 vs. `0.685` D), lose to 4-feature output confidence (`0.743`), add only `0.003–0.006` AUROC, and cross-predict each other's errors (`0.680`). The 18-family screen explains why: on one-token prompts L32 settles by loop 8 or flips wrong-to-wrong (116/133).

**What is weaker or overclaimed:** Calling `+0.165` centered-logit nudges on already-solved sums "steering" overstates behavioral control: Huginn digit pushes flipped only 8/50 answers (12/50 to unintended digits), truth `d*` is confounded with affirmative polarity, and probe-guided steering added zero accuracy (`huginn_addition/WRITEUP.md` §§3.1, 3.4, 5.3). The theory notes assemble classical tools (Schur complements, Haar conditioning, Kalman Gramians, moment log-convexity), though `docs/monitor_theory.typ` Thm 2 nicely explains why step-ratio bounds systematically underpredict.

**Rating:** **Solidity: 9/10** (pre-registration, float64/adjoint gates, independent recomputations, honest nulls). **Novelty: 5/10** (strong workshop/TMLR mechanistic note; not a headline ML result).

---

### 2. Is this a sensible point to stop?

**For GPU experiments: yes, stop.** Both tracks hit pre-registered stopping criteria with clean answers.

**Write-up gaps to fix first (zero GPU):**
1. `docs/steering_results.typ` omits the repo's own **Ouro-1.4B opposite-sign result** (`docs/freeloop.typ`). Huginn and L32 share the Raven architecture (prelude, linear adapter re-injection, looped core, nonlinear coda, random-depth terminal loss); omitting Ouro makes readout suppression look universal across looped LMs when your own data show it is not.
2. Present Propositions 1–2 upfront as classical sensitivity calculus so the real empirical finding—non-normal read-space suppression—is not oversold as "linear response works."
3. Rework or cut the Section 4 "supervision hypothesis" box: it conflates *adjoint sensitivity* ($A_T^\top g$: what perturbations reach future logits) with *forward class separability* (where right/wrong states already differ)—the exact vector-vs.-covector distinction `docs/primer.typ` §1 warns about.

---

### 3. Assessment of candidate next steps

- **a. Second, non-arithmetic steering task on L32 (~15 GPU-min + cycle):** *Low gain.* Taylor guarantees small-dose `FULL` accuracy around any settled state, and Huginn already demonstrated lever efficacy on truth (`huginn_addition/WRITEUP.md` §3.4) and read-space-wide suppression across 1,440 random vocabulary contrasts (`docs/freeloop.typ`). **P(changes conclusions): ~10%.**
- **b. Section 4 "supervision hypothesis" (pilot on saved `results/`):**
  - *Saved-data check (verified from directory listings, `mon_app.py`/`llever_app.py` schemas, and `meta.json`, without executing `.npz` loaders):* `results/mon_main-20261110-*` locally holds all 15 shards' `.npz` chunks (`x_last` at 12–15 loop counts, `post`, `pre`, `x_prev`/`h_prev`, readout summaries), **but stores no L32 weights, readout gradients `G`, or VJP chains `J^\top G` / `A_T^\top G`** (`results/weights-20261107-145443` is Huginn only; `llever_*.npz` covers sums `70..79` only, matching <4% of `add3`/`add4` and 0% of `add3w`/`arc_easy`; `scores.npz` omits probe weights). Zero-GPU work can test only trajectory/PCA/logit-lens proxies, not true lever subspaces.
  - *Conceptual flaw:* `M1-a` and `M3-a` *already* fitted unrestricted ridge probes across all of $\mathbb{R}^{2048}$ at loops 1–32 (never restricted to $G$) and plateaued at `0.68–0.69`, below confidence (`0.743`). At settled loop 32, signals carried forward have already reached the logits; at early loops, $G A_T x_r$ forecasts *L32's eventual token*, not *ground-truth correctness*. **P(changes conclusions): ~5%.**
- **c. Mihara-bot pair verification + monitor rerun (~80 GPU-min):** *Poor ROI.* `docs/pairs_check.md` (line 9) already shows a 2× parameter confound (`517M` `4x5` looped vs. `1,007M` non-looped), unverified training parity, only 5 loops (breaking the 32-loop transfer grid), and nearly 3× your per-run GPU limit—likely just to reconfirm that confidence beats linear probes on one-token tasks. **P(changes conclusions): <10%.**
- **d. Better uses of effort:**
  1. **Zero-GPU synthesis + theory:** Fold Ouro (`docs/freeloop.typ`) into `docs/steering_results.typ`, and study analytically *why* random-depth terminal-loss training (vs. per-iteration loss, cf. Wang & Reid `arXiv:2609.01924`) drives $\operatorname{tr}(P_{\text{read}} J)/r < \operatorname{tr}(J)/d$.
  2. **Zero-GPU loose-end check (~15 CPU-min):** `mon_main` already saved `x_prev` and `h_prev` at position `−2` for all 6,000 items. Running the 2×2 cross-label table (`LL, LD, DL, DD`) on saved position `−2` states tests whether `docs/monitor_results.typ` §7.4's `+0.02–0.04` gap is a state-column effect or a label-difficulty artifact.
  3. **If spending ≤5 GPU-min once:** Measure `docs/freeloop.typ`'s `C3` read-space self-gain excess on L32 (`llever` harness) to test whether L32 matches Huginn (`< 0`) against Ouro (`> 0`).

---

### 4. Overall recommendation

**Stop GPU collection.** Skip **3a**, GPU **3b**, and **3c**.
1. **First (0 GPU):** Revise `docs/steering_results.typ` to include the Ouro contrast, separate Taylor's theorem from non-normal read-space suppression, and deflate the §4 supervision box.
2. **Second (0 GPU, optional):** Run the exploratory 2×2 cross-label table on already-saved `mon_main` position `−2` arrays (`x_prev`/`h_prev`); if the column contrast is near zero, close the monitoring thread completely.
3. **Only if you want one final ≤5 GPU-min run:** Compute L32's `C3` read-space excess for a 3-model comparison (Huginn, L32, Ouro), then stop regardless of sign and pivot to pure theory if still curious.

---

## What was done in response (Draft 3 of `docs/steering_results.typ`)

1. **Ouro contrast added** (new section "Is the readout special, and is this general? Three models"); the claims
   are limited to the one design the two steered models share. Checked in the repository: `docs/freeloop.typ`
   reports the opposite sign for Ouro-1.4B, and Draft 2 did not mention it.
2. **Propositions labelled as standard calculus** in the summary and in Section 1, with the three things that had
   to be measured listed. Only the deep-equilibrium reference (Bai, Kolter and Koltun 2019) is cited; the
   "phantom gradient" reference was not checked and is not cited.
3. **Supervision box removed** and replaced by a paragraph giving the two reasons not to expect a gain (a lever
   is a sensitivity, not a separating direction; a reader restricted to levers is a special case of the linear
   readers already fitted).
4. **"Steering" toned down:** defined up front as moving scores; the counts of changed answers on L32, computed
   from the saved outputs, are reported (no lever push at any dose made the pushed candidate the answer).
5. **The ≤5 GPU-minute measurement of the read-space self-gain on L32 was run** (`docs/lread_design.md`), the
   reviewer's option 4.3. Options 3a, 3b, 3c and the cross-label table (d.2) were not pursued.
6. The reviewer's statement that linear predictions "fail" for random pushes refers to an earlier pilot with a
   different kind of push; the Limits section now cites that pilot with its numbers for L32 (0.14 and 0.42).
