# Correctness monitors for looped LMs: what is known, what is open (2026-11-07)

Status: literature check (step 1), revised after the GPT-6 Astra (xhigh) review.
- Astra verified the Kirin, Du and Pagadala claims below from the full texts.
- Its verdict is to **narrow the question and run a capped pilot**. There is no protocol yet; that is step 2, to be
  reviewed by Astra before any GPU run.

**How each source was checked:**
- [F] full paper read by a subagent, with quotes verified against the HTML;
- [A] abstract read by me;
- [S] seen only in search results.

## Candidate question

Can we tell from a looped model's hidden state when its answer is wrong, in a way that keeps working when the
**loop budget** or the **prompt format** changes? And does that beat the model's own confidence at a fixed abstention
rate?

Why loop budget matters: it is the deployment knob of these models, since running more loops is how they spend extra
test-time compute. A monitor that breaks when the knob is turned gives false confidence exactly when someone tries
to get more out of the model.

## Closest prior work

**Correctness probes on looped models (in distribution)**
- **Kirin 2026**, [2607.18553](https://arxiv.org/abs/2607.18553) [A].
  - Model and probe: frozen Ouro (2.6B, "Ouro-RLTT"). The probe sits before the answer and predicts success.
  - GSM8K: AUROC 0.797 against 0.731 for length/log-probability features.
  - Task-disjoint Horizon Logic: +0.111. Selective prediction improves in four sealed arms.
  - Steering, loop allocation and LoRA binding are all negative ("decision-usable but not generatively
    controllable"). This matches our own "control is only as good as the monitor".
  - Not stated in the abstract: transfer across loop budgets and prompt formats, and a max-softmax baseline.
  - Astra's full-text read:
    - 0.797 is hidden-state features **plus** the shortcut features, not hidden state alone.
    - Sec. 4.4 and App. E already move frozen candidate-quality probes between loops, including on Huginn, but only
      *within fixed-budget runs*.
    - App. L tests compute allocation at total depths 1–4.
    - Nothing in it trains a correctness monitor at one total budget and tests it on answers generated at another.
    - No prompt-format transfer. No max-softmax baseline: the confidence baselines are token log-probabilities and
      length.
- **Du et al. 2025**, [2509.26314](https://arxiv.org/abs/2509.26314) [A].
  - A classifier on Huginn-3.5B's latent states predicts correctness and is claimed to generalize across domains.
  - It is used to optimize the latent reasoning, not for abstention.
  - The abstract mentions no loop-budget or format tests.
  - Astra's full-text read:
    - App. H.4 evaluates total budgets 16/24/32, but the reward model is trained on trajectories mixed across budgets,
      so a budget it never saw is not tested.
    - App. E uses a substantial trajectory Transformer, so **trajectory monitoring is already established**.
- **Pagadala 2026**, [2611.02333](https://arxiv.org/abs/2611.02333) [A + Astra full text].
  - Model: frozen Ouro-1.4B with 4 loops.
  - A stability probe for halting is compared with the native gate and with the KL between loops. Its target is
    agreement with later answers, not correctness.
  - It shows no advantage over the untrained KL.
  - Out of distribution (Social IQa) it went over budget, with a 3.26 pp accuracy loss.
  - This is a close comparator under distribution shift.

**Monitoring and inspection of looped models without correctness probes**
- **Olszowiec et al. 2026**, [2610.18127](https://arxiv.org/abs/2610.18127) [F].
  - Scope: 8 models in 4 families, using only the models' own outputs: "without external probes or additional
    training". No task-correctness labels.
  - Most tokens are decided at iteration 1. Huginn is the exception: 33% at iteration 1, 90% only by iterations 8–11
    of 16.
  - Under entry-state noise, Ouro becomes *more* confident, so confidence monitors would miss the breakdown. This is
    argued, not tested.
  - 3–11% of prompt tokens have some iteration that predicts them right while the last iteration does not.
- **Wang et al. 2026**, [2610.02741](https://arxiv.org/abs/2610.02741) [F].
  - The monitor is an LLM reading the written reasoning for a planted hint.
  - More loops lower that monitorability on some tasks for Ouro-Thinking (T = 4, 6, 8).
  - No hidden-state monitors. Huginn's written reasoning was nearly unusable for this (score 0–1%).

**Loop dynamics bearing on monitor transfer**
- **Wang & Reid 2026**, [2609.01924](https://arxiv.org/abs/2609.01924) [A]: Jacobian lens on Ouro-2.6B and Huginn.
  - Ouro rebuilds content in every loop.
  - Huginn carries content forward but acts within a window of about 2 recurrences.
  - "Verbalisability of injected content tracks per-iteration supervision". This bears on our read-space excess
    result (docs/freeloop.pdf) and **must be cited there**, as must 2606.24898.
- **Fu 2025**, [2511.08577](https://arxiv.org/abs/2511.08577) [S/A]: latent overthinking. Extra loops turn correct
  tokens wrong.
- **Luo & Yu 2026**, [2609.21383](https://arxiv.org/abs/2609.21383) [S]: in Huginn and Ouro, agreeing with the final
  answer and being correct are different properties of an iteration.
- **Popescu 2026**, [2607.20519](https://arxiv.org/abs/2607.20519) [S]: simple confidence readouts often match or beat
  learned exit gates.

**General-LLM probe pitfalls (baselines and controls)**
- Probes transfer poorly across datasets and tasks: Orgad 2024 [2410.02707](https://arxiv.org/abs/2410.02707);
  Azizian 2025 [2506.08572](https://arxiv.org/abs/2506.08572); CH-Wang 2023
  [2312.17249](https://arxiv.org/abs/2312.17249); Stacey 2026 [2604.11662](https://arxiv.org/abs/2604.11662).
- Format shifts break truth probes: Ichmoukhamedov 2025 [2505.09807](https://arxiv.org/abs/2505.09807) [S].
- Negation and surface-feature traps: Levinstein & Herrmann 2023 [2307.00175](https://arxiv.org/abs/2307.00175);
  Farquhar 2023 [2312.10029](https://arxiv.org/abs/2312.10029) [S].
- Semantic-entropy probes generalize better than accuracy probes: Kossen 2024
  [2406.15927](https://arxiv.org/abs/2406.15927).

## What is open

No paper found tests a correctness monitor for a looped model across loop budgets or prompt formats, after about
ten targeted searches plus the two full reads. Specifically open:
1. **Loop-budget transfer:** train the monitor at r loops and test at other r. That includes budgets past
   training, where Ouro gets worse and extra loops can turn right answers wrong.
2. **Prompt-format transfer** for any looped model.
3. **Comparison under shift** against the model's own confidence (max softmax probability) and against
   log-probability/length features, at fixed coverage.
4. ~~Raw state versus trajectory~~: trajectory monitoring is already established (Du, App. E). Trajectory
   statistics are now a baseline, not a contribution.
5. **"Knows more than it says" with controls.** The fair comparison is not the raw 55%. It is 78.3% for a calibrated
   readout of the model's own output against 90% for a recalibrated probe. That is 60 statements in 30 units, with
   post-hoc recalibration (README, "Label-free sign selection").
   - Before any claim it needs:
     - fresh entity/unit splits;
     - balanced negation;
     - paraphrases;
     - answer-label swaps;
     - predictors that use surface features only;
     - calibration frozen on training data only;
     - paired uncertainty at the unit level.
   - Truth classification is not yet error monitoring.

## Design priors from these papers

- **Huginn** was trained at random depth (mean 32), so the loop budget is a natural, in-support shift axis. It is
  also the model whose decisions keep changing across iterations, which gives the richest per-loop signal.
- **Ouro** was trained at 4 fixed loops and gets worse at 6. A budget shift there means leaving the training support.
  Use the `Thinking` and base variants with care.
- **Tasks must be ones the model can do**, such as GSM8K-level arithmetic, our digit tasks and multiple choice where
  accuracy is above chance. Correctness labels must not be guessable from surface features.

## Realistic ceiling (Astra)

- A careful benchmark could support a workshop or Findings-style paper.
- A main-conference paper would need a substantial, replicated failure, or a mechanism that predicts transfer, beyond
  "probes dislike distribution shift".
- Format transfer alone is less distinctive than loop-budget transfer.

## Novelty risk and what would make this more than incremental

- The in-distribution version is done (Kirin; Du). The contribution has to be the **shift test** plus the
  **comparison at fixed coverage against the model's own confidence**.
- Our specific angle is a mechanism. A probe direction m reads q_r = m·s_r. Near a fixed point J_*, with
  Δs_r = s_{r+1} − s_r (Astra's correction):

  q_{r+k} − q_r ≈ Σ_{j<k} ⟨(J_*ᵀ)ʲ m, Δs_r⟩.

  - The pullback chain needs the displacement Δs_r: chain norms alone cannot predict the drift.
  - The drift of the score does not determine how well the monitor separates right from wrong answers, nor its
    calibration.
- Huginn at small r and Ouro's 4 loops are not near a fixed point. There the testable version is a **short-horizon
  local linear forecast**. Check its error against two simple forecasts: no change, and linear extrapolation.
- The forecast must use the full relevant recurrent state. A Jacobian of the own position alone misses the prefix
  states, which are also changing.
- Pre-answer states must be measured directly, and continuations generated at different budgets can diverge.
- So this is a hypothesis to test, not an established transfer theorem.

## Protocol decisions (from Astra's review; to be fixed in step 2)

1. **Models and tasks.**
   - Huginn first, on short-answer arithmetic plus one multiple-choice task that is not arithmetic.
   - Add Ouro only after feasibility is shown.
   - Avoid tasks with almost no errors.
2. **Budgets and formats.**
   - Train the monitor at Huginn r = 32 and test at 8/16/32/64.
   - Later, Ouro at 2/4/6, labelling 6 as extrapolation.
   - Cross budget and format independently, on paired questions.
3. **Extraction and labels.**
   - Freeze one precisely defined pre-answer extraction point.
   - Label each budget's own actual answer.
   - Split every variant by its underlying problem, and control the random initial state.
4. **Baselines and metric.**
   - Linear probe against: answer-option confidence; margin/entropy; statistics of trajectory change; probes trained
     on mixed budgets; probes refitted at the target budget.
   - Report error at fixed coverage, **and** the coverage and error actually achieved with thresholds frozen at the
     source budget.
5. **Minimum effect and stop rule.**
   - Provisional minimum effect: 3 pp of accepted-answer error at 80% coverage.
   - Stop after the capped pilot if monitoring at the source budget fails or the affordable precision is too low.
   - Otherwise run one sealed evaluation and keep null results.
