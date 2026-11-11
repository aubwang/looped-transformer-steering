# Prior-work memo: sustained steering and score levers

Search date and cutoff: 2026-11-11. I read the requested portions of `docs/steering_results.typ` and `huginn_addition/WRITEUP.md`. This is an adversarial prior-work search, not a certification of novelty. “Not found” below means not found after the queries and citation follow-ups recorded here, not proof of absence. Search indexing and abstract-only screening leave gaps; no exhaustive forward-citation database export was available in this search.

## 1. Verdicts

- **A1 — CLOSE PRIOR WORK.** The mathematics is already published in general form: Bai, Kolter and Koltun (2019), §3.1/Theorem 1 and Appendix A, give equilibrium sensitivity; Geng et al. (2021), §2.4, equations (10)–(17), give unrolled sensitivity and its Neumann-series limit. I did not find the exact combined empirical result—last-position constant forcing, a frozen settled-state Jacobian, and quantitative finite-dose prediction in both specified pretrained looped LMs—but Wang and Reid (2026), §7, already combine Jacobian-derived directions with repeated interventions in Huginn/Ouro.
- **A2 — CLOSE PRIOR WORK.** The optimal Euclidean direction is an ordinary pulled-back gradient: FishBack (Wang et al., 2026), §3 equation (4), explicitly constructs the downstream Jacobian-transpose covector for activation steering, and Wang and Reid, §7, already intervene along transported token directions in looped LMs. Prediction/control dissociation and distinct recurrent input/readout directions have substantial precedents; I did not find the particular near-zero cosine/effect ratios and simultaneous lever norm growth reported for these two models.
- **A3 — NOT FOUND as this exact displayed identity.** No opened paper states the same three normalized ratios together; nevertheless it follows immediately from defining the two unit vectors, so absence of a matching display is not a meaningful mathematical novelty claim. It is exact per item in the linearization, not generally for ratios of population-averaged measured effects or for averaged normalized directions.

**Convention that matters:** the draft implements `x[t+1] = F(x[t] + P)`, so the injected vector is passed through a loop before the next readout and the sum starts at J. An addition after the final loop followed by immediate readout contributes an identity term instead; describe the operational ordering explicitly. Specializing the general input derivative to `F_P(x)=F(x+P)` gives `D_P F_P=J`, hence `(I-J)^(-1)J` and its transpose pullback. The extra J is a chain-rule consequence, not a new sensitivity principle.

## 2. Closest ten papers, most damaging first

### 1. Jacobian-based interventions already performed in the same model family

**Wenlong Wang and Fergal Reid (2026), “Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?” [arXiv:2609.01924](https://arxiv.org/abs/2609.01924). Read: FULL TEXT, §§2–5 and 7–8, with Appendix B protocol and references.**

§7 explicitly uses normalized directions `J_v^T u_t`, with `J_v` a fitted mean downstream Jacobian, for causal interventions. §7.1 reports that Ouro writes spanning remaining loops outperform single-loop writes; subsequent computation can undo a successful early intervention. Huginn writes have an approximately two-recurrence valid transport window. This is direct prior art against either “first Jacobian-aware steering in a looped LM” or “first observation that later loops erase an intervention.”

Difference: its Jacobian is a transport lens to a target depth, not the settled one-loop Jacobian summed into a constant-forcing resolvent. It uses projection clamps across windows, not the same fixed additive vector at every loop, and does not establish the draft’s finite-dose response matrix or gradient/lever cosine. The Huginn horizon is also explicitly an instrument-coordinate limitation, not proof of global state contraction.

### 2. The equilibrium derivative itself

**Shaojie Bai, J. Zico Kolter and Vladlen Koltun (2019), “Deep Equilibrium Models.” [arXiv:1909.01377](https://arxiv.org/abs/1909.01377). Read: FULL TEXT, §3.1, Theorem 1, equations (8)–(11), and Appendix A.**

The theorem differentiates a fixed-point readout with respect to inputs as well as parameters, using the inverse residual Jacobian; the backward computation solves the corresponding adjoint linear system. It includes transformer language models, so this is not merely a distant numerical-analysis analogy.

Difference: no inference-time steering study of Huginn, no specific persistent state-addition protocol, and no claimed finite-dose or near-orthogonality measurements. Nevertheless the resolvent and equilibrium adjoint direction in A1/A2 are direct specializations of its theorem.

### 3. The truncated sum and cheap backward approximations

**Zhengyang Geng, Xin-Yu Zhang, Shaojie Bai, Yisen Wang and Zhouchen Lin (2021; revised 2022), “On Training Implicit Models.” [arXiv:2111.05177](https://arxiv.org/abs/2111.05177). Read: FULL TEXT, §§2.1–2.5 and §4.**

§2.1 explicitly treats derivatives with respect to input and parameters in the same framework. §2.4 gives the finite unrolling sum, the Neumann expansion, and their equivalence at an exact equilibrium; it also gives damped and truncated approximations. Substituting the persistent-push input map into this construction yields the draft’s sum of powers and adjoint.

Difference: phantom gradients are proposed and evaluated for training implicit models, rather than steering a frozen LM. Short truncations working well are therefore established methodology; their measured accuracy and direction alignment on the draft’s tasks remain separate empirical questions.

### 4. An explicit downstream-gradient steering construction

**Sihan Wang, Jiayi Zhao, Qingyan Cao, Hongbo Yao and Lin Shu (2026), “FishBack: Pullback Fisher Geometry for Optimal Activation Steering in Transformers.” [arXiv:2605.17231](https://arxiv.org/abs/2605.17231). Read: FULL TEXT, v1 §§2–4 and v2 §§3–5.**

In v2 §3 equation (4), `q=J^T beta_W` is the concept covector pulled back through the remaining network. §4 solves for an optimal intervention under an output-distribution metric; the Euclidean special case follows the pulled-back gradient. Thus downstream-effect-aware steering and separating a readout direction from an intervention direction are explicit prior work.

Difference: feed-forward downstream transport, generally Fisher/KL cost rather than fixed Euclidean push length, and no tied recurrence, constant forcing, or settled-state resolvent. “Optimal” is relative to its specified local objective, as it is in the draft.

### 5. Linear response, distributed forcing, and adjoint patch selection in transformers

**David N. Olivieri and Antonio F. Pérez Rodríguez (2026), “Transformer Field Theory: A Response-Theoretic Approach to Mechanistic Interpretability” (v1 title: “Continuous-Depth Field Theory for Transformer Patching and Mechanistic Interpretability”). [arXiv:2605.25225](https://arxiv.org/abs/2605.25225). Read: FULL TEXT, v2 §§I–V; abstract/contribution statements describing empirical §§VIII–XIII.**

§III permits interventions supported over an interval of layers; §IV linearizes their propagation and expresses scalar effects as a sensitivity/source pairing; §V formulates an adjoint inverse problem for selecting patches. The paper reports bounded local linearity and empirical sensitivity predictions in GPT-2-style transformers. This is close prior art for presenting small activation pushes through a response operator as a research framework.

Difference: depth-dependent feed-forward trajectories, not a stationary repeated block; no demonstrated `J+...+J^T` persistent-push response on Huginn/Recurrent-Llama. Its patch-inference formulation should not be mistaken for a completed empirical optimal-control result: the stated empirical contribution tests forward response objects.

### 6. Decodable latent content with weak behavioral steering

**Gaoxiang Huang and Lei Qi (2026), “When Steering Fails in Latent Reasoning: A Latent-to-Language Transition Gap.” [arXiv:2609.21662](https://arxiv.org/abs/2609.21662). Read: FULL TEXT, §§1–4.**

§2 constructs contrastive-mean activation additions; §3 finds task information remains identifiable even when interventions produce weak language-generation changes. It compares explicit-CoT and COCONUT-style latent-CoT variants and distinguishes representation displacement from downstream effect. This directly anticipates the broad “predictive directions need not control latent reasoning” motivation.

Difference: COCONUT-style token feedback and a latent-to-language boundary, not a settled recurrent-depth state. No resolvent steering, readout-gradient comparison, or quantitative explanation by near-orthogonal adjoint directions is supplied.

### 7. Probe-guided interventions that barely affect continuous thoughts

**Yuyi Zhang, Boyu Tang, Tianjie Ju, Sufeng Duan and Gongshen Liu (2025; revised 2026), “Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought.” [arXiv:2512.21711](https://arxiv.org/abs/2512.21711). Read: FULL TEXT, v2 §§3–4; ABSTRACT for the remaining claims.**

§4 trains linear concept probes, constructs minimal representation perturbations that change probe classification, and also swaps latent tokens. It reports low behavioral sensitivity of COCONUT tokens relative to explicit CoT. This is especially close to the earlier write-up’s diagnostic of probe success combined with intervention failure.

Difference: it does not compute a downstream readout adjoint or a recurrent fixed-point response. Its stronger interpretation that weak steering implies deficient reasoning is not established by the draft and should not be imported: weak control along selected directions alone cannot establish causal irrelevance of the entire state.

### 8. Non-normal recurrence separates optimal input and readout

**Giulio Bondanelli and Srdjan Ostojic (2020), “Coding with transient trajectories in recurrent neural networks.” [PLOS Computational Biology 16(2):e1007655](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007655). Read: FULL TEXT, Results on transient amplification, singular-vector coding, low-rank connectivity; Methods on the propagator and its SVD.**

Its propagator’s right singular vectors specify amplified inputs, while left singular vectors specify readouts. Low-rank non-normal networks map an input direction into a distinct, potentially nearly orthogonal output direction, while stable dynamics can amplify transients. This substantially weakens any general claim that readout/control misalignment or local amplification despite eventual settling is a newly discovered recurrent-network phenomenon.

Difference: initial-condition perturbations and transient output coding, not constant forcing and terminal scalar-score optimization in a language model. A measured `||J L||>||L||` alone neither establishes the same low-rank mechanism nor proves non-normality; it also does not contradict spectral stability.

### 9. Persistent input perturbations in DEQs are already an adversarial-control setting

**Zonghan Yang, Peng Li, Tianyu Pang and Yang Liu (2023), “Improving Adversarial Robustness of Deep Equilibrium Models with Explicit Regulations Along the Neural Dynamics.” [ICML/PMLR 202](https://proceedings.mlr.press/v202/yang23i.html). Read: FULL TEXT, §§2–4 and associated references.**

§3.1 explicitly compares clean and perturbed trajectories under an input supplied at every iteration, separating new input perturbation from accumulated state displacement. Subsequent sections optimize/regulate those trajectories and evaluate intermediate-state gradient attacks. This blocks a broad claim that persistent perturbations through an equilibrium neural computation are unexplored.

Difference: perturbed image inputs and robustness training, not a fixed last-token state vector optimized for a selected language-model score. Its immediate predecessor is Zonghan Yang, Tianyu Pang and Yang Liu, **“A Closer Look at the Adversarial Robustness of Deep Equilibrium Models” (NeurIPS 2022; arXiv posting 2023), [arXiv:2306.01429](https://arxiv.org/abs/2306.01429), ABSTRACT only**, which warns that mismatched forward/backward solvers can give misleading robustness gradients.

### 10. Predictive accuracy and intervention efficacy were already separated

**Samuel Marks and Max Tegmark (2023; revised 2024), “The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets.” [arXiv:2310.06824](https://arxiv.org/abs/2310.06824). Read: FULL TEXT, §§5–6 and discussion.**

§6.2 finds that mass-mean probes produce more effective interventions than alternative probes despite similar classification accuracy. This is direct prior evidence that finding a predictive linear decoder is not the same optimization problem as finding an effective causal intervention.

Difference: feed-forward Llama-2 models; no repeated-loop attenuation or response-operator explanation. Its mass-mean success also means the draft’s mass-mean failure should be stated as model/task/protocol specific, not a general refutation of that method.

### Other verified checks and citation follow-ups

These are secondary to the ten above, but establish what else was actually checked. Except where specified, the reading level here is **ABSTRACT only**; no absent result is inferred from an abstract alone.

- **Hugh Blayney, Álvaro Arroyo, Johan Obando-Ceron, Pablo Samuel Castro, Aaron Courville, Michael M. Bronstein and Xiaowen Dong (2026), “A Mechanistic Analysis of Looped Reasoning Language Models,” [arXiv:2604.11791](https://arxiv.org/abs/2604.11791). FULL TEXT: §§2–5 and conclusions.** Establishes cyclic layerwise limiting states and examines input injection/normalization, including retrofitted models. The inspected sections do not construct persistent-push adjoints.
- **Shaojie Bai, Vladlen Koltun and J. Zico Kolter (2021), “Stabilizing Equilibrium Models by Jacobian Regularization,” [arXiv:2106.14342](https://arxiv.org/abs/2106.14342).** Directly regularizes the fixed-point Jacobian to improve forward/backward stability; an established stability context, not evidence for the reported direction-specific ratios.
- **Jiaju Wu, Yi Hu and Muhan Zhang (2026), “Shared Weights, Selected Computations: How Looped Transformers Route What Each Loop Does,” [arXiv:2609.39892](https://arxiv.org/abs/2609.39892).** Learns a linear map to steer a frozen loop toward graph transitions and uses activation patching. Its map named J is learned, not the Jacobian in A1.
- **Cyprien Tamekue, Ruiqi Chen and ShiNung Ching (2024; revised 2025), “On the control of recurrent neural networks using constant inputs,” [arXiv:2410.17199](https://arxiv.org/abs/2410.17199).** Constant/piecewise-constant control of nonlinear continuous-time RNNs using Jacobian-based algebraic approximations. Relevant control precedent; not the LM experiment.
- **Jiaqian Li, Yanshu Li and Kuan-Hao Huang (2026), “Steering Vector Fields for Context-Aware Inference-Time Control in Large Language Models,” [arXiv:2602.01654](https://arxiv.org/abs/2602.01654).** Uses a learned differentiable concept score’s local gradient for context-dependent steering. This is not automatically the gradient of the actual downstream score through recurrent dynamics.
- **Luyang Zhang, Minxing Zheng and Jialu Wang (2026), “Correcting Attribution Patching with a Few Exact Measurements,” [arXiv:2610.27393](https://arxiv.org/abs/2610.27393).** Documents errors in gradient-based intervention predictions and calibrates them with exact measurements; relevant to finite-dose validation, not the constant-loop formula.
- **Hengyi Zhou, Haobo Fang, Xiaonan Huang, Dezhong Tong and M. Khalid Jawed (2026), “Self-Consistent Neural Control: Learning Trajectories for Implicit Equilibrium,” [arXiv:2611.03885](https://arxiv.org/abs/2611.03885), submitted November 2.** Recent adjoint control of physical equilibrium systems and sensitivity variation; not LM activation steering.
- **Jose Luis Lima de Jesus Silva (2026), “Response Renormalization for Critical Deep Equilibrium Models,” [arXiv:2608.23725](https://arxiv.org/abs/2608.23725).** Near-critical adjoint amplification and selected response channels for reliable training; no LM steering experiment established by the abstract.
- **Peter Jemley (2026), “Continuous-Depth Transformers with Learned Control Dynamics,” [arXiv:2601.10007](https://arxiv.org/abs/2601.10007).** Trained control inputs in a neural-ODE middle block; different from post-hoc control of a frozen settled recurrence.

Citation chasing: Wang/Reid’s references led to the original Jacobian-lens study, **Wes Gurnee, Nicholas Sofroniew, Adam Pearce, Mateusz Piotrowski, Isaac Kauvar, Runjin Chen, Anna Soligo, Paul Bogdan, Euan Ong, Rowan Wang, Ben Thompson, David Abrahams, Subhash Kantamneni, Emmanuel Ameisen, Joshua Batson and Jack Lindsey (2026), “Verbalizable Representations Form a Global Workspace in Language Models,” [arXiv:2607.15495](https://arxiv.org/abs/2607.15495), ABSTRACT only**. It also led to **Wenquan Lu, Yuechuan Yang, Kyle Lee, Yanshu Li and Enqi Liu (2025), “Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer,” [arXiv:2507.02199](https://arxiv.org/abs/2507.02199), ABSTRACT only**, reporting Huginn decoding inconsistencies. Yang et al.’s references led to the 2022 robustness paper above; Geng’s §4 and derivation confirmed the DEQ/input-gradient lineage. These are backward citation follow-ups, supplemented by the forward-looking model-ID queries below, not an exhaustive citation census.

The requested foundational pages were also opened: **Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini and Monte MacDiarmid (2023; revised 2024), “Steering Language Models With Activation Engineering,” [arXiv:2308.10248](https://arxiv.org/abs/2308.10248), ABSTRACT only**; **Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister and Martin Wattenberg (2023; revised 2024), “Inference-Time Intervention: Eliciting Truthful Answers from a Language Model,” [arXiv:2306.03341](https://arxiv.org/abs/2306.03341), ABSTRACT only**; and **Andy Zou and 20 coauthors (2023; revised 2025; full author list on the opened page), “Representation Engineering: A Top-Down Approach to AI Transparency,” [arXiv:2310.01405](https://arxiv.org/abs/2310.01405), ABSTRACT only**. They establish activation control as background; the stronger novelty objections come from the full-text items above.

## 3. Unverified leads

- **“Power Steering: Behavior Steering via Layer-to-Layer Jacobian Singular Vectors,” Omar Ayyub, February 2026.** Search returned a technical blog, not an opened arXiv/publisher paper. Potentially relevant to downstream Jacobian steering; not used as evidence.
- **“How Much Gradient Does Truncated Backprop Lose in Looped Transformers?”** A search snippet associated this title with 2604.08764. Opening that arXiv identifier instead produced *Revisiting Anisotropy in Language Transformers: The Geometry of Learning Dynamics*. The claimed title/identifier association is unverified and must not be cited as a paper.
- **“Interpreting Latent CoT Reasoning as Dynamical Systems” (search lead 2607.09698)** and **“Invariant Reasoning Directions in Latent Trajectories of Language Models.”** Secondary search results suggested potentially relevant dynamics/intervention work; primary pages were not opened, so no findings are attributed to them.
- An **Ouro CAA steering experiment repository** surfaced in searches, describing stronger effects in later loops. It is not an opened paper and is excluded from publication-priority judgments.

No paper used substantively in the verdict rests solely on a search snippet. Conversely, an opened abstract is only abstract-level verification, not a full-text negative search.

## 4. Queries run

The following 46 distinct web queries were run. Quotation marks below reproduce query constraints; all query terms were technical terms or literature names/identifiers.

1. `"Huginn" "steering"`
2. `"looped" "steering" transformer Jacobian`
3. `"2609.01924"`
4. `"2604.11791"`
5. `deep equilibrium adversarial robustness input gradients fixed point inverse Jacobian`
6. `Geng 2021 phantom gradients training implicit models`
7. `Jacobian regularization deep equilibrium models Bai 2021`
8. `gradient based activation steering downstream Jacobian 2025 2026`
9. `"Coconut" "steering" latent reasoning`
10. `"Ouro" "activation" "steering"`
11. `"Huginn" "interventions" 2026`
12. `"looped transformer" "resolvent"`
13. `"probe directions" "steering" gradient`
14. `"predict" "control" "steering" "Jacobian"`
15. `recurrent neural network nonnormal optimal input readout amplification`
16. `controllability observability Gramian recurrent neural networks stimulation`
17. `"Training Implicit Models" "Geng" phantom`
18. `"Stabilizing Equilibrium Models" Jacobian`
19. `"steering" "gradient" "backpropagation" 2025 arxiv`
20. `"probe" "causal" "steering" "2026" gradient`
21. `"2502.05171" "steering" 2026`
22. `"Scaling Latent Reasoning via Looped Language Models" interventions`
23. `"deep equilibrium" "adjoint" "sensitivity"`
24. `"activation" "linear response" "steering"`
25. `"activation steering" "Jacobian" arxiv gradient`
26. `"attribution patching" gradients interventions paper`
27. `"Coding with transient trajectories" authors`
28. `"steering" "deep equilibrium" adversarial`
29. `"Huginn" "steering" "2025" "2026" arxiv`
30. `"Ouro" "Jacobian" "steering" 2026`
31. `"steering" "universal transformer" interventions`
32. `"probe directions" "do not" "steer"`
33. `Yang Pang Liu 2022 deep equilibrium robustness gradient attacks`
34. `"2609.01924" steering Jacobian -site:themoonlight.io`
35. `"2502.05171" "intervention" "2026" arxiv`
36. `"2611" "looped" "steering"`
37. `"sustained" "steering" "Huginn"`
38. `"resolvent" "steering" language model`
39. `"adjoint" "steering" "transformer"`
40. `"gradient" "probe" "orthogonal" "looped"`
41. `"Steering Vector Fields" gradient scoring arxiv`
42. `"Huginn" "resolvent"`
43. `"looped" "sustained" "gradient" steering`
44. `"2607.15495" "steering" Jacobian`
45. `optimal stimulation readout direction cosine adjoint reciprocity identity`
46. `"steering" "gradient alignment" "cosine" "linear response"`

## 5. Likely referee assessment

The resolvent, its truncated expansion, and the maximum-score push are standard differentiation and Euclidean optimization, already concretely developed in the DEQ literature. Jacobian-derived intervention directions in looped LMs, erasure of early writes, and decodable-but-weakly-controllable latent directions also have published precedents, particularly Wang/Reid and the 2026 latent-steering papers. The defensible contribution is the narrowly specified empirical result: quantitative held-out prediction under finite constant forcing and the measured readout/adjoint geometry in these two pretrained models, provided numerical linearization and averaging caveats are retained. The three-ratio proposition usefully prevents counting one geometric observation three times, but is a bookkeeping identity rather than an independent theoretical discovery.
