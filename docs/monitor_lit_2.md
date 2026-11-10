# Monitoring literature extension — 2026-11-09

Read `monitoring_gap.md` first. This memo adds findings rather than reproducing its summaries. Scope: 24 targeted search queries, plus opened primary papers; negative findings mean “not found,” not proof of absence. References below were opened at least at abstract level; detailed claims use full text unless marked **abstract only**.

## Verdicts and overlap

**Q1 / M1 — appears open for the matched-pair comparison.** No located paper compares correctness/error probes against output confidence in a recurrent LM and its matched feedforward partner on identical problems. Kirin [1] remains the closest correctness-monitor precedent, already covered in the earlier memo. Blayney [2] compares recurrent and feedforward internal computation, but measures mechanistic stages, not error detection. Thus correctness probing itself is established; architectural attribution with the proposed matched partner is the opening.

**Q2 / M3 — partly done.** Rechecking Kirin [1] confirms frozen cross-loop candidate-quality readouts on Ouro and Huginn: first-pass Ouro readouts lose performance at loop 4, whereas loop-2/3 readouts transfer approximately at parity. This is real cross-iteration transfer, not merely separately fitted probes. Missing: a paired feedforward cross-layer matrix and training at total budget 32 then testing answers generated at 8/16/64. A standard-transformer comparator is Zacharopoulos [3]: late-layer syntactic probes transferred to earlier layers become below chance, with coefficient sign reversal (**abstract only**). Neither its task nor its models provide our matched baseline.

**Q3 / M4 — partly done; “label-free dynamics predict correctness” is not new broadly.** Geiping [4] uses successive-output KL below 5×10⁻⁴ for per-token adaptive exit and reports exit-depth distributions across MMLU subjects; mathematical tokens exit sooner than moral-scenario/logical-fallacy tokens. This evaluates adaptive computation, not residual-conditioned error AUROC beyond confidence. Zhu/Ouro [5] learns entropy-regularized exit probabilities and evaluates compute–performance allocation, rather than treating fixed-point residual as a correctness certificate. Bansal [6] measures accuracy versus recurrent iterations and problem size on prefix sums, mazes and chess; recall/progressive training prevents overthinking, but is not a calibrated uncertainty detector.

Anil [7] explicitly plots per-instance multi-initialization latent alignment against correctness on prefix sums, including OOD cases. Its residual/accuracy tables and solver experiments also separate convergence from path independence: accurate models can approach limit cycles. Missing for our M4: incremental predictive value beyond confidence on matched LMs, especially a period-2 feature. Answer-stabilization monitoring is already covered by the earlier memo; do not claim novelty for that feature alone.

**Q4 / Theory — partly done mathematically; monitor-budget application appears open.** Chen [8] certifies monDEQ input robustness, Lipschitz constants and uncertainty propagation; Jafarpour [9] encloses outputs under input boxes using contraction and certifies classifier margins; Gao [10] certifies smoothed image classifiers against input perturbations. These do not establish correctness or probe stability across loop budgets. Vázquez [11] directly bounds approximate-equilibrium state/readout-derivative errors from residuals, certifying PDE residual/solution error (**abstract only**): residual certification itself is established.

Our proposed implication is elementary: under a contraction on a forward-invariant region, every later iterate lies within ε = ||F(x_r)−x_r||/(1−q) of x_r. A linear threshold readout is stable if its margin exceeds ||w||*ε. A sampled Jacobian norm or observed residual ratio does not certify q throughout that region. Fixed prompt, full recurrent state and identical update map are essential; this cannot certify correctness or stability after different generated prefixes.

**Q5 — follow-up exists.** Blayney [2] explicitly studies retrofitted Llama/OLMo/TinyLlama: layer-wise cyclic fixed points, attention-matrix changes and repeated feedforward-like inference stages. No located work tests our correctness-probe matched pair.

**Q6 — partly done.** Anil’s multiple-start latent alignment [7] is the closest direct precedent. Random-start *answer disagreement* as an incremental LM error signal remains unlocated.

## Unverified leads

Borisevich, 2026, arXiv:2609.16485: abstract opened; selected-branch DEQ continuation certification, but finite-budget classifier/readout overlap needs full-text checking. Search-only descriptions claiming Ouro experiments for arXiv:2609.10739 are stale: its opened current abstract instead describes Gemma/Llama truth-label aliasing; do not cite it as recurrent-model evidence.

## References

1. Kirin (2026), arXiv [2607.18553](https://arxiv.org/html/2607.18553), candidate-quality/correctness readouts.
2. Blayney et al. (2026), *A Mechanistic Analysis of Looped Reasoning Language Models*, arXiv [2604.11791](https://arxiv.org/html/2604.11791).
3. Zacharopoulos et al. (2026), *Late Transformer Layers Recode Syntax Canonically*, arXiv [2609.00416](https://arxiv.org/abs/2609.00416).
4. Geiping et al. (2025; revised title *Scaling by Thinking in Continuous Space*), arXiv [2502.05171](https://arxiv.org/html/2502.05171).
5. Zhu et al. (2025; v6 2026), *Scaling Latent Reasoning via Looped Language Models*, arXiv [2510.25741](https://arxiv.org/pdf/2510.25741).
6. Bansal et al. (2022), *End-to-end Algorithm Synthesis with Recurrent Networks*, arXiv [2202.05826](https://arxiv.org/abs/2202.05826).
7. Anil et al. (2022), *Path Independent Equilibrium Models Can Better Exploit Test-Time Computation*, arXiv [2211.09961](https://arxiv.org/html/2211.09961), §§7–8.
8. Chen et al. (2021), *Semialgebraic Representation of Monotone Deep Equilibrium Models and Applications to Certification*, arXiv [2106.01453](https://arxiv.org/abs/2106.01453), abstract.
9. Jafarpour et al. (2022 proceedings; 2021 preprint), *Robustness Certificates for Implicit Neural Networks*, arXiv [2112.05310](https://arxiv.org/abs/2112.05310), abstract.
10. Gao et al. (2024), *Certified Robustness for Deep Equilibrium Models via Serialized Random Smoothing*, arXiv [2411.00899](https://arxiv.org/abs/2411.00899), abstract.
11. Vázquez et al. (2026), *Convex Equilibrium Potentials for Monge–Ampère*, arXiv [2610.31386](https://arxiv.org/abs/2610.31386).
