# Blayney et al. reading memo

Read the complete [arXiv v1 HTML](https://arxiv.org/html/2604.11791), including Appendices A–F (13 April 2026). Below, findings are the paper’s; the final section gives our assessment.

## 1. Models and setup

Appendix B, Tables 1–2: `smcleish/Recurrent-Llama-3.2-train-recurrence-32` (1B; base `meta-llama/Llama-3.2-1B`), `smcleish/Recurrent-OLMo-2-0425-train-recurrence-32` (1B), and `smcleish/Recurrent-TinyLlama-3T-train-recurrence-32` (0.8B). Llama/OLMo use (4,6,4), TinyLlama (4,8,4), with input injection, pre-norm, 32 training loops; fewer-loop checkpoints are mentioned but not identified. Others: `ByteDance/Ouro-1.4B` (0,24,0; four training loops), `ByteDance/Ouro-2.6B` (48-layer upcycling), `tomg-group-umd/huginn-0125` (3.5B; (2,4,2), injection; 32 training loops). Feedforward partners appear in Figs. 8, 37–39.

Default: 256 random GSM8k test examples, no system prompt; HellaSwag repeats this setup (§E.4). Illustrations use a short greeting (B) or “The square root of 16 is” (D.3). C.1 additionally uses Huginn’s long persona, padding-token replacement, and short mathematical-assistant prompts. Tests use 8, 32, 128 loops. §2.1 specifies Gaussian initial recurrent state, variance unspecified; numerical precision is unstated. Controls: random 12-layer models (three initializations; also 4/16 layers, D.4), and Nanochat models (width 512, four heads, 3.7B tokens, four fixed training loops; B, §5.1/E.6).

## 2. Meaning of cyclic fixed point

§4, Proposition 4.1 assumes S_k(X′)=X′ and states: “Then any cyclic permutation of blocks 1,…,k will also have reached a fixed point.” Thus each within-stack layer position repeats its own state across loops; different positions can differ (Fig. 1/5). This is a fixed point of the whole stack, not a period-k orbit of the loop map. Appendix A proves this by cyclic-shift induction (Eqs. 8–12).

## 3. Convergence measurements

§4.1/D.1–D.4: residual-stream successive-difference norms (Fig. 3), distance to layer-specific loop-128 reference (4,29), successive cosine similarities (24), reference cosine similarities (6,22,30–31), pairwise depth cosine similarities (19–20). Only 19–20 explicitly specify batch/sequence averaging; residual norm type and position reduction in 3–4 are unspecified. PCA uses final-token vectors (5,26–28). Attention matrices use whole-sequence Frobenius differences, batch/head averaged: pairwise depths (2,21), loop-128 reference (23), successive loops (25).

No fitted decay/contraction rates, relative residuals, Jacobian spectra, or explicit period-two finding. Figs. 3–4 show Llama/Huginn rapidly approaching zero; Ouro’s small steps fail to establish convergence. Attention stabilizes after Ouro’s first iteration or immediately after Llama/Huginn’s prelude (§4.1). Ouro can look stable at loops 8–16 then depart (D.3,28).

C.1, Algorithm 1 detects per-token orbits/sliders using cosine to loop 128, detrending and FFT; τ=.05, ρ=.9. Table 3 nonfixed-token percentages (persona/padded/none/math): Huginn .14/.05/.02/.01; Llama all round to .00. Table 4 corresponding example percentages: 2.81/.83/.76/.45 versus 0/0/.08/0. Llama’s detected orbit is multifrequency (13); intermediate layers also orbit (C.2,14–15). Classification is heuristic and threshold-sensitive.

## 4–6. Theory, attention, monitoring

Proposition 4.2/A, Eq. 15 proves ||Δattention||≤L_sm·2Bκ/√d·||Δstate|| under bounded inputs and tied weights; it does not prove state convergence. §6 acknowledges no analytic explanation of stability. §§5–5.2/E/F show repeated feedforward-like mixing stages: middle stages recur, prelude/coda occur once; Huginn lacks clear stages. ColSum concentration, sink rates, mixing entropy, residual entropy quantify this (Eqs. 16–18). Stable models retain patterns through 128 loops; Ouro drifts (11–12,47–53). Extrapolated task-performance claims cite prior work. No correctness probes, probe drift bounds, or evaluated early-exit policy; Algorithm 1 monitors dynamics only.

## Overlap with our planned claims

- (a) **Partly**: successive steps measured, but no ratio-based forecast (§4.1).
- (b) **Not addressed**: attention bounds concern neither linear probes nor budget-wise score drift (Prop. 4.2).
- (c) **Partly**: empirical orbits, but no parity-specific limits/theory (C).
- (d) **Not addressed**: matched partners compare mixing, not error-detection probes (§5/E).
