# Prior-work audit: period-two dynamics in Huginn

Search date and cutoff: 2026-11-11. Read the write-up's Summary, §2 (especially §2.3), §6 and §9 before searching. This is an adversarial literature search, not a proof of priority. “Not found” below means not found in the opened sources after the listed queries; indexing gaps, inaccessible papers and abstract-only screening remain limitations. No experiments were run.

## 1. Verdicts on C1–C3

**C1 — CLOSE PRIOR WORK.** Geiping et al. §7 already report context-dependent Huginn orbits, and Blayney et al. Appendix C already quantify their incidence and dependence on prompting. I did not find the specific result of settled period-two orbits in 26/200 fresh addition prompts (13%, write-up CI 8–18%); neither prior result establishes that period or that prevalence in this setting. [Geiping](https://arxiv.org/html/2502.05171), [Blayney](https://arxiv.org/html/2604.11791).

**C2 — CLOSE PRIOR WORK.** Gao et al. §§III.3–III.4 already demonstrate a −1 crossing and stable period-two motion in recurrent self-attention, while Kim and Jones already analyze transformer bifurcations quantitatively. I did not find the specific combination of a pretrained Huginn source-token own-block instability, causal downstream inheritance, and context-interpolation scaling; that narrower combination remains the defensible candidate contribution. [Gao](https://arxiv.org/html/2609.28448), [Kim](https://arxiv.org/html/2608.26556), [Jones](https://arxiv.org/html/2610.31692).

**C3 — CLOSE PRIOR WORK.** Movahedi et al. §3.3 and Appendix A.3 already give the same under-relaxed update and eigenvalue transformation, including a general spectral stability bound. The apparent addition is source-token targeting and measured suppression in pretrained Huginn, not a new damping principle or an eigenvector-selective intervention. [Movahedi](https://arxiv.org/html/2606.18206).

Two corrections are necessary when stating the claims:

- The write-up applies **whole-state under-relaxation at one token**, s ← (1−a)s+aF(s). It does not project onto or damp only a Jacobian eigenvector. For a real negative eigenvalue its threshold a*=2/(1−λ) is a specialization of the existing spectral bound.
- Its 24/26 result means the **selected source's blink stopped**. Whole-sequence stationarity occurred in only 15/26; downstream motion remained in at least six source-suppressed cases. The 26 source eigenvalues were −1.007 to −1.226, median −1.07, measured at hidden unstable fixed points; actual crossing/scaling evidence comes from two interpolation families, not 26 independently continued bifurcations.

## 2. Closest papers, most damaging first

### 1. Blayney et al.: the same model, orbits, prevalence and prompt dependence

Hugh Blayney, Álvaro Arroyo, Johan Obando-Ceron, Pablo Samuel Castro, Aaron Courville, Michael M. Bronstein and Xiaowen Dong (2026), **A Mechanistic Analysis of Looped Reasoning Language Models**, arXiv:2604.11791. **FULL TEXT read:** §§2–4, Proposition 4.2 and its Appendix A proof, Appendices B and C.1–C.3. [Paper](https://arxiv.org/abs/2604.11791), [text](https://arxiv.org/html/2604.11791).

Appendix C goes substantially beyond the write-up's “orbits on GSM8K” description: an FFT-based per-token classifier, full-test-set census, and system-prompt comparisons. Huginn orbit incidence ranges from 0.00–0.13% of tokens and 0.38–2.50% of examples across the tabulated prompt conditions. These denominators and thresholds differ from the addition census. The authors explicitly caution about classifier hyperparameters and leave detailed explanation open. They do not establish exact period two, a source-token flip or targeted suppression.

The main paper's “cyclic fixed points” also describe different constituent layers' states within one block; that is not itself a multi-iteration orbit of the complete block. Proposition 4.2 bounds attention changes by state changes; it is not evidence of a source oscillator driving downstream tokens.

### 2. Gao, Yang and Chen: an actual transformer flip and period-two state

Qucheng Gao, Zuyi Yang and Xiao Chen (2026), **Nonequilibrium Phases of Repulsive Self-Attention: Chaos, Attention Condensation, and Emergent Locality**, arXiv:2609.28448. **FULL TEXT read:** introduction, model definition, §§III.2–III.4.1 and the local period-doubling analysis in §III.7; abstract and overview for perturbation spreading. [Paper](https://arxiv.org/abs/2609.28448), [text](https://arxiv.org/html/2609.28448).

This rules out “first flip bifurcation/period-two dynamics in a recurrent transformer.” A normalized single-head model with Q=K=I and V=−I has explicit Fourier multipliers crossing −1; computed period-two branches have stable transverse two-step multipliers. It also discusses local instability and routing-mediated perturbation spreading. Crucially, this is a prescribed repulsive attention model, with collective Fourier modes and angular-neighbor geometry, not a pretrained causal LM with an earliest unstable sequence position. Its near-flip amplitude/branch analysis is close methodology, but the sections read do not establish the write-up's particular amplitude-squared-versus-own-block-eigenvalue regression. Relative periodicity is sometimes assessed modulo a common rotation.

### 3. Movahedi et al.: the damping formula is already in looped-transformer work

Sajad Movahedi, Vera Milovanović, Shlomo Libo Feigin, Alexander Theus, Thomas Hofmann, Valentina Boeva, T. Konstantin Rusch and Antonio Orvieto (2026), **Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers**, arXiv:2606.18206. **FULL TEXT read:** §3.3, surrounding architecture discussion, Appendix A.3. [Paper](https://arxiv.org/abs/2606.18206), [text](https://arxiv.org/html/2606.18206).

They observe non-convergent oscillation for some inputs and use gη=(1−η)I+ηf. Appendix A.3 derives μη=1−η+ηλ and the bound η<2(1−Re λ)/|λ−1|², minimized over modes; their practical controller reduces η when residual improvement stalls. Thus the update, preservation of fixed points and eigenvalue-based explanation all precede C3. They do not report Huginn's source-token localization, 24/26 intervention result or a rank-one eigenvector intervention. Distinguish their adaptive global damping from calibration at the selected source.

### 4. Geiping et al.: Huginn orbits were reported in the model paper

Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele and Tom Goldstein (2025), **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach**, arXiv:2502.05171. **FULL TEXT read:** §7, Figures 11–12 discussion, path independence and related-work passages. [Paper](https://arxiv.org/abs/2502.05171), [text](https://arxiv.org/html/2502.05171).

The authors show fixed-point approaches, orbits and drifting “sliders,” including an orbit for a numerical token in a GSM8K question; dynamics depend on context and recur across initializations. Consequently, neither orbital latent computation nor context-dependent convergence in Huginn is new. This is qualitative trajectory geometry rather than a period-two census, Jacobian continuation, bifurcation identification or source-specific intervention. The opened HTML is v2; terminology in later model-paper versions may differ.

### 5. Jones: quantitative flips and certified periodic orbits in LM feedback

Timothy D. Jones (2026), **Chaos and bifurcations in continuous language-model feedback**, arXiv:2610.31692. **FULL TEXT read:** §§I–II, relevant §§IV–VI, §IX and its claims/limitations table. [Paper](https://arxiv.org/abs/2610.31692), [text](https://arxiv.org/html/2610.31692).

The feedback state is the expected next-token embedding under a standard pretrained LM, with a fixed prefix/window. Section VI tracks real multipliers reaching −1 through period-doubling cascades; it reports quantitative scaling and computer-assisted proofs for selected attracting orbits, including a period-two orbit. This substantially precedes generic claims of derivative-validated LM bifurcations. The operator closes the loop through soft output probabilities and embeddings, unlike Huginn's internal core recurrence. It does not identify an upstream own-position block or suppress its instability by token-local under-relaxation.

### 6. Kim: trained looped-transformer bifurcations and derivative-based scaling

Gunn Kim (2026), **Dynamical phase selection controls compute scaling in looped transformers**, arXiv:2608.26556. **FULL TEXT read:** main article, setup, phase classification, scaling analysis and references. [Paper](https://arxiv.org/abs/2608.26556), [text](https://arxiv.org/html/2608.26556).

A small trained attention-plus-MLP recurrence supports fold, Neimark–Sacker and other dynamical phases. Local derivatives predict relaxation-time and spectral-gap scaling. Calling this merely “a fold bifurcation” understates it: the complex-pair transition explicitly produces bounded nonstationary motion. Conversely, the paper explicitly reports no period doubling in its explored models. Its task is a small sign-decision/memory system, not language-prompt source-token analysis. It already removes novelty from “use continuation and normal forms to explain a trained looped transformer's computation,” but not the particular Huginn flip.

### 7. Tomihari and Karakida: Jacobians, oscillation and criticality in recurrent attention

Akiyoshi Tomihari and Ryo Karakida (2025; opened v5 revised 2026-10-18), **Recurrent Self-Attention Dynamics: An Energy-Agnostic Perspective from Jacobians**, arXiv:2505.19458, NeurIPS 2025. **FULL TEXT read:** §§5.1–5.3, §6.3, limitations and relevant references. [Paper](https://arxiv.org/abs/2505.19458), [text](https://arxiv.org/html/2505.19458).

They connect normalization to Jacobian spectra, oscillatory complex modes and Lyapunov exponents near criticality, and develop spectral regularization. This is direct prior work for explaining iterated-attention dynamics using local derivatives rather than assuming an energy function. Their stated setting omits positional encoding, masking and MLP blocks. It does not identify a real −1 source-token mode in Huginn or demonstrate source-token damping. Attribute details checked in v5 to the version read rather than automatically to the initial 2025 submission.

### 8. Anil et al.: useful equilibrium-model computation need not converge to a point

Cem Anil, Ashwini Pokle, Kaiqu Liang, Johannes Treutlein, Yuhuai Wu, Shaojie Bai, Zico Kolter and Roger Grosse (2022), **Path Independent Equilibrium Models Can Better Exploit Test-Time Computation**, arXiv:2211.09961, NeurIPS 2022. **FULL TEXT read:** §§1–4, §7 and Figure 5 discussion, relevant Appendix B.7 setup. [Paper](https://arxiv.org/abs/2211.09961), [text](https://arxiv.org/html/2211.09961).

Section 7 explicitly reports different limit cycles under fixed-point iteration and Broyden's method in a weight-tied, input-injected maze network, with strong accuracy and path independence. Limit cycles in an equilibrium-model family are therefore old empirical work, not just a theoretical possibility. The paper does not identify their period, a −1 crossing, an attention-mediated source or mode-calibrated suppression. Its central distinction—same limiting behavior versus convergence to a fixed point—is directly relevant to interpreting Huginn.

### 9. Kiwitt et al.: identify the “uniform convergence” study correctly

Fynn Kiwitt, Christopher Irwin, Riccardo Ali and Pietro Liò (2026), **Lyapunov Spectral Analysis of Loop Transformer Dynamics**, ICML 2026 Mechanistic Interpretability Workshop. **FULL TEXT of the one-page workshop poster read:** all sections; **article inaccessible**, OpenReview verification blocked the paper. [Official workshop poster](https://mechinterpworkshop.com/poster-pdfs/748.pdf); article location: https://openreview.net/forum?id=enDyNuSD3L.

The poster reports 50 tracked exponents over 64 recurrences and 20 prompts: Huginn's spectrum is negative, while Ouro's leading exponent is positive; it attributes much of Huginn's contraction to the adapter/core balance. It also directly reports decay of successive-state differences. This is a workshop study, not an established global convergence theorem from the ICML main proceedings. Mathematical interpretation: a stable discrete period-two orbit can have negative Lyapunov exponents, so negative exponents alone cannot exclude it; the reported residual decay is separate evidence about those sampled trajectories. It does not test the addition census's oscillating prompts.

### 10. Wang and Reid: Jacobian interventions in these very models

Wenlong Wang and Fergal Reid (2026), **Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?**, arXiv:2609.01924. **ABSTRACT only.** [Paper](https://arxiv.org/abs/2609.01924).

This is an important near-neighbor for any broad claim of first Jacobian-based mechanistic intervention in Huginn/Ouro: it describes virtual unrolling, lens fitting and eleven causal experiment families, with loop-dependent transport and intervention windows. The abstract does not report period-two orbits, flip eigenvalues, source-token inheritance or damping. Because only the abstract was read, it cannot establish their absence from the full paper.

### Classical background and additional screening

**Gautam M. Shroff and Herbert B. Keller (1993), Stabilization of Unstable Procedures: The Recursive Projection Method**, SIAM Journal on Numerical Analysis 30(4), 1099–1120, DOI 10.1137/0730057. **ABSTRACT only**, opened on both the publisher and author repository; PDF retrieval failed. [Publisher](https://epubs.siam.org/doi/10.1137/0730057). It stabilizes an unstable fixed-point iteration by projecting onto its unstable subspace, applying Newton there and retaining ordinary iteration on the complement. Thus even genuinely subspace-selective stabilization has strong numerical-analysis precedent; this is not an LM experiment or the write-up's scalar relaxation.

**Razvan Pascanu, Tomas Mikolov and Yoshua Bengio (2013), On the difficulty of training recurrent neural networks**, ICML, PMLR 28(3):1310–1318. **FULL TEXT read:** §2.1 Jacobian products and §2.2 dynamical-systems/bifurcation discussion. [Publisher](https://proceedings.mlr.press/v28/pascanu13.html), [PDF](https://proceedings.mlr.press/v28/pascanu13.pdf). It makes attractors, bifurcations and basin changes part of RNN gradient analysis; it also cautions that a bifurcation crossing is not by itself necessary or sufficient for exploding gradients. It does not establish C1–C3 in transformers.

The following were opened and screened but are weaker matches. All are **ABSTRACT only**, except the noted related-work reading; no full-text absence claim is made:

- Francesco Pappone, Donato Crisostomi and Emanuele Rodolà (2025), **Two-Scale Latent Dynamics for Recurrent-Depth Transformers**, [arXiv:2509.23314](https://arxiv.org/abs/2509.23314). Local refinements, inter-block drift and second-difference halting; followed backward from Blayney's references.
- Xiao-Wen Yang, Ziyu Han, Xi-Hua Zhang, Wen-Da Wei, Jie-Jing Shao, Lan-Zhe Guo and Yu-Feng Li (2026), **Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models**, [ICML/PMLR 306:144670–144682](https://proceedings.mlr.press/v306/yang26t.html). STARS uses training-time Jacobian spectral-radius regularization, not source-token runtime control. This is distinct from Kiwitt's workshop study.
- Matthias M. M. Buehlmaier (2026), **Spectral Margins and Memory Routing in Looped Transformers**, [arXiv:2607.10681v2](https://arxiv.org/abs/2607.10681). Small-model Jacobian spectral margins, non-normality and non-settling initializations; v2, dated October 30, replaces the earlier title *LayerNorm as Implicit Gain Control in Looped Transformers*. Followed from Kim's references.
- Asher Labovich (2026), **Stability and Generalization in Looped Transformers**, [arXiv:2604.15259v3](https://arxiv.org/abs/2604.15259). Local attractivity, input dependence and geometry; v3 is dated November 4. Followed from Kim's references.
- Hayden Prairie, Zachary Novack, Taylor Berg-Kirkpatrick and Daniel Y. Fu (2026), **Parcae: Scaling Laws For Stable Looped Language Models**, [arXiv:2604.12946](https://arxiv.org/abs/2604.12946). Injection-parameter spectral control and training stability. Followed from Kim's references.
- Pawel Olszowiec, Paulina Tomaszewska, Maciej Żelaszczyk, Michal Byra and Neo Christopher Chung (2026), **Monitoring latent reasoning in looped language models**, [arXiv:2610.18127](https://arxiv.org/abs/2610.18127). Also **FULL TEXT §2 related work read**: it explicitly cites Blayney and directs attention to Wang–Reid. Its abstract surveys eight models through five inspection methods; it does not claim the source-flip mechanism.
- Yuekun Yao, Yupei Du, Dawei Zhu and Huan Sun (2026), **ARLoop: Fully-Open Pre-trained Looped Transformers with Attention Residuals**, [arXiv:2611.05234](https://arxiv.org/abs/2611.05234), November 3. Depth-attention residual architecture and scaling, without a flip claim in the abstract.
- Xuan Ouyang, Zefan Cai, Chenlong Zhang, Cheng Luo, Ke Wan and Junjie Hu (2026), **Loopback Transformer: Towards a Unified View of Residual Looped Transformers**, [arXiv:2611.07226](https://arxiv.org/abs/2611.07226), November 4. Geometric decay and depth-memory read weighting, without a source-orbit claim in the abstract.

Citation following therefore included Geiping → Anil; Blayney → Pappone; later work citing Blayney → Olszowiec → Wang–Reid; Kim → Movahedi/Parcae/Buehlmaier/Labovich; and Gao → Tomihari–Karakida. This was manual backward/forward searching, not an exhaustive citation-index export. No exact precursor for an earliest unstable **causal sequence position** driving downstream period-two motion was found. Gao's angular-order propagation is a serious conceptual neighbor, but angular order is not causal token order.

## 3. Unverified leads

These are leads, not evidence used to declare a claim already published:

- **Kenji Doya (1992), Bifurcations in the learning of recurrent neural networks**, ISCAS, pp. 2777–2780. Search identified [IEEE article 230622](https://ieeexplore.ieee.org/abstract/document/230622), DOI 10.1109/ISCAS.1992.230622; direct opening failed with HTTP 418 and DOI access also failed. The search abstract concerns learning across bifurcations. Do not claim a particular flip calculation without reading it. Pascanu's opened paper discusses related Doya work dated 1993, which should not be silently conflated with this 1992 item.
- **Robert Haschke and Jochen J. Steil (2005), Input space bifurcation manifolds of recurrent neural networks**, Neurocomputing 64:25–38, [publisher location](https://www.sciencedirect.com/science/article/pii/S0925231204005120), DOI 10.1016/j.neucom.2004.11.030. Publisher and abstract URLs returned 403; DOI access failed. Search results describe codimension-one bifurcations with inputs as parameters and an input-controlled oscillator. Potentially very damaging to a broad “context controls bifurcation” claim, but not full-text verified here.
- **Kiwitt et al.'s full workshop article:** bibliographic identity and poster are verified above; the paper's OpenReview PDF/forum and API endpoint were inaccessible. Detailed article-only claims remain unverified.
- **Learning Chaos Without Seeing Chaos: Extrapolation of Global Dynamics in Autoregressive Transformers**, arXiv:2609.38814, appeared in search with period-doubling reconstruction. Its arXiv page was not opened, so author list and full claims are not certified here. Likely a learned dynamical-system surrogate rather than a pretrained LM's native latent orbit.
- **Spherical Fixed-Point Reasoners: Adaptive Loop Halting on the Hypersphere**, arXiv:2610.25646, and **Reasoning on the Simplex: Geometric Fixed-Point Models**, arXiv:2609.33540, appeared while following FPRM. Pages were not opened; investigate their stabilization details before claiming comprehensive coverage of late-2026 fixed-point solvers.

## 4. Queries run

Literal queries, in execution order; IDs and names below came from the task's literature references or public literature results. No repository name, private person name or filesystem path was sent in a query.

1. `"2604.11791"`
2. `"2610.31692"`
3. `"2608.26556"`
4. `recurrent depth looped language models Lyapunov exponents ICML 2026 convergence`
5. `"looped" "Lyapunov" "2026" convergence`
6. `"Path independent equilibrium models" limit cycles`
7. `Huginn period doubling flip bifurcation oscillations`
8. `"A Mechanistic Analysis of Looped Reasoning Language Models" citations`
9. `"looped language" "Lyapunov exponents"`
10. `"Path Independent Equilibrium Models" Anil 2022`
11. `"Huginn" "oscillation" "2026"`
12. `"looped transformer" "flip bifurcation"`
13. `"recurrent" "Lyapunov" "ICML 2026" language`
14. `"Path Independent Equilibrium Models Can Better Exploit Test-Time Computation" arxiv`
15. `"Huginn" "Lyapunov"`
16. `"Doya" "1992" "bifurcations"`
17. `"Haschke" "Steil" "2005" bifurcation`
18. `"deep equilibrium" "eigenvector" damping`
19. `"Lyapunov" "uniform" "Huginn"`
20. `"Lyapunov" "looped" site:proceedings.mlr.press`
21. `transformer "period doubling" 2025 2026`
22. `"universal transformer" "limit cycles"`
23. `"deep equilibrium" "oscillation" eigenvalue damping`
24. `"looped" "source token" oscillation`
25. `"recurrent depth" "edge of chaos"`
26. `"Pascanu" "2013" bifurcations gradients`
27. `"Pappone" "two-scale" transformer`
28. `"deep equilibrium" "recursive projection"`
29. `"looped transformer" "damping" eigenvalue`
30. `"Huginn" "period-2"`
31. `"Fixed-Point Reasoners" arxiv`
32. `"Lyapunov Spectral Analysis of Loop Transformer Dynamics" arxiv`
33. `"Haschke" "Steil" "bifurcation" pdf`
34. `"recursive projection method" Shroff Keller 1993`
35. `"Lyapunov Spectral Analysis" Kiwitt pdf`
36. `"Bifurcations in the learning of recurrent neural networks" IEEE 1992`
37. `"Huginn" "flip" bifurcation`
38. `"looped" "token" "oscillations" attention 2026`
39. `"transformer" "Neimark-Sacker" 2026`
40. `"transformer" "flip bifurcation" 2025 2026`
41. `"looped" "eigenvector" stabilization oscillation`
42. `"Huginn" "Blayney" "November" 2026`
43. `Huginn looped transformer bifurcation November 2026`
44. `"deep equilibrium" "eigenvector" "stabilization"`
45. `"weight-tied" "period-2"`
46. `"2609.28448" transformer bifurcation`
47. `"Huginn" "two-cycle"`
48. `"looped transformer" "source" "non-convergence"`
49. `"recurrent self-attention" "period doubling"`
50. `"Jacobian eigenvector" "limit cycle" "deep equilibrium"`
51. `"Huginn" "period-doubling"`
52. `"Blayney" "orbits" 2026`
53. `"transformer" "eigenvector" "damping" oscillation`
54. `"looped language" "non-convergence" 2026 November`

## 5. Likely referee assessment

Orbits and context-dependent convergence in Huginn are already published, and recurrent-attention flip instabilities now have direct analytical and numerical precedent. Jacobian stability analysis, bifurcation continuation and scalar damping are established methods, with the exact relaxation spectrum already derived in looped-transformer work. The plausible contribution is a carefully delimited empirical mechanism in pretrained Huginn: a source-position flip, downstream inheritance, two contextual continuation tests and source-targeted suppression. A knowledgeable referee would ask §9 to acknowledge these stronger precedents and insist that 24/26 source suppression not be presented as global stabilization or eigenvector-selective control.
