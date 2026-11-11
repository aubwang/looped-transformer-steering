# Prior-work audit: read-space self-gain in looped language models
Date and cutoff: 2026-11-11. Search covers material available through this date, including October–November 2026 searches and current arXiv versions. This is a bounded novelty search, not an exhaustive citation census. “NOT FOUND” means not found after the queries below, not proof of absence.

Read first: the requested three-model diagnostic in docs/steering_results.typ, all of docs/lread_design.md, and the readout-suppression, second-model, and status/limits sections of docs/freeloop.typ. No experiments were run. Only this memo was written.

## 1. Verdicts

- **B1 — CLOSE PRIOR WORK.** Gurnee et al. (2026), §4.3 and Appendix A.19, already compare gains on readout-derived directions against isotropic directions; Wang and Reid (2026), §§2–4, transport unembedding directions through looped-model Jacobians. I did not find their combination as the specific statistic (u^TJu-\mathrm{tr}(J)/d), with per-context logit-gradient directions and a spherical probe baseline.
- **B2 — NOT FOUND.** No opened source reports the specified three-checkpoint signed-self-gain measurements or the negative-Huginn/positive-Ouro/near-zero-Recurrent-Llama pattern. However, differences between Huginn and Ouro in transport, convergence and readout interfaces are already central findings of Wang–Reid, the Kiwitt et al. workshop work, and later October diagnostics; the novelty cannot be “looped models have different dynamics.”
- **B3 — CLOSE PRIOR WORK.** Buehlmaier (2026), §5.3, already measures one-step amplification at stable looped-transformer fixed points, and Fernando–Guitchounts (2026), §§2–3, studies anisotropic, non-normal transformer Jacobians. The specific Recurrent-Llama measurements around 0.22 and 1.3 were not found; those two norms alone establish anisotropy, not non-normality.
- **B4 — ALREADY PUBLISHED as mathematical machinery.** Quadratic-form trace estimation and rotationally invariant probe analysis are established, e.g. Wimmer–Wu–Zhang (2014), §§1, 3–4; Bai–Koltun–Kolter (2021), §4.2, already uses Hutchinson probes on equilibrium Jacobians. The statistic is a **Rayleigh quotient** of the Jacobian, equivalently of its symmetric part over real vectors; its range is the **real numerical range/field of values**, and a chosen value is a diagonal entry in an orthonormal basis containing that direction.

For real (J), let (S=(J+J^T)/2), (\tau=\mathrm{tr}(J)/d), and (u) be uniform on the unit sphere. The elementary identities are
$$
u^TJu=u^TSu,\qquad
\mathbb E[u^TJu]=\tau,
$$
$$
\operatorname{Var}(u^TJu)
=\frac{2}{d(d+2)}\left(\|S\|_F^2-\frac{\mathrm{tr}(J)^2}{d}\right)
=\frac{\|J\|_F^2+\mathrm{tr}(J^2)-2\mathrm{tr}(J)^2/d}{d(d+2)}.
$$
These follow from spherical second/fourth moments; the displayed nonsymmetric rewrite is my algebraic verification, not a claim that the cited trace paper prints exactly this form. Chebyshev then gives a concentration bound. A (d^{-1/2}) scale requires a bounded normalized second moment, rather than following from dimension alone. “Self-gain” is a useful descriptive name but not a new mathematical object. Neither trace nor this real quadratic form detects the skew-symmetric part. The squared-length statistic is instead (\mathbb E\|Ju\|^2=\|J\|_F^2/d).

## 2. Closest papers, most damaging first

### 1. Wang and Reid: direct prior comparison of Huginn and Ouro

**Wenlong Wang and Fergal Reid (2026). _Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?_ arXiv:2609.01924v1.** [Paper and full text](https://arxiv.org/html/2609.01924).

**Read: FULL TEXT**, §§1–8, with the methods/transport analysis in §§2–4 and causal results in §7; also inspected appendices on the inventory and adapter validation. Their operator is a corpus/position-averaged source-to-target state Jacobian, with token directions obtained by pulling back unembedding vectors. They examine transport norms, semantic recoverability, singular-subspace overlap, composition of mean transports, and interventions. Huginn uses 16 recurrences; Ouro is 2.6B.

**Exact B1/B2 check:** I found no reported (u^TJu), diagonal entries in a readout-aligned basis, trace baseline, Hutchinson self-gain comparison, or signed read-minus-random excess. Their bulk norm comparison is not that statistic; their random-pair/subspace controls are not spherical trace probes. Nevertheless, this preempts a broad claim to discovering architecture-dependent readout transport. Importantly, “Huginn carries content” is qualified by §4.5: cutting input reinjection destroys it, supporting continual recomputation rather than passive storage. Low local self-gain is not a contradiction. §§4.5 and 7 are particularly relevant to any proposed training or maintenance explanation.

### 2. Gurnee et al.: readout-related directional gains versus random directions

**Wes Gurnee, Nicholas Sofroniew, Adam Pearce, Mateusz Piotrowski, Isaac Kauvar, Runjin Chen, Anna Soligo, Paul Bogdan, Euan Ong, Rowan Wang, Ben Thompson, David Abrahams, Subhash Kantamneni, Emmanuel Ameisen, Joshua Batson and Jack Lindsey (2026). _Verbalizable Representations Form a Global Workspace in Language Models._ arXiv:2607.15495.** [Full text](https://arxiv.org/html/2607.15495).

**Read: FULL TEXT, selected sections:** §2.1–2.3 definitions, §4.3 and Appendix A.19. §4.3.1 measures the output norm of the next MLP applied to unit J-lens directions, normalized by isotropic-random output norms. §4.3.2 measures attention OV gain and label preservation against random controls; Appendix A.19 examines read/write weight alignment.

This is a strong precedent for asking whether the output-related subspace is treated differently from random directions. It is not a per-context recurrent Jacobian quadratic form: the MLP metric is an output norm, and attention label preservation is a ranking statistic. It finds enhanced broadcast, not the claimed signed one-loop deficit. Cite this when introducing the diagnostic, rather than suggesting random-controlled readout-gain comparisons are new.

### 3. Buehlmaier: stable loops can amplify perturbations

**Matthias M. M. Buehlmaier (2026). _Spectral Margins and Memory Routing in Looped Transformers._ arXiv:2607.10681v2, revised 30 October 2026; v1 had a different title.** [Abstract](https://arxiv.org/abs/2607.10681), [PDF](https://arxiv.org/pdf/2607.10681).

**Read: FULL TEXT, selected sections:** introduction/related work, §5.3 and adjoining spectral discussion, §7 and relevant appendix passages. Small LayerNorm/Parcae-inspired models have fixed-point Jacobians with operator norm above one while spectral radius is below one. The paper explicitly distinguishes one-step amplification, asymptotic stability and non-normal transients.

This substantially preempts B3 as a general phenomenon. It does not establish the Recurrent-Llama numbers, optimize score-moving directions, or compare readout quadratic forms with trace. Also distinguish its small-model settings from trained billion-parameter checkpoints.

### 4. Fernando and Guitchounts: full transformer Jacobian geometry

**Jesseba Fernando and Grigori Guitchounts (2026). _Dynamics of the Transformer Residual Stream: Coupling Spectral Geometry to Network Topology._ arXiv:2605.14258.** [Full text](https://arxiv.org/html/2605.14258).

**Read: FULL TEXT, §§2–3.** The study computes layer Jacobians in Llama, OLMo and Gemma, examines spectra, singular-subspace alignment, cumulative rank collapse and Schur removal of non-normal structure. It also compares Jacobian column norms for graph-boundary versus other units.

This is close prior art for directional amplification/suppression and a stronger direct non-normality analysis than B3's two length summaries. The selected directions are graph/community related, not output-gradient directions, and the models are feed-forward rather than the tested recurrent checkpoints. Its “self-alignment” measures singular-subspace overlap, not (u^TJu).

### 5. Oskin: computation away from the readout

**Mark Oskin (2026). _Off-Axis, On Purpose: Where a Transformer Computes Concepts and Why it Does So._ arXiv:2608.10251.** [Full text](https://arxiv.org/html/2608.10251).

**Read: FULL TEXT, selected sections:** §§2–3, §10. A GPT-2-scale study examines writes relative to the final readout and matched rotation controls, separates off-axis computation from later answer writing, and connects frame motion to the antisymmetric part of a fitted linear response.

This preempts broad “the network computes outside what the output reads” framing. However, its write-angle measurement is in vocabulary space relative to final logits; its fitted regression operator is explicitly distinguished from the mean Jacobian. Neither near-orthogonal activation writes nor a low-rank approximate unembedding null space establishes a negative read-direction self-gain excess.

### 6. Lu et al.: October Jacobian analysis of the same model pair

**Dezhen Lu, Runguo Li, Zixiang Ni, Yuchen Tang, Xingquan Kong, Jiaxing Yang, Jiarui Guo and Lingxiang Gao (2026). _Where KV-Cache Error Lives in Looped Language Models: A Diagnostic Study of Ouro and Huginn._ arXiv:2610.41825.** [Full text](https://arxiv.org/html/2610.41825).

**Read: FULL TEXT, selected sections:** §6 and Appendix E/T4, plus abstract and contents. The authors compare KV-error tangents in a Fisher metric, analyze loop-gradient coherence, and estimate dominant loop-map Ritz values. On Huginn, similarly directed cache errors can have weakly aligned logit tangents; negative dominant modes and parity effects also appear.

This is genuine recent prior work on anisotropic sensitivity and loop Jacobians, including the same two architecture families. Its perturbations live in cache/weight space and its output metric is Fisher/KL; it does not report B1's state-space diagonal Rayleigh quotients or B2's trace-normalized three-model table. Do not describe every orthogonality result here as read-space suppression.

### 7. Sharma and Vu: normalized readout visibility is limited

**Rituraj Sharma and Tu Vu (2026). _Dense Supervision Is Not Enough: The Readout Blind Spot in Looped Language Models._ arXiv:2606.24898.** [Full text](https://arxiv.org/html/2606.24898).

**Read: FULL TEXT, §2 and abstract/experimental overview.** Lemmas 1–2 distinguish the radial coordinate hidden by a normalized readout from scale carried by pre-norm recurrence. Future-loop losses can provide indirect signals.

This is a direct warning about defining the read space through final normalization. It does not imply low (u^TJu) on visible gradients: an invisible radial direction and an under-self-coupled visible direction are different statements. Nor is the blind direction necessarily radial in the pre-coda state coordinates; a nonlinear coda pulls the constraint back.

### 8. Blayney et al.: repeated stages and cyclic limiting behavior

**Hugh Blayney, Álvaro Arroyo, Johan Obando-Ceron, Pablo Samuel Castro, Aaron Courville, Michael M. Bronstein and Xiaowen Dong (2026). _A Mechanistic Analysis of Looped Reasoning Language Models._ arXiv:2604.11791.** [Full text](https://arxiv.org/html/2604.11791).

**Read: FULL TEXT, selected sections:** preliminaries, §§4–5 and limitations. It studies cyclic trajectories, stable attention behavior, architectural changes and repeated inference stages.

This rules out novelty claims about simply applying mechanistic analysis to looped models or contrasting their convergence. The inspected analyses do not make the trace-versus-read-gradient comparison. Repeated prediction/suppression stages are conceptually adjacent but do not determine a local Jacobian self-gain.

### 9. Bai, Koltun and Kolter: equilibrium Jacobians and random VJPs

**Shaojie Bai, Vladlen Koltun and J. Zico Kolter (2021). _Stabilizing Equilibrium Models by Jacobian Regularization._ arXiv:2106.14342.** [Abstract](https://arxiv.org/abs/2106.14342), [full text](https://arxiv.org/html/2106.14342).

**Read: FULL TEXT, §§2.1 and 4.2.** The HTML carries the heading “Accelerating Equilibrium Models by Stabilizing Their Jacobians”; the abstract page supplies the cited title. Equation (3) estimates (\mathrm{tr}(JJ^T)) using Gaussian VJPs; equation (4) regularizes this quantity, and §2.1 gives the readout-loss gradient through the equilibrium resolvent.

This preempts treating random probes of a fixed-point Jacobian, output sensitivities through it, or the Frobenius baseline in B3 as new. Its trace is a squared-length statistic, not signed (\mathrm{tr}(J)); it does not compare visible directions with that signed baseline.

### 10. Wimmer, Wu and Zhang: the trace-estimation foundation

**Karl Wimmer, Yi Wu and Peng Zhang (2014). _Optimal query complexity for estimating the trace of a matrix._ arXiv:1405.7112; full version of ICALP 2014 paper.** [Full text](https://arxiv.org/html/1405.7112).

**Read: FULL TEXT, selected portions:** §§1–3 and definitions of random orthogonal unit vectors; inspected the §4 estimator framework. The basic oracle is precisely (x^TAx); the paper studies unbiased trace estimation, variance, and randomized orthogonal directions.

B4 and the random-baseline part of B1 are established numerical linear algebra. This source is not evidence that the proposed readout-conditioned empirical comparison was already performed.

### Additional screened sources and access record

These are supporting/negative-screen sources, not further ranked “closest” papers.

- **Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele and Tom Goldstein (2025), _Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach_, arXiv:2502.05171v2.** [Full text](https://arxiv.org/html/2502.05171). **FULL TEXT: §7, including path independence, and related-work discussion.** Trajectories can converge, orbit or drift; this is not the signed diagnostic. Followed its later mechanistic citations via Wang–Reid and the 2026 searches.
- **Rui-Jie Zhu, Zixuan Wang, Kai Hua, Tianyu Zhang, Ziniu Li, Haoran Que, Boyi Wei, Zixin Wen, Fan Yin, He Xing, Lu Li, Jiajun Shi, Kaijing Ma, Shanda Li, Taylor Kergan, Andrew Smith, Xingwei Qu, Mude Hui, Bohong Wu, Qiyang Min, Hongzhi Huang, Xun Zhou, Wei Ye, Jiaheng Liu, Jian Yang, Yunfeng Shi, Chenghua Lin, Enduo Zhao, Tianle Cai, Ge Zhang, Wenhao Huang, Yoshua Bengio and Jason Eshraghian (2025; current revision 2026), _Scaling Latent Reasoning via Looped Language Models_, arXiv:2510.25741v6.** [Page](https://arxiv.org/abs/2510.25741). **ABSTRACT only.** Establishes Ouro provenance; insufficient reading to rule out every detail in its full text.
- **Wenquan Lu, Yuechuan Yang, Kyle Lee, Yanshu Li and Enqi Liu (2025), _Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer_, arXiv:2507.02199v2.** [Page](https://arxiv.org/abs/2507.02199). **ABSTRACT only.** Followed from Wang–Reid's bibliography: logit/coda lenses and inconsistent decoding across core blocks, not an abstract-level claim of Jacobian self-gain.
- **Nora Belrose, Igor Ostrovsky, Lev McKinney, Zach Furman, Logan Smith, Danny Halawi, Stella Biderman and Jacob Steinhardt (2023; revised 2025), _Eliciting Latent Predictions from Transformers with the Tuned Lens_, arXiv:2303.08112v6.** [Page](https://arxiv.org/abs/2303.08112). **ABSTRACT only.** Learned affine probes and iterative refinement are prior art; no exact-diagnostic exclusion claimed from this reading.
- **Jett Janiak, Can Rager, James Dao and Yeu-Tong Lau (2024), _An Adversarial Example for Direct Logit Attribution: Memory Management in GELU-4L_.** [Publisher paper](https://aclanthology.org/2024.blackboxnlp-1.15/). **FULL TEXT: §§2–3 in PDF.** Projection ratios and causal writer-to-eraser interventions demonstrate cancellation of earlier writes. Finite activation erasure is not the same estimand as differential self-gain.
- **Sebastian Gerstner, Hilal AlQuabeh, Kentaro Inui and Hinrich Schütze (2026), _Weakening Neurons: An Input-Output Functionality in Transformers with Outsize Influence_, arXiv:2609.18612.** [Page](https://arxiv.org/abs/2609.18612). **ABSTRACT only.** Negative input/output weight cosines provide an especially relevant local negative-feedback analogue; not a full-loop Jacobian or trace comparison.
- **Giancarlo Kerg, Kyle Goyette, Maximilian Puelma Touzel, Gauthier Gidel, Eugene Vorontsov, Yoshua Bengio and Guillaume Lajoie (2019), _Non-normal Recurrent Neural Network (nnRNN): learning long time dependencies while improving expressivity with transient dynamics_, arXiv:1905.12080.** [Page](https://arxiv.org/abs/1905.12080). **ABSTRACT only.** Followed from Buehlmaier: spectral constraints and transient expressivity in trained RNNs long predate B3.
- **Ivan Viakhirev, Kirill Borodin, Maxim Abramov and Grach Mkrtchian (2026), _Agreement rather than normalisation keeps depth reusable in looped language models_, arXiv:2610.31943.** [Page](https://arxiv.org/abs/2610.31943). **ABSTRACT only**, plus inspected HTML introduction/contents. Readout/last-layer refitting and boundary normalization directly concern architecture confounds; no exact B1 result apparent.
- **Pawel Olszowiec, Paulina Tomaszewska, Maciej Żelaszczyk, Michal Byra and Neo Christopher Chung (2026), _Monitoring latent reasoning in looped language models_, arXiv:2610.18127.** [Page](https://arxiv.org/abs/2610.18127). **ABSTRACT only.** Eight-model/five-lens analysis emphasizes carry versus reinjected-input loops; cannot exclude overlap hidden in unread methods.
- **Aur Shalev Merin (2026), _Present but Hard to Read: Probing Latent Reasoning with Minimal Pairs_, arXiv:2610.28498v2.** [Page](https://arxiv.org/abs/2610.28498). **ABSTRACT only.** Ouro probing with synthetic/random controls and readable-but-unused values is adjacent; not evidence for the signed Jacobian statistic.

## 3. Unverified leads and partial access

- **Fynn Kiwitt, Christopher Irwin, Riccardo Ali and Pietro Liò (2026), _Lyapunov Spectral Analysis of Loop Transformer Dynamics_.** The [OpenReview manuscript](https://openreview.net/forum?id=enDyNuSD3L) and PDF returned a browser-verification barrier. Thus the **full manuscript is unverified**. I did open and read the [official ICML Mechanistic Interpretability workshop poster](https://mechinterpworkshop.com/poster-pdfs/748.pdf), sections 1–7: finite-time QR/JVP spectra, Ouro-1.4B versus Huginn, sublayer attributions, and 50 tracked directions. The poster compares Lyapunov-vector growth, not readout-selected Rayleigh quotients or a trace baseline. Treat this as verified poster evidence, not full-paper review or confirmation of main-track ICML publication.
- **Matthew T. Kaufman, Mark M. Churchland, Stephen I. Ryu and Krishna V. Shenoy (2014), _Cortical activity in the null space: permitting preparation without movement_, DOI 10.1038/nn.3643.** Nature access failed, PMC presented a CAPTCHA, and PubMed did not expose the abstract through the opened page. The familiar output-null/output-potent analogy remains an **unverified primary-source lead in this session**, rather than a fully read citation. The same applies to the successor “A neural population mechanism for rapid learning” found in search.
- **M. F. Hutchinson (1989), _A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines_, DOI 10.1080/03610918908812806:** publisher 403. **Haim Avron and Sivan Toledo (2011), _Randomized algorithms for estimating the trace of an implicit symmetric positive semi-definite matrix_, DOI 10.1145/1944345.1944349:** publisher 403 and author PDF timeout. Historical leads corroborated by the opened Wimmer and Bai papers, not independently read originals.
- **V. I. Khokhlov (2006), _The Uniform Distribution on a Sphere in R^s. Properties of Projections. I_, DOI 10.1137/S0040585X97981846:** search lead and Wimmer reference; publisher article not opened.
- **Guillaume Hennequin, Tim P. Vogels and Wulfram Gerstner (2012), _Non-normal amplification in random balanced neuronal networks_:** search hit, not opened publisher/arXiv page. Useful predecessor for amplification versus dynamical slowing.
- Search hits for “Variance Reduction for Matrix Computations with Applications to Gaussian Processes,” “J-Spaces in Transformers: Pointwise Causality, Corpus Averaging, and Sparse Geometry,” and “Improved memory in recurrent neural networks with sequential non-normal dynamics” were not fully verified.
- Scope limit: forward citation discovery used title/identifier searches and newer papers' references, not an exhaustive citation-index export. Abstract-only screens above remain an uncertainty, especially the October multi-model monitoring paper.

## 4. Queries run

Queries used only technical terms, public paper titles/identifiers and literature names. No repository or machine-derived personal identifiers were submitted.

1. `"looped" "Jacobian" "Lyapunov" 2026`
2. `"Huginn" "Ouro" "Jacobian" 2026`
3. `"recurrent-depth" "Jacobian" readout`
4. `"Scaling latent reasoning via looped language models"`
5. `"transformer" "Jacobian" "unembedding"`
6. `"residual stream Jacobian"`
7. `"tuned lens" "transport" Jacobian`
8. `"transformer" "suppression neurons" erasure`
9. `"unembedding" "null space" "dark"`
10. `"deep equilibrium" "Jacobian" "trace"`
11. `"deep equilibrium" "output" "Jacobian regularization"`
12. `"non-normal" "recurrent" "readout"`
13. `"output-null" "output-potent" Kaufman 2014`
14. `"Jacobian" "Rayleigh quotient" neural`
15. `"Hutchinson" "sphere" "variance" trace estimator`
16. `"looped language" "self-gain"`
17. `"Huginn" "Lyapunov" ICML`
18. `"transformers" "iterative refinement" residual networks`
19. `"2609.01924"`
20. `"2606.24898"`
21. `"2502.05171" "Jacobian" 2026`
22. `"Ouro" "singular" "Jacobian"`
23. `"universal transformer" "Jacobian" spectrum`
24. `"latent reasoning" "non-normal"`
25. `"field of values" "Jacobian" neural networks`
26. `"transformer" "logit" "erasure" late layers`
27. `"Jacobian lens" Wang Reid`
28. `"Lyapunov Spectral Analysis of Loop Transformer Dynamics"`
29. `"Spectral Margins and Memory Routing in Looped Transformers"`
30. `"Verbalizable Representations Form a Global Workspace"`
31. `"non-normal" recurrent networks amplified readout`
32. `"deep equilibrium models" "Jacobian regularization"`
33. `"Hutchinson" "1989" trace estimator`
34. `"Tuned Lens" arxiv`
35. `"output-null" "Kaufman" 2014 nature`
36. `"unembedding" "dark subspace"`
37. `"Transformer" "suppression neurons"`
38. `"Lyapunov Spectral Analysis of Loop Transformer Dynamics"`
39. `"Spectral Margins and Memory Routing in Looped Transformers"`
40. `"Verbalizable Representations Form a Global Workspace"`
41. `"non-normal" "recurrent" "amplification" Hennequin`
42. `"Hutchinson" "1989" "stochastic"`
43. `"Kaufman" "Cortical activity in the null space"`
44. `"Lyapunov Spectral Analysis" "Ali"`
45. `"randomized trace" "unit sphere" variance estimator`
46. `"Rayleigh quotient" "numerical range" symmetric part`
47. `Avron Toledo 2011 randomized algorithms estimating trace implicit symmetric positive semidefinite matrices`
48. `Hutchinson 1989 stochastic estimator trace influence matrix DOI`
49. `"Scaling Latent Reasoning" "Zhu" arxiv`
50. `"uniform distribution on a sphere" Khokhlov 2006 projections`
51. `"spherical" "trace estimator" variance n+2`
52. `"Huginn" "Jacobian" "2026" November`
53. `"Ouro" "readout" "trace" Jacobian`
54. `"Huginn" "unembedding" "gain"`
55. `"looped" "Jacobian" "2025"`
56. `"Huginn" "Lu" "coda lens"`
57. `"output-potent" "non-normal" recurrent`
58. `"unembedding" "null space" transformer Jacobian`
59. `"Huginn" "Ouro" "2026" mechanistic November`
60. `"Off-Axis, On Purpose" transformer`
61. `"self-gain" Jacobian readout`
62. `"read-space" suppression Jacobian`
63. `"Huginn" "trace" Jacobian -site:github.com`
64. `"Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer"`

Citation paths followed included Wang–Reid → Gurnee and Lu (2025); Oskin → Fernando–Guitchounts and the unembedding-null-space discussion; Buehlmaier → Kerg and sequential non-normal RNN work; Bai → Hutchinson/Avron–Toledo; Wimmer → spherical-projection work. Several broad queries returned irrelevant hits; those are not evidence of absence.

## 5. Likely referee response

The Rayleigh quotient, spherical trace baseline, equilibrium-Jacobian probing, selective readout transport, and non-normal amplification are established ideas, and the report should not claim any of them individually. The potentially new result is the narrowly specified signed-self-gain contrast on these three checkpoints and protocols, provided it is reproducible and not generalized to the entire read space or all looped models. A referee will ask why sensitivity-weighted logit gradients are the relevant sampling distribution, how normalization and coda geometry affect it, and whether differences survive matched states, supervision and architectures. They will also reject inferring non-normality solely from 0.22 random RMS gain versus 1.3 selected-direction gain: a normal anisotropic matrix can do that, so B3 needs independent non-normality evidence or narrower wording.

