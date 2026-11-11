# Verdict: ACCEPT WITH CHANGES

The numerical results are sound; the revised interpretation still needs the corrections below. I independently recomputed the statistics from the saved arrays using local CPU NumPy, without importing or calling analyze_lread.py (or its analysis helpers), launching Modal/GPU work, or committing. Only this review file was written. All five columns reproduce the saved numerical summaries and displayed values to rounding **under the calculation actually implemented**. The stated aggregation convention is different and must be corrected.

## Required changes

1. **Correct the false statement that no answers changed.** `docs/steering_results.typ:68`: “at the doses tested it did not change which answer the second model gives.”
   
   Replace with: “At the tested doses no push made a different, targeted candidate become the answer. At the largest lever dose, five of 360 problem–candidate pairs changed to another answer.”
   
   These are five interventions on three distinct problems, not five distinct problems. The closing summary and Section 5's detailed counts are correct when restricted to candidates different from the unpushed answer. Limits item 2 (“the tested doses did not make the second model give the pushed answer”) should likewise say “did not make a different, pushed candidate become the answer”; pushes toward the already-correct candidate of course leave that candidate as the answer. Add “Five largest-dose lever interventions changed to other answers” there for clarity.

2. **Correct the fixed-point derivative and specify evaluation at zero push.** Section 1, lines 183–187: “its limit … is the derivative of the fixed point itself with respect to that vector”.
   
   Replace the two sentences with: “At zero push, the lever is the gradient of the score after T forced loops with respect to the repeated push. Under the spectral condition of Proposition 1 its limit is \(J^\top(I-J^\top)^{-1}G_k^\top\), the gradient of the equilibrium **score** with respect to the push. Indeed, if \(x(P)=F(x(P)+P)\), then \(D_Px(0)=(I-J)^{-1}J\). Differentiating through this implicit equation uses the same principle as training deep-equilibrium models.”
   
   The original equates a d-vector with a d-by-d derivative. The limiting formula itself is correct. The connection to implicit differentiation in [Bai, Kolter and Koltun (2019)](https://arxiv.org/abs/1909.01377) is legitimate; it does not identify the extra J factor as a universal DEQ formula. That factor comes from this experiment's before-loop injection.

3. **Describe the table's actual aggregation.** Section 4, lines 422–423: “Medians are over directions and then over problems; intervals are 95% bootstrap intervals over problems.”
   
   Replace with: “The trace estimate is the median across problems of the mean of their 16 probe self-gains. Excesses are medians across problems of within-problem median excesses. Absolute self-gains, one-step norms and self-cosines are pooled medians over the sampled directions and problems. Intervals resample problems.”
   
   This is a real discrepancy in the claimed statistic, not rounding: Huginn-digits task alpha is 0.041656 pooled (displayed 0.042), versus 0.047442 with the stated nested convention (would display 0.047). Huginn-truth C3 alpha is 0.018456 pooled versus 0.017399 nested; Ouro probe norm is 0.820783 pooled versus 0.827661 nested. Retain the registered calculations and correct the description.

4. **Restrict the self-gain conclusions to the measured quantity and samples; do not equate failure to detect a deficit with equality or causation.** Revise the following together:
   
   - Section 4, lines 407–409: “The self-gain … separates the two”. Replace with “Self-gain tests whether the sampled read directions differ from the random-direction mean on this scalar diagnostic; it does not explain the steering response by itself.”
   - Section 4, Huginn bullet: “The loop treats what the output reads differently from the rest of the state; the gradient push's loss of effect … is a case of this.” Replace with “The sampled read directions have lower signed self-gain than the random probes. This is consistent with the small one-loop task response, but does not establish its mechanism: the steering experiment uses gradients averaged across training problems, whereas this diagnostic uses each problem's own gradients.”
   - Section 4, L32 bullet: “Read directions are the same … and so are the task readouts”. Replace with “The C3 excess on sums is −0.0002 [−0.0017, 0.0004], so no C3 deficit is detected; equality was not tested. Task self-gains are also small in absolute value, although their median excess is positive, +0.0033 [0.0027, 0.0048].”
   - Section 4, end of L32 bullet: “a typical direction also returns nothing to itself.” Replace with “The sampled uniformly random directions also have small self-components after one loop. This does not measure the length of the resulting push or prove that all directions have small self-gain.”
   - Summary item 5, “both keep essentially nothing … there the one-loop loss of a gradient push is what happens to a typical direction”. Replace with “On L32's sums, the sampled probes, read directions and task gradients all have self-gains of only a few hundredths in magnitude. No C3 deficit is detected; this does not establish equality or explain the finite steering response. The diagnostic measures the component along the original direction, not the length of the propagated push.”
   - Section 5, first “Not shown” item: “on the second model the readout is not distinguished from a typical direction.” Replace with “on the second model's sums no C3 deficit was detected, while small positive excesses occur for the task gradients and, on the truth prompts, for C3.”
   
   Calling the five columns “three different pictures” is fair as a descriptive comparison of three model/protocol combinations, especially with the existing caption and caveat; it is not evidence that architecture alone caused the differences. Small signed self-gain must not become “erasure” or damping. The present paragraph explicitly distinguishing \(J^\top u\) from unmeasured \(Ju\) norms is correct and should stay. Statements about the scalar component of a **push** are valid because \(u^\top Ju=u^\top J^\top u\); using that equality does not require a measured JVP norm.
   
   Section 5's opening “the third item has a different character” should instead read “The third item's small one-loop target response coexists with different random-versus-read self-gain patterns; this does not identify the mechanism of either steering experiment.”

5. **Correct the description of sampled read directions and the state used.** Section 4, lines 414–416: “random combination of all the model's output scores (coefficients drawn independently from a centred normal law) … sampled in proportion to how sensitive it is”.
   
   Replace with: “For each of eight read directions, draw independent mean-zero Gaussian coefficients on ordinary output tokens, subtract their mean over those tokens, and differentiate the resulting linear combination of scores with respect to the state; then normalise the gradient. Special tokens are excluded (and L32's added tokens are excluded). This sampling favours strongly read directions, but is not uniform over the read space and can miss weakly read directions.”
   
   After subtracting the mean the coefficients are correlated; the original coefficients are independent, the actual ones are not. “In proportion” is not the angular distribution after normalisation. If H is the derivative of these output scores and C the centering projection, the unnormalised covariance is \(H^\top C H\), not a probability density proportional to sensitivity.
   
   Also replace the opening “On each problem, at the settled state” with “On each problem, at the state used by its protocol (approximately settled for Huginn and L32, the final loop for Ouro)”. The later sources paragraph supplies this distinction, but the opening currently contradicts it.

6. **Do not infer the performance of a restricted fitted monitor from inclusion of function classes.** Section 5, lines 535–539: “one of the linear readers already fitted … restricting w could help only by needing fewer fitting problems”.
   
   Replace with: “A fixed linear reader whose weight lies in a fixed lever span belongs to the affine function class used by the monitoring experiment. That experiment fitted coordinate-standardised, L2-penalised logistic regressions, with the penalty chosen by nested cross-validation; it did not evaluate every weight in that class. A lever-span restriction would change the inductive bias and could improve or worsen generalisation at the same training size. That restriction was not tested, so no consequence for error detection follows.”
   
   The actual primary readers are affine over the whole state (2,048 coordinates), with an unpenalised intercept, training-fold coordinate standardisation, and penalty \(\lambda\|w\|^2/2\) in standardised coordinates. Nine penalties from 1e−6 to 1e2 are selected in four inner folds; five outer folds score held-out predictions; three split seeds are used. The thin SVD in `mon_probe.fit_path` is an exact solver reduction to the data row space, not a restriction to 128 principal components. The latter is a separate robustness check. “Only by needing fewer fitting problems” is too strong: a useful restriction can improve bias/variance and generalisation at the same n. The sentence “a lever is a sensitivity … not a direction along which the states of right and wrong answers differ” should say “does not by itself establish a direction separating right and wrong answers”; such correlation is possible, just unmeasured.

7. **Define the output scores and probabilities for a mathematical reader.** Section 1, “Readout”, currently introduces scores without specifying which model quantity they are; Section 4 then expands to all output scores and Section 6 uses confidence.
   
   Add after the first readout sentence: “The model outputs one real score, called a logit, for each possible next token (a token is an element of its text vocabulary). It assigns probabilities \(p_j=\exp(\ell_j)/\sum_i\exp(\ell_i)\) and chooses the token with largest score. Here the ten candidates are represented by single tokens; \(z_k=\ell_k-\frac1{10}\sum_j\ell_j\). Section 4 also uses scores for ordinary vocabulary tokens, meaning tokens used for text rather than reserved control symbols.”
   
   This connects the centred score, chosen answer, read-direction sampling and confidence without assuming ML terminology.

## Optional suggestions and assessment

- The numerical classifier was applied as fixed in Draft 1.2. A and B use the Z branch because median tau is at most 0.05. A's C3 interval crosses zero: NOT DETECTED, no positive-excess sublabel. B's interval is wholly positive: NOT DETECTED / POSITIVE EXCESS. Both tau intervals are inside ±0.05. Coordinate attribution is correctly DESCRIPTIVE ONLY when no C3 deficit is detected. This is not an equivalence test. The registered absolute task norms and cosines are in the saved summaries and generated columns but not in Table 2; displaying them would make the Draft 1.2 requirement more visible.
- The drift amendment is defensible as a disclosed, pilot-informed correction, not a fully untouched preregistration. The earlier lever evaluation arrays independently give median drift 2.108285866e−6 and maximum 2.090263692e−4 at loop 64; the original 1e−6 gate would fail on the reference states themselves. The 1e−5 median threshold was changed before production, with full-vector state and gradient identity gates added and sensitivity filtering retained. The Section 4 caveat adequately discloses the pilot, timing, reason and Section 8 source. Prefer “final design and reading rule fixed before production” to “design … fixed before the run”; optionally state both old and new thresholds explicitly.
- Zero contexts are removed from A by the 1e−3 drift filter. One is removed from B; its C3 excess remains positive, 0.004520 [0.003837, 0.006552], and both the main label and positive-excess sublabel persist. Filtered family adjoint medians still pass. B's 34/40 measures preference between True and False, not unrestricted whole-vocabulary correctness; the write-up says “prefers the labelled answer” correctly.
- Section 1's \(\alpha=c_1/c_0\) identity is correct for a nonzero **local** gradient and a positive-length push along it; add the nonzero condition. It is not the ratio for the averaged training direction U on an arbitrary held-out problem. The uniform-sphere identity \(E\alpha=\mathrm{tr}(J)/d\) is correct for any real J. For precision, replace “Whether … differently from a typical one can therefore be read” with “The difference from the uniformly random-direction mean is the excess”; a mean need not be typical without distributional evidence. The sampled probe ranges here supply such evidence locally.
- The warning about opposite-sign eigenvalues is a valid example of trace cancellation. For a possibly nonsymmetric J, the symmetric part \((J+J^\top)/2\) directly controls all self-gains; an optional sentence stating this would prevent a reader from treating the diagnostic as a spectral-norm estimate.
- Sections 2 and 3's steering findings do not conflict with the diagnostic. In particular Section 2's measured push length ratio 1.17 is a separate forward-propagation measurement, and Section 3's “the pattern … replicates” explicitly defines that pattern by R1/R2 and says the label certifies no more. Those sentences do not need reversal. Section 2's headline “Levers have an effect, gradient and centroid pushes do not” would be clearer as “Levers have a larger mean target effect than the tested gradient and centroid pushes”; small mean effects do not prove exact zero. Section 5's three empirical list items remain supported; the interpretive opening and first “Not shown” item need the replacements above.
- The new self-gain data do not establish a shared causal mechanism and do not refute the common linear-response formula. The same calculus can give different directional geometry. The broad architecture description (input reinjection, recurrent core, final layers) is appropriately coarse.
- The Table 2 last-row caption calls these labels “the label … fixed in advance”. Huginn's displayed “deficit” is a descriptive reduction of its earlier control analysis, not the same primary classifier as Ouro/L32. Say “qualitative deficit summary; the underlying registered reading rules differ between runs.” Ouro's positive C3 interval could also be labelled “positive excess” descriptively, without changing its registered NOT DETECTED label.
- The counted answers are final forced-run argmaxes, not evidence about transient answers at every intermediate loop.
- Runtime metadata give 149.061547 s production, 36.756206 s first smoke, and 21.184742 s second smoke: 2.5 and 3.5 minutes as printed. G1's saved maximum is 6.750155990e−14, displayed 6.8×10⁻¹⁴. These are checks of saved gate results; I did not rerun the model.

## Recomputed numbers against displayed values

Independent computation: \(\alpha=\mathrm{gram3}_{0,1}\); \(\tau_i\) is its mean over families 4/5; C3/task excess is the within-context median of \(\alpha-\tau_i\), followed by the across-context median. Bootstrap: 5,000 context resamples, NumPy default_rng(0), percentile endpoints 2.5% and 97.5%. Norm and cosine use \(\sqrt{\mathrm{gram3}_{1,1}}\) and \(\alpha/\sqrt{\mathrm{gram3}_{1,1}}\), pooled medians. No production analysis function was called.

The recomputed tau and C3/task excess triples agree exactly with the saved summary triples for every column (maximum absolute discrepancy 0). All numerical table entries below agree with rs_cols to displayed rounding. The aggregation-description mismatch is required change 3, not a mismatch with the arrays.


### Huginn, true/false

| Quantity | Recomputed | Displayed (rs_cols) |
|---|---:|---:|
| Contexts | 40 | 40 |
| Median probe-mean tau [CI] | 0.24078744 [0.23960822, 0.24159476] | 0.241 [0.240, 0.242] |
| C3 pooled alpha | 0.018455897 | 0.018 |
| C3 excess [CI] | -0.22357964 [-0.23053026, -0.21568041] | −0.224 [−0.231, −0.216] |
| task pooled alpha | 0.12190572 | 0.122 |
| task excess [CI] | -0.11880616 [-0.12143589, -0.11565922] | −0.119 [−0.121, −0.116] |
| C3 below all probes | 100% | 100% |
| probes norm / self-cosine | 0.63849156 / 0.37599095 | 0.64 / 0.38 |
| C3 norm / self-cosine | 0.62222736 / 0.029566451 | 0.62 / 0.03 |
| task norm / self-cosine | 0.70953762 / 0.17142003 | 0.71 / 0.17 |

Task intervals and task norm/cosine are present in rs_cols but not displayed in Table 2.

### Huginn, digits

| Quantity | Recomputed | Displayed (rs_cols) |
|---|---:|---:|
| Contexts | 20 | 20 |
| Median probe-mean tau [CI] | 0.25750856 [0.25585703, 0.25936897] | 0.258 [0.256, 0.259] |
| C3 pooled alpha | 0.058539744 | 0.059 |
| C3 excess [CI] | -0.19983711 [-0.20977908, -0.19228904] | −0.200 [−0.210, −0.192] |
| task pooled alpha | 0.041655890 | 0.042 |
| task excess [CI] | -0.21273993 [-0.23610085, -0.19826932] | −0.213 [−0.236, −0.198] |
| C3 below all probes | 100% | 100% |
| probes norm / self-cosine | 0.67236314 / 0.38337567 | 0.67 / 0.38 |
| C3 norm / self-cosine | 0.64249558 / 0.090921143 | 0.64 / 0.09 |
| task norm / self-cosine | 0.82368074 / 0.052368315 | 0.82 / 0.05 |

Task intervals and task norm/cosine are present in rs_cols but not displayed in Table 2.

### L32, sums

| Quantity | Recomputed | Displayed (rs_cols) |
|---|---:|---:|
| Contexts | 40 | 40 |
| Median probe-mean tau [CI] | -0.0014300664 [-0.0018031149, -0.00078992203] | −0.001 [−0.002, −0.001] |
| C3 pooled alpha | -0.0023345353 | −0.002 |
| C3 excess [CI] | -0.00023570838 [-0.0016659809, 0.00038476530] | 0.000 [−0.002, 0.000] |
| task pooled alpha | 0.0024421929 | 0.002 |
| task excess [CI] | 0.0032699106 [0.0026610297, 0.0048343758] | +0.003 [0.003, 0.005] |
| C3 below all probes | 11.875% | 12% |
| probes norm / self-cosine | 0.21296359 / -0.0058173306 | 0.21 / −0.01 |
| C3 norm / self-cosine | 0.23522979 / -0.0087381653 | 0.24 / −0.01 |
| task norm / self-cosine | 0.23952623 / 0.010747175 | 0.24 / 0.01 |

Task intervals and task norm/cosine are present in rs_cols but not displayed in Table 2.

### L32, true/false

| Quantity | Recomputed | Displayed (rs_cols) |
|---|---:|---:|
| Contexts | 40 | 40 |
| Median probe-mean tau [CI] | -0.000012362258 [-0.00056994291, 0.00029761155] | 0.000 [−0.001, 0.000] |
| C3 pooled alpha | 0.0053889784 | 0.005 |
| C3 excess [CI] | 0.0045843455 [0.0038707019, 0.0065737790] | +0.005 [0.004, 0.007] |
| task pooled alpha | 0.0040502019 | 0.004 |
| task excess [CI] | 0.0041034074 [-0.0014790686, 0.010900267] | +0.004 [−0.001, 0.011] |
| C3 below all probes | 1.5625% | 2% |
| probes norm / self-cosine | 0.23292972 / -0.00065845172 | 0.23 / 0.00 |
| C3 norm / self-cosine | 0.24243676 / 0.021681446 | 0.24 / 0.02 |
| task norm / self-cosine | 0.46687477 / 0.0097121758 | 0.47 / 0.01 |

Task intervals and task norm/cosine are present in rs_cols but not displayed in Table 2.

### Ouro-1.4B, true/false

| Quantity | Recomputed | Displayed (rs_cols) |
|---|---:|---:|
| Contexts | 40 | 40 |
| Median probe-mean tau [CI] | 0.059837081 [0.057047511, 0.061072825] | 0.060 [0.057, 0.061] |
| C3 pooled alpha | 0.080647925 | 0.081 |
| C3 excess [CI] | 0.021788799 [0.015959298, 0.025271835] | +0.022 [0.016, 0.025] |
| task pooled alpha | 0.15463118 | 0.155 |
| task excess [CI] | 0.096545643 [0.085623359, 0.10372036] | +0.097 [0.086, 0.104] |
| C3 below all probes | 3.75% | 4% |
| probes norm / self-cosine | 0.82078304 / 0.071927689 | 0.82 / 0.07 |
| C3 norm / self-cosine | 0.98786824 / 0.082547610 | 0.99 / 0.08 |
| task norm / self-cosine | 1.8417059 / 0.086338932 | 1.84 / 0.09 |

Task intervals and task norm/cosine are present in rs_cols but not displayed in Table 2.

### L32 absolute ranges and gates

| Quantity | Set A | Set B | Comparison |
|---|---:|---:|---|
| probes alpha min / max | -0.016974079 / 0.015456573 | -0.015360625 / 0.017327404 | RS: −0.017 / 0.015; −0.015 / 0.017 |
| C3 alpha min / max | -0.032384288 / 0.019033848 | -0.014352506 / 0.026425194 | RS: −0.032 / 0.019; −0.014 / 0.026 |
| task alpha min / max | -0.018583246 / 0.037752909 | -0.017798258 / 0.021093776 | RS: −0.019 / 0.038; −0.018 / 0.021 |
| Largest unit-start error | 3.5527137e-15 | 3.5527137e-15 | Matches saved summary |
| diag(x1) versus alpha | 1.1796120e-16 | 1.0061396e-16 | Matches saved summary |
| Minimum read-gradient norm | 0.031813601 | 0.026838408 | Matches saved summary |
| Minimum estimated tau(B²) | 0.019332883 | 0.025496554 | Matches saved summary |
| Labelled answers preferred | 40.000000 | 34.000000 | Matches saved summary |
| Median drift | 0.0000021082859 | 9.4071263e-9 | Matches saved summary |
| Maximum drift | 0.00020902637 | 0.0049587850 | Matches saved summary |
| Drift exclusions | 0.0000000 | 1.0000000 | Matches saved summary |
| Complete / finite | true / true | true / true | Pass |
| Exact 40 eval pids | true | n/a | Pass |
| Maximum relative task-gradient identity error | 8.2991310e-14 | n/a | RS: 8.3 × 10⁻¹⁴ |
| Maximum relative whole-state identity error | 2.9253391e-14 | n/a | RS: 2.9 × 10⁻¹⁴ |
| Filtered maximum family adjoint median | 5.1099669e-9 | 4.9359189e-9 | Pass |

All main-set family adjoint medians (family codes 0–6):

| Family | A | B |
|---|---:|---:|
| 0 | 4.4049832e-9 | 1.9863433e-9 |
| 1 | 4.1189897e-9 | 3.1553541e-9 |
| 2 | 5.1099669e-9 | 3.4308472e-9 |
| 3 | 4.2337408e-9 | 4.5494621e-9 |
| 4 | 4.9180862e-9 | 5.0335099e-9 |
| 5 | 4.5066783e-9 | 3.7235590e-9 |
| 6 | 4.5032222e-9 | 4.3955372e-9 |

The largest is 5.109966869e−9, matching RS.adjoint_max = 5.1×10⁻⁹. All unit/x1/positivity/adjoint/completeness gates pass. The A drift gate passes the amended 1e−5 threshold, not the original 1e−6 threshold. Correctness is A 40/40, B 34/40, matching the text. There are 640 probe observations in each set. RS.tau_A, c3_A and c3_B are the same triples above printed at four/four/three decimal places; no mismatch.

### Changed final answers

Each row has 360 eligible problem–candidate pairs (40 problems × 9 candidates different from their unpushed answer). Rows were selected independently by row_push and row_rho; baseline was the sham row. Candidate token IDs were mapped from ans and ans_token.

| Family | Dose | Target became answer | Changed to another answer | All 400 interventions changed | Generated counts |
|---|---:|---:|---:|---:|---|
| lever | 0.01 | 0 | 0 | 0 | Match |
| lever | 0.03 | 0 | 0 | 0 | Match |
| lever | 0.09 | 0 | 5 | 5 | Match |
| gradient | 0.01 | 0 | 0 | 0 | Match |
| gradient | 0.03 | 0 | 0 | 0 | Match |
| gradient | 0.09 | 0 | 0 | 0 | Match |
| centroid | 0.01 | 0 | 0 | 0 | Match |
| centroid | 0.03 | 0 | 0 | 0 | Match |
| centroid | 0.09 | 0 | 0 | 0 | Match |

All five changes occur on three distinct problems. Thus Section 5's detailed count, summary closing target-flip count, and numbers.json flip keys are correct; the summary opening's “did not change which answer” is not.

### Per-context probe means

In saved context order (row index 0–39); these are the independently computed tau_i, not pooled alpha medians.

| Row | Set A tau_i | Set B tau_i |
|---:|---:|---:|
| 0 | 0.0016298802 | 0.0020739249 |
| 1 | -0.0010229287 | -0.000040088938 |
| 2 | -0.0013447850 | 0.0011085574 |
| 3 | -0.00077764960 | -0.00014325295 |
| 4 | -0.0039616894 | -0.000095979030 |
| 5 | -0.00078646954 | -0.0012708330 |
| 6 | -0.0029419790 | 0.00016334598 |
| 7 | -0.0026613442 | 0.00020943320 |
| 8 | -0.0024942337 | 0.0022082395 |
| 9 | -0.0021777351 | 0.00012452710 |
| 10 | -0.0012722111 | 0.0020272210 |
| 11 | -0.00012672123 | 0.000031025083 |
| 12 | -0.0019477320 | -0.00061740576 |
| 13 | -0.0016413446 | -0.0015678714 |
| 14 | -0.0024804724 | 0.00037493675 |
| 15 | -0.00024027100 | -0.0012245738 |
| 16 | -0.0014976431 | 0.000015364422 |
| 17 | -0.00075410756 | 0.00033161405 |
| 18 | -0.00038887583 | -0.00052248007 |
| 19 | 0.00084071460 | -0.00077045772 |
| 20 | -0.0016694992 | -0.00098215157 |
| 21 | -0.0018134662 | -0.0020766995 |
| 22 | -0.00019662427 | 0.00036466721 |
| 23 | -0.0034251990 | 0.00091412016 |
| 24 | -0.0034715310 | 0.0012473988 |
| 25 | -0.00079337453 | 0.0016700414 |
| 26 | -0.0016721206 | 0.0016956580 |
| 27 | -0.0013624897 | -0.00088443699 |
| 28 | -0.0035015616 | 0.0024147977 |
| 29 | -0.0016602199 | 0.0026329551 |
| 30 | 0.00018286018 | 0.0012758326 |
| 31 | -0.00064941452 | -0.0017371074 |
| 32 | -0.0037431255 | -0.00022483713 |
| 33 | -0.00064696523 | 0.00026360905 |
| 34 | -0.0019660330 | -0.0018794466 |
| 35 | -0.0017927637 | -0.0021370339 |
| 36 | -0.00023024265 | -0.0036298947 |
| 37 | 0.00025592030 | -0.0015678929 |
| 38 | -0.0019154655 | -0.00035908727 |
| 39 | -0.0010296621 | -0.00091487232 |

