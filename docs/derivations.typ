#set document(title: "Deriving the Huginn results: a workbook")
#set page(paper: "us-letter", margin: (x: 1in, y: 0.9in), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#show heading.where(level: 1): it => [#v(0.6em) #it #v(0.2em)]

#let boxed(title, color, body) = block(
  fill: color, inset: 9pt, radius: 3pt, width: 100%, breakable: true,
  [*#title.* #body],
)
#let moves(body) = boxed("Moves", rgb("#eef3fb"), body)
#let check(body) = boxed("Check against the data", rgb("#eef8ee"), body)
#let warn(body) = boxed("Careful", rgb("#fbf3e6"), body)
#let try(body) = boxed("Try it yourself first", rgb("#f4eefb"), body)

#align(center)[
  #text(17pt, weight: "bold")[Deriving the Huginn results: a workbook]
  #v(0.2em)
  #text(10pt)[Companion to `huginn_addition/WRITEUP.md`. Draft, 2026-11-07.]
]

#v(0.6em)

*How to use this.* Every result in the study comes from the same short recipe, and the derivations reuse about
seven moves. The first section gives the recipe and the moves. D1–D10 apply them, one derivation per section, most ending with a
check against a number you measured. Then come exercises (answers at the end) and places where a real analytical
result might be found.

To learn these, don't read the derivation first. Read the *claim*, cover the rest, and try to get there with the recipe.
Then compare. If you get stuck, do the *scalar case* ($J = lambda$, a number) first: almost every step here is a
one-line fact about geometric series in the scalar case, and the matrix version is the same algebra with the order of
multiplication kept.

*Notation.* $A^top$ is the transpose. Superscripts like $J^t$ or $J^T$ are matrix powers. $T$ is the number of loops.

= The recipe and the toolkit

*The recipe.*
+ *Write the system as an update rule.* What is the state, what is held fixed, what is applied every step?
+ *Find the reference state.* Here, the fixed point $x^*$ with $x^* = F(x^*)$.
+ *Perturb and linearize.* Write $s = x^* + delta$ and keep first-order terms.
+ *Solve the linear recurrence.* Unroll a few steps, spot the pattern, sum it.
+ *Map to what you measured.* Usually the readout, via its gradient $g$.
+ *Check.* Scalar case, limits ($J = 0$, $T -> infinity$), and one number from the data.

*The moves.*
- *M1. First-order Taylor / chain rule.* $F(x^* + delta) = F(x^*) + J delta + O(norm(delta)^2)$.
- *M2. Unroll a linear recurrence.* Compute $delta_1, delta_2, delta_3$ by hand; the pattern is the answer.
- *M3. Geometric (Neumann) series.* $sum_(t=0)^infinity J^t = (I - J)^(-1)$ when the spectral radius $rho(J) < 1$.
  Scalar version: $1 + lambda + lambda^2 + dots.c = 1 / (1 - lambda)$.
- *M4. The adjoint trick.* $g^top (A P) = (A^top g)^top P$. Turns "how does the readout respond to $P$" into "inner
  product of $P$ with one fixed vector". Then Cauchy–Schwarz gives the best $P$.
- *M5. Diagonalize, then think one mode at a time.* If $J v = lambda v$, everything acts on $v$ as the number $lambda$.
- *M6. Block-triangular structure.* Causality makes the full-sequence Jacobian block lower-triangular; its eigenvalues
  are those of the diagonal blocks.
- *M7. Normal forms.* Near a bifurcation, reduce to the one critical direction and keep the lowest-order terms that
  matter.

= Setup

The core block $F$ takes the adapter output $w = W_s s + W_e E$ (Huginn's adapter has no bias), where $s$ is the recurrent state and $E$ is the
prelude output, re-injected every loop:
$ s_(t+1) = F(s_t; E), quad quad x^* = F(x^*; E) quad "(fixed point at answer position" q")". $
Define, at the fixed point,
$ J = (partial F_q) / (partial s_q), quad quad g = nabla R(x^*), $
where $R$ is the readout you care about (a centered digit logit, or $log p("True") - log p("False")$).

= D1. Linearization, and why the own-position Jacobian is exact

*Claim.* Small perturbations of the answer-position state evolve as $delta_(t+1) = J delta_t$, exactly to first order.

*Derivation (M1).* Put $s_t = x^* + delta_t$:
$ x^* + delta_(t+1) = F(x^* + delta_t) = F(x^*) + J delta_t + O(norm(delta_t)^2) = x^* + J delta_t + dots.c $
Cancel $x^*$. Done.

*Why only position $q$ (M6).* The full state is $S = (s_1, dots, s_q)$. The full Jacobian has blocks
$cal(J)_(i j) = partial F_i \/ partial s_j$. Attention is causal, so position $i$ never reads position $j > i$:
$cal(J)_(i j) = 0$ for $j > i$. The matrix is block lower-triangular:
$ cal(J) = mat(cal(J)_(1 1), 0, dots.c, 0; cal(J)_(2 1), cal(J)_(2 2), dots.c, 0; dots.v, , dots.down, dots.v; cal(J)_(q 1), cal(J)_(q 2), dots.c, cal(J)_(q q)). $
A perturbation only at $q$ is $delta S = (0, dots, 0, delta_q)$. Multiplying by $cal(J)$ uses only the last column,
whose only nonzero block is $cal(J)_(q q) = J$. So the perturbation stays at $q$ and evolves by $J$ alone. This is why
the lever analyses could use $J$ without approximation, and why they say nothing about cross-position effects.

= D2. One-shot edits are erased

*Claim.* Add $P$ once; the effect on the state and readout dies away.

*Derivation (M2, M5).* $delta_0 = P$, so $delta_1 = J P$, $delta_2 = J^2 P$, and $delta_t = J^t P$. The readout
change is $Delta R_t approx g^top J^t P$. If $J v = lambda v$ with $|lambda| < 1$, the component of $P$ along $v$ is
multiplied by $lambda^t -> 0$. So every component decays at a rate set by its eigenvalue, and the slowest by
$rho(J)$.

#check[With $rho approx 0.78$: $0.78^(16) = e^(16 ln 0.78) approx e^(-3.97) approx 0.02$. That matches the push
decaying to about zero within roughly 16 loops.]

#warn[Eigenvalues control the *long-run* rate, not the short run. For a non-normal $J$ (D5) a vector can *grow*
for a step even though all $|lambda| < 1$. You measured exactly this: $norm(J P) \/ norm(P) = 1.17$ for
readout-gradient pushes, while $rho approx 0.78$.]

= D3. Persistent forcing moves the fixed point

*Claim.* Adding $P$ before every loop gives
$ delta_T = A_T P, quad A_T = sum_(t=1)^T J^t, quad quad delta_infinity = (I - J)^(-1) J P. $

*Derivation (M1, M2, M3).* The update is $s_(t+1) = F(s_t + P)$. Linearize: $delta_(t+1) = J(delta_t + P)$, with
$delta_0 = 0$. Unroll:
$ delta_1 &= J P \
  delta_2 &= J(J P + P) = J^2 P + J P \
  delta_3 &= J(J^2 P + J P + P) = J^3 P + J^2 P + J P \
  delta_T &= (J + J^2 + dots.c + J^T) P = A_T P. $
Let $T -> infinity$ and use M3: $sum_(t >= 1) J^t = J sum_(t >= 0) J^t = J (I - J)^(-1) = (I - J)^(-1) J$ (the two
commute because both are functions of $J$).

*Shortcut worth knowing.* Skip the unrolling: at the new steady state, $delta = J(delta + P)$, so
$(I - J) delta = J P$. Same answer in one line. Use the unrolled form when you need finite $T$.

*Scalar case.* $J = lambda$ gives $delta_infinity = lambda \/ (1 - lambda) dot P$. A mode with $lambda = 0.78$ is
amplified $3.5 times$; one with $lambda = 0.5$ only $1 times$; one with $lambda < 0$ is *reversed* in sign (shrunk).

*Context equivalence.* $F$ sees $s$ only through $w = W_s s + W_e E$. Adding $P$ to $s$ changes $w$ by $W_s P$.
Changing $E$ by $delta E$ changes $w$ by $W_e delta E$. So a persistent state push is *the same thing* as a context
edit with $W_e delta E = W_s P$. That's why persistent forcing is the natural intervention: it's what a context change
does every loop.

#check[I computed this one; it isn't in the write-up. The steering push was along the digit-circle plane, where the
measured block of $J$ is about $0.71 dot I$. The scalar formula predicts a displacement of
$0.71 \/ (1 - 0.71) approx 2.45$ times the push norm. You observed $2.5 times$. Treat it as "consistent with": the full
dose was outside the linear regime for the *readout* (Section 5 of the write-up).]

= D4. The readout, and the optimal lever

*Claim.* Among pushes with $norm(P) = r$, the one that moves the readout most after $T$ loops is
$ P^* = r dot (A_T^top g) / norm(A_T^top g), quad "with gain" quad r norm(A_T^top g). $

*Derivation (M4).* The readout change is
$ Delta R_T approx g^top delta_T = g^top A_T P = (A_T^top g)^top P = sum_(t=1)^T c_t, quad c_t = g^top J^t P. $
For fixed $a = A_T^top g$, maximize $a^top P$ over $norm(P) = r$. Cauchy–Schwarz: $a^top P <= norm(a) norm(P)$, with
equality when $P$ points along $a$. Done.

*How you computed it.* You never form $A_T$. Run the adjoint recurrence
$ w_1 = J^top g, quad w_(t+1) = J^top w_t, quad A_T^top g = sum_(t=1)^T w_t. $
Each step is one vector–Jacobian product (one backward pass through the block). That's the 48 VJPs per item for $d^*$.

*Corollary: "fractions ≈ cosines".* For any unit direction $u$,
$ (Delta R(u)) / (Delta R(P^* "at norm" 1)) = (a^top u) / norm(a) = cos(u, a). $
So in the linear regime, *the fraction of the optimal effect a direction achieves is exactly its cosine with the
item lever*. Your truth table had fractions $0.68 \/ 0.39 \/ 0.09$ against cosines $0.70 \/ 0.40 \/ 0.10$.

#warn[This is exact *per item*. The population $d^*$ averages normalized item levers, so for the population it is an
approximation whose quality depends on how tightly the item levers cluster.]

= D5. Why the lever is nearly orthogonal to $g$: non-normality

*Claim.* If $J$ were symmetric with eigenvalues in $(0, 1)$, the lever would stay positively aligned with $g$.
(Normal is *not* enough: a rotation is normal and can point the lever against $g$. See `analytic.pdf`.) Near-
orthogonality (cosine $approx 0.1$) and gradient pushes dying after one step both point to a strongly non-normal $J$.

*Symmetric case (M5).* Write $J = sum_i lambda_i v_i v_i^top$ with orthonormal $v_i$. Then $A_T$ has the same
eigenvectors with eigenvalues $phi_T (lambda_i) = sum_(t=1)^T lambda_i^t$, and
$ A_T^top g = sum_i phi_T (lambda_i) (v_i^top g) v_i, quad quad g^top A_T^top g = sum_i phi_T (lambda_i) (v_i^top g)^2. $
If all $lambda_i in (0, 1)$, every weight $phi_T > 0$, so the lever has a positive inner product with $g$: it is $g$
with its components reweighted, not rotated away.

*A $2 times 2$ toy that reproduces your mechanism findings.* Let the readout see only coordinate 1:
$ J = mat(lambda_1, a; 0, lambda_2), quad g = e_1 = mat(1; 0), quad |lambda_1|, |lambda_2| < 1, quad a "large". $
- Lever (one step): $J^top g = (lambda_1, a)^top$. If $lambda_1$ is small, it points almost along $e_2$:
  *orthogonal to the readout direction.*
- Push along $g$: $c_0 = g^top e_1 = 1$, $c_1 = g^top J e_1 = lambda_1$. It *dies after one step*
  ($|c_1 \/ c_0| = |lambda_1|$).
- Push along $e_2$: $c_0 = 0$, $c_1 = g^top J e_2 = a$. *Nothing immediately, a lot after one step.*
- Growth: $norm(J e_2) = sqrt(a^2 + lambda_2^2)$ can exceed 1 although both eigenvalues are below 1.

Compare: readout-gradient pushes suppressed after one step ($|c_1 \/ c_0| = 0.05$); levers mostly outside the readout
span (in-span ratio 0.12); the readout span fed from outside itself ($rho = 0.17$); eigenvector condition number around 75.
These are all the same fact: $J^top$ moves $g$ out of its own span.

*The $c_0$ sign identity.* For an item lever $hat(a) = A_T^top g \/ norm(A_T^top g)$,
$ c_0 = g^top hat(a) = (g^top A_T^top g) / norm(A_T^top g) = (g^top A_T g) / norm(A_T^top g), $
using that a scalar equals its own transpose. The numerator is the accumulated response to a push along $g$ itself.
So the lever starts negative exactly when a $g$-push ends negative. In the symmetric case with positive eigenvalues
that can't happen; it needs negative eigenvalues or a skew (circulating) part of $J$. `analytic.pdf` makes this exact. The identity links the two signs; it does not
say why the sign is negative.

= D6. Cheap levers are truncations

*One-step lever.* Take $T = 1$: $A_1^top g = J^top g$. One backward pass.

*Adapter lever.* $s$ enters the block only through $w = W_s s + dots$, so by the chain rule
$ J = M W_s, quad M = (partial F) / (partial w), quad J^top g = W_s^top M^top g. $
Split $M$ by the last write on each path: $M = M_"direct" + M_"MLP" + M_"attn"$. If the direct path (residual plus
normalizations) acts on these covectors like a scalar, $M_"direct"^top g approx kappa g$, then the direct part of
$J^top g$ is $kappa W_s^top g$. You measured $kappa approx 2.76$ and cosine $0.9987$.

*Why RMSNorm can look scalar.* $"RMSNorm"(x) = gamma dot.o x \/ "rms"(x)$, with $"rms"(x) = norm(x) \/ sqrt(d)$.
Differentiating (quotient rule):
$ (partial "RMSNorm") / (partial x) = 1 / "rms"(x) op("diag")(gamma) (I - (x x^top) / norm(x)^2). $
The bracket projects out the direction of $x$. On covectors nearly orthogonal to $x$ it is close to the identity, so
the map is about $op("diag")(gamma) \/ "rms"(x)$, a scalar if $gamma$ is roughly flat on those directions. That's the
untested "candidate explanation" in §4.4 of the write-up.

= D7. Period-2 cycles: the flip normal form

*Claim.* Near a real eigenvalue crossing $-1$, the dynamics along that eigendirection reduce to
$ f(u) = -(1 + mu) u + c u^3, quad lambda = -(1 + mu), $
which predicts a period-2 cycle with amplitude $u^2 = mu \/ c$ and two-step multiplier $1 - 4 mu$.

*Why this form (M7).* Restrict to the critical direction $u$ (the center manifold). The linear term is the eigenvalue.
A quadratic term can be removed by a small change of coordinates near a flip (it ends up inside the effective $c$), so
the first term that matters is cubic.

*Derivation.* Apply $f$ twice and keep terms up to $u^3$ and first order in $mu$:
$ f(f(u)) &= -(1 + mu) f(u) + c f(u)^3 \
  &= (1 + mu)^2 u - c(1 + mu) u^3 - c (1 + mu)^3 u^3 + O(u^5) \
  &approx (1 + 2 mu) u - 2 c u^3. $
A period-2 point is a fixed point of $f compose f$ that isn't a fixed point of $f$:
$ (1 + 2 mu) u - 2 c u^3 = u quad => quad u^2 = mu / c. $
Stability: differentiate $f compose f$ at the cycle,
$ (f compose f)'(u) = 1 + 2 mu - 6 c u^2 = 1 + 2 mu - 6 mu = 1 - 4 mu. $
For small $mu > 0$ and $c > 0$ this is just below 1: a stable cycle that appears as $lambda$ passes $-1$
(supercritical). With $c < 0$ the cycle would exist only on the other side and be unstable (subcritical).

#check[Along the interpolation $alpha$, the tracked eigenvalue crossed $-1$; the squared amplitude was linear in
$mu$ (slope 0.0125 in your units), and the measured two-step multiplier matched $1 - 4 mu$. That is the textbook
supercritical flip, which is why the write-up can say "consistent with a flip bifurcation".]

= D8. Under-relaxation removes flips without changing answers

*Claim.* Replace $F$ at a token by $G(s) = (1 - beta) s + beta F(s)$ with $0 < beta <= 1$. Then:
+ $G$ has exactly the same fixed points as $F$.
+ Each eigenvalue maps as $lambda |-> 1 - beta + beta lambda$.
+ To move $lambda$ to a target $lambda'$, use $beta = (1 - lambda') \/ (1 - lambda)$. The flip disappears when
  $beta < beta^* = 2 \/ (1 - lambda)$.

*Derivation.* (1) $G(x) = x <=> (1 - beta) x + beta F(x) = x <=> beta (F(x) - x) = 0 <=> F(x) = x$.
(2) Differentiate: $J_G = (1 - beta) I + beta J$. If $J v = lambda v$ then $J_G v = (1 - beta + beta lambda) v$.
(3) Solve $1 - beta + beta lambda = lambda'$ for $beta$. Setting $lambda' = -1$ gives the threshold.

#check[A source with $lambda = -1.07$ and target $-0.85$ needs $beta = 1.85 \/ 2.07 approx 0.894$; the threshold is
$2 \/ 2.07 approx 0.966$. Your sweep bracketed the predicted threshold within $plus.minus 1%$, and the decay rate at
the $-0.85$ setting was 0.851. Point (1) is why global relaxation on ARC left all endpoints unchanged.]

= D9. Inheritance, and what downstream tokens should do

*Claim.* An oscillating upstream token forces period-2 motion downstream, and pinning upstream stops it.

*Spectrum (M6).* For a block lower-triangular matrix, $det(z I - cal(J)) = product_i det(z I - cal(J)_(i i))$. So the
eigenvalues of the full system are the union of each token's own-block eigenvalues. One token with an eigenvalue past
$-1$ is enough to destabilize the sequence, and the instability lives at that token.

*Forced response (an untested prediction).* Let an upstream token oscillate as $(-1)^t u$ and enter position $q$
through a coupling $B$. Then $delta_(t+1) = J delta_t + B (-1)^t u$. Guess a period-2 solution $delta_t = (-1)^t v$:
$ (-1)^(t+1) v = J (-1)^t v + (-1)^t B u quad => quad -v = J v + B u quad => quad v = -(I + J)^(-1) B u. $
So downstream positions amplify inherited blinking along modes with eigenvalues near $-1$: by $1 \/ (1 + lambda)$, up to
about $7.7 times$ for your validated modes at $-0.87$. This could explain why residual downstream motion persisted after
source damping. It hasn't been tested.

= D10. One formula for both contributions

Force with any sequence $P_t = z^t P$: $delta_(t+1) = J (delta_t + P_t)$. Guess $delta_t = z^t v$:
$ z v = J (v + P) quad => quad v = (z I - J)^(-1) J P. $
- $z = 1$ (constant forcing): $(I - J)^(-1) J P$. This is *steering* (D3), amplified by eigenvalues near $+1$.
- $z = -1$ (alternating forcing): $-(I + J)^(-1) J P$. This is *blinking inheritance* (D9), amplified by eigenvalues
  near $-1$.

This is the transfer function of the linearized loop. Steering and period-2 dynamics are its responses at the two ends
of the frequency range.

= Exercises

Do these with the derivations covered. Answers follow.

+ Show $sum_(t=1)^T J^t = J (I - J^T)(I - J)^(-1)$. Check it in the scalar case with $T = 1$.
+ For $lambda = 0.78$, compare the readout effect of a one-shot push after 10 loops with the steady-state effect of the
  same push applied persistently (scalar case, $g = 1$).
+ In the $2 times 2$ toy with $lambda_1 = 0.1$, $lambda_2 = 0.5$, $a = 2$: compute the one-step lever, its cosine with
  $g$, and the infinite-horizon lever $(I - J)^(-top) J^top g$.
+ Prove the fraction-equals-cosine corollary without looking.
+ A source token has $lambda = -1.226$. What $beta$ moves it to $-0.6$? What is the flip threshold?
+ Prove that $G(s) = (1 - beta) s + beta F(s)$ has the same fixed points as $F$.
+ Derive the forced period-2 response $v = -(I + J)^(-1) B u$.
+ (Harder) Show that for symmetric $J$ with eigenvalues in $(0, 1)$, the item lever has positive cosine with $g$.
+ (Harder) Derive the RMSNorm Jacobian in D6.
+ (Harder) Explain why, in the toy, a push along the lever can grow for one step while $rho(J) < 1$.

= Where an analytical result might come from

*A. The transfer-function framing (D10).* Promote D10 from a remark to the organizing result: the linearized loop's
transfer function $H(z) = g^top (z I - J)^(-1) J$ predicts steering at $z = 1$ and blinking at $z = -1$. The D9
prediction is cheap to test on existing census and relaxation runs, with the Jacobian tools you have.

*B. Decoding versus control as reachability versus observability.* (Partly done in `analytic.pdf`: for the readout
direction, the deciding quantity is circulation, not departure from normality.) A mass-mean probe is a difference of fixed points:
$ Delta x^* approx (I - J)^(-1) (partial F) / (partial E) Delta E quad "(where the context moves the state)". $
The infinite-horizon lever is
$ (I - J)^(-top) J^top g quad "(what the readout listens to)". $
For normal $J$ these can align; for non-normal $J$ they can separate arbitrarily. Conjecture: decode–control
misalignment grows with the departure from normality (for example $norm(J J^top - J^top J)$ or the eigenvector
condition number). This is the controllability/observability duality from control theory, and it carries over to
ordinary transformers by replacing $J^t$ with products of per-layer Jacobians. Caveat: the probe already had cosine
0.06 with $g$ before any looping, so part of the gap is readout geometry, not recurrence. A good theorem separates the
two.

*C. A provable stabilizer.* From D8: global relaxation with $beta < 2 \/ (1 - lambda_min)$ removes every real flip
instability and leaves all fixed points unchanged. It slows a mode with eigenvalue $lambda$ from rate $lambda$ to
$1 - beta (1 - lambda)$, which you can state exactly. Complex eigenvalues need $|1 - beta + beta lambda| < 1$.

*D. When are one-step levers enough?* $A_T^top g = J^top g + (J^top)^2 g + dots.c$. Bound
$cos(J^top g, A_T^top g)$ in terms of how fast the later terms shrink and rotate. That would explain the measured
$0.70$.

#pagebreak()

= Answers

+ Multiply $(J + dots.c + J^T)(I - J) = J - J^(T+1) = J (I - J^T)$ (the sum telescopes), then multiply on the right by
  $(I - J)^(-1)$. Scalar, $T = 1$: $lambda (1 - lambda) \/ (1 - lambda) = lambda$. ✓
+ One-shot after 10 loops: $0.78^(10) approx 0.083$. Persistent steady state: $0.78 \/ 0.22 approx 3.55$. About
  $43 times$ larger: this is the whole "one-shot erased, persistent kept" result in one line.
+ $J^top g = (0.1, 2)$, cosine with $e_1$ is $0.1 \/ sqrt(4.01) approx 0.05$. For infinite horizon, solve
  $(I - J)^top y = J^top g$: $0.9 y_1 = 0.1 => y_1 approx 0.111$; $-2 y_1 + 0.5 y_2 = 2 => y_2 approx 4.44$. Cosine
  with $g$: $0.111 \/ 4.45 approx 0.025$. Longer horizons make it *more* orthogonal here.
+ $Delta R(u) = a^top u = norm(a) cos(u, a)$ and the best unit push gives $norm(a)$. Divide.
+ $beta = 1.6 \/ 2.226 approx 0.719$. Threshold $2 \/ 2.226 approx 0.898$.
+ See D8 (1).
+ See D9.
+ $g^top A_T^top g = sum_i phi_T (lambda_i) (v_i^top g)^2 > 0$ since each $phi_T (lambda_i) > 0$ and $g != 0$.
+ $partial (x_i \/ "rms") \/ partial x_j = delta_(i j) \/ "rms" - x_i x_j \/ ("rms" dot norm(x)^2)$, using
  $partial "rms" \/ partial x_j = x_j \/ (d dot "rms") = "rms" dot x_j \/ norm(x)^2$. Factor out $1 \/ "rms"$ and
  multiply by $op("diag")(gamma)$.
+ $J e_2 = (a, lambda_2)$ has norm $sqrt(a^2 + lambda_2^2) > 1$ when $a > 1$. Eigenvalues bound growth only
  asymptotically: $norm(J^t) <= kappa(V) rho^t$, and the eigenvector condition number $kappa(V)$ is large when $J$ is
  far from normal.
