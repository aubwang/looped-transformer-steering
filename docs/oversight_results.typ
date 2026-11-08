#set document(title: "Oversight pilot: how looped models respond to small nudges")
#set page(paper: "us-letter", margin: (x: 1in, y: 0.9in), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#show table: set par(justify: false)
#set heading(numbering: "1.")
#show heading.where(level: 1): it => [#v(0.6em) #it #v(0.2em)]

#let boxed(title, color, body) = block(
  fill: color, inset: 9pt, radius: 3pt, width: 100%, breakable: true,
  [*#title.* #body],
)
#let plain(body) = boxed("In words", rgb("#fbf3e6"), body)
#let caveat(body) = boxed("Careful", rgb("#fdeeee"), body)

#align(center)[
  #text(16pt, weight: "bold")[Oversight pilot: how looped models respond to small nudges]
  #v(0.2em)
  #text(10.5pt)[Part 1 (easy prompts, `docs/oversight_pilot.md`) and part 2 (harder prompts, `docs/oversight_hard.md`)]
  #v(0.2em)
  #text(9.5pt)[Draft 2, 2026-11-08. Both parts' results reviewed by GPT-6 Astra (xhigh), which recomputed every
  comparison from the arrays. Theory: `docs/observability.pdf`.]
]

#v(0.4em)

#plain[
We took four looped models built from the same 1-billion-parameter parent and trained with random loop counts
of mean $M + 1$ for $M = 4, 8, 16, 32$ (called L4, L8, L16, L32), plus their ordinary non-looped partner D (same data,
same recipe). We asked two questions, first on 16 easy addition problems (part 1), then on 16 problems chosen in
advance because at least one model was unsure or wrong (part 2).

- *Nudge the loop's internal state: how much does the answer distribution move?* The checkpoints differ, but not in
  order of training depth. One difference appeared in both parts and both ways of measuring nudge size: L4's output
  reacts less than L32's (about 12 times less on easy prompts, about 6 times less on hard ones).
- *Push on an internal layer: how well does a straight-line (linear) prediction describe the effect?* For pushes
  matched by their effect on the output, the looped models showed no consistent disadvantage against the non-looped
  partner on easy prompts, and on hard prompts their median errors were lower for random pushes (for example 0.42–0.47
  against 0.76). The straight-line prediction is still poor for random pushes of moderate effect, and it fails badly
  on some individual prompts even for answer-directed pushes.

These are the same checkpoints under one protocol, on two prompt sets: a replication across prompts, not across
model families. No monitor was tested, so nothing here says that loops make monitoring or useful steering easier.
]

= The objects

*The model as a dynamical system.* A prompt is a list of $L$ tokens ($L = 39$ in part 1; 39 or 54 in part 2). The first layers turn it into a
matrix $e$ with one row of $d = 2048$ numbers per token. The looped part is a map $F$ from states to states, where a
state $x$ is also an $L times d$ matrix:
$ x_(t+1) = F(x_t), quad t = 0, 1, 2, dots $
The start $x_0$ is a small random matrix, as in training. After $t$ loops the last layers read the row of $x_t$ at the
final position $q$ and produce one score $z_i$ for every one of the $V = 128 thin 256$ tokens. The model's output
distribution is $pi_i = e^(z_i) \/ sum_j e^(z_j)$ (positive numbers that sum to 1).

*Measuring a change of output.* If the scores change by a vector $Delta$, we measure the change by its *Fisher norm*
$ norm(Delta)_H^2 = sum_i pi_i Delta_i^2 - (sum_i pi_i Delta_i)^2, $
the variance of $Delta$ when a token is drawn from $pi$. Adding the same number to every score changes nothing (the
distribution is unchanged), and the Fisher norm ignores it. For small changes, the Kullback–Leibler divergence between
the old and new distributions is $"KL" approx norm(Delta)_H^2 \/ 2$.

*Part A: nudging the state.* Run 64 loops; by then every model sits very close to a fixed point $x^* = F(x^*)$
(Section 2). Add a small random matrix $delta$ whose row for each token has length $sigma$ times that row of the
state ($sigma = 10^(-3)$). Run both copies $k$ more loops and record
$ F_k = norm(z^"nudged"_(64+k) - z_(64+k))_H^2 \/ sigma^2, quad k = 0, dots, 32. $
The *susceptibility* $A$ averages $F_k$ over $k = 1, dots, 32$ and over 4 random directions. Because
$norm(delta)_F^2 = sigma^2 norm(x_64)_F^2$ (Frobenius norm: square root of the sum of all squared entries), the
*absolute* version $A \/ norm(x_64)_F^2$ is the response per unit of nudge energy. A comparison of two models is the
per-prompt $log(A_1 \/ A_2)$; we report its median over prompts and a 95% bootstrap interval (resample the 16 prompts
with replacement 10,000 times). A log ratio of $-2.5$ means a factor $e^(-2.5) approx 1\/12$.

*Part C: pushing a layer.* Add a vector $epsilon u$ to the output of layer 4 at position $q$. In a looped model that
layer feeds every loop, so the push acts on every loop; in D it acts once. Here $epsilon$ is a dose and $u$ a fixed
direction: 4 random directions, and the 2 directions that most increase or decrease the probability of the answer.
Let $Delta(epsilon)$ be the change of scores. The *linear prediction* scales the response to a tiny dose,
$epsilon_0 = 10^(-3)$: $Delta_"lin" (epsilon) = (epsilon \/ epsilon_0) Delta(epsilon_0)$. Its relative error is
$ e(epsilon) = norm(Delta(epsilon) - Delta_"lin" (epsilon))_H \/ norm(Delta(epsilon))_H. $
To compare fairly, each model's error is read off at the dose that produces a given KL (from $10^(-6)$ to $10^(-1)$),
using only directions that reach that KL in both models.

= Runs and checks

- Five Modal runs, each under 11 GPU-minutes, about 27 GPU-minutes in all. Every computation was in 64-bit floating
  point. Two smoke runs on 2 prompts came first; the first, in 32-bit arithmetic, showed that rounding noise swamped
  these small responses and led to the switch.
- Every quality gate passed in every container. The loop reimplementation matches the model's own code to
  $10^(-15)$; finite-difference slopes agree across doses to about $10^(-3)$; noise floors are about $10^(-28)$.
- Run 2 is labelled *not clean*: its L16 container lost the last prompt to a slow first download of the weights. L16
  was rerun alone (clean). On the 15 prompts both L16 runs share, the measured numbers agree to a median relative
  difference of $10^(-10)$. That is numerical repeatability, not an independent replication. The combined data use the
  clean containers (L4, L32, D from run 1; L8 from run 2; L16 from the rerun).
- Base state: by loop 64 the median step size $norm(x_(t+1) - x_t) \/ norm(x_t)$ is about $10^(-13)$ to
  $10^(-12)$ for L4 and L8, $10^(-7)$ for L32 and $10^(-5)$ for L16. Different random starts end at the same output
  (median KL at most $2 times 10^(-11)$ by loop 32).

= Part 1 results: easy prompts

== Output sensitivity to state nudges (Part A, loop 64)

#figure(table(
  columns: 4, align: (left, center, center, center), stroke: 0.4pt,
  [*Comparison*], [*Relative nudge*], [*Absolute (per energy)*], [*Prompts (negative)*],
  [L4 / L32 (primary)], [$-2.50$ $[-3.65, -1.99]$], [$-1.20$ $[-2.35, -0.69]$], [15 (13)],
  [L4 / L8], [$-1.02$ $[-1.59, +0.56]$], [$-1.45$ $[-2.01, +0.14]$], [16 (9)],
  [L4 / L16], [$-0.66$ $[-0.83, +0.57]$], [$-0.84$ $[-1.01, +0.40]$], [15 (11)],
  [L8 / L16], [$+0.13$ $[-0.68, +0.81]$], [$+0.38$ $[-0.43, +1.06]$], [15 (5)],
  [L8 / L32], [$-1.42$ $[-2.76, -0.80]$], [$+0.30$ $[-1.03, +0.93]$], [15 (13)],
  [L16 / L32], [$-1.67$ $[-2.56, -0.74]$], [$-0.19$ $[-1.09, +0.74]$], [14 (12)],
), caption: [Median per-prompt $log(A_1 \/ A_2)$ with 95% bootstrap intervals. Negative: the first model reacts less.])

- The primary comparison, fixed in advance, is L4 versus L32: L4 reacts about 12 times less to the same relative
  nudge, and about 3 times less per unit of nudge energy. 13 of 15 prompts agree. It holds without the two smoke
  prompts ($-2.50$, 13 prompts).
- It is the only pairwise difference supported in *both* normalizations. L4's estimates against L8 and L16 point the
  same way but are uncertain. L32's larger relative response against L8 and L16 disappears per unit energy: L32's
  state matrix is about twice as large in Frobenius norm (median 2665 against 1124–1389).
- The median L4/L32 curve of $log(F_k^"L4" \/ F_k^"L32")$ stays negative for every $k$, from $-2.1$ at $k = 1$ to about
  $-12$ at $k = 32$ (not monotonically). Two of 15 individual prompts cross zero.

== How fast nudges fade (exploratory, not fixed in advance)

Between $k = 16$ and $k = 32$, the median per-loop factor by which the state nudge shrinks,
$(norm(Delta x_32) \/ norm(Delta x_16))^(1\/16)$, is

#align(center)[0.67 (L4), 0.70 (L8), 0.96 (L16), 0.85 (L32).]

These are *effective decay factors over a finite window for 4 sampled directions*, not eigenvalues: they neither
identify the slowest mode nor bound how strongly the loop can amplify some other direction. In one L8 prompt the nudge
even grew slightly over this window (factor 1.013). The defensible statement is that L16 has the slowest median decay
in this window. Training depth does not order these factors.

== Steering prediction (Part C)

#figure(table(
  columns: 5, align: (left, center, center, center, center), stroke: 0.4pt,
  [*Model (loops)*], [*KL $10^(-5)$*], [*KL $10^(-3)$*], [*KL $10^(-2)$*], [*KL $10^(-1)$*],
  [L4 (4)], [0.15 / 0.14], [0.65 / 0.72], [0.72 / 0.84], [0.88 / 0.92],
  [L8 (8)], [0.15 / 0.14], [0.61 / 0.72], [0.71 / 0.83], [0.80 / 0.97],
  [L16 (16)], [0.19 / 0.14], [0.67 / 0.72], [0.74 / 0.83], [0.91 / 0.94],
  [L32 (32)], [0.09 / 0.14], [0.51 / 0.72], [0.69 / 0.83], [0.82 / 0.95],
), caption: [Median relative error $e$ of the linear prediction for random pushes, looped / D, at matched KL and on
the same directions.])

- Random pushes: the looped models' errors are similar to D's or a little smaller (no consistent observed
  disadvantage; this is not a formal non-inferiority result). The full range of median log
  ratios (looped / D) is $-0.69$ to $+0.49$; intervals exclude zero only in some cases (for example L32 at KL
  $10^(-4)$: $-0.34$ $[-0.80, -0.03]$; L4 at KL $10^(-1)$: $-0.14$ $[-0.26, -0.07]$). These are pointwise intervals
  across many related comparisons, not corrected discoveries.
- The absolute errors are large. A random push needs a big dose to move these confident outputs at all, and at such
  doses the response is far from linear.
- Answer-directed pushes are much more linear (median error at KL $10^(-2)$: 0.04–0.23 for the looped models, 0.14
  for D). Only L32 differs clearly from D there ($-1.40$ $[-2.31, -0.17]$).
- Cost: counting a push that acts on every loop as $r$ pushes, the looped models need $e^3$ to $e^6$ times the
  squared push size of D. Per single push the excess is about $e^(1.2)$ to $e^(1.7)$ for random pushes at KL
  $>= 10^(-3)$. L4 needs the same per-push dose with 4 or with 32 loops: the response saturates within a few loops.

= Part 2: harder prompts

*Selection (fixed before screening).* A pool of 256 candidates in four families (three-digit subtraction with a
borrow, three-number sums, two-digit times one-digit, two-digit times two-digit; every answer below 1000, four solved
examples of the same family before the question; no example repeats its question). Each model answered every candidate
once, without any nudge or push. W = at least one model wrong; U = all right but at least one model gave the answer
probability below 0.9. Rule: 8 from W and 8 from U, round-robin over the families available in each group, in a fixed
random order. The cutoff 0.9 is an operational choice, not a definition of uncertainty.

*What the screen found.* Subtraction and two-digit times one-digit: all three models 100% right. Two-digit times
two-digit: one error (L32). Three-number sums: 66–70% right. So 34 candidates were W, 52 were U and 170 were easy. The
selected 16 are 9 three-number sums, 3 two-digit products, 2 subtractions and 2 one-digit products; W is 7 sums and
1 product. Family, prompt length (39 or 54 tokens) and group are therefore entangled.

#figure(table(
  columns: 6, align: (center, center, left, left, center, left), stroke: 0.4pt,
  [*uid*], [*group*], [*family*], [*question*], [*answer*], [*models that put it there*],
  [0], [W], [sum], [33 + 75 + 86], [194], [L32, D wrong],
  [1], [U], [sub], [880 − 786], [94], [L4, D unsure],
  [2], [W], [product], [13 × 71], [923], [L32 wrong],
  [3], [U], [sum], [82 + 78 + 77], [237], [L32, D unsure],
  [4], [W], [sum], [26 + 89 + 24], [139], [L4 wrong],
  [5], [U], [×1 digit], [64 × 5], [320], [L4, L32 unsure],
  [6], [W], [sum], [19 + 34 + 72], [125], [L4 wrong],
  [7], [U], [product], [43 × 20], [860], [L32 unsure],
  [8], [W], [sum], [51 + 89 + 19], [159], [all three wrong],
  [9], [U], [sub], [922 − 198], [724], [L4 unsure],
  [10], [W], [sum], [35 + 74 + 20], [129], [L4, D wrong],
  [11], [U], [sum], [60 + 84 + 24], [168], [D unsure],
  [12], [W], [sum], [49 + 25 + 56], [130], [L32, D wrong],
  [13], [U], [×1 digit], [34 × 4], [136], [L4 unsure],
  [14], [W], [sum], [41 + 69 + 12], [122], [L4 wrong],
  [15], [U], [product], [12 × 45], [540], [L32 unsure],
), caption: [The 16 selected prompts. "Unsure": answer probability below 0.9.])

*Run.* Same protocol and frozen analysis as part 1 (one run, about 9 GPU-minutes; all gates passed; the selection was
re-derived from the saved screen before launch). Accuracy: L4 11/16, L32 12/16, D 12/16; every U prompt stayed right,
and W accuracy was 3/8 (L4), 4/8 (L32) and 4/8 (D). The run's greedy answers equal the screen's for every model and depth (different
random starts); answer probabilities agree to within $10^(-3)$ at L4's loop 4 and within $3 times 10^(-6)$ at loop 64.
Agreement of two starts does not prove start independence.

*State nudges (Part A, loop 64).* All 16 prompts are in the low-residual subset for both models; two (uid 1 and 10)
are excluded because a few of L4's last responses fall below the noise floor.

#figure(table(
  columns: 4, align: (left, center, center, center), stroke: 0.4pt,
  [*L4 / L32*], [*Relative nudge*], [*Absolute (per energy)*], [*Prompts (negative)*],
  [hard set (primary)], [$-1.80$ $[-2.73, -1.35]$], [$-0.49$ $[-1.43, -0.04]$], [14 (13)],
  [easy set (part 1)], [$-2.50$ $[-3.65, -1.99]$], [$-1.20$ $[-2.35, -0.69]$], [15 (13)],
), caption: [Median per-prompt $log(A_"L4" \/ A_"L32")$ at loop 64 with 95% bootstrap intervals.])

- The direction recurs, with a smaller point estimate; that is not a demonstrated difference between the two sets.
  Per unit of nudge energy the interval only just excludes zero. The W and U groups (7 prompts each) are descriptive
  only: $-1.76$ and $-2.77$.
- Both models react far more on these prompts (median $A$: L4 0.016 against $2.3 times 10^(-4)$ on easy prompts,
  L32 0.11 against $2.1 times 10^(-3)$). This is consistent with less concentrated answer distributions (median entropy
  about 0.87–0.91 nats against 0.02–0.06), but the Fisher norm depends on how the score change lines up with the
  probability mass, and the prompts, states and readouts also changed, so it is not an explanation.

*Layer pushes (Part C).*

#figure(table(
  columns: 5, align: (left, center, center, center, center), stroke: 0.4pt,
  [*Random pushes*], [*L4, 4 loops*], [*L4, 32 loops*], [*L32, 32 loops*], [*D*],
  [KL $10^(-3)$], [0.17], [0.17], [0.14], [0.29–0.30],
  [KL $10^(-2)$], [0.46], [0.47], [0.42], [0.76],
  [KL $10^(-1)$], [0.78], [0.76], [0.71], [0.84],
), caption: [Median relative prediction error $e$ (a ratio of Fisher norms, not an accuracy) at matched KL, on
directions matched in both models.])

- Median paired log ratios (looped / D) are negative at every random-direction target; intervals exclude zero at KL
  $10^(-2)$ for L4 (both budgets; for example $-0.24$ $[-0.65, -0.03]$) and for L32 at $10^(-3)$ and $10^(-2)$ ($-0.60$
  $[-1.12, -0.22]$; the $10^(-3)$ interval only just excludes zero). These are pointwise exploratory intervals; L4's two
  budgets are not independent replications. "Somewhat more linear under the tested random pushes" is the defensible
  description, not "loops improve linearity".
- Answer-directed pushes: median errors at most 0.06 for every model, but single prompts fail badly at KL $10^(-1)$
  (worst prompt: 1.70, 1.40, 1.49 for the looped settings, 2.49 for D). Below KL $10^(-2)$ there are too few matched
  prompts (fewer than 12) to compare.
- Push size at matched KL $10^(-2)$ is similar for all models: about 0.50–0.54 of the site's typical scale for the
  looped models and 0.47 for D. Counted over loops, the cumulative squared push norm is $e^(2.6)$ to $e^(4.6)$ times D's;
  per single injection, $e^(0.8)$ to $e^(1.2)$. L4 with 4 or 32 loops gives nearly the same errors and push sizes, so
  more loops did not make the response more linear here. Forward cost: about 188 GFLOPs per prompt for L4 with 4
  loops, 1,180 with 32 loops, 93 for D.

= When is a looped response more linear? A conditional statement

Matching KL equalizes one number. Write the score change for a push of dose $epsilon$ as
$Delta(epsilon) = epsilon g + epsilon^2 h \/ 2 + O(epsilon^3)$, with $g, h in RR^V$. The linear prediction from a tiny
dose is $epsilon g$, so its relative error is $e approx epsilon norm(h)_H \/ (2 norm(g)_H)$. Since
$"KL" approx epsilon^2 norm(g)_H^2 \/ 2$, at a fixed KL
$ e approx sqrt(2 "KL") / 2 dot norm(h)_H / norm(g)_H^2 . $
So a lower matched error means *less curvature relative to the square of the gain*. Contraction alone does not
provide that.

*Proposition (conditional).* Let $x_(t+1) = F(x_t, epsilon)$ be twice continuously differentiable. Suppose there is a
closed convex region $C$ and an interval $I$ around $epsilon = 0$ such that, for every $epsilon in I$, $F(dot, epsilon)$
maps $C$ into itself, and on $C times I$: $norm(F_x) <= q < 1$, $norm(F_epsilon) <= B$, and the second derivative of
$F$ in $(x, epsilon)$ is bounded by $L$ (as a bilinear form, with the norm $norm(v) + |t|$ on pairs $(v, t)$). Then for
each $epsilon in I$ the fixed point $x_*(epsilon)$ in $C$ is unique, and
$ norm(x'_*) <= B / (1 - q), quad norm(x''_*) <= L / (1 - q) (1 + B / (1 - q))^2 . $

_Proof._ On the convex set $C$ the bound $norm(F_x) <= q$ makes $F(dot, epsilon)$ a contraction (mean-value
inequality), so Banach's fixed-point theorem gives a unique fixed point. Since $norm(F_x) <= q < 1$, the matrix $I - F_x$ is
invertible with $norm((I - F_x)^(-1)) <= sum_k q^k = 1 \/ (1 - q)$, so the implicit function theorem makes $x_*$ smooth
in $epsilon$. Differentiating $x_* = F(x_*, epsilon)$ once gives $(I - F_x) x'_* = F_epsilon$, hence the first bound.
Differentiating again gives $(I - F_x) x''_* = D^2 F[(x'_*, 1), (x'_*, 1)]$, whose right side has norm at most
$L (norm(x'_*) + 1)^2$; insert the first bound. $square$

A smooth readout then gives upper bounds on $g$ and $h$ for the output. The matched-KL error, however, needs an upper
bound on $h$ *and a lower bound* on the gain $g$, and contraction gives no lower bound: the push may hardly move the
output at all. So the proposition alone cannot show that a looped response is more linear.

*Counterexample to any general ordering.* $F(x, epsilon) = q x + (1 - q)(a epsilon + b epsilon^2)$ contracts with
factor $q$ for every $epsilon$, and its fixed point $x_* = a epsilon + b epsilon^2$ has arbitrary curvature $b$. A
linear feedforward map $epsilon |-> a epsilon$ has zero error. So nothing general can be said about loops versus no
loops from contraction; whether the measured advantage comes from curvature, readout geometry, the direction scales
or the adapter re-injection is not identified by these data. None of the proposition's assumptions were measured.

= What this does and does not say

#caveat[
- *No theorem was tested.* The pilot was registered as an exploratory association study after Astra showed that the
  note's Theorem 1 predicts no ordering between models.
- *Contraction is not established.* Proposition 3 of the theory note (linear predictability) needs a bound
  $norm(D F) <= q < 1$ on the whole region the trajectory visits. Fading sampled nudges do not give that, and the
  layer-4 push changes the loop map itself rather than adding a vector to the state, which is a different setting.
- *Theorem 1 can only be illustrated, conditionally.* It bounds how visible a mode with $|lambda| < 1$ can be,
  through $m_p (|lambda|^2) = sum_r p(r) |lambda|^(2r)$. If L32 had a mode with $|lambda| = 0.854$, its training depth
  distribution would give $m_p approx 0.005$, which allows such a mode to be quite visible. We measured neither
  eigenvalues nor the bound's right-hand side, so this explains nothing.
- *Weaker output sensitivity is not the same as easier oversight.* It can mean robustness, or that internal changes
  are harder to see from outside. No monitor was evaluated, and matched KL measures a change of distribution, not a
  successful steer toward a goal.
- *Scope.* Released checkpoints differ in more than training depth (curriculum, gradient horizon, operating point).
  Two prompt sets of 16 (one easy and confident, one screened for uncertainty and errors), the answer position only,
  the same checkpoints and protocol. Part 2's conclusions concern its jointly screened set; selection can favour or
  disfavour any model, and its W group is nearly all three-number sums.
- *Energy* is the declared cumulative squared push norm, not physical energy, computational cost or practical
  difficulty of steering. Answer-directed pushes use the known correct answer.
]

= Takeaway and next steps

Across an easy prompt set and a separately screened set of uncertain or wrong answers, L4's answer distribution is
less sensitive than L32's under the frozen state-nudge protocol, in both normalizations. The looped models show no
consistent disadvantage in local intervention prediction against their non-looped partner; the hard set adds
exploratory evidence of lower median errors for some matched-effect random pushes. Neither part shows that loops make
monitoring or useful steering easier: no monitor was evaluated, weaker sensitivity can hide internal changes as well
as give robustness, and changing a distribution is not the same as making a desired correction.

Astra's ranking of next steps:
+ stop GPU experiments here and write up (this note, with the descriptive tables in the result folders);
+ develop the conditional theory above without GPU work: persistent parameter forcing, finite horizons, readout
  bounds, and the curvature-to-gain condition, with the counterexample;
+ if one more capped pilot is warranted: non-answer positions, with perturbation sites and readout sites kept apart;
+ when larger evaluations resume: a second matched model family, then a held-out monitor experiment (the 16 prompts
  here have only four or five errors per model and are confounded by family);
+ eigenvalue (Arnoldi) estimates only for a specific spectral hypothesis.

#v(0.5em)
#text(9pt)[Data: `results/obs_run1-20261107-212210`, `results/obs_run2-20261107-212755` (not clean),
`results/obs_run2_L16-20261107-213455`, combined view `results/obs_combined-20261107` (report.txt, summary.json).
Part 2: screen `results/obs_screen-20261108-051120`, run `results/obs_hard-20261108-051314` (report.txt, summary.json,
descriptive.json). Code: `huginn_addition/obs_app.py`, `analyze_obs.py`, `report_hard.py`, `obs_local_test.py`.]
