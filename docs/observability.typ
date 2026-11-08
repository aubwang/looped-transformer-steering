#set document(title: "What training can see: loss-observability of looped models")
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
#let thm(title, body) = boxed(title, rgb("#eef3fb"), body)
#let plain(body) = boxed("In words", rgb("#fbf3e6"), body)
#let caveat(body) = boxed("Careful", rgb("#fdeeee"), body)
#let tr = math.op("tr")
#let ran = math.op("range")
#let Hh = $H^(1\/2)$

#align(center)[
  #text(16pt, weight: "bold")[What training can see: loss-observability of looped models]
  #v(0.2em)
  #text(10.5pt)[Random-depth training controls only the dynamics its loss can observe; what that means for
  monitoring and steering]
  #v(0.2em)
  #text(9.5pt)[Draft 2, 2026-11-07: merges GPT-6 Astra's independent proposals (Theorem 1′, Sections 4–5) and its math review (xhigh). Companion to `docs/analytic.pdf` and `docs/freeloop.pdf`.]
]

#v(0.4em)

#plain[
A looped model applies one map again and again. Some models (Huginn and the retrofitted Llama/OLMo models) are
trained from a *random* start state for a *random* number of loops. This note asks what that training forces on the
loop's dynamics, and what it leaves free.

- *The loss sees the dynamics only through the output distribution.* Near a fixed point, its view is a
  "Fisher-observability" matrix $W_p$. This matrix weights the readout's sensitivity after $r$ loops by the
  probability $p(r)$ of training at depth $r$ (Section 1).
- *Theorem 1 (observable stability).* If the output barely depends on small start perturbations, a dynamical mode
  that grows or decays like $lambda^r$ can move the output distribution only by a factor bounded through the
  generating function $sum_r p(r) |lambda|^(2r)$. If the depth distribution has a heavy tail (Huginn's does), every
  *expanding* mode must leave the output distribution unchanged to first order. Training that stops at a fixed depth
  gives no such guarantee beyond that depth. The theorem bounds how *visible* a slow mode can be, not how *long* it
  lasts: a model can satisfy it by making a slow mode faint (Section 7).
- *Theorem 2 (what training cannot see).* The loss constrains the loop only through the smallest subspace that
  contains the read directions and is closed under the transposed loop $J^top$. Off that subspace the loop can be
  changed arbitrarily without changing the loss. A *monitor* (a linear read of the state) inherits a bound on its
  perturbation-induced variance only through an *observability cost*. For a monitor with a component the loss cannot
  see, training gives no such bound.
- *Theorem 1′ (a converse; GPT-6 Astra).* If the loss observes every direction well within a window of training
  depths, small start sensitivity forces the whole loop to contract eventually. Together with Theorem 1: where the
  loss sees well enough, small sensitivity forces contraction; where it sees weakly, heavy-tailed depths exclude only
  *expanding* visible modes (not modes with $|lambda| = 1$); where it is blind, nothing is forced.
- *Predictability and control (Section 4).* Under contraction, the frozen-Jacobian prediction of extra loops and of
  persistent pushes has an explicit error bound. That bound is standard and holds for feedforward stacks as well.
  Counterexamples show that "more stable" implies neither "more predictable" nor "more controllable".
- *Monitoring (Section 5).* An invertible affine map of the state never changes what a linear monitor can read. With
  observation noise, contraction erodes it; re-injecting the input can restore it.
- These results are consistent with three existing observations (Section 6). Cheap predictions on matched
  looped/feedforward pairs are in Section 7.

*Honest novelty (Astra's assessment, which I share).* The mathematics is checked but mostly textbook, and
"dynamical systems for loops" alone is not new: Jacobian steering and loop monitoring already exist.
- The generating-function inequality is the Gramian evaluated on an eigenvector.
- The minimum-norm observability cost is classical.
- The broad idea that supervision controls the visible coordinates overlaps substantially with Sharma and Vu
  (arXiv 2606.24898).

What may be useful is the *synthesis* and its application to one question: what happens when the inference depth of
a single recurrent model changes. That claim is pending a systematic literature review. A strong paper would also need
a *validated mechanism* running from training to dynamics to oversight, across matched families.
]

= Setting

*The loop.* The state is $s in RR^d$. The input is fixed, and one loop step is $s |-> F(s)$, a $C^2$ map. A *fixed
point* is a state with $F(s^*) = s^*$. The *loop Jacobian* $J = D F(s^*)$ is the $d times d$ matrix of first
derivatives there: a small deviation $xi$ from $s^*$ becomes $J xi$ after one step and $J^r xi$ after $r$ steps,
to first order.

*The readout.* The coda maps the state to logits $z = h(s) in RR^V$, one per vocabulary token. The model's
output distribution is $pi = "softmax"(z)$, with $pi_i = e^(z_i) \/ sum_j e^(z_j)$. Let $G = D h(s^*)$
($V times d$) and $pi = "softmax"(h(s^*))$, and define the $V times V$ matrix
$ H = "diag"(pi) - pi pi^top. $

*Why $H$.* For a target token $y$ the loss is $ell(z) = -log pi_y$. Its gradient is $pi - e_y$ and its Hessian
is $H$, whatever $y$ is. If $y$ is distributed as $pi$, the gradient averages to $0$, so
$ EE_y [ell(z + Delta)] - EE_y [ell(z)] = 1/2 Delta^top H Delta + O(norm(Delta)^3). $
$H$ is symmetric and positive semidefinite. When every $pi_i > 0$, its null space is spanned by the all-ones vector
$bb(1)$. So $Hh G v = 0$ means that $v$ shifts all logits equally, i.e. it leaves the output distribution unchanged to
first order. We call such a $v$ *output-invisible*.

*Training depths.* The number of loops is drawn from a distribution $p$ on ${1, 2, dots}$. Its generating function
is $m_p (x) = sum_r p(r) x^r$. For Huginn and the retrofitted models, $r = 1 + "Poisson"(Lambda)$ with $Lambda$
lognormal of mean $M$, the mean Poisson rate. The mean loop count is $M + 1$, and $M = 32$ for Huginn. Then
$m_p (x) = x EE[e^(Lambda (x - 1))]$.

*Local start sensitivity.* Perturb the fixed point by isotropic noise, $xi tilde cal(N)(0, sigma^2 I)$ with
$sigma > 0$, run $r$ more loops of the *linearized* system, and measure the Fisher-weighted output change:
$ S_p (sigma) = EE_(r tilde p) EE_xi norm(Hh G J^r xi)^2 = sigma^2 tr W_p,
  quad W_p = sum_r p(r) (J^top)^r G^top H G J^r. $
$W_p$ is a depth-weighted *observability Gramian* (Kalman); we assume it is finite. Let
$Omega_p = norm(W_p)_"op"$ be its worst direction; then $Omega_p <= tr W_p = S_p \/ sigma^2$. $S_p$ is exactly the
sensitivity of the linearized system. If targets are distributed as $pi$, $S_p \/ 2$ is the leading term of the
excess loss of the *nonlinear* model only when the small-noise limit can be exchanged with the depth expectation. That
holds for depths capped at a finite horizon, or under a suitable domination assumption; pointwise $C^2$ smoothness
alone is not enough.

#caveat[
"Training makes $S_p$ small" is a *hypothesis*, not a theorem. Random-start, random-depth training penalizes any
dependence of the output on the start and on the depth; near the fixed point, that dependence is $S_p$. But Huginn's
random starts are far from $s^*$ (the nonlinear transient), so the local noise model is an idealization. $S_p$ is
probed in prediction P1, but finite-size random-direction measurements estimate (weighted) traces, not the
worst-direction $Omega_p$. The theorems below are pure linear algebra *given* $S_p$ or $Omega_p$.
]

= Theorem 1: observable stability

#thm("Theorem 1")[
For every unit vector $v in CC^d$,
$ sum_r p(r) norm(Hh G J^r v)^2 <= Omega_p. $
In particular, for every eigenpair $J v = lambda v$ with $norm(v) = 1$:
$ m_p (|lambda|^2) dot norm(Hh G v)^2 <= Omega_p. $
]

_Proof._ $v^* W_p v = sum_r p(r) norm(Hh G J^r v)^2$, and $v^* W_p v <= norm(W_p)_"op" norm(v)^2$ because $W_p$ is
Hermitian and positive semidefinite. For an eigenvector, $J^r v = lambda^r v$. $square$

The first form covers non-normal transients, where $norm(J^r v)$ can grow before it decays. The second is clearest
for single modes. A mode that keeps a factor $|lambda|$ per loop can move the output distribution by at most
$norm(Hh G v) <= sqrt(Omega_p \/ m_p (|lambda|^2))$. Slow modes ($|lambda|$ near 1) and expanding modes
($|lambda| > 1$) have large $m_p$, so they are pushed toward output-invisibility. This bounds *visibility*, not
*persistence*: a slow mode is allowed as long as it is faint enough.

#thm("Corollary 1a (heavy-tailed depths)")[
Suppose $m_p (x) = oo$ for every $x > 1$ and $Omega_p < oo$. Then every eigenvector of $J$ with $|lambda| > 1$ is
output-invisible: $G v in "span"(bb(1))$. Equivalently, every expanding mode lies outside what the loss observes.
The Poisson–lognormal depth distribution satisfies the hypothesis, because
$m_p (x) = x EE[e^(Lambda (x - 1))]$ and a lognormal $Lambda$ has no finite exponential moments. If the sampled
depths are capped at $r_max$ with $p(r_max) > 0$, the bound is quantitative:
$ norm(Hh G v)^2 <= Omega_p \/ (p(r_max) |lambda|^(2 r_max)). $
]

#thm("Corollary 1b (no guarantee beyond a bounded training depth)")[
If $p$ is supported in ${1, dots, R}$, then $m_p (x) <= x^R$ for $x >= 1$. A model can have $Omega_p$ arbitrarily
small and still an observable expanding mode whose effect on the output grows without bound beyond depth $R$.

_Example._ Take $d = 1$, $J = lambda > 1$, $p = delta_R$, and a scalar readout with $norm(Hh G) = a$. Then
$Omega_p = a^2 lambda^(2 R)$. Choose $a = eta lambda^(-R)$: then $Omega_p = eta^2$, while the output effect at
depth $k$ is $a lambda^k = eta lambda^(k - R) -> oo$.
]

#thm("Theorem 1′ (full observability forces eventual contraction; proposed by GPT-6 Astra)")[
Let $m >= 0$ and $kappa >= 1$ be integers and $sigma > 0$. Suppose $p(m + k) >= p_min > 0$ for $0 <= k < kappa$,
and the finite-horizon Fisher-observability Gramian satisfies
$W_kappa = sum_(k < kappa) (J^top)^k G^top H G J^k succ.eq alpha I$ with $alpha > 0$. Then
$ norm(J^m)_2^2 <= norm(J^m)_F^2 <= S_p \/ (sigma^2 p_min alpha). $
If the right-hand side is below 1, then $rho(J) < 1$: the loop is eventually contracting.
]

_Proof._ $S_p = sigma^2 sum_r p(r) tr((J^r)^top G^top H G J^r) >= sigma^2 p_min tr((J^m)^top W_kappa J^m)
>= sigma^2 p_min alpha norm(J^m)_F^2$. Also $rho(J)^(2m) <= norm(J^m)_2^2$. $square$

*The two theorems together.* Take $J = "diag"(1\/2, 2)$, with a readout that sees only the first coordinate.
The output's dependence on the start decays like $2^(-r)$, yet the hidden second coordinate doubles at every loop.
Theorem 1′ does not apply ($alpha = 0$), and Theorem 1 with heavy-tailed depths says the expanding mode must be
invisible, which it is. So random-depth training alone does not force stability. *Observability* decides where
stability is forced.

= Theorem 2: what training cannot see

Stack the Fisher-weighted read directions and their pullbacks into the observability matrix:
$ O_kappa = mat(Hh G; Hh G J; dots.v; Hh G J^(kappa - 1)), quad
  W_kappa = O_kappa^top O_kappa = sum_(k < kappa) (J^top)^k G^top H G J^k. $
The *observable subspace* $K = ran W_d$ is the smallest $J^top$-invariant subspace containing the columns of
$G^top Hh$ (a Krylov subspace; by Cayley–Hamilton, $kappa = d$ suffices). For a monitor direction $m in RR^d$, which
reads $q = m^top s$, define its *observability cost*
$ c_kappa (m) = m^top W_kappa^+ m quad (m in ran W_kappa), $
where $W_kappa^+$ is the pseudo-inverse. $c_kappa (m)$ is the smallest $norm(c)^2$ with $m = O_kappa^top c$: the
cheapest way to write the monitor as a combination of read directions pulled back through at most $kappa - 1$
loops.

#thm("Theorem 2")[
(a) *Only the observable compression matters.* In an orthonormal basis adapted to $K plus.o K^perp$,
$J^top = mat(A, B; 0, C)$. Any $J'$ with $J'^top = mat(A, B'; 0, C')$ has the same $W_p$, the same $S_p$ and the
same Fisher-weighted output response $Hh G J'^r xi$ for every $xi$ and $r$.

(b) *Monitors in $K$ inherit stability through their cost.* If $m in ran W_kappa$, then for every $r$,
$ EE_xi |m^top J^r xi|^2 <= c_kappa (m) dot sigma^2 sum_(k < kappa) tr(Hh G J^(r+k) (J^(r+k))^top G^top Hh). $
If also $p(r + k) >= p_min > 0$ for all $k < kappa$, the right-hand side is at most $c_kappa (m) S_p \/ p_min$.

(c) *No guarantee outside $K$.* If $m$ has a nonzero component in $K^perp$, then for every $M$ there is a $J'$ as
in (a) with $sup_r EE_xi |m^top J'^r xi|^2 >= M$.
]

_Proof._
(a) $K$ is $J^top$-invariant, so the lower-left block of $J^top$ is zero, and $(J^top)^r$ acts on $K$ as $A^r$.
The columns of $G^top Hh$ lie in $K$. Hence $(J'^top)^r G^top Hh = (J^top)^r G^top Hh$, and
$Hh G J'^r xi = ((J'^top)^r G^top Hh)^top xi$ is unchanged. So are $W_p$ and $S_p$.

(b) Write $m = O_kappa^top c$ with $norm(c)^2 = c_kappa (m)$, splitting $c$ into blocks $c_k$. Then
$m^top J^r xi = sum_(k < kappa) c_k^top Hh G J^(k + r) xi$. By Cauchy–Schwarz,
$|m^top J^r xi|^2 <= norm(c)^2 sum_k norm(Hh G J^(k + r) xi)^2$. Take the expectation using
$EE xi xi^top = sigma^2 I$. For the second bound, $S_p >= p_min sigma^2 sum_k tr(dots)$.

(c) Take $B' = B$ and $C' = t I$ with $t > 1$. Write $m = (m_K, m_U)$ with $m_U != 0$. The $K^perp$-component of
$(J'^top)^r m$ is $t^r m_U$. Therefore
$EE |m^top J'^r xi|^2 = sigma^2 norm((J'^top)^r m)^2 >= sigma^2 t^(2 r) norm(m_U)^2 -> oo$. $square$

*In words:* the training loss pins down the loop only as seen through the read directions, after any number of
loops. A monitor that is cheap to write in those terms is, under start perturbations, as stable as the output. A
monitor that reads elsewhere gets no stability from training, and the loop is free to amplify or flip what it reads.

Theorem 2(b) bounds the *perturbation-induced variance* of a monitor's reading. It says nothing directly about
classification performance, or about drift between budgets, prompts or operating points. Monitor transfer is a
hypothesis that needs further assumptions about labels and distribution shift (P3).

#caveat[
Mathematically, $K$ is often all of $RR^d$, while $c_kappa (m)$ is astronomically large. That is why the measured
quantity is the cost $c_kappa (m)$ at finite $kappa$, not membership in $K$.
- The theorem needs the exact range condition. A truncated pseudo-inverse can give zero cost to a weakly observed
  direction it discards, and ridge regularization changes the bound.
- So any estimate must report the reconstruction residual $m - O_kappa^top c$. That residual gets no guarantee, as in
  part (c).
- $ker G$ is the *one-step* blind spot. An example is the radial direction for a scale-invariant (normalized)
  readout, the "readout blind spot" of Sharma and Vu (arXiv 2606.24898).
- $K^perp$ is the *all-depth* blind spot, a smaller set: a direction in $ker G$ can still become visible after a few
  loops.
]

= Predictability and control (standard; GPT-6 Astra)

#thm("Proposition 3 (frozen-Jacobian prediction under contraction)")[
Let $F$ be $C^2$ on a convex set containing the trajectory, with $norm(D F)_2 <= q < 1$ and
$norm(D^2 F) <= M$ there. Start at $x_0 = x$ and iterate $x_(t+1) = F(x_t) + u$ for an integer horizon $h >= 1$.
Put $J = D F(x)$,
$v = F(x) - x + u$ and $S_h = sum_(j < h) J^j$. Then
$ norm(x_h - x - S_h v) <= M norm(v)^2 \/ (2 (1 - q)^3). $
]

_Proof._ Let $e_t = x_t - x$. Then $e_(t+1) = F(x_t) - F(x) + v$, so $norm(e_t) <= norm(v) \/ (1 - q)$. The linear
model satisfies $tilde(e)_(t+1) = J tilde(e)_t + v$, so $tilde(e)_h = S_h v$. The difference
$epsilon_t = e_t - tilde(e)_t$ satisfies $epsilon_(t+1) = J epsilon_t + R_t$ with
$norm(R_t) <= M norm(e_t)^2 \/ 2$. Sum the geometric series. $square$

- With $u = 0$, this predicts the effect of *extra loops* from the current residual $F(x) - x$.
- At a fixed point with $u = B p$, it predicts *persistent steering*.
- The best first-order gain per unit push on a linear readout $g$ is $norm((S_h B)^top g)$.
- The bound needs only contraction and smoothness, so an analogous bound holds for a feedforward stack. What
  recurrence adds is a reusable $J^j$ and a resolvent.

*Counterexamples (checked).*
- An expanding *linear* loop is perfectly predictable: linearization is exact. So stability is not needed for
  predictability.
- $J = mat(0, K; 0, 0)$ has spectral radius 0 but a transient amplification of size $K$. Eigenvalues alone do not
  bound gains.
- For a stable scalar mode $lambda$, constant forcing has gain $1\/(1 - lambda)$, but forcing that alternates in
  sign has gain $1\/(1 + lambda)$. Near a flip ($lambda -> -1$) the loop amplifies *alternating* pushes, not constant
  ones. So "more stable" does not mean "less controllable" either.

*Fair comparison with the feedforward partner:*
- push at the *same boundary* (the output of the prelude, which in the looped model re-enters at every loop);
- use doses normalized by the state covariance;
- charge persistent pushes their total energy $sum_t norm(u_t)^2$;
- report inference FLOPs separately;
- compare intervention prediction directly, and depth extrapolation separately, since extra loops have no
  counterpart in a fixed-depth model.

= Monitoring under contraction and noise (GPT-6 Astra)

#thm("Proposition 4")[
(a) A common invertible affine map of the state does not change the best AUROC of a linear monitor: pull the
monitor's weight vector back through the inverse. So noiseless contraction $h_t = q^t h_0$ with $q != 0$ loses
nothing.

(b) Assume equal class priors, $h_0 | Y = plus.minus 1 tilde cal(N)(plus.minus delta \/ 2, Sigma)$, and
observations $z_t = J^t h_0 + eta$ with independent $eta tilde cal(N)(0, sigma^2 I)$, $sigma > 0$ and
$norm(J) <= q < 1$. Then the best possible classification error satisfies
$ P_e (t) >= Phi(-q^t norm(delta) \/ (2 sigma)). $
]

_Proof of (b)._ The optimal error for two Gaussians with equal covariance and equal priors is $Phi(-D_t \/ 2)$,
where $D_t$ is the Mahalanobis distance. The observed covariance is at least $sigma^2 I$, so
$D_t <= norm(J^t delta) \/ sigma <= q^t norm(delta) \/ sigma$. $square$

Contraction plus noise erodes what a monitor can read, but re-injecting the input can regenerate the signal at every
loop. This concerns signal-to-noise, not weight sharing. A read-space self-gain of either sign does not, by itself,
establish that monitoring is hard.

= Consistency with existing observations (not tests)

+ *Huginn's blinking produced no observed answer changes.* Census positions blink with period 2. Even so, greedy
  outputs were identical at budgets 509–512 for all 26 blinking census prompts, and teacher-forced answers never
  flipped (README, census and orbit runs). On synthetic interpolated inputs (off-distribution, where $S_p$ need not be
  small), a *visible* parity flicker did occur.
  - This is consistent with Corollary 1a, but it does not establish that the output distributions were unchanged,
    or that $Hh G v = 0$.
  - A period-2 orbit is not a fixed point; its stability is governed by the two-step Jacobian product. What the runs
    established is weaker: each cycle is consistent with a flip bifurcation. Between the two cycle states lies an
    unstable fixed point with a validated real eigenvalue just past $-1$ (between $-1.226$ and $-1.007$ in the census).
    The linear statements here concern that fixed point, not the cycle.
+ *Noise at the loop entry* (Olszowiec et al. 2026, Fig. 14).
  - The random-start families shrink it: Rec-OLMo by 0.71–0.91; Huginn about 1.08–1.10, then 0.93–0.97.
  - LoopUS, trained from a deterministic start, amplifies it 9–10 times.
  - These are finite-amplitude hidden-state distances relative to the first iteration, not a local Fisher
    sensitivity. They are consistent with "random-start training makes $S_p$ small" but do not test it. Random start
    and input re-injection also coincide in these families, so the two causes cannot be separated with them.
+ *Ouro's next-token accuracy drops past its training depth of 4* (Olszowiec, App. G and Table 8: 0.605 to 0.580 at
  6 loops; not a task-success rate). This is consistent with Corollary 1b. But Ouro is never trained with start
  noise, so Theorem 1's hypothesis is not trained at all; this is not a test of the theory.

= Predictions on matched pairs (each run at most 30 GPU-min)

*Free depth ablation.* Recurrent-Llama-3.2 is released at mean Poisson rates $M = 4, 8, 16, 32$, with the same data
and the same surgery. Theorem 1 bounds the *visibility* of slow modes through $m_p (|lambda|^2)$.
- The training code draws $1 + "Poisson"(Lambda)$ loops, with $Lambda$ lognormal ($EE Lambda = M$, $sigma = 0.5$).
  For $|lambda| = 0.9$, $m_p (0.81) = 0.41, 0.22, 0.081, 0.016$ (Monte Carlo; Astra confirmed it by quadrature).
- So, at equal $Omega_p$, such a mode may show in the output with squared amplitude up to $Omega_p \/ 0.41$ at
  $M = 4$, against $Omega_p \/ 0.016$ at $M = 32$: a factor of 25.
- *This yields no ordering prediction* (Astra's counterexample). Give one scalar mode the readout gain $a$ with
  $a^2 = 10^(-6) \/ m_p (lambda^2)$, and take $lambda = 0.95$ at $M = 4$ but $lambda = 0.8$ at $M = 32$. Both
  models have the same $Omega_p = 10^(-6)$, yet the depth-4 model's perturbations persist *longer*
  ($F_8 \/ F_1 = lambda^14$). This can happen at exact fixed points.
- So the pilot (docs/oversight_pilot.md) is an *empirical association study* of released checkpoints. They also
  differ in operating point, compute, curriculum and gradient horizon. Neither outcome confirms or refutes Theorem 1.
- The released GSM8K numbers are descriptive only. The depth-4 model has settled by 4 loops (39.4 at 4, 40.0 at 32);
  the depth-32 model keeps improving up to 32 (44.6, 48.1, 49.4 at 4, 8, 32).


Models:
- Huginn-0125, and its non-recurrent twin from the same training run;
- Recurrent-Llama-3.2 trained at mean Poisson rates 4, 8, 16 and 32, against `Llama-3.2-non-recurrent-posttrained` (same
  50B tokens);
- Recurrent-OLMo-2;
- Ouro and LoopUS (deterministic start).

/ P1 ($S_p$ directly): perturb a converged (or depth-$R$) state by isotropic noise at several $sigma$, run $k$ more
  loops, and measure the KL divergence from the unperturbed output. Forward passes only.
  - Hypothesis (not derived): small and decaying in $k$ for random-start models.
  - Not predicted for Ouro and LoopUS beyond their training depth.
/ P2 (observable versus global spectral radius): run Arnoldi on $J^top$ (vector–Jacobian products), started from the
  Fisher read directions (contrasts among the top tokens) and, separately, from random vectors.
  - What the theory implies (Corollary 1a, heavy-tailed depths, finite $Omega_p$): no *visible expanding*
    eigenmodes. A strict observable contraction (observable radius below 1 where the global radius exceeds 1) is an
    empirical hypothesis; it would need the quantitative condition of Theorem 1′ on the observable block.
/ P3 (monitor transfer versus cost; a larger evaluation, deferred): train linear monitors at budget $r$, compute
  $c_kappa (m)$, and test across budgets and starts.
  - Hypothesis: perturbation-induced drift increases with cost (Theorem 2b bounds exactly that). Transfer across
    budgets needs further assumptions about labels and distribution shift.
/ P4 (the feedforward partner): it has no loop, so P1–P3 have no analogue.
  - The theory predicts nothing about which model is easier to monitor at the default budget; that comparison is
    empirical.
  - High-cost monitor failure is not unique to recurrence. The distinctive setting is transfer when the inference
    depth of one recurrent model changes, where the theory points to high-cost monitors as the ones at risk.

= Standard versus new; limits

- *Standard:* observability Gramians, the PBH test and the Kalman decomposition from linear control theory; the
  Fisher metric of the softmax.
- *A possibly useful synthesis* (pending a systematic literature review; overlaps with Sharma and Vu, arXiv
  2606.24898):
  - the depth-weighted Fisher-observability Gramian as the object that random-depth training controls;
  - the generating-function evaluation, and the output-invisibility of expanding modes under heavy-tailed depths;
  - observability cost as a candidate for whether a monitor survives a change of inference depth.
- *Limits:*
  - Everything is linearized at a fixed point; stability is assumed only where a result says so.
  - $S_p$ is a linearized quantity. Its link to the finite-noise loss needs the extra assumptions of Section 1.
  - "$S_p$ is small" is a hypothesis to measure, not a result.
  - Random training starts are far from $s^*$.
  - The second-order loss identity assumes targets distributed as $pi$.
  - Nothing here compares looped and feedforward models at the default budget.
