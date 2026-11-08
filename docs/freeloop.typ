#set document(title: "Generic and readout-specific steering geometry in looped models")
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
#let check(body) = boxed("Numerical check", rgb("#eef8ee"), body)
#let caveat(body) = boxed("Careful", rgb("#fdeeee"), body)
#let sgn = math.op("sgn")
#let tr = math.op("tr")
#let gh = $hat(g)$
#let ip(x, y) = $chevron.l #x, #y chevron.r$

#align(center)[
  #text(16pt, weight: "bold")[Generic and readout-specific steering geometry in looped models]
  #v(0.2em)
  #text(10.5pt)[Random directions, a spectral cancellation theorem, a rejected free-loop null model, and a read-space
  self-gain excess: negative in Huginn, positive in Ouro under the tested protocols]
  #v(0.2em)
  #text(9.5pt)[Companion to `docs/analytic.pdf`. Draft 3, 2026-11-07 (Drafts 1–2 are superseded). Verified by
  `docs/freeloop_check.py`.]
]

#v(0.4em)

#plain[
`analytic.pdf` decomposes the steering lever of a looped model exactly. This note asks which features of that
geometry a readout has *because it is a readout*, and which any direction would have. The baseline is a uniformly
random direction.

- *Random directions see traces (Theorem 1).* For a random unit direction $z$, every lever statistic concentrates
  on a normalized trace of the loop Jacobian $J$. The self-gain $alpha$, the return $psi$ and the sum $E + 2C$ depend
  only on the eigenvalues of $J$.
- *The cancellation is spectral (Theorem 2).* For random directions, $E + 2C$ tends to $(1 - gamma) S \/ (1 + S)$,
  where $gamma$ is the mean eigenvalue and $S$ is a series in the centred spectral moments. It is $0$ when the
  spectrum is rotation-invariant about $gamma$ and $>= 0$ when the spectrum is real. When the spectrum is concentrated
  near its centre, its leading term is $kappa_2 \/ (1 - gamma)$ with $kappa_2 = "mean" (lambda - gamma)^2$: positive
  when eigenvalues spread more along the real axis, negative when they spread more along the imaginary axis. For
  spread-out spectra higher moments can reverse this sign.
- *Non-normality (Theorem 3).* For given eigenvalues, non-normality makes the random lever longer and turns it away
  from its start.
- *Readout excess (Proposition R).* A readout's self-gain minus the mean eigenvalue is a quadratic form in the centred
  symmetric part of $J$. Over random directions it has mean $0$ and an explicit variance, which gives the readout
  comparison a natural unit.
- *Free loops (Theorems 4–5).* If $J$ is a gain profile times a uniformly random rotation, plus a self-gain, every
  lever statistic of every readout has an explicit limit. This is an exactly solvable null model.
- *Huginn: a pre-registered test.* Random probes at 40 truth and 20 digit contexts (2 runs, about 8 GPU-minutes).
  The free model fails the pre-registered adequacy criteria in both. It predicts that successive pullbacks align and
  decay slowly, and they do not. The *loop signature* has similar broad scales in the two contexts, with measurable
  differences: mean eigenvalue $tr(J) \/ d = 0.24$ and $0.26$, $kappa_2 = -0.021$ and $-0.019$, chain decay about
  $0.71$–$0.75$ per step, and a random-direction steering advantage of $4.1$ and $4.3$. Random directions have $E + 2C$
  slightly negative (about $-9%$ of $E$). This is consistent with $kappa_2 < 0$: the series of Theorem 2, evaluated
  with the measured trace moments, reproduces it to within a factor $0.92$–$1.08$ from third order on. The conditions under which
  Theorem 2 guarantees this are not verified for Huginn.
- *Readout suppression.* All 240 readouts tested (40 truth, 200 digit) have a self-gain below the probe estimate of
  their loop's mean eigenvalue, and below the self-gain of each of the 4 random probes at the same context: truth
  $0.12$ against $0.24$; digits a median excess of $-0.21$ (95% CI $[-0.24, -0.20]$, a pre-specified replication). In
  units of the random-direction spread (Proposition R) the median readout lies 15 (truth) and 26 (digits) standard
  deviations below a random direction, and none lies less than 7.9 below. So loop-aware steering pays more for
  readouts than for random directions: $2.7$ times more for truth, $4.4$ times for digits.
- *Beyond the task readouts (pre-registered control).* Sampled directions that the output layers read but that are
  not chosen by the task (contrasts between random tokens, between plausible tokens, and random combinations of all
  logits) have the same deficit: all 1,440 control directions lie below all 16 probes of their context; the random
  combinations (C3) have median standardized excess $-28.0$ (truth) and $-24.3$ (digits). The True/False readout is
  even *less* suppressed than a typical sampled read direction. This does not establish suppression throughout the
  complete read space. Exploratory: pullbacks of read directions have the size of random ones (one-step size $0.62$
  against $0.64$) but a much lower self-cosine ($0.03$ against $0.38$). A natural candidate mechanism, preferential
  forgetting of Huginn's random start state along the readouts, finds no support in the existing 48-step chains at
  the fixed point: in median, readout chains stay larger than random ones at every step up to 32.
- *A second looped model (pre-registered).* Ouro-1.4B, which loops its whole 24-layer stack 4 times, on the same 40
  prompts: no negative read-space excess detected; the measured C3 excess is positive ($+0.022$, CI $[0.016, 0.025]$),
  the mean eigenvalue is small ($0.06$), and the self-cosine is low for probes and sampled read directions alike. This
  does not establish that Ouro has no suppressed read directions. Under these protocols the read-space excess of
  @sec-metric has opposite signs in the two models.

So the circulation–passage cancellation of `analytic.pdf` is generic: random directions share it, and for them it is,
up to fluctuations of order $d^(-1\/2)$, a function of the spectrum. The near-orthogonality of the readout's lever is *not* generic. A random direction's lever
starts at $c_0 approx 0.25$, the truth readout's at $0.09$. What sets Huginn's readouts apart is that they are among
the directions its output layers read, and the sampled read directions are fed back much less than a typical
direction. This is not a property of looped models in general: in Ouro's tested loops the read-space excess is
positive. Draft 1 of this note called the near-orthogonality generic, on the basis of the free model; that claim is
withdrawn. Draft 2 attributed the deficit to the readouts; the control of @sec-suppression shows that it extends to
the sampled non-task read directions. @sec-metric turns these findings into a metric for comparing loops and
readouts.
]

= Setting <sec-setting>

$J in RR^(d times d)$ is the loop Jacobian at the fixed point, $g != 0$ the readout gradient, $gh = g \/ norm(g)$.
The *pullback chain* of a unit start vector $u$ (a readout $gh$, or a random probe $z$) is
$ w_0 = u, quad w_t = J^top w_(t-1) = (J^top)^t u . $
One step of the chain is one application of $J^top$ (one vector–Jacobian product, "VJP"). The $T$-step lever is
$a_T = norm(g) sum_(t=1)^T w_t$ (with $norm(g) = 1$ for a probe); when the spectral radius $rho(J) < 1$,
$a_T -> a = (I - J^top)^(-1) J^top g$, the lever of `analytic.pdf` (push added to the state before the loop map).
The statistics:

#table(columns: 2, stroke: none, inset: 3pt,
  [$alpha = ip(u, w_1)$], [*self-gain*: start component of one pullback,],
  [$s_g = norm(w_1 - alpha u)$], [*first-step gain* outside the start direction,],
  [$Gamma_T = norm(a_T) \/ norm(g)$], [*lever gain*,],
  [$c_0 = ip(u, a_T) \/ norm(a_T)$], [*start*: cosine between lever and start direction,],
  [$c_1 = ip(w_1, a_T) \/ norm(a_T)$], [*first-step response* per unit lever,],
  [$kappa = cos angle(w_1, a_T)$], [*one-step lever quality*: how well $J^top u$ points along the lever,],
  [$1 \/ |c_0|$], [*steering advantage*: effect of the best push over a push along $u$, at equal norm.],
)

The return, passage and circulation terms are those of `analytic.pdf`. For finite $T$ we use its estimates (§9
there): $x_T = ip(gh, a_T) \/ norm(g)$, $sigma_T = 1 \/ (1 + x_T)$, $psi_T = x_T sigma_T$,
$E_T = sigma_T (ip(w_1, a_T) \/ norm(g) - alpha x_T)$, and $C_T$ is *defined* by $psi_T = alpha + E_T + 2 C_T$. At
$T = infinity$ these are the exact quantities (Theorems 1–2 of `analytic.pdf`). So "$E_T + 2 C_T approx 0$" is the
same statement as "$psi_T approx alpha$". Defined this way, $C_T$ is *reconstructed*: it absorbs the truncation
residual. In Huginn the last chain term $norm(w_48)$ is below $3 times 10^(-5)$ of the lever for every start (the
validity gate allows $10^(-3)$), and a geometric extrapolation of the remaining tail is at most $1.6 times 10^(-4)$ of
the lever (@sec-test). Neither is a rigorous bound on the infinite tail.

= Random directions <sec-random>

Throughout, $z$ is uniform on the unit sphere of $RR^d$ and independent of $J$, and $tau(X) = tr(X) \/ d$. When
$I - J$ is invertible, the lever of a unit start $u$ is $a(u) = M^top u$ with $M = J (I - J)^(-1)$, so
$x(u) = ip(u, M^top u)$ and $Gamma(u) = norm(M^top u)$.

*Lemma S (uniform vectors).* If $xi$ is uniform on the unit sphere of a subspace $S$ of dimension $m$, $M$ is a
symmetric matrix and $v$ a vector (both fixed, or independent of $xi$), then $EE ip(xi, v)^2 = norm(P_S v)^2 \/ m$,
$EE xi^top M xi = tr(P_S M) \/ m$, and $"Var"(xi^top M xi) <= 2 norm(M)^2 \/ m$. These are the standard moments of the
uniform distribution on $S^(m-1)$.

For a non-symmetric $B$, $ip(xi, B xi) = ip(xi, B_s xi)$ with $B_s = (B + B^top) \/ 2$ and $norm(B_s) <= norm(B)$,
so the last two statements hold for every square $B$.

#thm("Theorem 1 (random directions see traces)")[
Let $J = J_d$ with $norm(J) <= C$, $norm((I - J)^(-1)) <= C$ and $|tau((I - J)^(-1))| >= 1 \/ C$ for all $d$. For
fixed $t, u >= 0$, each of the following differences is $O_p (d^(-1\/2))$:
$ ip((J^top)^t z, (J^top)^u z) - tau(J^t (J^top)^u), quad alpha(z) - tau(J), quad x(z) - tau(M), $
$ Gamma(z)^2 - tau(M M^top), quad ip(J^top z, a(z)) - tau(J M^top) . $
Here $tau(J) = (1 \/ d) sum_i lambda_i$ and $tau(M) = (1 \/ d) sum_i lambda_i \/ (1 - lambda_i)$ are averages over the
eigenvalues $lambda_i$ of $J$ (with algebraic multiplicity). Consequently $sigma(z)$, $psi(z)$ and
$E(z) + 2 C(z) = psi(z) - alpha(z)$ are within $O_p (d^(-1\/2))$ of functions of the eigenvalues alone:
$ E(z) + 2 C(z) = Phi_d + O_p (d^(-1\/2)), quad Phi_d := 1 - 1 \/ tau((I - J)^(-1)) - tau(J) . $
$Gamma$, $c_0 = x \/ Gamma$, $c_1$ and $E$ separately depend on more than the eigenvalues.
]

*Proof.* Each quantity has the form $ip(z, B z)$ for a fixed $B$ of norm at most a power of $C$:
$B = J^t (J^top)^u$, $J$, $M$, $M M^top$, $J M^top$. Lemma S with $m = d$ gives mean $tau(B)$ and variance at most
$2 norm(B)^2 \/ d$. For the eigenvalue statement, write $J = Q R Q^*$ (Schur form; $Q$ unitary, $R$ upper triangular
with diagonal $lambda_i$). For a rational function $h$ without poles on the spectrum, $h(J) = Q h(R) Q^*$, and $h(R)$
is upper triangular with diagonal $h(lambda_i)$, so $tr h(J) = sum_i h(lambda_i)$. Apply this to $h(lambda) = lambda$,
$lambda \/ (1 - lambda)$ and $1 \/ (1 - lambda)$. Finally $sigma = 1 \/ (1 + x)$, $1 + tau(M) = tau((I - J)^(-1))$, and
$psi = 1 - sigma$; these maps are Lipschitz where $|tau((I - J)^(-1))| >= 1 \/ C$. $square$

#thm("Theorem 2 (the cancellation is spectral)")[
Let $mu$ be a probability measure on $CC$ (for Theorem 1, $mu = (1 \/ d) sum_i delta_(lambda_i)$), let
$gamma = integral lambda dif mu$, and suppose $mu$ is supported in a disc $|lambda - gamma| <= r$ with
$r < |1 - gamma|$. Put $kappa_k = integral (lambda - gamma)^k dif mu$ and $S = sum_(k >= 2) kappa_k \/ (1 - gamma)^k$.
Then
$ integral (dif mu (lambda)) / (1 - lambda) = (1 + S) / (1 - gamma), quad
  Phi(mu) := 1 - (integral (dif mu (lambda)) / (1 - lambda))^(-1) - gamma = ((1 - gamma) S) / (1 + S)
  quad (1 + S != 0) . $
+ If $mu$ is invariant under rotations about $gamma$, then $kappa_k = 0$ for all $k >= 1$, and $Phi(mu) = 0$.
+ If $mu$ is supported in the real interval $(-infinity, 1)$ with $integral |lambda| dif mu < infinity$ and
  $integral dif mu \/ (1 - lambda) < infinity$ (automatic for finitely many atoms; the disc condition is not needed
  here), then $Phi(mu) >= 0$, with equality only for a point mass.
+ If $r <= |1 - gamma| \/ 2$, then $|Phi(mu) - kappa_2 \/ (1 - gamma)| <= 6 r^3 \/ |1 - gamma|^2$. For a measure
  symmetric under complex conjugation (the spectrum of a real $J$), $gamma$, $kappa_2$ and $Phi$ are real and
  $kappa_2 = integral ((op("Re") lambda - gamma)^2 - (op("Im") lambda)^2) dif mu$; if moreover $gamma < 1$, then
  $Phi$ has the sign of $kappa_2$ when $|kappa_2| > 6 r^3 \/ (1 - gamma)$. (For $gamma > 1$ the sign is that of
  $-kappa_2$: e.g. $mu = (delta_(1.9) + delta_(2.1)) \/ 2$ has $kappa_2 = 0.01$ and $Phi = -0.01$.)
]

*Proof.* On the support $|lambda - gamma| \/ |1 - gamma| <= r \/ |1 - gamma| < 1$, so
$1 \/ (1 - lambda) = (1 - gamma)^(-1) sum_(k >= 0) ((lambda - gamma) \/ (1 - gamma))^k$ converges uniformly.
Integrating term by term, with $kappa_0 = 1$ and $kappa_1 = 0$, gives $(1 + S) \/ (1 - gamma)$, and then
$Phi = 1 - (1 - gamma) \/ (1 + S) - gamma = (1 - gamma) S \/ (1 + S)$.
(1): $integral_0^(2 pi) e^(i k theta) dif theta = 0$ for $k >= 1$.
(2): $h(lambda) = 1 \/ (1 - lambda)$ is strictly convex on $(-infinity, 1)$, and $gamma < 1$. By Jensen,
$integral h dif mu >= h(gamma) = 1 \/ (1 - gamma) > 0$, with equality only for a point mass, so
$1 - 1 \/ integral h dif mu >= gamma$.
(3): Put $epsilon = r \/ |1 - gamma| <= 1\/2$. Since $|kappa_k| <= r^k$, $|S| <= epsilon^2 \/ (1 - epsilon) <= 2 epsilon^2 <= 1\/2$,
and $R := S - kappa_2 \/ (1 - gamma)^2$ has $|R| <= epsilon^3 \/ (1 - epsilon) <= 2 epsilon^3$. Then
$Phi - kappa_2 \/ (1 - gamma) = (1 - gamma)(R - S^2 \/ (1 + S))$ and $|S^2 \/ (1 + S)| <= 8 epsilon^4 <= 4 epsilon^3$, so the
difference is at most $6 |1 - gamma| epsilon^3$ in modulus. $square$

Concentration matters in (3). The conjugation-symmetric measure $0.1 delta_(0.9) + 0.1 delta_(-0.9) +
0.4 delta_(0.5 i) + 0.4 delta_(-0.5 i)$ satisfies the disc condition ($gamma = 0$, $r = 0.9 < 1$) but not
$r <= |1 - gamma| \/ 2$. It has $kappa_2 = -0.038$ but $Phi = +0.41$: the eigenvalues near $1$ dominate the higher
moments.

*Reading.* For random directions $E + 2C$ measures the *shape* of the spectrum around its centre. It vanishes for
circular spectra, whatever the radial profile and however non-normal $J$ is. It is positive for real spectra, as in
symmetric loops (consistent with Corollary 2a of `analytic.pdf`, where $C = 0$ and $E > 0$). For spectra concentrated
near their centre and stretched along the imaginary axis, as in rotation-dominated loops, it is negative (for $gamma < 1$). For a
*normal* $J$ the formulas hold exactly for every unit vector $u$, with $mu$ replaced by its spectral measure
$mu_u = sum_i |v_i^* u|^2 delta_(lambda_i)$ ($v_i$ orthonormal eigenvectors): $psi(u) - alpha(u) = Phi(mu_u)$. Theorem 2
then applies to $mu_u$, whose centre is $alpha(u)$, not $gamma$; a rotation-invariant spectrum need not give a
rotation-invariant $mu_u$.

#thm("Theorem 3 (non-normality lengthens and turns random levers)")[
In the setting of Theorem 1, let $nu_i = lambda_i \/ (1 - lambda_i)$ be the eigenvalues of $M$. Then
$ tau(M M^top) >= (1 \/ d) sum_i |nu_i|^2 , $
with equality if and only if $J$ is normal. Hence $Gamma(z)^2 >= "mean"_i |nu_i|^2 - O_p (d^(-1\/2))$. If moreover
$"mean"_i |nu_i|^2 >= 1 \/ C$ for all $d$, then $c_0 (z) = tau(M) \/ sqrt(tau(M M^top)) + O_p (d^(-1\/2))$ and
$ |c_0 (z)| <= |"mean"_i nu_i| \/ sqrt("mean"_i |nu_i|^2) + O_p (d^(-1\/2)) , $
with equality up to $O_p (d^(-1\/2))$ when $J$ is normal. So for a fixed spectrum the limiting start
$|tau(M)| \/ sqrt(tau(M M^top))$ of a random direction is largest when $J$ is normal: non-normality can only lower it,
and so only raise the random-direction steering advantage.
]

*Proof.* Schur's inequality: in a Schur form $M = Q (D + N) Q^*$ with $D$ diagonal and $N$ strictly upper triangular,
$norm(M)_F^2 = sum_i |nu_i|^2 + norm(N)_F^2$, and $N = 0$ exactly when $M$ is normal. $M = J (I - J)^(-1)$ and
$J = M (I + M)^(-1)$ are functions of each other, so $M$ is normal if and only if $J$ is. Combine with Theorem 1:
$c_0 = x \/ Gamma$ with $x = tau(M) + O_p (d^(-1\/2))$, $tau(M) = "mean"_i nu_i$ (real for real $J$), and
$Gamma^2 = tau(M M^top) + O_p (d^(-1\/2))$ with $tau(M M^top) >= 1 \/ C$, so $Gamma$ is bounded away from $0$ and the
quotient keeps the rate. Without the lower bound the rate can fail: for
$M = "diag"(1, d^(-1\/2), ..., d^(-1\/2))$, $c_0 (z)$ has a non-degenerate limit law. $square$

#thm("Proposition R (readout excess)")[
Let $H = (J + J^top) \/ 2$ and $B = H - tau(J) I$. For every unit vector $u$,
$ alpha(u) - tau(J) = u^top B u . $
For $z$ uniform on the unit sphere of $RR^d$, $EE alpha(z) = tau(J)$ and
$ "Var" alpha(z) = (2 tau(B^2)) / (d + 2), quad tau(B^2) = (tau(J^2) + tau(J J^top)) \/ 2 - tau(J)^2 . $
If $z$ is uniform on the unit sphere of a subspace with orthogonal projection $P$ of rank $r$, then
$EE alpha(z) = tr(P J) \/ r$.
]

*Proof.* $alpha(u) = ip(u, J^top u) = u^top H u$ and $tau(H) = tau(J)$. For $z$ uniform on $S^(d-1)$,
$EE z_i z_j z_k z_l = (delta_(i j) delta_(k l) + delta_(i k) delta_(j l) + delta_(i l) delta_(j k)) \/ (d (d + 2))$, so for
symmetric $B$, $EE (z^top B z)^2 = ((tr B)^2 + 2 tr(B^2)) \/ (d (d + 2))$; here $tr B = 0$. Also
$tau(H^2) = (tau(J^2) + tau(J J^top)) \/ 2$, because $tau(J^top J) = tau(J J^top)$ and $tau((J^top)^2) = tau(J^2)$. The
subspace statement is Lemma S. $square$

So the *standardized excess* $Z(u) = (alpha(u) - tau(J)) \/ sqrt(2 tau(B^2) \/ (d + 2))$ places a readout's self-gain
on the scale of random directions. By Chebyshev, a random direction has $|Z| >= k$ with probability at most $1 \/ k^2$.
All three traces are estimated by the probes: $m_1$, $m_2$ and $"mean" norm(J^top z)^2$.

= An exactly solvable case: free loops <sec-free>

Theorems 1–3 describe random directions in any loop. A readout is not random, so predicting its statistics needs a
model of how it sits in the loop. The simplest model has no readout-aligned structure at all.


*Definition.* A *free loop* is $J = A O$, where $O$ is distributed according to Haar measure on the orthogonal group
$O(d)$ and is independent of $A$ and of the readout. Its *loop gain* $s >= 0$ and *readout gain* $s_g >= 0$ are
$ s^2 = norm(A)_F^2 \/ d = (1 \/ d) sum_i sigma_i (J)^2, quad s_g = norm(A^top gh) = norm(J^top gh) . $
($O$ is orthogonal, so $J$ has the singular values of $A$, and $norm(J^top gh) = norm(O^top A^top gh) = norm(A^top gh)$
for every $O$.)

*Asymptotic regime.* Along a sequence $d -> infinity$, with $A_d$, the readouts and any test vector $v$ jointly
independent of $O$: $sup_d norm(A_d) < infinity$, $norm(A_d)_F^2 \/ d -> s^2$, $norm(A_d^top gh_d) -> s_g$, and
$sup_d norm(v_d) < infinity$. For several readouts $gh^((1)), ..., gh^((k))$ also
$ip(A_d^top gh^((j)), A_d^top gh^((l))) -> B_(j l)$. Limits are in probability; $T$ and $k$ stay fixed.

#thm("Lemma H (Haar conditioning)")[
Let $Q$ be Haar on $O(d)$ and $Z$ a random element independent of $Q$. Let $y_1 = f_0 (Z)$ and
$y_(k+1) = f_k (Z, Q y_1, ..., Q y_k)$ for measurable $f_k$. Let $P_k$ be the orthogonal projection onto
$"span"(y_1, ..., y_k)$ ($P_0 = 0$), $r_k = (I - P_k) y_(k+1)$, $S_k = "span"(Q y_1, ..., Q y_k)^perp$, and $cal(F)_k$
the information in $(Z, Q y_1, ..., Q y_k)$. Then $Q P_k y_(k+1)$ is $cal(F)_k$-measurable, and conditionally on
$cal(F)_k$, if $r_k != 0$,
$ Q r_k = norm(r_k) xi , $
with $xi$ uniform on the unit sphere of $S_k$ (dimension at least $d - k$).
]

*Proof.* We show by induction on $k$ that $Q = R Q_k$, where $Q_k$ is an $cal(F)_k$-measurable orthogonal matrix with
$Q_k y_j = Q y_j$ for $j <= k$, and, conditionally on $cal(F)_k$, $R$ is distributed according to Haar measure on the
($cal(F)_k$-measurable) group of orthogonal maps that fix $S_k^perp$ pointwise (equivalently, Haar on $O(S_k)$). For $k = 0$ this is the
independence of $Q$ and $Z$ ($Q_0 = I$). Given the claim at $k$: $y_(k+1)$ is $cal(F)_k$-measurable, so
$Q P_k y_(k+1) = Q_k P_k y_(k+1)$ is too, because $P_k y_(k+1)$ lies in the span of $y_1, ..., y_k$. The vector
$v = Q_k r_k$ is $cal(F)_k$-measurable and orthogonal to every $Q_k y_j = Q y_j$, so $v in S_k$, and
$Q r_k = R v$ is uniform on the sphere of radius $norm(r_k)$ in $S_k$. Revealing $Q y_(k+1) = Q_k P_k y_(k+1) + R v$
reveals $R v$. Conditionally on $R v = v'$, a Haar $R$ on $O(S_k)$ is $R' R_0$, with $R_0$ any fixed orthogonal map of
$S_k$ sending $v$ to $v'$ and $R'$ Haar on the maps of $S_k$ that fix $v'$. This is the claim at $k + 1$ (if $r_k = 0$,
nothing new is revealed). $square$

This is the conditioning device of Bolthausen's and Bayati–Montanari's analyses of iterative algorithms; the conditional
law of a Haar matrix given linear constraints is Lemma 4 of Rangan, Schniter and Fletcher (2019).

#thm("Theorem 4 (free chain)")[
Let $J = A O$ be a free loop in the asymptotic regime. Let $h$ be a start vector, independent of $O$, with
$sup_d norm(h) < infinity$ and $norm(A^top h) -> s_h$, and let $w_t = (J^top)^t h$. Then, in probability, for
$1 <= t, u <= T$:
+ $ip(w_t, w_u) -> delta_(t u) thin s_h^2 s^(2(t-1))$;
+ $ip(v, w_t) -> 0$ for every test vector $v$ of the regime (for example $v = h$, or another readout);
+ for several readouts, with $w_t^((j)) = (J^top)^t gh^((j))$: $ip(w_t^((j)), w_t^((l))) -> B_(j l) s^(2(t-1))$;
  for $t != u$, $ip(w_t^((j)), w_u^((l))) + ip(w_u^((j)), w_t^((l))) -> 0$; and $ip(gh^((j)), w_t^((l))) -> 0$.
]

For the readout chain, $h = gh$ and $s_h = s_g$.

*Proof.* Put $Q = O^top$ (also Haar), $Z = (A, h, v)$, $y_1 = A^top h$, $y_(t+1) = A^top w_t$. Then $w_t = Q y_t$,
the sequence $y_t$ has the form of Lemma H, and $ip(w_t, w_u) = ip(y_t, y_u)$. We prove by induction on $t$ that
the Gram matrix of $y_1, ..., y_t$ converges to $"diag"(s_h^2, s_h^2 s^2, ..., s_h^2 s^(2(t-1)))$ and that
$ip(v, w_u) -> 0$ for $u < t$. The case $t = 1$ is the regime assumption.

Step. Let $P$ project onto $"span"(y_1, ..., y_(t-1))$. First, $norm(P y_t) -> 0$: for $t = 1$, $P = 0$; for
$t >= 2$ and $s_h s > 0$, the limiting Gram matrix of $y_1, ..., y_(t-1)$ is invertible and $ip(y_j, y_t) -> 0$, so the
coefficients of $P y_t$ tend to $0$; for $t >= 2$ and $s_h s = 0$, the induction hypothesis gives $norm(y_t) -> 0$. By
Lemma H,
$ w_t = Q P y_t + norm((I - P) y_t) xi_t , $
with $xi_t$ conditionally uniform on a subspace $S$ of dimension $m >= d - t + 1$ (the second term is absent if
$(I - P) y_t = 0$). Conditionally, $A$, $v$ and $y_1, ..., y_t$ are fixed, so Lemma S applies to them.

(b): $|ip(v, Q P y_t)| <= norm(v) norm(P y_t) -> 0$, and $ip(v, xi_t) -> 0$ by Lemma S.

Next, $y_(t+1) = A^top Q P y_t + norm((I - P) y_t) A^top xi_t$. The first term has norm at most
$norm(A) norm(P y_t) -> 0$. By Lemma S, $norm(A^top xi_t)^2 = xi_t^top A A^top xi_t$ has conditional mean
$tr(P_S A A^top) \/ m$, which lies within $(t - 1) norm(A)^2 \/ m$ of $norm(A)_F^2 \/ m -> s^2$, and conditional
variance at most $2 norm(A)^4 \/ m -> 0$. Also $ip(A^top xi_t, y_u) = ip(xi_t, A y_u) -> 0$ for $u <= t$, since
$norm(A y_u)$ is bounded. Hence $norm(y_(t+1))^2 -> s^2 dot s_h^2 s^(2(t-1))$ and $ip(y_(t+1), y_u) -> 0$.
Conditional convergence in probability implies unconditional convergence by bounded convergence.

(c): by (b), $ip(gh^((j)), w_t^((l))) -> 0$. Apply (a) to the start vectors $h_(plus.minus) = gh^((j)) plus.minus gh^((l))$,
for which $norm(A^top h_(plus.minus))^2 -> B_(j j) + B_(l l) plus.minus 2 B_(j l)$, and subtract:
with $X_(t u) = ip(w_t^((j)), w_u^((l))) + ip(w_t^((l)), w_u^((j)))$,
$ 2 X_(t u) = ip(w_t (h_+), w_u (h_+)) - ip(w_t (h_-), w_u (h_-)) -> 4 delta_(t u) B_(j l) s^(2(t-1)) , $
using linearity of $h |-> w_t (h)$. For $t = u$ the two terms of $X_(t t)$ are equal; for $t != u$, the second term of
$X_(t u)$ is $ip(w_u^((j)), w_t^((l)))$. $square$

Polarization gives only the symmetric combination for $t != u$. That is all the lever needs:
$ip(a_T^((j)), a_T^((l))) = sum_(t, u) ip(w_t^((j)), w_u^((l)))$ pairs $(t, u)$ with $(u, t)$.

#thm("Corollary 4 (the free lever)")[
In the setting of Theorem 4, let $S_T = sum_(t=0)^(T-1) s^(2t)$ and $s_g > 0$. In probability (the statistics are
defined with probability tending to one):
$ alpha -> 0, quad c_0 -> 0, quad Gamma_T^2 -> s_g^2 S_T, quad c_1 -> s_g \/ sqrt(S_T), quad
  kappa -> 1 \/ sqrt(S_T), quad cos angle(a_t, a_T) -> sqrt(S_t \/ S_T) quad (1 <= t <= T), $
$ sigma_T -> 1, quad psi_T -> 0, quad E_T -> s_g^2, quad 2 C_T -> -s_g^2 . $
For several readouts with $B_(j j), B_(l l) > 0$: $cos angle(a_T^((j)), a_T^((l))) -> B_(j l) \/ sqrt(B_(j j) B_(l l))$,
which is the limit of $cos angle(J^top g^((j)), J^top g^((l)))$, and $ip(gh^((j)), hat(a)_T^((l))) -> 0$.

If moreover $sup_d norm(A_d) <= a_max < 1$, the same holds at $T = infinity$ with $S_infinity = 1 \/ (1 - s^2)$:
$ Gamma -> s_g \/ sqrt(1 - s^2), quad c_1 -> s_g sqrt(1 - s^2), quad kappa -> sqrt(1 - s^2) . $
]

*Proof.* With $norm(g) = 1$ without loss of generality, every quantity is a continuous function of the Gram matrix of
$gh, w_1, ..., w_T$ (and of the other readouts' chains, through the symmetric combinations of Theorem 4(c)), and its
denominators have positive limits when $s_g > 0$. Substitute Theorem 4. For example
$E_T = sigma_T (ip(w_1, a_T) \/ norm(g) - alpha x_T) -> 1 dot (s_g^2 - 0)$, and
$2 C_T = psi_T - alpha - E_T -> -s_g^2$. At $T = infinity$: $norm(w_t) <= a_max^t$, so
$norm(sum_(t > T) w_t) <= a_max^(T+1) \/ (1 - a_max)$ uniformly in $d$, and the limits $d -> infinity$ and
$T -> infinity$ can be exchanged. $square$

*Remarks.*
- *Typical readouts.* If the readout is itself uniformly random, independent of $A$, then $s_g -> s$, so at
  $T = infinity$ $c_1 -> s sqrt(1 - s^2) <= 1 \/ 2$ (largest at $s = 1 \/ sqrt(2)$) and
  $Gamma -> s \/ sqrt(1 - s^2)$. This bound is for the unshifted loop; with a self-gain (@sec-selfgain) $c_1$ can be
  larger (for $J = 0.8 I$, $c_1 = 0.8$).
- *Spectral radius.* Under hypotheses on the singular values that are stronger than our second-moment regime, the
  single ring theorem (Guionnet, Krishnapur and Zeitouni 2011; their Theorem 18 covers Haar orthogonal $O$) says that
  the eigenvalue distribution of $A O$ converges to a rotation-invariant law on an annulus with outer radius $s$. So
  $rho(J) >= s - o(1)$. The matching upper bound (no outlier eigenvalues) is proved by Guionnet and Zeitouni (2012) for
  Haar *unitary* rotations. For the orthogonal case we know no reference and treat $rho(J) -> s$ as an assumption
  (numerically, $d = 2000$: $rho = 0.621$, $s = 0.614$). Under it, $kappa -> sqrt(1 - rho(J)^2)$: the quality of the
  one-step lever is set by the spectral radius. The theorems do not use this.
- *Background.* If $A_d$ also converges in $*$-distribution, $J$ is asymptotically R-diagonal (Nica and Speicher 1997),
  and R-diagonal elements satisfy $tau(x^t (x^*)^u) = delta_(t u) tau(x x^*)^t$ (Haagerup and Larsen 2000). Theorem 4
  is the analogue for the vector state of a fixed readout, $gh^top J^t (J^top)^u gh$. It does not follow from trace
  convergence, and it needs only the second-moment regime.

- *Relation to Theorem 2.* The eigenvalue law of a free loop is rotation-invariant (single ring), so Theorem 2(1)
  gives $E + 2C -> 0$ for random directions. Corollary 4 is stronger: it gives the cancellation for *every* readout
  independent of $O$.

#caveat[Corollary 4 is about a *model*. It says what a loop with no readout-aligned structure would show. It does not
say that Huginn's loop is free; @sec-test tests that, and rejects it.]

== Self-gain: the only source of a readout component in the model <sec-selfgain>

Add a uniform gain $gamma$ (every direction feeds itself) and a readout-specific gain $beta$:
$ J = gamma I + beta gh gh^top + A O, quad lambda := gamma + beta . $

#thm("Theorem 5 (self-gain)")[
Assume the asymptotic regime for $A$ and $gh$, $sup_d norm(A_d) <= a_max < 1 - gamma$ (so $gamma < 1$), $lambda != 1$,
and $D > 0$ (that is, $lambda != 0$ or $s_g > 0$), where $F^2 = s_g^2 \/ ((1 - gamma)^2 - s^2)$ and
$D = sqrt(lambda^2 + F^2)$. For the lever $a = (I - J^top)^(-1) J^top g$, in probability:
$ alpha -> lambda, quad sigma -> 1 - lambda, quad psi -> lambda, quad norm(f) -> F, quad
  E -> s_g^2 \/ (1 - gamma), quad E + 2 C -> 0, $
$ c_0 -> sgn(1 - lambda) lambda \/ D, quad Gamma -> D \/ |1 - lambda|, quad
  c_1 -> sgn(1 - lambda) (lambda^2 + s_g^2 \/ (1 - gamma)) \/ D . $
]

*Proof.* Write $I - J = (1 - gamma)(I - beta' P - A' O)$ with $beta' = beta \/ (1 - gamma)$, $A' = A \/ (1 - gamma)$
and $P = gh gh^top$. Since $norm(A') <= a_max \/ (1 - gamma) < 1$, $R' = (I - A' O)^(-1) = sum_t (A' O)^t$ exists.
Put $r' = gh^top R' gh$. When $1 - beta' r' != 0$, the Sherman–Morrison formula gives the exact identities
$ (I - J^top)^(-1) gh = (R'^top gh) / ((1 - gamma)(1 - beta' r')), quad
  gh^top (I - J)^(-1) gh = r' / ((1 - gamma)(1 - beta' r')) . $
Now $A' O$ is a free loop with gain $s \/ (1 - gamma) < 1$ and readout gain $s_g \/ (1 - gamma)$. If $s_g > 0$, by
Corollary 4 ($T = infinity$), $R'^top gh = gh + ell'$ with $ip(gh, ell') -> 0$ and
$norm(ell')^2 -> (s_g \/ (1 - gamma))^2 \/ (1 - s^2 \/ (1 - gamma)^2) = F^2$. If $s_g = 0$, then
$ell' = sum_(t >= 1) (O^top A'^top)^t gh$ has $norm(ell') <= norm(A'^top gh) \/ (1 - norm(A')) -> 0 = F$ directly. So $r' -> 1$ and
$m := (1 - gamma)(1 - beta' r') -> 1 - lambda != 0$. The results of `analytic.pdf` apply because $I - K$ is
invertible: $K = U^top J U = gamma I + U^top A O U$ with $norm(U^top A O U) <= a_max < 1 - gamma$. Hence:

- By Lemma 1 of `analytic.pdf`, $1 \/ sigma = gh^top (I - J)^(-1) gh = r' \/ m -> 1 \/ (1 - lambda)$. So
  $sigma -> 1 - lambda$ and $psi = 1 - sigma -> lambda$.
- Exactly, $(I - J^top)^(-1) g = norm(g)(gh + ell') \/ m$ and $a = (I - J^top)^(-1) g - g$. So
  $ip(gh, a) \/ norm(g) -> lambda \/ (1 - lambda)$ and $norm(a) \/ norm(g) -> D \/ |1 - lambda|$. This yields $c_0$
  and $Gamma$.
- Theorem 1 of `analytic.pdf` gives $g + a = (norm(g) \/ sigma)(gh + U f)$. So
  $U f = (sigma \/ m)(gh + ell') - gh$ with $sigma \/ m -> 1$, and $norm(f) -> F$.
- $J^top gh = lambda gh + O^top A^top gh$, and $ip(gh, O^top A^top gh) -> 0$ by Theorem 4(b), so $alpha -> lambda$.
  The vector $O^top A^top gh$ is $(1 - gamma)$ times the first term of $ell'$, so by Theorem 4(a) (with the
  uniform tail bound) $ip(O^top A^top gh, ell') -> s_g^2 \/ (1 - gamma)$. With
  $E = ip(b, f) = ip((I - P) J^top gh, U f) = ip(O^top A^top gh, U f)$ ($U f perp gh$), this gives
  $E -> s_g^2 \/ (1 - gamma)$. Then $E + 2 C = psi - alpha -> 0$.
- Theorem 3 of `analytic.pdf` gives $c_1 = alpha c_0 + sgn(sigma) E \/ D_d$ with $D_d = sqrt(psi^2 + norm(f)^2) -> D$.
  $square$

*Resolvent lever versus persistent push.* Theorem 5 is about the resolvent at $1$. If also
$max(|gamma|, |lambda|) + a_max < 1$, then $norm(J) < 1$, the $T$-step levers converge to $a$, and $a$ is the response
to a persistent push. Otherwise convergence of the $T$-step levers is not guaranteed, and $a$ remains an algebraic
object (the resolvent at $1$).

*Consequences* (all within the model).
- *Start sign.* If $lambda < 1$ and $lambda != 0$, then with probability tending to one the lever starts against the
  readout ($c_0 < 0$) exactly when the self-gain is negative ($lambda < 0$).
- *Steering advantage.* The ratio of the best persistent push's effect to that of a push along $gh$, at equal norm,
  is $1 \/ |c_0| -> sqrt(1 + F^2 \/ lambda^2)$. It is large when the self-gain is small compared with the passage gain.
- *Circulation is generic.* With or without self-gain, $E + 2 C -> 0$: the circulation cancels the passage.
  Corollary 2b of `analytic.pdf` ("if $alpha + E > 0$, a negative start requires negative circulation") is then
  satisfied automatically, so circulation does not discriminate between readouts. The self-gain does.
- *$gamma$ versus $beta$.* For a single readout only $lambda = gamma + beta$ is visible (with $gamma$ also entering
  $F$). The uniform gain $gamma$ is a loop property, the same for every direction; $beta$ belongs to the readout.
  Random probe directions separate the two (@sec-test).

= Huginn: readouts against random directions <sec-huginn>

*Readouts* (existing per-item arrays, no new runs):
- *Truth*: 120 statements. Runs `dstar_vjp` (readout gradient $g$, 48-step lever $a_48$) and `truthcov`
  ($J^top g$), on the same items with identical $g$.
- *Digits*: 20 addition prompts × 10 centered digit-logit readouts. Runs `dglever_train` ($g$, $a_48$) and
  `adaptcov` ($J^top g$), joined by item id; the $g$ agree to relative $6 times 10^(-7)$.

*Random directions* (two new runs, @sec-test): 4 Gaussian unit probes in the own block at each of 40 truth contexts
(every third statement) and all 20 digit contexts, each pulled back for 48 steps at the same fixed point as the
readouts. A cross-run gate confirms the states are the same: the new runs reproduce the old readout gradients to
$1.3 times 10^(-5)$ and the old 48-step levers to $2.6 times 10^(-5)$ (relative).

$J$ is the own-block Jacobian at the answer position ($d = 5280$, fp32). Medians; all numbers are asserted by
`freeloop_check.py`.

#align(center, table(columns: 5, inset: 4pt, align: (left, center, center, center, center),
  [], [truth readout], [random, truth], [digit readouts], [random, digits],
  [], [$n = 120$], [$40 times 4$], [$n = 200$], [$20 times 4$],
  [$norm(J^top u)$], [0.71], [0.64], [0.82], [0.67],
  [$alpha$], [0.12], [0.24], [0.04], [0.26],
  [$c_0$], [0.09], [0.25], [0.03], [0.23],
  [$Gamma$], [1.24], [1.07], [1.45], [1.23],
  [$c_1$], [0.48], [0.47], [0.48], [0.46],
  [$kappa$], [0.68], [0.73], [0.59], [0.70],
  [$E$], [0.53], [0.34], [0.68], [0.39],
  [$2 C$], [$-0.55$], [$-0.38$], [$-0.66$], [$-0.43$],
  [$(E + 2C) \/ E$], [$-0.038$], [$-0.098$], [$0.009$], [$-0.088$],
))

*What readouts share with random directions.*
- *Cancellation.* $E + 2C$ is small relative to $E$ for readouts (truth $-3.8%$ with 80% of items in
  $[-4.4%, -2.7%]$; digits $+0.9%$, 80% in $[-5.5%, 6.8%]$) and for random directions ($-9.8%$, $-8.8%$).
  Equivalently $psi approx alpha$: correlation 0.90 (truth) and 0.87 (digits) across readouts. Since random directions
  share it, the cancellation is a property of the loop, not of the concept. For random directions Theorems 1–2 make
  it a function of the spectrum (@sec-test).
- *Start sign.* With $psi approx alpha$, the lever starts against the readout ($c_0 < 0$) essentially when $alpha < 0$:
  all 120 truth levers start positive with $alpha > 0$; 29% of digit levers start negative, and $sgn c_0 = sgn alpha$
  for 89.5% of digit readouts. The 59 negative-start digit readouts all satisfy the conditions of the regime proposed
  in `analytic.pdf` ($sigma_T > 0$, $E_T > 0$, $alpha + E_T > 0$, $c_1 > 0$, Corollary 3a); 53 of them have
  $alpha < 0$, against 15 of the 141 positive starts.
- *First step.* $c_1 approx 0.47$ for readouts and random directions alike.

*What sets readouts apart* is the self-gain, and through it the start angle and the steering advantage
(@sec-suppression).

#caveat[Draft 1 fitted the free model to the readout columns alone, with the loop gain inferred from $kappa$, and
reported agreement. The random-probe test below measures the loop directly, and the free model fails it, so those
fits were not evidence for it; they are withdrawn. The observations above do not depend on any model.]

= The random-probe test <sec-test>

*Design* (pre-registered in `huginn_addition/analyze_freeprobe.py` before the runs). At each context's fixed point,
the readout and $K = 4$ unit Gaussian probes are pulled back for $T = 48$ steps; the Gram matrix of all chain vectors
is saved. From the probes: $hat(gamma) = "mean" alpha(z)$ and $hat(s)^2 = "mean" norm(J^top z)^2 - hat(gamma)^2$
(Theorem 1: estimates of $tau(J)$ and $tau(J J^top) - tau(J)^2$). Predictions of the shifted free model
($J = gamma I + A O$; Theorem 5 with $beta = 0$), each probe predicted from the *other* three:
- *P1a* cancellation: median $|(E + 2C) \/ E| <= 0.10$;
- *P1b* lever: median $|log(Gamma \/ Gamma_"pred")| <= 0.10$, $|c_0 - c_(0,"pred")| <= 0.03$, $|kappa - kappa_"pred"| <= 0.05$;
- *P1c* chain norms $norm(w_t)$, $t = 2, 4, 8$, within $0.10$ in $|log|$;
- *P1d* ("exchangeability" in the registration; it measures probe agreement): within-context sd of $kappa$ across
  probes $<= 0.03$ (and of $log Gamma$ $<= 0.05$);
- *P1e* ("no shared direction"): median $|cos|$ between levers of different probes $<= 0.10$;
- *P2* the readout's $Gamma$ and $kappa$ from $(hat(gamma), hat(s))$ and its own $(alpha, s_g)$ (Theorem 5):
  median $|log|$ error $<= 0.10$ and $|kappa|$ error $<= 0.05$. Readouts: the 40 truth readouts; for digits the
  readouts of digits 0 and 5 at each context (40 in all; only these were pulled back in the new run);
- *Q3* the split of the readout's self-gain into a loop part $hat(gamma)$ and a readout part
  $hat(beta) = alpha - hat(gamma)$, with item-bootstrap CIs, read as "loop-wide", "readout-specific" or "mixed".
The verdict rule is a threshold-based adequacy check, not a significance test. Validity gates: unit starts, adjoint
check (finite-difference JVP against VJP), last chain term below $10^(-3)$ of the lever, and the cross-run identity
with the earlier runs. All passed in both runs.

*Calibration on synthetic loops* (`huginn_addition/calibrate_freeprobe.py`, CPU only; results in
`results/synth_calibration/calibration.json`). All loops have $d = 5280$ and 4 probes per context. A shifted free
loop with a readout gain (40 contexts, the truth design) passes every criterion: each P1 statistic is at least
2.9 times below its threshold, and each P2 error at least 7.6 times. The test fails, as designed, on a symmetric
loop (P1a–c) and on a weighted cyclic shift (P1b–c; its cancellation holds, so P1a alone would be fooled), with 10
contexts each. It can miss low-rank structure, as two examples show. A free loop plus a rank-one slow mode (a real outlier eigenvalue
near $0.9$; $T = 250$) passes P1 and P2. A readout placed in an exceptional $2 times 2$ block passes P1 and fails only
P2. A rank-one change moves the normalized traces of Theorem 1 by $O(1 \/ d)$ as long as $J$ and $M$ stay bounded
in norm, and random probes overlap its direction by only $O(1 \/ sqrt(d))$, so they can miss it (a mode very close
to 1, with large $norm(M)$, would still be visible); that is why readouts are measured directly.

#align(center, table(columns: 3, inset: 4pt, align: (left, center, center),
  [criterion (threshold)], [truth (40 contexts)], [digits (20 contexts)],
  [P1a $|(E + 2C) \/ E|$ ($0.10$)], [0.098 pass], [0.088 pass],
  [P1b $|log Gamma|$ ($0.10$)], [0.46 fail], [0.51 fail],
  [P1b $|c_0|$ ($0.03$), $|kappa|$ ($0.05$)], [0.057, 0.094 fail], [0.064, 0.127 fail],
  [P1c chain norms ($0.10$)], [0.45 fail], [0.74 fail],
  [P1d sd $kappa$ ($0.03$)], [0.019 pass], [0.016 pass],
  [P1e $|cos|$ ($0.10$)], [0.037 pass], [0.044 pass],
  [P2 readout $|log Gamma|$ ($0.10$)], [0.31 fail], [0.37 fail],
  [P1 verdict], [not supported], [not supported],
  [Q3 reading], [mixed], [mixed],
))

Excluding the two truth contexts whose state drift exceeds $10^(-3)$ changes no verdict. Q3 reads "mixed" in both
runs: the CI of the median $hat(beta)$ excludes $0$, but $hat(beta)$ is *negative*, a case the registered reading
rules did not anticipate (@sec-suppression).

*How freeness fails.* Fitted at step 1, the shifted free model gets step 2 right ($norm(w_2) = 0.457$ predicted,
$0.456$ observed, truth). After that it predicts that the uniform gain accumulates coherently: successive pullbacks
become nearly parallel ($cos angle(w_t, w_(t+1)) = 0.63$ at $t = 1$ and $0.94$ at $t = 10$) and decay slowly. In
Huginn they stay much less aligned (truth $0.30$ and $0.36$; digits $0.35$ and $0.69$) and decay faster
($norm(w_8) = 0.07$ against $0.11$ predicted for truth; $0.08$ against $0.16$ for digits). So the loop's mean
eigenvalue is not a coherent identity component, and the free model overstates the lever gain by about 60%.

*What holds: the loop signature.* Random directions agree closely within a context (P1d), and the levers of
different probes are nearly orthogonal (P1e; this does not exclude a shared slow mode, which P1e missed in the
calibration). The two very different contexts give similar broad scales, with context-dependent differences: random
$Gamma$ $1.07$ against $1.23$, lag cosine at $t = 20$ $0.54$ against $0.77$, and bootstrap CIs for the median
$hat(gamma)$ of $[0.240, 0.243]$ and $[0.254, 0.260]$.

#align(center, table(columns: 3, inset: 4pt, align: (left, center, center),
  [], [truth contexts], [digit contexts],
  [$m_1 = tau(J)$ (mean eigenvalue)], [0.242], [0.256],
  [$m_2, m_3, m_4$ ($m_k = tau(J^k)$)], [$0.037, -0.005, -0.008$], [$0.047, -0.003, -0.008$],
  [$kappa_2 = m_2 - m_1^2$], [$-0.021$], [$-0.019$],
  [$hat(s)$], [0.59], [0.62],
  [chain decay $norm(w_t) \/ norm(w_(t-1))$, $t = 2..10$], [0.71–0.73], [0.72–0.75],
  [$cos angle(w_t, w_(t+1))$ at $t = 1, 20$], [0.30, 0.54], [0.35, 0.77],
  [random lever gain $Gamma$, start $c_0$], [1.07, 0.25], [1.23, 0.23],
  [random steering advantage $1 \/ |c_0|$], [4.06], [4.33],
))

*Reading through Theorem 2.* The trace moments are spectral moments, and $kappa_k = integral (lambda - gamma)^k dif mu$
follows from $m_1, ..., m_k$. $kappa_2 < 0$ says the eigenvalues have a larger second moment along the imaginary axis
than along the real axis. Whether that fixes the sign of $E + 2C$ depends on how concentrated the spectrum is
(Theorem 2 needs the disc condition, and part (3) needs $r <= |1 - gamma| \/ 2$), which we have not verified: the
eigenvalues were not computed. Retrospectively the data agree with the expansion. The
second-order term $kappa_2 \/ (1 - gamma)$ gives 81% (truth) and 74% (digits) of the measured $E + 2C$ of random
directions. The measured $kappa_3$ is not negligible ($-0.003$ and $-0.005$), and the partial sums of
$(1 - gamma) S \/ (1 + S)$ through order $k = 3, ..., 8$ lie between $0.92$ and $1.08$ times the measured value in both contexts (medians
over contexts of per-context ratios).

= Readout and read-space suppression <sec-suppression>

The truth run showed every readout's self-gain below the probe estimate $hat(gamma)$ of its loop's mean eigenvalue.
This endpoint was not in the original registration. Before the digits production data were analysed, a replication
criterion was fixed in writing (`analyze_freeprobe.py`): use all 10 digit readouts per context from the existing
arrays (same states, by the cross-run gate) and $hat(gamma)$ from the probes; *replicated* if the median of
$alpha - hat(gamma)$ is negative with an item-cluster bootstrap 95% CI excluding 0.

#align(center, table(columns: 3, inset: 4pt, align: (left, center, center),
  [], [truth], [digits],
  [readouts], [40 (1 per context)], [200 (10 per context)],
  [$alpha < hat(gamma)$], [40 of 40], [200 of 200],
  [$alpha <$ every probe's $alpha(z)$], [40 of 40], [200 of 200],
  [median $alpha - hat(gamma)$ [95% CI]], [$-0.121$ [$-0.123$, $-0.117$]], [$-0.213$ [$-0.238$, $-0.198$]],
  [median $Z$ [95% CI]; largest $Z$], [$-15.3$ [$-15.4$, $-14.8$]; $-12.4$], [$-25.6$ [$-28.5$, $-24.0$]; $-7.9$],
  [rms sd of $alpha(z)$: Proposition R; observed], [0.0079; 0.0083], [0.0083; 0.0082],
  [advantage ratio $(1 \/ |c_0 (g)|) \/ (1 \/ |c_0 (z)|)$ [CI]], [2.75 [2.71, 2.82]], [4.35 [3.71, 5.90]],
))

$Z$ is the standardized excess of Proposition R, with the three traces estimated from the probes of the same context:
a descriptive plug-in scale, not a Gaussian significance score (4 probes estimate the traces, and Chebyshev's bound
applies only to the exact $Z$).
The "sd" row checks Proposition R on the probes themselves: the predicted sd of $alpha(z)$ against the
within-context sd of the 4 probes' $alpha(z)$, both as root mean squares over contexts (observed\/predicted $1.04$ and $0.99$). The CIs resample contexts, with the readouts
of a context kept together; they are intervals for the probe-estimated statistic, not for the exact $tau(J)$. The
probe mean $hat(gamma)$ has a standard error of about $0.004$, small against the smallest deficits ($0.095$ and
$0.066$).

*Replicated.* The loop feeds a perturbation along a readout back into that readout less than it does for a typical
direction. Since $1 \/ |c_0| = Gamma \/ |x|$ exactly, for each readout–probe pair the advantage ratio is
$(|x(z)| \/ |x(g)|) (Gamma(g) \/ Gamma(z))$. Descriptively, with $x(z)$ and $Gamma(z)$ averaged over the probes of a
context, the truth medians of the two factors are $2.32$ and $1.16$ (medians do not multiply exactly; $2.32 times 1.16
= 2.69$ against $2.75$). Most of the extra advantage comes from the smaller return $x approx alpha \/ (1 - alpha)$,
that is, from the self-gain deficit, and a smaller part from a longer lever. For digits many readouts have $alpha$ near or below zero (the negative starts of `analytic.pdf` are the extreme
end), where the advantage is very large.

*The control: the readout, or everything the coda reads?* (Pre-registered; `ctrlprobe_app.py`,
`analyze_ctrlprobe.py`; 2 runs, about 5 GPU-minutes, code reviewed before running.) The same contexts and states: the
4 old probes reproduce their self-gains to $5 times 10^(-8)$. A readout is the gradient at the answer position of a
logit contrast, $g = nabla(ell_a - ell_b)$ (or of a centred digit logit), where $ell$ is the vector of
log-probabilities the coda outputs. Every such gradient lies in the coda's *read space*. Three control families, 8
directions each per context: *C1* random-token contrasts $nabla(ell_a - ell_b)$ with $a, b$ uniform over the
vocabulary; *C2* contrasts among the 64 highest-logit tokens; *C3* random combinations $nabla(xi dot ell)$, $xi$ a
centred Gaussian vector, which is a direction the coda reads, weighted by how strongly it reads it. C1 and C2 exclude
the task tokens; C3 uses task-independent random coefficients over all non-special tokens, task tokens included. 16 probes per context (the 4 earlier ones and 12 new) estimate $tau(J)$. For a family $F$, $Delta_F$ is the
per-context median excess, $rho_F = "median" Delta_F \/ "median" Delta_"task"$ the share of the task deficit that the
controls have. The readings were fixed in advance and checked on synthetic loops with known answers: *shared* if
the CI of $Delta_F$ is below 0 and the CI of $rho_F$ above $0.75$; *task-specific* if the paired task-minus-control
CI is below 0 and the CI of $rho_F$ below $0.25$; *partial* if the control and paired CIs are both below 0 and the CI
of $rho_F$ lies inside $(0.25, 0.75)$; otherwise *unclear* (all after a check that the task deficit's CI is below 0).
The primary family is C3.

#align(center, table(columns: 3, inset: 4pt, align: (left, center, center),
  [], [truth (40 contexts)], [digits (20 contexts)],
  [$hat(tau)$ (16 probes) [95% CI]], [$0.241$ [$0.240$, $0.242$]], [$0.258$ [$0.256$, $0.259$]],
  [median excess: task readouts], [$-0.119$], [$-0.213$],
  [#h(1em) C1 random-token contrasts], [$-0.236$], [$-0.204$],
  [#h(1em) C2 plausible-token contrasts], [$-0.181$], [$-0.234$],
  [#h(1em) C3 random logit combinations [CI]], [$-0.224$ [$-0.231$, $-0.216$]], [$-0.200$ [$-0.210$, $-0.192$]],
  [share $rho_"C3"$ [CI]], [$1.88$ [$1.80$, $1.95$]], [$0.94$ [$0.84$, $1.03$]],
  [task minus C3, paired [CI]], [$+0.102$ [$0.098$, $0.115$]], [$-0.013$ [$-0.030$, $-0.005$]],
  [median $Z$: task; C3], [$-15.1$; $-28.0$], [$-25.3$; $-24.3$],
  [below all 16 probes: task; controls], [40/40; 960/960], [200/200; 480/480],
  [reading C1, C2, C3], [shared, shared, shared], [shared, shared, shared],
))

*Shared.* The deficit is not a property of the task readouts alone: all three sampled families of read directions
have it, in both contexts, and every control direction lies below every probe of its context. This does not establish
suppression throughout the complete read space: C3 samples directions in proportion to how strongly the coda reads
them, so weakly read directions can be missed. In the truth contexts the
True/False readout is *less* suppressed than a typical read direction ($rho approx 1.9$, paired difference $+0.10$);
the digit readouts are typical ($rho approx 0.94$). The readings do not change without the two truth contexts whose
drift exceeds $10^(-3)$, and projecting the task readouts' span out of each control changes $rho_"C3"$ by $0.002$
(truth) and $0.042$ (digits). The table's $Z$ and $tau$ use 16 probes, so they differ slightly from the 4-probe values above; Proposition R
again predicts the probe spread ($0.0079$ against $0.0081$; $0.0083$ against $0.0083$).

*Similar pullback norm, low self-alignment* (exploratory, defined after the control's smoke runs had been seen).
From the same runs, medians over contexts and directions; "read span" is the span of the 25 (truth) or 34 (digits)
sampled read directions of a context (numerical rank 24–25 and 32–33: the centred digit gradients are dependent).

#align(center, table(columns: 3, inset: 4pt, align: (left, center, center),
  [random probes; C3], [truth], [digits],
  [one-step size $norm(J^top u)$], [$0.64$; $0.62$], [$0.67$; $0.64$],
  [self-cosine $alpha(u) \/ norm(J^top u)$], [$0.38$; $0.03$], [$0.38$; $0.09$],
  [$ip(u, (J^top)^2 u)$], [$0.037$; $-0.005$], [$0.046$; $0.002$],
  [share of $norm(J^top u)^2$ in the read span, all sampled read directions (isotropic benchmark rank$\/d$)], [$0.038$ ($0.0047$)], [$0.042$ ($0.0063$)],
  [trace / dim of $J$ compressed to the read span (probe span)], [$0.061$ ($0.241$)], [$0.067$ ($0.257$)],
))

Pullbacks of read directions are not smaller than those of random directions: both contract by about a third, by
the same amount. They point elsewhere: the self-cosine is near zero, about 4% of the pulled-back squared norm stays in
the sampled read span (the isotropic benchmark is below 1%), and the signed two-step self-overlap is near zero. These
are adjoint (pullback) sensitivities within the own-position block, not measurements of how activations evolve, and
they do not show that the read component is rebuilt at each step. They are consistent with three findings of the
write-up: the readout reversal happens at the first loop step; 97% of the one-step sensitivity into the digit-gradient
span comes from outside the span; the loop-aware levers lie mostly outside it.

*A weights-only proxy* (CPU). Huginn's step is $s |-> "core"(W_s s + W_e e)$. On the tested digit direct-path
pullbacks the core's residual and norm chain acts nearly as a scalar (about $2.76$, write-up), so the adapter block
$W_s$ could carry the deficit. The proxy does not reproduce it: $tau(W_s) = 0.110$, and the median of $ip(u, W_s u)$
is $0.101$ over unembedding directions $u prop w_f dot.o E_v$ ($w_f$ the final norm weight, $E_v$ an unembedding row)
and $0.102$ over random combinations of them, close to the trace. For the actual readout gradients it is lower
($0.063$ truth, $0.049$ digits). So this proxy does not reproduce the observed geometry; it does not determine
whether the adapter, the coda or the state dependence of the core is responsible.

*A candidate mechanism: an exploratory chain test* (`analyze_chains.py`, existing arrays only). Huginn is trained
from a random start state with a random number of loops (mean 32). Suppose its output were exactly invariant to the
start state after $m$ loops. Differentiating at the fixed point gives $g^top J^m = 0$ for every readout gradient $g$.
The span $V$ of the vectors $(J^top)^j g$, $j < m$, is then invariant under $J^top$, which is nilpotent on it, so the
mean of $alpha$ over any orthonormal basis of $V$ is $tr(J^top|_V) \/ dim V = 0$. This basis average does not by itself
predict small self-gains for the sampled read directions, which are not uniformly spread in $V$: a nilpotent map can
have a large self-gain on a single vector ($A = mat(1, 1; -1, -1)$ has $A^2 = 0$ and $e_1^top A e_1 = 1$); further
assumptions would be needed. Empirically the C3 self-gains are small ($0.018$ and $0.059$), and the cosine between
$J^top u$ and $(J^top)^2 u$ is $0.015$ and $0.095$ for C3, against $0.30$ and $0.35$ for probes. Exact invariance
requires zero readout sensitivity from step $m$ on, but implies no ordering relative to probes before that step. We
therefore test the weaker hypothesis that readouts are attenuated preferentially. In the 48-step chains of
@sec-test, with $hat(g) = g \/ norm(g)$, the median ratio of $norm((J^top)^t hat(g))$ to the context's median probe
chain exceeds 1 at every $1 <= t <= 32$, ranging over $1.11$–$1.33$ (truth) and $1.22$–$1.64$ (digits). Over steps
1–8, 9 of 320 (2.8%, truth) and 3 of 320 (0.9%, digits) readout chain norms fall below the probe median. Strong
attenuation is also present in generic directions: the probe chains are down to $5 times 10^(-5)$ (truth) and
$1.4 times 10^(-4)$ (digits) at $t = 32$. These measurements do not support preferential attenuation of the sampled
task readouts through step 32. They do not rule out approximate start invariance, and they give no training
explanation for the self-gain deficit. Trajectories from random starts pass through states whose Jacobians differ
from the fixed point's; this test does not cover them.

#caveat[What this does *not* show. It is a statement about the linearized loop at the fixed point, at answer
positions. The families are typical sensitivity-weighted contrasts; no reading is a statement about *every* direction
the coda can read, and the self-alignment statistics refer to the sampled read span and the own-position block. It does
not say why Huginn's loop treats its read space this way (training or architecture). The advantage ratio is a
linear-response quantity; the behavioural comparison in the write-up ($d^*$ against $g$: $8.1 \/ 0.93 = 8.7$,
population pushes) is a different estimand.]

= A second looped model: Ouro-1.4B <sec-ouro>

Ouro-1.4B (ByteDance, Apache-2.0) loops its whole 24-layer stack 4 times with shared weights:
$h_(k+1) = F(h_k) = N(L(h_k))$, $L$ the 24 layers (pre-norm and sandwich norm, residual), $N$ the final RMSNorm,
$h_0$ the token embeddings, no input re-injection, and the logits after loop $k$ are $E h_k$ (linear). So its read
directions are exact unembedding contrasts. There is no fixed point: the state still moves in the last loop
(relative change $0.17$). It answers all 40 truth prompts correctly, with 99% of the probability on True/False.

*Design* (pre-registered, `ouro_app.py`, `analyze_ouro.py`, code reviewed; about 8 GPU-minutes). The own-position
Jacobian of the last loop ($J$ at $h_3$, the map that produces the read state; secondary: the loop before), the same
families and probes as for Huginn, and a coordinate control *C3flip*: the C3 directions with random coordinate signs
$S$. This keeps each coordinate's energy and, in expectation over the signs, removes the off-diagonal part of the
self-gain: $EE_S ip(S u, J^top S u) = sum_i J_(i i) u_i^2$. Primary reading on C3: *strong deficit* if the CI of
$Delta_"C3"$ is below 0, $-Delta_"C3" \/ tau >= 0.5$ and at least 90% of C3 directions lie below all probes (when the
median $tau <= 0.05$, a median $Z <= -10$ replaces the relative threshold; the loop before uses this branch); *weak
deficit* for other detected deficits; *not detected* if the CI of $Delta_"C3"$ is not below 0 (this is not evidence of
absence without an equivalence margin). Coordinate control: *coordinate-wise* if the C3flip deficit is at least 75%
of the C3 deficit, *alignment* if at most 25%. Gates: four manual loops reproduce the model's own states
exactly and its logits to $3 times 10^(-5)$; the adjoint check and the rest as before. All passed.

*Result: not detected, at both loops.* $hat(tau) = 0.060$ [$0.057$, $0.061$]. The C3 excess is *positive*, $+0.022$
[$0.016$, $0.025$] (median $Z = +1.1$; 4% of C3 directions below all probes), and the task readout's is $+0.097$
($Z = +5.1$). The sign-flipped control has nearly the same excess ($+0.020$, ratio $0.92$; exploratory paired C3
minus C3flip difference $-0.0004$, CI $[-0.006, 0.008]$), consistent with a substantial coordinate-energy
contribution. The registered "coordinate-wise" label was designed to attribute a detected deficit; applied to this
positive excess it is descriptive and does not exclude alignment effects. At the loop before: $hat(tau) = 0.049$, C3
excess $+0.024$. The self-cosine is low for probes and sampled read directions alike ($0.07$ and $0.08$), against
$0.38$ for Huginn's probes.



= A metric for loops and readouts <sec-metric>

*Loop signature.* For a loop (a model at a context), pull back $K$ random unit directions for $T$ steps and record the
trace moments $m_k = tau(J^k)$, the chain norms and successive-pullback cosines, and the random-direction lever
statistics ($Gamma$, $c_0$, steering advantage, $E + 2C$). By Theorem 1 these are trace functionals, so a few probes
suffice: within a context the probes agree to $0.02$ in $kappa$ (P1d). Cost: $K T$ VJPs per context. Theorems 2–3
give two of them a meaning. The unnormalized $E + 2C$ is a function of the spectrum (Theorem 2); the ratio
$(E + 2C) \/ E$ is not, because $E$ also depends on non-normality. Theorem 3 bounds the random start $c_0$ by a
spectral quantity attained for normal $J$, but evaluating that bound needs $"mean"_i |nu_i|^2$, i.e. the eigenvalues,
not just the trace moments.

*Readout excess.* For a readout $g$ of the same loop, record $Delta alpha = alpha(g) - m_1$, its standardized form
$Z(g)$ (Proposition R), and the advantage ratio $(1 \/ |c_0(g)|) \/ (1 \/ |c_0(z)|)$. These measure how a concept sits
in the loop relative to a typical direction. $Delta alpha$ is in eigenvalue units and comparable descriptively across
models once the loop map, state coordinates, operating point and read-direction distribution are specified;
$Z$ grows like $sqrt(d)$ for a fixed $Delta alpha$, so it says how atypical the readout is within one model.

*Read-space excess.* The same, with $u$ a random direction the output reads (C3 of @sec-suppression: the gradient of
a random centred combination of the log-probabilities), against a control with the same coordinate energies (C3flip
of @sec-ouro). It asks whether the loop treats what the model outputs differently from the rest of its state, and
it costs one VJP per direction. The readout excess should be read against it: in Huginn the readouts' deficit is
shared by the sampled read directions.

*Comparing models.* Two loops are compared through their signatures, and their concepts through their excesses. A
signature difference says "the loop is different". An excess difference at equal signatures says "the concepts sit
differently in the loop". The two looped models, on the same 40 truth prompts (Huginn also at 20 digit contexts):

#align(center, table(columns: (auto, auto, auto, auto), inset: 4pt, align: (left, center, center, center),
  [], [Huginn, truth], [Huginn, digits], [Ouro, truth (last loop)],
  [mean eigenvalue $hat(tau)$], [$0.241$], [$0.258$], [$0.060$],
  [random probes: self-cosine], [$0.38$], [$0.38$], [$0.07$],
  [read space (C3): median self-gain], [$0.018$], [$0.059$], [$0.081$],
  [read-space excess $Delta_"C3"$], [$-0.224$], [$-0.200$], [$+0.022$],
  [#h(1em) 95% CI], [[$-0.231$, $-0.216$]], [[$-0.210$, $-0.192$]], [[$0.016$, $0.025$]],
  [median $Z$ (C3)], [$-28.0$], [$-24.3$], [$+1.1$],
  [C3 below all 16 probes], [100%], [100%], [4%],
  [task readout excess], [$-0.119$], [$-0.213$], [$+0.097$],
))

The measured C3 excesses have opposite signs under these protocols. The comparison is between Huginn's recurrent core
near a fixed point, with read directions from a nonlinear coda, and Ouro's last whole-stack loop, with state drift
$0.17$ and linear unembedding directions. It establishes a protocol-dependent geometric difference, not a difference
in dynamical kind; the coordinate-matched control was measured only in Ouro. The random-probe self-cosine is $0.38$
in Huginn and $0.07$ in Ouro. Two models cannot say why. Plausible differences, untested: Huginn is trained to reach
a fixed point from a random state and reads only the final state through two more layers; Ouro reads the state after
every loop (an exit gate and a per-step loss), runs 4 loops and has no input re-injection. Within Huginn the
signature has similar broad scales at two very different kinds of position, with measurable differences; whether it
characterizes the model rather than the context needs more contexts, with an equivalence margin fixed in advance.

*Limits of the metric.* The signature is descriptive, not a sufficient statistic: the free model, which would make a
few numbers sufficient, fails (P1b, P1c, P2). Agreement between random probes cannot certify a particular
readout, and they can miss low-rank structure: in the calibration of @sec-test a slow mode, and a readout in an
exceptional subspace, pass the probe criteria. That is why the readout excess is measured on the readout itself.

= Status and limits <sec-status>

- *Mathematics.* Theorems 1–3 and Proposition R are elementary: concentration on the sphere, trace equals the sum of
  eigenvalues, a power series (mean-value property) and Jensen's inequality, Schur's inequality, and the fourth
  moments of the sphere. Theorems 4–5 use the classical
  Haar conditioning device and the Sherman–Morrison formula; Theorem 4 is a vector-state analogue of the R-diagonal
  moment identity. No literature search beyond the references was done. The likely contribution is the application:
  which parts of a looped model's steering geometry are spectral, which are generic, and which belong to the readout.
- *Related work.* Two recent papers bear directly on the read-space results; neither was known when these runs were
  designed.
  - Wang and Reid (arXiv 2609.01924) apply a Jacobian lens to Ouro-2.6B and Huginn. They report that Huginn carries
    content forward across loops while Ouro rebuilds it in every loop, and that the verbalisability of injected
    content tracks per-iteration supervision. That is the same training factor as our proposed training experiment
    (readout after every loop or after the last only), and it may bear on the Huginn/Ouro contrast in
    @sec-ouro.
  - Sharma and Vu (arXiv 2606.24898) describe a "readout blind spot": a normalized readout cannot see the scale of
    the state. Any read-space quantity that is computed through the final norm inherits that blind direction.
- *Experiments.* Five pre-registered production runs (about 21 GPU-minutes, plus smoke runs; code reviewed before
  running, validity gates passed), existing readout arrays and one CPU weight extraction.
  - The free null model fails the pre-registered criteria in both Huginn contexts.
  - The loop signature has similar broad scales in both, with measurable differences.
  - Readout suppression holds for 240 of 240 readouts (each below all 4 probes of its context; $Z <= -7.9$), with a
    replication criterion fixed before the digits data were analysed.
  - The control: the deficit extends to all three sampled families of non-task read directions (1,440 of 1,440 below
    all probes); the truth readout is less suppressed than a typical sampled read direction.
  - Ouro-1.4B: no negative read-space excess detected (a small positive one, similar for the sign-flipped control) and
    a much smaller mean eigenvalue.
  - Exploratory: normalized task-readout chains are larger than their context's median probe chain, in aggregate,
    at every step 1–32 of the fixed-point Jacobian. This does not support preferential readout attenuation;
    approximate start invariance and trajectories away from the fixed point remain untested.
- *What it does not do.*
  - Two models (Huginn-0125, Ouro-1.4B), answer positions only; Ouro on one prompt family (40 contexts), Huginn on
    two (60).
  - The eigenvalues are not computed; the spectral reading uses trace moments.
  - Random probes can miss low-rank structure (a slow mode, an exceptional block, in the calibration), so the
    loop signature is a bulk description.
  - No causal or training account of Huginn's read-space deficit, and two models cannot separate architecture from
    training.
  - Linear response only.
- *Next steps.*
  - The signature and read-space excess on more looped and recurrent-depth models (Ouro-2.6B, other fixed-point
    models) and across training checkpoints: does the read-space deficit go with fixed-point training, with reading
    only the final state, or with input re-injection?
  - Huginn at non-answer positions: is the read-space deficit a property of the model or of answer positions?
  - Direct spectral estimates, to test the conditions of Theorem 2 and the $kappa_2$ reading.
  - A matched training experiment: the same small looped architecture trained with a readout after the last loop
    only or after every loop, with and without input re-injection, over several seeds, with C3, C3flip and drift
    measured at every loop. A reproducible change with the training regime would turn the two-model contrast into
    evidence for a mechanism.
  - Whether the low self-alignment stabilizes decisions: a read direction that fed back on itself strongly would be
    amplified on every loop.
- *Consequences for `analytic.pdf`.* Its theorems stand. The circulation–passage cancellation it describes is shared
  by random directions and, for them, is a function of the spectrum. The start sign usually follows the self-gain
  (89.5% of digit readouts). In Huginn, readouts share with all tested control directions a self-gain far below the
  probe-estimated mean eigenvalue of their loop.

*References.*
M. Bayati, A. Montanari, The dynamics of message passing on dense graphs, _IEEE Trans. Inf. Theory_ 57 (2011).
E. Bolthausen, An iterative construction of solutions of the TAP equations for the Sherrington–Kirkpatrick model,
_Comm. Math. Phys._ 325 (2014).
A. Guionnet, M. Krishnapur, O. Zeitouni, The single ring theorem, _Ann. of Math._ 174 (2011).
A. Guionnet, O. Zeitouni, Support convergence in the single ring theorem, _Probab. Theory Related Fields_ 154 (2012).
U. Haagerup, F. Larsen, Brown's spectral distribution measure for R-diagonal elements, _J. Funct. Anal._ 176 (2000).
R. A. Horn, C. R. Johnson, _Matrix Analysis_ (Schur's inequality), Cambridge University Press.
M. F. Hutchinson, A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines,
_Comm. Statist. Simulation Comput._ 19 (1990).
A. Nica, R. Speicher, R-diagonal pairs: a common approach to Haar unitaries and circular elements, _Fields Inst.
Commun._ 12 (1997).
S. Rangan, P. Schniter, A. K. Fletcher, Vector approximate message passing, _IEEE Trans. Inf. Theory_ 65 (2019).
Sharma and Vu, readout blind spot of normalized readouts, arXiv 2606.24898 (2026).
Wang and Reid, a Jacobian lens on looped language models (Ouro-2.6B, Huginn), arXiv 2609.01924 (2026).
