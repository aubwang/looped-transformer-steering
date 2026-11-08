#set document(title: "Transformers as dynamical systems: a primer for the Huginn study")
#set page(paper: "us-letter", margin: (x: 1in, y: 0.9in), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => [#v(0.6em) #it #v(0.2em)]

#let boxed(title, color, body) = block(
  fill: color, inset: 9pt, radius: 3pt, width: 100%, breakable: true,
  [*#title.* #body],
)
#let defn(title, body) = boxed("Definition (" + title + ")", rgb("#eef3fb"), body)
#let fact(body) = boxed("Fact", rgb("#eef3fb"), body)
#let here(body) = boxed("In this project", rgb("#eef8ee"), body)
#let heur(body) = boxed("Heuristic, not a theorem", rgb("#fbf3e6"), body)

#align(center)[
  #text(17pt, weight: "bold")[Transformers as dynamical systems]
  #v(0.1em)
  #text(12pt)[A primer for the Huginn study, for a reader with math but no ML]
  #v(0.2em)
  #text(10pt)[Read before `derivations.pdf`. Draft, 2026-11-07.]
]

#v(0.6em)

*Plan.* Section 1 refreshes the linear algebra the derivations use, phrased as operator theory on $bb(R)^d$. Section 2
describes a standard transformer as a composition of maps. Section 3 ties the weights together to get a looped
("recurrent-depth") transformer, which is an honest discrete-time dynamical system; Huginn is one. Section 4 adds a
measure-theoretic view of the loop. Section 5 defines the ML vocabulary used in the write-up. Section 6 says how each
derivation in `derivations.pdf` sits in this picture.

Throughout, vectors are columns, $x^top y$ is the Euclidean inner product, and $norm(x) = sqrt(x^top x)$.

= Linear algebra, as operator theory on $bb(R)^d$

Everything here is finite-dimensional ($d = 5280$ for Huginn's state), so every operator is a matrix, every linear
functional is continuous, and all norms are equivalent. Most of what you remember from functional analysis applies
directly, and several facts get simpler.

*Inner products and angles.* $cos(x, y) = x^top y \/ (norm(x) norm(y))$. If $x$ and $y$ are centered data vectors this
is their sample correlation. "Nearly orthogonal" (cosine $approx 0.1$) means the two directions have almost nothing in
common.

#defn("adjoint")[For $A : bb(R)^d -> bb(R)^d$, the adjoint $A^top$ is the unique operator with
$(A x)^top y = x^top (A^top y)$ for all $x, y$. In coordinates it is the transpose. This identity is the single most
used move in the derivations: it moves an operator from the vector onto the functional.]

*Vectors versus covectors.* A linear functional $ell(x) = g^top x$ is represented by a vector $g$ (Riesz, trivially in
finite dimensions). Vectors are *pushed forward* by $A$: $x |-> A x$. Functionals are *pulled back*:
$ell compose A$ is represented by $A^top g$. Keep these two roles apart. Probes and pushes are vectors; readout
gradients are covectors. The central result of the study is that, for this system, the two kinds of object transform
very differently.

#defn("spectrum, spectral radius")[$sigma(A)$ is the set of eigenvalues ($A v = lambda v$, $v != 0$, $lambda in
bb(C)$). The spectral radius is $rho(A) = max_(lambda in sigma(A)) |lambda|$. Gelfand's formula holds:
$rho(A) = lim_(t -> infinity) norm(A^t)^(1 \/ t)$. So $A^t -> 0$ if and only if $rho(A) < 1$, and the decay is
eventually at rate $rho(A)^t$.]

#defn("Neumann series, resolvent")[If $rho(A) < 1$, then $sum_(t >= 0) A^t$ converges and equals $(I - A)^(-1)$.
More generally the resolvent $R(z) = (z I - A)^(-1)$ exists for $z in.not sigma(A)$, and
$R(z) = sum_(t >= 0) A^t \/ z^(t+1)$ for $|z| > rho(A)$.]

#defn("normal operator")[$A$ is normal if $A A^top = A^top A$ (for real matrices; symmetric matrices are the main
example). By the spectral theorem a normal $A$ has an orthonormal eigenbasis, so $norm(A^t) = rho(A)^t$ exactly: no
transient growth. On each eigendirection $A^top$ acts by the conjugate eigenvalue, so $A$ and $A^top$ agree only
for real eigenvalues; a rotation is normal but $A^top$ turns the other way.]

*Non-normal operators.* For non-normal $A$ the eigenvectors can be nearly parallel. Then:
- $norm(A^t)$ can grow for a while before it decays, although $rho(A) < 1$. One bound is
  $norm(A^t) <= kappa(V) rho(A)^t$, where $kappa(V) = norm(V) norm(V^(-1))$ is the condition number of the
  eigenvector matrix.
- The eigenvalues stop describing the operator well. The resolvent norm $norm((z I - A)^(-1))$ can be huge far from
  $sigma(A)$. The sets where it exceeds $1 \/ epsilon$ are the $epsilon$-pseudospectra (Trefethen–Embree), the
  standard tool for this situation.
- $A$ and $A^top$ have the same eigenvalues but can act very differently on a given vector. This is the mechanism
  behind the gap between decoding and control (Section 6).

#here[The measured eigenvector condition numbers had median 75, and a push could grow by a factor 1.17 in one step
while $rho approx 0.78$. The loop's linearization is strongly non-normal.]

#defn("derivative")[For $F : bb(R)^m -> bb(R)^m$, the (Fréchet) derivative at $x$ is the linear map $J = D F(x)$
with $F(x + h) = F(x) + J h + o(norm(h))$; its matrix is the Jacobian $partial F_i \/ partial x_j$. For a scalar
function $R$, $D R(x)$ is a linear functional, represented by the gradient $g = nabla R(x)$:
$R(x + h) = R(x) + g^top h + o(norm(h))$.]

#fact[*Spectral mapping (polynomial case).* If $p$ is a polynomial, $sigma(p(A)) = p(sigma(A))$, with the same
eigenvectors. *Triangular matrices.* For a block lower-triangular matrix, $sigma$ is the union of the spectra of the
diagonal blocks.]

= A standard transformer, as a composition of maps

*The statistical object.* Text is cut into *tokens*, integers in $\{1, dots, V\}$ ($V = 65536$ for Huginn). A
language model is a family of conditional distributions $p(x_(n+1) mid(|) x_1, dots, x_n)$, so it defines a
distribution on sequences by the chain rule. Training maximizes the log-likelihood of a large text corpus. ML calls
the negative log-likelihood "cross-entropy loss".

*Embedding.* Each token $k$ has a learned vector $e_k in bb(R)^d$. A sequence of $n$ tokens becomes a matrix
$H in bb(R)^(n times d)$, one row per *position*.

*Readout.* At the end, the last row $h_n$ is mapped to *logits* $ell = W_U h_n in bb(R)^V$, and
$ p(x_(n+1) = k mid(|) x_(1:n)) = op("softmax")(ell)_k = e^(ell_k) / (sum_j e^(ell_j)). $
Logits are log-probabilities up to an additive constant, so $ell_a - ell_b$ is the log-odds between tokens $a$ and $b$.
That is why the truth readout is $log p("True") - log p("False")$.

*Layers.* A transformer applies $L$ layers, each a *residual update*:
$ H^((l+1)) = H^((l)) + f_l (H^((l))), quad l = 0, dots, L - 1. $
The running matrix $H^((l))$ is called the *residual stream*. Each $f_l$ is built from three pieces.

+ *Normalization.* $N(x) = gamma dot.o x \/ "rms"(x)$ with $"rms"(x) = norm(x) \/ sqrt(d)$ and a learned gain
  vector $gamma$ (RMSNorm). Up to the gain, it radially projects $x$ onto the sphere of radius $sqrt(d)$.
+ *MLP*, applied to each row separately. Huginn's is gated:
  $x |-> W_2 (op("SiLU")(W_1 x) dot.o W_3 x)$, with hidden width 17920 and $op("SiLU")(u) = u \/ (1 + e^(-u))$.
  It is a nonlinear map $bb(R)^d -> bb(R)^d$, the same at every position.
+ *Attention*, the only piece that mixes positions. Row $i$ forms a query $q_i = W_Q x_i$, and each row $j$ forms a
  key $k_j = W_K x_j$ and a value $v_j = W_V x_j$. Then
  $ a_(i j) = (exp(q_i^top k_j \/ sqrt(d_h))) / (sum_(j' <= i) exp(q_i^top k_(j') \/ sqrt(d_h))) "for" j <= i,
    quad a_(i j) = 0 "for" j > i, quad quad "out"_i = W_O sum_(j <= i) a_(i j) v_j. $
  For a probabilist: $(a_(i j))$ is a *lower-triangular row-stochastic matrix*, a data-dependent Markov kernel from
  position $i$ to earlier positions. Each position outputs an average of earlier positions' values under that kernel.
  Huginn runs 55 such kernels ("heads") in parallel, each on a 96-dimensional slice, and concatenates them. Position
  information enters by rotating $q_i$ and $k_j$ by angles proportional to $i$ and $j$ (rotary embeddings).

*Causality.* Because $a_(i j) = 0$ for $j > i$, row $i$ of every layer's output depends only on rows $1, dots, i$ of its
input. This lower-triangular dependence carries through the whole network. It is what lets the model be trained on all
prefixes at once, and it is what makes the own-position Jacobian exact in D1.

*The dynamical reading.* $H^((l+1)) = H^((l)) + f_l (H^((l)))$ is a non-autonomous discrete-time system: "time" is
depth, the horizon is finite ($L$ steps), and the map changes at every step. It is also a forward-Euler step of an ODE
$dot(H) = f(H, l)$, which is the "neural ODE" view. Because each step has its own map, there are no fixed points or
long-run behavior to talk about. That changes in the next section.

= Looped transformers: an autonomous dynamical system

*Weight tying.* The Universal Transformer (Dehghani et al., 2018) uses the same layer at every step: $f_l = f$. Then
$H^((l+1)) = Phi(H^((l)))$ with one fixed map $Phi$. This is an autonomous system; you can run it for as many steps as
you like, and fixed points, cycles, attractors and stability all make sense.

*Huginn-0125* (Geiping et al., 2025; 3.5B parameters) splits the network into three parts:

#align(center)[#text(9.5pt)[
  #box(stroke: 0.6pt, inset: 5pt)[tokens] $->$
  #box(stroke: 0.6pt, inset: 5pt)[prelude $P$ (2 layers)] $-> E ->$
  #box(stroke: 0.6pt, inset: 5pt)[core $R$ (4 layers) $times r$ loops] $->$
  #box(stroke: 0.6pt, inset: 5pt)[coda $C$ (2 layers)] $->$ logits
]]

- The prelude computes $E = P(x_(1:n)) in bb(R)^(n times d)$ once. Call $E$ the *context*.
- The state starts at a random $s_0 in bb(R)^(n times d)$, drawn independently of the prompt.
- Each loop step is
  $ s_(t+1) = Phi_E (s_t) = R(W_s s_t + W_e E). $
  The linear map $[s; E] |-> W_s s + W_e E$ is the *adapter*; it re-injects the context at every step.
- After $r$ steps, the coda maps $s_r$ to logits.

The core's four layers are "sandwich" layers (normalization before and after each sub-block), and the core output is
normalized at the end. So every $s_t$ with $t >= 1$ lies, row by row, in the compact set
$K = {gamma dot.o u : norm(u) = sqrt(d)}$, a sphere stretched by the gains.

*Training.* For each training sequence, $r$ is drawn at random (a Poisson-lognormal distribution with mean 32), and
gradients flow through the last 8 loop steps. The loss is the next-token negative log-likelihood of
$C(Phi_E^r (s_0))$, averaged over data, over $r$, and over $s_0$.

*The object we studied.* For a fixed prompt, $Phi_E : K^n -> K^n$ is a continuous self-map of a compact set, with the
prompt entering only through the parameter $E$. Causality makes it *triangular*: row $i$ of $Phi_E (s)$ depends on
rows $1, dots, i$ of $s$. So position 1 evolves on its own, position 2 is driven by position 1, and so on. In
dynamics language, this is a tower of skew products. Our questions were the standard ones:
- Where does an orbit go? Fixed point, periodic orbit, something else?
- How stable is the limit? (Spectrum of the derivative.)
- How does the limit move when the parameter or the state is perturbed? (Linear response.)
- What does the readout $C$ see along the way?

#heur[*What training asks of the dynamics.* The loss averages over random $s_0$ and random $r$. A model that does well
must make the readout (a) forget $s_0$ and (b) stop depending on $r$ once $r$ is large. An attracting fixed point with
a large basin does both. But so does any attractor on which the *readout* is constant, for example a period-2 orbit
whose two points give the same logits. Training puts no pressure against such cycles. That fits what we found: about
13% of addition prompts had a period-2 position, and in free generation the outputs were unchanged.]

= Optional: the measure-theoretic view

Since $s_0$ is random, look at distributions rather than single orbits. Let $mu_0$ be the law of $s_0$. The law at
step $t$ is the pushforward $mu_t = (Phi_E^t)_* mu_0$.

- *Existence of invariant measures.* $Phi_E$ is a continuous self-map of the compact set $K^n$, so by
  Krylov–Bogolyubov it has at least one invariant probability measure: a limit point of the Cesàro averages
  $1/T sum_(t < T) mu_t$.
- *Attracting fixed point.* If $x^*$ is attracting and the support of $mu_0$ lies in its basin, then
  $mu_t -> delta_(x^*)$ weakly. The convergence rate is governed by $rho(D Phi_E (x^*))$, and the Lyapunov exponent at
  $x^*$ is $log rho approx log 0.78 approx -0.25$ per loop.
- *Attracting 2-cycle* ${a, b}$. Then $mu_t$ does not converge; it alternates near $delta_a$ and $delta_b$. The
  Cesàro averages converge to the invariant measure $(delta_a + delta_b) \/ 2$. All Lyapunov exponents can be
  negative, so a "uniform convergence" finding from Lyapunov exponents alone cannot rule cycles out.
- *What we could test.* With a handful of seeds per prompt we checked that different $s_0$ led to the same limit, which
  is evidence about the basin. We never established uniqueness of the attractor.

= Vocabulary used in the write-up

/ Prompt, few-shot demos: The input text. "Few-shot" means it starts with solved examples (e.g. "12+35=47") before the
  problem.
/ Position $q$, answer position: The row of the state we analyze, usually the last one, where the next token is the
  answer (digit) or the judgment (True/False).
/ Teacher forcing: For multi-digit answers, the correct earlier digits are placed in the prompt, so each digit is
  studied at a fixed position.
/ Readout $R$: A scalar function of the state at $q$, computed by running the coda: a *centered digit logit*
  $ell_k - 1/10 sum_(j = 0)^9 ell_j$, or the log-odds $ell_"True" - ell_"False"$. Its gradient $g = nabla R(x^*)$ is
  computed by automatic differentiation.
/ Logit lens: Applying the readout to an intermediate state $s_t$, to see what the model "would answer" at loop $t$.
/ Probe, decoding: A linear functional $w$ with $"sign"(w^top s + b)$ predicting a label. The *mass-mean* probe is
  $w = macron(s)_"True" - macron(s)_"False"$, a difference of class means of the states. "Decodable" means a probe
  separates the classes on held-out data. AUROC is the probability that a random positive item scores above a random
  negative one.
/ Steering, push, lever: An intervention that adds a vector $P$ to the state. *One-shot*: added once. *Persistent*:
  added before every loop step. A *lever* is a push direction chosen to move a readout.
/ Sham: The same pipeline with no push, to measure baseline effects.
/ Blinking: Our word for a position whose state settles into a period-2 orbit, $s_t$ alternating between two points.
/ Source token: The earliest position that blinks. Later positions inherit it through the triangular structure.
/ JVP, VJP: $J$ is $5280 times 5280$ and never formed. We only compute products: $J v$ (by finite differences
  $(Phi(x + epsilon v) - Phi(x - epsilon v)) \/ (2 epsilon)$) and $J^top w$ (by reverse-mode automatic
  differentiation, one backward pass).
/ Arnoldi, Ritz values: A Krylov method that estimates leading eigenvalues from $J v$ products alone: build an
  orthonormal basis of $"span"{v, J v, dots, J^(m-1) v}$ and diagonalize $J$ compressed to it. Its estimates are "Ritz
  values"; "validated" means the residual $norm(J v - lambda v)$ was checked to be small.
/ fp32, bf16: Floating-point formats. bf16 keeps 8 bits of mantissa, a relative precision of about $2^(-8)$, so
  rounding acts like a fresh perturbation at every step and hides convergence below a noise floor. Claims about
  convergence and cycles therefore used fp32.
/ Held-out: Evaluated on prompts not used to build the direction being tested.

= From here to the derivations

Now fix a prompt, let $x^*$ be a fixed point of $Phi_E$, and look at the answer position $q$. Each derivation in
`derivations.pdf` is one standard operator-theory fact applied to this system.

/ D1, linearization: $J = D(Phi_E)_q (x^*)$. Triangularity means a perturbation at the last position $q$ never
  affects earlier positions, so its evolution is governed by $J$ alone, exactly to first order.
/ D2, one-shot pushes: $delta_t = J^t P$. Gelfand gives eventual decay at rate $rho approx 0.78$; non-normality allows
  growth first.
/ D3, persistent pushes: Neumann series at $z = 1$: $sum_(t >= 1) J^t = J (I - J)^(-1)$. The adapter shows that a
  persistent push is the same as changing the context $E$.
/ D4, optimal lever: $P |-> Delta R_T approx g^top A_T P$ is a linear functional. Its representer is $A_T^top g$, the
  pulled-back covector. The best push in a ball points along the representer.
/ D5, non-normality: For normal $J$ the pulled-back covector $A_T^top g$ stays aligned with $g$. Here it is nearly
  orthogonal. A *probe* is a vector (a difference of state means: where the dynamics push the state). A *lever* is a
  pulled-back covector (what the readout is sensitive to). The adjoint relates them, and for a strongly non-normal $J$
  the adjoint sends things somewhere very different. That is the mathematical content of "decodable is not
  controllable".
/ D6, cheap levers: Truncate the Neumann series after one term ($J^top g$), or replace $J$ by its adapter factor.
/ D7, period-2 orbits: A real eigenvalue crossing $-1$ at a source position: a flip (period-doubling) bifurcation,
  reduced to a one-dimensional normal form on a center manifold.
/ D8, damping: $G = (1 - beta) "id" + beta Phi_E$ has the same fixed points, and by spectral mapping its derivative has
  eigenvalues $1 - beta + beta lambda$.
/ D9, inheritance: The full derivative is block lower-triangular, so its spectrum is the union of the per-position
  spectra.
/ D10, unification: Steering and blinking are the resolvent at $z = 1$ and at $z = -1$.

Suggested order: Section 1 here, then D1–D5 in `derivations.pdf` (the main contribution), then D7–D9 (the dynamics),
then the exercises.
