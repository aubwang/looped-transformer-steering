---
title: Building a Transformer on Eleven Words
---

# Building a Transformer on Eleven Words

I've looked around the internet and haven't really found a guide to transformers that felt satisfying from both a motivational and a mathematical perspective, so I set out to write my own. My goal here is to make it so that every component of the transformer arrives naturally as the solution to a concrete obstacle we run into, and also to express every step cleanly in math. 

## 1 · Eleven words and four sentences

Suppose we live in a world where the English language has only $11$ words. Call this set our **vocabulary**, $\mathcal V$:

$$\begin{aligned}
\mathcal V = \{&\code{the},\ \code{a},\ \code{on},\ \code{cat},\ \code{dog},\ \code{bird},\\
&\code{mat},\ \code{hat},\ \code{grass},\ \code{sat},\ \code{chased}\}
\end{aligned}$$

Now suppose you go out into the world, and you read some books and decide to write down the following $4$ sentences:

1. `the cat sat on the mat`
2. `the cat sat on the mat`
3. `the cat sat on the hat`
4. `the cat sat on the grass`


Suppose someone gives you the following words. Can we use the four sentences we wrote down to guess what goes in the blank?

`the cat sat on the _____`

Let's call the words before the blank the **context**. All four sentences start with `the cat sat on the`, so we can count how often each word follows that context.

{{widget: vocab-counts}}

This table defines the **observed frequency** of each word after `the cat sat on the`:

$$f: \mathcal V \longrightarrow [0, 1], \qquad f(v) = \frac{\operatorname{count}(v)}{N}$$

where $N$ is the number of observed endings ($N = 4$ here). So $f(\code{mat}) = \tfrac{2}{4}$, $f(\code{hat}) = \tfrac{1}{4}$, $f(\code{grass}) = \tfrac{1}{4}$, and $f(v) = 0$ for the other eight words. Summing across every candidate accounts for all four observations:

$$\sum_{v \in \mathcal V} f(v) = \frac{2 + 1 + 1 + 0 + \cdots + 0}{4} = 1$$

We can also ask about a group of words—an **event**, written as a subset $A \subseteq \mathcal V$—by adding the frequencies of its members:

$$P(A) = \sum_{v \in A} f(v)$$

For example, the event `mat` or `hat` is the union $\{\code{mat}\} \cup \{\code{hat}\} = \{\code{mat}, \code{hat}\}$. Because the two one-word sets share no elements, their frequencies can be added:

$$P(\{\code{mat}\} \cup \{\code{hat}\}) = f(\code{mat}) + f(\code{hat}) = \frac{2}{4} + \frac{1}{4} = \frac{3}{4}$$

Suppose we pick one of our four sentences at random, each equally likely. Let $Y$ be its last word. That's a **random variable** with values in $\mathcal V$. The **probability measure** $P$ we just built tells us the chance that $Y$ belongs to any group of words $A$: $\Pr(Y\in A)=P(A)$.

## 2 · Turning words into numbers

So counting works fine when a context appears verbatim in our data; we can say that when `the cat sat on the` appears, the next word is `mat` 50% of the time, `hat` 25% of the time, and `grass` 25% of the time.

OK, but what if someone asks us to predict what word comes after `a cat sat on the ____`?

Since none of our four sentences start that way, our percentage lookup table can't answer that question.

Changing the first word from `the` to `a` still leaves us looking for something a cat could sit on.

And if we swapped `cat` for `dog`, many of the same endings would still make sense.


As computers have no intuitive sense of language, we need to figure out a way we can represent these words as numbers. That way, the model can compare words with each other. Then our question becomes: how can we assign numerical representations to every word in our word universe $\mathcal V$, and how can we make it so that these numbers have some sort of semantic meaning?

Let's start with the simplest idea and give each word a number from $1$ to $11$ (call it their "ID"):

{{widget: word-ids}}

So now each word has a numeric representation, but we have a problem.

If the word `cat` has an ID equal to $4$, what does that mean? The words `on` ($3$) and `dog` ($5$) are both a value of $1$ away from `cat`, so does that mean `cat` semantically is just as close to the word `dog` as it is to the word `on`? No, we would expect the words that are close in meaning to `cat` be words that correspond with small animals. 



Instead of a single index, let's give each word a short vector of length $d$, e.g. $\v{e}_{\mathrm{word}} = (e_1, \dots, e_d)^\top$ where each $e_k \in \mathbb R$ is an adjustable number. We call this a word's **embedding**. Stack one embedding per row to form the **embedding matrix** $\m{E} \in \mathbb R^{|\mathcal V| \times d}$. Here $|\mathcal V| = 11$. A word's ID now simply tells us which row of $\m{E}$ contains that word's representation vector.

**Notation.** A bold lowercase letter such as $\v{e}_{\mathrm{cat}}$ denotes a column vector. Its transpose $\v{e}_{\mathrm{cat}}^\top$ is the row for `cat` in the matrix $\m{E}$. We use bold capitals for matrices and plain letters for individual numbers. For example, $E_{4,1}$ is the first coordinate of the embedding for word $4$, `cat`. All indices start at $1$.

In a real model, the numbers $e_k$ in each embedding are learned during training. 

For now, let's assume that we have $d = 2$ dimensions, and we set the numbers of each word vector so that words with similar meanings are clustered together.

{{widget: embedding-plane}}

Each word embedding is a vector starting at the origin. Notice that `cat` and `dog` point in nearly the same direction ($19.5^\circ$ apart), whereas `cat` and `mat` open a wider angle ($69.2^\circ$), and `cat` and `on` are pointing in nearly opposite directions ($138.4^\circ$).

To measure this numerically for two vectors $\v{u} = (u_1, u_2)^\top$ and $\v{v} = (v_1, v_2)^\top$, we take their **inner product** (or **dot product**):

$$\v{u}^\top \v{v} = \v{u} \cdot \v{v} = u_1 v_1 + u_2 v_2$$

For `cat` $(0.75, 0.75)$ and `dog` $(1.05, 0.50)$, multiplying matching coordinates gives

$$\v{e}_{\mathrm{cat}}^\top \v{e}_{\mathrm{dog}} = (0.75)(1.05) + (0.75)(0.50) = 1.1625,$$

compared with $0.4125$ for $\v{e}_{\mathrm{cat}}^\top \v{e}_{\mathrm{mat}}$ and $-0.6750$ for $\v{e}_{\mathrm{cat}}^\top \v{e}_{\mathrm{on}}$. Geometrically, if $\theta$ is the angle between two nonzero vectors,

$$\cos \theta = \frac{\v{u}^\top \v{v}}{\lVert \v{u} \rVert \, \lVert \v{v} \rVert}$$

We call $\cos \theta$ **cosine similarity**. Geometrically, when $\cos \theta$ is close to $1$, $\v{u}$ and $\v{v}$ are nearly parallel. When $\cos \theta$ is close to $0$, $\v{u}$ and $\v{v}$ are nearly perpendicular. When $\cos \theta$ is close to $-1$, $\v{u}$ and $\v{v}$ are pointing nearly in opposite directions.

### Where is each word in the sentence?

For our context `the cat sat on the ____`, look up each word in $\m{E}$ and stack the five rows in sentence order. Call this matrix $\m{E}_{\mathrm{ctx}}$. Its $i$th row is $\v{e}_{w_i}^\top$, where $w_i$ is the word at position $i$.

{{widget: embed-before-pos}}

Look at rows $1$ and $5$: both copies of `the` become $(-1.00, 0.40)$. Even though the matrix lists the rows in order, an individual row vector by itself doesn't carry information of which position it is in the sentence. Then, any calculation involving this vector on its own also loses positional information. So, we need to figure out a way to slightly tweak these vectors so that we can inject position information, without distorting the semantic information in the vectors cosine similarity. In English, how can we shift a vector slightly so that it still looks like the original, but the amount its shifted tells us where it is in a sentence.

Mathematically, we call this shift that injects position a **position offset** $\v{p}_i$. We add $\v{p}_i$ to our original embedding vector $\v{e}_{w_i}$, and we call the resulting vector:

$$\v{x}_i = \v{e}_{w_i} + \v{p}_i$$

What should $\v{p}_i$ be? Suppose we try the most straightforward option, adding the word's position index to each coordinate of its embedding. For example, adding $(1, 1)$ to $\v{e}_{w_1}$ at position $1$, $(5, 5)$ to $\v{e}_{w_5}$ at position $5$, and so on. In our example, doing this would result in the following matrix, where each row is now unique:

{{widget: index-offset-matrix}}

Now imagine a $5{,}000$-word document. At positions $4{,}999$ and $5{,}000$, the word embeddings contribute numbers near $1$, while the index adds $4{,}999$ and $5{,}000$:

{{widget: index-offset-5000}}

Look at what that does to **cosine similarity**. Without offsets, `the` compared to itself has $\cos\theta = 1.000$, while `the` and `cat` have $\cos\theta = -0.394$. After adding the position index, every vector near the end of the document is shoved into the same $45^\circ$ direction $(1, 1)$:

| Pair of words | Cosine similarity $\cos\theta$ |
|:--|:--:|
| Raw `the` vs. `the` (no offsets) | $1.000$ |
| Raw `the` vs. `cat` (no offsets) | $-0.394$ |
| `the` (pos $1$) vs. `the` (pos $5{,}000$) | $0.707$ |
| `cat` (pos $4{,}999$) vs. `the` (pos $5{,}000$) | $1.000$ |

Our position offset swamps our word identities: the exact same word `the` at position $1$ and position $5{,}000$ drops to $\cos\theta = 0.707$, while `cat` at position $4{,}999$ and `the` at position $5{,}000$ look completely aligned ($\cos\theta \approx 1.000$). Every pair of words late in the sequence is pushed onto the same angle, so this is not the small nudge that we're looking for.

Ideally, we want position offsets with three properties:

- **Offsets are small and bounded**, so our word embeddings don't get drowned out as sentence length grows.
- **Distinct positions get distinct offsets**, so that we can cleanly tell which position a word is in.
- **Stepping from one position to the next is consistent**, so the spacing between two neighboring words stays the same, no matter where they are in the sentence.

A short arrow of radius $r = 0.15$ rotating around a circle gives all three. Each step turns the arrow by $\omega = 45^\circ$, producing eight distinct offsets before repeating—plenty for our short contexts:

$$\v{p}_i = r \begin{bmatrix} \cos(i\omega) \\ \sin(i\omega) \end{bmatrix}, \qquad r = 0.15,\quad \omega = 45^\circ$$

{{widget: tip-to-tail-offset}}

::: details Optional: Rotation matrices and scaling to longer contexts
{{widget: position-circle}}

### Rotation as a linear map on $\mathbb R^2$

In our 2D example, each position offset $\v{p}_i$ is an element of $\mathbb R^2$. Stepping forward one position—moving from $\v{p}_i$ to $\v{p}_{i+1}$—is equivalent to applying the linear map $T: \mathbb R^2 \to \mathbb R^2$ to the offset vector, given by multiplying by the $2 \times 2$ rotation matrix $\m{R}$ (rotating counterclockwise by $\omega$):

$$T(\v{v}) = \m{R} \v{v}, \qquad \m{R} = \begin{bmatrix} \cos\omega & -\sin\omega \\ \sin\omega & \cos\omega \end{bmatrix} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix} \quad (\text{at } \omega = 45^\circ)$$

Because $T$ is linear and $\m{R}$ is orthogonal ($\m{R}^\top \m{R} = \m{I}$, $\det \m{R} = 1$), $T$ preserves vector lengths ($\lVert \m{R} \v{p}_i \rVert = \lVert \v{p}_i \rVert = r$) and the angles between any two offset vectors since ($(\m{R} \v{p}_i)^\top (\m{R} \v{p}_j) = \v{p}_i^\top \m{R}^\top \m{R} \v{p}_j = \v{p}_i^\top \v{p}_j$). Therefore the relative spacing between two vectors stays identical.
### Combining multiple circles for longer contexts

With $d=2$, we have one plane. Turning $45^\circ$ per step brings us back to where we started after eight tokens, so positions $1$ and $9$ get the same offset.

What if we had room for another circle? For even $d$, we can group the coordinates into $d/2$ pairs, each with its own plane:

$$\mathbb R^d = \underbrace{\mathbb R^2 \oplus \mathbb R^2 \oplus \cdots \oplus \mathbb R^2}_{d/2 \text{ planes}}$$

Think of the hands of a clock. The minute hand can return to the same place while the hour hand has moved on. With six coordinates, we can try three circles that make a full turn every $8$, $27$, and $200$ steps. For all three hands to return together takes $5{,}400$ steps, so every position in our $200$-token example has a different offset.

The plots compare each offset with the one at position $100$ using cosine similarity. For one circle, that gives $\cos((j-100)\omega)$. For three circles of equal radius, it gives the average of the three cosines.
- **One circle (top):** hits $+1$ every $8$ steps ($j = 4, 12, \dots, 100, \dots, 196$), note that all of these positions have identical offsets.
- **Three circles (bottom):** hits $+1$ **only** at $j = 100$.

(Modern architectures like Huginn and Llama apply these multi-plane rotations directly to queries and keys inside attention, as we will see in the RoPE discussion in Section 5, rather than adding $\m{P}$ upfront.)
:::

Stack the rounded position offsets as the rows $\v{p}_i^\top$ of $\m{P}$. Adding them to the word embeddings gives the sequence matrix $\m{X} = \m{E}_{\mathrm{ctx}} + \m{P}$.

{{widget: sequence-matrix-x}}

We can also write $\m{X}$ as a stack of row vectors, for a given context length $n$ we have

$$\m{X} = \begin{bmatrix} \v{x}_1^\top \\ \vdots \\ \v{x}_n^\top \end{bmatrix} \in \mathbb R^{n \times d}$$

Here $n = 5$ and $d = 2$, so $\m{X}$ is $5 \times 2$ (longer contexts/sequences add rows). Each $\v{x}_i$ now encodes *which word* appears at position $i$ and *where* it appears, recall that $\v{x}_i = \v{e}_{w_i} + \v{p}_i$.

## 3 · From a sequence to a prediction

Given the matrix $\m{X}$ for `the cat sat on the`, how can we predict the word that comes next?

We want a **probability distribution over $\mathcal V$**. Each word gets a nonnegative number, and the eleven numbers have to add up to $1$.

{{widget: sequence-to-dist}}

That means we have two tasks to do:

1. **Scoring:** Given $\m{X}$, how can we assign some sort of number to each candidate word $v \in \mathcal V$? How can we create a rule so that better candidate words get higher scores than worse words?
2. **Normalization:** Once we have our collection of 11 scores (one for each word $v \in \mathcal V$), how can we turn those scores into valid probabilities?

Let's start by solving the second problem first.

### Turning any eleven scores into probabilities

Suppose we have some arbitrary scoring function that takes a candidate word $v \in \mathcal V$ and our matrix $\m{X}$, and returns some real number. Let's call this scoring function $s$. Mathematically, we can write this function as:

$$s: \mathcal V \times \mathbb R^{n \times d} \longrightarrow \mathbb R$$

Let's write this function in a more readable manner, and set its output (the score) for some candidate word $v$ equal to $z_v$:

$$ z_v = s(v, \m{X}) $$

So in our example, we have 

$$z_1 = s(\code{the}, \m{X}) \quad z_2 = s(\code{a}, \m{X}) \quad  ... \quad z_{11} = s(\code{chased}, \m{X})$$ 

We don't have to define exactly $s(v, \m{X})$ yet, but remember that our only hard requirement is that $s(v, \m{X})$ return a real number. That's it. We could set  $z_v = s(v, \m{X}) = 1$ and have it always return $1$ no matter which word candidate $v$ and context matrix $\m{X}$, and all the math in this section would work out. 

For now, lets just say we've obtained eleven raw scores $z_v \in \mathbb R$, we can call these scores **logits**. Stacking them gives the vector of logits $\v{z} = (z_1, \dots, z_{11})^\top \in \mathbb R^{11}$, and we need to turn it into a valid probability distribution over $\mathcal V$. This means we need to transform each $z_v$ into a probability $\mu(v)$ so that every $\mu(v) \geq 0$ and $\sum_{v \in \mathcal V} \mu(v) = 1$. (We write $\mu$, the usual letter for a probability measure, for our model's distribution, to keep it apart from the observed frequencies $f$.)

One way we can do this is called **softmax**. First, recall that $e^x > 0$ for any $x$. Then, lets exponentiate each $z_v$ value and sum them up:

$$Z = \sum_{u \in \mathcal V} e^{z_u} > 0$$

Then, we can define each entry $\mu(v)$ for $v \in \mathcal V$ and stack them into the prediction vector $\vmu = \operatorname{softmax}(\v{z}) \in \mathbb R^{11}$ as:

$$\mu(v) = \frac{e^{z_v}}{Z}, \qquad \vmu = \operatorname{softmax}(\v{z}) = \frac{1}{Z}\begin{bmatrix} e^{z_1} \\ \vdots \\ e^{z_{11}} \end{bmatrix}$$

We know that $\mu(v) > 0$, and note that

$$\sum_{v \in \mathcal V} \mu(v) = \sum_{v \in \mathcal V} \frac{e^{z_v}}{Z} = \frac{1}{Z} \sum_{v \in \mathcal V} e^{z_v} = \frac{\sum_{v \in \mathcal V} e^{z_v}}{\sum_{u \in \mathcal V} e^{z_u}} = 1$$

Therefore, the probabilities $\mu(v)$ define a distribution over $\mathcal V$, no matter how $s(v, \m{X})$ is computed. For example in the case where $s(v, \m{X}) = 1$ is constant, then the value $\mu(v)$ for every word is $e^1 / (11 e^1) = 1/11 \approx 9.1\%$, and the values of each $\mu(v)$ add up to $1$. So under softmax and a scoring rule $s(v, \m{X}) = 1$, we've created a uniform distribution over $\mathcal V$.

### What should produce the scores?

So we've figured out the normalization component of our problem, now we must develop an effective scoring rule. We saw above that a scoring rule $s(v, \m{X}) = 1$ produces a uniform distribution no matter the context $\m{X}$, meaning that it just randomly picks a word out of the basket and isn't a good model.

We want a scoring rule so that it gives scores to each candidate word in our dictionary, and also gives higher scores to words that are a better fit for inserting in the blank, given context $\m{X}$. One nice property of softmax is for any two logits $z_a < z_b$, then $\mu(a) < \mu(b)$, meaning that the softmax operation passes through the ranking of candidate words given by our scoring function $s$. So a higher logit score means a higher probability as well, which is desirable (better words should have bigger probability mass).





Let's try to come up with a possible scoring function. What if we look at the **last row** of $\m{X}$, the last word before the blank? For `the cat sat on the`, row $5$ represents the second `the`. 

$$\v{x}_n = \v{x}_5 = \v{e}_{\text{the}} + \v{p}_5 = \begin{bmatrix} -1.11 \\ 0.29 \end{bmatrix}$$

To score a candidate word $v$, take the dot product between its embedding $\v{e}_v$ and $\v{x}_n$ (basically our scoring rule becomes how similar is our candidate word compared to the last word in the sentence).

$$z_v = \v{e}_v^\top \v{x}_n $$

For candidate `mat`, $z_{\text{mat}} = (1.00)(-1.11) + (-0.45)(0.29) = -1.2405$. Let's do this for all eleven candidate scores (logits), we can do this by calculating the vector $\v{z} = \m{E} \v{x}_n$, and then apply softmax:

$$\vmu = \operatorname{softmax}(\v{z}), \qquad \v{z} = \m{E} \v{x}_n$$

{{widget: first-predictor-readout}}

We have now constructed our first end to end predictor! Let's note two additional properties of softmax:

- **Only differences between logits matter:** $\mu(a) / \mu(b) = e^{z_a - z_b}$, so our probability ratios depend only on the difference between two logit values.
- **Finite logits never yield exact zero:** since $e^x > 0$, every word under softmax gets assigned a positive probability score.

Right now, the best guess for our predictor for what word to fill in `the cat sat on the ____` is actually the word `the` again, with $\mu(\code{the}) = 26.8\%$, meaning our best guess sentence completion is `the cat sat on the the`.

This makes sense, since according to our scoring rule, we are scoring our candidates based on how similar they are to the last word in the sequence, and if our last word is `the`, then the word most similar to it is itself. Worse yet, looking only at $\v{x}_5$ throws away rows $1$ through $4$ completely—so `the cat sat on the ____` and `on the mat sat the ____` share the exact same $\v{x}_5 = (-1.11, 0.29)^\top$ and produce identical predictions!

### Using all previous words

You might ask, why are we only using $\v{x}_5$ when all $n$ rows are sitting right there in $\m{X}$? Wouldn't including all words in our scoring improve our model?

Lets look at the score $z_v$ for a **single** candidate word $v$ first. Earlier we had $z_v = \v{e}_v^\top \v{x}_n$. To let every word in the prompt contribute, take candidate $v$'s dot product against the vector $\v{x}_j$ from *each* row ($j = 1, \dots, n$), give each position $j$ an adjustable weight $\alpha_j \in \mathbb R$, and add them up:

$$z_v = s(v, \m{X}) = \alpha_1(\v{e}_v^\top \v{x}_1) + \alpha_2(\v{e}_v^\top \v{x}_2) + \cdots + \alpha_n(\v{e}_v^\top \v{x}_n) = \sum_{j=1}^{n} \alpha_j (\v{e}_v^\top \v{x}_j)$$

Because $\v{e}_v^\top$ doesn't depend on $j$, we can factor $\v{e}_v^\top$ out to the left and look at the two halves of that dot product and define a vector $\v{h}$:

$$z_v = \underbrace{\v{e}_v^\top}_{\text{candidate word } v} \; \underbrace{(\alpha_1 \v{x}_1 + \alpha_2 \v{x}_2 + \cdots + \alpha_n \v{x}_n)}_{\v{h} \in \mathbb R^d} = \v{e}_v^\top \v{h}$$

Take `the cat sat on the`. The weight $\alpha_2$ belongs to slot $2$, so whatever word lands there gets multiplied by $\alpha_2$. For now we pick the weights ourselves. Later we'll let the model learn them from the endings it sees.

Click through the words to build $\v{h}$ one step at a time. Then try sliding $\alpha_2$ from $1$ down to $0$. The `cat` step shrinks until it disappears. Slide past $0$ and it points the other way.

{{widget: summary-walk}}

The arrow from the origin to our final point is $\v{h}$. That's what we mean by a **summary**. You can also think of it as a blend of our five words, where $\alpha_j$ says how much of word $j$ goes in. When we score `mat`, all its embedding gets to see is where we ended up. Two prompts that land at the same point get the same score for every candidate.

Suppose we replace `cat` with `dog` in slot $2$. Four of our five steps stay the same. Both words get the same position offset in slot $2$, and therefore the endpoint moves by $\alpha_2(\v{e}_{\mathrm{dog}}-\v{e}_{\mathrm{cat}})$. So what if `cat` showed up in slot $4$ of another prompt? Making the same replacement there moves the endpoint by $\alpha_4(\v{e}_{\mathrm{dog}}-\v{e}_{\mathrm{cat}})$. We reuse the word embeddings wherever the words appear, but how much they count still depends on the slot.

In matrix form, that same sum is $\m{X}^\top\valpha$. The first row of $\m{X}^\top$ holds the horizontal coordinates of our five input vectors, and the second holds their vertical coordinates. Multiplying by $\valpha$ weights and adds each row. Once we have $\v{h}$, we can score all $11$ candidate words at once:

$$\v{h} = \m{X}^\top \valpha, \qquad \v{z} = \m{E}\v{h}, \qquad \vmu = \operatorname{softmax}(\v{z})$$

(Combining the right-hand side into $\v{h}$ first also means we only multiply by the dictionary matrix $\m{E}$ once instead of $n$ separate times $\m{E}\v{x}_1, \dots, \m{E}\v{x}_n$, which saves a ton of work when $\m{E}$ has tens of thousands of words.)

Notice that our previous last-row-only scoring rule was just the special case $\valpha = (0, 0, 0, 0, 1)^\top$: we multiply every row except for the last one by zero, so we have $\v{h} = 0\cdot\v{x}_1 + \cdots + 0\cdot\v{x}_4 + 1\cdot\v{x}_5 = \v{x}_5 = (-1.11,\ 0.29)^\top$ and therefore $\v{z} = \m{E}\v{h} = \m{E}\v{x}_5$.

To start before training, let's initialize every position with equal weight $\alpha_j = \tfrac{1}{5} = 0.2$, $\valpha = (\tfrac{1}{5}, \tfrac{1}{5}, \tfrac{1}{5}, \tfrac{1}{5}, \tfrac{1}{5})^\top$. Drag the $\alpha_1, \dots, \alpha_5$ sliders or click the presets below to watch the right-hand side steer $\v{h}$ on the 2D plane and see how each candidate $\v{e}_v^\top \v{h}$ scores against it:

{{widget: ar-weights-preview}}

Try clicking **Swap cat ↔ sat** above while **Equal weights** ($\alpha_j = \tfrac{1}{5}$) is selected. Even though the sentence scrambles into `the sat cat on the`, the summary vector $\v{h} = (-0.512, 0.132)^\top$ and every predicted probability $\mu(v)$ stay **identical** to the unscrambled sentence.

Why did our position offsets $\v{p}_j$ fail to protect word order here? Compare what happens when all weights are equal versus when all weights are different from each other:

**Case 1 · All weights equal ($\alpha_1 = \cdots = \alpha_5 = \tfrac{1}{5}$).** Before summing, each position offset is paired with a specific word in its own row ($\v{e}_{\text{cat}} + \v{p}_2$ vs. $\v{e}_{\text{sat}} + \v{p}_3$). Because every row gets the **same** multiplier $\tfrac{1}{5}$, we're able to factor $\tfrac{1}{5}$ out and group all the $\v{e}$'s into one pile and all the $\v{p}$'s into a separate pile:

$$\begin{aligned}
\text{\code{the cat sat on the}:}\quad \v{h}_A &= \frac{1}{5}\big(\v{e}_{\text{the}} + \v{e}_{\text{cat}} + \v{e}_{\text{sat}} + \v{e}_{\text{on}} + \v{e}_{\text{the}}\big) + \frac{1}{5}\big(\v{p}_1 + \v{p}_2 + \v{p}_3 + \v{p}_4 + \v{p}_5\big), \\
\text{\code{the sat cat on the}:}\quad \v{h}_B &= \frac{1}{5}\big(\v{e}_{\text{the}} + \v{e}_{\text{sat}} + \v{e}_{\text{cat}} + \v{e}_{\text{on}} + \v{e}_{\text{the}}\big) + \frac{1}{5}\big(\v{p}_1 + \v{p}_2 + \v{p}_3 + \v{p}_4 + \v{p}_5\big)
\end{aligned}$$

Note that the above two equations are equal to each other, $\v{h}_A = \v{h}_B$. **You can no longer tell which $\v{e}$ came with which $\v{p}$**: the $\v{e}$ pile still remembers *which* words appeared, but $\v{e}_{\text{cat}} + \v{e}_{\text{sat}} = \v{e}_{\text{sat}} + \v{e}_{\text{cat}}$ forgets their order, while the $\v{p}$ pile is just the same constant $\tfrac{1}{5}(\v{p}_1 + \cdots + \v{p}_5)$ and the values $\v{p}_i$ aren't bound to any particular word.

**Case 2 · All weights different (say $\valpha = (0.10, 0.15, 0.20, 0.25, 0.30)^\top$).** Now try doing that exact same regrouping when every position has its own distinct weight:

$$\begin{aligned}
\text{\code{the cat sat on the}:}\quad \v{h}_C &= \big(\alpha_1 \v{e}_{\text{the}} + \alpha_2 \v{e}_{\text{cat}} + \alpha_3 \v{e}_{\text{sat}} + \alpha_4 \v{e}_{\text{on}} + \alpha_5 \v{e}_{\text{the}}\big) + \big(\alpha_1 \v{p}_1 + \cdots + \alpha_5 \v{p}_5\big), \\
\text{\code{the sat cat on the}:}\quad \v{h}_D &= \big(\alpha_1 \v{e}_{\text{the}} + \alpha_2 \v{e}_{\text{sat}} + \alpha_3 \v{e}_{\text{cat}} + \alpha_4 \v{e}_{\text{on}} + \alpha_5 \v{e}_{\text{the}}\big) + \big(\alpha_1 \v{p}_1 + \cdots + \alpha_5 \v{p}_5\big)
\end{aligned}$$

Note how in the $\v{e}$ pile, each position gets its own distinct weight $\alpha_j$.
- In `the cat sat on the`, $\v{e}_{\text{cat}}$ is scaled by $\alpha_2 = 0.15$ and $\v{e}_{\text{sat}}$ by $\alpha_3 = 0.20$.
- In `the sat cat on the`, $\v{e}_{\text{sat}}$ gets $0.15$ and $\v{e}_{\text{cat}}$ gets $0.20$.

Because $\alpha_1\v{e}_{\text{the}}$, $\alpha_4\v{e}_{\text{on}}$, $\alpha_5\v{e}_{\text{the}}$, and the $\v{p}$ pile are identical in $\v{h}_C$ and $\v{h}_D$, subtracting $\v{h}_D$ from $\v{h}_C$ cancels everything except positions $2$ and $3$:

$$\begin{aligned}
\v{h}_C - \v{h}_D &= (\alpha_2 \v{e}_{\text{cat}} + \alpha_3 \v{e}_{\text{sat}}) - (\alpha_2 \v{e}_{\text{sat}} + \alpha_3 \v{e}_{\text{cat}}) \\
&= (\alpha_2 - \alpha_3)\v{e}_{\text{cat}} + (\alpha_3 - \alpha_2)\v{e}_{\text{sat}} \\
&= (\alpha_2 - \alpha_3)(\v{e}_{\text{cat}} - \v{e}_{\text{sat}}) \\
&= (0.15 - 0.20)\!\left(\begin{bmatrix} 0.75 \\ 0.75 \end{bmatrix} - \begin{bmatrix} -0.20 \\ -1.10 \end{bmatrix}\right) = (-0.05)\begin{bmatrix} 0.95 \\ 1.85 \end{bmatrix} = \begin{bmatrix} -0.0475 \\ -0.0925 \end{bmatrix} \ne \v{0}
\end{aligned}$$

So this predictor can tell the two word orders apart. Swap any two words with different embeddings and the same calculation works, as long as their slots have different weights. Whether the predictor actually gives a good score to the next word when `cat` turns up somewhere new is a separate question. We'd have to try it on prompts we held out of training.

In any case, using equal weights and therefore assigning equal coefficients to each preceding word in a sentence is probably the wrong approach. For example, in the sentence `the cat sat on the ____`, the words in positions $1$ and $5$ (`the`) are less important than the words in positions $2, 3, 4$ (`cat sat on`) with regards to predicting which word belongs in the blank. So we would expect that $\alpha_1$ and $\alpha_5$ should be smaller than $\alpha_2, \alpha_3, \alpha_4$. We now need to figure out which values to use for our $\alpha_i$.

## 4 · How wrong is it, and what can be changed?

Go back to `the cat sat on the ____`. In our four sentences, `mat` fills the blank twice. With equal position weights, our model gives it about a $5\%$ chance. How wrong is that prediction?

### A number for "how wrong"

Suppose we observe $N$ endings $y_1, \dots, y_N$ after a given context (`the cat sat on the dog`, `the cat sat on the mat`, etc). Treating them as independent draws from $\mu$, we can define the joint probability the model assigns to the observed data—its **likelihood**—as

$$\mathcal L(\mu) = \prod_{i=1}^{N} \mu(y_i)$$

In our example data ($N = 4$), the endings are `mat`, `mat`, `hat`, `grass`, so

$$\mathcal L(\mu) = \mu(\code{mat})^2 \, \mu(\code{hat}) \, \mu(\code{grass}) = \prod_{\substack{v \in \mathcal V \\ f(v) > 0}} \mu(v)^{N f(v)},$$

since each word $v$ appears $\operatorname{count}(v) = N f(v)$ times. So for `mat`, we have $\operatorname{count}(\code{mat}) = N f(\code{mat}) = 4 (2/4)$. 

What we want to do is to **maximize** our likelihood $\mathcal L(\mu)$ (maximize how well we can "fit" the data). To make our math easier, maximizing $\mathcal L(\mu)$ is equivalent to **minimizing** the following expression:

$$L(\mu) = -\frac{1}{N} \ln \mathcal L(\mu) = -\sum_{v \in \mathcal V} f(v) \ln \mu(v)$$

we call the unfancy $L(\mu)$ the **average negative log-likelihood** (also called **cross-entropy**).

This works because the logarithm turns products into sums without changing which $\mu$ wins (since $\ln$ is strictly increasing), the minus sign flips maximization into minimization, and dividing by $N$ averages the penalty per observation. On our four examples (starting from $\alpha_j = 0.2$):

| Next word | Observed $f(v)$ | Current model $\mu(v)$ |
|:--|--:|--:|
| `mat` | $0.50$ | $0.0535$ |
| `hat` | $0.25$ | $0.0552$ |
| `grass` | $0.25$ | $0.0453$ |
| All other 8 words | $0.00$ | $0.8460$ |

Plugging those in gives our loss for the current model that we've built:

$$L(\mu) = -\big(0.50 \ln 0.0535 + 0.25 \ln 0.0552 + 0.25 \ln 0.0453\big) \approx 2.962$$

What is the lowest possible loss *any* model can have? Given our data, suppose we had a magical model such that it perfectly reproduces our data, i.e. $(\mu = f)$. Even under this assumption, our loss cannot be zero. Why? We have three different endings that occur in the data. Remember that we have a perfect model, $(\mu = f)$, so we can write our loss as:

$$L(f) = H(f) = -\sum_{v \in \mathcal V} f(v) \ln f(v) = -\big(0.50 \ln 0.50 + 0.25 \ln 0.25 + 0.25 \ln 0.25\big) \approx 1.040$$

This baseline level of loss $H(f)$ is called the **entropy** of the observed distribution. Loss cannot be lower than $H(f)$.

Then, we can define the gap between any model $\mu$ and the floor $H(f)$ as the **KL divergence**:

$$D_{\mathrm{KL}}(f \| \mu) := L(\mu) - H(f) = \sum_{v : f(v) > 0} f(v) \ln \frac{f(v)}{\mu(v)} \ge 0,$$

Now this value is equal to $0$ if and only if $\mu=f$ (since that implies we have a perfect model). Minimizing cross-entropy $L(\mu)$ is identical to pushing $D_{\mathrm{KL}}(f \| \mu)$ down towards zero.



### What can be changed?

Take a step back and look at the pieces of our predictor so far:

| Component | Role | Fixed or adjustable? |
|:--|:--|:--|
| Training frequencies $f$ | What was observed in the data | Fixed |
| Position offsets $\m{P}$ | Distinguish positions $1, \dots, n$ | Fixed |
| Scoring & softmax $\operatorname{softmax}(\m{E} \v{h})$, $\v{h} = \sum_j \alpha_j \v{x}_j$ | Turn the summary vector into probabilities | Fixed rule |
| Position weights $\valpha = (\alpha_1, \dots, \alpha_5)^\top$ | How much each position contributes | **5 adjustable parameters** |
| Embedding matrix $\m{E}$ ($11 \times 2$) | Two coordinates per word | **22 adjustable parameters** |

Without changing our model's formula, we can try to lower $L(\mu)$ by adjusting the $5$ position weights $\alpha_j$ and the $22$ entries of $\m{E}$. We call these $27$ values our model's **parameters** (or **weights**).

To know which direction to nudge them, first ask how $L$ responds when we nudge a single logit $z_v$. If we write $Z = \sum_{u \in \mathcal V} e^{z_u}$, then that means we can write $\ln \mu(u) = z_u - \ln Z$. Let's use this to simplify $L$:

$$L = -\sum_{u \in \mathcal V} f(u)(z_u - \ln Z) = -\sum_{u \in \mathcal V} f(u) z_u + \ln Z$$

since $\sum_u f(u) = 1$. By differentiating this expression with respect to $z_v$ and using the chain rule, where $\frac{\partial \ln Z}{\partial z_v} = \frac{e^{z_v}}{Z} = \mu(v)$, we can write the derivative of $L$ as:

$$\frac{\partial L}{\partial z_v} = \mu(v) - f(v)$$

If the model puts too much probability on $v$, meaning $\mu(v) > f(v)$, the slope is positive, so lowering $z_v$ reduces the loss; if it puts too little probability on $v$, meaning $\mu(v) < f(v)$, then raising $z_v$ reduces the loss. The chain rule then carries $\frac{\partial L}{\partial \v{z}}$ back to a partial derivative for every single parameter in our model, including all $5$ entries of $\valpha$ and all $22$ entries of $\m{E}$. Collect the parameters in a column vector $\boldsymbol{\theta}$, with the five position weights first and then the entries of $\m{E}$ in row order. The gradient of the loss with respect to these parameters is

$$\nabla_{\boldsymbol{\theta}} L = \left( \frac{\partial L}{\partial \alpha_1}, \dots, \frac{\partial L}{\partial \alpha_5},\; \frac{\partial L}{\partial E_{1,1}}, \frac{\partial L}{\partial E_{1,2}}, \dots, \frac{\partial L}{\partial E_{11,1}}, \frac{\partial L}{\partial E_{11,2}} \right)^\top \in \mathbb R^{27}$$

Because $L$ is a tangled mix of exponentials, logarithms, and products, we can't set $\nabla_{\boldsymbol{\theta}} L = \v{0}$ and solve for the minimum algebraically like you would do in a calculus course. Instead, we use **gradient descent** to minimize cross-entropy numerically: at each step, evaluate those $27$ partial derivatives at the current parameters and nudge every parameter a small step $\eta$ (e.g. $\eta = 0.05$) downhill via the calculation $\boldsymbol{\theta}_{\text{new}} = \boldsymbol{\theta}_{\text{old}} - \eta \nabla_{\boldsymbol{\theta}} L$. Then repeat.

{{widget: training-descent}}

After $2{,}000$ steps, the model assigns positive weight on `cat` ($\alpha_2 = +0.38$) and `sat` ($\alpha_3 = +0.65$) while assigning lower weights to the function words `the` and `on`, just like our earlier predictions. Now our model $\mu(v)$ is nearly identical to $f(v)$, and our calculated loss drops from $2.9622$ to $1.0400$. That's just barely above our calculated entropy floor of $H(f) = 1.0397$.

## 5 · The problem with position-only weights

Squashing into $\v{h} \in \mathbb R^2$ already lets our trained model generalize across similar words *in the same slot* (swapping `cat` for `dog` at position $2$, or `the` for `a` at position $1$). And notice that having "fixed parameters" after training isn't a bug by itself: once training finishes, *every* model's learned parameters stay fixed.

What holds our Section 4 model back is *what* those weights $\valpha = (\alpha_1, \dots, \alpha_5)$ are attached to: they are **position-only weights**, storing one number per slot $j$ without looking at the word vector $\v{x}_j$ sitting there. That creates two big problems, and the second one splits into three.

**1. The model breaks when we shuffle or reword the sentence**

Remember that we can interpret the learned weights $\alpha_j$ as assigning slot $j$ greater or lesser importance: if $\alpha_a < \alpha_b$, the model listens more to whatever word happens to sit in slot $b$ than in slot $a$.

What if we ask our model to fill in the blank for `dog on the grass chased _____`?

{{widget: fixed-alpha-ranking}}

Because $\alpha_j$ is tied to the slot number, `the cat sat on the ____` puts its largest weights on positions $2$ and $3$ (`cat` and `sat`), and `dog on the grass chased _____` *also* puts its largest weights on positions $2$ and $3$ (`on` and `the`). Clearly `on` and `the` are not the words that tell you what the dog chased. Even if you keep the exact same words and shuffle `the cat sat on the ____` into `on the cat sat the ____`, slot $2$ (`the`) still gets high weight.

Position-only weights generalize across words in the *same* slot, but they can't follow important words when they move to *different* slots.

**2. Iterative generation**

In `the cat sat on the ____`, we only predicted position $6$. In reality, a language model has to generate word $2$ from word $1$, word $3$ from words $1$–$2$, word $4$ from words $1$–$3$, and so on, at every length. Doing that with position-only weights means giving each prefix $i \in \{1, \dots, n\}$ its own row of slot weights to build $\v{h}_i = \sum_{j=1}^i \alpha_{ij}\v{x}_j$:

$$\begin{aligned}
\text{predict word 2 after \code{the}:}\quad \v{h}_1 &= \alpha_{11}\v{x}_1 \\
\text{predict word 3 after \code{the cat}:}\quad \v{h}_2 &= \alpha_{21}\v{x}_1 + \alpha_{22}\v{x}_2 \\
\text{predict word 4 after \code{the cat sat}:}\quad \v{h}_3 &= \alpha_{31}\v{x}_1 + \alpha_{32}\v{x}_2 + \alpha_{33}\v{x}_3 \\
\text{predict word 5 after \code{the cat sat on}:}\quad \v{h}_4 &= \alpha_{41}\v{x}_1 + \alpha_{42}\v{x}_2 + \alpha_{43}\v{x}_3 + \alpha_{44}\v{x}_4 \\
\text{predict word 6 after \code{the cat sat on the}:}\quad \v{h}_5 &= \alpha_{51}\v{x}_1 + \alpha_{52}\v{x}_2 + \alpha_{53}\v{x}_3 + \alpha_{54}\v{x}_4 + \alpha_{55}\v{x}_5
\end{aligned}$$

If we stack all $n$ summary vectors $\v{h}_i^\top$ as rows in a matrix, we have the matrix $\m{H} \in \mathbb R^{n \times d}$, which is the same shape as $\m X$. In fact, we can write $\m{H}$ as

$$\m{H} = \begin{bmatrix} \text{—} & \v{h}_1^\top & \text{—} \\ \text{—} & \v{h}_2^\top & \text{—} \\ & \vdots & \\ \text{—} & \v{h}_n^\top & \text{—} \end{bmatrix} = \begin{bmatrix} \alpha_{11} & 0 & \cdots & 0 \\ \alpha_{21} & \alpha_{22} & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ \alpha_{n1} & \alpha_{n2} & \cdots & \alpha_{nn} \end{bmatrix} \begin{bmatrix} \text{—} & \v{x}_1^\top & \text{—} \\ \text{—} & \v{x}_2^\top & \text{—} \\ & \vdots & \\ \text{—} & \v{x}_n^\top & \text{—} \end{bmatrix} = \m{A}\m{X}$$

Where $\m{A}$ is a lower-triangular matrix of position-pair weights: row $i$ is the step asking for context, column $j$ is the earlier slot supplying it, and $\alpha_{52}$ means "how much slot $5$ listens to slot $2$" (Notice that each step $i$ still squashes $\m{X}$ down into a $2$-number vector $\v{h}_i$, just with its own row of weights).

Iterative generation exposes the following problems:

First, rows we never trained don't exist. In our training data, we have $4$ sentences that are all $6$ words long (`the cat sat on the mat/hat/grass`), and therefore training only gives us the weights for $\v{h}_1, \dots, \v{h}_5$ (and $\v{h}_5$ is the one that predicts the blank in `the cat sat on the ___`). However, if we wanted to take this a step further and predict `the cat sat on the mat ____` (or beyond), we need $\v{h}_6$, but we can't compute it because it needs weights that weren't ever trained ($\alpha_{61}, \dots, \alpha_{66}$).

Second, trained rows don't share what they learned. Each $\alpha_{ij}$ is only updated by examples where position $i$ reads position $j$. This means that if our training teaches $\alpha_{52}$ something like "after `the`, look back at the noun", that heuristic only gets used when position is exactly $i = 5$ and it's evaluating position $j=2$. Essentially we are forcing our weights to work in a very granular and compartmentalized manner, which is not good for language, where what matters is which words are nearby and how far back they are, not their *absolute* position. For example, if you read an article like `a` or `the`, you expect a noun like `cat` to follow the article. This doesn't change if you read the article at the start of a book or at the end of a book.

Third, there are way too many weights. A context of length $n$ needs $1 + 2 + \cdots + n = \frac{n(n+1)}{2}$ separate weights. Doubling the context roughly quadruples the weights. If you want a context length of $5{,}000$, you would need $12{,}502{,}500$ weights. 

{{widget: weight-triangle}}

### Making the weights depend on the context

Let's revisit our original autoregressive style model:

$$\v{h}_i = \sum_{j=1}^{i} \alpha_{ij}\, \v{x}_j$$

All of the problems we listed above come from the same thing: each $\alpha_{ij}$ is a constant tied only to a pair of positions $(i, j)$, no matter which of our $11$ words actually sit there. It's kind of like if someone said: "The third word in every sentence is the most important". Maybe sometimes, but definitely not always, and you didn't even look at what the word was.

Instead of storing a fixed matrix $\m{A}$ of constants, we can compute $\m{H}$ from the input $\m{X}$ in three steps:

$$\underbrace{s_{ij} = g(\v{x}_i, \v{x}_j)}_{\text{1. score pairs } (\m{S})} \quad\longrightarrow\quad \underbrace{\alpha_{ij}(\m{X}) = \frac{e^{s_{ij}}}{\sum_{k \le i} e^{s_{ik}}}}_{\text{2. mask \& normalize } (\m{A})} \quad\longrightarrow\quad \underbrace{\v{h}_i = \sum_{j=1}^{i} \alpha_{ij}(\m{X})\,\v{x}_j}_{\text{3. summarize into } \m{H} = \m{A}\m{X}}$$

Let's walk through each of those three steps.

**Step 1 · Define $g$ to score word pairs ($s_{ij} = g(\v{x}_i, \v{x}_j)$).** Instead of looking up a stored number for $(i, j)$, we pick a function $g: \mathbb R^d \times \mathbb R^d \to \mathbb R$ that reads the two rows $\v{x}_i$ and $\v{x}_j$ and outputs a number $s_{ij}$ for how much position $i$ cares about position $j$. We'll call $s_{ij}$ an **attention score**. Careful, this isn't the same kind of number as the logits $z_v$ from Section 3. A logit compares a candidate word against our summary. An attention score compares two rows of the prompt against each other, and its only job is to build the weights. The simplest thing we can try is the dot product from Section 2, $g(\v{x}_i, \v{x}_j) = \v{x}_i^\top \v{x}_j$, so let's use that for now. Let's go back through our problems and see what having a shared rule $g$ immediately fixes.

Take shuffling first. Before, $\alpha_{5j}$ never looked at $\v{x}_j$ at all. Now $g(\v{x}_i, \v{x}_j)$ reads $\v{x}_j = \v{e}_{w_j} + \v{p}_j$, and since the position offset $\v{p}_j$ is small, $g(\v{x}_i, \v{x}_j) \approx g(\v{e}_{w_i}, \v{e}_{w_j})$ is mostly decided by which words are at $i$ and $j$. If `dog` moves from position $1$ to position $4$, $\v{e}_{\text{dog}}$ moves with it, so its score moves with it as well.

Now the rows we never trained. Suppose we have a 67-word prompt and want $\v{h}_{67}$ to predict word $68$. We can just plug in $g(\v{x}_{67}, \v{x}_1), \dots, g(\v{x}_{67}, \v{x}_{67})$. Every input vector is already available to us: $\v{x}_{67} = \v{e}_{w_{67}} + \v{p}_{67}$ is just the last word's corresponding embedding in $\m{E}$ plus its position offset, so we have everything we need to compute $\v{h}_{67}$.

What about trained rows not sharing? Previously, the weight for `the` at position $5$ looking at `cat` at position $2$ was its own number $\alpha_{52}$, so training it did nothing for $\alpha_{96}$, even if `the` and `cat` were at positions $9$ and $6$. Now that the attention score is a function of two vectors, training must change the numbers that $g$ reads (and any weights we later put inside $g$). Let's expand our dot product with $\v{x} = \v{e} + \v{p}$, so $g(\v{x}_5, \v{x}_2) = \v{e}_{\text{the}}^\top \v{e}_{\text{cat}} + \v{e}_{\text{the}}^\top \v{p}_2 + \v{p}_5^\top \v{e}_{\text{cat}} + \v{p}_5^\top \v{p}_2$. Since $\v{p}$ is small, the last three terms are  also small, so if training wants to increase $g$, it mostly does it by making the vectors $\v{e}_{\text{the}}$ and $\v{e}_{\text{cat}}$ in $\m{E}$ line up more. Now if `the` shows up at position $9$ and `cat` at position $6$, we know that expanding $g(\v{x}_9, \v{x}_6)$ will yield the same first term $\v{e}_{\text{the}}^\top \v{e}_{\text{cat}}$ with different (small) offset terms, so the score goes up by about the same amount even though we never explicitly trained `the` and `cat` on positions $9$ and  $6$.

And instead of storing $\frac{n(n+1)}{2}$ separate numbers in $\m{A}$, we only store whatever weights end up inside $g$ (a raw dot product has none), and those don't grow with the context length $n$ at all.

Just like we stacked the weights $\alpha_{ij}$ into $\m{A}$, we can stack the attention scores into an $n \times n$ matrix $\m{S}$, with $s_{ij}$ in row $i$ and column $j$. We don't have to fill it in one cell at a time, either. Entry $(i, j)$ of $\m{X}\m{X}^\top$ is row $i$ of $\m{X}$ dotted with row $j$ of $\m{X}$, which is exactly $\v{x}_i^\top \v{x}_j$. So with our dot product, $\m{S} = \m{X}\m{X}^\top$.

**Step 2 · Mask and normalize $s_{ij}$ into $\alpha_{ij}(\m{X})$.** Remember that $\v{h}_i$ predicts word $i+1$, so it can only look at positions $1, \dots, i$. If $\v{h}_5$ could look at position $6$, it would just read the answer (`mat`), and when we're generating, word $6$ doesn't exist yet. So just like the zeros above the diagonal of $\m{A}$ earlier, we set $\alpha_{ij}(\m{X}) = 0$ for every $j > i$. This is the **causal mask**. For the positions row $i$ can see, we use softmax so the weights are positive and add up to $1$:

$$\alpha_{ij}(\m{X}) = \frac{e^{s_{ij}}}{\sum_{k \le i} e^{s_{ik}}} \qquad (j \le i)$$

Normalizing each row to $1$ does three things for us. First, every row is on the same scale at every length: whether $i = 5$ or $i = 67$, $\alpha_{i1} + \cdots + \alpha_{ii} = 1$, so $\v{h}_{67}$ stays around the same size as $\v{h}_5$ instead of growing as a sum of $67$ unnormalized terms.

Second, words compete for weight. To get $\alpha_{ij}$, we divide $e^{s_{ij}}$ by the sum of $e^{s_{ik}}$ over every position $k$ that row $i$ can see. So $\alpha_{ij}$ depends on how $s_{ij}$ compares to the other scores in the row. If one word's score goes up, its weight goes up and every other word's weight goes down, since the row still has to add up to $1$.

Third, there are no negative weights. Since $e^{s_{ij}} > 0$, every $\alpha_{ij}(\m{X})$ sits between $0$ and $1$. Unlike Section 4's weights (where $\alpha_1 = -1.16$ could subtract a vector), $\v{h}_i$ blends $\v{x}_1, \dots, \v{x}_i$ and always lands somewhere in between (which comes back in Section 6).

**Step 3 · Aggregate into $\m{H} = \m{A}\m{X}$.** Finally, row $i$ uses its normalized weights $\alpha_{i1}(\m{X}), \dots, \alpha_{ii}(\m{X})$ to take a weighted average of the visible rows, giving its summary vector $\v{h}_i = \sum_{j=1}^i \alpha_{ij}(\m{X})\,\v{x}_j$. Stacking all $n$ rows gives $\m{H} = \m{A}\m{X}$:

{{widget: content-attention-raw}}

Notice how the three matrices in the widget match our three steps ($\m{S} \to \m{A} \to \m{H}$), and none of them are saved parameters: click **“dog on the grass chased”** ($n = 5$), **“a dog chased”** ($n = 3$), or **“the cat sat on the mat”** ($n = 6$), and $\m{S}$, $\m{A}$, and $\m{H}$ recompute and resize on the spot.

Row $i$ of $\m{A}$ gathers context into $\v{h}_i$ to predict word $i+1$, and column $j$ ($j \le i$) is the earlier word at position $j$ being looked back at, so cell $(i, j)$ is its share $\alpha_{ij}(\m{X})$. To predict the next word after an $n$-word prompt, we read the bottom row $\v{h}_n$ via $\v{z} = \m{E}\v{h}_n$. During training, one $6$-word training sentence gives $5$ predictions in one pass (rows $1$ through $5$), all training the same rule $g$.

Because no row of $\m{A}$ is stored, there is no "last row" to run out of when we generate past the training length. Pick word $n+1$ from $\m{E}\v{h}_n$, append it so $\m{X}$ now has $n+1$ rows, and run the same three steps again to get $\v{h}_{n+1}$ and keep generating.

::: details Optional: Scaling the scores before softmax (and why real transformers divide by $\sqrt{d}$)
Suppose we multiply every score by a positive number $c$ before applying softmax, $\operatorname{softmax}(c\, s_{ij})$:
- **$c = 0$:** every score becomes $0$, giving equal shares $\alpha_{ij} = 1/i$, which is our equal-weights average from Section 3.
- **Huge $c$:** whichever position has the largest score takes nearly all of the weight, so $\v{h}_i$ copies that single row.

So $c$ acts like a sharpness dial between averaging every visible word equally and copying the single top-scoring word. In real transformers with large dimension $d$, dot products grow roughly like $\sqrt{d}$, which would push that dial all the way toward copying one word, so they divide scores by $\sqrt{d}$ to keep the shares balanced.
:::

### Queries and keys ($\m{W}_Q$ and $\m{W}_K$)

Look closely at $\m{A}$ in the raw dot-product widget above: in `the cat sat on the`, row $5$ (`the`) gives its biggest shares to row $5$ (`the`, $32.6\%$) and row $1$ (`the`, $27.2\%$), and barely $4.9\%$ to `cat`!

Why? The raw score $s_{ij} = \v{x}_i^\top \v{x}_j$ is big when the two vectors point the same way, so every row is really asking "who looks like me?". The word that looks most like `the` is `the`. It's the same thing that went wrong in Section 3, when we scored every candidate against the last word and our best guess came out as `the cat sat on the the`.

To fill in the blank, the last `the` doesn't need to find more `the`s. It would be a lot more useful if it found `cat` or `sat`. So what if `the` could search in a different direction than the one it sits in? We can do that by putting one matrix between the two vectors:

$$s_{ij} = \v{x}_i^\top \m{W} \v{x}_j$$

Since $\v{x}_i^\top \m{W} = (\m{W}^\top \v{x}_i)^\top$, this moves the arrow for $\v{x}_i$ to $\m{W}^\top \v{x}_i$ first and then takes the same dot product as before. Setting $\m{W} = \m{I}$ gives back the raw score.

**A worked example.** Let's try one $\m{W}$ by hand on row $5$ of `the cat sat on the`: turn the arrow by $90^\circ$ and stretch it by $2$.

$$\m{W} = \begin{bmatrix} 0 & 2 \\ -2 & 0 \end{bmatrix}, \qquad \m{W}^\top \v{x}_5 = \begin{bmatrix} 0 & -2 \\ 2 & 0 \end{bmatrix} \begin{bmatrix} -1.11 \\ 0.29 \end{bmatrix} = \begin{bmatrix} -0.58 \\ -2.22 \end{bmatrix}$$

`the` sits in the upper left of our plane, and `sat` is near the bottom. After the turn, the arrow for `the` points almost straight down, right at `sat`. Now we dot it with each row $j = 1, \dots, 5$ that row $5$ can see:

| $j$ | Word at $j$ | $\v{x}_j$ | $s_{5j} = \v{x}_5^\top \m{W} \v{x}_j$ | $\alpha_{5j}$ | $\alpha_{5j}$ with raw dot products |
|:--:|:--:|:--:|:--:|:--:|:--:|
| $1$ | `the` | $(-0.89, 0.51)$ | $-0.62$ | $3.8\%$ | $27.2\%$ |
| $2$ | `cat` | $(0.75, 0.90)$ | $-2.43$ | $0.6\%$ | $4.9\%$ |
| $3$ | `sat` | $(-0.31, -0.99)$ | $2.38$ | $74.8\%$ | $9.2\%$ |
| $4$ | `on` | $(-1.00, -0.05)$ | $0.69$ | $13.9\%$ | $26.1\%$ |
| $5$ | `the` | $(-1.11, 0.29)$ | $0.00$ | $6.9\%$ | $32.6\%$ |

The share for `sat` goes from $9.2\%$ to $74.8\%$. Now look at the last row, where $s_{55} = 0.00$. Why exactly zero? Turn any arrow by $90^\circ$ and it ends up perpendicular to where it started, so if $\m{R}_{90^\circ}$ is a quarter turn, then $\v{x}^\top \m{R}_{90^\circ} \v{x} = 0$ for every $\v{x}$. Every word's score with itself is zero, so `the` stops finding itself. Row $1$ is also `the`, but its position offset is different, so $\v{x}_1$ isn't quite $\v{x}_5$ and its score comes out to $-0.62$.

To be clear, we picked this turn by hand. Nothing says the best $\m{W}$ is a rotation. It's a $2 \times 2$ matrix, and training is free to set its four numbers to whatever lowers the loss.

A transformer doesn't store $\m{W}$ as one matrix. It writes it as a product of two, $\m{W} = \m{W}_Q \m{W}_K^\top$, and gives each half its own job:

$$\v{q}_i = \m{W}_Q^\top \v{x}_i, \qquad \v{k}_j = \m{W}_K^\top \v{x}_j, \qquad s_{ij} = \v{q}_i^\top \v{k}_j = \v{x}_i^\top \m{W}_Q \m{W}_K^\top \v{x}_j$$

$\v{q}_i$ is the **query**. It's what row $i$ is asking for. $\v{k}_j$ is the **key**. It's how row $j$ advertises itself to whichever row is searching. In our example $\m{W}_Q$ was the turn and $\m{W}_K = \m{I}$, so the turned arrow $(-0.58, -2.22)^\top$ was $\v{q}_5$ and each key was just $\v{x}_j$.

Earlier we stacked the raw attention scores into $\m{S} = \m{X}\m{X}^\top$. We can do the same thing here. The same two matrices act on every row, so let's stack the queries as the rows of a matrix $\m{Q}$ and the keys as the rows of $\m{K}$. Row $i$ of $\m{X}$ is $\v{x}_i^\top$, so row $i$ of $\m{Q}$ is $\v{q}_i^\top = \v{x}_i^\top \m{W}_Q$, and the keys work the same way:

$$\m{Q} = \m{X} \m{W}_Q, \qquad \m{K} = \m{X} \m{W}_K$$

Entry $(i, j)$ of $\m{Q}\m{K}^\top$ is row $i$ of $\m{Q}$ dotted with row $j$ of $\m{K}$, which is $\v{q}_i^\top \v{k}_j = s_{ij}$. So we get every attention score at once:

$$\underbrace{\m{S}}_{n \times n} = \underbrace{\m{Q}}_{n \times d} \; \underbrace{\m{K}^\top}_{d \times n} = \m{X}\m{W}_Q\m{W}_K^\top\m{X}^\top$$

Rows $1$ through $4$ do the same thing with their own queries $\v{q}_1, \dots, \v{q}_4$ against the keys they can see, which fills in the rest of $\m{S}$.

Does splitting $\m{W}$ in two buy us anything? At $d = 2$, no. $\m{W}_Q$ and $\m{W}_K$ hold $8$ numbers between them, but the score only ever sees their product, which has $4$. The split starts to matter once queries and keys are smaller than the embedding, which is how multi-head attention works. We haven't gotten there yet.

Notice why those $8$ fixed parameters succeed where Section 4's $5$ slot parameters broke: once trained, $\m{W}_Q$ and $\m{W}_K$ are still fixed numbers, they just act on the **$d = 2$ embedding coordinates** instead of the **slot numbers**. When the words move slots or the sentence length changes, each word carries its embedding coordinates with it, so the importance bars move right along with the words:

{{widget: qk-circuit-lookup}}

::: details Optional: Expanding $s_{ij}$ and how 8 numbers cover all $11 \times 11$ word pairs (the QK circuit)
Substitute $\v{x}_i = \v{e}_{w_i} + \v{p}_i$ and $\v{x}_j = \v{e}_{w_j} + \v{p}_j$ directly into $s_{ij} = \v{x}_i^\top \m{W}_Q \m{W}_K^\top \v{x}_j$ and expand:

$$\begin{aligned}
s_{ij} &= (\v{e}_{w_i} + \v{p}_i)^\top \m{W}_Q \m{W}_K^\top (\v{e}_{w_j} + \v{p}_j) \\
&= \underbrace{\v{e}_{w_i}^\top \m{W}_Q \m{W}_K^\top \v{e}_{w_j}}_{\text{word–word score}}
\;+\; \underbrace{\v{e}_{w_i}^\top \m{W}_Q \m{W}_K^\top \v{p}_j + \v{p}_i^\top \m{W}_Q \m{W}_K^\top \v{e}_{w_j} + \v{p}_i^\top \m{W}_Q \m{W}_K^\top \v{p}_j}_{\text{position terms (small, since }\lVert \v{p}\rVert \text{ is small)}}
\end{aligned}$$

Because $\lVert\v{p}\rVert$ is small compared to the word embeddings, $\v{p}_i$ and $\v{p}_j$ are nearly $\v{0}$ relative to $\v{e}_{w_i}$ and $\v{e}_{w_j}$. Canceling those three small position terms for a moment leaves just the word-to-word backbone:

$$s_{ij} \approx \v{e}_{w_i}^\top \m{W}_Q \m{W}_K^\top \v{e}_{w_j}$$

Evaluating that formula across all $11$ possible query words $u$ and all $11$ possible key words $v$ produces the $11 \times 11$ compatibility table $\m{E}\m{W}_Q\m{W}_K^\top\m{E}^\top$ (often called the **QK circuit**).

How can $8$ numbers in $\m{W}_Q$ and $\m{W}_K$ govern all $11^2 = 121$ entries of that table? In fact, $\m{W}_Q\m{W}_K^\top$ is a $2 \times 2$ matrix, so it really has only **$4$ degrees of freedom**, forcing $\m{E}\m{W}_Q\m{W}_K^\top\m{E}^\top$ to have **rank at most $2$**. That means it *cannot* store $121$ unrelated numbers, and that's okay because our vocabulary isn't $121$ unrelated pairs. Words with similar grammatical roles already cluster together in $\m{E}$ (`cat`, `dog`, `bird` in the top-right; `sat`, `chased` at the bottom), so one $2 \times 2$ linear rule applies to whole clusters at once.

Adding the three $\v{p}$ terms back in then shifts those baseline word–word scores slightly by slot, breaking ties between duplicate words (like `the` at slot $1$ vs. slot $5$). The real limit of $d = 2$ is that a single query arrow $\v{q}_i \in \mathbb R^2$ can only point in one direction at a time. It can't simultaneously pull heavily from *both* the verb (`sat`) and the animal (`cat`), which is what will force us to introduce **multi-head attention** later.
:::

Setting $\m{W}_Q = \m{W}_K = \m{I}$ recovers raw dot products $\m{X} \m{X}^\top$. To try other matrices, the explorer below keeps $\m{W}_K = \m{I}$ and lets you steer $\m{W}_Q$ by hand with a turn angle $\theta$ and stretch $\beta$. Our worked example is $\theta = 90^\circ$, $\beta = 2$. Drag $\beta$ or $\theta$ (which also updates the comparison bars above!) and click any cell in $\m{S}$ or $\m{A}$ to inspect its query-key geometry:

{{widget: qk-explorer}}

::: callout
**Two softmaxes, two different jobs.**
- **Attention softmax** runs across positions $1, \dots, i$. Its output (a row of $\m{A}$) is a set of mixing weights telling row $i$ *where to read*; nothing is ever sampled from it.
- **Readout softmax** (Section 3) runs across the $11$ vocabulary logits $\v{z}$. Its output is a distribution over $\mathcal V$ deciding *which word comes next*.
:::

What about word order? Swapping `cat` and `sat` changes $\v{x}_2$ and $\v{x}_3$, so their queries and keys change and the weights change with them, and we saw in Section 3 that unequal weights are enough to tell the two orders apart. Position offsets plus input-dependent weights keep word order intact.

::: details Optional: Putting position inside the score (RoPE)
Why did we put positions on a circle $\v{p}_j = r(\cos j\omega, \sin j\omega)^\top$ back in Section 2? If we expand $s_{ij}$ with $\v{x}_i = \v{e}_{w_i} + \v{p}_i$ and $\v{x}_j = \v{e}_{w_j} + \v{p}_j$, one of the terms only involves positions. With $\m{W}_Q = \m{W}_K = \m{I}$ it's

$$\v{p}_i^\top \v{p}_j = r^2\big(\cos i\omega\cos j\omega + \sin i\omega\sin j\omega\big) = r^2\cos\!\big((i - j)\omega\big)$$

By the trig angle-difference identity, $i$ and $j$ only show up as $i - j$, how many words apart they are. And if $\m{W}_Q$ turns by $\theta = 3\omega$, then $\v{p}_i^\top \m{W}_Q \v{p}_j = r^2\cos((i - j)\omega - 3\omega)$ peaks at $i - j = 3$, whether you're at positions $(5, 2)$ or $(100, 97)$. That's a rule like "look 3 words back", stored once in $\m{W}_Q$ instead of once per position pair.

The problem is the other two position terms, $\v{e}_{w_i}^\top \m{W}_Q \m{W}_K^\top \v{p}_j$ and $\v{p}_i^\top \m{W}_Q \m{W}_K^\top \v{e}_{w_j}$. Each one mixes a word with a single position, so they depend on $j$ or $i$ by itself, and they get in the way of the clean $i - j$ term.

Most modern LLMs (Llama, for example) use **RoPE** (rotary position embeddings) to get rid of them. RoPE leaves the word embedding alone and rotates the query and the key by an angle that depends on their position, using the rotation $\m{R}$ from Section 2 applied $i$ times, $\m{R}_i = \m{R}^i$:

$$\v{q}_i = \m{R}_i \m{W}_Q^\top \v{e}_{w_i}, \qquad \v{k}_j = \m{R}_j \m{W}_K^\top \v{e}_{w_j}$$

Then the score is

$$s_{ij} = \v{q}_i^\top \v{k}_j = \v{e}_{w_i}^\top \m{W}_Q \, \m{R}_i^\top \m{R}_j \, \m{W}_K^\top \v{e}_{w_j} = \v{e}_{w_i}^\top \m{W}_Q \, \m{R}_{j - i} \, \m{W}_K^\top \v{e}_{w_j}$$

Rotating back by $i\omega$ and then forward by $j\omega$ is the same as rotating by $(j - i)\omega$, so $\m{R}_i^\top \m{R}_j = \m{R}_{j-i}$. Now each score depends on exactly two things: which two words they are, and how far apart they are. There are no cross-terms, and since a rotation never changes a vector's length, position can't drown out the word like adding the index did in Section 2. Here position really is a linear map $\m{R}_i$ acting on the query and key.

Our widgets keep adding $\v{p}_i$ because it's easier to draw in 2D, but everything else in this section works the same with RoPE.
:::

## 6 · What should a row send?

OK, so here's where we are. We have five words, `the cat sat on the`, and row $5$ has to fill in the blank. With the query we picked by hand in Section 5, it puts about $75\%$ of its weight on `sat`. So what does our model predict?

To find out, we need logits. In Section 3, the logit for one candidate word $v$ was

$$z_v = \v{e}_v^\top \v{h}, \qquad \v{h} = \alpha_1 \v{x}_1 + \alpha_2 \v{x}_2 + \cdots + \alpha_n \v{x}_n$$

Back then we had one blend $\v{h}$ for the whole prompt, and its weights were the same for every sentence. With attention, every row $i$ gets its own blend $\v{h}_i$, and the weights depend on the words in the sentence. The blend that fills in our blank is $\v{h}_5$. Its weights are row $5$ of $\m{A}$, which is why they have two subscripts:

$$z_v = \v{e}_v^\top \v{h}_5, \qquad \v{h}_5 = \alpha_{51} \v{x}_1 + \alpha_{52} \v{x}_2 + \alpha_{53} \v{x}_3 + \alpha_{54} \v{x}_4 + \alpha_{55} \v{x}_5$$

Keep in mind that the $\alpha_{5j}$ aren't trained parameters anymore. What gets trained is $\m{W}_Q$ and $\m{W}_K$, and the weights come out of them fresh for every sentence. First come the attention scores $s_{ij}$, then softmax:

$$\alpha_{ij} = \frac{e^{s_{ij}}}{\sum_{k \le i} e^{s_{ik}}}, \qquad s_{ij} = \v{x}_i^\top \m{W}_Q \m{W}_K^\top \v{x}_j$$

A different five words would give us different weights. For our five, the worked example in Section 5 gave us

$$\v{h}_5 = 0.038\,\v{x}_1 + 0.006\,\v{x}_2 + 0.748\,\v{x}_3 + 0.139\,\v{x}_4 + 0.069\,\v{x}_5 = (-0.48, -0.70)$$

So for this sentence, our blend is about $75\%$ `sat`.

Now we can finally compute all eleven logits with $\v{z} = \m{E}\v{h}_5$ and apply softmax:

| Ending | `sat` | `chased` | `on` | `the` | `hat` | `grass` | `mat` |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Probability | $19.2\%$ | $18.3\%$ | $12.5\%$ | $9.8\%$ | $9.4\%$ | $7.5\%$ | $6.9\%$ |

Our best guess is `the cat sat on the sat`. How wrong is that? Let's go back to our data. After `the cat sat on the`, we saw `mat` twice, `hat` once and `grass` once. Section 4 gave us a number for how far a prediction is from those endings, the loss $L$. Our model gives the three of them $6.86\%$, $9.43\%$ and $7.52\%$, so

$$L(\mu) = -\big(0.50 \ln 0.0686 + 0.25 \ln 0.0943 + 0.25 \ln 0.0752\big) \approx 2.577$$

A model that gives all eleven words the same chance gets $\ln 11 \approx 2.398$. So we're doing worse than a model that knows nothing.

What went wrong? Let's distribute $\v{e}_v$ across the sum inside $\v{h}_5$:

$$z_v = \alpha_{51}\,\v{e}_v^\top\v{x}_1 + \alpha_{52}\,\v{e}_v^\top\v{x}_2 + \alpha_{53}\,\v{e}_v^\top\v{x}_3 + \alpha_{54}\,\v{e}_v^\top\v{x}_4 + \alpha_{55}\,\v{e}_v^\top\v{x}_5$$

Every term is still a candidate dotted against one of the words in our prompt. Attention only changed how much each term counts. Add the terms back up and the logit asks how much $v$ looks like our blend $\v{h}_5$, the same as it did in Section 3. Remember our very first predictor there? It used $\v{h} = \v{x}_5$, a blend that's $100\%$ the last word, and it predicted `the`. Now our blend is mostly `sat`, and the two words that look most like it are `sat` and `chased`.

We picked our query by hand, though. Could better weights fix this? Let's find out. We'll leave the embeddings in $\m{E}$ exactly where Section 2 put them, with similar words close together, and ask what the best possible weights could do.

Look at `mat` and `the` in our table. `mat` filled the blank in two of our four sentences and `the` never did, but our model gives `the` $9.8\%$ and `mat` only $6.9\%$. How far ahead of `the` could better weights put `mat`? We saw in Section 3 that only the difference between two logits matters, $\mu(\code{mat}) / \mu(\code{the}) = e^{z_{\text{mat}} - z_{\text{the}}}$. So let's subtract the two logits term by term:

$$z_{\text{mat}} - z_{\text{the}} = \sum_{j=1}^{5} \alpha_{5j}\,(\v{e}_{\text{mat}} - \v{e}_{\text{the}})^\top \v{x}_j$$

Here $\v{e}_{\text{mat}} - \v{e}_{\text{the}} = (2.00, -0.85)^\top$. Let's dot it with each of our five rows:

| $j$ | Word at $j$ | $\v{x}_j$ | $(\v{e}_{\text{mat}} - \v{e}_{\text{the}})^\top \v{x}_j$ |
|:--:|:--:|:--:|:--:|
| $1$ | `the` | $(-0.89, 0.51)$ | $-2.2135$ |
| $2$ | `cat` | $(0.75, 0.90)$ | $0.7350$ |
| $3$ | `sat` | $(-0.31, -0.99)$ | $0.2215$ |
| $4$ | `on` | $(-1.00, -0.05)$ | $-1.9575$ |
| $5$ | `the` | $(-1.11, 0.29)$ | $-2.4665$ |

For `cat`, that's $(2.00)(0.75) + (-0.85)(0.90) = 0.735$. Softmax weights are always positive and add up to $1$, so $z_{\text{mat}} - z_{\text{the}}$ is a weighted average of these five numbers, and an average can never be bigger than the biggest number in it. That's the $0.735$ from `cat`. So whatever $\m{W}_Q$ and $\m{W}_K$ we choose, `mat` can be at most $e^{0.735} \approx 2.09$ times as likely as `the`. To get even that close we'd have to put all of our weight on `cat`, and then our top three guesses would be `bird`, `cat` and `dog`.

The loss tells the same story. No set of weights gets $L$ below $2.207$. The best ones put about $40\%$ on `cat` and $60\%$ on `sat`, and that only just beats the uniform guess at $2.398$.

So better weights can't fix this. It's the catch we flagged in Section 5: a blend always lands somewhere in between the rows it's made of.

What else is there to change? Look at one term, $\alpha_{5j}\,\v{e}_v^\top\v{x}_j$. The weight can't save us, and we're leaving the embeddings where they are. What's left is the dot product itself.

We've been here before. In Section 5, the raw attention score was a plain dot product too, and it had `the` looking for more `the`s, because a plain dot product is big when two vectors point the same way. We fixed that by putting a matrix between the two vectors, so $\v{x}_i^\top\v{x}_j$ became $\v{x}_i^\top\m{W}\v{x}_j$. Let's try the same thing here:

$$\v{e}_v^\top\v{x}_j \quad\longrightarrow\quad \v{e}_v^\top\m{W}_V^\top\v{x}_j$$

Just like before, this moves the arrow for $\v{x}_j$ to $\m{W}_V^\top\v{x}_j$ first and then takes the same dot product. Up to now we've really been using $\m{W}_V = \m{I}$ without saying so. Putting $\m{W}_V$ into every term gives us

$$z_v = \v{e}_v^\top \underbrace{(\alpha_{51} \m{W}_V^\top\v{x}_1 + \alpha_{52} \m{W}_V^\top\v{x}_2 + \alpha_{53} \m{W}_V^\top\v{x}_3 + \alpha_{54} \m{W}_V^\top\v{x}_4 + \alpha_{55} \m{W}_V^\top\v{x}_5)}_{\v{h}_5}$$

We call $\v{v}_j = \m{W}_V^\top \v{x}_j$ the **value** of row $j$. It's what row $j$ adds to the blend in place of $\v{x}_j$:

$$\v{v}_j = \m{W}_V^\top \v{x}_j, \qquad \v{h}_i = \sum_{j \le i} \alpha_{ij}\v{v}_j$$

**A worked example.** Which matrix should we try? We don't know, so let's start with the one we already have. Here's the quarter turn from Section 5, without the stretch:

$$\m{W}_V^\top = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}, \qquad \v{v}_3 = \m{W}_V^\top \v{x}_3 = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} -0.31 \\ -0.99 \end{bmatrix} = \begin{bmatrix} 0.99 \\ -0.31 \end{bmatrix}$$

The other four words get the same turn. Just like in Section 3, we can lay the five weighted steps tip to tail and see where they end up. Click through the words to build $\v{h}_5$ one step at a time, with $\alpha_{5j}\v{x}_j$ on the left and $\alpha_{5j}\v{v}_j$ on the right:

{{widget: summary-turn}}

Every step is the same length on both sides, because the weights haven't changed. Row $5$ still gives $74.8\%$ to `sat`. Each step just points a quarter turn away from where it did, so the whole path does too, and we end up at $(0.70, -0.48)$ and not $(-0.48, -0.70)$. Here's the readout now:

| Ending | `grass` | `hat` | `mat` | `chased` | `dog` | `sat` | `cat` |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Probability | $20.2\%$ | $15.6\%$ | $14.7\%$ | $10.8\%$ | $9.7\%$ | $8.6\%$ | $7.0\%$ |

The top three are exactly the three endings we saw, and `sat` dropped from first to sixth. `mat` is now $6.12$ times as likely as `the`, well past the $2.09$ that no weights could reach. Our loss went from $2.577$ down to $1.821$, below the $2.207$ that was the best the weights alone could do.

Did we just get lucky with that turn? Let's try the other quarter turns and see:

| Turn | $\v{h}_5$ | Top three guesses | Loss $L$ |
|:--:|:--:|:--:|:--:|
| $0^\circ$ ($\m{W}_V = \m{I}$) | $(-0.48, -0.70)$ | `sat`, `chased`, `on` | $2.577$ |
| $90^\circ$ | $(0.70, -0.48)$ | `grass`, `hat`, `mat` | $1.821$ |
| $180^\circ$ | $(0.48, 0.70)$ | `bird`, `cat`, `dog` | $2.586$ |
| $270^\circ$ | $(-0.70, 0.48)$ | `the`, `a`, `on` | $3.451$ |

So yes, a little lucky. Each turn points our blend at a different kind of word: the verbs, then the places to sit, then the animals, then the small words. Our blend is $75\%$ `sat` in all four rows. What changes is what reading `sat` makes us predict, and only the $90^\circ$ turn beats the uniform guess at $2.398$. So we don't get to pick the right matrix. The data does. $\m{W}_V$ is four more numbers that training can adjust to lower $L$, the same way it adjusted the $\alpha$'s in Section 4, and the result doesn't have to be a turn.

So each row's vector now gets used in two separate ways. Its key decides how much weight the row gets, and its value is what gets added into the sum. Before $\m{W}_V$, the thing added into the sum was just $\v{x}_j$ itself.

Stack the values as rows and the whole calculation becomes

$$\m{V} = \m{X}\m{W}_V, \qquad \m{H} = \m{A}\m{V}$$

$\m{A}$ is the same $\m{A}$ we built in Section 5. So $\m{W}_Q$ and $\m{W}_K$ decide where row $i$ reads, and $\m{W}_V$ decides what it gets from there.

We've just built one **causal self-attention head**. “Self” means that the queries, keys, and values all come from the same sequence. “Causal” means that row $i$ can read through position $i$, so the word it's trying to predict stays out of view.

The widget below starts at our quarter turn. Try turning the value vector yourself. The weight on `sat` stays where it was. What changes is the direction `sat` pulls $\v{h}_5$, and you can see which candidate words that favors when we read out $\m{E}\v{h}_5$.

{{widget: value-explorer}}

### What does the $74.8\%$ mean?

Suppose we draw a position $J$ from the five words in `the cat sat on the`, using row $5$ of $\m{A}$ as our probabilities. At the widget's starting settings we'd draw position $3$ about three quarters of the time. Whenever we draw it, we get the value vector for `sat`. What vector would we get on average?

$$\Pr(J=j)=\alpha_{5j}, \qquad \mathbb E[\v{v}_J]=\sum_{j=1}^{5}\alpha_{5j}\v{v}_j=\v{h}_5$$

That's the sum we just computed. The draw is a way to think about it. The head gets the average directly from all five values.

::: details Optional: From a distribution on positions to a distribution on vectors
Row $5$ gives us a probability measure on the positions $1,\ldots,5$. Each position $j$ maps to a value $\v{v}_j$. What's the probability of getting a vector in some region $B$ of our plane? Just like we did for the event “`mat` or `hat`” in Section 1, add the probabilities of all the outcomes that qualify:

$$\Pr(\v{v}_J\in B)=\sum_{j:\,\v{v}_j\in B}\alpha_{5j}$$

This distribution on vectors is called the **pushforward** of the distribution on positions. Two positions that send exactly the same value put their combined weight on one point. The head keeps the mean of this distribution. Different distributions can have the same mean, so knowing $\v{h}_5$ alone needn't tell us which positions supplied it.

Vuckovic, Baratin, and Tachet des Combes call replacing the distribution with a point mass at its mean a **moment projection**. The query and key projections in Section 5 are linear maps acting on individual vectors. [A Mathematical Theory of Attention, Definition 8 and Proposition 9](https://arxiv.org/pdf/2007.02876#page=5)
:::

### How much does reading `sat` help `mat`?

Take the dot product of $\v{e}_{\mathrm{mat}}$ with the sum we just built. We can distribute it across the five terms and pick out the one from position $3$:

$$z_{\mathrm{mat}}=\v{e}_{\mathrm{mat}}^\top\v{h}_5=\sum_{j=1}^5\alpha_{5j}\,\v{e}_{\mathrm{mat}}^\top\v{v}_j$$

With our quarter turn, $\v{v}_3 = (0.99,-0.31)^\top$ and $\v{e}_{\mathrm{mat}} = (1.00,-0.45)^\top$, so position $3$ contributes

$$\alpha_{53}\,\v{e}_{\mathrm{mat}}^\top\v{v}_3 \approx 0.7484\big[(1.00)(0.99)+(-0.45)(-0.31)\big]\approx +0.845$$

Adding the other four terms brings the logit for `mat` to $0.918$. Softmax across the eleven endings turns that into the $14.7\%$ from our table. So we can follow one path through the head all the way to the blank: the query and key determine how much row $5$ reads from `sat`, and the value and readout determine how that term affects the logit for `mat`.

What if we turn the value stretch all the way down to zero? Every word now sends the zero vector. Row $5$ still has its original input $\v{x}_5$ available at the start of the calculation, but by the time we read out $\v{h}_5$, all we have is zero. Every logit is zero too, so all eleven endings get the same probability. If attention has nothing to add, shouldn't we at least get to keep the word we started with?

## 7 · Keeping what we started with

Let's answer that. With the value stretch at zero, $\v{h}_5 = \v{0}$ and every ending gets $1/11 \approx 9.1\%$. The row has forgotten everything, even its own word $\v{x}_5$.

Why did we lose $\v{x}_5$? Because $\m{H} = \m{A}\m{V}$ replaces each row with what it gathered. The only way row $5$'s own word gets through is the weight it puts on itself, $\alpha_{55}$. With our $90^\circ$ query that was $6.9\%$, and even that share arrives as $\v{v}_5$ and not $\v{x}_5$.

So what if we add what attention gathered to the row, instead of replacing the row with it?

$$\v{h}_i = \v{x}_i + \sum_{j \le i} \alpha_{ij}\v{v}_j, \qquad \m{H} = \m{X} + \m{A}\m{V}$$

Try $\m{W}_V = \m{0}$ again. Every value is still zero, but now $\v{h}_5 = \v{x}_5$, and reading it out gives $\m{E}\v{x}_5$. That's exactly the Section 3 predictor, the one that guessed `the` at $26.8\%$. That wasn't a good predictor, but now each row starts from its own word and attention only has to add a correction on top of it. The logits split the same way:

$$\v{z} = \m{E}\v{h}_n = \underbrace{\m{E}\v{x}_n}_{\text{Section 3}} + \underbrace{\m{E}\sum_{j \le n}\alpha_{nj}\v{v}_j}_{\text{what attention adds}}$$

Keeping the input and adding to it is called a **residual connection**. The widgets above show $\m{H} = \m{A}\m{V}$ without the added $\m{X}$.

$\m{H}$ still has the same shape as $\m{X}$, so nothing stops us from feeding it into another layer just like this one. In a follow-up post we'll stack layers, then reuse one layer for every step, loop it, and ask whether the loop settles.
