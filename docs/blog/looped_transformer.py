# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="A looped transformer on eleven words",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np

    np.set_printoptions(precision=3, suppress=True)
    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A looped transformer on eleven words

    Suppose only eleven words exist, and the only text ever written is five short sentences. This notebook starts
    there and asks one question: given the words so far, how likely is each word to come next?

    **Part I** answers by construction. We build the simplest thing that could work, find exactly how it fails, and
    make the smallest change to the mathematics that repairs it. After about ten such repairs, the thing on the page
    is a transformer. **Part II** treats that transformer as an object of study: what its internal vectors contain,
    what happens when one layer is applied again and again, and how to test whether something inside the model
    causes what comes out. **Part III** reports the same questions asked of two trained looped models, Huginn and
    Recurrent-Llama, which is what the experiments in this repository measure.

    Every number below is computed by the code in this notebook from the context you type. Vectors have two
    coordinates, so every matrix fits on the screen and can be checked with pencil and paper. Three kinds of numbers
    appear, and they are kept apart:

    - **Hand-set** numbers are starting values chosen by a person: the embedding table, the attention matrices, the
      MLP. Nothing learned them. Most can be changed with sliders.
    - **Trained** numbers come out of an optimization that runs here: the table in section 4, the probes in Part II.
    - **Measured** numbers come from experiments on trained models. They appear in Part III.
    """)
    return


@app.cell(hide_code=True)
def _(E, VOCAB, mo, np):
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure

    INK, MUTED, RULE, SURFACE = "#0b0b0b", "#52514e", "#dddcd6", "#fcfcfb"
    BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
    SEQ = LinearSegmentedColormap.from_list("seq", [SURFACE, "#cde2fb", "#6da7ec", "#2a78d6", "#104281"])

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 120,
            "font.size": 10,
            "text.color": INK,
            "axes.edgecolor": RULE,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
        }
    )

    LABEL_SHIFT = {
        "the": (-0.08, 0.05, "right"),
        "a": (-0.08, 0.05, "right"),
        "on": (-0.08, 0.0, "right"),
        "hat": (-0.08, -0.03, "right"),
        "sat": (0.0, 0.13, "center"),
        "bird": (-0.08, 0.05, "right"),
        "chased": (0.08, -0.04, "left"),
    }

    # ---- figures ----------------------------------------------------------------------------------------------

    def new_fig(ncols=1, width=9.0, height=3.6, **kw):
        """A figure that is not registered with pyplot, so redrawing never piles figures up."""
        fig = Figure(figsize=(width, height), layout="constrained")
        return fig, fig.subplots(1, ncols, **kw)

    def plane(ax, table=None, lim=1.7, labels=True, alpha=1.0, skip=()):
        """The two embedding coordinates as a plane, with the eleven words marked."""
        table = E if table is None else table
        ax.axhline(0, color=RULE, lw=0.8, zorder=0)
        ax.axvline(0, color=RULE, lw=0.8, zorder=0)
        ax.scatter(table[:, 0], table[:, 1], s=22, color=MUTED, alpha=alpha, linewidths=0, zorder=2)
        if labels:
            for word, (x, y) in zip(VOCAB, table):
                if word in skip:
                    continue
                dx, dy, ha = LABEL_SHIFT.get(word, (0.08, 0.05, "left"))
                scale = lim / 1.7
                ax.text(x + dx * scale, y + dy * scale, word, fontsize=8.5, color=MUTED, ha=ha, va="center",
                        alpha=alpha)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ticks = [-1, 0, 1] if lim < 2.2 else [-int(lim * 0.75), 0, int(lim * 0.75)]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(length=0, labelsize=8)
        for side in ax.spines.values():
            side.set_visible(False)

    def arrow(ax, start, end, color=INK, lw=1.8, head=11, **kw):
        ax.annotate(
            "",
            xy=tuple(end),
            xytext=tuple(start),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=head),
            **kw,
        )

    def hbars(ax, values, title=None, color=BLUE, bad=None, fmt="{:.0%}", ticklabels=None):
        """One horizontal bar per vocabulary token, in vocabulary order."""
        values = np.asarray(values, dtype=float)
        shown = np.nan_to_num(values)
        y = np.arange(len(VOCAB))[::-1]
        colors = [RED if (bad is not None and bad[k]) else color for k in range(len(VOCAB))]
        ax.barh(y, shown, height=0.66, color=colors)
        span = max(np.abs(shown).max(), 1e-9)
        for yy, v, raw in zip(y, shown, values):
            label = fmt.format(raw) if np.isfinite(raw) else "undefined"
            ax.text(v + 0.02 * span * (1 if v >= 0 else -1), yy, label, va="center",
                    ha="left" if v >= 0 else "right", fontsize=8.5, color=MUTED)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(y, [""] * len(VOCAB) if ticklabels is False else (ticklabels or VOCAB))
        ax.set_xticks([])
        ax.tick_params(length=0)
        ax.margins(x=0.34 if shown.min() < 0 else 0.2)
        for side in ax.spines.values():
            side.set_visible(False)
        if title:
            ax.set_title(title)

    def end_label(ax, x, y, label, color=INK, dx=6, dy=0):
        ax.annotate(label, xy=(x, y), xytext=(dx, dy), textcoords="offset points", va="center", fontsize=9,
                    color=color)

    def head_picture(ax, X, K, Q, A, update, labels, row=-1, title=None):
        """One head, seen from one row: its query direction, the weight it puts on each key, the update it gets."""
        lim = max(1.7, float(np.abs(np.vstack([X, K, X + update])).max()) + 0.3)
        plane(ax, lim=lim, labels=False, alpha=0.22)
        q = Q[row]
        if np.linalg.norm(q) > 1e-9:
            arrow(ax, (0, 0), 0.8 * lim * q / np.linalg.norm(q), color=ORANGE, lw=1.6)
        ax.scatter(K[:, 0], K[:, 1], s=30 + 420 * A[row], color=BLUE, alpha=0.85, edgecolors=SURFACE,
                   linewidths=1.5, zorder=3)
        me = row % len(X)
        for k in range(len(X)):
            ax.text(K[k, 0], K[k, 1] + 0.1 * lim, labels[k], ha="center", fontsize=8,
                    color=INK if k == me else MUTED, fontweight="bold" if k == me else "normal")
        arrow(ax, X[row], X[row] + update[row], color=INK, lw=1.8, head=9, zorder=5)
        if title:
            ax.set_title(title)

    # ---- numbers on the page ----------------------------------------------------------------------------------

    _CSS = (
        "<style>"
        ".lt-eq{display:flex;flex-wrap:wrap;align-items:center;gap:10px 12px;margin:.55rem 0;"
        "font-variant-numeric:tabular-nums}"
        ".lt-m{display:inline-flex;flex-direction:column;align-items:center;gap:4px}"
        ".lt-cap{font-size:.8rem;opacity:.85;white-space:nowrap}"
        ".lt-cap i{font-family:'KaTeX_Math','Times New Roman',serif;font-size:.95rem}"
        ".lt-cap small{opacity:.7;margin-left:4px}"
        ".lt-row{display:inline-flex;align-items:flex-start}"
        "table.lt-t{border-collapse:collapse!important;font-size:.84rem;margin:0!important;width:auto!important;"
        "border:none!important}"
        "table.lt-t td,table.lt-t th{height:22px;line-height:22px;padding:0 7px!important;border:none!important;"
        "text-align:right;white-space:nowrap;background:none}"
        "table.lt-t th{font-weight:400;font-size:.74rem;opacity:.7}"
        "table.lt-t tr{border:none!important;background:none!important}"
        "table.lt-t tr.lt-rule th{border-bottom:1px solid currentColor!important}"
        "table.lt-t tr.lt-foot td{border-top:1px solid currentColor!important;opacity:.85}"
        "table.lt-t td.lt-l,table.lt-t th.lt-l{text-align:left}"
        "table.lt-lab td{font-size:.74rem;opacity:.7;padding:0 6px 0 0!important}"
        ".lt-b{position:relative;padding:0 5px}"
        ".lt-b:before,.lt-b:after{content:'';position:absolute;top:var(--top,0px);bottom:0;width:5px;"
        "border:1.5px solid currentColor;opacity:.65}"
        ".lt-b:before{left:0;border-right:none}.lt-b:after{right:0;border-left:none}"
        "table.lt-t td.lt-hi{background:rgba(42,120,214,.22)}"
        "table.lt-t td.lt-hi2{background:rgba(235,104,52,.26)}"
        "table.lt-t td.lt-dim{opacity:.3}"
        ".lt-op{font-size:1.05rem;padding:0 2px}"
        ".lt-pc{display:flex;align-items:center;gap:6px;min-width:118px}"
        ".lt-pc span:first-child{width:3.3em;text-align:right}"
        ".lt-bar{display:inline-block;height:9px;background:currentColor;opacity:.4;border-radius:1px}"
        "</style>"
    )

    def _txt(v, fmt="{:.2f}"):
        if isinstance(v, str):
            return v
        if v is None or not np.isfinite(v):
            return "·"
        s = fmt.format(v)
        if s.strip("-+0.%") == "":
            s = s.replace("-", "").replace("+", "")
        return s.replace("-", "−")

    def num(v, fmt="{:.2f}"):
        """A number as text, with a real minus sign."""
        return _txt(float(v), fmt)

    def vec(v, fmt="{:.2f}"):
        """A short vector as text: (a, b)."""
        return "(" + ", ".join(_txt(float(t), fmt) for t in v) + ")"

    def pct_cell(p):
        """A probability as a percentage with a small bar."""
        return (f'<div class="lt-pc"><span>{_txt(float(p), "{:.1%}")}</span>'
                f'<span class="lt-bar" style="width:{max(0.0, 70 * float(p)):.1f}px"></span></div>')

    def pct_plain(text):
        """Text aligned like the number of a pct_cell, for a totals row."""
        return f'<div class="lt-pc"><span>{text}</span></div>'

    def grid(M, rows=None, cols=None, fmt="{:.2f}", mark=None, mark2=None, dim=None, name=None, bracket=True,
             shape=True, left=(), foot=None):
        """A matrix as an HTML table: optional row and column labels, optional highlighted cells."""
        M = [list(r) for r in (np.atleast_2d(M) if isinstance(M, np.ndarray) else M)]
        nr, nc = len(M), len(M[0])
        fmts = fmt if isinstance(fmt, (list, tuple)) else [fmt] * nc

        def has(mask, r, c):
            return mask is not None and bool(np.asarray(mask)[r, c])

        body = ""
        if cols is not None:
            body += ('<tr class="' + ("" if bracket else "lt-rule") + '">'
                     + "".join(f'<th class="{"lt-l" if c in left else ""}">{lab}</th>' for c, lab in enumerate(cols))
                     + "</tr>")
        for r in range(nr):
            body += "<tr>"
            for c in range(nc):
                cls = "lt-hi" if has(mark, r, c) else "lt-hi2" if has(mark2, r, c) else ""
                if has(dim, r, c):
                    cls += " lt-dim"
                if c in left:
                    cls += " lt-l"
                body += f'<td class="{cls.strip()}">{_txt(M[r][c], fmts[c])}</td>'
            body += "</tr>"
        if foot is not None:
            body += ('<tr class="lt-foot">'
                     + "".join(f'<td class="{"lt-l" if c in left else ""}">{t}</td>' for c, t in enumerate(foot))
                     + "</tr>")
        table = f'<table class="lt-t">{body}</table>'
        if bracket:
            table = f'<div class="lt-b" style="--top:{22 if cols is not None else 0}px">{table}</div>'
        if rows is not None:
            lab = ("<tr><td>&nbsp;</td></tr>" if cols is not None else "") + "".join(
                f"<tr><td>{r}</td></tr>" for r in rows
            )
            table = f'<div class="lt-row"><table class="lt-t lt-lab">{lab}</table>{table}</div>'
        cap = ""
        if name:
            cap = f'<div class="lt-cap">{name}' + (f"<small>{nr}×{nc}</small>" if shape and bracket else "") + "</div>"
        return f'<div class="lt-m">{cap}{table}</div>'

    def show(*parts):
        """Lay matrices and symbols out in one row, like an equation."""
        html = "".join(p if p.lstrip().startswith("<") else f'<span class="lt-op">{p}</span>' for p in parts)
        return mo.Html(_CSS + f'<div class="lt-eq">{html}</div>')

    def vocab_table(columns, name=None, mark=None, foot=None):
        """One row per vocabulary word. columns: (header, values, kind); kind is 'pct', 'vec', 'text' or a format."""
        header = ["word"] + [c[0] for c in columns]
        body = []
        for v, word in enumerate(VOCAB):
            row = [word]
            for _, values, kind in columns:
                x = values[v]
                if kind == "pct":
                    row.append(pct_cell(x))
                elif kind == "vec":
                    row.append(vec(x))
                elif kind == "text":
                    row.append(str(x))
                else:
                    row.append(_txt(float(x), kind))
            body.append(row)
        return grid(body, cols=header, bracket=False, name=name, mark=mark, left=(0,), foot=foot)

    def top_words(p, k=3):
        """The k most probable next words, as text."""
        return ", ".join(f"{VOCAB[j]} {p[j]:.0%}" for j in np.argsort(-np.asarray(p))[:k])

    def last_row_mask(shape):
        """A highlight mask that marks the last row of a matrix."""
        mask = np.zeros(shape, dtype=bool)
        mask[-1] = True
        return mask

    return (
        AQUA,
        BLUE,
        INK,
        LABEL_SHIFT,
        MUTED,
        ORANGE,
        RULE,
        SEQ,
        SURFACE,
        arrow,
        end_label,
        grid,
        hbars,
        head_picture,
        last_row_mask,
        mpl,
        new_fig,
        num,
        pct_plain,
        plane,
        show,
        top_words,
        vec,
        vocab_table,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part I · Constructing a transformer

    ## 1 · Eleven words, five sentences

    Here is everything that exists.
    """)
    return


@app.cell
def _(np):
    VOCAB = ["the", "a", "on", "cat", "dog", "bird", "mat", "hat", "grass", "sat", "chased"]
    ID = {word: i for i, word in enumerate(VOCAB)}

    CORPUS = [
        "the cat sat on the mat",
        "the cat sat on the mat",
        "the cat sat on the mat",
        "the cat sat on the hat",
        "the cat sat on the grass",
    ]

    def encode(text):
        """Text to token ids: one integer per word."""
        return np.array([ID[word] for word in text.split()])

    return CORPUS, ID, VOCAB, encode


@app.cell(hide_code=True)
def _(CORPUS, VOCAB, mo):
    mo.Html(
        '<div style="border-left:3px solid currentColor;padding:.5rem 1.1rem;margin:.6rem 0">'
        '<div style="font-size:.8rem;opacity:.7">The vocabulary: eleven words, each word one token</div>'
        f'<div style="font-size:1.05rem;margin-bottom:.8rem">{" · ".join(VOCAB)}</div>'
        '<div style="font-size:.8rem;opacity:.7">The training data: five sentences</div>'
        '<ol style="font-size:1.2rem;line-height:1.7;margin:.1rem 0 0 0;padding-left:1.5rem;list-style:decimal">'
        + "".join(f"<li>{_s}</li>" for _s in CORPUS)
        + "</ol></div>"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Three of the five sentences are identical. That is deliberate. How often something occurs is the only
    information this data contains.

    ### What we are trying to learn

    Cover the last word of each sentence. All five now read *the cat sat on the*. Which word comes next? The data
    does not give one answer. It gives a count.
    """)
    return


@app.cell
def _(CORPUS, ID, VOCAB, np):
    def continuations(context):
        """Count which word follows this context in the five sentences. None if the context never occurs."""
        words = context.split()
        counts = np.zeros(len(VOCAB))
        for sentence in CORPUS:
            s = sentence.split()
            if s[: len(words)] == words and len(s) > len(words):
                counts[ID[s[len(words)]]] += 1
        return counts if counts.sum() > 0 else None

    QUESTION = "the cat sat on the"
    counts = continuations(QUESTION)
    target = counts / counts.sum()  # the observed distribution for this context
    return QUESTION, continuations, counts, target


@app.cell(hide_code=True)
def _(counts, pct_plain, show, target, vocab_table):
    show(
        vocab_table(
            [
                ("times it came next", counts, "{:.0f}"),
                ("as a share", [f"{int(_c)}/5" for _c in counts], "text"),
                ("observed f(v)", target, "pct"),
            ],
            foot=["total", f"{int(counts.sum())}", "5/5", pct_plain("100.0%")],
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Dividing by five gives the **observed distribution** for this context: mat 3/5, hat 1/5, grass 1/5, and 0 for
    each of the other eight words. Call it $f$.

    $f$ is a probability measure on the vocabulary $\mathcal V$. On a finite set that means nothing more than one
    number per element, none of them negative, adding up to 1. The probability of an **event**, any set of words
    $A \subseteq \mathcal V$, is the sum over its members:

    $$P(A) = \sum_{v \in A} f(v)$$
    """)
    return


@app.cell
def _(ID, np, target):
    def is_measure(p):
        """Eleven numbers are a probability measure on the vocabulary if none is negative and they add up to 1."""
        return bool(np.all(p >= 0) and np.isclose(p.sum(), 1.0))

    def prob(event, p):
        """P(A) for an event A, a set of words: add up its members."""
        return float(p[[ID[word] for word in event]].sum())

    {
        "f is a probability measure": is_measure(target),
        "P(mat or hat)": prob({"mat", "hat"}, target),
        "P(something to sit on)": prob({"mat", "hat", "grass"}, target),
        "P(an animal)": prob({"cat", "dog", "bird"}, target),
    }
    return (is_measure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two cautions before building anything.

    $f$ describes the data, for this one context. It is the target a model should come close to. It is not the
    model.

    And counting is not a model either. Ask the counts about *a dog sat on the* and there is nothing to count,
    because that context never occurs. Eleven words make $11^5 = 161{,}051$ contexts of length five, and the data
    covers one of them. What we want is a rule that *computes* a probability for every word from the words of the
    context, whatever the context is, and that agrees with the counts where counts exist.

    The context below feeds every computation in the notebook. Change it (using only the eleven words) and
    everything recomputes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    context_box = mo.ui.text(value="the cat sat on the", label="Context:", full_width=True)
    context_box
    return (context_box,)


@app.cell(hide_code=True)
def _(ID, context_box, mo):
    _typed = context_box.value.lower().split()
    _unknown = [_w for _w in _typed if _w not in ID]
    _usable = 2 <= len(_typed) <= 8 and not _unknown
    text = " ".join(_typed) if _usable else "the cat sat on the"
    (
        None
        if _usable
        else mo.callout(
            mo.md(
                (f"Not in the vocabulary: **{', '.join(_unknown)}**. " if _unknown else "Use two to eight words. ")
                + f"Using *{text}* instead."
            ),
            kind="warn",
        )
    )
    return (text,)


@app.cell(hide_code=True)
def _(VOCAB, continuations, mo, text):
    _c = continuations(text)
    f_obs = None if _c is None else _c / _c.sum()  # what the five sentences say about the typed context, if anything
    mo.md(
        f"In the five sentences, *{text}* is followed by: "
        + ", ".join(f"**{VOCAB[_v]}** {int(_c[_v])}/{int(_c.sum())}" for _v in range(len(VOCAB)) if _c[_v] > 0)
        + "."
        if _c is not None
        else f"*{text}* never occurs at the start of any of the five sentences, so the data says nothing about what "
        "follows it. A model will still have to give an answer."
    )
    return (f_obs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Words as numbers: we have a matrix, now what?

    A rule that computes needs numbers to compute with. The obvious first step gives each word its index in the
    vocabulary.
    """)
    return


@app.cell
def _(encode, text):
    words = text.split()
    ids = encode(text)
    n = len(ids)  # number of positions
    return ids, n, words


@app.cell(hide_code=True)
def _(grid, ids, n, show, words):
    row_labels = [f"{_i + 1} {_w}" for _i, _w in enumerate(words)]
    show(
        grid(
            [words, [str(_k) for _k in ids]],
            rows=["word", "id"],
            cols=[f"position {_i + 1}" for _i in range(n)],
            bracket=False,
        )
    )
    return (row_labels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    An index is a name. "mat" is 6 and "cat" is 3, but mat is not twice cat, and arithmetic on these numbers means
    nothing. So give every word a short list of numbers of its own, a vector with $d$ coordinates, and leave those
    numbers free to take whatever values turn out to be useful. Store the vectors as the rows of a table $E$, one
    row per word. We take $d = 2$ so that every vector can be drawn. (Huginn uses 5,280.)

    The vector of a word is called its **embedding**. The values below are **hand-set**: words that are used alike
    were placed near each other, by eye. Whether they are good values is the subject of section 4.
    """)
    return


@app.cell
def _(np):
    E = np.array([
        [-1.00,  0.40],  # the
        [-0.70,  0.75],  # a
        [-0.85, -0.05],  # on
        [ 0.75,  0.75],  # cat
        [ 1.05,  0.50],  # dog
        [ 0.50,  1.05],  # bird
        [ 1.00, -0.45],  # mat
        [ 0.85, -0.80],  # hat
        [ 1.25, -0.75],  # grass
        [-0.20, -1.10],  # sat
        [ 0.05, -1.20],  # chased
    ])
    return (E,)


@app.cell(hide_code=True)
def _(BLUE, E, MUTED, SURFACE, VOCAB, grid, ids, mo, new_fig, plane, show):
    _fig, _ax = new_fig(1, width=5.0, height=4.6)
    plane(_ax)
    _in_text = sorted(set(ids))
    _ax.scatter(E[_in_text, 0], E[_in_text, 1], s=70, color=BLUE, edgecolors=SURFACE, linewidths=2, zorder=3)
    for _name, _xy in {"function words": (-1.6, 1.1), "animals": (0.95, 1.3), "places": (1.0, -1.15),
                       "verbs": (-0.85, -1.45)}.items():
        _ax.text(*_xy, _name, fontsize=8.5, style="italic", color=MUTED, alpha=0.75)
    _ax.set_title("Each dot is one row of E (blue: words in the context)")
    mo.hstack(
        [_fig, show(grid(E, rows=VOCAB, cols=["coordinate 1", "coordinate 2"], name="<i>E</i>"))],
        justify="start",
        align="center",
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The group names in the figure are ours. The model has no such labels, only the numbers.

    Looking up each word of the context and stacking the rows gives a matrix with one row per position. It has a
    flaw we can see at once: a word that occurs twice gives two identical rows, so nothing in the numbers tells the
    first *the* from the second. Each row should also say where it is. So we add to each row a small **position
    offset** $p_i$ that depends only on the position $i$:

    $$x_i = e_{w_i} + p_i$$

    where $w_i$ is the word at position $i$. The offsets used here are short arrows that turn 45° from one position
    to the next, rounded to two decimals so that the arithmetic below stays exact. Where that choice comes from is
    in the aside at the end of this section.
    """)
    return


@app.cell
def _(E, encode, np):
    OMEGA = np.pi / 4  # 45° per position

    def offsets(n):
        """One short arrow per position: all the same length, each turned 45° from the one before."""
        i = np.arange(1, n + 1)
        return np.round(0.15 * np.column_stack([np.cos(i * OMEGA), np.sin(i * OMEGA)]), 2)

    def embed(text):
        """Text to matrix: one row per position, embedding plus position offset."""
        ids = encode(text)
        return E[ids] + offsets(len(ids))

    return OMEGA, embed, offsets


@app.cell
def _(E, ids, n, offsets):
    X_tok = E[ids]  # one embedding per position
    P = offsets(n)  # one offset per position
    X = X_tok + P  # the same as embed(text)
    return P, X, X_tok


@app.cell(hide_code=True)
def _(P, X, X_tok, grid, ids, mo, n, np, row_labels, show):
    _repeat = np.array([list(ids).count(_k) > 1 for _k in ids])
    _mark = np.repeat(_repeat[:, None], 2, axis=1)
    _note = (
        "The highlighted rows hold the same word. They are identical on the left and different on the right."
        if _repeat.any()
        else "No word occurs twice in this context. Try one where a word repeats."
    )
    mo.vstack(
        [
            show(
                grid(X_tok, rows=row_labels, cols=["1", "2"], name="<i>E</i>[ids]", mark=_mark),
                "+",
                grid(P, rows=[f"<i>p</i><sub>{_i + 1}</sub>" for _i in range(n)], cols=["1", "2"], name="<i>P</i>"),
                "=",
                grid(X, rows=row_labels, cols=["1", "2"], name="<i>X</i>", mark=_mark),
            ),
            mo.md(_note),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, n):
    mo.md(rf"""
    $X \in \mathbb R^{{n \times d}}$, here ${n} \times 2$. Row $i$ is the word at position $i$. The two columns are
    coordinates and have no names. A longer context adds rows, never columns.

    One fact about $X$ matters more than any other for what follows. **Row $i$ was computed from word $i$ and from
    the number $i$, and from nothing else.** It records which word stands there and where. It knows nothing about
    its neighbours.

    So a sequence of words has become a matrix of numbers. But the goal was eleven probabilities, one for each word
    that could come next. How do we get from a ${n} \times 2$ matrix to a probability measure on $\mathcal V$?
    """)
    return


@app.cell(hide_code=True)
def _(BLUE, INK, OMEGA, ORANGE, RULE, arrow, mo, n, new_fig, np):
    _unit = np.column_stack([np.cos(np.arange(1, n + 1) * OMEGA), np.sin(np.arange(1, n + 1) * OMEGA)])
    _R = np.array([[np.cos(OMEGA), -np.sin(OMEGA)], [np.sin(OMEGA), np.cos(OMEGA)]])
    _steps_on = bool(np.allclose(_unit[:-1] @ _R.T, _unit[1:]))

    _fig, (_ax, _one, _three) = new_fig(3, width=9.6, height=2.9, width_ratios=[1, 1.5, 1.5])
    _circle = np.linspace(0, 2 * np.pi, 200)
    _ax.plot(np.cos(_circle), np.sin(_circle), color=RULE, lw=1)
    for _i in range(n):
        arrow(_ax, (0, 0), _unit[_i], color=ORANGE, lw=1.6, head=10)
        _ax.text(*(1.2 * _unit[_i]), str(_i + 1), ha="center", va="center", fontsize=9)
    _ax.set_xlim(-1.45, 1.45)
    _ax.set_ylim(-1.45, 1.45)
    _ax.set_aspect("equal")
    _ax.axis("off")
    _ax.set_title("Position arrows")

    _j = np.arange(1, 201)

    def _match(periods):
        return np.mean([np.cos((100 - _j) * 2 * np.pi / _period) for _period in periods], axis=0)

    for _a, _vals, _title in [(_one, _match([8]), "One circle, 45° per step"),
                              (_three, _match([8, 27, 200]), "Three circles: periods 8, 27, 200")]:
        _a.plot(_j, _vals, color=BLUE, lw=1.2)
        _a.axhline(0, color=RULE, lw=0.8)
        _a.plot([100], [1], "o", ms=5, color=INK)
        _a.set_title(_title)
        _a.set_xlabel("position j")
        _a.set_ylim(-1.1, 1.15)
        _a.tick_params(length=0)
    _one.set_ylabel("match with position 100")

    mo.accordion(
        {
            "Aside: why position offsets are arrows on a circle": mo.vstack(
                [
                    mo.md(
                        r"""
                        A counter (1, 2, 3, …) in an extra column fails in one of two ways. Left alone, it reads
                        1,000 at position 1,000 beside word coordinates of about ±1, and the word becomes a
                        rounding error. Divided by a maximum length, it turns positions 500 and 501 into 0.500 and
                        0.501, which can hardly be told apart.

                        An arrow of fixed length that turns by a fixed angle $\omega$ per position avoids both:

                        $$p_i = r\,(\cos i\omega,\ \sin i\omega), \qquad r = 0.15,\ \omega = 45°$$

                        Every arrow has the same length, so it never swamps the word. And turning by $\omega$ is
                        multiplication by one fixed matrix $R$, which carries each position's arrow onto the next,
                        $R\,p_j = p_{j+1}$. So "the position just before me" is the same computation everywhere in
                        the text. """
                        + f"(Checked for the unrounded arrows of this context: `{_steps_on}`.)"
                        + r"""

                        One circle is not enough for a long text. With 45° steps the arrow comes back around after
                        eight positions, so position 9 lands exactly on position 1; that is why the context box
                        stops at eight words. Real encodings use many circles at once, each in its own pair of
                        coordinates and turning at its own speed, like the hands of a clock. The charts show how
                        well position 100 matches every other position, averaged over the circles in use. With one
                        circle, every eighth position is a perfect look-alike. With three, only position 100 scores
                        1. The original transformer uses $d/2$ circles. Huginn and Recurrent-Llama apply the same
                        rotations inside the attention step of section 6 instead of adding offsets.
                        """
                    ),
                    _fig,
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · The simplest predictor

    We need a function that takes $X$ and returns eleven numbers, none negative, adding up to 1. Build it in two
    steps.

    **Step 1: one score per word.** The next word will stand right after the last position, so start with the last
    row, $x_n$. We need a number for every candidate word $v$, and we already have a vector for every word. The
    simplest way to make one number out of two vectors is their inner product:

    $$z_v = e_v^{\top} x_n, \qquad \text{or for all eleven words at once} \qquad z = E\,x_n$$

    A word scores high when its embedding points the same way as the last row. These scores are called **logits**.
    Nothing restricts their sign or size.

    **Step 2: scores to probabilities.** The logits are indexed by the right set, but some are negative and they
    add up to nothing in particular. Two repairs are needed. Exponentiate each logit, which makes it positive. Then
    divide by the total, which makes the results add up to 1:

    $$p(v \mid \text{context}) = \frac{e^{z_v}}{\sum_{u \in \mathcal V} e^{z_u}}$$

    This map from scores to probabilities is called **softmax**.
    """)
    return


@app.cell
def _(encode, np, offsets):
    def softmax(z):
        positive = np.exp(z - z.max(axis=-1, keepdims=True))  # subtracting the largest logit only avoids overflow
        return positive / positive.sum(axis=-1, keepdims=True)

    def first_predictor(table, context):
        """Read the last row, score every word against it, normalize."""
        ids = encode(context)
        x_n = table[ids[-1]] + offsets(len(ids))[-1]  # the last row of X
        z = table @ x_n  # one logit per word
        return x_n, z, softmax(z)

    return first_predictor, softmax


@app.cell
def _(E, X, first_predictor, is_measure, np, text):
    x_last, z0, p0 = first_predictor(E, text)

    exp_z0 = np.exp(z0)  # repair 1: every entry is now above 0
    denominator0 = exp_z0.sum()  # one number
    # repair 2: exp_z0 / denominator0 adds up to 1, and equals p0

    {
        "x_n is the last row of X": bool(np.array_equal(x_last, X[-1])),
        "exp(z) / total equals softmax(z)": bool(np.allclose(exp_z0 / denominator0, p0)),
        "the output is a probability measure": is_measure(p0),
    }
    return denominator0, exp_z0, p0, x_last, z0


@app.cell(hide_code=True)
def _(
    E,
    ID,
    VOCAB,
    X,
    denominator0,
    exp_z0,
    grid,
    last_row_mask,
    mo,
    np,
    num,
    p0,
    pct_plain,
    row_labels,
    show,
    text,
    vocab_table,
    x_last,
    z0,
):
    _w = "mat"
    _k = ID[_w]
    _mark = np.zeros((len(VOCAB), 5), dtype=bool)
    _mark[_k, 1:3] = True
    mo.vstack(
        [
            show(
                grid(X, rows=row_labels, cols=["1", "2"], name="<i>X</i>", mark=last_row_mask(X.shape)),
                "→",
                grid([x_last], name="<i>x<sub>n</sub></i>, the last row", mark=np.ones((1, 2), dtype=bool)),
            ),
            show(
                vocab_table(
                    [
                        ("embedding e<sub>v</sub>", E, "vec"),
                        ("logit z<sub>v</sub> = e<sub>v</sub>·x<sub>n</sub>", z0, "{:.4f}"),
                        ("exp(z<sub>v</sub>)", exp_z0, "{:.4f}"),
                        ("p(v | context)", p0, "pct"),
                    ],
                    mark=_mark,
                    foot=["total", "", num(z0.sum(), "{:.4f}"), num(denominator0, "{:.4f}"), pct_plain("100.0%")],
                )
            ),
            mo.md(
                f"The context is *{text}*. One row worked by hand, the highlighted one: "
                f"z({_w}) = ({num(E[_k, 0])})({num(x_last[0])}) + ({num(E[_k, 1])})({num(x_last[1])}) "
                f"= {num(z0[_k], '{:.4f}')}. Its probability is exp({num(z0[_k], '{:.4f}')}) = "
                f"{num(exp_z0[_k], '{:.4f}')} divided by the denominator {num(denominator0, '{:.4f}')}, "
                f"which is {p0[_k]:.1%}."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, p0, text, top_words):
    mo.md(rf"""
    Every entry of the last column is positive and the column adds up to 1. That holds for any table $E$ and any
    context, because $e^{{z}}$ is positive for every real $z$ and the division forces the total. So
    $X \mapsto \operatorname{{softmax}}(E\,x_n)$ is a complete next-word predictor: it turns any context into a
    probability measure on $\mathcal V$.

    Valid is all it is. After *{text}* its three favourite words are {top_words(p0)}.

    Two properties of softmax are worth keeping. Only the gaps between logits matter, since
    $p(a)/p(b) = e^{{z_a - z_b}}$: each extra unit of gap multiplies the odds by $e \approx 2.7$. And no word ever
    gets probability exactly 0, a fact that returns in the next section.

    *Forced or chosen?* The output had to be eleven non-negative numbers adding up to 1. Everything else here is a
    choice: reading only the last row, scoring by inner product, scoring against the same table $E$ that supplied
    the rows (many models learn a second table for that), and softmax itself, which is one of many ways to turn
    scores into a measure.
    """)
    return


@app.cell(hide_code=True)
def _(VOCAB, hbars, mo, new_fig, np, softmax, z0):
    def _divide(z):
        return z / z.sum()

    def _clip(z):
        z = np.maximum(z, 0)
        return z / z.sum()

    _fig, _axes = new_fig(3, width=9.6, height=3.7, width_ratios=[1.25, 1, 1])
    for _ax, (_name, _recipe) in zip(
        _axes, [("divide by the total", _divide), ("clip, then divide", _clip), ("softmax", softmax)]
    ):
        with np.errstate(divide="ignore", invalid="ignore"):
            _values = _recipe(z0)
        _bad = ~np.isfinite(_values) | (_values < 0)
        _zeros = int(np.sum(_values == 0))
        if _bad.any():
            _verdict = "not a measure"
        elif _zeros:
            _verdict = f"a measure, but {_zeros} words get exactly 0"
        else:
            _verdict = "a measure, every word above 0"
        hbars(_ax, _values, bad=_bad, title=_name, ticklabels=VOCAB if _ax is _axes[0] else False)
        _ax.set_xlabel(_verdict)

    mo.accordion(
        {
            "Aside: why exponentiate? Two simpler recipes and how they fail": mo.vstack(
                [
                    mo.md(
                        r"""
                        The same logits, turned into numbers that add up to 1 in three ways.

                        - **Divide by the total.** The results add up to 1, but a negative logit becomes a negative
                          "probability" (or, if the total is negative, a positive one does), and a total near 0
                          blows everything up. Red bars are not valid probabilities.
                        - **Clip negatives to 0, then divide.** Valid whenever some logit is positive. But every
                          word with a negative logit gets exactly 0, and the loss of the next section is infinite
                          when a word given 0 then occurs. Doubling every logit also changes nothing, so the model
                          cannot express how sure it is.
                        - **Softmax.** Positive everywhere, never exactly 0, and doubling the logits squares the
                          odds.
                        """
                    ),
                    _fig,
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · How wrong is it, and what can be changed?

    The predictor produces a measure. The data produced a measure too, for the one context it knows about. This
    section always uses that context, *the cat sat on the*, whatever is typed in the box, because it is the only
    one for which the data has more than one thing to say.
    """)
    return


@app.cell
def _(E, QUESTION, first_predictor, np, target):
    def nll(p, f):
        """Average of -log p(word that came next) over data in which word v came next a share f[v] of the time."""
        return float(-np.sum(f[f > 0] * np.log(p[f > 0])))

    _, _, p_question = first_predictor(E, QUESTION)

    loss_start = nll(p_question, target)
    loss_floor = nll(target, target)  # the loss of a model that reproduced the observed distribution exactly
    {"loss of the section 3 predictor": round(loss_start, 4), "loss of a perfect model (the floor)": round(loss_floor, 4)}
    return loss_floor, loss_start, nll, p_question


@app.cell(hide_code=True)
def _(ID, loss_floor, loss_start, mo, num, p_question):
    _m, _h, _g = (p_question[ID[_w]] for _w in ("mat", "hat", "grass"))
    mo.md(
        rf"""
    ### A number for "how wrong"

    Go through the five sentences. In each, the model gave some probability to the word that actually came next.
    A good model gives those words high probability. Multiply the five probabilities to get the probability the
    model assigns to the data as a whole, $p(\text{{mat}})^3\,p(\text{{hat}})\,p(\text{{grass}})$. Take the
    logarithm to turn the product into a sum, flip the sign so that smaller is better, and average over the five:

    $$L = -\tfrac15\big[\,3\log p(\text{{mat}}) + \log p(\text{{hat}}) + \log p(\text{{grass}})\,\big]
        = -\sum_{{v \in \mathcal V}} f(v)\,\log p(v)$$

    This is the **negative log-likelihood**, the **loss**. A word the model was sure of costs nearly 0. A word it
    gave probability near 0 costs without limit. With the numbers of section 3:

    $$L = -\big(0.6\,\ln {num(_m, "{:.4f}")} + 0.2\,\ln {num(_h, "{:.4f}")} + 0.2\,\ln {num(_g, "{:.4f}")}\big)
        = {num(loss_start, "{:.3f}")}$$

    Among all measures $p$, the loss is smallest exactly when $p = f$ (Gibbs' inequality), and its value there is
    $-\sum f \log f = {num(loss_floor, "{:.3f}")}$. So lowering the loss and approaching the observed distribution
    are the same thing, and {num(loss_floor, "{:.3f}")} is the floor.

    ### What can be changed

    Not the data. Not softmax, and not the rule "read the last row": those are fixed formulas. The only numbers in
    the predictor that we picked freely are the 22 entries of $E$. They are the model's **parameters**, also called
    its **weights**, and written $\theta$ when there are too many to name. **Training** means adjusting $\theta$
    to lower $L$. Nothing else is learned: when people say a model "learned" something, the whole of what changed
    is a list of numbers like this one.

    ### Which way to change them

    $L$ depends on $E$ through the logits. Its slope with respect to each logit has a simple form,

    $$\frac{{\partial L}}{{\partial z_v}} = p(v) - f(v)$$

    the model's probability minus the observed one. Words the model overrates get their logits pushed down, words
    it underrates get pushed up, and the pushes stop only when $p = f$. The chain rule carries those slopes back to
    $E$, which enters twice: every row scores a word, and the row for *the* is also the word being read. One step
    of **gradient descent** moves every entry of $E$ a little way down its slope. Then repeat.
    """
    )
    return


@app.cell
def _(E, QUESTION, encode, np, offsets, softmax, target):
    def slope_of_loss(table, context, f):
        """dL/dE for the first predictor on one context, by the chain rule."""
        ids = encode(context)
        last = ids[-1]
        x_n = table[last] + offsets(len(ids))[-1]
        slope_z = softmax(table @ x_n) - f  # dL/dz
        slope = np.outer(slope_z, x_n)  # E as the scorer: z = E x_n
        slope[last] += table.T @ slope_z  # E as the source of x_n
        return slope

    def train_table(table, context, f, steps, rate=0.5):
        """Gradient descent on the 22 numbers of E. Returns the table after 0, 1, ..., steps steps."""
        tables = [table.copy()]
        for _ in range(steps):
            table = table - rate * slope_of_loss(table, context, f)
            tables.append(table.copy())
        return tables

    tables = train_table(E, QUESTION, target, steps=2000)
    E_trained = tables[-1]
    return E_trained, slope_of_loss, tables


@app.cell
def _(E, QUESTION, first_predictor, nll, np, slope_of_loss, target):
    def slope_by_nudging(table, context, f, eps=1e-6):
        """The same slope without calculus: nudge each of the 22 numbers and watch the loss."""
        slope = np.zeros_like(table)
        for r in range(table.shape[0]):
            for c in range(table.shape[1]):
                up, down = table.copy(), table.copy()
                up[r, c] += eps
                down[r, c] -= eps
                slope[r, c] = (nll(first_predictor(up, context)[2], f) - nll(first_predictor(down, context)[2], f)) / (2 * eps)
        return slope

    {
        "the chain-rule slope equals the slope found by nudging": bool(
            np.allclose(slope_of_loss(E, QUESTION, target), slope_by_nudging(E, QUESTION, target), atol=1e-6)
        )
    }
    return


@app.cell(hide_code=True)
def _(mo):
    train_step = mo.ui.slider(
        steps=[0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
        value=0,
        label="Training steps taken",
        show_value=True,
        full_width=True,
    )
    mo.vstack(
        [
            mo.md(
                "The check above confirms the chain-rule slope against plain nudging. Now run the descent. Drag "
                "the slider to move through it; every number below is the state of the predictor after that many "
                "steps."
            ),
            train_step,
        ]
    )
    return (train_step,)


@app.cell(hide_code=True)
def _(
    BLUE,
    E,
    ID,
    INK,
    MUTED,
    QUESTION,
    RULE,
    SURFACE,
    VOCAB,
    first_predictor,
    grid,
    loss_floor,
    mo,
    new_fig,
    nll,
    np,
    plane,
    show,
    tables,
    target,
    train_step,
    vec,
    vocab_table,
):
    _t = train_step.value
    _E = tables[_t]
    _x, _z, _p = first_predictor(_E, QUESTION)
    _loss = nll(_p, target)
    _losses = [nll(first_predictor(_tab, QUESTION)[2], target) for _tab in tables]
    _seen = [ID[_w] for _w in ("mat", "hat", "grass")]
    _rest = 1 - _p[_seen].sum()

    _fig, (_ax, _curve) = new_fig(2, width=9.4, height=3.9, width_ratios=[1, 1.25])
    _lim = max(1.7, float(np.abs(_E).max()) + 0.6)
    plane(_ax, table=_E, lim=_lim)
    _ax.scatter(E[:, 0], E[:, 1], s=18, facecolors="none", edgecolors=MUTED, linewidths=0.8, zorder=1)
    for _v in range(len(VOCAB)):
        _ax.plot([E[_v, 0], _E[_v, 0]], [E[_v, 1], _E[_v, 1]], color=RULE, lw=1, zorder=0)
    _ax.plot(*_x, "x", ms=7, mew=2, color=INK, zorder=4)
    _ax.set_title(f"E after {_t} steps (rings: start, ×: the last row)")

    _steps = np.arange(len(_losses))
    _curve.plot(_steps + 1, _losses, color=BLUE, lw=2)
    _curve.axhline(loss_floor, color=MUTED, lw=1, ls=(0, (4, 3)))
    _curve.text(len(_losses), loss_floor - 0.06, f"floor {loss_floor:.3f}", ha="right", va="top", fontsize=8.5,
                color=MUTED)
    _curve.plot([_t + 1], [_loss], "o", ms=8, color=INK, mec=SURFACE, mew=1.5, zorder=4)
    _curve.set_xscale("log")
    _curve.set_ylim(0.6, None)
    _curve.set_xlabel("steps taken + 1")
    _curve.set_title("Loss during the descent")
    _curve.tick_params(length=0)

    mo.vstack(
        [
            mo.hstack(
                [
                    show(
                        vocab_table(
                            [
                                ("row of E", _E, "vec"),
                                ("logit", _z, "{:.3f}"),
                                ("model p(v)", _p, "pct"),
                                ("observed f(v)", target, "pct"),
                            ],
                            name=f"After {_t} steps",
                        )
                    ),
                    show(
                        grid(
                            [
                                ["steps taken", str(_t)],
                                ["last row x<sub>n</sub>", vec(_x)],
                                ["loss L", f"<b>{_loss:.4f}</b>"],
                                ["the floor", f"{loss_floor:.4f}"],
                                ["gap to the floor", f"{_loss - loss_floor:.4f}"],
                                ["probability left on the eight words that never came next", f"{_rest:.2%}"],
                            ],
                            bracket=False,
                            left=(0,),
                        )
                    ),
                ],
                justify="start",
                align="start",
                gap=2,
            ),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(E_trained, ID, QUESTION, first_predictor, loss_floor, mo, nll, target):
    _p = first_predictor(E_trained, QUESTION)[2]
    _rest = 1 - sum(_p[ID[_w]] for _w in ("mat", "hat", "grass"))
    mo.md(
        rf"""
    After 2,000 steps the model gives mat {_p[ID["mat"]]:.1%}, hat {_p[ID["hat"]]:.1%} and grass
    {_p[ID["grass"]]:.1%}, with a loss of {nll(_p, target):.4f} against a floor of {loss_floor:.4f}. Those numbers
    are **trained**: they came out of the loop above, starting from the hand-set table.

    **The floor is never reached.** The observed distribution gives eight words probability exactly 0. Softmax
    cannot: $e^{{z}}$ is positive for every finite $z$. So $p = f$ is unattainable with finite logits, and the
    descent can only keep widening the gaps between the three words that occurred and the eight that did not. After
    2,000 steps the eight still share {_rest:.2%}. More steps shrink that number and never make it 0. This is a
    property of softmax, and a reasonable one: a model that says "never" pays an infinite loss the first time it is
    wrong.

    **What this run did not do.** It fitted one context. The five sentences contain four shorter contexts as well
    (*the* is followed by *cat*, and so on), and a full training run would descend on all of them at once.

    From here on the notebook goes back to the hand-set table of section 2. The defect found next does not depend
    on which numbers are in $E$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · What the last row cannot know

    Look again at what the predictor reads. It reads $x_n = e_{w_n} + p_n$: the embedding of the last word plus the
    offset of the last position. So

    $$p(\,\cdot \mid w_1, \dots, w_n) \text{ depends only on } w_n \text{ and } n.$$

    Take any two contexts of the same length that end in the same word. Their matrices differ, but their last rows
    are the same row, so their logits are equal and so are their measures. This holds for every possible table $E$,
    trained or not.

    Here is a second context to compare with the first. It must have the same length and the same last word.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    contrast_box = mo.ui.text(value="on the mat sat the", label="Second context:", full_width=True)
    contrast_box
    return (contrast_box,)


@app.cell(hide_code=True)
def _(ID, contrast_box, mo, words):
    def _fallback(ws):
        if ws == "the cat sat on the".split():
            return "on the mat sat the".split()
        alt = ws[:-1][::-1] + ws[-1:]
        if alt == ws:
            alt = [("a" if _w != "a" else "the") for _w in ws[:-1]] + ws[-1:]
        return alt

    _typed = contrast_box.value.lower().split()
    _ok = (
        all(_w in ID for _w in _typed)
        and len(_typed) == len(words)
        and _typed[-1] == words[-1]
        and _typed != words
    )
    text_b = " ".join(_typed if _ok else _fallback(words))
    (
        None
        if _ok
        else mo.callout(
            mo.md(
                f"The second context has to be a different context with {len(words)} words that ends in "
                f"*{words[-1]}*. Using *{text_b}* instead."
            ),
            kind="warn",
        )
    )
    return (text_b,)


@app.cell
def _(E, E_trained, embed, first_predictor, np, p0, text, text_b):
    X_b = embed(text_b)
    _, z0_b, p0_b = first_predictor(E, text_b)

    {
        "the two matrices are equal": bool(np.array_equal(embed(text), X_b)),
        "their last rows are equal": bool(np.array_equal(embed(text)[-1], X_b[-1])),
        "the two predictions are equal": bool(np.array_equal(p0, p0_b)),
        "and equal with the trained table too": bool(
            np.array_equal(first_predictor(E_trained, text)[2], first_predictor(E_trained, text_b)[2])
        ),
    }
    return X_b, p0_b


@app.cell(hide_code=True)
def _(
    E_trained,
    ID,
    X,
    X_b,
    first_predictor,
    grid,
    last_row_mask,
    mo,
    p0,
    p0_b,
    row_labels,
    show,
    text,
    text_b,
    vocab_table,
):
    _labels_b = [f"{_i + 1} {_w}" for _i, _w in enumerate(text_b.split())]
    _p_tr = first_predictor(E_trained, text)[2]
    mo.vstack(
        [
            show(
                grid(X, rows=row_labels, cols=["1", "2"], name=f"<i>X</i> for “{text}”", mark=last_row_mask(X.shape)),
                "&nbsp;",
                grid(X_b, rows=_labels_b, cols=["1", "2"], name=f"<i>X</i> for “{text_b}”",
                     mark=last_row_mask(X_b.shape)),
                "&nbsp;",
                vocab_table([(f"p(v | {text})", p0, "pct"), (f"p(v | {text_b})", p0_b, "pct")]),
            ),
            mo.md(
                f"The matrices differ in the rows that are not highlighted. The predictor never looks at those. "
                f"Training does not help: the table trained in section 4 gives *mat* {_p_tr[ID['mat']]:.1%} after "
                f"*{text}*, and the same {_p_tr[ID['mat']]:.1%} after *{text_b}*."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With the default pair the failure is easy to feel. *The cat sat on the …* should continue with a place. *On the
    mat sat the …* should continue with an animal. One measure cannot be right for both.

    **What this does and does not show.** The five sentences alone do not prove the predictor inadequate. A
    function of the last word and the position can tell apart every context that occurs in the data: *the* stands
    at position 1 (followed by *cat*) and at position 5 (followed by *mat*, *hat* or *grass*), and those two rows
    differ by their position offsets. Section 4 showed that the predictor can fit 60/20/20. The second context
    above is not in the data, and we are not adding it. It is a counterexample to the *form* of the predictor: it
    shows a distinction that matters for language and that no choice of $E$ can express.

    So the next requirement is clear, and it is forced. **Before the last row is read, it has to come to depend on
    the other rows.** Row $n$ of $X$ is an *initial* description of one word at one position. What we need in its
    place is a description of that position *in its context*. Building that is the rest of Part I.
    """)
    return


@app.cell(hide_code=True)
def _(
    BLUE,
    E,
    LABEL_SHIFT,
    MUTED,
    SURFACE,
    VOCAB,
    first_predictor,
    mo,
    new_fig,
    np,
    plane,
    softmax,
):
    def history_measure(logits_of, length):
        """The probability of every context of this length when the model writes it itself (first word uniform)."""
        mu = {(_v,): 1 / len(VOCAB) for _v in range(len(VOCAB))}
        for _ in range(length - 1):
            mu = {_w + (_v,): _m * _q for _w, _m in mu.items() for _v, _q in enumerate(softmax(logits_of(_w)))}
        return mu

    def pushforward(logits_of, mu):
        """The measure on logit vectors induced by a measure on contexts."""
        mu_Z = {}
        for _w, _m in mu.items():
            _z = tuple(np.round(logits_of(_w), 9))
            mu_Z[_z] = mu_Z.get(_z, 0.0) + _m  # contexts that give the same logits pool their probability
        return mu_Z

    def atoms(ax, measure, scale=4500, names=None):
        """Draw a measure on logit vectors: one dot per vector, at the last row it comes from, area = probability."""
        logits = np.array(list(measure))
        mass = np.array(list(measure.values()))
        rows = logits @ np.linalg.pinv(E).T  # z = E h has exactly one solution h, because E has rank 2
        plane(ax, lim=max(1.7, float(np.abs(rows).max()) + 0.25), labels=False, alpha=0.25)
        ax.scatter(rows[:, 0], rows[:, 1], s=scale * mass, color=BLUE, alpha=0.6, edgecolors=SURFACE,
                   linewidths=0.4, zorder=3)
        if names:
            for row, key, m in zip(rows, measure, mass):
                dx, dy, ha = LABEL_SHIFT.get(names[key], (0.08, 0.05, "left"))
                ax.text(row[0] + 2.2 * dx, row[1] + 2.2 * dy, f"{names[key]} {m:.0%}", ha=ha, va="center",
                        fontsize=8.5, color=MUTED)

    def _first(context_ids):
        return first_predictor(E, " ".join(VOCAB[_k] for _k in context_ids))[1]

    mu_first = history_measure(_first, 3)
    mu_Z_first = pushforward(_first, mu_first)
    _names = {tuple(np.round(_first(_w), 9)): VOCAB[_w[-1]] for _w in mu_first}
    _fig, _ax = new_fig(1, width=5.2, height=4.4)
    atoms(_ax, mu_Z_first, names=_names)
    _ax.set_title("The distribution of Z after three words")

    mo.accordion(
        {
            "Aside: counting what the first predictor can say (pushforward measures)": mo.vstack(
                [
                    mo.md(
                        rf"""
                        There are two probability distributions in play, on different spaces. For one fixed
                        context, softmax of the logits is a distribution on $\mathcal V$: which word comes next.
                        But if the context is itself random, for example because the model wrote it one word at a
                        time, then the logit vector $Z$ is random too. Its distribution is the **pushforward** of
                        the distribution over contexts: the probability of a logit vector is the total probability
                        of the contexts that produce it.

                        With eleven words this can be computed exactly. Let the first predictor write three words,
                        the first uniformly at random and each later one from its own softmax. There are
                        {len(mu_first):,} contexts of three words, and the predictor can produce only
                        **{len(mu_Z_first)}** different logit vectors from them: one per last word. That is the
                        limitation of this section, counted. Each dot below is one possible logit vector, drawn at
                        the last row it comes from, with area equal to its probability. Section 12 repeats the
                        count for the finished layer.
                        """
                    ),
                    _fig,
                ]
            )
        }
    )
    return atoms, history_measure, mu_Z_first, pushforward


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Letting rows read earlier rows

    We want a transformation

    $$T : \mathbb R^{n \times d} \to \mathbb R^{n \times d}, \qquad X \mapsto H$$

    whose output row $h_i$ can depend on rows of $X$ other than row $i$. The output keeps the shape of the input for
    a practical reason: the last row of $H$ can then be scored against $E$ exactly as before.

    **Which rows may row $i$ use?** We read only the last row, but nothing is special about it. Every beginning of
    a sentence is itself a context (*the*, *the cat*, *the cat sat*, …), and row $i$ is the last row of the first
    $i$ words. In training, each of those rows is asked to predict the word that follows it. If row $i$ could see
    row $i + 1$ it would be reading the answer. So row $i$ may use rows $j \le i$ and no others.

    The simplest way to combine several vectors into one is a weighted sum:

    $$h_i = \sum_{j \le i} \alpha_{ij}\, x_j$$

    Everything now turns on the weights $\alpha_{ij}$. We try three ways of choosing them, each fixing the failure
    of the one before.

    ### First try: equal weights

    Give every permitted row the same weight, $\alpha_{ij} = 1/i$. Row $i$ becomes the average of rows 1 to $i$.
    """)
    return


@app.cell
def _(X, X_b, n, np):
    U = np.tril(np.ones((n, n))) / np.arange(1, n + 1)[:, None]  # row i: weight 1/i on rows 1..i, 0 on later rows
    H_avg = U @ X
    H_avg_b = U @ X_b
    return H_avg, H_avg_b, U


@app.cell(hide_code=True)
def _(
    E,
    H_avg,
    H_avg_b,
    U,
    X,
    embed,
    grid,
    last_row_mask,
    mo,
    n,
    np,
    row_labels,
    show,
    softmax,
    text,
    text_b,
    top_words,
    vec,
    words,
):
    _nums = [str(_i + 1) for _i in range(n)]
    _nice = {"the cat sat on the": "on the cat sat the"}
    _alt = _nice.get(text, " ".join(words[:-1][::-1] + words[-1:]))
    if _alt == text:
        _alt = " ".join(words[1:-1] + words[:1] + words[-1:])
    _same = bool(np.allclose((U @ embed(_alt))[-1], H_avg[-1]))
    _differ = (
        "The two contexts now get different last rows and different predictions. That is the first time anything "
        "before the last word has had a say."
        if not np.allclose(H_avg[-1], H_avg_b[-1])
        else "These two contexts contain the same words in a different order, and they still get the same last "
        "row. The next paragraph says why."
    )
    _order = (
        f"Reorder the words of the context, for example into *{_alt}*. The new last row is "
        f"{vec(H_avg[-1], '{:.3f}')} again (`{_same}`)."
        if _alt != text
        else "This context has only one distinct ordering of its words, so there is nothing to compare."
    )
    mo.vstack(
        [
            show(
                grid(U, rows=row_labels, cols=_nums, name="weights <i>α</i>", mark=last_row_mask(U.shape),
                     dim=np.triu(np.ones((n, n)), 1)),
                "×",
                grid(X, cols=["1", "2"], name="<i>X</i>", mark=np.ones(X.shape, dtype=bool)),
                "=",
                grid(H_avg, rows=row_labels, cols=["1", "2"], name="<i>H</i>", mark=last_row_mask(X.shape)),
            ),
            mo.md(
                f"""
    Blue marks what produces the last row: its weights, the rows they multiply, and the result. Read out with the
    same $E$ as before:

    | context | last row after averaging | three most probable next words |
    |---|---|---|
    | *{text}* | {vec(H_avg[-1], "{:.3f}")} | {top_words(softmax(E @ H_avg[-1]))} |
    | *{text_b}* | {vec(H_avg_b[-1], "{:.3f}")} | {top_words(softmax(E @ H_avg_b[-1]))} |

    {_differ}

    **What fails.** The average loses the order of the words completely, position offsets and all. The sum
    $\\sum_j (e_{{w_j}} + p_j)$ splits into $\\sum_j e_{{w_j}} + \\sum_j p_j$, and neither part depends on which
    word stood where. {_order} And every word has the same say, $1/n$, whether or not it bears on what comes next.
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Second try: unequal weights, fixed in advance

    Let the weights depend on the two positions, say with recent rows counting more. Order now matters. But the
    weights are the same for every text: position $i - 3$ gets its weight whether it holds *cat* or *on*. In *the
    cat sat on the*, the word that says most about what comes next is three back. In *on the mat sat the*, it is
    one or two back. A table of weights fixed in advance cannot follow that.

    ### Third try: weights computed from the rows themselves

    Let $\alpha_{ij}$ be a function of the two rows involved, $x_i$ and $x_j$. Two requirements shape it. The
    weights of one row should be non-negative and add up to 1, so that $h_i$ is a weighted *average*: it stays on
    the scale of the rows however long the text is. And we would like to start from an unrestricted score, any real
    number per pair of rows. Section 3 already has the tool for turning arbitrary scores into non-negative numbers
    that add up to 1:

    $$\alpha_{ij} = \frac{e^{s_{ij}}}{\sum_{k \le i} e^{s_{ik}}}, \qquad s_{ij} = \text{a score for how much row } i \text{ should take from row } j$$

    The simplest score for two vectors is again their inner product, $s_{ij} = x_i^{\top} x_j$.
    """)
    return


@app.cell
def _(X, np, softmax):
    def causal_softmax(S):
        """Scores to weights: row i keeps only rows j ≤ i, then softmax along the row."""
        allowed = np.tri(len(S)) > 0
        return softmax(np.where(allowed, S, -np.inf))  # exp(-inf) = 0: a forbidden row gets weight exactly 0

    S_raw = X @ X.T  # S_raw[i, j] = x_i · x_j
    A_raw = causal_softmax(S_raw)
    return A_raw, S_raw, causal_softmax


@app.cell(hide_code=True)
def _(A_raw, S_raw, grid, last_row_mask, mo, n, np, row_labels, show, words):
    _nums = [f"{_i + 1} {_w}" for _i, _w in enumerate(words)]
    _upper = np.triu(np.ones((n, n)), 1)
    _best = int(np.argmax(A_raw[-1]))
    _selfish = int(np.sum(np.argmax(A_raw, axis=1) == np.arange(n)))
    mo.vstack(
        [
            show(
                grid(S_raw, rows=row_labels, cols=_nums, name="scores <i>S</i> = <i>X X</i><sup>⊤</sup>",
                     mark=last_row_mask(S_raw.shape), dim=_upper),
                "→",
                grid(A_raw, rows=row_labels, cols=_nums, name="weights <i>A</i>: softmax along each row",
                     mark=last_row_mask(A_raw.shape), dim=_upper),
            ),
            mo.md(
                f"""
    Row $i$ of $S$ holds row $i$'s scores for every row, and row $i$ of $A$ the weights they turn into. Faded
    entries are the forbidden ones, $j > i$; they get weight 0. Each row of $A$ adds up to 1.

    **What fails.** Look at where the weight goes. The last row, *{words[-1]}*, puts its largest weight
    ({A_raw[-1, _best]:.0%}) on row {_best + 1}, *{words[_best]}*, and {_selfish} of the {n} rows put their largest
    weight on themselves. That is built into the score: $x_i^{{\\top}} x_j$ is large when $x_j$ resembles $x_i$, and
    nothing resembles $x_i$ more than $x_i$. A row that asks "who looks like me?" learns nothing it did not already
    know. The last word of a context usually needs rows that are *unlike* it: a determiner needs the noun or the
    verb. The score is also symmetric, $s_{{ij}} = s_{{ji}}$, as if what row $i$ wants from row $j$ had to equal
    what row $j$ wants from row $i$.
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The repair: let the question differ from the asker

    Before comparing two rows, transform each side with its own matrix:

    $$q_i = x_i\,W_Q \quad \text{(what row } i \text{ is looking for)} \qquad k_j = x_j\,W_K \quad \text{(how row } j \text{ presents itself)} \qquad s_{ij} = q_i^{\top} k_j$$

    $q_i$ is called row $i$'s **query** and $k_j$ row $j$'s **key**. As matrices, $Q = X W_Q$, $K = X W_K$ and
    $S = Q K^{\top}$. With $W_Q = W_K = I$ this is the previous try, so nothing has been lost. Otherwise
    $s_{ij} = x_i\,W_Q W_K^{\top} x_j^{\top}$ is a general bilinear score: it need not be symmetric, and a row need
    not score highly against itself. The entries of $W_Q$ and $W_K$ are eight new parameters.

    Rows here are row vectors, so a matrix acts from the right. Two sliders set $W_Q$ as "turn the row by an angle
    $\theta$ and stretch it by $\beta$". $W_K$ stays the identity. These weights are **hand-set**; at the default,
    a query is its row turned a quarter turn and doubled.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    beta1 = mo.ui.slider(0, 6, step=0.5, value=2, label="Stretch β", show_value=True, debounce=True)
    theta1 = mo.ui.slider(-180, 180, step=15, value=90, label="Turn θ (degrees)", show_value=True, debounce=True)
    mo.hstack([beta1, theta1], justify="start", gap=2)
    return beta1, theta1


@app.cell
def _(X, beta1, causal_softmax, np, theta1):
    def turn(degrees, stretch=1.0):
        """The 2×2 matrix that turns a row vector anticlockwise by this angle and stretches it."""
        a = np.deg2rad(degrees)
        return np.round(stretch * np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]), 10)

    WQ1 = turn(theta1.value, beta1.value)
    WK1 = np.eye(2)

    Q1 = X @ WQ1  # one query per row
    K1 = X @ WK1  # one key per row
    S1 = Q1 @ K1.T  # S1[i, j] = q_i · k_j
    A1 = causal_softmax(S1)  # each row: weights over rows j ≤ i
    return A1, K1, Q1, S1, WK1, WQ1, turn


@app.cell(hide_code=True)
def _(
    A1,
    K1,
    Q1,
    S1,
    WK1,
    WQ1,
    X,
    grid,
    last_row_mask,
    mo,
    n,
    np,
    num,
    row_labels,
    show,
    words,
):
    _nums = [f"{_i + 1} {_w}" for _i, _w in enumerate(words)]
    _upper = np.triu(np.ones((n, n)), 1)
    _best = int(np.argmax(A1[-1]))
    mo.vstack(
        [
            show(
                grid(X, rows=row_labels, name="<i>X</i>"),
                "×",
                grid(WQ1, name="<i>W<sub>Q</sub></i>"),
                "=",
                grid(Q1, name="<i>Q</i>", mark=last_row_mask(Q1.shape)),
                "&nbsp;&nbsp;",
                grid(X, name="<i>X</i>"),
                "×",
                grid(WK1, name="<i>W<sub>K</sub></i>"),
                "=",
                grid(K1, name="<i>K</i>", mark=np.ones(K1.shape, dtype=bool)),
            ),
            show(
                grid(S1, rows=row_labels, cols=_nums, name="<i>Q K</i><sup>⊤</sup>", mark=last_row_mask(S1.shape),
                     dim=_upper),
                "→",
                grid(A1, rows=row_labels, cols=_nums, name="<i>A</i>: mask, then softmax along each row",
                     mark=last_row_mask(A1.shape), dim=_upper),
            ),
            mo.md(
                f"""
    $Q$ and $K$ are ${n} \\times 2$: one query and one key per row. $Q K^{{\\top}}$ is ${n} \\times {n}$: entry
    $(i, j)$ is the inner product of row $i$'s query with row $j$'s key. The last row's entry for row {_best + 1}
    is ({num(Q1[-1, 0])})({num(K1[_best, 0])}) + ({num(Q1[-1, 1])})({num(K1[_best, 1])}) = {num(S1[-1, _best], "{:.3f}")}.
    $A$ is ${n} \\times {n}$ too, lower-triangular, with every row adding up to 1. Entry $(i, j)$ is the share of
    row $j$ in row $i$'s weighted average.

    With these weights the last row, *{words[-1]}*, puts {A1[-1, _best]:.0%} on row {_best + 1}, *{words[_best]}*,
    and {A1[-1, -1]:.0%} on itself.
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    **Two softmaxes, two different sample spaces.**

    - The softmax of this section runs along one row of scores. Its output, a row of $A$, is a distribution over
      the **positions** $1, \dots, i$ of the text. Nothing is ever drawn from it. It is used as mixing weights.
    - The softmax of section 3 runs over the eleven logits. Its output is a distribution over the **vocabulary**,
      and the next word is drawn from it.

    Same formula, different jobs. The first decides where a row reads. The second decides what the model writes.
    """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The weights $\alpha_{ij}$ are called **attention weights**, and the construction **attention**: row $i$
    attends to row $j$ with weight $\alpha_{ij}$. The rule $j \le i$ is the **causal mask**.

    *Forced or chosen?* What section 5 forced is only that the last row must depend on earlier rows, in a way that
    can tell contexts apart. Everything about *how* is a choice: a weighted average instead of some other
    combination, softmax to normalize, an inner product of two linear maps as the score. Other designs meet the
    same requirement (a running summary passed from row to row, or fixed windows of neighbours). Attention is the
    one that transformers use, not the only one that could exist.

    ## 7 · What should a row send?

    The weights are settled. Now look at what is being averaged. So far each row contributes itself, $x_j$. Here is
    the last row after averaging raw rows with the weights of section 6, read out with $E$ as before.
    """)
    return


@app.cell
def _(A1, E, X, softmax):
    H_copy = A1 @ X  # every row replaced by a weighted average of raw rows
    p_copy = softmax(E @ H_copy[-1])
    return H_copy, p_copy


@app.cell(hide_code=True)
def _(A1, H_copy, mo, np, p_copy, top_words, vec, words):
    _best = int(np.argmax(A1[-1]))
    mo.md(
        rf"""
    The last row becomes {vec(H_copy[-1], "{:.3f}")}, and the three most probable next words are
    {top_words(p_copy)}.

    **What fails.** The model predicts words like the ones it has just read. That is no accident of these weights.
    The readout scores a word $v$ by $e_v^{{\top}} h_n$, and $h_n$ is an average of rows $x_j \approx e_{{w_j}}$.
    An embedding has its largest inner product with vectors like itself, so the words that win are the words that
    were attended to: here the last row leans on *{words[_best]}*, and the readout answers with words near
    *{words[_best]}*. Averaging raw rows can only say "more of what I read".

    But what a word *is* and what it *implies about the next word* are different things. *sat* in the context does
    not make *sat* likely to come next. It makes a place likely. So a row should not send itself. It should send a
    message computed from itself, and the simplest computation is one more matrix:

    $$v_j = x_j\,W_V, \qquad h_i = \sum_{{j \le i}} \alpha_{{ij}}\, v_j, \qquad \text{{as matrices: }} V = X W_V,\quad H = A V$$

    $v_j$ is row $j$'s **value**. Each row now has three vectors derived from it, each with its own job: a query
    (what it looks for), a key (how it is found) and a value (what it sends when read). $W_V$ adds four parameters.
    Two more sliders set it as a turn $\varphi$ and a stretch $\gamma$; at the default, a value is its row turned a
    quarter turn.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    gamma1 = mo.ui.slider(0, 2, step=0.25, value=1, label="Value stretch γ", show_value=True, debounce=True)
    phi1 = mo.ui.slider(-180, 180, step=15, value=90, label="Value turn φ (degrees)", show_value=True, debounce=True)
    mo.hstack([gamma1, phi1], justify="start", gap=2)
    return gamma1, phi1


@app.cell
def _(A1, X, gamma1, phi1, turn):
    WV1 = turn(phi1.value, gamma1.value)

    V1 = X @ WV1  # one value per row
    AV1 = A1 @ V1  # row i: the weighted average of the values of rows j ≤ i
    return AV1, V1, WV1


@app.cell(hide_code=True)
def _(
    A1,
    AV1,
    V1,
    WV1,
    X,
    grid,
    last_row_mask,
    mo,
    n,
    np,
    row_labels,
    show,
    words,
):
    _nums = [str(_i + 1) for _i in range(n)]
    _upper = np.triu(np.ones((n, n)), 1)
    mo.vstack(
        [
            show(
                grid(X, rows=row_labels, name="<i>X</i>"),
                "×",
                grid(WV1, name="<i>W<sub>V</sub></i>"),
                "=",
                grid(V1, name="<i>V</i>", mark=np.ones(V1.shape, dtype=bool)),
            ),
            show(
                grid(A1, rows=row_labels, cols=_nums, name="<i>A</i>", mark=last_row_mask(A1.shape), dim=_upper),
                "×",
                grid(V1, name="<i>V</i>", mark=np.ones(V1.shape, dtype=bool)),
                "=",
                grid(AV1, rows=row_labels, name="<i>A V</i>", mark=last_row_mask(AV1.shape)),
            ),
            mo.md(
                f"$V$ is ${n} \\times 2$: row $j$ is what row $j$ sends to any row that reads it. $AV$ is "
                f"${n} \\times 2$: row $i$ is what row $i$ receives. The last row of $AV$ is "
                + " + ".join(f"{A1[-1, _j]:.3f}·v<sub>{_j + 1}</sub>" for _j in range(n))
                + f", the values of *{' '.join(words)}* mixed in the proportions of the last row of $A$."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Do not throw away what we had

    Should $AV$ *replace* $X$? If it does, a row's own content survives only through the weight it happens to put
    on itself. There is also a cleaner objection. Compare the family of predictors we can express now with the
    family of section 3. With replacement, no choice of $W_V$ gives back the section 3 predictor: $W_V = 0$ gives
    $h_n = 0$, all logits 0, and the uniform distribution. The new family does not contain the old one, so a step
    meant as an improvement could make every model worse.

    Adding instead of replacing fixes both:

    $$h_i = x_i + \sum_{j \le i} \alpha_{ij}\, v_j, \qquad H = X + A V$$

    Now $W_V = 0$ gives exactly the section 3 predictor, and anything else is a correction to it. The logits split
    the same way:

    $$E\,h_n = \underbrace{E\,x_n}_{\text{section 3}} + \underbrace{E\,(AV)_n}_{\text{what attention adds}}$$

    Keeping the input and adding to it is called a **residual connection**.
    """)
    return


@app.cell
def _(AV1, E, X, softmax):
    H1 = X + AV1  # every row: what it was, plus what it gathered

    z1 = E @ H1[-1]
    p1 = softmax(z1)
    return H1, p1, z1


@app.cell(hide_code=True)
def _(
    AV1,
    E,
    H1,
    X,
    f_obs,
    grid,
    last_row_mask,
    mo,
    nll,
    p0,
    p1,
    row_labels,
    show,
    text,
    top_words,
    vocab_table,
    z0,
    z1,
):
    _columns = [
        ("section 3 logit", z0, "{:.3f}"),
        ("+ e<sub>v</sub>·(AV)<sub>n</sub>", E @ AV1[-1], "{:+.3f}"),
        ("= new logit", z1, "{:.3f}"),
        ("section 3 p(v)", p0, "pct"),
        ("with attention p(v)", p1, "pct"),
    ]
    if f_obs is not None:
        _columns.append(("observed f(v)", f_obs, "pct"))
        _verdict = (
            f"On *{text}*, the loss against the observed distribution moves from **{nll(p0, f_obs):.3f}** to "
            f"**{nll(p1, f_obs):.3f}** ({'lower' if nll(p1, f_obs) < nll(p0, f_obs) else 'higher'})."
        )
    else:
        _verdict = f"The data has no observed distribution for *{text}*, so there is no loss to compare."
    mo.vstack(
        [
            show(
                grid(X, rows=row_labels, name="<i>X</i>", mark=last_row_mask(X.shape)),
                "+",
                grid(AV1, name="<i>A V</i>", mark=last_row_mask(X.shape)),
                "=",
                grid(H1, rows=row_labels, name="<i>H</i>", mark=last_row_mask(X.shape)),
            ),
            show(vocab_table(_columns)),
            mo.md(
                f"""
    The most probable next words were {top_words(p0)}. They are now {top_words(p1)}. {_verdict}

    **Nothing guarantees that this is an improvement.** $W_Q$ and $W_V$ were set by hand, and the sliders make the
    loss worse as easily as better. Set the value stretch γ to 0 and the last two probability columns coincide: the
    section 3 predictor is still in there. Finding weights that lower the loss is what training is for, and it
    would proceed exactly as in section 4, with the twelve new numbers added to $\\theta$.
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hidden states

    The rows of $H$ are a new kind of object. $h_i$ is not an input: no lookup in a table produces it. It is not an
    output: it is not a probability. It is a vector the model computes on the way from one to the other. Such
    vectors are called **hidden states**, or **representations**: $h_i$ is the model's representation of position
    $i$ *in its context*.

    Keep the two apart. $x_i$ is the initial representation, a function of one word and one position. $h_i$ is a
    function of every row up to $i$. Everything the model will ever say about the next word has to pass through
    $h_n$, which makes $h_n$ the natural place to look when we ask, in Part II, what the model "knows".
    """)
    return


@app.cell(hide_code=True)
def _(A1, AV1, K1, Q1, X, head_picture, mo, new_fig, row_labels):
    _fig, _ax = new_fig(1, width=5.4, height=4.8)
    head_picture(_ax, X, K1, Q1, A1, AV1, row_labels, title="The last row's view of the head")
    mo.hstack(
        [
            _fig,
            mo.md(
                """
                **The same step as a picture.** Each dot is the key of one row; with $W_K = I$ a key is the row
                itself. The orange arrow is the direction of the last row's query. A key that lies far along that
                direction gets a high score, and the dot's size is the attention weight it ends up with. The black
                arrow is the update the last row receives: from $x_n$ to $h_n = x_n + (AV)_n$.

                Turn θ and the orange arrow swings round to pick out other rows. Raise β and the largest dot takes
                everything. Turn φ and the black arrow rotates without any dot changing size: the weights say where
                to read, the values say what is brought back.
                """
            ),
        ],
        justify="start",
        align="center",
        gap=2,
        widths=[1.1, 1],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 · The whole computation so far, live

    Sections 2 to 8 in one place. Change the context or a slider, and every number below recomputes from the same
    arrays the sections above used. Blue marks the cells that produce the prediction: the last row, the weights it
    reads with, and what it reads.
    """)
    return


@app.cell(hide_code=True)
def _(beta1, context_box, gamma1, mo, phi1, theta1):
    mo.vstack(
        [
            context_box,
            mo.hstack([beta1, theta1, gamma1, phi1], justify="start", gap=1.5, wrap=True),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    A1,
    AV1,
    E,
    H1,
    K1,
    P,
    Q1,
    S1,
    V1,
    WK1,
    WQ1,
    WV1,
    X,
    X_tok,
    f_obs,
    grid,
    ids,
    last_row_mask,
    mo,
    n,
    nll,
    np,
    p0,
    p1,
    row_labels,
    show,
    text,
    vocab_table,
    words,
    z0,
    z1,
):
    _nums = [f"{_i + 1} {_w}" for _i, _w in enumerate(words)]
    _upper = np.triu(np.ones((n, n)), 1)
    _last = last_row_mask(X.shape)
    _all = np.ones(X.shape, dtype=bool)
    _columns = [
        ("baseline p(v)", p0, "pct"),
        ("new logit", z1, "{:.3f}"),
        ("new p(v)", p1, "pct"),
    ]
    if f_obs is not None:
        _columns.append(("observed f(v)", f_obs, "pct"))
        _loss = (
            f"**8 · Against the data.** Loss of the baseline: {nll(p0, f_obs):.3f}. Loss with attention: "
            f"{nll(p1, f_obs):.3f}. The floor for this context is {nll(f_obs, f_obs):.3f}."
        )
    else:
        _loss = (
            f"**8 · Against the data.** *{text}* does not occur in the five sentences, so there is no observed "
            "distribution to compare with."
        )
    mo.vstack(
        [
            mo.md("**1 · Words and ids**"),
            show(grid([words, [str(_k) for _k in ids]], rows=["word", "id"],
                      cols=[f"position {_i + 1}" for _i in range(n)], bracket=False)),
            mo.md("**2 · Embeddings and position offsets**, and **3 · the input matrix**"),
            show(
                grid(X_tok, rows=row_labels, name="<i>E</i>[ids]"), "+", grid(P, name="<i>P</i>"), "=",
                grid(X, rows=row_labels, name="<i>X</i>", mark=_last),
            ),
            mo.md("**4 · Baseline logits and probabilities**, from the last row of $X$"),
            show(
                grid([X[-1]], name="<i>x<sub>n</sub></i>", mark=np.ones((1, 2), dtype=bool)),
                "→",
                vocab_table([("embedding e<sub>v</sub>", E, "vec"), ("baseline logit e<sub>v</sub>·x<sub>n</sub>", z0, "{:.3f}"),
                             ("baseline p(v)", p0, "pct")]),
            ),
            mo.md("**5 · Attention scores and weights**"),
            show(
                grid(WQ1, name="<i>W<sub>Q</sub></i>"), grid(WK1, name="<i>W<sub>K</sub></i>"),
                grid(Q1, rows=row_labels, name="<i>Q</i> = <i>X W<sub>Q</sub></i>", mark=_last),
                grid(K1, name="<i>K</i> = <i>X W<sub>K</sub></i>", mark=_all),
            ),
            show(
                grid(S1, rows=row_labels, cols=_nums, name="<i>Q K</i><sup>⊤</sup>", mark=last_row_mask(S1.shape),
                     dim=_upper),
                "→",
                grid(A1, rows=row_labels, cols=_nums, name="<i>A</i>", mark=last_row_mask(A1.shape), dim=_upper),
            ),
            mo.md("**6 · Values and the updated representations**"),
            show(
                grid(WV1, name="<i>W<sub>V</sub></i>"),
                grid(V1, rows=row_labels, name="<i>V</i> = <i>X W<sub>V</sub></i>", mark=_all),
                grid(AV1, name="<i>A V</i>", mark=_last),
                "→",
                grid(X, rows=row_labels, name="<i>X</i>", mark=_last), "+", grid(AV1, name="<i>A V</i>", mark=_last),
                "=",
                grid(H1, rows=row_labels, name="<i>H</i>", mark=_last),
            ),
            mo.md("**7 · New logits and probabilities**, from the last row of $H$"),
            show(
                grid([H1[-1]], name="<i>h<sub>n</sub></i>", mark=np.ones((1, 2), dtype=bool)),
                "→",
                vocab_table(_columns),
            ),
            mo.md(_loss),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10 · One question per row is not enough

    The construction of sections 6 to 8, with its three matrices, is called an attention **head**. Look at what one
    head gives the last row: one row of weights, and one matrix $W_V$ applied to whatever those weights select.
    That is one question with one way of using the answer.

    Predicting the word after *the cat sat on the* plausibly needs two things from the context: what is being done
    (*sat on*) and who is doing it (*cat*). One head handles that badly, for two reasons that can be stated exactly.

    - The score is linear in the key, $s_{nj} = q_n^{\top} k_j$. If two rows have keys that point in nearly opposite
      directions, no single query can score both highly: whatever raises one lowers the other.
    - Even with the weight split between the two rows, both values pass through the same $W_V$. The message that
      *sat* should send and the message that *cat* should send are produced by one matrix.

    The repair costs no new idea. Run several heads side by side, each with its own $W_Q$, $W_K$, $W_V$, and add
    all their updates to the same row:

    $$H = X + A^{(1)} V^{(1)} + A^{(2)} V^{(2)}$$

    Here is a second head with its own four sliders. At the default it is the first head with both quarter turns
    reversed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    beta2 = mo.ui.slider(0, 6, step=0.5, value=2, label="Head 2: stretch β", show_value=True, debounce=True)
    theta2 = mo.ui.slider(-180, 180, step=15, value=-90, label="turn θ", show_value=True, debounce=True)
    gamma2 = mo.ui.slider(0, 2, step=0.25, value=1, label="value stretch γ", show_value=True, debounce=True)
    phi2 = mo.ui.slider(-180, 180, step=15, value=-90, label="value turn φ", show_value=True, debounce=True)
    mo.hstack([beta2, theta2, gamma2, phi2], justify="start", gap=1.5, wrap=True)
    return beta2, gamma2, phi2, theta2


@app.cell
def _(
    AV1,
    E,
    WK1,
    WQ1,
    WV1,
    X,
    beta2,
    causal_softmax,
    gamma2,
    np,
    phi2,
    softmax,
    theta2,
    turn,
):
    def head(X, WQ, WK, WV):
        """Sections 6 and 7 as one function: queries, keys, values, scores, weights, and the update."""
        Q, K, V = X @ WQ, X @ WK, X @ WV
        S = Q @ K.T
        A = causal_softmax(S)
        return Q, K, V, S, A, A @ V

    WQ2, WK2, WV2 = turn(theta2.value, beta2.value), np.eye(2), turn(phi2.value, gamma2.value)
    heads = [(WQ1, WK1, WV1), (WQ2, WK2, WV2)]  # every attention weight of the model, in one list

    Q2, K2, V2, S2, A2, AV2 = head(X, *heads[1])
    H2 = X + AV1 + AV2  # both heads add to the same rows

    z2 = E @ H2[-1]
    p2 = softmax(z2)
    return A2, AV2, H2, K2, Q2, V2, WQ2, WV2, head, heads, p2


@app.cell(hide_code=True)
def _(
    A1,
    A2,
    AV1,
    AV2,
    H2,
    K1,
    K2,
    Q2,
    V2,
    WQ2,
    WV2,
    X,
    f_obs,
    grid,
    head_picture,
    last_row_mask,
    mo,
    n,
    new_fig,
    nll,
    np,
    num,
    p0,
    p1,
    p2,
    row_labels,
    show,
    text,
    top_words,
    words,
):
    _j1, _j2 = int(np.argmax(A1[-1])), int(np.argmax(A2[-1]))
    _cos = float(K1[_j1] @ K1[_j2] / (np.linalg.norm(K1[_j1]) * np.linalg.norm(K1[_j2])))
    _last = last_row_mask(X.shape)
    _both = grid(
        np.column_stack([A1[-1], A2[-1]]),
        rows=[f"{_i + 1} {_w}" for _i, _w in enumerate(words)],
        cols=["head 1", "head 2"],
        name=f"The last row's weights",
        bracket=False,
        mark=np.column_stack([np.arange(n) == _j1, np.arange(n) == _j2]),
    )
    _fig, _ax = new_fig(1, width=4.6, height=4.2)
    head_picture(_ax, X, K2, Q2, A2, AV2, row_labels, title="The last row's view of head 2")
    _opposed = (
        f"Head 1's favourite is row {_j1 + 1}, *{words[_j1]}*; head 2's is row {_j2 + 1}, *{words[_j2]}*. The "
        f"cosine between those two keys is {num(_cos, '{:+.2f}')}"
        + (", so a single query could not have favoured both." if _cos < 0 else ".")
        if _j1 != _j2
        else f"With these settings both heads favour row {_j1 + 1}, *{words[_j1]}*."
    )
    _cancel = (
        " Row 1 is unchanged: it has only itself to read, and with these two value matrices its two messages "
        "cancel."
        if np.allclose(AV1[0] + AV2[0], 0)
        else ""
    )
    _loss = (
        f" Loss on *{text}*: {nll(p0, f_obs):.3f} with no attention, {nll(p1, f_obs):.3f} with head 1, "
        f"**{nll(p2, f_obs):.3f}** with both."
        if f_obs is not None
        else ""
    )
    mo.vstack(
        [
            show(
                grid(WQ2, name="<i>W<sub>Q</sub></i><sup>(2)</sup>"),
                grid(WV2, name="<i>W<sub>V</sub></i><sup>(2)</sup>"),
                grid(V2, rows=row_labels, name="<i>V</i><sup>(2)</sup>", mark=np.ones(X.shape, dtype=bool)),
                "&nbsp;",
                _both,
            ),
            show(
                grid(X, rows=row_labels, name="<i>X</i>", mark=_last),
                "+",
                grid(AV1, name="<i>A</i><sup>(1)</sup><i>V</i><sup>(1)</sup>", mark=_last),
                "+",
                grid(AV2, name="<i>A</i><sup>(2)</sup><i>V</i><sup>(2)</sup>", mark=_last),
                "=",
                grid(H2, rows=row_labels, name="<i>H</i>", mark=_last),
            ),
            mo.hstack(
                [
                    _fig,
                    mo.md(
                        f"""
    {_opposed}{_cancel}

    The most probable next words are now {top_words(p2)}.{_loss}

    As before, these are hand-set weights and the numbers are whatever they are. What changed for certain is what
    the model *can* express: two questions per row, each with its own way of using the answer. Real models run many
    heads per layer (55 in Huginn), each with queries and keys much shorter than the rows.
    """
                    ),
                ],
                justify="start",
                align="center",
                gap=2,
                widths=[1, 1.2],
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11 · Votes that only add

    Hold the attention weights fixed for a moment and look at how the last row depends on what it read:

    $$E\,h_n = E\,x_n + \sum_{\text{heads}} \sum_{j \le n} \alpha_{nj}\; E\,W_V^{\top} x_j$$

    Each row that is read casts a vote on every word's logit, the vote depends on that row alone, and the votes
    add. That has a consequence. Take four contexts that differ in two places, such as *cat* or *dog* as the
    animal and *sat* or *chased* as the verb, and suppose the weights are the same in all four. Then for every word
    $v$,

    $$z_v(\text{cat}, \text{sat}) + z_v(\text{dog}, \text{chased}) = z_v(\text{cat}, \text{chased}) + z_v(\text{dog}, \text{sat})$$

    because each side adds up the same four votes. A model built this way cannot say "a place is likely when the
    animal is a cat *and* the verb is sat, and not otherwise". Evidence can only accumulate; it cannot interact.
    All the non-linearity we have so far sits in choosing where to read. None is in what is done with what was
    read.

    Adding one more matrix after attention does not help: a linear map of a linear map is a linear map, and the sum
    rule above survives. What is needed is a function that is not a straight line. The simplest place to put one is
    on a single number:

    $$a = w^{\top} h + b, \qquad \text{output } \sigma(a)\; c$$

    This unit measures one combination $a$ of the row's coordinates, bends it through a fixed curve $\sigma$, and
    uses the result as the coefficient of a vector $c$ that it writes back. The unit is called a **neuron**. $a$ is
    its **pre-activation** and $\sigma(a)$ its **activation**; $w$, $b$ and $c$ are its weights. With
    $\sigma(u) = \max(0, u)$, one neuron already computes "both":
    """)
    return


@app.cell
def _():
    def both(a, b):
        return max(0, a + b - 1)  # one neuron: w = (1, 1), bias −1

    {(a, b): both(a, b) for a in (0, 1) for b in (0, 1)}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The output is 1 only when both inputs are 1, which no sum of separate votes can do. A sum of $k$ such neurons,
    applied to each row separately and added to it, is called an **MLP**:

    $$m(h) = \sum_{r=1}^{k} \sigma\big(w_r^{\top} h + b_r\big)\, c_r = \sigma\big(h\,W_{\text{in}} + b\big)\,W_{\text{out}}, \qquad y = h + m(h)$$

    Ours has $k = 2$ neurons (Huginn has 17,920) and uses $\sigma = \tanh$: steep near 0, level beyond about ±0.5.
    The weights are **hand-set**, chosen so that Part II has something to investigate:

    - Neuron 1 reads the second coordinate, shifted by a bias, and writes *against* it with strength $g$.
    - Neuron 2 reads the first coordinate and writes along it, reinforcing it a little.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    gain = mo.ui.slider(0, 4.5, step=0.1, value=2.5, label="Neuron 1's write strength g", show_value=True,
                        debounce=True)
    gain
    return (gain,)


@app.cell
def _(H2, gain, np):
    W_IN = np.array([[0.0, 4.0],
                     [4.0, 0.0]])  # column r: the direction neuron r reads
    B_IN = np.array([0.8, 0.0])  # one bias per neuron

    def w_out(g):
        return np.array([[0.00, -g / 4],  # row r: the vector neuron r writes
                         [0.25, 0.00]])

    def mlp(H, g):
        """Every row separately: two pre-activations, two activations, one written vector."""
        return np.tanh(H @ W_IN + B_IN) @ w_out(g)

    pre2 = H2 @ W_IN + B_IN  # pre-activations: one row per position, one column per neuron
    act2 = np.tanh(pre2)  # activations
    Y1 = H2 + act2 @ w_out(gain.value)  # the same as H2 + mlp(H2, gain.value)
    return B_IN, W_IN, Y1, act2, mlp, pre2, w_out


@app.cell(hide_code=True)
def _(
    B_IN,
    E,
    H2,
    W_IN,
    Y1,
    act2,
    f_obs,
    gain,
    grid,
    last_row_mask,
    mo,
    nll,
    num,
    p2,
    pre2,
    row_labels,
    show,
    softmax,
    text,
    top_words,
    vec,
    w_out,
):
    p3 = softmax(E @ Y1[-1])
    _Wo = w_out(gain.value)
    _h = H2[-1]
    _rows = [
        [
            f"neuron {_r + 1}",
            vec(W_IN[:, _r], "{:.0f}"),
            num(B_IN[_r], "{:.1f}"),
            f"({num(W_IN[0, _r], '{:.0f}')})({num(_h[0], '{:.3f}')}) + ({num(W_IN[1, _r], '{:.0f}')})"
            f"({num(_h[1], '{:.3f}')}) + {num(B_IN[_r], '{:.1f}')} = <b>{num(pre2[-1, _r], '{:.3f}')}</b>",
            f"tanh = <b>{num(act2[-1, _r], '{:.3f}')}</b>",
            vec(_Wo[_r], "{:.3f}"),
            vec(act2[-1, _r] * _Wo[_r], "{:.3f}"),
        ]
        for _r in range(2)
    ]
    _last = last_row_mask(H2.shape)
    _loss = (
        f" Loss on *{text}*: {nll(p2, f_obs):.3f} before the MLP, **{nll(p3, f_obs):.3f}** after."
        if f_obs is not None
        else ""
    )
    mo.vstack(
        [
            show(
                grid(W_IN, name="<i>W</i><sub>in</sub>"),
                grid([B_IN], name="<i>b</i>", fmt="{:.1f}"),
                grid(_Wo, name="<i>W</i><sub>out</sub>", fmt="{:.3f}"),
            ),
            show(
                grid(
                    _rows,
                    cols=["", "reads w", "bias b", "pre-activation a = w·h + b", "activation σ(a)", "writes c",
                          "contribution σ(a)·c"],
                    bracket=False,
                    left=(0, 3),
                    name=f"The two neurons on the last row, h<sub>n</sub> = {vec(_h, '{:.3f}')}",
                )
            ),
            show(
                grid(H2, rows=row_labels, name="<i>H</i>, after attention", mark=_last),
                "→",
                grid(pre2, name="pre-activations", cols=["neuron 1", "neuron 2"], mark=_last),
                "→",
                grid(act2, name="activations", cols=["neuron 1", "neuron 2"], mark=_last),
                "→",
                grid(Y1 - H2, name="<i>m</i>(<i>H</i>)", mark=_last),
                "→",
                grid(Y1, rows=row_labels, name="<i>Y</i> = <i>H</i> + <i>m</i>(<i>H</i>)", mark=_last),
            ),
            mo.md(
                f"""
    Every row goes through the same two neurons, separately: there is no traffic between rows in this step. The
    most probable next words after the MLP are {top_words(p3)}.{_loss} Once more the weights are hand-set, and
    nothing says the MLP should help here.

    **A neuron is not a coordinate of the row.** The row $h$ lives in $\\mathbb R^d$, here $d = 2$. The activations
    live in a different space, $\\mathbb R^k$, with one coordinate per neuron; that $k$ is also 2 in this toy is a
    coincidence of its size. In the activation space the axes are special, because $\\sigma$ acts on each
    coordinate separately. In the row space nothing acts coordinate by coordinate: every matrix that touches a row
    could be re-expressed in a rotated basis without changing what the model computes. That difference matters in
    Part II.
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12 · Reading rows that have already read

    Attention followed by the MLP, each adding to the rows, is one **layer**. Written as a single map, with
    $\theta$ standing for all its weights:

    $$T_\theta(X) = H + m(H), \qquad H = X + \sum_{\text{heads}} A^{(h)} V^{(h)}$$
    """)
    return


@app.cell
def _(Y1, embed, gain, head, heads, mlp, np, text):
    def attend(X, heads):
        """The rows after attention: X plus the update of every head."""
        return X + sum(head(X, *weights)[-1] for weights in heads)

    def layer(X, heads, g):
        """One layer, T_θ: attention, then the MLP, each adding to the rows."""
        H = attend(X, heads)
        return H + mlp(H, g)

    {"layer() reproduces sections 6 to 11": bool(np.allclose(layer(embed(text), heads, gain.value), Y1))}
    return attend, layer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One layer still has a blind spot, and it is the limitation of section 5 one level up. In a single layer, the
    weight $\alpha_{nj}$ compares row $n$ with the *initial* row $j$, and the value $v_j = x_j W_V$ is computed
    from the initial row $j$. So what *sat* sends is the same whoever sat. A row can be read, but only as the bare
    word it started as.

    The repair is to apply a layer to the output of a layer. That is possible because $T$ returns a matrix of the
    same shape it was given, which is the second reason for insisting on that in section 6. In the second pass,
    every key and value is computed from a row that has already gathered from its own context.

    Here is that claim as an experiment. Change only the first word of the context and compare the values that head
    1 sends, in a first pass and in a second.
    """)
    return


@app.cell
def _(X, Y1, embed, gain, heads, layer, words):
    text_var = " ".join([("a" if words[0] != "a" else "the")] + words[1:])  # the context with its first word changed
    X_var = embed(text_var)

    Y2 = layer(Y1, heads, gain.value)  # two passes through the same layer

    WV = heads[0][2]
    V_pass1, V_pass1_var = X @ WV, X_var @ WV  # what each row sends in the first pass
    V_pass2, V_pass2_var = Y1 @ WV, layer(X_var, heads, gain.value) @ WV  # and in the second
    return V_pass1, V_pass1_var, V_pass2, V_pass2_var, Y2, text_var


@app.cell(hide_code=True)
def _(
    V_pass1,
    V_pass1_var,
    V_pass2,
    V_pass2_var,
    grid,
    mo,
    np,
    row_labels,
    show,
    text,
    text_var,
):
    _d1 = ~np.isclose(V_pass1, V_pass1_var)
    _d2 = ~np.isclose(V_pass2, V_pass2_var)
    _labels_var = [f"{_i + 1} {_w}" for _i, _w in enumerate(text_var.split())]
    mo.vstack(
        [
            show(
                grid(V_pass1, rows=row_labels, name=f"first pass, “{text}”", mark2=_d1),
                grid(V_pass1_var, rows=_labels_var, name=f"first pass, “{text_var}”", mark2=_d1),
                "&nbsp;&nbsp;",
                grid(V_pass2, rows=row_labels, name=f"second pass, “{text}”", mark2=_d2),
                grid(V_pass2_var, rows=_labels_var, name=f"second pass, “{text_var}”", mark2=_d2),
            ),
            mo.md(
                f"Orange marks every entry that differs between the two contexts. In the first pass the only row "
                f"that differs is the one whose word changed ({int(_d1.any(axis=1).sum())} of {len(_d1)}). In the "
                f"second pass {int(_d2.any(axis=1).sum())} of {len(_d2)} rows differ: what a row sends now depends "
                "on what came before it."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A transformer

    That is the whole architecture. A **transformer** embeds the context, applies $L$ layers in a row, each with
    its own weights, reads the last row and normalizes:

    $$p(\,\cdot \mid w_1 \dots w_n) = \operatorname{softmax}\!\big(E\,[\,T_{\theta_L} \circ \dots \circ T_{\theta_1}(X)\,]_n\big)$$

    To write text, draw a word from that distribution (below, the most probable one), append it to the context, and
    run the whole computation again on the longer context. That outer loop is called **autoregressive generation**.
    Each turn of it adds one row.
    """)
    return


@app.cell
def _(E, VOCAB, embed, gain, heads, layer, np, softmax, text):
    def next_word_probs(text):
        return softmax(E @ layer(embed(text), heads, gain.value)[-1])

    def generate(text, n_words=3):
        for _ in range(n_words):
            text += " " + VOCAB[np.argmax(next_word_probs(text))]
        return text

    {"the context, continued by three words": generate(text)}
    return


@app.cell(hide_code=True)
def _(
    E,
    H1,
    H2,
    VOCAB,
    X,
    Y1,
    Y2,
    f_obs,
    grid,
    mo,
    nll,
    np,
    show,
    softmax,
    text,
    vec,
):
    _stages = [
        ("section 3: the last row of X", X[-1]),
        ("section 8: one head", H1[-1]),
        ("section 10: two heads", H2[-1]),
        ("section 11: one full layer", Y1[-1]),
        ("this section: the layer twice", Y2[-1]),
    ]
    _seen = [] if f_obs is None else [_v for _v in range(len(VOCAB)) if f_obs[_v] > 0]
    _rows = []
    for _name, _h in _stages:
        _p = softmax(E @ _h)
        _row = [_name, vec(_h, "{:.3f}"), f"{VOCAB[int(np.argmax(_p))]} {_p.max():.0%}"]
        _row += [f"{_p[_v]:.1%}" for _v in _seen]
        _row += [f"{nll(_p, f_obs):.3f}"] if f_obs is not None else []
        _rows.append(_row)
    _cols = ["what is read out", "the row", "top word"] + [f"p({VOCAB[_v]})" for _v in _seen] + (
        ["loss"] if f_obs is not None else []
    )
    _foot = None
    if f_obs is not None:
        _foot = ["observed", "", ""] + [f"{f_obs[_v]:.1%}" for _v in _seen] + [f"{nll(f_obs, f_obs):.3f}"]
    mo.vstack(
        [
            show(grid(_rows, cols=_cols, bracket=False, left=(0,), foot=_foot,
                      name=f"Every stage of Part I on “{text}”")),
            mo.md(
                """
    The generated text above is not English, because almost none of these weights were trained. Where the loss
    in the table falls from one stage to the next, the credit goes to how the hand-set weights were picked (by
    looking at the picture in section 2 and aiming the values of *sat* and *cat* at the places), not to learning.
    Move a slider and it can as easily rise. The mechanics are complete; the training is not. Training would tune
    every number in $\\theta$ by the descent of section 4, on every position of every sentence.

    We have hand-set weights for only one layer, so the last row of the table applies that same layer twice. Hold
    on to that. Using one set of weights again and again is the idea Part II starts from.
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What was forced, what was chosen

    | Section | The problem | The repair | Forced or chosen |
    |---|---|---|---|
    | 2 | Words are names, not quantities | A vector per word, plus a position offset | Numbers are forced; this encoding is chosen |
    | 3 | A matrix is not a probability measure | Score the last row against $E$, then softmax | Non-negative and summing to 1 is forced; the rest is chosen |
    | 4 | The measure is far from the data | Descend the loss over the parameters | The data forces a target; the loss and the descent are chosen |
    | 5–6 | The last row ignores every earlier word | Each row takes a weighted average of earlier rows | Depending on earlier rows is forced; averaging is chosen |
    | 6 | Equal or fixed weights ignore content | Weights from scores between rows | Chosen |
    | 6 | A row scores highest against itself | Queries and keys | Chosen |
    | 7 | Averaging rows repeats the context | Values | Chosen |
    | 8 | The new family lost the old predictor | Add the update instead of replacing | Chosen |
    | 10 | One question per row | Several heads | Chosen |
    | 11 | Evidence can only add up | A non-linear function of each row | Some non-linearity is forced; this one is chosen |
    | 12 | Rows are read as bare words | Apply a layer to the output of a layer | Chosen |

    Left out of the toy: real models rescale the rows before each step to keep their size under control, use
    queries and keys much shorter than the rows with a fourth matrix that maps values back to $d$ coordinates,
    divide the scores by $\sqrt{d_k}$, and give every layer its own weights. None of those changes the logic above.

    The model so far has 22 parameters in $E$, 12 in each head and 10 in the MLP: 56 numbers. Trained language
    models have billions.
    """)
    return


@app.cell(hide_code=True)
def _(
    E,
    VOCAB,
    atoms,
    embed,
    gain,
    heads,
    history_measure,
    layer,
    mo,
    mu_Z_first,
    new_fig,
    pushforward,
):
    def _layered(context_ids):
        return E @ layer(embed(" ".join(VOCAB[_k] for _k in context_ids)), heads, gain.value)[-1]

    _mu = history_measure(_layered, 3)
    _mu_Z = pushforward(_layered, _mu)
    _fig, (_before, _after) = new_fig(2, width=9.2, height=4.3)
    atoms(_before, mu_Z_first)
    _before.set_title(f"Section 3: {len(mu_Z_first):,} possible logit vectors")
    atoms(_after, _mu_Z)
    _after.set_title(f"One layer: {len(_mu_Z):,} possible logit vectors")
    mo.accordion(
        {
            "Aside: the count of section 5, repeated for the layer": mo.vstack(
                [
                    mo.md(
                        f"Of the {len(_mu):,} contexts of three words, the section 3 predictor could produce "
                        f"{len(mu_Z_first)} different logit vectors. With one layer in between, the same contexts "
                        f"produce **{len(_mu_Z):,}**. Contexts that end in the same word no longer collapse. In "
                        "both panels a dot's area is its probability when the model writes the three words itself."
                    ),
                    _fig,
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part II · Investigating a transformer

    Part I ended with a function we can run. Part II asks what happens inside it.

    That may sound like an odd question about a model we built by hand, with all 56 weights on the screen. But try
    to answer this from the weights alone: *when the last row is read, does it know whether an animal has been
    mentioned?* The weights do not say. The answer depends on what the weights do to particular inputs. For a
    trained model with billions of weights the gap is far wider: every weight is known, and inspection of them
    explains almost nothing. So the questions have to be put to the hidden states, as experiments. The toy is the
    place where every answer can be checked.

    ## 13 · What does a hidden state contain?

    Take the hidden state that Part I worked hardest to build: $h_n$, the last row after attention and before the
    MLP. Section 5 complained that the last row knew nothing about the earlier words, and sections 6 to 10 were the
    repair. Did the repair work? Is there information about the earlier words in $h_n$, in a form that can be read
    out?

    Five sentences cannot answer that. There is too little variety in them. So we set up a diagnostic experiment:
    a set of contexts, chosen by us to vary one property, used only to examine the model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    **Diagnostic experiment. This is not training data.** 24 contexts in two families, all five words long.

    - **Family A**, *the/a* + animal + *sat on* + *the/a*, for example *a dog sat on the*. An animal has been
      named, and a place should come next.
    - **Family B**, *on* + *the/a* + place + *sat* + *the/a*, for example *on the mat sat the*. No animal yet, and
      an animal should come next.

    The property to detect is **which family a context belongs to**. Nothing in the model is trained on these
    contexts. The model stays exactly as Part I left it: hand-set.
    """),
        kind="info",
    )
    return


@app.cell
def _(attend, embed, heads, np):
    FAMILY_A = [f"{d1} {animal} sat on {d2}" for d1 in ("the", "a") for animal in ("cat", "dog", "bird")
                for d2 in ("the", "a")]
    FAMILY_B = [f"on {d1} {place} sat {d2}" for d1 in ("the", "a") for place in ("mat", "hat", "grass")
                for d2 in ("the", "a")]
    DIAG = FAMILY_A + FAMILY_B
    diag_y = np.array([1] * len(FAMILY_A) + [0] * len(FAMILY_B))  # 1 = family A: an animal has appeared

    diag_x = np.array([embed(c)[-1] for c in DIAG])  # the last row before any layer: the initial representation
    diag_h = np.array([attend(embed(c), heads)[-1] for c in DIAG])  # the last row after attention: the hidden state
    return DIAG, diag_h, diag_x, diag_y


@app.cell(hide_code=True)
def _(diag_h, diag_x, diag_y, mo, np):
    _values, _inverse = np.unique(diag_x, axis=0, return_inverse=True)
    _inverse = np.ravel(_inverse)
    _best = sum(
        max(int(np.sum((_inverse == _k) & (diag_y == 1))), int(np.sum((_inverse == _k) & (diag_y == 0))))
        for _k in range(len(_values))
    ) / len(diag_y)
    eye_accuracy = float(np.mean((diag_h[:, 1] < 0) == (diag_y == 1)))
    mo.md(
        rf"""
    **Before attention there is nothing to find.** Across the 24 contexts the initial last row $x_n$ takes only
    {len(_values)} distinct values, one for each possible last word, and each value occurs equally often in both
    families. So no function of $x_n$ whatever, linear or not, can identify the family more than {_best:.0%} of the
    time. That is section 5 again, as a measurement.

    **After attention** the last row depends on the whole context. The left panel below shows the 24 hidden
    states $h_n$. With the default weights, family A sits low and family B sits high, so try the simplest possible
    reading: call a context family A when the second coordinate of $h_n$ is negative. That rule is right for
    {eye_accuracy:.0%} of the 24.

    ### Features and probes

    A property of the input that can be computed from a hidden state is called a **feature** of the
    representation. The simplest features are read along a direction:

    $$s(h) = u^{{\top}} h + b$$

    positive for one class and negative for the other. Geometrically $u$ is a direction in the space the rows live
    in, and $s$ measures how far along it the state lies.

    Instead of choosing $u$ by eye, fit it. A **linear probe** is a classifier of exactly this form, trained on
    pairs (hidden state, label). Its loss is the negative log-likelihood of section 4 with two outcomes instead of
    eleven, taking $P(\text{{family A}} \mid h) = 1 / (1 + e^{{-s(h)}})$, and it is trained by the same descent. To
    check that it has found something general, train it on 16 of the 24 contexts and test it on the 8 it has never
    seen.
    """
    )
    return


@app.cell
def _(DIAG, diag_h, diag_y, np):
    def train_probe(H, y, steps=3000, rate=1.0, shrink=1e-3):
        """Fit s(h) = u·h + b by gradient descent, so that 1 / (1 + exp(-s)) matches the labels y."""
        centre = H.mean(axis=0)
        u, b = np.zeros(H.shape[1]), 0.0
        for _ in range(steps):
            p = 1 / (1 + np.exp(-np.clip((H - centre) @ u + b, -60, 60)))  # the probe's probability of class 1
            u = u - rate * ((H - centre).T @ (p - y) / len(y) + shrink * u)
            b = b - rate * np.mean(p - y)
        return u, b - centre @ u

    def accuracy(H, y, u, b):
        return float(np.mean((H @ u + b > 0) == (y == 1)))

    def unit(v):
        """The same direction at length 1 (or zero, if there is no direction to speak of)."""
        length = np.linalg.norm(v)
        return v / length if length > 1e-12 else np.zeros_like(v)

    _order = np.random.default_rng(0).permutation(len(DIAG))
    train_idx, test_idx = _order[:16], _order[16:]

    probe_u, probe_b = train_probe(diag_h[train_idx], diag_y[train_idx])  # the only training in this section
    probe_dir = unit(probe_u)  # the probe's direction, at unit length

    {
        "accuracy on the 16 training contexts": accuracy(diag_h[train_idx], diag_y[train_idx], probe_u, probe_b),
        "accuracy on the 8 held-out contexts": accuracy(diag_h[test_idx], diag_y[test_idx], probe_u, probe_b),
    }
    return (
        accuracy,
        probe_b,
        probe_dir,
        probe_u,
        test_idx,
        train_idx,
        train_probe,
        unit,
    )


@app.cell
def _(E, ID, diag_h, gain, mlp):
    PLACES, ANIMALS = ["mat", "hat", "grass"], ["cat", "dog", "bird"]
    d_place = E[[ID[w] for w in PLACES]].mean(axis=0) - E[[ID[w] for w in ANIMALS]].mean(axis=0)

    def place_preference(row):
        """Mean logit of the three places minus mean logit of the three animals, for a row about to be read out."""
        return row @ d_place

    diag_out = diag_h + mlp(diag_h, gain.value)  # the same 24 last rows after the MLP: what is actually read out
    diag_r = place_preference(diag_out)
    return d_place, diag_r


@app.cell(hide_code=True)
def _(
    BLUE,
    B_IN,
    DIAG,
    INK,
    MUTED,
    ORANGE,
    RULE,
    SURFACE,
    W_IN,
    arrow,
    diag_h,
    diag_r,
    diag_y,
    new_fig,
    np,
    probe_b,
    probe_dir,
    probe_u,
    test_idx,
    text,
):
    _fig, (_ax, _beh) = new_fig(2, width=9.6, height=4.5, width_ratios=[1.3, 1])
    _lim = max(1.1, float(np.abs(diag_h).max()) + 0.25)
    _ax.axhline(0, color=RULE, lw=0.8, zorder=0)
    _ax.axvline(0, color=RULE, lw=0.8, zorder=0)
    _held = np.isin(np.arange(len(DIAG)), test_idx)
    for _cls, _color, _marker, _name in [(1, BLUE, "o", "family A"), (0, ORANGE, "s", "family B")]:
        for _is_test in (False, True):
            _sel = (diag_y == _cls) & (_held == _is_test)
            _ax.scatter(diag_h[_sel, 0], diag_h[_sel, 1], s=46, marker=_marker,
                        facecolors=SURFACE if _is_test else _color, edgecolors=_color, linewidths=1.6, zorder=3,
                        label=_name + (" (held out)" if _is_test else ""))
    _foot = -probe_b * probe_u / (probe_u @ probe_u)  # the point of the boundary nearest the origin
    _along = np.array([-probe_dir[1], probe_dir[0]])
    _ax.plot(*np.column_stack([_foot - 3 * _lim * _along, _foot + 3 * _lim * _along]), color=INK, lw=1.4, zorder=2)
    arrow(_ax, _foot, _foot + 0.45 * _lim * probe_dir, color=INK, lw=1.8, head=10, zorder=4)
    _ax.text(*(_foot + 0.55 * _lim * probe_dir), "u", fontsize=10, style="italic", ha="center", va="center")
    for _r, _style in [(0, (0, (4, 3))), (1, (0, (1, 3)))]:
        _w = W_IN[:, _r]
        _p = -B_IN[_r] * _w / (_w @ _w)
        _t = np.array([-_w[1], _w[0]]) / np.linalg.norm(_w)
        _ax.plot(*np.column_stack([_p - 3 * _lim * _t, _p + 3 * _lim * _t]), color=MUTED, lw=1, ls=_style, zorder=1)
        _ax.text(*(_p + 0.72 * _lim * _t * (-1 if _r == 0 else 1)), f"neuron {_r + 1}\nswitches here", fontsize=7.5,
                 color=MUTED, ha="center", va="center", bbox=dict(fc=SURFACE, ec="none", pad=1))
    if text in DIAG:
        _k = DIAG.index(text)
        _ax.plot(*diag_h[_k], "o", ms=13, mfc="none", mec=INK, mew=1.3, zorder=5)
    _ax.set_xlim(-_lim, _lim)
    _ax.set_ylim(-_lim, _lim)
    _ax.set_aspect("equal")
    _ax.tick_params(length=0, labelsize=8)
    _ax.legend(frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.9, fontsize=7.5, loc="lower left",
               handletextpad=0.2, borderaxespad=0.3)
    _ax.set_title("The hidden state h_n of the 24 contexts, and the probe")

    _rng = np.random.default_rng(1)
    for _cls, _color, _marker, _x in [(1, BLUE, "o", 0), (0, ORANGE, "s", 1)]:
        _vals = diag_r[diag_y == _cls]
        _beh.scatter(_x + _rng.uniform(-0.13, 0.13, len(_vals)), _vals, s=40, marker=_marker, color=_color,
                     alpha=0.85, edgecolors=SURFACE, linewidths=1, zorder=3)
        _beh.plot([_x - 0.25, _x + 0.25], [_vals.mean()] * 2, color=INK, lw=1.6, zorder=4)
    _beh.axhline(0, color=RULE, lw=0.8)
    _beh.set_xticks([0, 1], ["family A\n(place should follow)", "family B\n(animal should follow)"])
    _beh.set_xlim(-0.6, 1.6)
    _beh.tick_params(length=0)
    _beh.set_ylabel("place preference r of the output")
    _beh.set_title("What the model does with it (bars: means)")
    _fig
    return


@app.cell(hide_code=True)
def _(
    B_IN,
    DIAG,
    W_IN,
    accuracy,
    diag_h,
    diag_r,
    diag_y,
    grid,
    mo,
    np,
    num,
    probe_b,
    probe_dir,
    probe_u,
    show,
    test_idx,
    train_idx,
    vec,
):
    _acc_train = accuracy(diag_h[train_idx], diag_y[train_idx], probe_u, probe_b)
    _acc_test = accuracy(diag_h[test_idx], diag_y[test_idx], probe_u, probe_b)
    _w1 = W_IN[:, 0] / np.linalg.norm(W_IN[:, 0])
    _angle = float(np.degrees(np.arccos(min(1.0, abs(float(probe_dir @ _w1))))))
    _pre1 = diag_h @ W_IN[:, 0] + B_IN[0]
    _acc_n1 = float(np.mean((_pre1 > 0) == (diag_y == 1)))
    _acc_n1 = max(_acc_n1, 1 - _acc_n1)
    _a, _b = diag_r[diag_y == 1], diag_r[diag_y == 0]
    _pairs = float(np.mean(_a[:, None] > _b[None, :]))
    _scores = diag_h @ probe_u + probe_b
    _rows = [
        [
            DIAG[_k],
            "A" if diag_y[_k] else "B",
            "held out" if _k in test_idx else "train",
            vec(diag_h[_k], "{:.3f}"),
            num(_scores[_k], "{:+.2f}"),
            "A" if _scores[_k] > 0 else "B",
            num(diag_r[_k], "{:+.3f}"),
        ]
        for _k in range(len(DIAG))
    ]
    _near = (
        f"The probe's direction lies {_angle:.0f}° from the direction neuron 1 reads. In this toy the feature sits "
        f"close to a neuron: the sign of neuron 1's pre-activation alone separates the families in {_acc_n1:.0%} "
        "of the 24 contexts. That closeness is an accident of size, with two neurons in a two-dimensional space."
        if _angle < 30
        else f"The probe's direction lies {_angle:.0f}° from the direction neuron 1 reads, and the sign of neuron "
        f"1's pre-activation alone separates the families in {_acc_n1:.0%} of the 24 contexts. The feature is not "
        "the property of any one neuron."
    )
    mo.vstack(
        [
            mo.md(
                f"""
    **The trained probe** is three numbers: $u$ = {vec(probe_u)} and $b$ = {num(probe_b)}. As a unit direction,
    $u / \\lVert u \\rVert$ = {vec(probe_dir)}. It classifies {_acc_train:.0%} of its 16 training contexts and
    **{_acc_test:.0%} of the 8 held-out contexts** correctly. In the left panel, the black line is where
    $s(h) = 0$ and the arrow is $u$. Filled markers were used for training; hollow ones were held out.

    Be exact about what was trained. Two different trainings appear in this notebook. Section 4 trained a
    *model*, the table $E$, on the five sentences. Here we trained a *probe*, on pairs of a hidden state and a
    label that we assigned. The descent adjusted $u$ and $b$, and nothing else. The model's weights were not
    touched, and the model never sees $u$. A probe is a measuring instrument that we build outside the model and
    hold against one of its hidden states. What it finds is a fact about this hand-set model on these 24
    contexts; nothing here was learned by the model. (The small `shrink` term in the code keeps $u$ finite. Without it,
    perfectly separable data would drive $u$ toward infinity, for the same reason the logits of section 4 kept
    growing.)

    **A feature is not a neuron.** A neuron is a part of the model: it has weights we can look up, and its
    activation is a coordinate of the MLP's hidden vector. A feature is a pattern we go looking for, and the
    direction that reads it is whatever the search finds. {_near} With thousands of dimensions there are vastly
    more directions than neurons, and a feature generally lines up with none of them.

    ### What the probe has shown, and what it has not

    It has shown that the family is **linearly decodable** from $h_n$. The information reached the last row, which
    is what Part I set out to arrange.

    It has not shown that the model *does* anything with that information. For that we have to look at behaviour.
    Define the model's **place preference** as the mean logit of the three places minus the mean logit of the three
    animals. It is a linear function of the row that is read out:

    $$r = d^{{\\top}} y_n, \\qquad d = \\tfrac13(e_\\text{{mat}} + e_\\text{{hat}} + e_\\text{{grass}}) - \\tfrac13(e_\\text{{cat}} + e_\\text{{dog}} + e_\\text{{bird}})$$

    If the model used the family the way English does, $r$ would be clearly higher in family A than in family B.
    The right panel shows $r$ for the same 24 contexts, computed from the output of the layer. The means are
    {num(_a.mean(), "{:+.3f}")} for family A and {num(_b.mean(), "{:+.3f}")} for family B, and an A context has the
    higher $r$ in {_pairs:.0%} of the 144 A–B pairs (50% would mean no relation at all). Compare that with the
    probe, which names the family correctly for {accuracy(diag_h, diag_y, probe_u, probe_b):.0%} of the 24 at
    $h_n$. What can be read from a hidden state and what the model's output reflects are two separate
    measurements. **Decodable is not the same as used.**
    """
            ),
            mo.accordion(
                {
                    "All 24 contexts: hidden state, probe reading, place preference": show(
                        grid(
                            _rows,
                            cols=["context", "family", "probe saw it in", "h_n", "s(h) = u·h + b", "probe says",
                                  "place preference r"],
                            bracket=False,
                            left=(0,),
                        )
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14 · Does the model use it? Interventions

    Observation alone cannot settle whether a direction matters to the model. Two things can vary together without
    one causing the other. The remedy is the one every experimental science uses: change one thing ourselves and
    see what follows. Take the hidden state, move it along the probe's direction, and rerun everything downstream:

    $$h' = h + \alpha\,\hat u, \qquad y' = h' + m(h'), \qquad r' = d^{\top} y'$$

    where $\hat u = u / \lVert u \rVert$. This is a **causal intervention**. Nothing else in the forward pass is
    touched, so whatever changes in the output was caused by the change we made.

    There is a naive prediction to test it against. If $\hat u$ were simply "the family, as the model stores it",
    and nothing downstream reacted to the move, then $r$ would change in a straight line, by $\alpha\,d^{\top}\hat u$.
    Pick a context and a push.
    """)
    return


@app.cell(hide_code=True)
def _(DIAG, mo, text):
    push_context = mo.ui.dropdown(
        ([text] if text not in DIAG else []) + DIAG,
        value=text,
        label="Context to intervene on:",
    )
    push_alpha = mo.ui.slider(-1.5, 1.5, step=0.05, value=0.0, label="Push α along the probe direction",
                              show_value=True, full_width=True)
    mo.vstack([push_context, push_alpha])
    return push_alpha, push_context


@app.cell
def _(
    attend,
    d_place,
    diag_h,
    embed,
    gain,
    heads,
    mlp,
    np,
    probe_b,
    probe_dir,
    probe_u,
    push_context,
    unit,
):
    def after_mlp(h, g):
        """Everything downstream of the hidden state h: the MLP adds to it, and the result is what gets read out."""
        return h + mlp(h[None, :], g)[0]

    def jacobian(func, x, eps=1e-5):
        """The derivative of func at x by nudging: column j is the response to a nudge of coordinate j."""
        return np.column_stack([(func(x + eps * v) - func(x - eps * v)) / (2 * eps) for v in np.eye(len(x))])

    h_site = attend(embed(push_context.value), heads)[-1]  # the hidden state we intervene on

    alphas = np.linspace(-1.5, 1.5, 121)
    pushed = h_site + alphas[:, None] * probe_dir  # h' for every push size
    probe_line = pushed @ probe_u + probe_b  # what the probe reads at each h'
    r_actual = np.array([d_place @ after_mlp(h, gain.value) for h in pushed])  # recomputed downstream
    r_naive = d_place @ after_mlp(h_site, gain.value) + alphas * (d_place @ probe_dir)  # if nothing reacted

    def best_direction(h, g):
        """The direction at h that raises r fastest: the readout direction d, carried back through the MLP."""
        return unit(jacobian(lambda v: after_mlp(v, g), h).T @ d_place)

    cos_all = np.array([best_direction(h, gain.value) @ probe_dir for h in diag_h])
    return (
        after_mlp,
        alphas,
        best_direction,
        cos_all,
        h_site,
        jacobian,
        probe_line,
        r_actual,
        r_naive,
    )


@app.cell(hide_code=True)
def _(
    BLUE,
    B_IN,
    INK,
    MUTED,
    ORANGE,
    RULE,
    SURFACE,
    W_IN,
    after_mlp,
    alphas,
    best_direction,
    cos_all,
    d_place,
    diag_r,
    diag_y,
    gain,
    grid,
    h_site,
    mo,
    new_fig,
    np,
    num,
    probe_b,
    probe_dir,
    probe_line,
    probe_u,
    push_alpha,
    push_context,
    r_actual,
    r_naive,
    show,
    vec,
):
    _a = push_alpha.value
    _g = gain.value
    _h0, _h1 = h_site, h_site + _a * probe_dir
    _s0, _s1 = _h0 @ probe_u + probe_b, _h1 @ probe_u + probe_b
    _y0, _y1 = after_mlp(_h0, _g), after_mlp(_h1, _g)
    _r0, _r1 = d_place @ _y0, d_place @ _y1
    _flip = -_s0 / max(float(np.linalg.norm(probe_u)), 1e-12)  # the push at which the probe changes its verdict
    _r_flip = d_place @ after_mlp(_h0 + _flip * probe_dir, _g)
    _pre = _h0 @ W_IN[:, 0] + B_IN[0]
    _slope = 1 - _g * (1 - np.tanh(_pre) ** 2)
    _best = best_direction(_h0, _g)

    _fig, (_ps, _pr, _pc) = new_fig(3, width=9.8, height=3.5, width_ratios=[1, 1.25, 0.8])
    _ps.axhline(0, color=RULE, lw=0.8)
    _ps.plot(alphas, probe_line, color=INK, lw=2)
    _ps.plot([_a], [_s1], "o", ms=8, color=INK, mec=SURFACE, mew=1.5, zorder=4)
    if abs(_flip) < 1.5:
        _ps.axvline(_flip, color=MUTED, lw=1, ls=(0, (4, 3)))
    _ps.set_xlabel("push α")
    _ps.set_title("What the probe reads")
    _ps.text(0.03, 0.95, "above 0: family A", transform=_ps.transAxes, fontsize=8, color=MUTED, va="top")
    _ps.text(0.03, 0.05, "below 0: family B", transform=_ps.transAxes, fontsize=8, color=MUTED, va="bottom")

    _pr.axhline(_r0, color=RULE, lw=0.8)
    _pr.plot(alphas, r_naive, color=MUTED, lw=1.4, ls=(0, (4, 3)), label="if nothing downstream reacted")
    _pr.plot(alphas, r_actual, color=BLUE, lw=2.2, label="recomputed through the MLP")
    _pr.plot([_a], [_r1], "o", ms=8, color=INK, mec=SURFACE, mew=1.5, zorder=4)
    if abs(_flip) < 1.5:
        _pr.axvline(_flip, color=MUTED, lw=1, ls=(0, (4, 3)))
    _pr.set_xlabel("push α")
    _pr.set_title("What the model does: place preference r")
    _pr.legend(frameon=False, fontsize=8, loc="upper left")

    _rng = np.random.default_rng(2)
    for _cls, _color, _marker in [(1, BLUE, "o"), (0, ORANGE, "s")]:
        _vals = cos_all[diag_y == _cls]
        _pc.scatter(_vals, _cls + _rng.uniform(-0.15, 0.15, len(_vals)), s=34, marker=_marker, color=_color,
                    alpha=0.85, edgecolors=SURFACE, linewidths=1)
    _pc.axvline(0, color=RULE, lw=0.8)
    _pc.set_xlim(-1.08, 1.08)
    _pc.set_ylim(-0.6, 1.6)
    _pc.set_yticks([1, 0], ["A", "B"])
    _pc.set_xlabel("cosine")
    _pc.set_title("Probe direction against\nthe locally best direction")
    for _axis in (_ps, _pr, _pc):
        _axis.tick_params(length=0)

    _other = 1 - int(_s0 > 0)
    _gap = diag_r[diag_y == _other].mean() - diag_r[diag_y == int(_s0 > 0)].mean()
    _rows = [
        ["hidden state h", vec(_h0, "{:.3f}"), vec(_h1, "{:.3f}")],
        ["probe reading s(h)", num(_s0, "{:+.2f}") + (" (A)" if _s0 > 0 else " (B)"),
         num(_s1, "{:+.2f}") + (" (A)" if _s1 > 0 else " (B)")],
        ["neuron activations", vec(np.tanh(_h0 @ W_IN + B_IN), "{:.3f}"), vec(np.tanh(_h1 @ W_IN + B_IN), "{:.3f}")],
        ["row read out, y", vec(_y0, "{:.3f}"), vec(_y1, "{:.3f}")],
        ["place preference r", num(_r0, "{:+.3f}"), num(_r1, "{:+.3f}")],
    ]
    mo.vstack(
        [
            _fig,
            show(grid(_rows, cols=["", "untouched", f"pushed by α = {num(_a, '{:+.2f}')}"], bracket=False, left=(0,),
                      name=f"“{push_context.value}”")),
            mo.md(
                f"""
    **Reading the figure.** Left: the probe's reading moves in a straight line with the push, by construction, and
    crosses zero at α = {num(_flip, "{:+.2f}")} (dashed line). At that push the probe says the context has changed
    family. Middle: the model's place preference at the same pushes. The dashed line is the naive prediction; the
    blue curve is the actual recomputation. At the push that flips the probe, $r$ has moved from
    {num(_r0, "{:+.3f}")} to {num(_r_flip, "{:+.3f}")}, a change of {num(_r_flip - _r0, "{:+.3f}")}. The naive
    straight line predicts {num(_flip * (d_place @ probe_dir), "{:+.3f}")}, and the two families differ in their
    average $r$ by only {num(_gap, "{:+.3f}")} in the first place.

    **Why the curve bends.** The MLP sits between the hidden state and the output. With the default weights,
    neuron 1 reads a direction close to the one we are pushing along (section 13) and writes against it. For a small push the second coordinate of the
    output moves by a factor $1 - g\\,(1 - \\tanh^2 a_1)$ times the push, where $a_1$ is neuron 1's pre-activation.
    For this context $a_1$ = {num(_pre, "{:.3f}")}, so the factor is 1 − {num(_g, "{:.1f}")}·(1 −
    {num(np.tanh(_pre) ** 2, "{:.3f}")}) = **{num(_slope, "{:+.3f}")}**. Where the neuron is saturated the factor
    is near 1 and the push passes through. Where the neuron is in its steep part and $g$ is above 1, the factor
    is negative, and the push comes out *reversed*. The same intervention has different effects in different
    states.

    **Which direction would work.** At a given state, the direction that raises $r$ fastest is the gradient of
    $r$ with respect to $h$. By the chain rule that is the readout direction $d$ carried back through the
    derivative of everything downstream, $(\\partial y / \\partial h)^{{\\top}} d$. For this context it is
    {vec(_best)}, against the probe's {vec(probe_dir)}: cosine {num(float(_best @ probe_dir), "{:+.2f}")}. The
    right panel shows that cosine for all 24 contexts. It ranges from {num(cos_all.min(), "{:+.2f}")} to
    {num(cos_all.max(), "{:+.2f}")}, and in {int(np.sum(cos_all < 0))} of the 24 it is negative: there, pushing
    along the probe direction lowers what the best direction would raise.
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Three conclusions about this model with its default weights, each supported by the numbers above.

    1. **A direction can be highly predictive.** The probe reads the family from $h_n$ without error on contexts it
       never saw.
    2. **Moving along it need not produce the predicted change in behaviour.** The output's response is bent,
       cancelled or reversed by what lies downstream, depending on the state.
    3. **The direction that best changes behaviour is a different object.** It is the readout direction carried
       back through the downstream computation. It depends on the state and on the weights after the site, and it
       coincides with the probe direction only where those happen to agree.

    A probe answers "what can be read here?". An intervention answers "what happens if this changes?". The second
    question is the one about mechanism, and it can only be answered by recomputing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 15 · Why stop after one pass?

    The model of Part I transforms the rows a fixed number of times, once per layer, and the number of layers is
    settled before any input is seen. An easy context and a hard one get the same amount of computation. Why should
    that be? What if the model could keep refining its hidden state for as long as that helped?

    The simplest way to try uses what we already have. Take the one layer and apply it again and again:

    $$H^{(0)} = X, \qquad H^{(k+1)} = T_\theta\big(H^{(k)}\big)$$

    The weights stay the same from pass to pass. Only the state changes. Section 12 took the first two steps of
    this already.
    """)
    return


@app.cell
def _(X, gain, heads, layer, np):
    plain = [X]
    for _ in range(12):
        plain.append(layer(plain[-1], heads, gain.value))  # the same layer, applied to its own output

    {"largest entry of the state after 0, 1, 2, … passes": [round(float(np.abs(H).max()), 2) for H in plain]}
    return (plain,)


@app.cell(hide_code=True)
def _(mo, np, plain):
    _size = [float(np.abs(_H).max()) for _H in plain]
    _grew = _size[-1] > 10 * _size[0]
    mo.md(
        (
            f"**It blows up.** The largest entry of the state goes from {_size[0]:.2f} to {_size[4]:.2f} after four "
            f"passes and {_size[-1]:.1f} after twelve. "
            if _grew
            else f"With these weights the largest entry goes from {_size[0]:.2f} to {_size[-1]:.2f} in twelve "
            "passes. With the default weights it grows without limit. "
        )
        + r"""
    The cause is the residual connection that section 8 introduced: every pass adds its updates to the rows and
    nothing ever takes anything away. After a few passes nothing of the prompt is left to steer by.

    **The repair** is to hand the prompt back in on every pass. Before applying the layer, average the current
    state with $X$:

    $$H^{(k+1)} = F\big(H^{(k)}\big), \qquad F(H) = T_\theta\big(\tfrac12 (H + X)\big)$$

    The average keeps the size in check and the prompt in view. It is a design choice, the simplest of its kind;
    Huginn learns its mixing as a linear map of the state and the embedded prompt. With $H^{(0)} = X$, the first
    step is $F(X) = T_\theta(X)$, so iteration 1 is exactly the one-layer model of Part I. A model that reuses one
    layer this way is called a **looped** or **recurrent-depth** transformer.

    For a fixed prompt, $F$ is one fixed map from $\mathbb R^{n \times d}$ to itself, applied over and over. That
    makes the sequence $H^{(0)}, H^{(1)}, H^{(2)}, \dots$ a **discrete dynamical system**: a state, and a rule
    that produces the next state from the current one. The questions to put to it are the classical ones. Where
    does the state go? Does it settle? If it is disturbed, does the disturbance die out or grow?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    **Two loops. Keep them apart.**

    - **Generation** (section 12) is a loop over *words*. Predict, draw a word, append it, and start again on a
      context that is one row longer. Each turn commits to a word that cannot be taken back.
    - **Recurrence** (this section) is a loop over *iterations* $k$ on one fixed context. The same $n$ rows are
      transformed again and again, and no word is produced. Everything happens in the hidden state.

    A looped model that writes text runs the second loop inside the first: some number of iterations for every
    word it writes.
    """),
        kind="info",
    )
    return


@app.cell
def _(attend, mlp, np):
    def loop_step(S, X, heads, g, a=1.0):
        """One iteration: feed the prompt back in, apply the layer. a < 1 takes only part of the step (damping)."""
        H = attend((S + X) / 2, heads)
        return (1 - a) * S + a * (H + mlp(H, g))

    def run_loop(X, heads, g, loops, a=1.0, seed=None):
        """Iterate from the prompt (or from a random state). Returns every state and every post-attention state."""
        S = X.copy() if seed is None else np.random.default_rng(seed).normal(size=X.shape)
        states, mids = [S], []
        for _ in range(loops):
            H = attend((S + X) / 2, heads)  # the hidden state after attention, inside this iteration
            S = (1 - a) * S + a * (H + mlp(H, g))
            states.append(S)
            mids.append(H)
        return np.array(states), np.array(mids)

    def keeps_moving(states):
        """One answer per row: is it still moving at the end of the run, and no longer slowing down?"""
        late = np.linalg.norm(states[-1] - states[-2], axis=-1)
        earlier = np.linalg.norm(states[-21] - states[-22], axis=-1)
        return (late > 1e-6) & (late > 0.5 * earlier)

    return keeps_moving, loop_step, run_loop


@app.cell(hide_code=True)
def _(gain, mo, n):
    loop_k = mo.ui.slider(0, 40, step=1, value=1, label="Iteration k to inspect", show_value=True, full_width=True)
    start_box = mo.ui.dropdown(
        {"the prompt X": None, "random state 1": 1, "random state 2": 2, "random state 3": 3},
        value="the prompt X",
        label="Start from",
    )
    inspect_row = mo.ui.slider(1, n, step=1, value=n, label="Row whose neurons to show", show_value=True)
    mo.vstack([loop_k, mo.hstack([gain, start_box, inspect_row], justify="start", gap=1.5, wrap=True)])
    return inspect_row, loop_k, start_box


@app.cell
def _(
    B_IN,
    E,
    W_IN,
    X,
    gain,
    heads,
    np,
    probe_b,
    probe_u,
    run_loop,
    softmax,
    start_box,
):
    K_MAX = 400

    loop_states, loop_mids = run_loop(X, heads, gain.value, K_MAX, seed=start_box.value)

    loop_moves = np.linalg.norm(loop_states[1:] - loop_states[:-1], axis=2)  # how far each row moved, per iteration
    loop_probs = softmax(loop_states[:, -1] @ E.T)  # the next-word distribution if we stopped after k iterations
    loop_acts = np.tanh(loop_mids @ W_IN + B_IN)  # every neuron activation, at every iteration, for every row
    loop_probe = loop_mids[:, -1] @ probe_u + probe_b  # the section 13 probe, held against the last row each time
    return (
        loop_acts,
        loop_mids,
        loop_moves,
        loop_probe,
        loop_probs,
        loop_states,
    )


@app.cell(hide_code=True)
def _(
    B_IN,
    VOCAB,
    W_IN,
    f_obs,
    grid,
    inspect_row,
    loop_acts,
    loop_k,
    loop_mids,
    loop_moves,
    loop_probe,
    loop_probs,
    loop_states,
    mo,
    nll,
    np,
    num,
    row_labels,
    show,
    start_box,
    top_words,
    vec,
):
    _k = loop_k.value
    _i = inspect_row.value - 1
    _cur = loop_states[_k]
    _mask = np.zeros(_cur.shape, dtype=bool)
    _mask[_i] = True
    if _k == 0:
        _matrices = show(grid(_cur, rows=row_labels, name="<i>H</i><sup>(0)</sup>, the starting state", mark=_mask))
        _neurons = "At k = 0 nothing has been computed yet: no neuron has fired and there is nothing to probe."
    else:
        _prev = loop_states[_k - 1]
        _matrices = show(
            grid(_prev, rows=row_labels, name=f"<i>H</i><sup>({_k - 1})</sup>", mark=_mask),
            "→",
            grid(_cur, rows=row_labels, name=f"<i>H</i><sup>({_k})</sup>", mark=_mask),
            "&nbsp;",
            grid(_cur - _prev, name=f"<i>H</i><sup>({_k})</sup> − <i>H</i><sup>({_k - 1})</sup>", fmt="{:+.3f}",
                 mark=_mask),
            grid(loop_moves[_k - 1][:, None], name="size of each row's move", fmt="{:.4f}", shape=False),
        )
        _pre = loop_mids[_k - 1, _i] @ W_IN + B_IN
        _neurons = (
            f"**Neurons of row {_i + 1} during iteration {_k}.** Hidden state after attention "
            f"{vec(loop_mids[_k - 1, _i], '{:.3f}')}; pre-activations {vec(_pre, '{:.3f}')}; activations "
            f"**{vec(loop_acts[_k - 1, _i], '{:.3f}')}**"
            + (f" (they were {vec(loop_acts[_k - 2, _i], '{:.3f}')} one iteration earlier)." if _k > 1 else ".")
            + f" **Probe of section 13 on the last row:** s = {num(loop_probe[_k - 1], '{:+.2f}')}, which reads as "
            f"family {'A' if loop_probe[_k - 1] > 0 else 'B'}."
        )

    _shown = list(range(0, 13)) + ([_k] if _k > 12 else [])
    _rows = []
    for _j in _shown:
        _total = float(np.linalg.norm(loop_states[_j] - loop_states[_j - 1])) if _j > 0 else None
        _before = float(np.linalg.norm(loop_states[_j - 1] - loop_states[_j - 2])) if _j > 1 else None
        _rows.append(
            [
                str(_j),
                "·" if _total is None else num(_total, "{:.4f}"),
                "·" if not _before else num(_total / _before, "{:.2f}"),
                vec(loop_states[_j, -1], "{:.3f}"),
                "·" if _j == 0 else vec(loop_acts[_j - 1, _i], "{:.2f}"),
                "·" if _j == 0 else num(loop_probe[_j - 1], "{:+.2f}"),
                f"{VOCAB[int(np.argmax(loop_probs[_j]))]} {loop_probs[_j].max():.0%}",
            ]
            + ([num(nll(loop_probs[_j], f_obs), "{:.3f}")] if f_obs is not None else [])
        )
    _table_mask = np.zeros((len(_rows), len(_rows[0])), dtype=bool)
    _table_mask[_shown.index(_k)] = True
    mo.vstack(
        [
            _matrices,
            mo.md(
                _neurons
                + f"\n\n**Next word if we stopped here:** {top_words(loop_probs[_k])}."
                + (f" Loss against the data: {nll(loop_probs[_k], f_obs):.3f}." if f_obs is not None else "")
            ),
            show(
                grid(
                    _rows,
                    cols=["k", "size of the step ‖H(k) − H(k−1)‖", "ratio to the step before", "last row",
                          f"activations of row {_i + 1}", "probe reading", "top next word"]
                    + (["loss"] if f_obs is not None else []),
                    bracket=False,
                    mark=_table_mask,
                    name=f"Iteration by iteration, starting from {start_box.selected_key}",
                )
            ),
        ]
    )
    return


@app.cell
def _(
    DIAG,
    accuracy,
    diag_y,
    embed,
    gain,
    heads,
    np,
    probe_b,
    probe_u,
    run_loop,
    train_idx,
    train_probe,
):
    K_DIAG = 20
    _runs = [run_loop(embed(c), heads, gain.value, 60) for c in DIAG]
    diag_mids = np.array([mids[:K_DIAG, -1] for _, mids in _runs])  # (24 contexts, 20 iterations, 2): h_n each time
    diag_settled = np.array([states[-1, -1] for states, _ in _runs])  # the last row after 60 iterations

    acc_fixed = [accuracy(diag_mids[:, k], diag_y, probe_u, probe_b) for k in range(K_DIAG)]  # one probe, reused
    acc_refit = [
        accuracy(diag_mids[:, k], diag_y, *train_probe(diag_mids[train_idx, k], diag_y[train_idx], steps=800))
        for k in range(K_DIAG)
    ]  # a new probe fitted at every iteration
    return K_DIAG, acc_fixed, acc_refit, diag_settled


@app.cell(hide_code=True)
def _(
    AQUA,
    BLUE,
    INK,
    K_DIAG,
    MUTED,
    ORANGE,
    RULE,
    SEQ,
    SURFACE,
    VOCAB,
    acc_fixed,
    acc_refit,
    end_label,
    f_obs,
    inspect_row,
    loop_acts,
    loop_moves,
    loop_probe,
    loop_probs,
    mo,
    mpl,
    n,
    new_fig,
    nll,
    np,
    row_labels,
):
    _shown = 40
    _i = inspect_row.value - 1
    _ks = np.arange(1, _shown + 1)

    _fig1, (_mv, _pb) = new_fig(2, width=9.6, height=3.0, width_ratios=[1.25, 1])
    _im = _mv.imshow(loop_moves[:_shown].T, aspect="auto", cmap=SEQ, vmin=0,
                     vmax=max(float(loop_moves[3:_shown].max()), 0.2), extent=(0.5, _shown + 0.5, n - 0.5, -0.5))
    _mv.set_yticks(np.arange(n), row_labels)
    _mv.set_xlabel("iteration k")
    _mv.tick_params(length=0)
    _mv.set_title("How far each row moves at each iteration")
    _fig1.colorbar(_im, ax=_mv, pad=0.01, aspect=12).ax.tick_params(length=0, labelsize=8)
    _top = np.argsort(-loop_probs[1 : _shown + 1].mean(axis=0))[:3]
    for _token, _color in zip(_top, (BLUE, ORANGE, AQUA)):
        _pb.plot(np.arange(0, _shown + 1), loop_probs[: _shown + 1, _token], color=_color, lw=1.8)
        end_label(_pb, _shown, loop_probs[_shown, _token], VOCAB[_token], dx=6)
    _pb.set_xlim(0, _shown + 7)
    _pb.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    _pb.set_ylim(0, None)
    _pb.grid(axis="y", color=RULE, lw=0.6)
    _pb.tick_params(length=0)
    _pb.set_xlabel("iterations before reading out")
    _pb.set_title("Probability of three next words")

    _fig2, (_na, _pr, _ac) = new_fig(3, width=9.6, height=2.9)
    for _r, _color in [(0, BLUE), (1, ORANGE)]:
        _na.plot(_ks, loop_acts[:_shown, _i, _r], "-o", color=_color, lw=1.4, ms=3, label=f"neuron {_r + 1}")
    _na.axhline(0, color=RULE, lw=0.8)
    _na.legend(frameon=False, fontsize=8, loc="center right")
    _na.set_xlim(0, _shown + 1)
    _na.set_ylim(-1.1, 1.1)
    _na.set_xlabel("iteration k")
    _na.set_title(f"Activations of row {_i + 1}")
    _pr.axhline(0, color=MUTED, lw=0.8)
    _pr.plot(_ks, loop_probe[:_shown], "-o", color=INK, lw=1.4, ms=3)
    _pr.set_xlabel("iteration k")
    _pr.set_title("Probe reading of the last row\n(above 0: family A)")
    _kd = np.arange(1, K_DIAG + 1)
    _ac.plot(_kd, acc_refit, "-o", color=AQUA, lw=1.6, ms=3.5, mec=SURFACE, label="refitted at each k")
    _ac.plot(_kd, acc_fixed, "-o", color=INK, lw=1.6, ms=3.5, mec=SURFACE, label="the probe of section 13")
    _ac.axhline(0.5, color=RULE, lw=0.8)
    _ac.legend(frameon=False, fontsize=8, loc="lower right")
    _ac.set_xlim(0, K_DIAG + 1)
    _ac.set_ylim(0.4, 1.04)
    _ac.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    _ac.set_xticks([1, 5, 10, 15, 20])
    _ac.set_xlabel("iteration k")
    _ac.set_title("Probe accuracy on the 24\ndiagnostic contexts")
    for _axis in (_na, _pr, _ac):
        _axis.tick_params(length=0)

    _worst = int(np.argmin(acc_fixed))
    _loss = (
        f" On this context the loss against the data is {nll(loop_probs[1], f_obs):.3f} after one iteration and "
        f"{nll(loop_probs[_shown], f_obs):.3f} after {_shown}."
        if f_obs is not None
        else ""
    )
    mo.vstack(
        [
            _fig1,
            _fig2,
            mo.md(
                f"""
    **What to look at.** Drag *k* and watch the matrix: the state changes at every iteration, by less and less or
    not. The table lists the size of each step. The neurons of a row are recomputed at every iteration from that
    row's current hidden state, so an activation is a function of the iteration as well as of the input.

    **More iterations do not mean a better prediction.**{_loss} Nothing trained these weights so that the place the
    loop goes to is a good answer. In a trained looped model that is exactly what training is for; here we are
    studying the dynamics, not the quality.

    **A probe belongs to the place and the moment it was trained on.** The probe of section 13 was fitted to
    $h_n$ during iteration 1. Held against $h_n$ at later iterations (black curve, bottom right) its accuracy is
    {acc_fixed[0]:.0%} at iteration 1, falls to {acc_fixed[_worst]:.0%} at iteration {_worst + 1}, and is
    {acc_fixed[-1]:.0%} at iteration {K_DIAG}. A probe refitted at each iteration (green) reaches
    {min(acc_refit):.0%} or better throughout. So where the fixed probe fails, the information has not gone: the
    direction that carries it has moved. Reading a model across iterations with one frozen probe measures two
    things at once, and they have to be separated.
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 16 · Does the sequence settle?

    Look at the step sizes in the table of section 15. Whether they shrink toward 0 is the first question to ask
    of any iteration.

    **Fixed points.** Suppose the sequence does converge, to some state $H^*$. Because $F$ is continuous, applying
    it to both sides of $H^{(k)} \to H^*$ gives

    $$H^* = F(H^*)$$

    A state that the map leaves unchanged is called a **fixed point**. So "the loop settles" means precisely "the
    state approaches a fixed point of $F$". To find one even when the plain loop does not settle, take only part
    of each step, $H \leftarrow (1 - a) H + a F(H)$ with $a = 0.3$. That changes how the state moves without
    changing where the fixed points are.
    """)
    return


@app.cell
def _(X, gain, heads, jacobian, keeps_moving, loop_step, np, run_loop):
    H_star = run_loop(X, heads, gain.value, 1500, a=0.3)[0][-1]  # a heavily damped run ends on a fixed point
    residual = float(np.abs(loop_step(H_star, X, heads, gain.value) - H_star).max())  # is F(H*) = H*?

    def F_flat(v):
        """F as a map on one long vector of n·d numbers, so that its derivative is an ordinary matrix."""
        return loop_step(v.reshape(X.shape), X, heads, gain.value).ravel()

    J_full = jacobian(F_flat, H_star.ravel())  # the derivative of F at H*: (n·d) × (n·d)
    eig_full = np.linalg.eigvals(J_full)
    rho = float(np.abs(eig_full).max())  # spectral radius: the largest |eigenvalue|
    stretch = float(np.linalg.norm(J_full, 2))  # the most any small displacement can grow in one step

    _ends = [run_loop(X, heads, gain.value, 400, seed=s)[0] for s in range(1, 21)]
    starts_settled = sum(not keeps_moving(st).any() for st in _ends)
    starts_same = sum(bool(np.abs(st[-1] - H_star).max() < 1e-5) for st in _ends)

    {"|F(H*) − H*|": residual, "spectral radius": round(rho, 4), "largest stretch": round(stretch, 4)}
    return (
        H_star,
        J_full,
        eig_full,
        residual,
        rho,
        starts_same,
        starts_settled,
        stretch,
    )


@app.cell(hide_code=True)
def _(
    H_star,
    J_full,
    X,
    eig_full,
    gain,
    grid,
    heads,
    jacobian,
    keeps_moving,
    loop_moves,
    loop_states,
    loop_step,
    mo,
    n,
    np,
    num,
    residual,
    rho,
    row_labels,
    show,
    starts_same,
    starts_settled,
    stretch,
):
    def own_block(S, X, heads, i, g):
        """J_i: how row i's next state responds to row i's current state."""
        def _row(x):
            S2 = S.copy()
            S2[i] = x
            return loop_step(S2, X, heads, g)[i]

        return jacobian(_row, S[i])

    row_eigs = [np.linalg.eigvals(own_block(H_star, X, heads, _i, gain.value)) for _i in range(n)]
    _blocks_match = bool(
        np.allclose(np.sort_complex(np.concatenate(row_eigs)), np.sort_complex(eig_full), atol=1e-5)
    )
    _upper_zero = all(
        np.abs(J_full[2 * _i : 2 * _i + 2, 2 * _j : 2 * _j + 2]).max() < 1e-6
        for _i in range(n)
        for _j in range(_i + 1, n)
    )
    _total = np.linalg.norm(loop_moves, axis=1)
    _late = [(_total[_k] / _total[_k - 1]) for _k in range(20, 60) if _total[_k - 1] > 1e-11]
    _late_ratio = float(np.median(_late)) if _late else float("nan")
    _settles = not keeps_moving(loop_states).any()

    def _show_eig(value):
        return num(value.real, "{:+.2f}") if abs(value.imag) < 1e-7 else f"{value.real:+.2f}{value.imag:+.2f}i"

    _is_fixed = (
        "It is a fixed point."
        if residual < 1e-8
        else "That is not small enough to call it a fixed point: with these weights the damped run has not "
        "converged, and the rest of this section does not apply."
    )
    _observed = (
        f"In the run of section 15 the step shrinks by a factor of about **{_late_ratio:.2f}** per iteration between "
        f"iterations 20 and 60, to compare with ρ = {rho:.2f}."
        if _settles and np.isfinite(_late_ratio)
        else "The run of section 15 does not settle at the current weights, so there is no shrinking factor to "
        "compare with. Section 17 is about this case."
    )
    if stretch < 1:
        _contract = (f"Here that is **{stretch:.2f}**, below 1, so close to $H^*$ every pair of states is pulled "
                     "together at every step.")
    elif rho < 1:
        _contract = (
            f"Here that is **{stretch:.2f}**, above 1. So $F$ is *not* a contraction in the ordinary sense of "
            "distance, even right at its fixed point: two nearby states can move apart for a step before they come "
            "together. What the eigenvalues promise is weaker and still sufficient near $H^*$: because ρ is below "
            "1, there is another way of measuring distance, adapted to $J$, in which $F$ does contract there."
        )
    else:
        _contract = (
            f"Here that is **{stretch:.2f}**, and ρ itself is above 1. With the current weights $F$ is not a "
            "contraction near $H^*$ under any way of measuring distance: some displacements grow at every step."
        )
    mo.vstack(
        [
            show(
                grid(H_star, rows=row_labels, name="<i>H</i>*, the fixed point", fmt="{:.3f}"),
                "&nbsp;",
                grid(J_full, name="<i>J</i>, the derivative of <i>F</i> at <i>H</i>*", fmt="{:+.2f}",
                     dim=np.abs(J_full) < 5e-7),
            ),
            mo.md(
                rf"""
    The damped run ends at the state on the left, and $F$ moves it by at most {residual:.0e}. {_is_fixed}

    **Does every start lead there?** Run the plain loop from 20 random starting states.
    {starts_settled} of the 20 settle, and {starts_same} of them arrive at this same $H^*$.

    **Why a fixed point attracts: linearize.** Close to $H^*$ a smooth map is nearly linear:

    $$F(H^* + \delta) \approx H^* + J\,\delta$$

    where $J$ is the **Jacobian**, the matrix of all partial derivatives of $F$ at $H^*$. With the state written
    as one long vector of $n \cdot d = {2 * n}$ numbers, $J$ is the ${2 * n} \times {2 * n}$ matrix on the right,
    computed by nudging each coordinate in turn. After $k$ iterations a small displacement $\delta$ has become
    $J^k \delta$, and what happens to that is decided by the eigenvalues of $J$. If every eigenvalue has
    $|\lambda| < 1$, then $J^k \delta \to 0$: the fixed point is **stable**, and in the long run displacements
    shrink by a factor of about $\rho = \max |\lambda|$ per iteration. If some eigenvalue has $|\lambda| > 1$,
    displacements along its direction grow and the fixed point repels.

    The eigenvalues here are {", ".join(_show_eig(_v) for _v in sorted(eig_full, key=lambda _v: -abs(_v)))}, so
    **ρ = {rho:.3f}**. {_observed}

    **The causal mask shapes $J$.** Row $i$'s next state depends on rows 1 to $i$ and on nothing later, so every
    block of $J$ above the diagonal is zero (the faded entries; checked: `{_upper_zero}`). $J$ is block
    lower-triangular, and its eigenvalues are simply the eigenvalues of its $n$ diagonal blocks taken together
    (checked: `{_blocks_match}`). Each row therefore has two eigenvalues of its own, which section 17 lists.

    **Contraction mappings.** There is a stronger property that would settle everything at once. $F$ is a
    **contraction** on a region if some $q < 1$ satisfies $\lVert F(a) - F(b) \rVert \le q\,\lVert a - b \rVert$
    for all $a, b$ in it. Banach's fixed-point theorem says that a contraction mapping a closed region into itself
    has exactly one fixed point there, and that the iteration converges to it from every start in the region, with
    an error that shrinks like $q^k$. Does $F$ qualify? Near $H^*$ the best possible $q$, measuring distance in
    the ordinary way, is the largest factor by which $J$ can stretch a vector. {_contract} Far from $H^*$ nothing
    is promised at all. $\tanh$ and softmax make $F$ non-linear, the theorem's assumption has to be checked and
    cannot be assumed, and the random starts above are an experiment, not a proof.

    **Stability is forgetting.** Read the linearization once more, as a statement about interference. Disturb a
    settled state by $\delta$, and after $k$ more iterations what is left of the disturbance is $J^k \delta$. A
    stable loop erases what is done to its state. Section 18 depends on this.
    """
            ),
        ]
    )
    return own_block, row_eigs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 17 · When it does not settle

    Neuron 1 writes against what it reads, with strength $g$. So far $g$ has been a number we set and left alone.
    Sweep it. For each value of $g$, run the loop 300 times from the prompt and record where the state is on the
    last two iterations.
    """)
    return


@app.cell
def _(X, heads, keeps_moving, np, run_loop):
    G_SWEEP = np.round(np.linspace(1.5, 4.5, 41), 3)
    _tails = [run_loop(X, heads, g, 300)[0][-22:] for g in G_SWEEP]  # the end of each run
    sweep_ends = np.array([tail[-2:] for tail in _tails])  # (41 values of g, 2 states, n, d)
    sweep_moving = np.array([keeps_moving(tail) for tail in _tails])  # (41 values of g, n rows)
    return G_SWEEP, sweep_ends, sweep_moving


@app.cell(hide_code=True)
def _(mo):
    damp_a = mo.ui.slider(0.5, 1.0, step=0.05, value=1.0, label="Step size a (1 = no damping)", show_value=True,
                          debounce=True)
    damp_where = mo.ui.radio(["every row", "only the source row"], value="every row", inline=True, label="Damp:")
    return damp_a, damp_where


@app.cell
def _(
    E,
    X,
    damp_a,
    damp_where,
    gain,
    heads,
    keeps_moving,
    loop_states,
    n,
    np,
    run_loop,
    softmax,
):
    still_moving = keeps_moving(loop_states)
    source = int(np.argmax(still_moving)) if still_moving.any() else None  # the first row that has not settled

    step_sizes = np.ones((n, 1))
    if damp_where.value == "every row":
        step_sizes[:] = damp_a.value
    elif source is not None:
        step_sizes[source] = damp_a.value

    damped_states, _ = run_loop(X, heads, gain.value, 400, a=step_sizes)
    damped_moves = np.linalg.norm(damped_states[1:] - damped_states[:-1], axis=2)
    damped_probs = softmax(damped_states[:, -1] @ E.T)
    return (
        damped_moves,
        damped_probs,
        damped_states,
        source,
        step_sizes,
        still_moving,
    )


@app.cell(hide_code=True)
def _(
    BLUE,
    G_SWEEP,
    INK,
    VOCAB,
    gain,
    inspect_row,
    loop_acts,
    loop_probs,
    loop_states,
    mo,
    n,
    new_fig,
    np,
    num,
    row_labels,
    source,
    still_moving,
    sweep_ends,
    sweep_moving,
    words,
):
    _i = inspect_row.value - 1
    _fig, (_ax, _all) = new_fig(2, width=9.6, height=3.3, width_ratios=[1.25, 1])
    for _phase in (0, 1):
        _ax.plot(G_SWEEP, sweep_ends[:, _phase, _i, 1], "o", ms=3.2, color=BLUE)
    _ax.axvline(gain.value, color=INK, lw=1)
    _ax.text(gain.value, _ax.get_ylim()[1], f" g = {gain.value:.1f}", fontsize=8.5, va="top")
    _ax.set_xlabel("neuron 1's write strength g")
    _ax.set_ylabel("second coordinate")
    _ax.set_title(f"Row {_i + 1} on the last two of 300 iterations")
    _im = _all.imshow(sweep_moving.T, aspect="auto", cmap="Blues", vmin=0, vmax=1.6,
                      extent=(G_SWEEP[0], G_SWEEP[-1], n - 0.5, -0.5))
    _all.axvline(gain.value, color=INK, lw=1)
    _all.set_yticks(np.arange(n), row_labels)
    _all.set_xlabel("g")
    _all.set_title("Rows that never stop moving")
    for _axis in (_ax, _all):
        _axis.tick_params(length=0)

    _moving = np.where(still_moving)[0]
    _period_two = bool(np.abs(loop_states[-1] - loop_states[-3]).max() < 1e-5)
    _pa, _pb = loop_probs[-2], loop_probs[-1]
    if len(_moving) == 0:
        _text = (f"**Settled at g = {gain.value:.1f}.** After 400 iterations no row moves, so the next-word "
                 f"distribution is the same at every further iteration: “{VOCAB[_pb.argmax()]}” leads with "
                 f"{_pb.max():.0%}. Raise g (the slider is in section 15, and again below).")
        _kind = "success"
    elif _period_two:
        _rows = ", ".join(f"{_k + 1} (“{words[_k]}”)" for _k in _moving)
        _text = (f"**A period-two orbit at g = {gain.value:.1f}.** Rows {_rows} keep moving and return to the same "
                 f"state every second iteration. Every row before row {source + 1} has settled, and a row reads "
                 f"only earlier rows, so row {source + 1} is not inheriting its motion. It is a source. Neuron 1 "
                 f"of that row alternates between activations {num(loop_acts[-2, source, 0], '{:+.2f}')} and "
                 f"{num(loop_acts[-1, source, 0], '{:+.2f}')}. ")
        if _pa.argmax() != _pb.argmax():
            _text += (f"The most probable next word **depends on when you stop**: “{VOCAB[_pa.argmax()]}” "
                      f"({_pa.max():.0%}) after 399 iterations, “{VOCAB[_pb.argmax()]}” ({_pb.max():.0%}) after "
                      "400.")
        else:
            _top = _pb.argmax()
            _text += (f"The most probable next word is “{VOCAB[_top]}” on both halves of the orbit, but its "
                      f"probability alternates between {_pa[_top]:.1%} and {_pb[_top]:.1%}.")
        _kind = "warn"
    else:
        _text = (f"**Still moving at g = {gain.value:.1f}.** After 400 iterations some rows have neither settled "
                 "nor fallen into a period-two orbit.")
        _kind = "neutral"
    mo.vstack(
        [
            _fig,
            mo.md(
                "Left: one dot per iteration. Where the two dots coincide the row has settled. Where they separate, "
                "the row alternates between two states for ever: a **period-two orbit**. Right: for every row, the "
                "values of g at which it is still moving at the end of the run and no longer slowing down. The "
                "vertical line marks the current value of g."
            ),
            mo.callout(mo.md(_text), kind=_kind),
        ]
    )
    return


@app.cell(hide_code=True)
def _(damp_a, damp_where, gain, mo):
    mo.vstack(
        [
            mo.md(
                r"""
    ### Why a row flips

    The fixed point is still there when the loop stops settling. It has become unstable. The table below lists each
    row's own two eigenvalues at the fixed point, which the damped run of section 16 finds whether or not the plain
    loop settles.

    - If both are inside the unit circle, a displacement of that row shrinks, as long as the earlier rows hold
      still.
    - If one is real and below −1, a displacement along its direction changes sign and grows at every iteration.
      The growth stops only when $\tanh$ saturates, and the row ends up alternating between two states. Rows that
      read it through attention alternate with it, even if their own eigenvalues are harmless.

    In the toy the eigenvalue below −1 comes from neuron 1. Where its pre-activation is near 0 it answers a
    displacement of the second coordinate with $-g$ times that displacement, so the row's block has an eigenvalue
    near $\tfrac12(1 - g)$ times what attention contributes. Which rows sit in that steep part depends on the
    text.

    **Damping** replaces the step by $H \leftarrow (1 - a)\,H + a\,F(H)$. It keeps every fixed point and moves each
    eigenvalue $\lambda$ to $1 - a + a\lambda$, so a small enough $a$ brings an eigenvalue back above −1; the
    crossing is at $a^* = 2 / (1 - \lambda)$. Try it on every row, then on the source row alone.
    """
            ),
            mo.hstack([gain, damp_a, damp_where], justify="start", gap=1.5, wrap=True),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    BLUE,
    ORANGE,
    RULE,
    SEQ,
    SURFACE,
    VOCAB,
    damp_a,
    damped_moves,
    damped_probs,
    damped_states,
    end_label,
    gain,
    grid,
    keeps_moving,
    mo,
    mpl,
    n,
    new_fig,
    np,
    row_eigs,
    row_labels,
    show,
    source,
    step_sizes,
    words,
):
    def _show_eig(value):
        return f"{value.real:+.2f}".replace("-", "−") if abs(value.imag) < 1e-7 else f"{value.real:+.2f}{value.imag:+.2f}i"

    def _verdict(values):
        if np.abs(values).max() < 1:
            return "settles on its own"
        if np.any((np.abs(values.imag) < 1e-7) & (values.real < -1)):
            return "<b>flips</b>: a real eigenvalue below −1"
        return "does not settle"

    _rows = []
    for _k in range(n):
        _vals = sorted(row_eigs[_k], key=lambda _v: _v.real)
        _flip = [_v.real for _v in _vals if abs(_v.imag) < 1e-7 and _v.real < -1]
        _damped = [1 - step_sizes[_k, 0] + step_sizes[_k, 0] * _v for _v in _vals]
        _rows.append(
            [
                f"{_k + 1} {words[_k]}",
                ", ".join(_show_eig(_v) for _v in _vals),
                _verdict(row_eigs[_k]),
                f"{2 / (1 - _flip[0]):.2f}" if _flip else "·",
                f"{step_sizes[_k, 0]:.2f}",
                ", ".join(_show_eig(_v) for _v in _damped),
            ]
        )

    _fig, (_mv, _co, _pr) = new_fig(3, width=9.8, height=3.2, width_ratios=[1.3, 1, 1])
    _shown = 48
    _im = _mv.imshow(damped_moves[:_shown].T, aspect="auto", cmap=SEQ, vmin=0,
                     vmax=max(float(damped_moves[4:].max()), 0.2), extent=(0.5, _shown + 0.5, n - 0.5, -0.5))
    _mv.set_yticks(np.arange(n), row_labels)
    _mv.set_xlabel("iteration")
    _mv.set_title("Movement with the damping above")
    _last = np.arange(len(damped_states) - 8, len(damped_states))
    _watch = [n - 1] if source in (None, n - 1) else [source, n - 1]
    for _k, _color in zip(_watch, (BLUE, ORANGE)):
        _co.plot(_last, damped_states[_last, _k, 1], "-o", color=_color, lw=2, ms=5, mec=SURFACE, mew=1.2)
        end_label(_co, _last[-1], damped_states[-1, _k, 1], row_labels[_k], dx=7)
    _co.set_xlim(_last[0] - 0.4, _last[-1] + 2.6)
    _seen = damped_states[_last][:, _watch, 1]
    _mid, _half = (_seen.max() + _seen.min()) / 2, max(0.7 * (_seen.max() - _seen.min()), 0.08)
    _co.set_ylim(_mid - _half, _mid + _half)
    _co.grid(axis="y", color=RULE, lw=0.6)
    _co.set_xticks(_last[::2])
    _co.set_xlabel("iteration")
    _co.set_title("Second coordinate, last 8 iterations")
    _top = np.argsort(-damped_probs[-2:].mean(axis=0))[:2]
    for _token, _color in zip(_top, (BLUE, ORANGE)):
        _pr.plot(_last, damped_probs[_last, _token], "-o", color=_color, lw=2, ms=5, mec=SURFACE, mew=1.2)
        end_label(_pr, _last[-1], damped_probs[-1, _token], VOCAB[_token], dx=7)
    _pr.set_xlim(_last[0] - 0.4, _last[-1] + 2.6)
    _pr.set_xticks(_last[::2])
    _pr.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    _pr.set_ylim(0, max(0.3, float(damped_probs[_last][:, _top].max()) + 0.06))
    _pr.grid(axis="y", color=RULE, lw=0.6)
    _pr.set_xlabel("iterations before reading out")
    _pr.set_title("Two leading next words")
    for _axis in (_mv, _co, _pr):
        _axis.tick_params(length=0)

    _still = [f"{_k + 1}" for _k in np.where(keeps_moving(damped_states))[0]]
    mo.vstack(
        [
            show(
                grid(
                    _rows,
                    cols=["row", "its two eigenvalues", "on its own", "damping needed a*", "step size a now",
                          "eigenvalues after damping"],
                    bracket=False,
                    left=(0, 2),
                    name=f"Each row's own block at the fixed point, g = {gain.value:.1f}",
                )
            ),
            _fig,
            mo.md(
                (
                    f"With the damping set above (a = {damp_a.value:.2f}), "
                    + ("no row is moving after 400 iterations." if not _still
                       else f"rows {', '.join(_still)} are still moving after 400 iterations.")
                )
                + " A row that moves in the left panel while its own eigenvalues say it settles is being driven by "
                "an earlier row."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Aside: what replaces a fixed point (invariant measures)": mo.md(
                r"""
                When the state settles, "where the loop ends up" is a point, $H^*$. On a period-two orbit there is
                no such point, but there is still a well-defined answer to "where is the state, on average, in the
                long run?": half the time at one state $a$ of the orbit and half the time at the other, $b$. The
                measure $\mu = \tfrac12 \delta_a + \tfrac12 \delta_b$ is **invariant** under $F$: push it forward
                through $F$ and you get it back, because $F$ swaps $a$ and $b$. A fixed point is the special case
                $\mu = \delta_{H^*}$.

                The practical reading is modest. On an orbit, any quantity computed from the state has two values,
                one per phase, and its long-run average is the mean of the two. A reported probability for a
                looped model on an orbit therefore needs either the iteration count it was read at or the average
                over both phases.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 18 · Pushing on a looped model

    Section 14 intervened on a model that makes one pass, where "downstream" meant the MLP and the readout. In a
    looped model, what lies downstream of a hidden state is the loop itself, as many more times as it runs. This
    section repeats the experiment of section 14 there. The state we push is the last row between iterations, and
    the behaviour is again the place preference $r = d^{\top} h_n$ of the row that is read out.

    ### A push given once

    Let the loop settle at $H^*$, add a vector $P$ to its last row, and keep iterating. By section 16 the
    displacement after $t$ more iterations is $J^t P$, where $J$ is now the last row's own block (pushing the last
    row changes no earlier row). At a stable fixed point that goes to 0. The stability that makes the loop settle
    is the same property that erases a one-off edit.

    ### A push given at every iteration

    To move where the state settles, the push has to be applied each time: $H^{(k+1)} = F(H^{(k)} + P)$. To first
    order the readout then changes by

    $$\text{at once: } d^{\top} P \qquad\qquad \text{after } T \text{ iterations: } d^{\top}\big(J + J^2 + \dots + J^T\big)\,P$$

    and as $T$ grows the sum tends to $(I - J)^{-1} J$. The unit direction that makes the second quantity largest
    is not $d$. It is proportional to

    $$\big(J + J^2 + \dots + J^T\big)^{\top} d$$

    the readout direction carried back through every remaining iteration of the loop. Call it the **lever**. It is
    the same object as the "locally best direction" of section 14, with the loop in the place of the MLP.

    ### Four directions, one test

    Which direction should we push along to raise $r$? Here are four candidates, each at unit length, tested by
    running the loop with a push of length 0.03 for 24 iterations and recomputing the output.

    - The **probe direction**: a probe for the family, fitted to the settled last rows of the 24 diagnostic
      contexts. It says where the two families differ.
    - The **difference of class means** of those same states, the simplest diagnostic direction there is.
    - The **readout gradient** $d$: the direction that raises $r$ fastest right now.
    - The **lever**.
    """)
    return


@app.cell
def _(
    H_star,
    X,
    d_place,
    diag_settled,
    diag_y,
    gain,
    heads,
    loop_step,
    n,
    np,
    own_block,
    train_probe,
    unit,
):
    T_PUSH, PUSH_SIZE = 24, 0.03

    J_last = own_block(H_star, X, heads, n - 1, gain.value)  # the last row's own derivative at the fixed point
    powers = np.array([np.linalg.matrix_power(J_last, t) for t in range(1, T_PUSH + 1)])  # J, J², …, J^T

    loop_probe_u, loop_probe_b = train_probe(diag_settled, diag_y)  # a probe for this site: the settled last row
    directions = {
        "probe direction": unit(loop_probe_u),
        "difference of class means": unit(diag_settled[diag_y == 1].mean(axis=0) - diag_settled[diag_y == 0].mean(axis=0)),
        "readout gradient": unit(d_place),
        "lever": unit(powers.sum(axis=0).T @ d_place),
    }

    def predicted(P):
        """First-order change of r after 1, 2, …, T iterations of pushing."""
        return np.cumsum(powers @ P, axis=0) @ d_place

    def measured(P, every=True):
        """Run the loop from the fixed point, adding P to the last row before every iteration (or only the first)."""
        S, changes = H_star.copy(), []
        for t in range(T_PUSH):
            pushed = S.copy()
            if every or t == 0:
                pushed[-1] += P
            S = loop_step(pushed, X, heads, gain.value)
            changes.append(d_place @ (S[-1] - H_star[-1]))
        return np.array(changes)

    return (
        J_last,
        PUSH_SIZE,
        T_PUSH,
        directions,
        loop_probe_b,
        loop_probe_u,
        measured,
        predicted,
    )


@app.cell(hide_code=True)
def _(
    AQUA,
    BLUE,
    J_last,
    MUTED,
    ORANGE,
    PUSH_SIZE,
    RULE,
    SURFACE,
    T_PUSH,
    accuracy,
    arrow,
    d_place,
    diag_settled,
    diag_y,
    directions,
    gain,
    grid,
    keeps_moving,
    loop_probe_b,
    loop_probe_u,
    loop_states,
    measured,
    mo,
    new_fig,
    np,
    num,
    plane,
    predicted,
    show,
    text,
    vec,
):
    _settled = bool(np.abs(np.linalg.eigvals(J_last)).max() < 1) and not keeps_moving(loop_states).any()
    _colors = {"probe direction": AQUA, "difference of class means": MUTED, "readout gradient": ORANGE, "lever": BLUE}
    _fig, (_ax, _cv) = new_fig(2, width=9.6, height=4.0, width_ratios=[1, 1.4])
    plane(_ax, labels=False, alpha=0.2)
    for _name, _dir in directions.items():
        arrow(_ax, (0, 0), 1.3 * _dir, color=_colors[_name], lw=2.2)
    _ax.set_title("Four unit directions for the last row")

    _t = np.arange(0, T_PUSH + 1)
    _rows = []
    for _name, _dir in directions.items():
        _P = PUSH_SIZE * _dir
        _pred = np.concatenate([[d_place @ _P], predicted(_P)])
        _meas = np.concatenate([[d_place @ _P], measured(_P)])
        _cv.plot(_t, _pred, color=_colors[_name], lw=2, label=_name)
        _cv.plot(_t, _meas, "o", ms=4, color=_colors[_name], mec=SURFACE, mew=0.8)
        _rows.append(
            [_name, vec(_dir), num(float(_dir @ directions["lever"]), "{:+.2f}"), num(d_place @ _P, "{:+.4f}"),
             num(_meas[1], "{:+.4f}"), num(_meas[-1], "{:+.4f}"), num(_pred[-1], "{:+.4f}")]
        )
    _cv.axhline(0, color=RULE, lw=0.8)
    _cv.legend(frameon=False, fontsize=8, loc="upper right", ncols=2)
    _cv.set_xlim(-0.6, T_PUSH + 0.6)
    _cv.set_xticks([0, 4, 8, 12, 16, 20, 24])
    _cv.grid(axis="y", color=RULE, lw=0.6)
    _cv.tick_params(length=0)
    _cv.set_xlabel("iterations with the push applied (0 = at once)")
    _cv.set_title("Change in place preference r: formula (line), actual loop (dots)")

    _once = measured(0.1 * directions["readout gradient"], every=False)
    _acc = accuracy(diag_settled, diag_y, loop_probe_u, loop_probe_b)
    _g_row, _l_row, _p_row = _rows[2], _rows[3], _rows[0]
    _warning = mo.callout(
        mo.md(f"With this text and these weights the loop does not settle at g = {gain.value:.1f}, so there is no "
              "stable fixed point to push on and the first-order formula does not apply. Lower g in section 15."),
        kind="warn",
    )
    _body = [
        mo.md(
            f"""
    **A push given once** (length 0.1 along the readout gradient, on *{text}*). At once, $r$ rises by
    {num(0.1 * np.linalg.norm(d_place), "{:+.3f}")}. Over the next six iterations what remains of that is
    {", ".join(num(_v, "{:+.3f}") for _v in _once[:6])}. The edit is forgotten.

    **A push given at every iteration.** The last row's own block at the fixed point is $J$ = {vec(J_last[0])} /
    {vec(J_last[1])} (its two rows), with eigenvalues
    {", ".join(num(_v.real, "{:+.2f}") for _v in np.linalg.eigvals(J_last))}.
    """
        ),
        show(
            grid(
                _rows,
                cols=["direction", "unit vector", "cosine with the lever", "change in r at once",
                      "after 1 iteration", f"after {T_PUSH} iterations", "formula predicts"],
                bracket=False,
                left=(0,),
                name=f"Pushes of length {PUSH_SIZE} on the last row, g = {gain.value:.1f}",
            )
        ),
        _fig,
        mo.md(
            f"""
    The probe for this site classifies {_acc:.0%} of the 24 settled states correctly, so its direction is a real
    summary of where the families differ. As a push it changes $r$ by {_p_row[5]} after {T_PUSH} iterations.

    The readout gradient wins at once, by construction: {_g_row[3]}. After {T_PUSH} iterations of the loop it
    stands at **{_g_row[5]}**. The lever starts at {_l_row[3]} and ends at **{_l_row[5]}**. The dots (the actual
    loop) lie on the lines (the formula) because a push of this size is small enough for the linearization to
    hold; a longer push departs from it.

    With the default weights the sign change has a plain cause. $r$ depends mostly on the second coordinate, and
    the loop answers a push on the second coordinate by pushing back harder, through neuron 1. So the direction
    that raises $r$ at once lowers it once the loop has responded, and the direction that works looks wrong on the
    first step.
    """
        ),
    ]
    mo.vstack(_body if _settled else [_warning] + _body)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The three conclusions of section 14 hold again, in a stronger form.

    1. **A direction can be highly predictive.** A probe reads the family from the settled state.
    2. **Moving along it need not cause the predicted change.** A push given once is erased by the stability of
       the fixed point. A push given every iteration is reshaped by the loop: its lasting effect is
       $d^{\top}(I - J)^{-1} J\,P$, which can be small, or have the opposite sign to its immediate effect.
    3. **The direction that steers is a different object.** It is the readout direction carried back through the
       dynamics. Neither the probe nor the readout itself reveals it. The Jacobian of the loop does.

    Everything in Part II was arranged by hand: the weights were chosen partly so that these effects would show.
    The toy establishes that the effects are possible and how to compute them. Whether trained models behave this
    way is an empirical question, and it is the subject of Part III.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part III · The same questions, asked of trained models

    The experiments in this repository put the questions of Part II to **Huginn-0125** (Geiping and others, 2025),
    a looped transformer trained on text, and repeated the steering experiment on **Recurrent-Llama-3.2** (McLeish
    and others, 2025). Everything in this part is **Measured**: the numbers come from the write-ups listed at the
    end, not from this notebook.

    Huginn's rows have 5,280 coordinates. Two layers embed the prompt. A core of four layers is then applied again
    and again to a state that starts as random noise, with the embedded prompt mixed back in before every
    iteration. Two final layers read out the next token. The correspondence with the toy:

    | In the toy | In Huginn |
    |---|---|
    | $F(H) = T_\theta\big(\tfrac12(H + X)\big)$ | the core block applied to a learned linear map of the state and the embedded prompt |
    | start from $X$, or from a random state | start from a seeded random state |
    | the last row | the last token, where the answer is predicted |
    | place preference $r = d^{\top} h_n$ | a digit's logit minus the average over digits, or $\log p(\text{True}) - \log p(\text{False})$ |
    | the context *the cat sat on the* | few-shot addition problems; true-or-false statements about cities and number comparisons |
    | $g$, set by a slider | nothing: every weight was trained |

    ## 19 · Do the states settle?

    **Mostly, yes.** At the positions where Huginn predicts an answer digit, late states are close to stationary,
    and the largest eigenvalue magnitude found for the local dynamics has a median of 0.78 (the toy's spectral
    radius is shown in section 16). The answer settles long before the state does: among positions that
    converged, the most probable digit stops changing at a median of 7 iterations and the state at a median of 74.

    **Sometimes, no.** On 200 fresh addition problems, 26 (13%) did not settle and ended on a period-two orbit. In
    each of the 26 the oscillation began at one earlier token and spread to later ones through attention, and the
    derivative at that source token had a real eigenvalue between −1.01 and −1.23, the case of section 17. Damping
    the source token alone stopped its oscillation in 24 of the 26. It stilled the whole text in only 15, because
    other rows often keep moving for reasons of their own.

    **Does the oscillation change what the model writes?** Usually not. The two states of an orbit are close, so
    the probabilities differ by a few points, and the top choice changes only where two candidates were nearly
    tied. In the first 44 oscillating prompts examined, no answer digit changed. A targeted search (400 new sums
    screened for near-ties, the 85 closest run for 512 iterations) found four answer digits that alternate between
    the correct digit and a wrong one, out of 358 answer positions on an orbit. One of them: after three worked
    sums, Huginn continues `816 + 254 =` with 1070 at 511 iterations per token and with 1060 at 512. The chart
    replays its saved probabilities for the third digit. From another random starting state the same prompt still
    oscillates, but 7 stays ahead. These examples were found by a search built to find them; they do not say how
    often this happens.
    """)
    return


@app.cell(hide_code=True)
def _(BLUE, MUTED, ORANGE, RULE, SURFACE, end_label, mpl, new_fig, np):
    _replay = {
        "Start A (the start used in the search)": ([0.5809, 0.4861] * 4, [0.4065, 0.5015] * 4),
        "Start B (another random start)": ([0.5657, 0.5058] * 4, [0.4218, 0.4818] * 4),
    }
    _loops = np.arange(505, 513)
    _fig, _axes = new_fig(2, width=9.4, height=3.0, sharey=True)
    for _ax, (_title, (_p7, _p6)) in zip(_axes, _replay.items()):
        _ax.plot(_loops, _p7, "-o", color=BLUE, lw=2, ms=6, mec=SURFACE, mew=1.5)
        _ax.plot(_loops, _p6, "-o", color=ORANGE, lw=2, ms=6, mec=SURFACE, mew=1.5)
        end_label(_ax, _loops[-1], _p7[-1], "7 (correct)", dx=9, dy=-5 if _p7[-1] < _p6[-1] else 5)
        end_label(_ax, _loops[-1], _p6[-1], "6", dx=9, dy=5 if _p7[-1] < _p6[-1] else -5)
        _ax.set_xticks(_loops)
        _ax.set_xlim(504.6, 514.2)
        _ax.set_ylim(0.36, 0.62)
        _ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
        _ax.grid(axis="y", color=RULE, lw=0.6)
        _ax.tick_params(length=0)
        _ax.set_xlabel("iterations before reading out")
        _ax.set_title(_title)
    _fig.suptitle("Measured · Huginn · probability of the digit after “816 + 254 = 10”", x=0.01, ha="left",
                  fontsize=10, color=MUTED)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 20 · Reading is not controlling

    **A property can be read.** Whether a statement is true can be decoded from Huginn's state at the end of the
    statement by a simple linear probe (the difference of class means), with an AUROC of 0.997 on statements about
    cities and 0.914 on number comparisons. That is decoding of a label, as in section 13. It is not evidence that
    the model has a notion of truth.

    **A push given once is erased.** Pushing a digit's state along a fitted direction moved the immediate readout
    toward the intended neighbouring digit by 2.3 in log-odds. Under continued iteration the push decayed, and the
    final answer equalled the unpushed one on 168 of 168 prompts. The same kind of edit applied at every iteration
    settled at a shifted state.

    **The probe direction is a weak lever.** On 60 held-out true-or-false statements, with pushes of equal length
    applied at every iteration, the median shift of the True/False log-odds was 0.25 along the probe direction,
    0.93 along the readout gradient, and 8.13 along the lever. The lever is nearly orthogonal to both of the others
    (cosine 0.10 with the readout gradient, 0.03 with the probe direction). What it moves is the readout: it raises the token "True". The experiments designed to
    separate that from changing the model's judgment were inconclusive.

    **The same on digits.** Unit directions were averaged over training sums and tested on held-out sums, each
    push added before every iteration. The numbers are the change in the pushed answer's score.

    | direction | Huginn (20 held-out sums, 48 iterations, push length 3) | Recurrent-Llama-3.2 (40 held-out sums, push 3% of the state's length) |
    |---|---|---|
    | lever | +1.78 | +0.165 |
    | readout gradient | −0.05 | −0.001 |
    | centroid of the target's states | −0.05 | 0.000 |

    In Huginn the lever was positive on 189 of 200 problem–digit pairs. It started slightly negative at once
    (median −0.13) and turned positive after a single iteration (+0.94), while the gradient and the centroid sat at
    the level of random directions (±0.05). That is the pattern of the table in section 18, in a model nobody
    arranged. In Recurrent-Llama the cosine between lever and gradient is −0.005. The chart shows its three push
    lengths.

    **The first-order formula predicts the measurements.** Ten lever pushes times ten candidate answers give 100
    score changes per problem. Using each problem's own $J$, the formula of section 18 predicts them with median
    correlation 0.9997 at the 1% push, 0.997 at 3% and 0.976 at 9%, against 0.11 with the loop left out (in Huginn
    0.978, against −0.59).

    **The catch.** On Recurrent-Llama the correct answer leads the runner-up by a median of 4.84, about ten times
    the largest change in the chart. No push made its target the model's answer: 0 of 360 tries at every length.
    And in Huginn, in a retrospective test where a probe chose which way to steer, steering added no accuracy
    beyond what the probe's own verdict already gave.
    """)
    return


@app.cell(hide_code=True)
def _(AQUA, BLUE, MUTED, ORANGE, RULE, SURFACE, end_label, new_fig, np):
    _doses = np.array([1, 3, 9])
    _series = [
        ("lever", BLUE, [0.055, 0.165, 0.480], [0.051, 0.154, 0.443], [0.059, 0.177, 0.519]),
        ("gradient", ORANGE, [0.000, -0.001, -0.003], [-0.001, -0.002, -0.006], [0.000, 0.000, 0.001]),
        ("centroid", AQUA, [0.000, 0.000, -0.001], [-0.001, -0.002, -0.005], [0.000, 0.001, 0.004]),
    ]
    _fig, _ax = new_fig(1, width=6.4, height=3.3)
    _ax.axhline(0, color=MUTED, lw=0.8)
    for _name, _color, _v, _lo, _hi in _series:
        _ax.plot(_doses, _v, "-o", color=_color, lw=2, ms=7, mec=SURFACE, mew=1.5, zorder=3)
        _ax.vlines(_doses, _lo, _hi, color=_color, lw=2, zorder=2)
    end_label(_ax, 9, 0.480, "lever", dx=10)
    end_label(_ax, 9, 0.0, "gradient and centroid", dx=10, dy=9)
    _ax.set_xticks(_doses, ["1%", "3%", "9%"])
    _ax.set_xlim(0.4, 12.5)
    _ax.grid(axis="y", color=RULE, lw=0.6)
    _ax.tick_params(length=0)
    _ax.set_xlabel("length of the push, as a share of the length of the state")
    _ax.set_title("Measured · Recurrent-Llama-3.2 · change of the pushed answer's score")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 21 · What this shows, and what it does not

    **The toy** shows that each mechanism fits in a few lines of numpy and can be checked by hand. A hidden state
    can carry a property perfectly and the output can ignore it. A row whose own block has an eigenvalue below −1
    drives a period-two orbit that later rows inherit. The direction that moves a readout after looping is
    $(J + \dots + J^T)^{\top} d$, which need not resemble either the probe direction or $d$. Its weights were
    chosen by hand to show exactly that, so the toy says nothing about how often any of this happens in a trained
    model.

    **The experiments on dynamics** connect a source position's local instability to a change in the model's
    next-token probabilities, and show that damping that position can sometimes stop the oscillation. They do not
    estimate how often answers flip in general use: the answer-changing examples came from searches for near-ties,
    and the outcome also depends on the starting state.

    **The experiments on control** show that the response of the loop decides which direction steers a readout,
    and that a local linear model of the loop predicts the size of the effect. They measure changes in target
    scores on the tested tasks and models. In Recurrent-Llama none of the tested pushes made the target the answer,
    even though its score rose. In neither model do they show that a concept, as opposed to a readout, was
    changed.

    **A way of asking.** The thread through all three parts is a habit more than a result. Say what the hidden
    state is and where it sits. Separate what can be read from it from what the model does with it. Test the
    second by changing the state and recomputing. And when the model is a loop, remember that everything downstream
    of an intervention includes the loop.

    Sources in this repository: `huginn_addition/WRITEUP.md`, `docs/flicker2_design.md`,
    `docs/steering_results.pdf`, `docs/followup_results.md`. Models: Huginn (Geiping and others, 2025),
    Recurrent-Llama-3.2 (McLeish and others, 2025). The way of writing an attention head as "where to read" and
    "what to bring back" is from Elhage and others, "A Mathematical Framework for Transformer Circuits" (2021);
    the damping rule is in Movahedi and others (2026); the lever is the calculus used to train fixed-point models
    since Bai, Kolter and Koltun (2019).
    """)
    return


if __name__ == "__main__":
    app.run()
