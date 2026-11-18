(() => {
  "use strict";

  const VOCAB = ["the", "a", "on", "cat", "dog", "bird", "mat", "hat", "grass", "sat", "chased"];
  const ID = Object.fromEntries(VOCAB.map((w, i) => [w, i]));
  // Words shown in widget tables use inline-code style, like quoted words in the text
  const cw = w => `<code>${w}</code>`;

  const E_BASE = [
    [-1.00,  0.40], // the
    [-0.70,  0.75], // a
    [-0.85, -0.05], // on
    [ 0.75,  0.75], // cat
    [ 1.05,  0.50], // dog
    [ 0.50,  1.05], // bird
    [ 1.00, -0.45], // mat
    [ 0.85, -0.80], // hat
    [ 1.25, -0.75], // grass
    [-0.20, -1.10], // sat
    [ 0.05, -1.20], // chased
  ];

  const LABEL_SHIFT = {
    the: [-0.09, 0.06, "end"],
    a: [-0.09, 0.06, "end"],
    on: [-0.09, 0.00, "end"],
    hat: [-0.09, -0.04, "end"],
    sat: [0.00, 0.14, "middle"],
    bird: [-0.09, 0.06, "end"],
    chased: [0.09, -0.05, "start"],
  };

  const QUESTION = "the cat sat on the";
  const COUNTS = [0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0];
  const TARGET = COUNTS.map(c => c / 4);
  const OMEGA = Math.PI / 4;

  // ---- Math helpers -------------------------------------------------------------------------

  const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
  const norm = a => Math.hypot(a[0], a[1]);

  function round2(x) {
    return Math.round((x + Number.EPSILON) * 100) / 100;
  }

  function offsets(n) {
    const out = [];
    for (let i = 1; i <= n; i++) {
      out.push([round2(0.15 * Math.cos(i * OMEGA)), round2(0.15 * Math.sin(i * OMEGA))]);
    }
    return out;
  }

  function encode(text) {
    return text.trim().split(/\s+/).map(w => ID[w]);
  }

  function embed(text, table = E_BASE) {
    const ids = encode(text);
    const P = offsets(ids.length);
    return ids.map((id, i) => [table[id][0] + P[i][0], table[id][1] + P[i][1]]);
  }

  function softmax(z) {
    const finite = z.filter(Number.isFinite);
    const max = finite.length ? Math.max(...finite) : 0;
    const exps = z.map(v => (v === -Infinity ? 0 : Math.exp(v - max)));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => v / sum);
  }

  function causalSoftmax(S) {
    return S.map((row, i) => softmax(row.map((v, j) => (j <= i ? v : -Infinity))));
  }

  function matMul(A, B) {
    const m = A.length, k = A[0].length, n = B[0].length;
    const out = Array.from({ length: m }, () => Array(n).fill(0));
    for (let r = 0; r < m; r++) {
      for (let c = 0; c < n; c++) {
        let s = 0;
        for (let t = 0; t < k; t++) s += A[r][t] * B[t][c];
        out[r][c] = s;
      }
    }
    return out;
  }

  function transpose(M) {
    return M[0].map((_, c) => M.map(r => r[c]));
  }

  function turnMatrix(deg, stretch = 1.0) {
    const a = (deg * Math.PI) / 180;
    const c = Math.cos(a), s = Math.sin(a);
    const clean = x => (Math.abs(x) < 1e-12 ? 0 : x);
    return [
      [clean(stretch * c), clean(stretch * s)],
      [clean(-stretch * s), clean(stretch * c)],
    ];
  }

  function nll(p, f = TARGET) {
    let s = 0;
    for (let i = 0; i < f.length; i++) {
      if (f[i] > 0) s -= f[i] * Math.log(p[i]);
    }
    return s;
  }

  const LOSS_FLOOR = nll(TARGET, TARGET);

  // Precompute 2000 steps of gradient descent on (alpha, E) (27 parameters)
  const GD_HISTORY = (() => {
    let tab = E_BASE.map(r => [r[0], r[1]]);
    let alpha = [0.2, 0.2, 0.2, 0.2, 0.2];
    const P = offsets(5);
    const qIds = encode(QUESTION);
    const snaps = {};
    const keepSteps = new Set([0, 1, 2, 5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000]);
    const losses = [];

    for (let step = 0; step <= 2000; step++) {
      const X = qIds.map((id, i) => [tab[id][0] + P[i][0], tab[id][1] + P[i][1]]);
      const c = [
        alpha.reduce((acc, a, i) => acc + a * X[i][0], 0),
        alpha.reduce((acc, a, i) => acc + a * X[i][1], 0),
      ];
      const z = tab.map(e => dot(e, c));
      const p = softmax(z);
      losses.push(nll(p, TARGET));
      if (keepSteps.has(step)) {
        snaps[step] = {
          E: tab.map(r => [r[0], r[1]]),
          alpha: alpha.slice(),
          c: [c[0], c[1]],
        };
      }
      if (step === 2000) break;

      const dz = p.map((pv, k) => pv - TARGET[k]);
      const dc = [
        tab.reduce((acc, e, k) => acc + e[0] * dz[k], 0),
        tab.reduce((acc, e, k) => acc + e[1] * dz[k], 0),
      ];
      const gE = dz.map(d => [d * c[0], d * c[1]]);
      const gAlpha = X.map((xi, i) => {
        gE[qIds[i]][0] += alpha[i] * dc[0];
        gE[qIds[i]][1] += alpha[i] * dc[1];
        return xi[0] * dc[0] + xi[1] * dc[1];
      });
      tab = tab.map((r, k) => [r[0] - 0.05 * gE[k][0], r[1] - 0.05 * gE[k][1]]);
      alpha = alpha.map((a, i) => a - 0.05 * gAlpha[i]);
    }
    return { snaps, stepList: Array.from(keepSteps).sort((a, b) => a - b), losses, trained: snaps[2000].E };
  })();

  // ---- Formatting & HTML Matrix Builders ----------------------------------------------------

  function fmtNum(v, digits = 2, signed = false) {
    if (typeof v === "string") return v;
    if (v === null || v === undefined || !Number.isFinite(v)) return "·";
    const clean = Math.abs(v) < 0.5 * Math.pow(10, -digits) ? 0 : v;
    const absStr = Math.abs(clean).toFixed(digits);
    if (clean < 0) return "−" + absStr;
    if (signed && clean > 0) return "+" + absStr;
    return absStr;
  }

  function fmtPct(p, digits = 1) {
    if (!Number.isFinite(p)) return "·";
    const clean = Math.abs(p * 100) < 0.5 * Math.pow(10, -digits) ? 0 : p * 100;
    return (clean < 0 ? "−" : "") + Math.abs(clean).toFixed(digits) + "%";
  }

  function fmtVec(v, digits = 2) {
    return `(${fmtNum(v[0], digits)}, ${fmtNum(v[1], digits)})`;
  }

  function pctBar(p, isTarget = false) {
    const w = Math.max(0, Math.min(1, p)) * 68;
    const cls = isTarget ? "lt-bar lt-bar-target" : "lt-bar";
    return `<div class="lt-pc"><span>${fmtPct(p, 1)}</span><span class="${cls}" style="width:${w.toFixed(1)}px"></span></div>`;
  }

  function gridHTML(M, opts = {}) {
    const {
      rows = null,
      cols = null,
      digits = 2,
      signed = false,
      mark = null,
      mark2 = null,
      dim = null,
      name = null,
      bracket = true,
      shape = true,
      left = [],
      foot = null,
      rowClickAttr = null,
      cellClickAttr = null,
    } = opts;

    const nr = M.length;
    const nc = M[0].length;
    const leftSet = new Set(left);

    let body = "";
    if (cols) {
      body += `<tr class="${bracket ? "" : "lt-rule"}">` +
        cols.map((c, idx) => `<th class="${leftSet.has(idx) ? "lt-l" : ""}">${c}</th>`).join("") +
        `</tr>`;
    }
    for (let r = 0; r < nr; r++) {
      const rAttr = rowClickAttr ? ` class="lt-click" data-${rowClickAttr}="${r}"` : "";
      body += `<tr${rAttr}>`;
      for (let c = 0; c < nc; c++) {
        const classes = [];
        if (mark && mark(r, c)) classes.push("lt-hi");
        else if (mark2 && mark2(r, c)) classes.push("lt-hi2");
        if (dim && dim(r, c)) classes.push("lt-dim");
        if (leftSet.has(c)) classes.push("lt-l");
        if (cellClickAttr && (!dim || !dim(r, c))) classes.push("lt-cell-click");
        const cAttr = cellClickAttr && (!dim || !dim(r, c)) ? ` data-${cellClickAttr}="${r},${c}"` : "";
        const val = M[r][c];
        const txt = typeof val === "string" ? val : fmtNum(val, Array.isArray(digits) ? digits[c] : digits, signed);
        body += `<td class="${classes.join(" ")}"${cAttr}>${txt}</td>`;
      }
      body += `</tr>`;
    }
    if (foot) {
      body += `<tr class="lt-foot">` +
        foot.map((f, idx) => `<td class="${leftSet.has(idx) ? "lt-l" : ""}">${f}</td>`).join("") +
        `</tr>`;
    }

    let table = `<table class="lt-t">${body}</table>`;
    if (bracket) {
      table = `<div class="lt-b" style="--top:${cols ? 23 : 0}px">${table}</div>`;
    }
    if (rows) {
      const labRows = (cols ? `<tr><td>&nbsp;</td></tr>` : "") +
        rows.map(r => `<tr><td>${r}</td></tr>`).join("");
      table = `<div class="lt-row"><table class="lt-t lt-lab">${labRows}</table>${table}</div>`;
    }
    const cap = name
      ? `<div class="lt-cap">${name}${shape && bracket ? `<small>${nr}×${nc}</small>` : ""}</div>`
      : "";
    return `<div class="lt-m">${cap}${table}</div>`;
  }

  function eqRow(...parts) {
    return `<div class="lt-eq">${parts.map(p => p.trim().startsWith("<") ? p : `<span class="lt-op">${p}</span>`).join("")}</div>`;
  }

  // ---- SVG Plane Helper ---------------------------------------------------------------------

  function makePlaneSVG({ width = 340, height = 320, lim = 1.65, table = E_BASE, showWords = true, wordAlpha = 1.0, extra = "" }) {
    const pad = 26;
    const W = width, H = height;
    const sx = x => pad + ((x + lim) / (2 * lim)) * (W - 2 * pad);
    const sy = y => H - pad - ((y + lim) / (2 * lim)) * (H - 2 * pad);

    const ticks = lim < 2.2 ? [-1, 0, 1] : [-Math.round(lim * 0.7), 0, Math.round(lim * 0.7)];
    let gridSvg = `
      <line x1="${sx(-lim)}" y1="${sy(0)}" x2="${sx(lim)}" y2="${sy(0)}" stroke="var(--rule)" stroke-width="1"/>
      <line x1="${sx(0)}" y1="${sy(-lim)}" x2="${sx(0)}" y2="${sy(lim)}" stroke="var(--rule)" stroke-width="1"/>
    `;
    for (const t of ticks) {
      if (t === 0) continue;
      gridSvg += `<text x="${sx(t)}" y="${H - 8}" text-anchor="middle" font-size="10" fill="var(--ink3)">${t}</text>`;
      gridSvg += `<text x="10" y="${sy(t) + 3}" text-anchor="middle" font-size="10" fill="var(--ink3)">${t}</text>`;
    }

    let wordsSvg = "";
    if (table) {
      table.forEach(([x, y], idx) => {
        const w = VOCAB[idx];
        wordsSvg += `<circle cx="${sx(x).toFixed(1)}" cy="${sy(y).toFixed(1)}" r="3.5" fill="var(--ink2)" opacity="${wordAlpha}" data-word-dot="${idx}" style="cursor:pointer"/>`;
        if (showWords) {
          const [dx, dy, anchor] = LABEL_SHIFT[w] || [0.09, 0.05, "start"];
          const scale = lim / 1.65;
          wordsSvg += `<text x="${sx(x + dx * scale).toFixed(1)}" y="${(sy(y + dy * scale) + 3).toFixed(1)}" text-anchor="${anchor}" font-size="11" fill="var(--ink2)" opacity="${wordAlpha}" data-word-dot="${idx}" style="cursor:pointer">${w}</text>`;
        }
      });
    }

    const defs = `
      <defs>
        <marker id="arr-ink" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--ink)"/>
        </marker>
        <marker id="arr-blue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--blue)"/>
        </marker>
        <marker id="arr-orange" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--orange)"/>
        </marker>
      </defs>
    `;

    return {
      sx,
      sy,
      wrap: inner => `<svg class="w-svg" viewBox="0 0 ${W} ${H}">${defs}${gridSvg}${wordsSvg}${inner}${extra}</svg>`,
    };
  }

  function svgArrow(sx, sy, from, to, colorVar = "ink", width = 1.8, opacity = 1) {
    return `<line x1="${sx(from[0]).toFixed(1)}" y1="${sy(from[1]).toFixed(1)}" x2="${sx(to[0]).toFixed(1)}" y2="${sy(to[1]).toFixed(1)}" stroke="var(--${colorVar})" stroke-width="${width}" opacity="${opacity}" marker-end="url(#arr-${colorVar})"/>`;
  }

  // ---- Shared Attention State (syncs Sections 6 and 7!) -------------------------------------

  const sharedAttn = {
    beta: 2.0,
    theta: 90,
    gamma: 1.0,
    phi: 90,
    listeners: [],
    notify() {
      this.listeners.forEach(fn => fn());
    },
  };

  // ---- Widget Implementations ---------------------------------------------------------------

  const WIDGETS = {
    // 1. Observed counts + interactive event P(A)
    "vocab-counts"(el) {
      const selected = new Set([ID["mat"], ID["hat"]]);

      function render() {
        const rows = VOCAB.map((w, i) => [
          cw(w),
          String(COUNTS[i]),
          `${COUNTS[i]}/4`,
          pctBar(TARGET[i]),
        ]);
        const sumSelected = Array.from(selected).reduce((acc, i) => acc + COUNTS[i], 0);
        const setStr = selected.size
          ? `{${Array.from(selected).map(i => cw(VOCAB[i])).join(", ")}}`
          : "∅";

        el.innerHTML = `
          <div class="w-card">
            <div class="w-head">
              <span class="w-title">Continuations after <code>${QUESTION} ____</code></span>
              <span class="w-hint">Click any rows to form an event <i>A</i> ⊆ <span class=cal-v>𝒱</span></span>
            </div>
            ${eqRow(
              gridHTML(rows, {
                cols: ["candidate word", "count", "share", "observed frequency f(v)"],
                foot: ["total", "4", "4/4", `<div class="lt-pc"><span>100.0%</span></div>`],
                bracket: false,
                left: [0],
                mark: r => selected.has(r),
                rowClickAttr: "ev-row",
              })
            )}
            <div class="w-note">
              Selected event <strong><i>A</i> = ${setStr}</strong> &nbsp;→&nbsp;
              <strong><i>P</i>(<i>A</i>) = ${sumSelected}/4 = ${fmtPct(sumSelected / 4, 1)}</strong>
            </div>
          </div>
        `;
        el.querySelectorAll("[data-ev-row]").forEach(tr => {
          tr.addEventListener("click", () => {
            const idx = Number(tr.dataset.evRow);
            if (selected.has(idx)) selected.delete(idx);
            else selected.add(idx);
            render();
          });
        });
      }
      render();
    },

    // 2. Word IDs lookup & context encoding
    "word-ids"(el) {
      const ids = encode(QUESTION);
      const words = QUESTION.split(" ");
      const dictRows = VOCAB.map((w, i) => [`<code>${w}</code>`, String(i + 1)]);  // IDs start at 1, like every index in the post
      const seqGrid = [words.map(w => `<code>${w}</code>`), ids.map(id => String(id + 1))];
      el.innerHTML = `
        <div class="w-card">
          <div class="w-head">
            <span class="w-title">Assigning integer IDs to the vocabulary</span>
          </div>
          ${eqRow(
            gridHTML(dictRows, { cols: ["word", "ID"], bracket: false, left: [0], name: "Dictionary" }),
            "→",
            gridHTML(seqGrid, {
              rows: ["word", "ID"],
              cols: words.map((_, i) => `pos ${i + 1}`),
              bracket: false,
              name: `Encoded context: <code>${QUESTION}</code>`,
            })
          )}
        </div>
      `;
    },

    // 3. Interactive 2D Embedding Plane + Inner Product
    "embedding-plane"(el) {
      let compareIdx = ID["dog"];
      const anchorIdx = ID["cat"];

      function render() {
        const lim = 1.65;
        const p = makePlaneSVG({ width: 340, height: 320, lim, table: E_BASE });
        const catVec = E_BASE[anchorIdx];
        const otherVec = E_BASE[compareIdx];

        const a1 = Math.atan2(catVec[1], catVec[0]);
        const a2 = Math.atan2(otherVec[1], otherVec[0]);
        let diff = a2 - a1;
        while (diff > Math.PI) diff -= 2 * Math.PI;
        while (diff < -Math.PI) diff += 2 * Math.PI;

        const steps = 32;
        const rArc = 0.42 * (lim / 1.65);
        const pts = [];
        for (let s = 0; s <= steps; s++) {
          const ang = a1 + (diff * s) / steps;
          pts.push(`${p.sx(rArc * Math.cos(ang)).toFixed(1)},${p.sy(rArc * Math.sin(ang)).toFixed(1)}`);
        }
        const midAng = a1 + diff / 2;

        const groupLabels = [
          ["function words", -1.18 * (lim / 1.65), 1.18 * (lim / 1.65)],
          ["animals", 1.05 * (lim / 1.65), 1.32 * (lim / 1.65)],
          ["places", 1.08 * (lim / 1.65), -1.18 * (lim / 1.65)],
          ["verbs", -0.62 * (lim / 1.65), -1.38 * (lim / 1.65)],
        ]
          .map(([txt, gx, gy]) => `<text x="${p.sx(gx).toFixed(1)}" y="${p.sy(gy).toFixed(1)}" font-size="11" font-style="italic" fill="var(--ink3)" text-anchor="middle">${txt}</text>`)
          .join("");

        const innerProduct = dot(catVec, otherVec);
        const cosSim = innerProduct / (norm(catVec) * norm(otherVec));
        const deg = (Math.acos(Math.max(-1, Math.min(1, cosSim))) * 180) / Math.PI;

        const svg = p.wrap(`
          ${groupLabels}
          <polyline points="${pts.join(" ")}" fill="none" stroke="var(--orange)" stroke-width="1.6"/>
          <text x="${p.sx(1.32 * rArc * Math.cos(midAng)).toFixed(1)}" y="${(p.sy(1.32 * rArc * Math.sin(midAng)) + 3).toFixed(1)}" font-size="11" font-weight="600" fill="var(--orange)" text-anchor="middle">${deg.toFixed(1)}°</text>
          ${svgArrow(p.sx, p.sy, [0, 0], otherVec, "orange", 2.0)}
          ${svgArrow(p.sx, p.sy, [0, 0], catVec, "ink", 2.0)}
        `);

        el.innerHTML = `
          <div class="w-card">
            <div class="w-head">
              <span class="w-title">Hand-placed 2D embeddings <b class=mm>E</b> (11 × 2)</span>
              <span class="w-hint">Click any word in the plot or table to compare with <code>cat</code></span>
            </div>
            <div class="w-controls">
              <div class="w-chips">
                ${["dog", "bird", "mat", "on", "sat"].map(w => `
                  <button class="w-chip ${compareIdx === ID[w] ? "active-orange" : ""}" data-pick="${ID[w]}"><code>cat</code> vs <code>${w}</code></button>
                `).join("")}
              </div>
            </div>
            <div class="w-split">
              <div>${svg}</div>
              <div style="justify-self:center">
                ${gridHTML(E_BASE, {
                  rows: VOCAB.map(cw),
                  cols: ["coord 1", "coord 2"],
                  name: "<b class=mm>E</b>",
                  mark: r => r === anchorIdx,
                  mark2: r => r === compareIdx,
                  rowClickAttr: "pick-row",
                })}
              </div>
            </div>
            <div class="w-note">
              <code>cat</code> ${fmtVec(catVec)} · <code>${VOCAB[compareIdx]}</code> ${fmtVec(otherVec)}
              = (${fmtNum(catVec[0])})(${fmtNum(otherVec[0])}) + (${fmtNum(catVec[1])})(${fmtNum(otherVec[1])})
              = <strong>${fmtNum(innerProduct, 4)}</strong>
              &nbsp;|&nbsp; angle θ = <strong>${deg.toFixed(1)}°</strong> (cos θ = ${fmtNum(cosSim, 3)})
            </div>
          </div>
        `;

        el.querySelectorAll("[data-pick]").forEach(btn => {
          btn.addEventListener("click", () => {
            compareIdx = Number(btn.dataset.pick);
            render();
          });
        });
        el.querySelectorAll("[data-pick-row]").forEach(tr => {
          tr.addEventListener("click", () => {
            compareIdx = Number(tr.dataset.pickRow);
            render();
          });
        });
        el.querySelectorAll("[data-word-dot]").forEach(node => {
          node.addEventListener("click", () => {
            compareIdx = Number(node.dataset.wordDot);
            render();
          });
        });
      }
      render();
    },

    // 4. Embeddings before position offsets
    "embed-before-pos"(el) {
      const words = QUESTION.split(" ");
      const ids = encode(QUESTION);
      const rows = ids.map(id => E_BASE[id]);
      el.innerHTML = `
        <div class="w-card">
          ${eqRow(
            gridHTML(rows, {
              rows: words.map((w, i) => `${i + 1} ${cw(w)}`),
              cols: ["coord 1", "coord 2"],
              name: "Embeddings before position offsets, <b class=mm>E</b><sub>ctx</sub>",
              mark: r => r === 0 || r === 4,
            })
          )}
          <div class="w-note">Rows 1 and 5 both hold <code>the</code>: without position information, their vectors are identical.</div>
        </div>
      `;
    },

    // 4b. Naive position-index offsets
    "index-offset-matrix"(el) {
      const words = ["the", "cat", "sat", "on", "the"];
      const Xtok = words.map(w => E_BASE[ID[w]]);
      const Pidx = words.map((_, i) => [i + 1, i + 1]);
      const Xout = Xtok.map((r, i) => [r[0] + Pidx[i][0], r[1] + Pidx[i][1]]);
      const rowLabels = words.map((w, i) => `${i + 1} ${cw(w)}`);

      el.innerHTML = `
        <div class="w-card">
          ${eqRow(
            gridHTML(Xtok, { rows: rowLabels, cols: ["1", "2"], name: "Word embeddings<br><b class=mm>E</b><sub>ctx</sub>", mark: r => r === 0 || r === 4 }),
            "+",
            gridHTML(Pidx, { rows: words.map((_, i) => `<b class=vm>p</b><sub>${i + 1}</sub>`), cols: ["1", "2"], name: "Position index<br><b class=mm>P</b>" }),
            "=",
            gridHTML(Xout, { rows: rowLabels, cols: ["1", "2"], name: "Sequence matrix<br><b class=mm>X</b>", mark: r => r === 0 || r === 4 })
          )}
        </div>
      `;
    },

    // 4c. 5000-word document index blow-up
    "index-offset-5000"(el) {
      const Erows = [
        E_BASE[ID["the"]],
        E_BASE[ID["cat"]],
        ["⋮", "⋮"],
        E_BASE[ID["cat"]],
        E_BASE[ID["the"]],
      ];
      const Prows = [
        [1, 1],
        [2, 2],
        ["⋮", "⋮"],
        [4999, 4999],
        [5000, 5000],
      ];
      const Xrows = [
        [E_BASE[ID["the"]][0] + 1, E_BASE[ID["the"]][1] + 1],
        [E_BASE[ID["cat"]][0] + 2, E_BASE[ID["cat"]][1] + 2],
        ["⋮", "⋮"],
        [E_BASE[ID["cat"]][0] + 4999, E_BASE[ID["cat"]][1] + 4999],
        [E_BASE[ID["the"]][0] + 5000, E_BASE[ID["the"]][1] + 5000],
      ];
      const wordRows = [`1 ${cw("the")}`, `2 ${cw("cat")}`, "⋮", `4999 ${cw("cat")}`, `5000 ${cw("the")}`];
      const posRows = ["<b class=vm>p</b><sub>1</sub>", "<b class=vm>p</b><sub>2</sub>", "⋮", "<b class=vm>p</b><sub>4999</sub>", "<b class=vm>p</b><sub>5000</sub>"];

      el.innerHTML = `
        <div class="w-card">
          ${eqRow(
            gridHTML(Erows, { rows: wordRows, cols: ["1", "2"], name: "Word embeddings<br><b class=mm>E</b><sub>ctx</sub>", shape: false, mark: r => r === 0 || r === 4, mark2: r => r === 3 }),
            "+",
            gridHTML(Prows, { rows: posRows, cols: ["1", "2"], name: "Position index<br><b class=mm>P</b>", shape: false }),
            "=",
            gridHTML(Xrows, { rows: wordRows, cols: ["1", "2"], name: "Sequence matrix<br><b class=mm>X</b> (5000×2)", shape: false, mark: r => r === 0 || r === 4, mark2: r => r === 3 })
          )}
        </div>
      `;
    },

    // 5a. Tip-to-tail vector addition: e_w + p_i = x_i
    "tip-to-tail-offset"(el) {
      let word = "the";
      let pos = 1;

      function render() {
        const lim = 1.55;
        const p = makePlaneSVG({ width: 360, height: 320, lim, table: E_BASE, wordAlpha: 0.35 });
        const ew = E_BASE[ID[word]];
        const pi = [round2(0.15 * Math.cos(pos * OMEGA)), round2(0.15 * Math.sin(pos * OMEGA))];
        const xi = [ew[0] + pi[0], ew[1] + pi[1]];
        const rPx = (0.15 / (2 * lim)) * (360 - 52);

        const mx = ew[0] * 0.52;
        const my = ew[1] * 0.52;
        const nLen = Math.hypot(ew[0], ew[1]) || 1;
        const nx = (-ew[1] / nLen) * 0.14;
        const ny = (ew[0] / nLen) * 0.14;

        const pLen = Math.hypot(pi[0], pi[1]) || 1;
        const pxLab = xi[0] + (pi[0] / pLen) * 0.16;
        const pyLab = xi[1] + (pi[1] / pLen) * 0.16;

        const svg = p.wrap(`
          <circle cx="${p.sx(ew[0]).toFixed(1)}" cy="${p.sy(ew[1]).toFixed(1)}" r="${rPx.toFixed(1)}" fill="none" stroke="var(--blue)" stroke-width="1.2" stroke-dasharray="3 3" opacity="0.65"/>
          ${svgArrow(p.sx, p.sy, [0, 0], xi, "orange", 1.8, 0.85)}
          ${svgArrow(p.sx, p.sy, [0, 0], ew, "ink", 2.1)}
          ${svgArrow(p.sx, p.sy, ew, xi, "blue", 2.1)}
          <text x="${p.sx(mx + nx).toFixed(1)}" y="${(p.sy(my + ny) + 3).toFixed(1)}" font-size="12" font-weight="700" font-family="KaTeX_Main, serif" fill="var(--ink)" text-anchor="middle">e<tspan dy="3" font-size="9" font-style="normal">${word}</tspan></text>
          <text x="${p.sx(pxLab).toFixed(1)}" y="${(p.sy(pyLab) + 3).toFixed(1)}" font-size="12" font-weight="700" font-family="KaTeX_Main, serif" fill="var(--blue)" text-anchor="middle">p<tspan dy="3" font-size="9" font-style="normal">${pos}</tspan></text>
        `);

        el.innerHTML = `
          <div class="w-card">
            <div class="w-head">
              <span class="w-title">Adding a small rotating offset: <b class=vm>x</b><sub>${pos}</sub> = <b class=vm>e</b><sub>${word}</sub> + <b class=vm>p</b><sub>${pos}</sub></span>
              <span class="w-hint">Click a word or position</span>
            </div>
            <div class="w-controls">
              <div class="w-chips">
                ${["the", "cat", "sat", "on", "mat"].map(w => `
                  <button class="w-chip ${word === w ? "active" : ""}" data-tt-word="${w}"><code>${w}</code></button>
                `).join("")}
              </div>
              <div class="w-chips">
                ${[1, 2, 3, 4, 5, 6, 7, 8].map(i => `
                  <button class="w-chip ${pos === i ? "active-orange" : ""}" data-tt-pos="${i}">pos ${i}</button>
                `).join("")}
              </div>
            </div>
            <div style="max-width:420px;margin:0 auto">${svg}</div>
            <div class="w-note">
              <strong><b class=vm>e</b><sub>${word}</sub></strong> ${fmtVec(ew)} + <strong style="color:var(--blue)"><b class=vm>p</b><sub>${pos}</sub></strong> ${fmtVec(pi)}
              = <strong style="color:var(--orange)"><b class=vm>x</b><sub>${pos}</sub></strong> ${fmtVec(xi)}
            </div>
          </div>
        `;

        el.querySelectorAll("[data-tt-word]").forEach(b => {
          b.addEventListener("click", () => { word = b.dataset.ttWord; render(); });
        });
        el.querySelectorAll("[data-tt-pos]").forEach(b => {
          b.addEventListener("click", () => { pos = Number(b.dataset.ttPos); render(); });
        });
        el.querySelectorAll("[data-word-dot]").forEach(node => {
          node.addEventListener("click", () => { word = VOCAB[Number(node.dataset.wordDot)]; render(); });
        });
      }
      render();
    },

    // 5. Three-plane clock hands + 1-circle vs 3-circle period plot
    "position-circle"(el) {
      let activePos = 108;
      const planes = [
        { label: "Plane 1", sub: "period 8", period: 8, color: "var(--blue)" },
        { label: "Plane 2", sub: "period 27", period: 27, color: "var(--orange)" },
        { label: "Plane 3", sub: "period 200", period: 200, color: "var(--aqua)" },
      ];

      function render() {
        // Left SVG: 3 mini clock-hand circles for R^6 = R^2 + R^2 + R^2
        const W = 310, H = 220;
        const r = 34;
        const centers = [
          [56, 72],
          [155, 72],
          [254, 72],
        ];
        const clocks = planes.map((pl, idx) => {
          const [cx, cy] = centers[idx];
          const ang = (activePos * 2 * Math.PI) / pl.period;
          const ang100 = (100 * 2 * Math.PI) / pl.period;
          const hx = cx + r * Math.cos(ang);
          const hy = cy - r * Math.sin(ang);
          const gx = cx + r * Math.cos(ang100);
          const gy = cy - r * Math.sin(ang100);
          return `
            <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="var(--rule)" stroke-width="1.4"/>
            <line x1="${cx - r - 4}" y1="${cy}" x2="${cx + r + 4}" y2="${cy}" stroke="var(--rule)" stroke-width="0.8"/>
            <line x1="${cx}" y1="${cy - r - 4}" x2="${cx}" y2="${cy + r + 4}" stroke="var(--rule)" stroke-width="0.8"/>
            <line x1="${cx}" y1="${cy}" x2="${gx.toFixed(1)}" y2="${gy.toFixed(1)}" stroke="var(--ink3)" stroke-width="1.3" stroke-dasharray="2.5 2.5" opacity="0.7"/>
            <line x1="${cx}" y1="${cy}" x2="${hx.toFixed(1)}" y2="${hy.toFixed(1)}" stroke="${pl.color}" stroke-width="2.2"/>
            <circle cx="${hx.toFixed(1)}" cy="${hy.toFixed(1)}" r="3.5" fill="${pl.color}"/>
            <text x="${cx}" y="${cy + r + 18}" text-anchor="middle" font-size="11.5" font-weight="600" fill="var(--ink)">${pl.label}</text>
            <text x="${cx}" y="${cy + r + 32}" text-anchor="middle" font-size="10.5" fill="var(--ink3)">${pl.sub}</text>
          `;
        }).join("");

        const clocksSvg = `
          <svg class="w-svg" viewBox="0 0 ${W} ${H}">
            ${clocks}
            <text x="${W / 2}" y="${H - 42}" text-anchor="middle" font-size="11" fill="var(--ink2)">
              Solid hand: position <tspan font-weight="700" fill="var(--ink)">j = ${activePos}</tspan>
              &nbsp;|&nbsp; Dashed: <tspan font-weight="600">j = 100</tspan>
            </text>
            <text x="${W / 2}" y="${H - 22}" text-anchor="middle" font-size="10.5" fill="var(--ink3)">
              ${activePos === 108 ? "At j = 108, Plane 1 collides with 100, but Planes 2 & 3 do not." :
                activePos === 9 ? "At j = 9, Plane 1 wraps back to j = 1, while Planes 2 & 3 advance." :
                activePos === 100 ? "At j = 100, all three hands align with the reference." :
                "Each plane rotates at its own speed ωₖ = 360° / period."}
            </text>
          </svg>
        `;

        // Right SVG: 1 circle vs 3 circles match with position 100 + live cursor at activePos
        const W2 = 340, H2 = 260;
        const chartPanel = (yOff, title, periods) => {
          const left = 36, right = W2 - 14, top = yOff + 22, bot = yOff + 108;
          const xMap = j => left + ((j - 1) / 199) * (right - left);
          const yMap = v => bot - ((v + 1.05) / 2.15) * (bot - top);
          const pts = [];
          for (let j = 1; j <= 200; j++) {
            const val = periods.reduce((acc, per) => acc + Math.cos(((100 - j) * 2 * Math.PI) / per), 0) / periods.length;
            pts.push(`${xMap(j).toFixed(1)},${yMap(val).toFixed(1)}`);
          }
          const curVal = periods.reduce((acc, per) => acc + Math.cos(((100 - activePos) * 2 * Math.PI) / per), 0) / periods.length;
          return `
            <text x="${left}" y="${yOff + 14}" font-size="11" font-weight="600" fill="var(--ink)">${title}</text>
            <line x1="${left}" y1="${yMap(0).toFixed(1)}" x2="${right}" y2="${yMap(0).toFixed(1)}" stroke="var(--rule)"/>
            <line x1="${xMap(activePos).toFixed(1)}" y1="${top}" x2="${xMap(activePos).toFixed(1)}" y2="${bot}" stroke="var(--orange)" stroke-width="1" stroke-dasharray="3 3" opacity="0.8"/>
            <polyline points="${pts.join(" ")}" fill="none" stroke="var(--blue)" stroke-width="1.2"/>
            <circle cx="${xMap(100).toFixed(1)}" cy="${yMap(1).toFixed(1)}" r="3.2" fill="var(--ink)"/>
            <circle cx="${xMap(activePos).toFixed(1)}" cy="${yMap(curVal).toFixed(1)}" r="4" fill="var(--orange)"/>
            <text x="${left - 6}" y="${yMap(1) + 3}" font-size="9.5" fill="var(--ink3)" text-anchor="end">1</text>
            <text x="${left - 6}" y="${yMap(-1) + 3}" font-size="9.5" fill="var(--ink3)" text-anchor="end">−1</text>
          `;
        };
        const aliasSvg = `
          <svg class="w-svg" viewBox="0 0 ${W2} ${H2}">
            ${chartPanel(8, "One circle (period 8): every 8th position collides", [8])}
            ${chartPanel(130, "Three circles (periods 8, 27, 200): unique at pos 100", [8, 27, 200])}
            <text x="${(W2 + 22) / 2}" y="${H2 - 6}" font-size="10" fill="var(--ink3)" text-anchor="middle">position j (compared against position 100)</text>
          </svg>
        `;

        el.innerHTML = `
          <div class="w-controls" style="margin-bottom:10px">
            <div class="w-chips">
              ${[1, 9, 100, 108].map(j => `
                <button class="w-chip ${activePos === j ? "active-orange" : ""}" data-pos="${j}">j = ${j}</button>
              `).join("")}
            </div>
            <div class="w-slider">
              <label><span>Position <i>j</i></span><span class="val">${activePos}</span></label>
              <input type="range" min="1" max="200" step="1" value="${activePos}" data-pos-slider>
            </div>
          </div>
          <div class="w-split">
            <div>${clocksSvg}</div>
            <div>${aliasSvg}</div>
          </div>
        `;
        el.querySelectorAll("[data-pos]").forEach(n => {
          n.addEventListener("click", () => {
            activePos = Number(n.dataset.pos);
            render();
          });
        });
        el.querySelector("[data-pos-slider]").addEventListener("input", e => {
          activePos = Number(e.target.value);
          render();
        });
      }
      render();
    },

    // 6. Sequence representation X = E_ctx + P (with preset switcher!)
    "sequence-matrix-x"(el) {
      const presets = [
        "the cat sat on the",
        "on the mat sat the",
        "a dog chased a cat",
      ];
      let curText = presets[0];

      function render() {
        const words = curText.split(" ");
        const ids = encode(curText);
        const Xtok = ids.map(id => E_BASE[id]);
        const P = offsets(ids.length);
        const X = embed(curText);
        const rowLabels = words.map((w, i) => `${i + 1} ${cw(w)}`);

        el.innerHTML = `
          <div class="w-card">
            <div class="w-head">
              <span class="w-title">Building the sequence matrix <b class=mm>X</b> = <b class=mm>E</b><sub>ctx</sub> + <b class=mm>P</b></span>
              <div class="w-chips">
                ${presets.map(p => `<button class="w-chip ${curText === p ? "active" : ""}" data-seq="${p}"><code>${p}</code></button>`).join("")}
              </div>
            </div>
            <div class="w-seq-fixed">
              ${eqRow(
                gridHTML(Xtok, { rows: rowLabels, cols: ["1", "2"], name: "Word embeddings<br><b class=mm>E</b><sub>ctx</sub>" }),
                "+",
                gridHTML(P, { rows: words.map((_, i) => `<b class=vm>p</b><sub>${i + 1}</sub>`), cols: ["1", "2"], name: "Position offsets<br><b class=mm>P</b>" }),
                "=",
                gridHTML(X, { rows: rowLabels, cols: ["1", "2"], name: "Sequence matrix<br><b class=mm>X</b>" })
              )}
            </div>
            <div class="w-note">Notice how <b class=mm>P</b> remains the same across every sentence.</div>
          </div>
        `;
        el.querySelectorAll("[data-seq]").forEach(b => {
          b.addEventListener("click", () => {
            curText = b.dataset.seq;
            render();
          });
        });
      }
      render();
    },

    // 7. Sequence to distribution schematic
    "sequence-to-dist"(el) {
      const list = VOCAB.map(
        w => `<div style="display:grid;grid-template-columns:4.5rem 1.8rem 2rem;gap:4px;font-size:13.5px;line-height:1.55">
          <span>${cw(w)}</span><span style="color:var(--ink3)">→</span><strong>?</strong>
        </div>`
      ).join("");
      el.innerHTML = `
        <div class="w-card" style="display:flex;justify-content:center;padding:18px">
          <div style="display:flex;align-items:center;gap:20px">
            <div style="text-align:center">
              <div class="mm" style="font-size:30px;line-height:1.1">X</div>
              <div style="font-size:12px;color:var(--ink2)">n × d matrix</div>
            </div>
            <span style="font-size:24px;color:var(--ink2)">→</span>
            <div style="border-left:2px solid var(--rule);padding-left:16px">
              <div style="font-size:11.5px;color:var(--ink2);margin-bottom:4px">candidate v → probability μ(v)</div>
              ${list}
            </div>
          </div>
        </div>
      `;
    },

    // 8. Softmax vs naive normalizations
    "softmax-alternatives"(el) {
      const X = embed(QUESTION);
      const xn = X[X.length - 1];
      const z = E_BASE.map(e => dot(e, xn));
      const sumZ = z.reduce((a, b) => a + b, 0);
      const divRaw = z.map(v => v / sumZ);
      const clipped = z.map(v => Math.max(0, v));
      const sumClip = clipped.reduce((a, b) => a + b, 0);
      const divClip = clipped.map(v => v / sumClip);
      const sm = softmax(z);

      const panel = (title, vals, verdict, showLabels) => {
        const W = showLabels ? 245 : 195, H = 248;
        const left = showLabels ? 56 : 16, right = W - 48;
        const zeroX = vals.some(v => v < 0)
          ? left + ((-Math.min(...vals, 0)) / (Math.max(...vals, 0.1) - Math.min(...vals, 0))) * (right - left)
          : left;
        const span = Math.max(...vals.map(Math.abs), 0.01);
        const scale = (vals.some(v => v < 0) ? (right - left) * 0.48 : (right - left)) / span;

        let bars = `<text x="${left}" y="16" font-size="11.5" font-weight="600" fill="var(--ink)">${title}</text>`;
        bars += `<line x1="${zeroX.toFixed(1)}" y1="24" x2="${zeroX.toFixed(1)}" y2="${H - 26}" stroke="var(--ink3)" stroke-width="0.9"/>`;
        VOCAB.forEach((w, idx) => {
          const y = 28 + idx * 17.5;
          const v = vals[idx];
          const bad = v < 0 || !Number.isFinite(v);
          const bw = Math.abs(v) * scale;
          const bx = v >= 0 ? zeroX : zeroX - bw;
          bars += `<rect x="${bx.toFixed(1)}" y="${y}" width="${Math.max(0.8, bw).toFixed(1)}" height="12" rx="1.5" fill="var(--${bad ? "red" : "blue"})" opacity="0.82"/>`;
          if (showLabels) {
            bars += `<text x="${left - 6}" y="${y + 9.5}" text-anchor="end" font-size="10.5" fill="var(--ink2)">${w}</text>`;
          }
          const tx = v >= 0 ? zeroX + bw + 4 : zeroX - bw - 4;
          bars += `<text x="${tx.toFixed(1)}" y="${y + 9.5}" text-anchor="${v >= 0 ? "start" : "end"}" font-size="9.5" fill="var(--ink2)">${fmtPct(v, 0)}</text>`;
        });
        bars += `<text x="${(left + right) / 2}" y="${H - 7}" text-anchor="middle" font-size="10" fill="var(--ink2)">${verdict}</text>`;
        return `<svg class="w-svg" viewBox="0 0 ${W} ${H}">${bars}</svg>`;
      };

      el.innerHTML = `
        <div style="display:grid;grid-template-columns:1.22fr 1fr 1fr;gap:10px">
          ${panel("1. Divide by sum", divRaw, "invalid (negative entries)", true)}
          ${panel("2. Clip, then divide", divClip, "valid, but 5 exact zeros", false)}
          ${panel("3. Softmax", sm, "valid, every word > 0", false)}
        </div>
      `;
    },

    // 9. First predictor readout E * x_n = z -> softmax(z)
    "first-predictor-readout"(el) {
      let sel = ID["mat"];
      const X = embed(QUESTION);
      const xn = X[X.length - 1];
      const z = E_BASE.map(e => dot(e, xn));
      const expZ = z.map(Math.exp);
      const denom = expZ.reduce((a, b) => a + b, 0);
      const p = softmax(z);

      function render() {
        const probRows = VOCAB.map((w, i) => [
          cw(w),
          fmtNum(z[i], 4),
          fmtNum(expZ[i], 4),
          pctBar(p[i]),
        ]);

        el.innerHTML = `
          <div class="w-card">
            <div class="w-head">
              <span class="w-title">Section 3 predictor on <code>${QUESTION} ____</code></span>
              <span class="w-hint">Click any candidate word to trace its score and probability</span>
            </div>
            <div class="w-hint">Step 1 · take the last row of <b class=mm>X</b></div>
            ${eqRow(
              gridHTML(X, {
                rows: QUESTION.split(" ").map((w, i) => `${i + 1} ${cw(w)}`),
                cols: ["1", "2"],
                name: "<b class=mm>X</b>",
                mark: r => r === X.length - 1,
              }),
              "→",
              gridHTML([[xn[0]], [xn[1]]], { name: "<b><b class=vm>x</b></b><sub>5</sub> (row 5, as a column)" })
            )}
            <div class="w-hint">Step 2 · score every word by its dot product with <b><b class=vm>x</b></b><sub>5</sub>, then softmax</div>
            ${eqRow(
              gridHTML(E_BASE, { rows: VOCAB.map(cw), name: "<b class=mm>E</b>", mark: r => r === sel, rowClickAttr: "fp-row" }),
              "×",
              gridHTML([[xn[0]], [xn[1]]], { name: "<b><b class=vm>x</b></b><sub>5</sub>" }),
              "=",
              gridHTML(z.map(v => [v]), { name: "<b class=vm>z</b>", digits: 4, mark: r => r === sel, rowClickAttr: "fp-row" }),
              "→",
              gridHTML(probRows, {
                cols: ["word", "logit <i>z</i><sub>v</sub>", "exp(<i>z</i><sub>v</sub>)", "μ(v)"],
                foot: ["total", "", fmtNum(denom, 3), `<div class="lt-pc"><span>100.0%</span></div>`],
                bracket: false,
                left: [0],
                mark: r => r === sel,
                rowClickAttr: "fp-row",
              })
            )}
            <div class="w-note">
              For <code>${VOCAB[sel]}</code>:
              <b class=vm>e</b><sub>${VOCAB[sel]}</sub><sup>⊤</sup><b class=vm>x</b><sub>5</sub> = (${fmtNum(E_BASE[sel][0])})(${fmtNum(xn[0])}) + (${fmtNum(E_BASE[sel][1])})(${fmtNum(xn[1])}) = <strong>${fmtNum(z[sel], 4)}</strong>
              &nbsp;→&nbsp; exp(${fmtNum(z[sel], 4)}) / ${fmtNum(denom, 3)} = <strong>${fmtPct(p[sel], 1)}</strong>.
            </div>
          </div>
        `;
        el.querySelectorAll("[data-fp-row]").forEach(tr => {
          tr.addEventListener("click", () => {
            sel = Number(tr.dataset.fpRow);
            render();
          });
        });
      }
      render();
    },

    "summary-walk"(el) {
      const words = QUESTION.split(" ");
      const weights = [0.4, 0.5, 1, 0.6, 0.5];
      const vectors = embed(QUESTION);
      el.innerHTML = `
        <div class="w-card summary-walk">
          <div class="w-chips" role="group" aria-label="Word steps and position weights">
            <button type="button" class="w-chip">Start</button>
            ${words.map((w, j) => `<div class="walk-control">
              <button type="button" class="w-chip" aria-label="Through word ${j + 1}, ${w}">${cw(w)}</button>
              <label>α<sub>${j + 1}</sub> = <span>${weights[j]}</span>
                <input type="range" min="-1" max="1" step="0.05" value="${weights[j]}" aria-label="Weight for position ${j + 1}, ${w}">
              </label>
            </div>`).join("")}
          </div>
          <svg viewBox="0 0 400 245" role="img" aria-label="Weighted word vectors placed tip to tail">
            <defs>
              <marker id="walk-step" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="var(--blue)"/></marker>
              <marker id="walk-sum" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="var(--ink)"/></marker>
            </defs>
            <g data-axes></g>
            <g data-path></g>
          </svg>
          <output aria-live="polite"></output>
        </div>`;
      const buttons = Array.from(el.querySelectorAll("button"));
      const values = el.querySelectorAll("label span");
      const axes = el.querySelector("[data-axes]");
      const path = el.querySelector("[data-path]");
      const output = el.querySelector("output");
      const labelOffsets = [[-27, 5], [-15, -3], [16, 4], [0, 17], [0, 19]];
      function show(count) {
        const points = [[0, 0]];
        vectors.forEach(([x, y], j) => {
          const [a, b] = points[j];
          points.push([a + weights[j] * x, b + weights[j] * y]);
        });
        // Fit the entire walk, even when showing a prefix. Keep equal units on both axes.
        const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
        const left = Math.min(-1.6, ...xs), right = Math.max(0.25, ...xs);
        const top = Math.max(0.7, ...ys), bottom = Math.min(-0.4, ...ys);
        const scale = Math.min(320 / (right - left), 185 / (top - bottom));
        const ox = 200 - (left + right) * scale / 2;
        const oy = 122 + (top + bottom) * scale / 2;
        const plot = points.map(([x, y]) => [ox + scale * x, oy - scale * y]);
        axes.innerHTML = `<path d="M18 ${oy}H382 M${ox} 12V232" fill="none" stroke="var(--rule)"/>
          <circle cx="${ox}" cy="${oy}" r="3" fill="var(--ink)"/>
          <text x="${ox + 5}" y="${oy + 15}" fill="var(--ink2)">origin</text>`;
        buttons.forEach((button, j) => {
          button.classList.toggle("active", j === count);
          button.setAttribute("aria-pressed", String(j === count));
        });
        const steps = words.slice(0, count).map((word, j) => {
          if (weights[j] === 0) return "";
          const [x, y] = plot[j];
          const [u, v] = plot[j + 1];
          const [dx, dy] = labelOffsets[j];
          return `<line x1="${x}" y1="${y}" x2="${u}" y2="${v}" stroke="var(--blue)" stroke-width="3" marker-end="url(#walk-step)"/>
            <text x="${(x + u) / 2 + dx}" y="${(y + v) / 2 + dy}" text-anchor="middle" fill="var(--ink2)">${word}</text>`;
        });
        const [x, y] = plot[count];
        if (count === words.length) {
          steps.push(`<line x1="${ox}" y1="${oy}" x2="${x}" y2="${y}" stroke="var(--ink)" stroke-width="2.5" stroke-dasharray="5 4" marker-end="url(#walk-sum)"/>
            <text x="${x - 12}" y="${y + 4}" text-anchor="end" fill="var(--ink)" font-weight="bold" font-style="italic">h</text>`);
        }
        path.innerHTML = steps.join("");
        output.textContent = count === words.length
          ? `All five steps · h = ${fmtVec(points[count], 3)}`
          : `${count} of 5 steps · ${fmtVec(points[count], 3)}`;
      }
      buttons.forEach((button, j) => button.addEventListener("click", () => show(j)));
      el.querySelectorAll("input").forEach((input, j) => {
        input.addEventListener("input", () => {
          weights[j] = Number(input.value);
          values[j].textContent = fmtNum(weights[j], 2);
          show(words.length);
        });
      });
      show(words.length);
    },

    // 9b. Autoregressive weighted sum over all rows of X (live sliders + vertical row-squash + 2D plane + readout)
    "ar-weights-preview"(el) {
      const PRESETS = {
        equal: [0.2, 0.2, 0.2, 0.2, 0.2],
        distinct: [0.10, 0.15, 0.20, 0.25, 0.30],
        last: [0, 0, 0, 0, 1],
      };
      let mode = "equal"; // "equal" | "distinct" | "last" | "custom"
      let alpha = PRESETS.equal.slice();
      let swapped = false;
      let sel = ID["mat"];

      el.innerHTML = `
        <div class="w-card">
          <div class="w-head">
            <span class="w-title"><b class=vm>z</b><sub><i>v</i></sub> = <b class=vm>e</b><sub><i>v</i></sub><sup>⊤</sup><b class=vm>h</b>, where <b class=vm>h</b> = ∑ α<sub><i>j</i></sub> <b class=vm>x</b><sub><i>j</i></sub></span>
            <div class="w-chips" data-ar-chips></div>
          </div>
          <div class="w-controls" style="display:grid;grid-template-columns:repeat(5,minmax(95px,1fr));gap:10px;margin-bottom:12px" data-ar-sliders>
            ${[0, 1, 2, 3, 4].map(j => `
              <div class="w-slider" style="min-width:0">
                <label><span data-ar-slab="${j}"></span><span class="val" data-ar-sval="${j}"></span></label>
                <input type="range" min="-1.2" max="1.2" step="0.05" value="0.2" data-ar-s="${j}">
              </div>
            `).join("")}
          </div>
          <div data-ar-body></div>
        </div>
      `;

      const chipsEl = el.querySelector("[data-ar-chips]");
      const bodyEl = el.querySelector("[data-ar-body]");
      const sliderInputs = [0, 1, 2, 3, 4].map(j => el.querySelector(`[data-ar-s="${j}"]`));
      const sliderLabs = [0, 1, 2, 3, 4].map(j => el.querySelector(`[data-ar-slab="${j}"]`));
      const sliderVals = [0, 1, 2, 3, 4].map(j => el.querySelector(`[data-ar-sval="${j}"]`));

      sliderInputs.forEach((inp, j) => {
        inp.addEventListener("input", () => {
          alpha[j] = Number(inp.value);
          mode = "custom";
          update(false);
        });
      });

      function update(syncSliders = true) {
        const curText = swapped ? "the sat cat on the" : QUESTION;
        const words = curText.split(" ");
        const X = embed(curText);

        if (syncSliders) {
          sliderInputs.forEach((inp, j) => {
            inp.value = String(alpha[j]);
          });
        }
        words.forEach((w, j) => {
          sliderLabs[j].innerHTML = `α<sub>${j + 1}</sub> (${w})`;
          sliderVals[j].textContent = fmtNum(alpha[j], 2, true);
        });

        chipsEl.innerHTML = `
          <button class="w-chip ${mode === "equal" ? "active" : ""}" data-ar-mode="equal">Equal weights</button>
          <button class="w-chip ${mode === "distinct" ? "active" : ""}" data-ar-mode="distinct">Distinct weights</button>
          <button class="w-chip ${mode === "last" ? "active" : ""}" data-ar-mode="last">Last-row only</button>
          <button class="w-chip ${swapped ? "active-orange" : ""}" data-ar-swap>Swap cat ↔ sat</button>
        `;
        chipsEl.querySelectorAll("[data-ar-mode]").forEach(b => {
          b.addEventListener("click", () => {
            mode = b.dataset.arMode;
            alpha = PRESETS[mode].slice();
            update(true);
          });
        });
        chipsEl.querySelector("[data-ar-swap]").addEventListener("click", () => {
          swapped = !swapped;
          update(false);
        });

        const c = [
          alpha.reduce((acc, a, i) => acc + a * X[i][0], 0),
          alpha.reduce((acc, a, i) => acc + a * X[i][1], 0),
        ];
        const z = E_BASE.map(e => dot(e, c));
        const p = softmax(z);
        const top = [...p.keys()].sort((a, b) => p[b] - p[a]).slice(0, 3);

        const squashRows = X.map((xj, j) => [
          `${j + 1} ${cw(words[j])}`,
          fmtNum(alpha[j], 2, true),
          `(${fmtNum(xj[0], 2, true)}, ${fmtNum(xj[1], 2, true)})`,
          fmtNum(alpha[j] * xj[0], 3, true),
          fmtNum(alpha[j] * xj[1], 3, true),
        ]);

        const squashTable = gridHTML(squashRows, {
          cols: ["row j", "α<sub>j</sub>", "<b class=vm>x</b><sub>j</sub><sup>⊤</sup>", "α<sub>j</sub><i>X</i><sub>j,1</sub>", "α<sub>j</sub><i>X</i><sub>j,2</sub>"],
          foot: [
            "sum ↓",
            fmtNum(alpha.reduce((a, b) => a + b, 0), 2, true),
            "<b class=vm>h</b><sup>⊤</sup> = <b class=vm-greek>α</b><sup>⊤</sup><b class=mm>X</b>",
            `<strong>${fmtNum(c[0], 3, true)}</strong>`,
            `<strong>${fmtNum(c[1], 3, true)}</strong>`,
          ],
          bracket: false,
          left: [0],
          mark2: r => swapped && (r === 1 || r === 2),
          dim: r => Math.abs(alpha[r]) < 1e-6,
        });

        const lim = Math.max(1.48, Math.abs(c[0]) + 0.35, Math.abs(c[1]) + 0.35);
        const plane = makePlaneSVG({
          width: 310,
          height: 255,
          lim,
          table: E_BASE,
          showWords: true,
          wordAlpha: 0.22,
        });

        let planeInner = "";
        X.forEach((xj, j) => {
          const aj = alpha[j];
          const active = Math.abs(aj) > 1e-6;
          const col = aj < 0 ? "red" : "blue";
          if (active) {
            planeInner += `<line x1="${plane.sx(xj[0]).toFixed(1)}" y1="${plane.sy(xj[1]).toFixed(1)}" x2="${plane.sx(c[0]).toFixed(1)}" y2="${plane.sy(c[1]).toFixed(1)}" stroke="var(--${col})" stroke-width="1" stroke-dasharray="3 3" opacity="0.4"/>`;
          }
          const rDot = active ? Math.min(15, 3.2 + 9 * Math.abs(aj)) : 2.8;
          planeInner += `<circle cx="${plane.sx(xj[0]).toFixed(1)}" cy="${plane.sy(xj[1]).toFixed(1)}" r="${rDot.toFixed(1)}" fill="var(--${col})" opacity="${active ? "0.24" : "0.10"}"/>`;
          planeInner += `<circle cx="${plane.sx(xj[0]).toFixed(1)}" cy="${plane.sy(xj[1]).toFixed(1)}" r="3.2" fill="var(--${col})" opacity="${active ? "0.95" : "0.35"}"/>`;
          const isLeft = xj[0] < 0;
          const lx = plane.sx(xj[0]) + (isLeft ? -7 : 7);
          const ly = plane.sy(xj[1]) + (j === 0 ? -5 : j === 4 ? 11 : 4);
          planeInner += `<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" text-anchor="${isLeft ? "end" : "start"}" font-size="10.5" font-weight="${swapped && (j === 1 || j === 2) ? "700" : "500"}" fill="var(--${col})" opacity="${active ? "1" : "0.45"}">x${j + 1} (${words[j]})</text>`;
        });
        if (norm(c) > 1e-6) {
          planeInner += svgArrow(plane.sx, plane.sy, [0, 0], c, "orange", 2.4);
        }
        planeInner += `<circle cx="${plane.sx(c[0]).toFixed(1)}" cy="${plane.sy(c[1]).toFixed(1)}" r="4" fill="var(--orange)"/>`;
        planeInner += `<text x="${(plane.sx(c[0]) + 8).toFixed(1)}" y="${(plane.sy(c[1]) - 6).toFixed(1)}" font-size="11.5" font-weight="700" fill="var(--orange)">h (${fmtNum(c[0], 2)}, ${fmtNum(c[1], 2)})</text>`;

        bodyEl.innerHTML = `
          <div class="w-hint">Right-hand side (<b class=vm>h</b> = <b class=mm>X</b><sup>⊤</sup><b class=vm-greek>α</b>) · scale each row of <b class=mm>X</b> by α<sub><i>j</i></sub> and combine into one 2D vector <b class=vm>h</b> (orange arrow)</div>
          <div class="w-split" style="align-items:center;margin-bottom:12px">
            <div style="overflow-x:auto">${squashTable}</div>
            <div>${plane.wrap(planeInner)}</div>
          </div>
          <div class="w-hint">Left-hand side (<b class=vm>e</b><sub><i>v</i></sub><sup>⊤</sup><b class=vm>h</b>, stacked as <b class=vm>z</b> = <b class=mm>E</b> <b class=vm>h</b>) · test every candidate word against <b class=vm>h</b></div>
          ${eqRow(
            gridHTML(E_BASE, { rows: VOCAB.map(cw), name: "<b class=mm>E</b>", mark: r => r === sel, rowClickAttr: "ar-row" }),
            "×",
            gridHTML([[c[0]], [c[1]]], { digits: 3, name: "<b class=vm>h</b>" }),
            "=",
            gridHTML(z.map(v => [v]), { name: "<b class=vm>z</b>", digits: 3, mark: r => r === sel, rowClickAttr: "ar-row" }),
            "→",
            gridHTML(VOCAB.map((w, i) => [cw(w), fmtNum(z[i], 3), pctBar(p[i])]), {
              cols: ["word", "logit <i>z</i><sub>v</sub>", "μ(v)"],
              bracket: false,
              left: [0],
              mark: r => r === sel,
              rowClickAttr: "ar-row",
            })
          )}
          <div class="w-note" style="min-height: 6em">
            For <code>${VOCAB[sel]}</code>:
            <i>z</i><sub>${VOCAB[sel]}</sub> = <b class=vm>e</b><sub>${VOCAB[sel]}</sub><sup>⊤</sup><b class=vm>h</b> = (${fmtNum(E_BASE[sel][0])})(${fmtNum(c[0], 3)}) + (${fmtNum(E_BASE[sel][1])})(${fmtNum(c[1], 3)}) = <strong>${fmtNum(z[sel], 3)}</strong>, so <i>μ</i>(${VOCAB[sel]}) = <strong>${fmtPct(p[sel], 1)}</strong>.<br>
            ${mode === "last"
              ? `Setting <b class=vm-greek>α</b> = (0, 0, 0, 0, 1)<sup>⊤</sup> puts 100% of the weight on row 5, so <b class=vm>h</b> lands right on <b class=vm>x</b><sub>5</sub>.`
              : mode === "distinct"
                ? swapped
                  ? `With distinct weights (α<sub>2</sub> = +0.15 ≠ α<sub>3</sub> = +0.20), swapping <code>cat</code> and <code>sat</code> pulls <b class=vm>h</b> from (−0.620, +0.062) to <strong>${fmtVec(c, 3)}</strong>—word order is preserved!`
                  : `Distinct weights (+0.10, +0.15, +0.20, +0.25, +0.30)<sup>⊤</sup>: click <strong>Swap cat ↔ sat</strong> and watch <b class=vm>h</b> move!`
                : mode === "custom"
                  ? `Custom weights (${alpha.map(a => fmtNum(a, 2, true)).join(", ")})<sup>⊤</sup> → top predictions: ${top.map(k => `<code>${VOCAB[k]}</code> ${fmtPct(p[k], 1)}`).join(", ")}.`
                  : swapped
                    ? `With equal weights α<sub>j</sub> = +0.20, the center of mass <b class=vm>h</b> = ${fmtVec(c, 3)} didn't move at all when swapping <code>cat</code> and <code>sat</code>! Switch to <strong>Distinct weights</strong> to compare.`
                    : `With equal weights α<sub>j</sub> = +0.20, <b class=vm>h</b> sits at the center of mass of all 5 points; top predictions: ${top.map(k => `<code>${VOCAB[k]}</code> ${fmtPct(p[k], 1)}`).join(", ")}.`}
          </div>
        `;
        bodyEl.querySelectorAll("[data-ar-row]").forEach(tr => {
          tr.addEventListener("click", () => {
            sel = Number(tr.dataset.arRow);
            update(false);
          });
        });
      }
      update(true);
    },

    // 9c. Fixed learned alpha as importance bars under each position; the words move, the bars don't (Section 5)
    "fixed-alpha-ranking"(el) {
      const learnedAlpha = GD_HISTORY.snaps[2000].alpha;
      // The second preset puts filler words under the two largest weights (positions 2 and 3) and the
      // words that carry the meaning under the three smallest; its blank ("the"/"a") is in our vocabulary.
      const presets = ["the cat sat on the", "dog on the grass chased"];
      // Both notes are rendered in the same grid cell (the inactive one hidden), so the card keeps its height.
      const notes = [
        `In this sentence, <code>cat</code> and <code>sat</code> occupy slots 2 and 3, which have the largest positive coefficients.`,
        `In this sentence, those same coefficients multiply <code>on</code> and <code>the</code>. The words we wanted to follow, <code>dog</code> and <code>chased</code>, have moved.`,
      ];
      // Importance = alpha rescaled from the smallest weight (0) to the largest (1).
      const lo = Math.min(...learnedAlpha), hi = Math.max(...learnedAlpha);
      const importance = learnedAlpha.map(a => (a - lo) / (hi - lo));
      const MIN_H = 5, MAX_H = 110;
      let curText = presets[0];

      function render() {
        const words = curText.split(" ");
        // Green when the fixed weights fit the sentence, red when they land on the wrong words.
        const hue = curText === presets[0] ? "var(--aqua)" : "var(--red)";
        // Axis column: invisible word/position slots keep the label lined up with the plot area.
        const axis = `
            <div class="fab-col fab-axis">
              <div class="fab-word fab-ghost">&nbsp;</div>
              <div class="fab-pos">&nbsp;</div>
              <div class="fab-plot" style="height:${MAX_H}px"><span class="fab-axis-label">coefficient →</span></div>
              <div class="fab-val">&nbsp;</div>
            </div>`;
        const cols = axis + words.map((w, i) => {
          const t = importance[i];
          const h = MIN_H + t * (MAX_H - MIN_H);
          const pct = Math.round(15 + 85 * t);
          return `
            <div class="fab-col">
              <div class="fab-word">${w}</div>
              <div class="fab-pos">position ${i + 1}</div>
              <div class="fab-plot" style="height:${MAX_H}px">
                <div class="fab-bar" style="height:${h.toFixed(1)}px;background:color-mix(in srgb, ${hue} ${pct}%, var(--rule))"></div>
              </div>
              <div class="fab-val">α<sub>${i + 1}</sub> = ${fmtNum(learnedAlpha[i], 2, true)}</div>
            </div>`;
        }).join("") + `
            <div class="fab-col">
              <div class="fab-word fab-blank">____</div>
              <div class="fab-pos">position 6</div>
              <div class="fab-plot" style="height:${MAX_H}px"></div>
              <div class="fab-val">&nbsp;</div>
            </div>`;

        el.innerHTML = `
          <div class="w-card">
            <div class="w-head">
              <span class="w-title">Position-only weights belong to slots, not words</span>
              <div class="w-chips">
                ${presets.map(p => `<button class="w-chip ${curText === p ? "active" : ""}" data-far-seq="${p}"><code>${p}</code></button>`).join("")}
              </div>
            </div>
            <div class="fab-row">${cols}</div>
            <div class="w-note fab-notes">
              ${notes.map((n, k) => `<div class="${presets[k] === curText ? "" : "fab-hidden"}">${n}</div>`).join("")}
            </div>
          </div>
        `;
        el.querySelectorAll("[data-far-seq]").forEach(b => {
          b.addEventListener("click", () => {
            curText = b.dataset.farSeq;
            render();
          });
        });
      }
      render();
    },

    // 10. Interactive Gradient Descent on (alpha, E) (27 parameters)
    "training-descent"(el) {
      let stepIdx = 0;
      let timer = null;
      const steps = GD_HISTORY.stepList;
      const qWords = QUESTION.split(" ");

      function stopAnim() {
        if (timer) {
          clearInterval(timer);
          timer = null;
        }
      }

      function render() {
        const t = steps[stepIdx];
        const snap = GD_HISTORY.snaps[t];
        const curE = snap.E;
        const curAlpha = snap.alpha;
        const c = snap.c;
        const z = curE.map(e => dot(e, c));
        const p = softmax(z);
        const curLoss = GD_HISTORY.losses[t];
        const rest = 1 - (p[ID["mat"]] + p[ID["hat"]] + p[ID["grass"]]);

        const maxCoord = Math.max(1.7, ...curE.flat().map(Math.abs), Math.abs(c[0]) * 0.55, Math.abs(c[1]) * 0.55) + 0.35;
        const plane = makePlaneSVG({ width: 320, height: 290, lim: maxCoord, table: curE });

        let trails = "";
        E_BASE.forEach(([x0, y0], k) => {
          const [x1, y1] = curE[k];
          trails += `<line x1="${plane.sx(x0).toFixed(1)}" y1="${plane.sy(y0).toFixed(1)}" x2="${plane.sx(x1).toFixed(1)}" y2="${plane.sy(y1).toFixed(1)}" stroke="var(--rule)" stroke-width="1.1"/>`;
          trails += `<circle cx="${plane.sx(x0).toFixed(1)}" cy="${plane.sy(y0).toFixed(1)}" r="3.2" fill="none" stroke="var(--ink3)" stroke-width="1"/>`;
        });
        // Draw direction of c = sum alpha_j x_j (clamped smoothly to frame if long)
        const cNorm = Math.hypot(c[0], c[1]) || 1;
        const clampScale = cNorm > maxCoord * 0.85 ? (maxCoord * 0.85) / cNorm : 1;
        const cVis = [c[0] * clampScale, c[1] * clampScale];
        const cArrow = svgArrow(plane.sx, plane.sy, [0, 0], cVis, "orange", 2.2);
        const planeSvg = plane.wrap(trails + cArrow);

        // Loss curve (log scale x: step + 1)
        const W2 = 340, H2 = 290, lPad = 42, rPad = 16, tPad = 24, bPad = 34;
        const xLog = s => lPad + (Math.log10(s + 1) / Math.log10(2001)) * (W2 - lPad - rPad);
        const yLoss = L => (H2 - bPad) - ((L - 0.65) / (3.25 - 0.65)) * (H2 - tPad - bPad);
        const curvePts = [];
        for (let s = 0; s <= 2000; s += (s < 50 ? 1 : s < 300 ? 5 : 20)) {
          curvePts.push(`${xLog(s).toFixed(1)},${yLoss(GD_HISTORY.losses[s]).toFixed(1)}`);
        }
        const curveSvg = `
          <svg class="w-svg" viewBox="0 0 ${W2} ${H2}">
            <text x="${lPad}" y="16" font-size="11.5" font-weight="600" fill="var(--ink)">Cross-entropy loss L(p) during descent</text>
            <line x1="${lPad}" y1="${yLoss(LOSS_FLOOR).toFixed(1)}" x2="${W2 - rPad}" y2="${yLoss(LOSS_FLOOR).toFixed(1)}" stroke="var(--ink3)" stroke-dasharray="4 3"/>
            <text x="${W2 - rPad}" y="${yLoss(LOSS_FLOOR) + 13}" font-size="10" fill="var(--ink2)" text-anchor="end">entropy floor H(f) = ${LOSS_FLOOR.toFixed(3)}</text>
            <polyline points="${curvePts.join(" ")}" fill="none" stroke="var(--blue)" stroke-width="2"/>
            <circle cx="${xLog(t).toFixed(1)}" cy="${yLoss(curLoss).toFixed(1)}" r="5" fill="var(--ink)" stroke="var(--page)" stroke-width="1.5"/>
            ${[0, 9, 99, 999, 2000].map(s => `<text x="${xLog(s).toFixed(1)}" y="${H2 - 10}" font-size="10" fill="var(--ink3)" text-anchor="middle">${s === 0 ? "0" : s + 1 === 2001 ? "2000" : s + 1}</text>`).join("")}
            ${[1.04, 2, 2.96].map(v => `<text x="${lPad - 6}" y="${yLoss(v) + 3}" font-size="10" fill="var(--ink3)" text-anchor="end">${v.toFixed(v === 2 ? 0 : 2)}</text>`).join("")}
          </svg>
        `;

        const rows = VOCAB.map((w, k) => [
          cw(w),
          fmtVec(curE[k]),
          fmtNum(z[k], 3),
          pctBar(p[k]),
          pctBar(TARGET[k], true),
        ]);

        el.innerHTML = `
          <div class="w-card">
            <div class="w-head">
              <span class="w-title">Gradient descent on position weights <b class=vm-greek>α</b> (5) and embeddings <b class=mm>E</b> (22)</span>
              <span class="w-hint">Rings = initial <b class=mm>E</b> · orange arrow = direction of summary vector <b class=vm>h</b></span>
            </div>
            <div class="w-controls">
              <button class="w-btn" data-play>${timer ? "Pause" : t >= 2000 ? "Replay" : "Play"}</button>
              <div class="w-slider">
                <label><span>Training steps taken</span><span class="val">${t} / 2000</span></label>
                <input type="range" min="0" max="${steps.length - 1}" step="1" value="${stepIdx}" data-step>
              </div>
            </div>
            <div style="margin-bottom:12px">
              ${eqRow(
                gridHTML([curAlpha], {
                  cols: qWords.map((w, i) => `α<sub>${i + 1}</sub> (${cw(w)})`),
                  signed: true,
                  name: "Learned position weights <b class=vm-greek>α</b><sup>⊤</sup>",
                  mark: (_, cIdx) => curAlpha[cIdx] > 0.21,
                })
              )}
            </div>
            <div class="w-split" style="margin-bottom:14px">
              <div>${planeSvg}</div>
              <div>${curveSvg}</div>
            </div>
            ${eqRow(
              gridHTML(rows, {
                cols: ["word", "row of <b class=mm>E</b>", "logit <i>z</i><sub>v</sub>", "model μ(v)", "observed f(v)"],
                bracket: false,
                left: [0],
                mark: r => TARGET[r] > 0,
              })
            )}
            <div class="w-note">
              Step <strong>${t}</strong>: loss <strong>${curLoss.toFixed(4)}</strong> (excess KL above floor: ${(curLoss - LOSS_FLOOR).toFixed(4)}) ·
              remaining mass on the 8 unobserved words: <strong>${fmtPct(rest, 2)}</strong>.
            </div>
          </div>
        `;

        el.querySelector("[data-step]").addEventListener("input", e => {
          stopAnim();
          stepIdx = Number(e.target.value);
          render();
        });
        el.querySelector("[data-play]").addEventListener("click", () => {
          if (timer) {
            stopAnim();
            render();
            return;
          }
          if (stepIdx >= steps.length - 1) stepIdx = 0;
          timer = setInterval(() => {
            if (stepIdx < steps.length - 1) {
              stepIdx++;
              render();
            } else {
              stopAnim();
              render();
            }
          }, 180);
          render();
        });
      }
      render();
    },

    // 11. Last-row collision explorer (Section 5)
    "last-row-collision"(el) {
      const presets = ["on the mat sat the", "a dog chased a the", "bird sat on hat the"];
      let textB = presets[0];
      let useTrained = false;

      function render() {
        const tab = useTrained ? GD_HISTORY.trained : E_BASE;
        const XA = embed(QUESTION, tab);
        const XB = embed(textB, tab);
        const pA = softmax(tab.map(e => dot(e, XA[4])));
        const pB = softmax(tab.map(e => dot(e, XB[4])));

        const topRows = ["mat", "hat", "grass", "cat", "dog", "bird", "the"].map(w => {
          const id = ID[w];
          return [cw(w), pctBar(pA[id]), pctBar(pB[id])];
        });

        el.innerHTML = `
          <div class="w-card">
            <div class="w-head">
              <span class="w-title">Two different contexts, one identical last row</span>
            </div>
            <div class="w-controls">
              <div class="w-chips">
                ${presets.map(pr => `<button class="w-chip ${textB === pr ? "active" : ""}" data-b="${pr}"><code>${pr}</code></button>`).join("")}
              </div>
              <button class="w-chip ${useTrained ? "active-orange" : ""}" data-toggle-tr>
                Using: ${useTrained ? "Section 4 trained E" : "Hand-placed E"}
              </button>
            </div>
            ${eqRow(
              gridHTML(XA, {
                rows: QUESTION.split(" ").map((w, i) => `${i + 1} ${cw(w)}`),
                cols: ["1", "2"],
                name: `<b class=mm>X</b> for <code>${QUESTION}</code>`,
                mark: r => r === 4,
              }),
              gridHTML(XB, {
                rows: textB.split(" ").map((w, i) => `${i + 1} ${cw(w)}`),
                cols: ["1", "2"],
                name: `<b class=mm>X</b> for <code>${textB}</code>`,
                mark: r => r === 4,
              }),
              gridHTML(topRows, {
                cols: ["candidate", `μ(v | ${QUESTION})`, `μ(v | ${textB})`],
                bracket: false,
                left: [0],
              })
            )}
            <div class="w-note">
              Rows 1–4 differ completely, yet row 5 is <strong>${fmtVec(XA[4])}</strong> in both—so every logit and probability matches bit-for-bit.
            </div>
          </div>
        `;
        el.querySelectorAll("[data-b]").forEach(b => {
          b.addEventListener("click", () => {
            textB = b.dataset.b;
            render();
          });
        });
        el.querySelector("[data-toggle-tr]").addEventListener("click", () => {
          useTrained = !useTrained;
          render();
        });
      }
      render();
    },

    // 12. Pushforward measure on 3-word contexts (1,331 -> 11 atoms)
    "pushforward-atoms"(el) {
      // Compute exact distribution over w3 after 3 words
      const P1 = offsets(1)[0], P2 = offsets(2)[1], P3 = offsets(3)[2];
      const trans = posOff => E_BASE.map(ew => softmax(E_BASE.map(ev => dot(ev, [ew[0] + posOff[0], ew[1] + posOff[1]]))));
      const T1 = trans(P1), T2 = trans(P2);

      let dist2 = Array(11).fill(0);
      for (let w1 = 0; w1 < 11; w1++) {
        for (let w2 = 0; w2 < 11; w2++) dist2[w2] += (1 / 11) * T1[w1][w2];
      }
      let dist3 = Array(11).fill(0);
      for (let w2 = 0; w2 < 11; w2++) {
        for (let w3 = 0; w3 < 11; w3++) dist3[w3] += dist2[w2] * T2[w2][w3];
      }

      const rows3 = E_BASE.map(ew => [ew[0] + P3[0], ew[1] + P3[1]]);
      const plane = makePlaneSVG({ width: 360, height: 300, lim: 1.7, table: E_BASE, showWords: false, wordAlpha: 0.25 });

      let atoms = "";
      rows3.forEach(([x, y], k) => {
        const r = Math.sqrt(dist3[k]) * 38;
        const [dx, dy, anchor] = LABEL_SHIFT[VOCAB[k]] || [0.09, 0.05, "start"];
        atoms += `<circle cx="${plane.sx(x).toFixed(1)}" cy="${plane.sy(y).toFixed(1)}" r="${r.toFixed(1)}" fill="var(--blue)" opacity="0.62" stroke="var(--page)" stroke-width="1"/>`;
        atoms += `<text x="${plane.sx(x + 1.8 * dx).toFixed(1)}" y="${(plane.sy(y + 1.8 * dy) + 3).toFixed(1)}" text-anchor="${anchor}" font-size="10.5" fill="var(--ink2)">${VOCAB[k]} ${fmtPct(dist3[k], 0)}</text>`;
      });

      el.innerHTML = `
        <div style="max-width:400px;margin:0 auto">
          ${plane.wrap(atoms)}
          <div class="w-note" style="text-align:center">1,331 three-word contexts collapse onto 11 atoms (area = probability).</div>
        </div>
      `;
    },

    // 13. Fixed per-position weights: the lower-triangular A grows like n(n+1)/2 (Section 5)
    "weight-triangle"(el) {
      const SIZES = [1, 10, 100, 1000];
      const BOX = 270;     // px: drawing area for the dot pictures, and the height reserved for every size
      let n = 10;
      // One "rule" = one fixed distance back (row i reads row i - k). Its weights lie on a diagonal of A.
      const RULES = [0, 1, 3];
      let rule = 0;
      const onRule = (r, c) => rule > 0 && r - c === rule;
      const tri = k => (k * (k + 1)) / 2;
      const fmtInt = k => k.toLocaleString("en-US");
      // alpha_ij as it appears in the text: alpha_52, and alpha_10,3 once an index has two digits
      const sym = (i, j) => `α<sub>${i > 9 || j > 9 ? `${i},${j}` : `${i}${j}`}</sub>`;

      // n = 1, 10: every symbol written out
      function symbolMatrix() {
        const M = Array.from({ length: n }, (_, r) =>
          Array.from({ length: n }, (__, c) => (c > r ? "0" : sym(r + 1, c + 1))));
        return gridHTML(M, {
          rows: M.map((_, r) => `<b class=vm>h</b><sub>${r + 1}</sub>`),
          name: "<b class=mm>A</b>",
          mark: onRule,
          dim: (r, c) => c > r,
        });
      }

      // n = 100, 1000: too many to write, so one tiny dot per weight
      function dotMatrix() {
        return `
          <div class="lt-m">
            <div class="lt-cap"><b class=mm>A</b><small>${fmtInt(n)}×${fmtInt(n)}</small></div>
            <div class="lt-row">
              <div class="wt-ends" style="height:${BOX}px">
                <span><b class=vm>h</b><sub>1</sub></span><span>⋮</span><span><b class=vm>h</b><sub>${fmtInt(n)}</sub></span>
              </div>
              <div class="lt-b" style="--top:0px"><canvas class="wt-canvas" style="width:${BOX}px;height:${BOX}px"></canvas></div>
            </div>
          </div>`;
      }

      function drawDots() {
        const cv = el.querySelector(".wt-canvas");
        if (!cv) return;
        const dpr = window.devicePixelRatio || 1;
        cv.width = cv.height = Math.round(BOX * dpr);
        const g = cv.getContext("2d");
        const ink = getComputedStyle(el).getPropertyValue("--ink").trim() || "#888";
        const cell = cv.width / n;
        const blue = getComputedStyle(el).getPropertyValue("--blue").trim() || "#4d97f0";
        g.fillStyle = ink;
        // at n = 1000 the dots are smaller than a pixel and merge, so draw them faint: a grey texture, not a white slab
        g.globalAlpha = n > 100 ? 0.45 : 1;
        for (let i = 0; i < n; i++) {
          for (let j = 0; j <= i; j++) {
            if (n <= 100) {
              g.beginPath();
              g.arc((j + 0.5) * cell, (i + 0.5) * cell, cell * 0.3, 0, 2 * Math.PI);
              g.fill();
            } else {
              g.fillRect(j * cell, i * cell, Math.max(cell * 0.6, 0.5), Math.max(cell * 0.6, 0.5));
            }
          }
        }
        // the chosen rule's diagonal, drawn on top in blue
        if (rule > 0 && n > rule) {
          g.globalAlpha = 1;
          g.fillStyle = blue;
          const rad = Math.max(cell * 0.45, 1.4 * dpr);
          for (let i = rule; i < n; i++) {
            g.beginPath();
            g.arc((i - rule + 0.5) * cell, (i + 0.5) * cell, rad, 0, 2 * Math.PI);
            g.fill();
          }
        }
      }

      function ruleNote() {
        if (!rule) return `Pick a rule above to see where its weights live.`;
        const copies = Math.max(0, n - rule);
        const name = `“look ${rule} word${rule > 1 ? "s" : ""} back”`;
        if (copies === 0) return `With <i>n</i> = ${n} there is nothing ${rule} word${rule > 1 ? "s" : ""} back yet.`;
        const list = copies <= 3
          ? Array.from({ length: copies }, (_, k) => sym(rule + k + 1, k + 1)).join(", ")
          : `${sym(rule + 1, 1)}, ${sym(rule + 2, 2)}, …, ${sym(n, n - rule)}`;
        // two concrete copies of the same rule, e.g. alpha_52 and alpha_96 for "3 words back"
        const a = [rule + 2, 2], b = rule + 6 <= n ? [rule + 6, 6] : [n, n - rule];
        const pair = copies > 1 && b[0] !== a[0]
          ? ` Learning ${sym(...a)} doesn't change ${sym(...b)}, so the model has to learn the same thing over again at every position.`
          : "";
        return `The rule ${name} shows up as <strong>${fmtInt(copies)} separate weight${copies > 1 ? "s" : ""}</strong>: ${list}.${pair}`;
      }

      function render() {
        const total = tri(n);
        el.innerHTML = `
          <div class="w-card">
            <div class="w-head">
              <span class="w-title">How many weights <b class=mm>A</b> needs</span>
              <div class="w-chips">
                ${SIZES.map(k => `<button class="w-chip ${n === k ? "active" : ""}" data-wt-n="${k}"><i>n</i> = ${fmtInt(k)}</button>`).join("")}
              </div>
            </div>
            <div class="w-controls wt-rules">
              <span class="wt-rules-lab">highlight one rule:</span>
              <div class="w-chips">
                ${RULES.map(k => `<button class="w-chip ${rule === k ? "active" : ""}" data-wt-rule="${k}">${k ? `look ${k} word${k > 1 ? "s" : ""} back` : "none"}</button>`).join("")}
              </div>
            </div>
            <div class="wt-wrap" style="height:${BOX + 40}px">
              ${n <= 10 ? symbolMatrix() : dotMatrix()}
            </div>
            <div class="w-note wt-note">
              <div><i>n</i> = ${fmtInt(n)}: ${n === 1 ? "" : `1 + 2 + ⋯ + ${fmtInt(n)} = `}<strong>${fmtInt(total)} weight${total > 1 ? "s" : ""}</strong>${n > 10 ? `, one dot each` : ""}.</div>
              <div>${ruleNote()}</div>
            </div>
          </div>
        `;
        drawDots();
        el.querySelectorAll("[data-wt-n]").forEach(b => {
          b.addEventListener("click", () => {
            n = Number(b.dataset.wtN);
            render();
          });
        });
        el.querySelectorAll("[data-wt-rule]").forEach(b => {
          b.addEventListener("click", () => {
            rule = Number(b.dataset.wtRule);
            render();
          });
        });
      }
      render();
    },

    // 14. Equal weights averaging H = U X + Swap cat <-> sat toggle!
    "equal-weights-averaging"(el) {
      let swapped = false;

      function render() {
        const curText = swapped ? "the sat cat on the" : QUESTION;
        const words = curText.split(" ");
        const X = embed(curText);
        const n = words.length;
        const U = Array.from({ length: n }, (_, i) =>
          Array.from({ length: n }, (__, j) => (j <= i ? 1 / (i + 1) : 0))
        );
        const H = matMul(U, X);
        const hn = H[n - 1];
        const z = E_BASE.map(e => dot(e, hn));
        const p = softmax(z);
        const top3 = [...p.keys()].sort((a, b) => p[b] - p[a]).slice(0, 3);

        el.innerHTML = `
          <div class="w-card">
            <div class="w-head">
              <span class="w-title">Equal causal averaging: <b class=mm>H</b> = <b class=mm>U X</b></span>
              <button class="w-chip ${swapped ? "active-orange" : ""}" data-swap>
                ${swapped ? "Swapped: <code>the sat cat on the</code> (click to restore)" : "Swap cat ↔ sat"}
              </button>
            </div>
            ${eqRow(
              gridHTML(U, {
                rows: words.map((w, i) => `${i + 1} ${cw(w)}`),
                cols: ["1", "2", "3", "4", "5"],
                name: "equal weights <b class=mm>U</b>",
                mark: r => r === 4,
                dim: (r, c) => c > r,
              }),
              "×",
              gridHTML(X, {
                cols: ["1", "2"],
                name: "<b class=mm>X</b>",
                mark2: r => swapped && (r === 1 || r === 2),
              }),
              "=",
              gridHTML(H, {
                rows: words.map((w, i) => `${i + 1} ${cw(w)}`),
                cols: ["1", "2"],
                digits: 3,
                name: "<b class=mm>H</b> = <b class=mm>U X</b>",
                mark: r => r === 4,
              })
            )}
            <div class="w-note">
              Final row <strong><b class=vm>h</b><sub>5</sub> = ${fmtVec(hn, 3)}</strong> ${swapped ? "(unchanged despite swapping rows 2 & 3!)" : ""}
              &nbsp;→&nbsp; top predictions: ${top3.map(k => `<code>${VOCAB[k]}</code> ${fmtPct(p[k], 1)}`).join(", ")}.
            </div>
          </div>
        `;
        el.querySelector("[data-swap]").addEventListener("click", () => {
          swapped = !swapped;
          render();
        });
      }
      render();
    },

    // 15. Context-dependent raw dot-product attention
    "content-attention-raw"(el) {
      const presets = [
        "the cat sat on the",
        "dog on the grass chased",
        "a dog chased",
        "the cat sat on the mat",
      ];
      let curText = presets[0];
      let selRow = 4; // default bottom row of "the cat sat on the"
      let selCol = 1; // default col 2 ("cat")

      function render() {
        const words = curText.split(" ");
        const n = words.length;
        if (selRow >= n) selRow = n - 1;
        if (selCol > selRow) selCol = Math.min(1, selRow);

        const X = embed(curText);
        const S = matMul(X, transpose(X));
        const A = causalSoftmax(S);
        const H = matMul(A, X);
        const colLabels = words.map((_, i) => String(i + 1));
        const rowLabels = words.map((w, i) => `${i + 1} ${cw(w)}`);

        el.innerHTML = `
          <div class="w-card">
            <div class="w-head">
              <span class="w-title">Raw dot-product attention: <b class=mm>S</b> = <b class=mm>X X</b><sup>⊤</sup> → <b class=mm>A</b> → <b class=mm>H</b> = <b class=mm>A X</b></span>
              <div class="w-chips">
                ${presets.map(p => `<button class="w-chip ${curText === p ? "active" : ""}" data-car-seq="${p}">“${p}”</button>`).join("")}
              </div>
            </div>
            <div class="w-hint">Click any permitted cell (<i>i</i>, <i>j</i>) in <b class=mm>S</b> or <b class=mm>A</b> (or a row of <b class=mm>H</b>) · prompt length <i>n</i> = ${n}</div>
            ${eqRow(
              gridHTML(S, {
                rows: rowLabels,
                cols: colLabels,
                name: "scores <b class=mm>S</b> = <b class=mm>X X</b><sup>⊤</sup>",
                mark: (r, c) => r === selRow && c <= r && c !== selCol,
                mark2: (r, c) => r === selRow && c === selCol,
                dim: (r, c) => c > r,
                cellClickAttr: "car-cell",
              }),
              "→",
              gridHTML(A, {
                cols: colLabels,
                name: "weights <b class=mm>A</b>",
                mark: (r, c) => r === selRow && c <= r && c !== selCol,
                mark2: (r, c) => r === selRow && c === selCol,
                dim: (r, c) => c > r,
                cellClickAttr: "car-cell",
              }),
              "→",
              gridHTML(H, {
                cols: ["1", "2"],
                name: "summaries <b class=mm>H</b> = <b class=mm>A X</b>",
                mark: r => r === selRow,
                rowClickAttr: "car-row",
              })
            )}
            <div class="w-note">
              <strong>Row ${selRow + 1}</strong> (${cw(words[selRow])}, building <b class=vm>h</b><sub>${selRow + 1}</sub> = ${fmtVec(H[selRow])} to predict word ${selRow + 2}) looking at <strong>col ${selCol + 1}</strong> (${cw(words[selCol])}):
              <i>s</i><sub>${selRow + 1},${selCol + 1}</sub> = <b class=vm>x</b><sub>${selRow + 1}</sub><sup>⊤</sup><b class=vm>x</b><sub>${selCol + 1}</sub> = <strong>${fmtNum(S[selRow][selCol], 2)}</strong>
              &nbsp;→&nbsp; <strong>α<sub>${selRow + 1},${selCol + 1}</sub> = ${fmtPct(A[selRow][selCol], 1)}</strong> of row ${selRow + 1}'s budget.
            </div>
          </div>
        `;
        el.querySelectorAll("[data-car-seq]").forEach(b => {
          b.addEventListener("click", () => {
            curText = b.dataset.carSeq;
            const newLen = curText.split(" ").length;
            selRow = newLen - 1;
            selCol = Math.min(1, selRow);
            render();
          });
        });
        el.querySelectorAll("[data-car-cell]").forEach(td => {
          td.addEventListener("click", () => {
            const [r, c] = td.dataset.carCell.split(",").map(Number);
            if (c <= r) {
              selRow = r;
              selCol = c;
              render();
            }
          });
        });
        el.querySelectorAll("[data-car-row]").forEach(tr => {
          tr.addEventListener("click", () => {
            selRow = Number(tr.dataset.carRow);
            if (selCol > selRow) selCol = selRow;
            render();
          });
        });
      }
      render();
    },

    // 15b. Position-only slot bars vs. Query-Key bars + 2D query arrow
    "qk-circuit-lookup"(el) {
      const presets = [
        "the cat sat on the",
        "dog on the grass chased",
        "a dog chased",
      ];
      let curText = presets[0];
      const learnedAlpha = GD_HISTORY.snaps[2000].alpha;
      const lo = Math.min(...learnedAlpha), hi = Math.max(...learnedAlpha);
      const slotImp = learnedAlpha.map(a => (a - lo) / (hi - lo));

      function render() {
        const words = curText.split(" ");
        const n = words.length;
        const WQ = turnMatrix(sharedAttn.theta, sharedAttn.beta);
        const WK = [[1, 0], [0, 1]];

        const X = embed(curText);
        const Q = matMul(X, WQ);
        const K = matMul(X, WK);
        const qVec = Q[n - 1];
        const scores = K.map(kj => dot(qVec, kj));
        const shares = softmax(scores);

        const MIN_H = 5, BAR_H = 82;
        const fitsOld = curText === presets[0];
        const oldHue = fitsOld ? "var(--aqua)" : "var(--red)";

        // Top row: Section 4 position-only bars (always 5 fixed slots)
        const oldAxis = `
          <div class="fab-col fab-axis">
            <div class="fab-word fab-ghost" style="font-size:18px">&nbsp;</div>
            <div class="fab-pos">&nbsp;</div>
            <div class="fab-plot" style="height:${BAR_H}px"><span class="fab-axis-label">slot α →</span></div>
            <div class="fab-val">&nbsp;</div>
          </div>`;
        let oldCols = oldAxis;
        for (let j = 0; j < 5; j++) {
          const hasWord = j < n;
          const t = slotImp[j];
          const h = MIN_H + t * (BAR_H - MIN_H);
          const pct = Math.round(18 + 82 * t);
          oldCols += `
            <div class="fab-col" style="width:82px">
              <div class="fab-word ${hasWord ? "" : "fab-blank"}" style="font-size:18px;${hasWord ? "" : "color:var(--red);border-color:var(--red)"}">${hasWord ? words[j] : "empty"}</div>
              <div class="fab-pos">slot ${j + 1}</div>
              <div class="fab-plot" style="height:${BAR_H}px">
                <div class="fab-bar" style="height:${h.toFixed(1)}px;background:color-mix(in srgb, ${oldHue} ${pct}%, var(--rule))"></div>
              </div>
              <div class="fab-val">α<sub>${j + 1}</sub> = ${fmtNum(learnedAlpha[j], 2, true)}</div>
            </div>`;
        }
        oldCols += `
          <div class="fab-col" style="width:74px">
            <div class="fab-word fab-blank" style="font-size:18px">____</div>
            <div class="fab-pos">pos ${n + 1}</div>
            <div class="fab-plot" style="height:${BAR_H}px"></div>
            <div class="fab-val">&nbsp;</div>
          </div>`;

        // Bottom-left: Query-Key bars for the actual n words in the prompt
        const newAxis = `
          <div class="fab-col fab-axis">
            <div class="fab-word fab-ghost" style="font-size:18px">&nbsp;</div>
            <div class="fab-pos">&nbsp;</div>
            <div class="fab-plot" style="height:${BAR_H}px"><span class="fab-axis-label">share →</span></div>
            <div class="fab-val">&nbsp;</div>
          </div>`;
        let newCols = newAxis;
        words.forEach((w, j) => {
          const sh = shares[j];
          const isQuery = j === n - 1;
          const h = MIN_H + sh * (BAR_H - MIN_H);
          const pct = Math.round(22 + 78 * Math.min(1, sh * 1.4));
          newCols += `
            <div class="fab-col" style="width:74px">
              <div class="fab-word" style="font-size:18px;${isQuery ? "border-color:var(--orange)" : ""}">${w}</div>
              <div class="fab-pos" style="${isQuery ? "color:var(--orange);font-weight:600" : ""}">pos ${j + 1}${isQuery ? " (q)" : ""}</div>
              <div class="fab-plot" style="height:${BAR_H}px">
                <div class="fab-bar" style="height:${h.toFixed(1)}px;background:color-mix(in srgb, var(--blue) ${pct}%, var(--rule))"></div>
              </div>
              <div class="fab-val"><strong>${fmtPct(sh, 1)}</strong></div>
            </div>`;
        });

        // Bottom-right: Compact 2D plane showing orange query arrow q_n aiming at blue key vectors k_1..k_n
        const maxCoord = Math.max(1.45, ...K.flat().map(Math.abs), Math.abs(qVec[0]), Math.abs(qVec[1])) * 1.22;
        const plane = makePlaneSVG({ width: 230, height: 195, lim: maxCoord, table: null, showWords: false });
        let vecSvg = "";
        K.forEach((kv, j) => {
          const sh = shares[j];
          const strong = sh >= 0.2;
          vecSvg += svgArrow(plane.sx, plane.sy, [0, 0], kv, "blue", strong ? 2.1 : 1.15, strong ? 0.95 : 0.42);
          vecSvg += `<circle cx="${plane.sx(kv[0]).toFixed(1)}" cy="${plane.sy(kv[1]).toFixed(1)}" r="${(2.8 + 8 * sh).toFixed(1)}" fill="var(--blue)" opacity="0.32"/>`;
          const isLeft = kv[0] < 0;
          const ly = plane.sy(kv[1]) + (j === 0 && n === 5 && words[0] === "the" ? -5 : 4);
          vecSvg += `<text x="${(plane.sx(kv[0]) + (isLeft ? -6 : 6)).toFixed(1)}" y="${ly.toFixed(1)}" text-anchor="${isLeft ? "end" : "start"}" font-size="10" font-weight="${strong ? "700" : "400"}" fill="var(--blue)" opacity="${strong ? "1" : "0.65"}">k${j + 1} (${words[j]})</text>`;
        });
        if (norm(qVec) > 1e-6) {
          vecSvg += svgArrow(plane.sx, plane.sy, [0, 0], qVec, "orange", 2.3);
        }
        vecSvg += `<text x="${(plane.sx(qVec[0]) + (qVec[0] < 0 ? -6 : 6)).toFixed(1)}" y="${(plane.sy(qVec[1]) + 12).toFixed(1)}" text-anchor="${qVec[0] < 0 ? "end" : "start"}" font-size="10.5" font-weight="700" fill="var(--orange)">q${n} (${words[n - 1]})</text>`;

        const topWordIdx = shares.indexOf(Math.max(...shares));

        el.innerHTML = `
          <div class="w-card">
            <div class="w-head">
              <span class="w-title">Slot-only weights (5 params) vs. Query–Key attention (8 params)</span>
              <div class="w-chips">
                ${presets.map(p => `<button class="w-chip ${curText === p ? "active" : ""}" data-qkl-seq="${p}"><code>${p}</code></button>`).join("")}
              </div>
            </div>
            <div class="w-hint">1 · Position-only weights (Section 4): bars stay bolted to <strong>slots 2 and 3</strong>, no matter which words sit there</div>
            <div class="fab-row" style="justify-content:flex-start;overflow-x:auto">${oldCols}</div>
            <div class="w-hint" style="border-top:1px solid var(--rule);padding-top:10px;margin-top:6px">
              2 · Query–Key weights (<b class=mm>W</b><sub>Q</sub>, <b class=mm>W</b><sub>K</sub> on <i>d</i> = 2 coordinates): last word aims query <b class=vm style="color:var(--orange)">q</b><sub>${n}</sub>, so the bars follow the matching words at any length
            </div>
            <div style="display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap">
              <div class="fab-row" style="margin:4px 0;overflow-x:auto">${newCols}</div>
              <div style="width:225px;flex:none;margin:0 auto">${plane.wrap(vecSvg)}</div>
            </div>
            <div class="w-note">
              ${curText === presets[0]
                ? `On <code>the cat sat on the</code>, query <b class=vm>q</b><sub>5</sub> (<code>the</code>) turns toward <code>sat</code> (<strong>${fmtPct(shares[2], 1)}</strong>) instead of matching <code>the</code>.`
                : curText === presets[1]
                  ? `Switch to <code>dog on the grass chased</code>: slot weights stay stuck on <code>on</code> and <code>the</code>, while query <b class=vm>q</b><sub>5</sub> (<code>chased</code>) swings right to light up <code>grass</code> (<strong>${fmtPct(shares[3], 1)}</strong>) and <code>dog</code> (<strong>${fmtPct(shares[0], 1)}</strong>)!`
                  : `Switch to a 3-word prompt <code>a dog chased</code>: the 5-slot table runs out of words at slots 4–5, while <b class=vm>q</b><sub>3</sub> (<code>chased</code>) cleanly puts <strong>${fmtPct(shares[topWordIdx], 1)}</strong> on <code>${words[topWordIdx]}</code>.`}
            </div>
          </div>
        `;

        el.querySelectorAll("[data-qkl-seq]").forEach(b => {
          b.addEventListener("click", () => {
            curText = b.dataset.qklSeq;
            render();
          });
        });
      }

      sharedAttn.listeners.push(render);
      render();
    },

    "sat-quarter-turn"(el) {
      const X = embed(QUESTION);
      const key = X[2];
      const px = x => 129 + 67 * x;
      const py = y => 50 - 67 * y;
      function panel(angle, title) {
        const [q] = matMul([X[4]], turnMatrix(angle));
        const scores = X.map(x => dot(q, x));
        const weight = softmax(scores)[2];
        const tipX = px(q[0]), tipY = py(q[1]);
        return `<div class="sat-turn-panel">
          <strong>${title}</strong>
          <svg viewBox="0 0 220 150" aria-hidden="true">
            <defs><marker id="sat-turn-tip-${angle}" markerWidth="5" markerHeight="5" refX="5" refY="3" viewBox="0 0 6 6" orient="auto"><path d="M0 0L6 3L0 6Z" fill="var(--orange)"/></marker></defs>
            <path d="M16 50H204 M129 12V140" stroke="var(--rule)" fill="none"/>
            <line x1="129" y1="50" x2="${px(key[0])}" y2="${py(key[1])}" stroke="var(--blue)" stroke-dasharray="3 3"/>
            <line x1="129" y1="50" x2="${tipX}" y2="${tipY}" stroke="var(--orange)" stroke-width="2.5" marker-end="url(#sat-turn-tip-${angle})"/>
            <circle cx="${px(key[0])}" cy="${py(key[1])}" r="4" stroke="var(--blue)" stroke-width="2" fill="var(--page)"/>
            <text x="${px(key[0]) + 10}" y="${py(key[1]) + 2}" fill="var(--ink)">sat</text>
            <text x="${tipX - 6}" y="${tipY + 4}" text-anchor="end" fill="var(--ink)">query</text>
          </svg>
          <span>Score ${fmtNum(scores[2], 3)} · weight <b>${fmtPct(weight)}</b></span>
        </div>`;
      }
      el.innerHTML = `<figure class="sat-turn" aria-label="Comparing the search arrow with the fixed vector for sat in slot 3">
        ${panel(0, "Before the turn")}${panel(90, "After a quarter turn")}
      </figure>`;
    },

    "summary-turn"(el) {
      const words = QUESTION.split(" ");
      const X = embed(QUESTION);
      const Q = matMul(X, turnMatrix(90, 2.0));
      const alpha = causalSoftmax(matMul(Q, transpose(X)))[4];
      const V = matMul(X, turnMatrix(90, 1.0));
      const panels = [
        { title: "Without <b class=mm>W</b><sub>V</sub>: steps α<sub>5<i>j</i></sub><b class=vm>x</b><sub><i>j</i></sub>", rows: X },
        { title: "With <b class=mm>W</b><sub>V</sub>: steps α<sub>5<i>j</i></sub><b class=vm>v</b><sub><i>j</i></sub>", rows: V },
      ];
      el.innerHTML = `
        <div class="w-card summary-walk">
          <div class="w-chips" role="group" aria-label="Word steps for row 5">
            <button type="button" class="w-chip">Start</button>
            ${words.map((w, j) => `<div class="walk-control">
              <button type="button" class="w-chip" aria-label="Through word ${j + 1}, ${w}">${cw(w)}</button>
              <label>α<sub>5${j + 1}</sub> = ${fmtPct(alpha[j], 1)}</label>
            </div>`).join("")}
          </div>
          <figure class="sat-turn summary-turn">
            ${panels.map((_, k) => `<div class="sat-turn-panel"><strong>${panels[k].title}</strong><div data-plane="${k}"></div><span data-cap="${k}"></span></div>`).join("")}
          </figure>
        </div>`;
      const buttons = Array.from(el.querySelectorAll("button"));

      function draw(k, count) {
        const rows = panels[k].rows;
        const plane = makePlaneSVG({ width: 300, height: 270, lim: 1.65, table: E_BASE, showWords: true, wordAlpha: 0.7 });
        const pts = [[0, 0]];
        rows.forEach((r, j) => pts.push([pts[j][0] + alpha[j] * r[0], pts[j][1] + alpha[j] * r[1]]));
        const px = q => [plane.sx(q[0]), plane.sy(q[1])];
        const end = pts[count];
        const mid = px([pts[5][0] / 2, pts[5][1] / 2]);
        // A label beside a segment, pushed to the side facing away from `avoid`
        function sideLabel(from, to, avoid, text, attrs) {
          const [x1, y1] = px(from), [x2, y2] = px(to);
          const L = Math.hypot(x2 - x1, y2 - y1) || 1;
          let nx = -(y2 - y1) / L, ny = (x2 - x1) / L;
          const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
          if (nx * (avoid[0] - mx) + ny * (avoid[1] - my) > 0) { nx = -nx; ny = -ny; }
          return `<text x="${(mx + 14 * nx).toFixed(1)}" y="${(my + 14 * ny + 4).toFixed(1)}" text-anchor="middle" font-size="12" ${attrs}>${text}</text>`;
        }
        let inner = "";
        for (let j = 0; j < count; j++) {
          const [x1, y1] = px(pts[j]), [x2, y2] = px(pts[j + 1]);
          const current = j === count - 1 && count < words.length;
          const color = current ? "orange" : "blue";
          const head = Math.hypot(x2 - x1, y2 - y1) >= 10 ? ` marker-end="url(#arr-${color})"` : "";
          inner += `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="var(--${color})" stroke-width="3" stroke-linecap="round"${head}/>`;
        }
        if (count >= 3) inner += sideLabel(pts[2], pts[3], mid, "sat", `fill="var(--ink)"`);
        if (count === words.length) {
          const [x, y] = px(end);
          inner += `<line x1="${plane.sx(0).toFixed(1)}" y1="${plane.sy(0).toFixed(1)}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="var(--ink)" stroke-width="2.2" stroke-dasharray="5 4" marker-end="url(#arr-ink)"/>`;
          const satMid = px([(pts[2][0] + pts[3][0]) / 2, (pts[2][1] + pts[3][1]) / 2]);
          inner += sideLabel([0, 0], end, satMid, "h₅", `fill="var(--ink)" font-weight="700"`);
        }
        el.querySelector(`[data-plane="${k}"]`).innerHTML = plane.wrap(inner);
        const cap = el.querySelector(`[data-cap="${k}"]`);
        if (count === words.length) {
          const p = softmax(E_BASE.map(e => dot(e, end)));
          const top = [...p.keys()].sort((a, b) => p[b] - p[a]).slice(0, 3);
          cap.innerHTML = `<b class=vm>h</b><sub>5</sub> = ${fmtVec(end, 2)} · top guesses: ${top.map(i => `<code>${VOCAB[i]}</code>`).join(", ")}`;
        } else if (count === 0) {
          cap.innerHTML = `Start at the origin`;
        } else {
          const step = [alpha[count - 1] * rows[count - 1][0], alpha[count - 1] * rows[count - 1][1]];
          cap.innerHTML = `Step ${count} (${cw(words[count - 1])}) adds ${fmtVec(step, 2)} · now at ${fmtVec(end, 2)}`;
        }
      }
      function show(count) {
        buttons.forEach((button, j) => {
          button.classList.toggle("active", j === count);
          button.setAttribute("aria-pressed", String(j === count));
        });
        draw(0, count);
        draw(1, count);
      }
      buttons.forEach((button, j) => button.addEventListener("click", () => show(j)));
      show(words.length);
    },

    "one-query"(el) {
      const words = QUESTION.split(" ");
      const X = embed(QUESTION);
      const rules = [
        { label: "Original", angle: 0, stretch: 1 },
        { label: "Quarter turn", angle: 90, stretch: 1 },
        { label: "Turn + double", angle: 90, stretch: 2 },
      ];
      let ruleIndex = 0;
      el.innerHTML = `
        <div class="w-card one-query">
          <label>Predict after
            <select>${words.map((_, i) => `<option value="${i}" ${i === words.length - 1 ? "selected" : ""}>${words.slice(0, i + 1).join(" ")} ____</option>`).join("")}</select>
          </label>
          <div class="query-controls"><span>Shared query rule</span>
            <div class="w-chips" role="group" aria-label="Shared query rule">
              ${rules.map(rule => `<button class="w-chip" type="button">${rule.label}</button>`).join("")}
            </div>
          </div>
          <div class="query-body" aria-live="polite"></div>
        </div>`;
      const prefix = el.querySelector("select");
      const buttons = Array.from(el.querySelectorAll("button"));
      const body = el.querySelector(".query-body");
      const plane = makePlaneSVG({ width: 215, height: 215, lim: 2.4, table: null });
      function render() {
        const i = Number(prefix.value);
        const { angle, stretch } = rules[ruleIndex];
        // Only this prefix's final query is needed. Keys and values are X in this example.
        const [query] = matMul([X[i]], turnMatrix(angle, stretch));
        const keys = X.slice(0, i + 1);
        const scores = keys.map(key => dot(query, key));
        const alpha = softmax(scores);
        const [summary] = matMul([alpha], keys);
        const prediction = softmax(E_BASE.map(e => dot(e, summary)));
        const best = prediction.indexOf(Math.max(...prediction));
        buttons.forEach((button, j) => {
          button.setAttribute("aria-pressed", String(j === ruleIndex));
          button.classList.toggle("active-orange", j === ruleIndex);
        });
        const origin = [0, 0];
        const arrows = svgArrow(plane.sx, plane.sy, origin, X[i], "blue", 2.8)
          + svgArrow(plane.sx, plane.sy, origin, query, "orange", 1.8);
        body.innerHTML = `
          <div class="query-columns">
            <div>
              <div class="query-heading">One query from slot ${i + 1} · ${cw(words[i])}</div>
              <div aria-hidden="true">${plane.wrap(arrows)}</div>
              <div class="query-coord"><span class="query-input">Input x<sub>${i + 1}</sub></span> ${fmtVec(X[i])}</div>
              <div class="query-coord"><span class="query-result">Query q<sub>${i + 1}</sub></span> ${fmtVec(query)}</div>
            </div>
            <div>
              <div class="query-heading">Compare q<sub>${i + 1}</sub> with each visible key</div>
              <table><thead><tr><th>Slot</th><th>Fixed key</th><th>Score</th><th>Weight</th></tr></thead>
                <tbody>${keys.map((k, j) => `<tr>
                  <th scope="row">${j + 1} · ${cw(words[j])}</th><td>${fmtVec(k)}</td><td>${fmtNum(scores[j], 3)}</td>
                  <td>${fmtPct(alpha[j])}<span class="query-bar"><i style="width:${alpha[j] * 100}%"></i></span></td>
                </tr>`).join("")}</tbody>
              </table>
            </div>
          </div>
          <div class="query-readout">These weights mix the input vectors into <b>h<sub>${i + 1}</sub></b> = ${fmtVec(summary)}.
            For word ${i + 2}, ${cw(VOCAB[best])} has the highest probability (${fmtPct(prediction[best])}).${best === ID.mat ? "" : ` The chance of ${cw("mat")} is ${fmtPct(prediction[ID.mat])}.`}</div>`;
      }
      prefix.addEventListener("change", render);
      buttons.forEach((button, j) => button.addEventListener("click", () => {
        ruleIndex = j;
        render();
      }));
      render();
    },

    // 16. Linked Q/K Explorer (synced with Section 7)
    "qk-explorer"(el) {
      let qRow = 4; // default row 5 ("the")
      let kRow = 1; // default row 2 ("cat")
      const X = embed(QUESTION);
      const words = QUESTION.split(" ");
      const rowLabels = words.map((w, i) => `${i + 1} ${cw(w)}`);

      function render() {
        const WQ = turnMatrix(sharedAttn.theta, sharedAttn.beta);
        const WK = [[1, 0], [0, 1]];
        const Q = matMul(X, WQ);
        const K = matMul(X, WK);
        const S = matMul(Q, transpose(K));
        const A = causalSoftmax(S);

        // Vector comparison plane for selected qRow vs permitted keys 0..qRow
        const qVec = Q[qRow];
        const keys = K.slice(0, qRow + 1);
        const maxCoord = Math.max(1.5, ...keys.flat().map(Math.abs), Math.abs(qVec[0]), Math.abs(qVec[1])) * 1.25;
        const plane = makePlaneSVG({ width: 310, height: 285, lim: maxCoord, table: null, showWords: false });

        let vecSvg = "";
        keys.forEach((kv, j) => {
          const w = A[qRow][j];
          const isSel = j === kRow;
          vecSvg += svgArrow(plane.sx, plane.sy, [0, 0], kv, "blue", isSel ? 2.4 : 1.3, isSel ? 1 : 0.55);
          vecSvg += `<circle cx="${plane.sx(kv[0]).toFixed(1)}" cy="${plane.sy(kv[1]).toFixed(1)}" r="${(3 + 10 * w).toFixed(1)}" fill="var(--blue)" opacity="0.35"/>`;
          vecSvg += `<text x="${(plane.sx(kv[0]) + 6).toFixed(1)}" y="${(plane.sy(kv[1]) - 5).toFixed(1)}" font-size="10.5" font-weight="${isSel ? "700" : "400"}" fill="var(--blue)">k${j + 1} (${words[j]} ${fmtPct(w, 0)})</text>`;
        });
        if (norm(qVec) > 1e-6) {
          vecSvg += svgArrow(plane.sx, plane.sy, [0, 0], qVec, "orange", 2.4);
        }
        vecSvg += `<text x="${(plane.sx(qVec[0]) + 6).toFixed(1)}" y="${(plane.sy(qVec[1]) + 13).toFixed(1)}" font-size="11" font-weight="700" fill="var(--orange)">q${qRow + 1} (${words[qRow]})</text>`;

        // Compare cat vs bird at position 2
        const Xbird = embed("the bird sat on the");
        const Sbird = matMul(matMul(Xbird, WQ), transpose(matMul(Xbird, WK)));
        const Abird = causalSoftmax(Sbird);

        el.innerHTML = `
          <div class="w-card">
            <div class="w-head">
              <span class="w-title">Interactive Query–Key Explorer: <b class=mm>Q</b> = <b class=mm>X W</b><sub>Q</sub>, <b class=mm>K</b> = <b class=mm>X W</b><sub>K</sub></span>
              <span class="w-hint">Click any permitted cell (<i>i</i>, <i>j</i>) in <b class=mm>S</b> or <b class=mm>A</b> to inspect</span>
            </div>
            <div class="w-controls">
              <div class="w-slider">
                <label><span>Query stretch β</span><span class="val">${sharedAttn.beta.toFixed(1)}</span></label>
                <input type="range" min="0" max="6" step="0.5" value="${sharedAttn.beta}" data-beta>
              </div>
              <div class="w-slider">
                <label><span>Query turn θ</span><span class="val">${sharedAttn.theta}°</span></label>
                <input type="range" min="-180" max="180" step="15" value="${sharedAttn.theta}" data-theta>
              </div>
              <button class="w-btn" data-reset>Reset (β=2, θ=90°)</button>
            </div>
            <div class="w-split" style="margin-bottom:12px">
              <div>${plane.wrap(vecSvg)}</div>
              <div>
                ${eqRow(
                  gridHTML(WQ, { name: "<b class=mm>W</b><sub>Q</sub>" }),
                  gridHTML(Q, { rows: rowLabels, name: "<b class=mm>Q</b>", mark2: r => r === qRow })
                )}
                <div style="margin-top:10px;font-size:12.5px;color:var(--ink2)">
                  <strong>Same <b class=mm>W</b><sub>Q</sub>, different noun at pos 2:</strong><br>
                  • <code>the cat sat on the</code> → α<sub>52</sub> = <strong>${fmtPct(A[4][1], 1)}</strong> (score ${fmtNum(S[4][1], 2)})<br>
                  • <code>the bird sat on the</code> → α<sub>52</sub> = <strong>${fmtPct(Abird[4][1], 1)}</strong> (score ${fmtNum(Sbird[4][1], 2)})
                </div>
              </div>
            </div>
            ${eqRow(
              gridHTML(S, {
                rows: rowLabels,
                cols: ["1", "2", "3", "4", "5"],
                name: "scores <b class=mm>S</b> = <b class=mm>Q K</b><sup>⊤</sup>",
                mark: (r, c) => r === qRow && c <= r && c !== kRow,
                mark2: (r, c) => r === qRow && c === kRow,
                dim: (r, c) => c > r,
                cellClickAttr: "qk-cell",
              }),
              "→",
              gridHTML(A, {
                cols: ["1", "2", "3", "4", "5"],
                name: "weights <b class=mm>A</b>",
                mark: (r, c) => r === qRow && c <= r && c !== kRow,
                mark2: (r, c) => r === qRow && c === kRow,
                dim: (r, c) => c > r,
                cellClickAttr: "qk-cell",
              })
            )}
            <div class="w-note">
              Selected pair <strong>(row ${qRow + 1} ${cw(words[qRow])}, key ${kRow + 1} ${cw(words[kRow])})</strong>:
              <b class=vm>q</b><sub>${qRow + 1}</sub><sup>⊤</sup><b class=vm>k</b><sub>${kRow + 1}</sub>
              = (${fmtNum(Q[qRow][0])})(${fmtNum(K[kRow][0])}) + (${fmtNum(Q[qRow][1])})(${fmtNum(K[kRow][1])})
              = <strong>${fmtNum(S[qRow][kRow], 3)}</strong>
              &nbsp;→&nbsp; weight <strong>α<sub>${qRow + 1},${kRow + 1}</sub> = ${fmtPct(A[qRow][kRow], 1)}</strong>.
            </div>
          </div>
        `;

        el.querySelector("[data-beta]").addEventListener("input", e => {
          sharedAttn.beta = Number(e.target.value);
          sharedAttn.notify();
        });
        el.querySelector("[data-theta]").addEventListener("input", e => {
          sharedAttn.theta = Number(e.target.value);
          sharedAttn.notify();
        });
        el.querySelector("[data-reset]").addEventListener("click", () => {
          sharedAttn.beta = 2.0;
          sharedAttn.theta = 90;
          sharedAttn.notify();
        });
        el.querySelectorAll("[data-qk-cell]").forEach(td => {
          td.addEventListener("click", () => {
            const [r, c] = td.dataset.qkCell.split(",").map(Number);
            if (c <= r) {
              qRow = r;
              kRow = c;
              render();
            }
          });
        });
      }

      sharedAttn.listeners.push(render);
      render();
    },

    // 17. Linked Value Explorer V = X W_V, H = A V (synced with Section 6's A!)
    "value-explorer"(el) {
      const X = embed(QUESTION);
      const words = QUESTION.split(" ");
      const rowLabels = words.map((w, i) => `${i + 1} ${cw(w)}`);

      function render() {
        const WQ = turnMatrix(sharedAttn.theta, sharedAttn.beta);
        const WV = turnMatrix(sharedAttn.phi, sharedAttn.gamma);
        const Q = matMul(X, WQ);
        const K = X;
        const A = causalSoftmax(matMul(Q, transpose(K)));
        const V = matMul(X, WV);
        const AV = matMul(A, V);

        const pOut = softmax(E_BASE.map(e => dot(e, AV[4])));
        const top3 = [...pOut.keys()].sort((a, b) => pOut[b] - pOut[a]).slice(0, 3);

        // Head picture for row 5: keys sized by A[4] (frozen under WV!) + output vector AV[4]
        const lim = Math.max(1.65, ...X.flat().map(Math.abs), ...AV.flat().map(Math.abs)) + 0.3;
        const plane = makePlaneSVG({ width: 300, height: 275, lim, table: E_BASE, showWords: true, wordAlpha: 0.25 });

        let headSvg = "";
        K.forEach((kv, j) => {
          const r = 3.5 + 13 * A[4][j];
          headSvg += `<circle cx="${plane.sx(kv[0]).toFixed(1)}" cy="${plane.sy(kv[1]).toFixed(1)}" r="${r.toFixed(1)}" fill="var(--blue)" opacity="0.82" stroke="var(--page)" stroke-width="1.2"/>`;
          headSvg += `<text x="${plane.sx(kv[0]).toFixed(1)}" y="${(plane.sy(kv[1]) - r - 4).toFixed(1)}" text-anchor="middle" font-size="10" fill="var(--ink)">${j + 1} ${words[j]}</text>`;
        });
        if (norm(AV[4]) > 1e-6) {
          headSvg += svgArrow(plane.sx, plane.sy, [0, 0], AV[4], "ink", 2.3);
        }
        headSvg += `<text x="${(plane.sx(AV[4][0]) + 7).toFixed(1)}" y="${(plane.sy(AV[4][1]) + 4).toFixed(1)}" font-size="11" font-weight="700" fill="var(--ink)">(AV)₅</text>`;

        el.innerHTML = `
          <div class="w-card">
            <div class="w-head">
              <span class="w-title">Interactive Value Explorer: <b class=mm>V</b> = <b class=mm>X W</b><sub>V</sub>, <b class=mm>H</b> = <b class=mm>A V</b></span>
              <span class="w-hint">Move γ or φ: blue attention weights <b class=mm>A</b> stay fixed while <b class=mm>V</b> and (<b class=mm>AV</b>)<sub>5</sub> move!</span>
            </div>
            <div class="w-controls">
              <div class="w-slider">
                <label><span>Value stretch γ</span><span class="val">${sharedAttn.gamma.toFixed(2)}</span></label>
                <input type="range" min="0" max="2" step="0.25" value="${sharedAttn.gamma}" data-gamma>
              </div>
              <div class="w-slider">
                <label><span>Value turn φ</span><span class="val">${sharedAttn.phi}°</span></label>
                <input type="range" min="-180" max="180" step="15" value="${sharedAttn.phi}" data-phi>
              </div>
              <button class="w-btn" data-reset-v>Reset (γ=1, φ=90°)</button>
            </div>
            <div class="w-split" style="margin-bottom:12px">
              <div>${plane.wrap(headSvg)}</div>
              <div>
                ${eqRow(
                  gridHTML(WV, { name: "<b class=mm>W</b><sub>V</sub>" }),
                  gridHTML(V, { rows: rowLabels, name: "<b class=mm>V</b> = <b class=mm>X W</b><sub>V</sub>" })
                )}
              </div>
            </div>
            ${eqRow(
              gridHTML(A, {
                rows: rowLabels,
                cols: ["1", "2", "3", "4", "5"],
                name: "<b class=mm>A</b> (from Section 5)",
                mark: r => r === 4,
                dim: (r, c) => c > r,
              }),
              "×",
              gridHTML(V, { name: "<b class=mm>V</b>", mark: () => true }),
              "=",
              gridHTML(AV, { rows: rowLabels, name: "<b class=mm>H</b> = <b class=mm>A V</b>", digits: 3, mark: r => r === 4 })
            )}
            <div class="w-note">
              Message received at row 5: <strong><b class=vm>h</b><sub>5</sub><sup>⊤</sup> = ${fmtVec(AV[4], 3)}</strong>
              &nbsp;→&nbsp; readout softmax(<b class=mm>E</b> <b class=vm>h</b><sub>5</sub>) favors:
              ${top3.map(k => `<code>${VOCAB[k]}</code> ${fmtPct(pOut[k], 1)}`).join(", ")}.
            </div>
          </div>
        `;

        el.querySelector("[data-gamma]").addEventListener("input", e => {
          sharedAttn.gamma = Number(e.target.value);
          sharedAttn.notify();
        });
        el.querySelector("[data-phi]").addEventListener("input", e => {
          sharedAttn.phi = Number(e.target.value);
          sharedAttn.notify();
        });
        el.querySelector("[data-reset-v]").addEventListener("click", () => {
          sharedAttn.gamma = 1.0;
          sharedAttn.phi = 90;
          sharedAttn.notify();
        });
      }

      sharedAttn.listeners.push(render);
      render();
    },
  };

  // Mount all widgets on DOM ready
  function initWidgets() {
    document.querySelectorAll(".widget-mount[data-widget]").forEach(el => {
      const name = el.dataset.widget;
      const fn = WIDGETS[name];
      if (fn) {
        try {
          fn(el);
        } catch (err) {
          el.innerHTML = `<div class="callout callout-warn">Widget error (${name}): ${err.message}</div>`;
          console.error(err);
        }
      } else {
        el.innerHTML = `<div class="callout callout-warn">Unknown widget: ${name}</div>`;
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initWidgets);
  } else {
    initWidgets();
  }
})();
