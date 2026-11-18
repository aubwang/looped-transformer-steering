#!/usr/bin/env python3
"""
Compile Markdown posts (e.g. docs/blog/transformer.md) into self-contained
GitHub Pages HTML using local KaTeX and interactive JS widgets.

Usage:
    python3 docs/blog/build.py              # one-shot build
    python3 docs/blog/build.py --serve      # local server with auto-rebuild & live reload
"""

from __future__ import annotations

import argparse
import html
import http.server
import json
import os
import re
import socketserver
import threading
import time
from pathlib import Path

BLOG_DIR = Path(__file__).resolve().parent


def slugify(text: str) -> str:
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"[^\w\s-]", "", clean).strip().lower()
    return re.sub(r"[\s_-]+", "-", clean)


class MarkdownCompiler:
    def __init__(self) -> None:
        self.tokens: list[str] = []

    def _stash(self, rendered_html: str) -> str:
        idx = len(self.tokens)
        self.tokens.append(rendered_html)
        return f"\x00TOK{idx}END\x00"

    def _unstash(self, text: str) -> str:
        # Unstash repeatedly in case tokens nest inside block structures
        prev = None
        while prev != text and "\x00TOK" in text:
            prev = text
            text = re.sub(
                r"\x00TOK(\d+)END\x00",
                lambda m: self.tokens[int(m.group(1))],
                text,
            )
        return text

    def _protect_math_and_code(self, src: str) -> str:
        # 1. Fenced code blocks ```lang ... ```
        def repl_fence(m: re.Match) -> str:
            lang = (m.group(1) or "").strip()
            code = html.escape(m.group(2))
            cls = f' class="language-{html.escape(lang)}"' if lang else ""
            return self._stash(f"<pre><code{cls}>{code}</code></pre>")

        src = re.sub(r"^```([^\n]*)\n(.*?)^```[ \t]*$", repl_fence, src, flags=re.M | re.S)

        # 2. Inline code `...`
        src = re.sub(
            r"`([^`\n]+)`",
            lambda m: self._stash(f"<code>{html.escape(m.group(1))}</code>"),
            src,
        )

        # 3. Display math $$ ... $$
        def repl_display_math(m: re.Match) -> str:
            tex = html.escape(m.group(1).strip(), quote=True)
            return self._stash(f'<div class="eq" data-tex="{tex}"></div>')

        src = re.sub(r"\$\$(.*?)\$\$", repl_display_math, src, flags=re.S)

        # 4. Inline math $ ... $ (non-empty, not starting/ending with space unless single char)
        def repl_inline_math(m: re.Match) -> str:
            tex = html.escape(m.group(1).strip(), quote=True)
            return self._stash(f'<span class="math-inline" data-tex="{tex}"></span>')

        src = re.sub(r"(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\\)\$(?!\$)", repl_inline_math, src)
        return src

    def _format_inline(self, text: str) -> str:
        # Escape raw HTML characters except intentional <br>, <sub>, <sup>
        text = html.escape(text, quote=False)
        for tag in ("br", "sub", "sup", "/sub", "/sup"):
            text = text.replace(f"&lt;{tag}&gt;", f"<{tag}>")

        # Links [label](url)
        text = re.sub(
            r"\[([^\]]+)\]\(([^)]+)\)",
            lambda m: f'<a href="{html.escape(m.group(2), quote=True)}">{m.group(1)}</a>',
            text,
        )
        # Bold+italic ***...***
        text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", text)
        # Bold **...**
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        # Italic *...*
        text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", text)
        # Em-dash --- and en-dash -- (optional typographic nicety if written in ASCII)
        return self._unstash(text)

    def _parse_table(self, lines: list[str]) -> str:
        def split_row(row: str) -> list[str]:
            row = row.strip()
            if row.startswith("|"):
                row = row[1:]
            if row.endswith("|"):
                row = row[:-1]
            return [c.strip() for c in row.split("|")]

        headers = split_row(lines[0])
        align_cells = split_row(lines[1])
        aligns: list[str] = []
        for c in align_cells:
            if c.startswith(":") and c.endswith(":"):
                aligns.append("center")
            elif c.endswith(":"):
                aligns.append("right")
            else:
                aligns.append("left")

        thead = "<tr>" + "".join(
            f'<th style="text-align:{aligns[idx] if idx < len(aligns) else "left"}">{self._format_inline(h)}</th>'
            for idx, h in enumerate(headers)
        ) + "</tr>"

        tbody_rows = []
        for row_line in lines[2:]:
            if not row_line.strip():
                continue
            cells = split_row(row_line)
            tr = "<tr>" + "".join(
                f'<td style="text-align:{aligns[idx] if idx < len(aligns) else "left"}">{self._format_inline(c)}</td>'
                for idx, c in enumerate(cells)
            ) + "</tr>"
            tbody_rows.append(tr)

        return f'<div class="table-wrap"><table class="md-table"><thead>{thead}</thead><tbody>{"".join(tbody_rows)}</tbody></table></div>'

    def _parse_blocks(self, lines: list[str], toc: list[tuple[int, str, str]]) -> str:
        out: list[str] = []
        i = 0
        n = len(lines)

        while i < n:
            line = lines[i]
            stripped = line.strip()

            if not stripped:
                i += 1
                continue

            # HTML comment <!-- ... -->
            if stripped.startswith("<!--"):
                while i < n and "-->" not in lines[i]:
                    i += 1
                i += 1
                continue

            # Widget tag {{widget: name}} or {{widget: name | key=val}}
            m_widget = re.match(r"^\{\{\s*widget:\s*([a-zA-Z0-9_-]+)(?:\s*\|\s*(.+?))?\s*\}\}$", stripped)
            if m_widget:
                w_name = html.escape(m_widget.group(1), quote=True)
                w_opts = html.escape(m_widget.group(2) or "", quote=True)
                attr_opts = f' data-opts="{w_opts}"' if w_opts else ""
                out.append(f'<div class="widget-mount" data-widget="{w_name}"{attr_opts}></div>')
                i += 1
                continue

            # Standalone stashed block (e.g. display math $$...$$ or fenced code block)
            if re.match(r"^\x00TOK\d+END\x00$", stripped):
                out.append(self._unstash(stripped))
                i += 1
                continue

            # Container directives ::: details Title / ::: callout [kind]
            m_open = re.match(r"^(:{3,})\s*(\w+)(?:\s+(.*))?$", stripped)
            if m_open:
                fence = m_open.group(1)
                kind = m_open.group(2).lower()
                rest = (m_open.group(3) or "").strip()
                i += 1
                inner_lines: list[str] = []
                while i < n:
                    if lines[i].strip() == fence:
                        i += 1
                        break
                    inner_lines.append(lines[i])
                    i += 1
                inner_html = self._parse_blocks(inner_lines, toc)
                if kind == "details":
                    summary = self._format_inline(rest or "Details")
                    out.append(f'<details class="md-details"><summary>{summary}</summary><div class="details-body">{inner_html}</div></details>')
                elif kind == "callout":
                    variant = html.escape(rest or "info", quote=True)
                    out.append(f'<aside class="callout callout-{variant}">{inner_html}</aside>')
                else:
                    out.append(f'<div class="block-{html.escape(kind, quote=True)}">{inner_html}</div>')
                continue

            # Horizontal rule
            if re.match(r"^(---|\*\*\*|___)\s*$", stripped):
                out.append("<hr>")
                i += 1
                continue

            # Headings
            m_head = re.match(r"^(#{1,4})\s+(.+)$", stripped)
            if m_head:
                level = len(m_head.group(1))
                raw_title = m_head.group(2).strip()
                rendered_title = self._format_inline(raw_title)
                hid = slugify(rendered_title)
                if level in (2, 3):
                    toc.append((level, hid, re.sub(r"<[^>]+>", "", rendered_title)))
                out.append(f'<h{level} id="{hid}">{rendered_title}</h{level}>')
                i += 1
                continue

            # GFM Table
            if "|" in stripped and i + 1 < n and re.match(r"^\|?[\s:-]+\|[\s|:-]*$", lines[i + 1].strip()):
                t_lines = [lines[i], lines[i + 1]]
                i += 2
                while i < n and "|" in lines[i] and lines[i].strip():
                    t_lines.append(lines[i])
                    i += 1
                out.append(self._parse_table(t_lines))
                continue

            # Blockquote
            if stripped.startswith(">"):
                q_lines = []
                while i < n and lines[i].strip().startswith(">"):
                    q_lines.append(re.sub(r"^>\s?", "", lines[i].strip()))
                    i += 1
                out.append(f"<blockquote>{self._parse_blocks(q_lines, toc)}</blockquote>")
                continue

            # Unordered or ordered list
            m_ul = re.match(r"^(\s*)([-*+])\s+(.+)$", line)
            m_ol = re.match(r"^(\s*)(\d+)\.\s+(.+)$", line)
            if m_ul or m_ol:
                ordered = bool(m_ol)
                tag = "ol" if ordered else "ul"
                pat = r"^\s*\d+\.\s+(.+)$" if ordered else r"^\s*[-*+]\s+(.+)$"
                items: list[list[str]] = []
                while i < n:
                    l_cur = lines[i]
                    if not l_cur.strip():
                        # Check if next non-empty line continues the list or is indented
                        if i + 1 < n and (re.match(pat, lines[i + 1]) or lines[i + 1].startswith("  ")):
                            if items:
                                items[-1].append("")
                            i += 1
                            continue
                        break
                    m_item = re.match(pat, l_cur)
                    if m_item:
                        items.append([m_item.group(1)])
                        i += 1
                    elif l_cur.startswith("  ") and items:
                        items[-1].append(l_cur.strip())
                        i += 1
                    else:
                        break
                li_html = []
                for item_lines in items:
                    joined = " ".join(x for x in item_lines if x)
                    li_html.append(f"<li>{self._format_inline(joined)}</li>")
                out.append(f"<{tag}>{''.join(li_html)}</{tag}>")
                continue

            # Paragraph
            p_lines = []
            while i < n:
                cur = lines[i].strip()
                if (
                    not cur
                    or cur.startswith("#")
                    or cur.startswith("<!--")
                    or cur.startswith(":::")
                    or cur.startswith(">")
                    or re.match(r"^\{\{\s*widget:", cur)
                    or re.match(r"^\x00TOK\d+END\x00$", cur)
                    or re.match(r"^(---|\*\*\*|___)\s*$", cur)
                    or re.match(r"^[-*+]\s+", cur)
                    or re.match(r"^\d+\.\s+", cur)
                    or ("|" in cur and i + 1 < n and re.match(r"^\|?[\s:-]+\|[\s|:-]*$", lines[i + 1].strip()))
                ):
                    break
                p_lines.append(cur)
                i += 1
            out.append(f"<p>{self._format_inline(' '.join(p_lines))}</p>")

        return "\n".join(out)

    def compile(self, raw_md: str) -> tuple[dict[str, str], str, list[tuple[int, str, str]]]:
        meta: dict[str, str] = {}
        body_md = raw_md
        if raw_md.startswith("---\n"):
            end = raw_md.find("\n---\n", 4)
            if end != -1:
                front = raw_md[4:end]
                body_md = raw_md[end + 5:]
                for line in front.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        meta[k.strip()] = v.strip().strip('"').strip("'")

        protected = self._protect_math_and_code(body_md)
        toc: list[tuple[int, str, str]] = []
        html_body = self._parse_blocks(protected.splitlines(), toc)
        return meta, html_body, toc


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="assets/katex/katex.min.css">
<link rel="stylesheet" href="assets/transformer.css?v={css_v}">
</head>
<body>
<div class="page-shell">
  <nav class="toc-rail" aria-label="Table of contents">
    <div class="toc-title">Contents</div>
    {toc_html}
  </nav>
  <main class="prose">
    {body_html}
  </main>
</div>
<script src="assets/katex/katex.min.js"></script>
<script src="assets/math.js?v={math_v}"></script>
<script src="assets/transformer-widgets.js?v={widgets_v}"></script>
{live_reload_script}
</body>
</html>
"""


def build_post(md_path: Path, out_path: Path, live_reload: bool = False) -> None:
    compiler = MarkdownCompiler()
    raw = md_path.read_text(encoding="utf-8")
    meta, body_html, toc = compiler.compile(raw)
    title = html.escape(meta.get("title", "A Transformer on Eleven Words"))

    css_v = int((BLOG_DIR / "assets" / "transformer.css").stat().st_mtime)
    math_v = int((BLOG_DIR / "assets" / "math.js").stat().st_mtime)
    widgets_v = int((BLOG_DIR / "assets" / "transformer-widgets.js").stat().st_mtime)

    toc_items = []
    for level, hid, label in toc:
        cls = "toc-h3" if level == 3 else "toc-h2"
        toc_items.append(f'<a class="{cls}" href="#{hid}">{html.escape(label)}</a>')
    toc_html = "\n    ".join(toc_items)

    lr_script = ""
    if live_reload:
        lr_script = """<script>
(() => {
  let last = null;
  setInterval(async () => {
    try {
      const r = await fetch('/__mtime', { cache: 'no-store' });
      const d = await r.json();
      if (last !== null && d.mtime !== last) location.reload();
      last = d.mtime;
    } catch (_) {}
  }, 500);
})();
</script>"""

    rendered = TEMPLATE.format(
        title=title,
        toc_html=toc_html,
        body_html=body_html,
        live_reload_script=lr_script,
        css_v=css_v,
        math_v=math_v,
        widgets_v=widgets_v,
    )
    out_path.write_text(rendered, encoding="utf-8")


def watched_mtime() -> float:
    targets = [
        BLOG_DIR / "transformer.md",
        BLOG_DIR / "assets" / "transformer.css",
        BLOG_DIR / "assets" / "transformer-widgets.js",
        BLOG_DIR / "build.py",
    ]
    return max((p.stat().st_mtime for p in targets if p.exists()), default=0.0)


def serve(port: int = 8080) -> None:
    md_path = BLOG_DIR / "transformer.md"
    out_path = BLOG_DIR / "transformer.html"
    build_post(md_path, out_path, live_reload=True)
    last_built = [watched_mtime()]

    def watcher() -> None:
        while True:
            time.sleep(0.35)
            m = watched_mtime()
            if m > last_built[0]:
                try:
                    build_post(md_path, out_path, live_reload=True)
                    last_built[0] = watched_mtime()
                    print(f"[build] Rebuilt {out_path.name}")
                except Exception as exc:
                    print(f"[build] Error: {exc}")

    threading.Thread(target=watcher, daemon=True).start()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(BLOG_DIR), **kwargs)

        def do_GET(self):
            if self.path.startswith("/__mtime"):
                payload = json.dumps({"mtime": last_built[0]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
                return
            return super().do_GET()

        def log_message(self, format, *args):
            pass

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        print(f"Serving at http://127.0.0.1:{port}/transformer.html (editing transformer.md auto-reloads)")
        httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Watch files and serve with live reload")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    md_path = BLOG_DIR / "transformer.md"
    out_path = BLOG_DIR / "transformer.html"
    if args.serve:
        serve(args.port)
    else:
        build_post(md_path, out_path, live_reload=False)
        print(f"Built {out_path}")


if __name__ == "__main__":
    main()
