(() => {
  function renderMath(root = document) {
    if (!window.katex) return;
    root.querySelectorAll('[data-tex]').forEach(element => {
      if (element.dataset.mathRendered) return;
      katex.render(element.dataset.tex, element, {
        displayMode: element.classList.contains('eq'),
        throwOnError: true,
        // Notation: bold = an array of numbers (\v lowercase vectors, \m capital matrices), plain italic = a single number.
        // Change the style of every vector or matrix in the post here.
        macros: {
          '\\v': '\\mathbf{#1}',
          '\\m': '\\mathbf{#1}',
          // Greek has no upright bold in KaTeX, so bold Greek vectors stay bold italic
          '\\valpha': '\\boldsymbol{\\alpha}',
          // Quoted words inside equations: monospace with the same background as inline `code`
          '\\code': '\\htmlClass{tex-code}{\\texttt{#1}}',
          '\\vmu': '\\boldsymbol{\\mu}'
        },
        // \htmlClass is allowed so equations can colour parts with the page's theme colours
        strict: code => (code === 'htmlExtension' ? 'ignore' : 'error'),
        trust: context => context.command === '\\htmlClass'
      });
      element.dataset.mathRendered = 'true';
    });
  }
  window.renderBlogMath = renderMath;
  renderMath();
})();
