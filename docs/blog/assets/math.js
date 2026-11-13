(() => {
  function renderMath(root = document) {
    if (!window.katex) return;
    root.querySelectorAll('[data-tex]').forEach(element => {
      if (element.dataset.mathRendered) return;
      katex.render(element.dataset.tex, element, {
        displayMode: element.classList.contains('eq'),
        throwOnError: true,
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
