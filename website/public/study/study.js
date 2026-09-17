/* ============================================================================
   aman.study — runtime for HTML study modules (no framework, ES2020)
   ----------------------------------------------------------------------------
   window.Study.init(root)   — idempotent; safe to call again after navigation.
   window.Study.destroy(root)
   Works standalone (html.study-standalone) and embedded in the Next.js site.
   Vendor assets (KaTeX, highlight.js) are lazy-loaded from ./vendor/ relative
   to this script's own URL, only when the root actually needs them.
   ========================================================================== */
(function () {
  'use strict';
  if (window.Study && window.Study.__v) return;

  var SCRIPT_URL = (document.currentScript && document.currentScript.src) || '/study/study.js';
  var VENDOR = new URL('./vendor/', SCRIPT_URL).href;

  /* ---------- tiny helpers ---------- */
  var q = function (sel, el) { return (el || document).querySelector(sel); };
  var qa = function (sel, el) { return Array.prototype.slice.call((el || document).querySelectorAll(sel)); };
  var el = function (tag, attrs, html) {
    var e = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { if (k === 'class') e.className = attrs[k]; else e.setAttribute(k, attrs[k]); });
    if (html != null) e.innerHTML = html;
    return e;
  };
  var loaded = {};
  function loadScript(url) {
    if (!loaded[url]) loaded[url] = new Promise(function (res, rej) {
      var s = el('script', { src: url, async: 'true' });
      s.onload = res; s.onerror = function () { delete loaded[url]; rej(new Error('failed ' + url)); };
      document.head.appendChild(s);
    });
    return loaded[url];
  }
  function loadCss(url) {
    if (!loaded[url]) loaded[url] = new Promise(function (res) {
      if (qa('link[href="' + url + '"]').length) return res();
      var l = el('link', { rel: 'stylesheet', href: url }); l.onload = res; l.onerror = res;
      document.head.appendChild(l);
    });
    return loaded[url];
  }

  /* ---------- 1. Code blocks ---------- */
  var LANG_ALIAS = { cuda: 'cpp', metal: 'cpp', 'c++': 'cpp', cc: 'cpp', h: 'c', hpp: 'cpp', py: 'python', ts: 'typescript', js: 'javascript',
    sh: 'bash', shell: 'bash', zsh: 'bash', console: 'bash', asm: 'x86asm', ptx: 'llvm', mlir: 'llvm', toml: 'ini', txt: 'plaintext', text: 'plaintext',
    hlsl: 'glsl', wgsl: 'glsl', yml: 'yaml', jsonc: 'json', proto: 'protobuf', docker: 'dockerfile', diff: 'diff', hs: 'haskell', jl: 'julia',
    triton: 'python', jax: 'python', torch: 'python', rs: 'rust', go: 'go', golang: 'go' };
  var LANG_LABEL = { cpp: 'C++', c: 'C', python: 'Python', bash: 'Shell', x86asm: 'x86 asm', armasm: 'ARM asm', llvm: 'LLVM IR', plaintext: 'Text',
    typescript: 'TypeScript', javascript: 'JavaScript', rust: 'Rust', go: 'Go', json: 'JSON', yaml: 'YAML', sql: 'SQL', glsl: 'Shader', cmake: 'CMake',
    dockerfile: 'Dockerfile', protobuf: 'Protobuf', verilog: 'Verilog', julia: 'Julia', haskell: 'Haskell', scala: 'Scala', nginx: 'nginx', ini: 'INI', diff: 'Diff', makefile: 'Makefile' };
  var EXTRA_LANGS = ['x86asm', 'armasm', 'llvm', 'cmake', 'dockerfile', 'protobuf', 'verilog', 'glsl', 'julia', 'haskell', 'scala', 'nginx'];

  function initCode(root) {
    var pres = qa('pre.code', root).filter(function (p) { return !p.closest('.code-block'); });
    if (!pres.length) return;
    var need = {};
    pres.forEach(function (pre) {
      var raw = (pre.getAttribute('data-lang') || 'plaintext').toLowerCase();
      var lang = LANG_ALIAS[raw] || raw;
      var label = pre.getAttribute('data-label') || LANG_LABEL[lang] || raw.toUpperCase();
      if (raw === 'cuda') label = 'CUDA'; if (raw === 'metal') label = 'Metal'; if (raw === 'ptx') label = 'PTX'; if (raw === 'triton') label = 'Triton';
      var wrap = el('div', { class: 'code-block' });
      var head = el('div', { class: 'code-head' });
      var left = el('div', { style: 'display:flex;align-items:center;gap:.5rem;min-width:0' });
      left.appendChild(el('span', { class: 'lang' }, label));
      var title = pre.getAttribute('data-title'); if (title) left.appendChild(el('span', { class: 'code-title' }, title));
      head.appendChild(left);
      var copy = el('button', { class: 'copy', type: 'button', 'aria-label': 'Copy code' }, 'Copy');
      copy.addEventListener('click', function () {
        var code = q('code', pre) || pre;
        navigator.clipboard.writeText(code.textContent).then(function () {
          copy.textContent = 'Copied'; copy.classList.add('is-done');
          setTimeout(function () { copy.textContent = 'Copy'; copy.classList.remove('is-done'); }, 1600);
        }).catch(function () {});
      });
      head.appendChild(copy);
      pre.parentNode.insertBefore(wrap, pre); wrap.appendChild(head); wrap.appendChild(pre);
      var code = q('code', pre);
      if (code && lang !== 'plaintext' && !pre.hasAttribute('data-nohl')) { code.className = 'language-' + lang; need[lang] = code; }
    });
    // data-hl="3,5-7" → wrap those lines (applied after syntax highlighting so colours survive)
    function applyLineHl(pre) {
      var hl = pre.getAttribute('data-hl'), code = q('code', pre);
      if (!hl || !code || pre.hasAttribute('data-hl-done')) return;
      var set = {};
      hl.split(',').forEach(function (r) { var m = r.trim().split('-'); var a = +m[0], b = +(m[1] || m[0]); for (var i = a; i <= b; i++) set[i] = 1; });
      var lines = code.innerHTML.split('\n');
      code.innerHTML = lines.map(function (l, i) { return set[i + 1] ? '<span class="hl">' + l + '</span>' : l; }).join('\n');
      pre.setAttribute('data-hl-done', '1');
    }
    var langs = Object.keys(need);
    if (!langs.length) { pres.forEach(applyLineHl); return; }
    loadScript(VENDOR + 'hljs/highlight.min.js').then(function () {
      return Promise.all(langs.filter(function (l) { return EXTRA_LANGS.indexOf(l) >= 0 && !(window.hljs.getLanguage && window.hljs.getLanguage(l)); })
        .map(function (l) { return loadScript(VENDOR + 'hljs/languages/' + l + '.min.js').catch(function () {}); }));
    }).then(function () {
      if (!window.hljs) return;
      langs.forEach(function (l) {
        qa('pre.code code.language-' + l, root).forEach(function (code) {
          if (window.hljs.getLanguage(l)) { try { window.hljs.highlightElement(code); } catch (e) {} }
        });
      });
      pres.forEach(applyLineHl);
    }).catch(function () { pres.forEach(applyLineHl); });
  }

  /* ---------- 2. Math (KaTeX) — delimiters: $$…$$ display, \(…\) inline, \[…\] display. NOT single $. ---------- */
  function initMath(root) {
    var txt = root.textContent;
    if (txt.indexOf('$$') < 0 && txt.indexOf('\\(') < 0 && txt.indexOf('\\[') < 0) return;
    loadCss(VENDOR + 'katex/katex.min.css');
    loadScript(VENDOR + 'katex/katex.min.js').then(function () { return loadScript(VENDOR + 'katex/auto-render.min.js'); }).then(function () {
      if (!window.renderMathInElement) return;
      window.renderMathInElement(root, {
        delimiters: [{ left: '$$', right: '$$', display: true }, { left: '\\[', right: '\\]', display: true }, { left: '\\(', right: '\\)', display: false }],
        ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option', 'svg'],
        ignoredClasses: ['nomath', 'code-block'],
        throwOnError: false, strict: false, trust: true, errorColor: 'var(--accent-red)'
      });
    }).catch(function () {});
  }

  /* ---------- 3. Tables → scroll wrapper ---------- */
  function initTables(root) {
    qa('table.tbl', root).forEach(function (t) {
      if (t.parentNode.classList.contains('tbl-wrap')) return;
      var w = el('div', { class: 'tbl-wrap' }); t.parentNode.insertBefore(w, t); w.appendChild(t);
    });
  }

  /* ---------- 4. Figures → click to zoom ---------- */
  function initFigures(root) {
    qa('figure.fig > svg', root).forEach(function (svg) {
      if (svg.__zoom) return; svg.__zoom = 1;
      if (svg.closest('.stepper')) { svg.style.cursor = 'default'; return; }
      svg.addEventListener('click', function () {
        var ov = el('div', { class: 'fig-zoom', role: 'dialog', 'aria-label': 'Zoomed figure' });
        var clone = svg.cloneNode(true); clone.removeAttribute('style'); ov.appendChild(clone);
        var host = root.closest('.study') || root; host.appendChild(ov);
        function close() { ov.remove(); document.removeEventListener('keydown', onKey); }
        function onKey(e) { if (e.key === 'Escape') close(); }
        ov.addEventListener('click', close); document.addEventListener('keydown', onKey);
      });
    });
  }

  /* ---------- 5. Tabs ---------- */
  function initTabs(root) {
    qa('.tabs', root).forEach(function (tabs) {
      if (tabs.__init) return; tabs.__init = 1;
      var btns = qa('.tabs-nav button[data-tab]', tabs), panels = qa('.tabs-panel[data-panel]', tabs);
      function show(id) {
        btns.forEach(function (b) { b.classList.toggle('is-active', b.getAttribute('data-tab') === id); b.setAttribute('aria-selected', b.getAttribute('data-tab') === id); });
        panels.forEach(function (p) { p.classList.toggle('is-active', p.getAttribute('data-panel') === id); });
      }
      btns.forEach(function (b) { b.setAttribute('type', 'button'); b.setAttribute('role', 'tab'); b.addEventListener('click', function () { show(b.getAttribute('data-tab')); }); });
      var active = btns.filter(function (b) { return b.classList.contains('is-active'); })[0] || btns[0];
      if (active) show(active.getAttribute('data-tab'));
    });
  }

  /* ---------- 6. Stepper: svg [data-step="n"] revealed progressively; ol.steps li describe each step ---------- */
  function initSteppers(root) {
    qa('.stepper', root).forEach(function (st) {
      if (st.__init) return; st.__init = 1;
      var nodes = qa('[data-step]', st), items = qa('ol.steps > li', st);
      var max = Math.max(items.length, nodes.reduce(function (m, n) { return Math.max(m, +String(n.getAttribute('data-step')).split('-')[0] || 0); }, 0));
      if (!max) return;
      var cur = 1;
      var ctrl = el('div', { class: 'stepper-controls' });
      var prev = el('button', { type: 'button' }, '← Prev'), next = el('button', { type: 'button' }, 'Next →'), pos = el('span', { class: 'stepper-pos' });
      ctrl.appendChild(prev); ctrl.appendChild(next); ctrl.appendChild(pos);
      var fig = q('figure', st); (fig ? fig.parentNode : st).insertBefore(ctrl, fig ? fig.nextSibling : st.firstChild);
      function render() {
        nodes.forEach(function (n) {
          var spec = String(n.getAttribute('data-step')), m = spec.split('-'), from = +m[0], to = m[1] ? +m[1] : (n.hasAttribute('data-step-until') ? +n.getAttribute('data-step-until') : Infinity);
          n.classList.remove('is-past', 'is-current', 'is-future');
          if (cur < from) n.classList.add('is-future'); else if (cur > to) n.classList.add('is-future'); else if (cur === from || (cur >= from && cur <= to && m[1])) n.classList.add('is-current'); else n.classList.add('is-past');
        });
        items.forEach(function (li, i) { li.classList.remove('is-past', 'is-current', 'is-future'); li.classList.add(i + 1 < cur ? 'is-past' : i + 1 === cur ? 'is-current' : 'is-future'); });
        prev.disabled = cur <= 1; next.disabled = cur >= max; pos.textContent = 'Step ' + cur + ' / ' + max;
      }
      prev.addEventListener('click', function () { if (cur > 1) { cur--; render(); } });
      next.addEventListener('click', function () { if (cur < max) { cur++; render(); } });
      items.forEach(function (li, i) { li.addEventListener('click', function () { cur = i + 1; render(); }); });
      st.setAttribute('tabindex', '0');
      st.addEventListener('keydown', function (e) { if (e.key === 'ArrowRight' && cur < max) { cur++; render(); e.preventDefault(); } if (e.key === 'ArrowLeft' && cur > 1) { cur--; render(); e.preventDefault(); } });
      render();
    });
  }

  /* ---------- 7. Calculators ---------- */
  var FMT = {
    int: function (v) { return Math.round(v).toLocaleString(); },
    si: function (v) { var u = ['', 'k', 'M', 'G', 'T', 'P']; var i = 0; var a = Math.abs(v); while (a >= 1000 && i < u.length - 1) { a /= 1000; i++; } return (v < 0 ? '-' : '') + (a >= 100 ? a.toFixed(0) : a >= 10 ? a.toFixed(1) : a.toFixed(2)) + u[i]; },
    bytes: function (v) { var u = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']; var i = 0; var a = Math.abs(v); while (a >= 1024 && i < u.length - 1) { a /= 1024; i++; } return (a >= 100 ? a.toFixed(0) : a >= 10 ? a.toFixed(1) : a.toFixed(2)) + ' ' + u[i]; },
    bytes10: function (v) { var u = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']; var i = 0; var a = Math.abs(v); while (a >= 1000 && i < u.length - 1) { a /= 1000; i++; } return (a >= 100 ? a.toFixed(0) : a >= 10 ? a.toFixed(1) : a.toFixed(2)) + ' ' + u[i]; },
    pct: function (v) { return (v * 100).toFixed(1) + '%'; },
    raw: function (v) { return String(v); }
  };
  function fmt(v, spec) {
    if (!isFinite(v)) return '—';
    spec = spec || 'auto';
    if (spec.indexOf('fixed:') === 0) return v.toFixed(+spec.slice(6));
    if (FMT[spec]) return FMT[spec](v);
    if (Number.isInteger(v)) return v.toLocaleString();
    var a = Math.abs(v); return a >= 1000 ? Math.round(v).toLocaleString() : a >= 100 ? v.toFixed(1) : a >= 1 ? v.toFixed(2) : v.toPrecision(3);
  }
  function initCalcs(root) {
    qa('.calc', root).forEach(function (c) {
      if (c.__init) return; c.__init = 1;
      var inputs = qa('input[name], select[name]', c), outs = qa('output[data-expr]', c);
      inputs.forEach(function (inp) {
        if (inp.type === 'range' && !inp.nextElementSibling?.classList?.contains('calc-val')) {
          var v = el('span', { class: 'calc-val' }); inp.parentNode.insertBefore(v, inp.nextSibling);
        }
      });
      function scope() {
        var s = {}; inputs.forEach(function (i) { var n = i.tagName === 'SELECT' ? parseFloat(i.value) : parseFloat(i.value); s[i.name] = isNaN(n) ? i.value : n; }); return s;
      }
      function update() {
        var s = scope(), names = Object.keys(s), vals = names.map(function (n) { return s[n]; });
        inputs.forEach(function (i) { var v = i.type === 'range' ? i.nextElementSibling : null; if (v && v.classList.contains('calc-val')) v.textContent = fmt(parseFloat(i.value), i.getAttribute('data-format')) + (i.getAttribute('data-unit') ? ' ' + i.getAttribute('data-unit') : ''); });
        outs.forEach(function (o) {
          try { var f = new Function(names.join(','), 'with (Math) { return (' + o.getAttribute('data-expr') + '); }'); var r = f.apply(null, vals); o.textContent = fmt(r, o.getAttribute('data-format')) + (o.getAttribute('data-unit') ? ' ' + o.getAttribute('data-unit') : ''); o.setAttribute('data-value', String(r)); }
          catch (e) { o.textContent = 'expr error'; }
        });
        c.dispatchEvent(new CustomEvent('calc:update', { detail: s }));
      }
      inputs.forEach(function (i) { i.addEventListener('input', update); i.addEventListener('change', update); });
      c.__update = update; update();
    });
  }

  /* ---------- 8. Citation popovers ---------- */
  function initCites(root) {
    var host = root.closest('.study') || root;
    var pop = null, timer = null;
    function hide() { if (pop) { pop.remove(); pop = null; } }
    qa('a.cite', root).forEach(function (a) {
      if (a.__init) return; a.__init = 1;
      var href = a.getAttribute('href') || ''; if (href.charAt(0) !== '#') return;
      var id = href.slice(1);
      function show() {
        clearTimeout(timer);
        var ref = document.getElementById(id); if (!ref) return;
        hide();
        pop = el('div', { class: 'cite-pop', role: 'tooltip' }, ref.innerHTML);
        host.style.position = host.style.position || 'relative';
        host.appendChild(pop);
        var r = a.getBoundingClientRect(), h = host.getBoundingClientRect();
        var left = Math.max(8, Math.min(r.left - h.left, h.width - pop.offsetWidth - 8));
        pop.style.left = left + 'px'; pop.style.top = (r.bottom - h.top + 6) + 'px';
        pop.addEventListener('mouseenter', function () { clearTimeout(timer); });
        pop.addEventListener('mouseleave', function () { timer = setTimeout(hide, 150); });
      }
      a.addEventListener('mouseenter', show); a.addEventListener('focus', show);
      a.addEventListener('mouseleave', function () { timer = setTimeout(hide, 200); });
      a.addEventListener('blur', hide);
    });
  }

  /* ---------- 9. Standalone chrome: TOC, theme toggle, progress ---------- */
  function slug(t) { return t.toLowerCase().replace(/[^\w\s-]/g, '').trim().replace(/\s+/g, '-').replace(/-+/g, '-').slice(0, 60); }
  function initStandalone(root) {
    var html = document.documentElement;
    if (!html.classList.contains('study-standalone')) return function () {};
    // theme
    var stored = null; try { stored = localStorage.getItem('study-theme'); } catch (e) {}
    var forced = html.getAttribute('data-theme');
    var dark = forced ? forced === 'dark' : stored ? stored === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
    html.classList.toggle('dark', dark);
    if (!q('.study-theme-toggle')) {
      var tg = el('button', { class: 'study-theme-toggle', type: 'button', title: 'Toggle theme', 'aria-label': 'Toggle theme' }, dark ? '☾' : '☀');
      tg.addEventListener('click', function () { dark = !dark; html.classList.toggle('dark', dark); tg.textContent = dark ? '☾' : '☀'; try { localStorage.setItem('study-theme', dark ? 'dark' : 'light'); } catch (e) {} });
      document.body.appendChild(tg);
    }
    // progress bar
    var bar = q('.study-progress') || document.body.appendChild(el('div', { class: 'study-progress' }));
    function onScroll() { var d = document.documentElement; var max = d.scrollHeight - d.clientHeight; bar.style.width = (max > 0 ? Math.min(100, d.scrollTop / max * 100) : 0) + '%'; }
    window.addEventListener('scroll', onScroll, { passive: true }); onScroll();
    // ids on headings
    var heads = qa('h2, h3', root);
    heads.forEach(function (h) { if (!h.id) h.id = slug(h.textContent); });
    // TOC
    var shell = root.closest('.study-shell');
    if (!shell) { shell = el('div', { class: 'study-shell' }); root.parentNode.insertBefore(shell, root); shell.appendChild(root); }
    var toc = q('.study-toc', shell) || shell.appendChild(el('nav', { class: 'study-toc', 'aria-label': 'On this page' }));
    toc.innerHTML = '<p class="toc-title">On this page</p>';
    var ol = el('ol'); toc.appendChild(ol);
    var lis = [];
    heads.forEach(function (h) { if (h.closest('.study-header')) return; var li = el('li', { class: h.tagName === 'H3' ? 'l3' : 'l2' }); li.appendChild(el('a', { href: '#' + h.id }, h.textContent)); ol.appendChild(li); lis.push([h, li]); });
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { lis.forEach(function (p) { p[1].classList.toggle('is-active', p[0] === e.target); }); } });
    }, { rootMargin: '-10% 0px -80% 0px', threshold: 0 });
    lis.forEach(function (p) { io.observe(p[0]); });
    return function () { window.removeEventListener('scroll', onScroll); io.disconnect(); };
  }

  /* ---------- public API ---------- */
  var Study = {
    __v: 1,
    vendor: VENDOR,
    fmt: fmt,
    init: function (root) {
      root = root || q('main.study') || document.body;
      if (root.__studyInit) return root.__studyCleanup;
      root.__studyInit = 1;
      var cleanupStandalone = initStandalone(root);
      initCode(root); initTables(root); initFigures(root); initTabs(root); initSteppers(root); initCalcs(root); initCites(root); initMath(root);
      var cleanup = function () { cleanupStandalone(); root.__studyInit = 0; };
      root.__studyCleanup = cleanup;
      return cleanup;
    },
    destroy: function (root) { if (root && root.__studyCleanup) root.__studyCleanup(); }
  };
  window.Study = Study;

  if (document.documentElement.classList.contains('study-standalone')) {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', function () { Study.init(); });
    else Study.init();
  }
})();
