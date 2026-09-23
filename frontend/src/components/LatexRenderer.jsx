import React, { useState, useMemo, useRef, useEffect, memo } from 'react';
import 'katex/dist/katex.min.css';
import './LatexRenderer.css';
import DOMPurify from 'dompurify';
import { renderLatexToHtml } from '../utils/latexConverter';
import { chunkLatex } from '../utils/latexChunking';
import { renderLatexAsync, scheduleLatexMainTask } from '../utils/latexRenderService';

export const DOMPURIFY_CONFIG = {
  ALLOWED_TAGS: [
    'div', 'span', 'p', 'br', 'hr', 'pre', 'code', 'strong', 'b', 'em', 'i', 'u', 's', 'sub', 'sup', 'small',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'dl', 'dt', 'dd', 'table', 'thead', 'tbody', 'tr', 'th', 'td',
    'math', 'semantics', 'mrow', 'mi', 'mo', 'mn', 'msup', 'msub', 'mfrac', 'mroot', 'msqrt', 'mtext', 'mspace', 'mtable', 'mtr', 'mtd', 'annotation', 'annotation-xml',
    'svg', 'path', 'line', 'rect', 'circle', 'g', 'use', 'defs', 'clippath',
  ],
  ALLOWED_ATTR: [
    'class', 'id', 'title', 'style', 'mathvariant', 'encoding', 'xmlns', 'displaystyle', 'scriptlevel',
    'columnalign', 'rowalign', 'columnspacing', 'rowspacing', 'stretchy', 'symmetric', 'fence', 'separator', 'lspace', 'rspace', 'accent', 'accentunder', 'movablelimits', 'minsize', 'maxsize', 'width', 'height',
    'd', 'viewBox', 'preserveAspectRatio', 'fill', 'stroke', 'stroke-width', 'transform', 'x', 'y', 'dx', 'dy', 'x1', 'y1', 'x2', 'y2', 'r', 'cx', 'cy', 'href', 'xlink:href', 'clip-path',
  ],
  ALLOW_DATA_ATTR: false,
  ALLOW_ARIA_ATTR: false,
  FORBID_TAGS: ['script', 'iframe', 'object', 'embed', 'form', 'input', 'button', 'textarea', 'select', 'option', 'link', 'style', 'base', 'meta'],
  FORBID_ATTR: ['onerror', 'onclick', 'onload', 'onmouseover', 'onfocus', 'onblur', 'onchange', 'onsubmit', 'onkeydown', 'onkeyup', 'onmousedown', 'onmouseup'],
  SANITIZE_DOM: true,
};

// Identity is independent of source revision. Match exact blocks first (including
// duplicates in occurrence order), then reuse unmatched positional slots only.
function reconcileBlocks(previous, texts, allocate) {
  const byText = new Map();
  previous.forEach(block => { const entries = byText.get(block.text) || []; entries.push(block); byText.set(block.text, entries); });
  const used = new Set();
  const next = texts.map(text => {
    const block = byText.get(text)?.shift();
    if (block) used.add(block.id);
    return block || null;
  });
  return next.map((block, index) => {
    if (block) return block;
    const positional = previous[index];
    if (positional && !used.has(positional.id)) {
      used.add(positional.id);
      return { ...positional, text: texts[index], revision: positional.revision + 1 };
    }
    return { id: allocate(), text: texts[index], revision: 0 };
  });
}
function interactionWithin(node) {
  if (!node) return false;
  if (node.contains(document.activeElement)) return true;
  const selection = window.getSelection();
  if (!selection || selection.isCollapsed) return false;
  for (let i = 0; i < selection.rangeCount; i++) {
    try { if (selection.getRangeAt(i).intersectsNode(node)) return true; } catch { /* Detached selection. */ }
  }
  return false;
}
const RenderedChunk = memo(({ text, index }) => {
  const container = useRef(null);
  const height = useRef(Math.max(48, text.length * 0.15));
  const [near, setNear] = useState(false);
  const [pinned, setPinned] = useState(false);
  const [result, setResult] = useState(null);
  const [retry, setRetry] = useState(0);
  const [displayText, setDisplayText] = useState(text);
  useEffect(() => { if (!pinned && !interactionWithin(container.current)) setDisplayText(text); }, [text, pinned]);
  const visible = near || pinned;
  useEffect(() => {
    const node = container.current;
    if (!node) return;
    const pin = () => {
      const selection = window.getSelection();
      let selected = false;
      if (selection && !selection.isCollapsed) {
        for (let i = 0; i < selection.rangeCount; i++) {
          try { if (selection.getRangeAt(i).intersectsNode(node)) selected = true; } catch { /* Detached selection. */ }
        }
      }
      setPinned(selected || node.contains(document.activeElement));
    };
    document.addEventListener('selectionchange', pin);
    document.addEventListener('focusin', pin);
    document.addEventListener('focusout', pin);
    if (typeof IntersectionObserver === 'undefined') {
      // Missing viewport APIs must not eagerly schedule the entire document.
      const update = () => { const rect = node.getBoundingClientRect(); setNear(rect.bottom >= -600 && rect.top <= window.innerHeight + 600); };
      update(); window.addEventListener('scroll', update, true); window.addEventListener('resize', update);
      return () => { document.removeEventListener('selectionchange', pin); document.removeEventListener('focusin', pin); document.removeEventListener('focusout', pin); window.removeEventListener('scroll', update, true); window.removeEventListener('resize', update); };
    }
    const observer = new IntersectionObserver(([entry]) => setNear(entry.isIntersecting), { rootMargin: '600px 0px' });
    observer.observe(node);
    return () => { observer.disconnect(); document.removeEventListener('selectionchange', pin); document.removeEventListener('focusin', pin); document.removeEventListener('focusout', pin); };
  }, []);
  useEffect(() => {
    if (!visible) { setResult(null); return; }
    const controller = new AbortController();
    setResult(null);
    renderLatexAsync(displayText, { signal: controller.signal, priority: 'screen' })
      .then(html => scheduleLatexMainTask(() => DOMPurify.sanitize(html, DOMPURIFY_CONFIG), { signal: controller.signal }))
      .then(html => { if (!controller.signal.aborted) setResult({ html }); })
      .catch(error => { if (!controller.signal.aborted) setResult({ error: error.message || 'Rendering unavailable.' }); });
    return () => controller.abort();
  }, [displayText, visible, retry]);
  useEffect(() => {
    const node = container.current;
    if (!visible || !result || !node) return;
    const measure = () => { height.current = Math.max(48, node.getBoundingClientRect().height); };
    measure();
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(measure); observer.observe(node);
    return () => observer.disconnect();
  }, [visible, result]);
  return <div ref={container} data-chunk={index} className={`latex-chunk${!visible ? ' latex-chunk-placeholder' : ''}`} style={!visible || !result ? { minHeight: height.current } : undefined}>
    {visible && (result?.error ? <div className="latex-render-fallback"><p role="status">Rendered view unavailable for this block: {result.error} Full raw block shown below.</p><button type="button" onClick={() => setRetry(value => value + 1)}>Retry rendered block</button><pre className="latex-raw-content">{displayText}</pre></div>
      : result ? <div dangerouslySetInnerHTML={{ __html: result.html }} /> : <span className="latex-render-pending" role="status">Preparing section…</span>)}
  </div>;
});

/** Trailing updates coalesce, but continuous streaming cannot postpone progress forever. */
function useRenderedSnapshot(content, documentId, enabled) {
  const [snapshot, setSnapshot] = useState({ content, documentId });
  const [wasEnabled, setWasEnabled] = useState(enabled);
  if (wasEnabled !== enabled || snapshot.documentId !== documentId) {
    setWasEnabled(enabled);
    setSnapshot({ content, documentId });
  }
  const latest = useRef(content);
  latest.current = content;
  useEffect(() => {
    if (!enabled) return;
    const interval = setInterval(() => setSnapshot(old => old.documentId === documentId && old.content === latest.current ? old : { content: latest.current, documentId }), 1500);
    return () => clearInterval(interval);
  }, [documentId, enabled]);
  useEffect(() => {
    if (!enabled) return;
    const timer = setTimeout(() => setSnapshot({ content, documentId }), 1500);
    return () => clearTimeout(timer);
  }, [content, documentId, enabled]);
  // A changed document identity never displays the previous document during debounce.
  return snapshot.documentId === documentId ? snapshot.content : content;
}
function LatexRenderer({ content = '', className = '', defaultRaw = false, showToggle = true, showLatex, documentId }) {
  const [mode, setMode] = useState(defaultRaw ? 'raw' : 'rendered');
  const rendered = showLatex === undefined ? mode === 'rendered' : showLatex;
  const snapshot = useRenderedSnapshot(content, documentId, rendered);
  const root = useRef(null);
  const serial = useRef(0);
  const [interacting, setInteracting] = useState(false);
  const [blocks, setBlocks] = useState({ documentId, source: null, items: [] });
  const texts = useMemo(() => rendered ? chunkLatex(snapshot) : [], [snapshot, rendered]);
  // Defer whole reconciliation during selection/focus so even merges/deletions
  // cannot detach a user's selection. Document changes remain immediate.
  if (blocks.documentId !== documentId || (rendered && blocks.source !== snapshot && !interacting && !interactionWithin(root.current))) {
    setBlocks({ documentId, source: snapshot, items: reconcileBlocks(blocks.documentId === documentId ? blocks.items : [], texts, () => `block-${++serial.current}`) });
  }
  useEffect(() => {
    const update = () => {
      setInteracting(interactionWithin(root.current));
      // focusout fires before activeElement settles in some browsers.
      queueMicrotask(() => setInteracting(interactionWithin(root.current)));
    };
    document.addEventListener('selectionchange', update);
    document.addEventListener('focusin', update);
    document.addEventListener('focusout', update);
    return () => { document.removeEventListener('selectionchange', update); document.removeEventListener('focusin', update); document.removeEventListener('focusout', update); };
  }, []);
  const chunks = blocks.items;
  if (!content) return <div className={`latex-renderer ${className}`}>No content</div>;
  return <div className={`latex-renderer ${className}`}>
    {showToggle && showLatex === undefined && <div className="latex-toggle-bar"><div className="latex-toggle-buttons">
      <button className={`latex-toggle-btn ${rendered ? 'active' : ''}`} onClick={() => setMode('rendered')}>Rendered View</button>
      <button className={`latex-toggle-btn ${!rendered ? 'active' : ''}`} onClick={() => setMode('raw')}>Raw Text View</button>
    </div>{rendered && <span className="latex-indicator">Progressive rendered view · {chunks.length} sections</span>}</div>}
    <div className="latex-content-container">
      {!rendered ? <pre className="latex-raw-content">{content}</pre> : <>
        <div className="latex-print-warning">This is a viewport-only preview, not the complete document. Use Download PDF for a complete rendered export, or Raw Text View to print the full source.</div>
        <p className="latex-viewport-notice">Rendered preview loads nearby sections only. Browser Find and Select All may omit offscreen text. Use Raw Text View or download the complete document to search or copy everything.{interacting && blocks.source !== snapshot ? ' Live updates are paused while text is selected or a block has focus.' : ''}</p>
        <div ref={root} className="latex-rendered-content" key={documentId}>{chunks.map(({ text, id }, index) => <RenderedChunk key={id} text={text} index={index} />)}</div>
      </>}
    </div>
  </div>;
}
export { renderLatexToHtml };
export default LatexRenderer;
