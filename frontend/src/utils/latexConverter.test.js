import { describe, expect, it } from 'vitest';
import { decodeHtmlEntities, renderLatexToHtml } from './latexConverter';
describe('DOM-free converter compatibility', () => {
  it.each(['a\rb', 'a\r\nb', '\r\nleading', 'a\0b', '&#13; &#0;', '&amp;lt; &copy; &#x1F600;', '<b>literal</b>'])('matches textarea RCDATA decoding for %j', source => {
    const textarea = document.createElement('textarea');
    textarea.innerHTML = source;
    expect(decodeHtmlEntities(source)).toBe(textarea.textContent);
  });
  it('renders literal CR and CRLF as line breaks', () => {
    expect(renderLatexToHtml('a\rb\r\nc')).toBe('a<br/>b<br/>c');
  });
  it.each(['\\gdef\\compatmacro{z}', '\\global\\def\\compatmacro{z}', '\\global\\let\\compatmacro=z'])('shares global definitions within a call but not between calls: %s', definition => {
    const html = renderLatexToHtml(`$${definition}\\compatmacro$ then $\\compatmacro$`);
    expect(html).not.toContain('katex-error');
    expect(html).not.toContain('#cc0000');
    expect(renderLatexToHtml('$\\compatmacro$')).toContain('#cc0000');
  });
  it.each(['', '*'])('renders only matched equation stars (%s)', star => {
    expect(renderLatexToHtml(`\\begin{equation${star}}x=1\\end{equation${star}}`)).toContain('latex-display');
    expect(renderLatexToHtml(`\\begin{equation${star}}x=1\\end{equation${star ? '' : '*'}}`)).not.toContain('latex-display');
  });
  it('decodes named, numeric, astral, and exactly one entity layer', () => {
    expect(decodeHtmlEntities('&copy; &#x1F600; &amp;lt; &#39; &NotEqualTilde;')).toBe('© 😀 &lt; \' ≂̸');
  });
  it('retains theorem, nested heading, citation, list and footnote semantics', () => {
    const html = renderLatexToHtml('\\section{Nested {heading}} \\cite{ref} \\footnote{note} \\begin{theorem}[Title]$x=1$\\end{theorem} \\begin{itemize}\\item One\\item Two\\end{itemize}');
    expect(html).toContain('<h2 class="latex-section">Nested {heading}</h2>');
    expect(html).toContain('[ref]'); expect(html).toContain('latex-footnote');
    expect(html).toContain('Theorem (Title).'); expect(html).toContain('katex'); expect(html).toContain('<li>Two</li>');
  });
  it.each(['\\[\\begin{tikzcd}A & B\\end{tikzcd}\\]', '$$\\begin{tikzcd}A & B\\end{tikzcd}$$', '\\begin{tikzcd}A & B\\end{tikzcd}'])('preserves TikZ fallback before KaTeX', source => {
    const html = renderLatexToHtml(source); expect(html).toContain('tikz-code'); expect(html).not.toContain('katex-error');
  });
  it('keeps fenced code literal and escaped', () => {
    const html = renderLatexToHtml('```js\n<script>x</script> $x$ \\n```');
    expect(html).toContain('&lt;script&gt;'); expect(html).not.toContain('class="katex"'); expect(html).toContain('latex-code-block');
  });
  it('preserves matrix row separators through KaTeX', () => {
    expect(renderLatexToHtml('$$\\begin{pmatrix}1 & 2\\\\3 & 4\\end{pmatrix}$$')).toContain('katex');
  });
});
