import katex from 'katex';
import { decodeHTML } from 'entities';

export const escapeHtml = (text) => String(text).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
// Match textarea RCDATA preprocessing before decoding exactly one entity layer.
// Character references such as &#13; must not undergo literal-input normalization.
export const decodeHtmlEntities = (text) => text ? decodeHTML(text.replace(/\r\n?/g, '\n').replace(/\0/g, '\uFFFD')) : text;
const extendedMacros = Object.fromEntries([
  ...Object.entries({ R: 'R', N: 'N', Z: 'Z', Q: 'Q', C: 'C', F: 'F', P: 'P', A: 'A', H: 'H', K: 'K' }).map(([key, value]) => [`\\${key}`, `\\mathbb{${value}}`]),
  ...'GL SL PSL PGL SO Sp SU Lie Aut End Hom Gal Spec Proj Ad ad Val Ind Res Tr Vol rank Out Inn Cent Stab Orb supp sgn id Id diag disc STr Irr Rep cusp temp tw im re ker coker deg codim dim ord char Char Pic Div Cl NS Br Ext Tor colim holim hocolim'.split(' ').map(key => [`\\${key}`, `\\mathrm{${key}}`]),
  ['\\Op', '\\mathrm{O}'],
]);
const MATH_COMMAND_PATTERNS = 'alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi omicron pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega mathcal mathbb mathfrak mathscr mathbf mathrm mathsf mathtt mathit boldsymbol text textrm textit textbf textsf texttt frac tfrac dfrac sqrt root sum prod coprod int oint iint iiint bigcup bigcap bigsqcup bigvee bigwedge bigoplus bigotimes lim limsup liminf sup inf max min log ln exp sin cos tan cot sec csc arcsin arccos arctan sinh cosh tanh coth det gcd lcm Pr hom arg leq geq neq approx equiv cong sim simeq subset supset subseteq supseteq subsetneq supsetneq in notin ni owns prec succ preceq succeq ll gg lll ggg perp parallel mid nmid vdash dashv models vDash Vdash to rightarrow leftarrow leftrightarrow Rightarrow Leftarrow Leftrightarrow mapsto longmapsto hookrightarrow hookleftarrow twoheadrightarrow rightarrowtail uparrow downarrow updownarrow nearrow searrow nwarrow swarrow leadsto rightsquigarrow left right big Big bigg Bigg langle rangle lfloor rfloor lceil rceil lvert rvert lVert rVert { } | hat widehat tilde widetilde bar overline vec overrightarrow overleftarrow dot ddot dddot acute grave breve check underline overbrace underbrace quad qquad , ; ! infty partial nabla forall exists nexists emptyset varnothing setminus times otimes oplus circ bullet cdot cdots ldots vdots ddots star ast dagger ddagger pm mp div wedge vee cap cup neg lnot land lor prime backprime aleph beth gimel ell wp Re Im hbar hslash triangle square Diamond Box clubsuit diamondsuit heartsuit spadesuit stackrel overset underset xrightarrow xleftarrow binom choose atop matrix pmatrix bmatrix vmatrix Vmatrix array cases GL SL PSL PGL SO Sp SU Aut End Hom Gal Spec Proj Tr Ad ad Val Ind Res Cent Stab Orb STr Irr Rep Ext Tor Pic Div Cl Br'.split(' ').map(cmd => `\\\\${cmd}`);
const findMatchingBrace = (text, startPos) => {
  let braces = 1, i = startPos;
  while (i < text.length && braces > 0) {
    if (text[i] === '\\' && i + 1 < text.length) { i += 2; continue; }
    if (text[i] === '{') braces++;
    if (text[i] === '}') braces--;
    i++;
  }
  return braces === 0 ? i - 1 : -1;
};
const autoWrapMath = (text) => {
  if (!text || text.includes('$') || text.includes('\\[') || text.includes('\\(')) return text;
  let result = text;
  const regions = [];
  const patterns = [/\$\$[\s\S]*?\$\$/g, /\\\[[\s\S]*?\\\]/g, /(?<!\$)\$(?!\$)(?:[^$\\]|\\.)+?\$(?!\$)/g, /\\\([\s\S]*?\\\)/g];
  patterns.forEach(pattern => { result = result.replace(pattern, match => { const index = regions.push(match) - 1; return `⟦MATHRGN⟧${index}⟦/MATHRGN⟧`; }); });
  const pattern = new RegExp('(' + MATH_COMMAND_PATTERNS.join('|') + ')' + '(?:[_{^](?:\\{[^}]*\\}|[^\\s{},;.!?)]))*' + '(?:\\{[^}]*\\})*' + '(?:[_{^](?:\\{[^}]*\\}|[^\\s{},;.!?]))*', 'g');
  result = result.replace(pattern, match => match.trim().length < 3 ? match : `$${match}$`);
  result = result.replace(/(\$[^$]+\$)\s*([A-Za-z])([_^])(\{[^}]+\}|[A-Za-z0-9])/g, (_, before, letter, op, sub) => `${before} $${letter}${op}${sub}$`);
  result = result.replace(/\$\s*\$/g, '');
  result = result.replace(/\$\$([^$]+)\$\$/g, (match, inner) => !inner.includes('\n') && inner.length < 100 ? `$${inner}$` : match);
  for (let i = regions.length - 1; i >= 0; i--) result = result.replace(`⟦MATHRGN⟧${i}⟦/MATHRGN⟧`, regions[i]);
  return result;
};
const replaceSectionCommand = (text, command, tag, endTag) => {
  const regex = new RegExp(`\\\\${command}\\s*\\{`, 'g');
  let result = '', lastIndex = 0, match;
  while ((match = regex.exec(text)) !== null) {
    const open = match.index + match[0].length - 1;
    const close = findMatchingBrace(text, open + 1);
    if (close !== -1) {
      result += text.substring(lastIndex, match.index) + tag + text.substring(open + 1, close) + endTag;
      lastIndex = close + 1;
    }
  }
  return result + text.substring(lastIndex);
};
const renderKatexSafely = (latex, displayMode, originalMatch, macros) => {
  if (!latex.trim()) return '';
  try {
    const html = katex.renderToString(latex.trim(), { displayMode, throwOnError: false, strict: false, trust: true, macros, maxExpand: 5000, maxSize: 500 });
    return displayMode ? `<div class="latex-display">${html}</div>` : `<span class="latex-inline">${html}</span>`;
  } catch (error) {
    const tag = displayMode ? 'div' : 'span';
    return `<${tag} class="latex-error${displayMode ? ' latex-display-error' : ''}" title="${escapeHtml(error.message)}">${escapeHtml(originalMatch)}</${tag}>`;
  }
};
const isEscapedAt = (text, index) => {
  let count = 0;
  for (let i = index - 1; i >= 0 && text[i] === '\\'; i--) count++;
  return count % 2 === 1;
};
const isSingleDollarDelimiter = (text, index) => text[index] === '$' && text[index - 1] !== '$' && text[index + 1] !== '$' && !isEscapedAt(text, index);
const renderInlineDollarMath = (text, macros) => {
  let output = '', start = 0, index = 0;
  while (index < text.length) {
    if (!isSingleDollarDelimiter(text, index)) { index++; continue; }
    const open = index;
    let close = -1;
    for (let scan = open + 1; scan < text.length; scan++) if (isSingleDollarDelimiter(text, scan)) { close = scan; break; }
    if (close === -1) break;
    const latex = text.slice(open + 1, close), match = text.slice(open, close + 1);
    output += text.slice(start, open) + (latex.includes('<div') || latex.includes('class=') ? match : renderKatexSafely(latex, false, match, macros));
    index = close + 1; start = index;
  }
  return output + text.slice(start);
};
const cleanTikzContent = content => content.trim().replace(/&lt;br\/&gt;/g, '\n').replace(/<br\s*\/?>/g, '\n');
const processTheoremEnvironments = (text) => {
  let result = text;
  for (const env of ['tikzcd', 'tikzpicture', 'pgfpicture']) {
    const inner = `\\\\begin\\{${env}\\}([\\s\\S]*?)\\\\end\\{${env}\\}`;
    for (const source of [`\\\\\\[\\s*${inner}\\s*\\\\\\]`, `\\$\\$\\s*${inner}\\s*\\$\\$`, inner]) {
      result = result.replace(new RegExp(source, 'gi'), (_, content) => `<div class="latex-tikz-placeholder"><div class="tikz-label">[Commutative Diagram - ${env}]</div><pre class="tikz-code">${escapeHtml(cleanTikzContent(content))}</pre></div>`);
    }
  }
  for (const name of ['theorem', 'lemma', 'proposition', 'corollary', 'definition', 'example', 'remark', 'note', 'proof', 'claim', 'conjecture', 'axiom', 'assumption']) {
    result = result.replace(new RegExp(`\\\\begin\\{${name}\\}(?:\\[([^\\]]+)\\])?([\\s\\S]*?)\\\\end\\{${name}\\}`, 'gi'), (_, title, content) => `<div class="latex-${name}"><strong>${name[0].toUpperCase() + name.slice(1)}${title ? ` (${title})` : ''}.</strong> ${content.trim()}${name === 'proof' ? '<span class="qed">∎</span>' : ''}</div>`);
  }
  result = result.replace(/\\begin\{equation\}([\s\S]*?)\\end\{equation\}/gi, (_, content) => `$$${content.trim()}$$`);
  result = result.replace(/\\begin\{equation\*\}([\s\S]*?)\\end\{equation\*\}/gi, (_, content) => `$$${content.trim()}$$`);
  result = result.replace(/\\begin\{align\*?\}([\s\S]*?)\\end\{align\*?\}/gi, (_, content) => `$$\\begin{gathered}${content.replace(/&/g, '')}\\end{gathered}$$`);
  result = result.replace(/\\begin\{gather\*?\}([\s\S]*?)\\end\{gather\*?\}/gi, (_, content) => `$$\\begin{gathered}${content.trim()}\\end{gathered}$$`);
  result = result.replace(/\\begin\{split\}([\s\S]*?)\\end\{split\}/gi, (_, content) => `$$\\begin{aligned}${content.trim()}\\end{aligned}$$`);
  result = result.replace(/\\begin\{multline\*?\}([\s\S]*?)\\end\{multline\*?\}/gi, (_, content) => `$$\\begin{gathered}${content}\\end{gathered}$$`);
  return result;
};
/** Pure, DOM-free conversion. UNSANITIZED: all consumers must sanitize before HTML sinks. */
export const renderLatexToHtml = (text) => {
  if (!text) return '';
  // Definitions survive expressions in this conversion, never another document/job.
  const macros = { ...extendedMacros };
  let result = decodeHtmlEntities(text);
  // Fenced code is literal, including delimiter-looking text and HTML.
  const codeBlocks = [];
  let codeMarker = '⟦LATEXCODE⟧';
  while (result.includes(codeMarker)) codeMarker += 'X';
  const lines = result.split(/(?<=\n)/);
  let code = null;
  result = lines.map(line => {
    const match = line.match(/^ {0,3}(`{3,}|~{3,})(.*?)(?:\r?\n)?$/);
    if (code) {
      code.text += line;
      if (match && match[1][0] === code.char && match[1].length >= code.length && !match[2].trim()) {
        const token = `${codeMarker}${codeBlocks.length}⟦/CODE⟧`;
        codeBlocks.push(code.text); code = null; return token;
      }
      return '';
    }
    if (match) { code = { text: line, char: match[1][0], length: match[1].length }; return ''; }
    return line;
  }).join('');
  if (code) { result += `${codeMarker}${codeBlocks.length}⟦/CODE⟧`; codeBlocks.push(code.text); }
  result = result.replace(/\\\[\s*\\n/g, '\\[').replace(/\\n\s*\\\]/g, '\\]').replace(/\$\$\s*\\n/g, '$$').replace(/\\n\s*\$\$/g, '$$').replace(/\\n(?=[^a-zA-Z])/g, ' ');
  result = result.replace(/\\igl\(/g, '\\bigl(').replace(/\\igr\)/g, '\\bigr)').replace(/\\igl\[/g, '\\bigl[').replace(/\\igr\]/g, '\\bigr]').replace(/\\igl\{/g, '\\bigl\\{').replace(/\\igr\}/g, '\\bigr\\}').replace(/\\igl(?![a-zA-Z])/g, '\\bigl').replace(/\\igr(?![a-zA-Z])/g, '\\bigr').replace(/\\ig\|/g, '\\big|');
  for (const cmd of ['underline', 'overline', 'widehat', 'widetilde', 'mathcal', 'mathbb', 'mathrm', 'mathbf', 'mathit', 'mathsf', 'operatorname']) result = result.replace(new RegExp(`\\\\\\\\${cmd}\\{`, 'g'), `\\${cmd}{`);
  result = result.replace(/\\\s*\nho_/g, '\\rho_').replace(/\\\s*\nho\{/g, '\\rho{').replace(/\\\s*\nho\(/g, '\\rho(').replace(/\\\s*\nho\|/g, '\\rho|').replace(/\\ho_/g, '\\rho_').replace(/\\ho\{/g, '\\rho{').replace(/\\ho\(/g, '\\rho(').replace(/\\ho\|/g, '\\rho|').replace(/\\\to/g, '\\to');
  result = processTheoremEnvironments(autoWrapMath(result));
  for (const [cmd, level] of [['chapter', 1], ['subsubsection', 4], ['subsection', 3], ['section', 2], ['paragraph', 5]]) result = replaceSectionCommand(result, cmd, `<h${level} class="latex-${cmd}">`, `</h${level}>`);
  result = replaceSectionCommand(result, 'cite', '[', ']');
  for (const [cmd, tag, end] of [['textbf', 'strong', 'strong'], ['textit', 'em', 'em'], ['texttt', 'code class="latex-texttt"', 'code'], ['emph', 'em', 'em'], ['underline', 'u', 'u'], ['textsc', 'span class="latex-smallcaps"', 'span'], ['textsf', 'span class="latex-sans"', 'span']]) result = replaceSectionCommand(result, cmd, `<${tag}>`, `</${end}>`);
  result = replaceSectionCommand(result, 'footnote', '<sup class="latex-footnote">[', ']</sup>');
  for (const [env, tag] of [['enumerate', 'ol'], ['itemize', 'ul']]) {
    result = result.replace(new RegExp(`\\\\begin\\{${env}\\}([\\s\\S]*?)\\\\end\\{${env}\\}`, 'g'), (_, content) => {
      let list = content.replace(/\\item\s*/g, '</li><li>').trim();
      if (list.startsWith('</li>')) list = list.substring(5);
      if (!list.endsWith('</li>')) list += '</li>';
      return `<${tag} class="latex-${env}">${list}</${tag}>`;
    });
  }
  result = result.replace(/\\begin\{description\}([\s\S]*?)\\end\{description\}/g, (_, content) => {
    let list = content.replace(/\\item\s*\[([^\]]+)\]\s*/g, '</dd><dt>$1</dt><dd>').trim();
    if (list.startsWith('</dd>')) list = list.substring(5);
    if (!list.endsWith('</dd>')) list += '</dd>';
    return `<dl class="latex-description">${list}</dl>`;
  });
  result = result.replace(/\\begin\{(?:table|tabular)\}(?:\[[^\]]*\])?(?:\{[^}]*\})?([\s\S]*?)\\end\{(?:table|tabular)\}/gi, (_, content) => `<table class="latex-table">${content.split('\\\\').filter(row => row.trim()).map(row => `<tr>${row.split('&').map(cell => `<td>${cell.replace(/\\hline/g, '').trim()}</td>`).join('')}</tr>`).join('')}</table>`);
  result = result.replace(/\\item\s+/g, '• ').replace(/\\qed/g, '<span class="qed">∎</span>').replace(/\\blacksquare/g, '<span class="qed">■</span>').replace(/\\square/g, '<span class="qed">□</span>');
  for (const pattern of [/\$\$([\s\S]*?)\$\$/g, /\\\[([\s\S]*?)\\\]/g]) result = result.replace(pattern, (match, latex) => latex.includes('<div') || latex.includes('class=') ? match : renderKatexSafely(latex, true, match, macros));
  result = renderInlineDollarMath(result, macros);
  result = result.replace(/\\\(([\s\S]*?)\\\)/g, (match, latex) => latex.includes('<div') || latex.includes('class=') ? match : renderKatexSafely(latex, false, match, macros));
  result = result.replace(/\\\\(?![^<]*>)/g, '<br/>').replace(/\\newline(?![^<]*>)/g, '<br/>').replace(/\\linebreak(?![^<]*>)/g, '<br/>').replace(/\\hrule/g, '<hr class="latex-hrule"/>').replace(/\\rule\{[^}]*\}\{[^}]*\}/g, '<hr class="latex-hrule"/>');
  const parts = [];
  let inTag = false, start = 0;
  for (let i = 0; i < result.length; i++) {
    if (result[i] === '<') inTag = true;
    else if (result[i] === '>') inTag = false;
    else if (result[i] === '\n' && !inTag) { if (i > start) parts.push(result.slice(start, i)); parts.push('<br/>'); start = i + 1; }
  }
  if (start < result.length) parts.push(result.slice(start));
  result = parts.join('').replace(/(<br\s*\/?>\s*){3,}/g, '<br/><br/>');
  codeBlocks.forEach((codeText, index) => { result = result.replace(`${codeMarker}${index}⟦/CODE⟧`, `<pre class="latex-code-block"><code>${escapeHtml(codeText)}</code></pre>`); });
  return result;
};
