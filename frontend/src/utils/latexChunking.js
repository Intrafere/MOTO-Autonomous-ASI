import { decodeHTML } from 'entities';

/** Source-preserving linear scanner. Boundaries are permitted only outside atomic syntax.
 * Macro-bearing documents conservatively remain one atomic conversion job, even when
 * definitions occur in comments/code. This preserves per-call macro state without
 * unordered worker/chunk state leakage. The service's 64K atomic limit supplies its
 * explicit complete-source fallback rather than splitting an oversized macro document.
 */
export const LATEX_CHUNK_TARGET = 3000;
export function chunkLatex(text, target = LATEX_CHUNK_TARGET) {
  if (!text) return [text || ''];
  // Inspect the decoded spelling too: entity-encoded backslashes become commands.
  if (/\\(?:newcommand|renewcommand|providecommand|def|gdef|edef|xdef|global|let|futurelet|globaldefs)(?![A-Za-z])/.test(decodeHTML(text))) return [text];
  const chunks = [];
  let start = 0, braces = 0, math = null, fence = null, pendingWhitespace = -1;
  const environments = [];
  for (let i = 0; i < text.length;) {
    const lineStart = i === 0 || text[i - 1] === '\n';
    if (lineStart) {
      const lineEnd = text.indexOf('\n', i);
      const end = lineEnd < 0 ? text.length : lineEnd;
      const marker = text.slice(i, end).match(/^ {0,3}(`{3,}|~{3,})(.*)$/);
      if (fence) {
        if (marker && marker[1][0] === fence.char && marker[1].length >= fence.length && !marker[2].trim()) fence = null;
        i = end < text.length ? end + 1 : end;
        continue;
      }
      if (marker && !math && !braces && !environments.length) {
        fence = { char: marker[1][0], length: marker[1].length };
        i = end < text.length ? end + 1 : end;
        continue;
      }
    }
    if (text[i] === '`' && !math && !braces && !environments.length) {
      let end = i;
      while (text[end] === '`') end++;
      const marker = text.slice(i, end);
      let close = text.indexOf(marker, end);
      while (close >= 0 && (text[close - 1] === '`' || text[close + marker.length] === '`')) close = text.indexOf(marker, close + marker.length);
      i = close < 0 ? text.length : close + marker.length;
      continue;
    }
    if (text[i] === '\\') {
      const env = text.slice(i, i + 160).match(/^\\(begin|end)\s*\{([^{}\r\n]+)\}/);
      if (env) {
        if (env[1] === 'begin') environments.push(env[2]);
        else if (environments.at(-1) === env[2]) environments.pop();
        // Mismatched ends never release an open atomic environment.
        i += env[0].length; continue;
      }
      const pair = text.slice(i, i + 2);
      if (pair === '\\[' || pair === '\\(') { if (!math) math = pair === '\\[' ? '\\]' : '\\)'; }
      else if (pair === math) math = null;
      i += Math.min(2, text.length - i); continue;
    }
    if (text[i] === '$') {
      const delimiter = text[i + 1] === '$' ? '$$' : '$';
      if (!math) math = delimiter;
      else if (math === delimiter) math = null;
      i += delimiter.length; continue;
    }
    if (text[i] === '{') braces++;
    else if (text[i] === '}') braces = Math.max(0, braces - 1);
    if (!math && !braces && !environments.length && /\s/.test(text[i]) && i + 1 - start >= target) {
      // Prefer a nearby newline, but do not leave a single prose paragraph atomic.
      if (pendingWhitespace < 0) pendingWhitespace = i + 1;
      if (text[i] === '\n' || i + 1 - pendingWhitespace >= Math.min(256, target)) {
        const boundary = text[i] === '\n' ? i + 1 : pendingWhitespace;
        chunks.push(text.slice(start, boundary)); start = boundary;
        pendingWhitespace = -1;
      }
    }
    i++;
  }
  if (start < text.length) chunks.push(text.slice(start));
  return chunks.length ? chunks : [text];
}
