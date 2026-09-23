import { describe, it, expect } from 'vitest';
import { chunkLatex } from './latexChunking';
describe('source-preserving atomic scanner', () => {
  it.each(['newcommand', 'renewcommand', 'providecommand', 'def', 'gdef', 'edef', 'xdef', 'global', 'let', 'futurelet', 'globaldefs'])('keeps macro-bearing %s documents atomic', command => {
    for (const prefix of ['\\', '&#92;']) {
      const source = 'prose '.repeat(1000) + `$${prefix}${command}\\foo{z}$` + ' more prose'.repeat(1000);
      expect(chunkLatex(source)).toEqual([source]);
    }
  });
  it('retains oversized macro documents for explicit service fallback', () => {
    const source = '$\\gdef\\foo{z}$' + 'prose '.repeat(12000);
    expect(source.length).toBeGreaterThan(64000);
    expect(chunkLatex(source)).toEqual([source]);
  });
  it('bounds ordinary single-paragraph prose at safe whitespace', () => {
    const source = 'ordinary prose '.repeat(10000);
    const chunks = chunkLatex(source);
    expect(chunks.join('')).toBe(source);
    expect(Math.max(...chunks.map(chunk => chunk.length))).toBeLessThan(3300);
    expect(chunks.length).toBeGreaterThan(10);
  });
  it('prefers a nearby newline over ordinary whitespace', () => {
    const source = 'word '.repeat(21) + '\n' + 'word '.repeat(30);
    expect(chunkLatex(source, 100)[0]).toBe('word '.repeat(21) + '\n');
  });
  it.each(['`code with $ and { and spaces`', '``code with ` nested and spaces``'])('protects inline code %s', code => {
    const source = 'prose '.repeat(30) + code + ' suffix '.repeat(30);
    const chunks = chunkLatex(source, 20);
    expect(chunks.join('')).toBe(source);
    expect(chunks.some(chunk => chunk.includes(code))).toBe(true);
  });
  it.each(['$a\n\nb$', '$$a\n\nb$$', '\\[a\n\nb\\]', '\\(a\n\nb\\)', '\\begin{theorem}a\n\n\\begin{proof}b\n\nc\\end{proof}\\end{theorem}', '\\textbf{a\n\nb}', '```js\na\n\nb\n```', '~~~\na\n\nb\n~~~'])('never splits atomic %s', atomic => {
    const source = 'prefix\n\n' + atomic + '\n\nsuffix';
    const chunks = chunkLatex(source, 4);
    expect(chunks.join('')).toBe(source);
    expect(chunks.some(chunk => chunk.includes(atomic))).toBe(true);
  });
  it.each([
    '$' + 'x + '.repeat(2000) + 'x$',
    '\\begin{theorem}' + 'prose '.repeat(2000) + '\\end{theorem}',
    '\\textbf{' + 'prose '.repeat(2000) + '}',
    '```\n' + 'code '.repeat(2000) + '\n```',
  ])('keeps long protected structures intact despite whitespace fallback', atomic => {
    const source = 'prefix '.repeat(500) + atomic + ' suffix'.repeat(500);
    const chunks = chunkLatex(source);
    expect(chunks.join('')).toBe(source);
    expect(chunks.some(chunk => chunk.includes(atomic))).toBe(true);
  });
  it('preserves escaped dollar/brace syntax and all separators', () => {
    const source = 'cost \\$5 \\{ literal\n\nnext\n\\section{Heading}\ntext';
    expect(chunkLatex(source, 5).join('')).toBe(source);
    expect(chunkLatex(source, 5).length).toBeGreaterThan(1);
  });
  it('keeps oversized or unterminated atomic content intact', () => {
    for (const source of ['x'.repeat(100000), '$$' + 'a\n\n'.repeat(30000), '\\begin{proof}' + 'x\n\n'.repeat(30000)]) expect(chunkLatex(source)).toEqual([source]);
  });
  it('does not interpret math markers inside fenced code', () => {
    const code = '```\n$ { \\begin{proof}\n\n```\n';
    const source = code + '\n' + 'after\n\n'.repeat(20);
    expect(chunkLatex(source, 10).join('')).toBe(source);
    expect(chunkLatex(source, 10).length).toBeGreaterThan(1);
  });
});
