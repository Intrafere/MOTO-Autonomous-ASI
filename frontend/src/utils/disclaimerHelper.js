/**
 * Frontend-only disclaimer injection for brainstorms and in-progress papers.
 *
 * Disclaimers are prepended at the display/download layer so models never see
 * them in their context window. Completed papers already have a richer
 * backend-embedded disclaimer (AUTONOMOUS AI SOLUTION) — those are detected
 * and left unchanged.
 */

const SEPARATOR = '='.repeat(80);

export const PAPER_DISCLAIMER =
  `${SEPARATOR}\n` +
  'DISCLAIMER\n' +
  '\n' +
  'This content is provided for informational and experimental purposes only.\n' +
  'This paper was autonomously generated with the novelty-seeking MOTO harness\n' +
  'without peer review or user oversight beyond the original prompt. It may\n' +
  'contain incorrect, incomplete, misleading, or fabricated claims presented\n' +
  'with high confidence. Use of this content is at your own risk. You are\n' +
  'solely responsible for reviewing and independently verifying any output\n' +
  'before relying on it, and the developers, operators, and contributors are\n' +
  'not responsible for errors, omissions, decisions made from this content, or\n' +
  'any resulting loss, damage, cost, or liability.\n' +
  `${SEPARATOR}`;

export const BRAINSTORM_DISCLAIMER =
  `${SEPARATOR}\n` +
  'DISCLAIMER\n' +
  '\n' +
  'This content is provided for informational and experimental purposes only.\n' +
  'This brainstorm database was autonomously generated with the novelty-seeking\n' +
  'MOTO harness without peer review or user oversight beyond the original\n' +
  'prompt. It may contain incorrect, incomplete, misleading, or\n' +
  'fabricated claims presented with high confidence. Use of this content is at\n' +
  'your own risk. You are solely responsible for reviewing and independently\n' +
  'verifying any output before relying on it, and the developers, operators,\n' +
  'and contributors are not responsible for errors, omissions, decisions made\n' +
  'from this content, or any resulting loss, damage, cost, or liability.\n' +
  `${SEPARATOR}`;

/**
 * Returns true when the content already carries a disclaimer header
 * (either the backend-embedded AUTONOMOUS AI SOLUTION or the frontend
 * DISCLAIMER block).
 */
export function hasDisclaimer(content) {
  if (!content) return false;
  const head = content.slice(0, 200);
  return (
    head.includes('AUTONOMOUS AI SOLUTION') ||
    head.includes('DISCLAIMER')
  );
}

/**
 * Prepend the appropriate disclaimer to content unless one is already present.
 *
 * @param {string} content  Raw content string
 * @param {'paper'|'brainstorm'} type  Which disclaimer variant to use
 * @returns {string} Content with disclaimer prepended (or unchanged)
 */
export function prependDisclaimer(content, type) {
  if (!content || hasDisclaimer(content)) return content;
  const disclaimer = type === 'brainstorm' ? BRAINSTORM_DISCLAIMER : PAPER_DISCLAIMER;
  return `${disclaimer}\n\n${content}`;
}
