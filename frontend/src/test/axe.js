import axe from 'axe-core';
import { expect } from 'vitest';

// jsdom has no layout engine, so colour contrast cannot be computed here; it is
// reviewed manually. Every other WCAG A/AA rule axe can evaluate is enforced.
const AXE_OPTIONS = {
  runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'best-practice'] },
  rules: {
    'color-contrast': { enabled: false },
    // Isolated component renders have no page-level landmarks or h1.
    region: { enabled: false },
    'page-has-heading-one': { enabled: false },
    'landmark-one-main': { enabled: false },
  },
};

export async function expectNoAxeViolations(container) {
  const results = await axe.run(container, AXE_OPTIONS);
  const summary = results.violations.map((violation) => ({
    id: violation.id,
    impact: violation.impact,
    nodes: violation.nodes.map((node) => node.target.join(' ')),
  }));
  expect(summary).toEqual([]);
}
