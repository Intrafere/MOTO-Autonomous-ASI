import React from 'react';
import { render, screen } from '@testing-library/react';
import { expect, test } from 'vitest';
import UsageLevelBadge from './UsageLevelBadge';
import AllowedOutputOption from './AllowedOutputOption';

test('renders filled bars matching the level rank plus a visible text label', () => {
  const { container } = render(<UsageLevelBadge featureId="allowed_outputs.mathematical_proofs" />);
  const badge = container.querySelector('.usage-level-badge');
  expect(badge).toHaveAttribute('data-usage-level', 'very_high');
  expect(container.querySelectorAll('.usage-level-bar')).toHaveLength(4);
  expect(container.querySelectorAll('.usage-level-bar.is-filled')).toHaveLength(4);
  expect(badge).toHaveTextContent('Very high');
});

test('supports an ad-hoc level and renders nothing for unknown input', () => {
  const { container, rerender } = render(<UsageLevelBadge level="medium" reason="Example." />);
  expect(container.querySelectorAll('.usage-level-bar.is-filled')).toHaveLength(2);
  expect(container.querySelector('.usage-level-badge')).toHaveAttribute('title', 'Medium usage. Example.');
  rerender(<UsageLevelBadge featureId="not.registered" />);
  expect(container.querySelector('.usage-level-badge')).toBeNull();
  rerender(<UsageLevelBadge level="extreme" />);
  expect(container.querySelector('.usage-level-badge')).toBeNull();
});

test('allowed output checkbox keeps its name and describes its usage level', () => {
  render(
    <AllowedOutputOption usageFeatureId="allowed_outputs.research_papers" checked onChange={() => {}}>
      Research Papers
    </AllowedOutputOption>,
  );
  const checkbox = screen.getByRole('checkbox', { name: 'Research Papers' });
  expect(checkbox).toHaveAccessibleDescription(/^Medium usage\./);
});
