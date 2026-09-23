import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import CodexReasoningControl from './CodexReasoningControl';
import { getCodexReasoningInfo } from '../utils/codexReasoning';
import { normalizeOpenRouterReasoningEffort } from '../utils/openRouterSelection';

test('catalog strings and objects select maximum in effort order', () => {
  expect(getCodexReasoningInfo({ supported_reasoning_levels: ['max', { effort: 'high' }, 'medium'] })).toEqual({ status: 'supported', levels: ['medium', 'high', 'max'], maximum: 'max' });
  expect(getCodexReasoningInfo({ supported_reasoning_levels: [] }).status).toBe('unsupported');
  expect(getCodexReasoningInfo({}).status).toBe('unknown');
  expect(getCodexReasoningInfo({ supported_reasoning_levels: [{}] }).status).toBe('unknown');
  expect(normalizeOpenRouterReasoningEffort('max', 'openai_codex_oauth')).toBe('max');
  expect(normalizeOpenRouterReasoningEffort('max', 'openrouter')).toBe('auto');
});

test('control uses catalog options and preserves Auto maximum semantics', () => {
  const onChange = vi.fn();
  render(<CodexReasoningControl model={{ supported_reasoning_levels: ['medium', 'max'] }} onChange={onChange} />);
  expect(screen.getByRole('option', { name: 'Auto (Maximum: max)' })).toBeInTheDocument();
  expect(screen.queryByRole('option', { name: 'High' })).not.toBeInTheDocument();
  fireEvent.change(screen.getByRole('combobox'), { target: { value: 'max' } });
  expect(onChange).toHaveBeenCalledWith('max');
});

test('fixed Spark High and unavailable catalogs are explicit and disabled', () => {
  const { rerender } = render(<CodexReasoningControl modelId="gpt-5.3-codex-spark-high" value="max" />);
  expect(screen.getByRole('combobox')).toBeDisabled();
  expect(screen.getByRole('combobox')).toHaveValue('high');
  rerender(<CodexReasoningControl model={{}} value="max" />);
  expect(screen.getByText(/capabilities are unknown/)).toBeInTheDocument();
  expect(screen.getByRole('combobox')).toHaveValue('max');
  expect(screen.getByRole('combobox')).toBeDisabled();
  rerender(<CodexReasoningControl model={{ supported_reasoning_levels: [] }} />);
  expect(screen.getByText(/does not support configurable/)).toBeInTheDocument();
});
