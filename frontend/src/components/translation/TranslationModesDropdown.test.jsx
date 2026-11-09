import React, { useState } from 'react';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, test, vi } from 'vitest';
import TranslationModesDropdown, {
  DEFAULT_TRANSLATION_MODES,
  TRANSLATION_MODES,
  translationModesPayload,
  translationModesSummary,
  visibleTranslationModes,
} from './TranslationModesDropdown';
import { expectNoAxeViolations } from '../../test/axe';

function Harness({ initialModes = DEFAULT_TRANSLATION_MODES, onChange = () => {}, ...props }) {
  const [modes, setModes] = useState(initialModes);
  return (
    <div>
      <p>Outside the list</p>
      <TranslationModesDropdown
        modes={modes}
        onChange={(next) => {
          onChange(next);
          setModes(next);
        }}
        {...props}
      />
    </div>
  );
}

function trigger() {
  return screen.getByRole('button', { name: /^Current runs:/ });
}

function panel() {
  return document.getElementById(trigger().getAttribute('aria-controls'));
}

afterEach(() => {
  cleanup();
});

describe('mode helpers', () => {
  test('LeanOJ is offered only in developer mode, and the default is every standard mode', () => {
    expect(TRANSLATION_MODES.map((mode) => mode.value)).toEqual(['autonomous', 'aggregator', 'compiler', 'leanoj']);
    expect(visibleTranslationModes(false).map((mode) => mode.label))
      .toEqual(['Autonomous Research', 'Aggregator', 'Single Paper Writer']);
    expect(visibleTranslationModes(true).map((mode) => mode.label))
      .toEqual(['Autonomous Research', 'Aggregator', 'Single Paper Writer', 'LeanOJ Proof Solver']);
    expect(DEFAULT_TRANSLATION_MODES).toEqual(['autonomous', 'aggregator', 'compiler']);
  });

  test('the payload is canonical, de-duplicated, and limited to the modes the user can see', () => {
    expect(translationModesPayload(['compiler', 'leanoj', 'autonomous', 'compiler', 'unknown'], false))
      .toEqual(['autonomous', 'compiler']);
    expect(translationModesPayload(['leanoj', 'aggregator'], true)).toEqual(['aggregator', 'leanoj']);
    expect(translationModesPayload(undefined, true)).toEqual([]);
  });

  test('the summary names all, none, or the checked modes', () => {
    expect(translationModesSummary(DEFAULT_TRANSLATION_MODES, false)).toBe('All modes');
    expect(translationModesSummary(DEFAULT_TRANSLATION_MODES, true)).toBe('Autonomous, Aggregator, Single Paper Writer');
    expect(translationModesSummary([...DEFAULT_TRANSLATION_MODES, 'leanoj'], true)).toBe('All modes');
    expect(translationModesSummary(['leanoj'], false)).toBe('None — only items you open');
    expect(translationModesSummary(['aggregator', 'autonomous'], false)).toBe('Autonomous, Aggregator');
    expect(translationModesSummary(['leanoj'], true)).toBe('LeanOJ');
  });
});

describe('TranslationModesDropdown', () => {
  test('is a disclosure whose inline list stays hidden until opened', async () => {
    const user = userEvent.setup();
    const { container } = render(<Harness />);
    expect(trigger()).toHaveAccessibleName('Current runs: All modes');
    expect(trigger()).toHaveAttribute('aria-expanded', 'false');
    expect(container).toContainElement(panel());
    expect(panel()).toHaveAttribute('hidden');
    expect(screen.queryByRole('checkbox')).not.toBeInTheDocument();

    await user.click(trigger());
    expect(trigger()).toHaveAttribute('aria-expanded', 'true');
    expect(panel()).not.toHaveAttribute('hidden');
    expect(screen.getByRole('group', { name: 'Translate the current run of' })).toBe(panel());
    await user.click(trigger());
    expect(panel()).toHaveAttribute('hidden');
  });

  test('reports every change in canonical order and keeps modes the user cannot see', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Harness initialModes={['leanoj', 'compiler']} onChange={onChange} />);
    await user.click(trigger());
    await user.click(screen.getByRole('checkbox', { name: 'Autonomous Research' }));
    expect(onChange).toHaveBeenLastCalledWith(['autonomous', 'compiler', 'leanoj']);
    expect(trigger()).toHaveAccessibleName('Current runs: Autonomous, Single Paper Writer');
    await user.click(screen.getByRole('checkbox', { name: 'Single Paper Writer' }));
    expect(onChange).toHaveBeenLastCalledWith(['autonomous', 'leanoj']);
  });

  test('Escape closes the open list without reaching document listeners; closed, it passes through', async () => {
    const user = userEvent.setup();
    const documentKeyDown = vi.fn();
    document.addEventListener('keydown', documentKeyDown);
    try {
      render(<Harness />);
      await user.click(trigger());
      screen.getByRole('checkbox', { name: 'Aggregator' }).focus();
      await user.keyboard('{Escape}');
      expect(panel()).toHaveAttribute('hidden');
      expect(trigger()).toHaveFocus();
      expect(documentKeyDown).not.toHaveBeenCalled();

      await user.keyboard('{Escape}');
      expect(documentKeyDown).toHaveBeenCalledTimes(1);
      expect(documentKeyDown.mock.calls[0][0].key).toBe('Escape');
    } finally {
      document.removeEventListener('keydown', documentKeyDown);
    }
  });

  test('closes on a mousedown outside it, but not on one inside it', async () => {
    const user = userEvent.setup();
    render(<Harness help={<button type="button">Help</button>} />);
    await user.click(trigger());
    fireEvent.mouseDown(screen.getByRole('button', { name: 'Help' }));
    fireEvent.mouseDown(screen.getByText('The Live Paper and the proofs found in it.'));
    expect(panel()).not.toHaveAttribute('hidden');
    fireEvent.mouseDown(screen.getByText('Outside the list'));
    expect(panel()).toHaveAttribute('hidden');
  });

  test('clicking a description toggles its mode, and the checkbox is named by its label only', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Harness onChange={onChange} />);
    await user.click(trigger());
    const aggregator = screen.getByRole('checkbox', { name: 'Aggregator' });
    expect(aggregator).toHaveAccessibleDescription(
      'Live Results and proofs found in the Aggregator database (including Single Paper Writer proofs-only runs).',
    );
    await user.click(screen.getByText(/^Live Results and proofs found/));
    expect(onChange).toHaveBeenLastCalledWith(['autonomous', 'compiler']);
    expect(aggregator).not.toBeChecked();
  });

  test('groups LeanOJ separately in developer mode', async () => {
    const user = userEvent.setup();
    render(<Harness developerModeEnabled />);
    await user.click(trigger());
    expect(within(panel()).getAllByRole('group')).toHaveLength(3);
    expect(within(screen.getByRole('group', { name: 'LeanOJ' })).getByRole('checkbox', { name: 'LeanOJ Proof Solver' }))
      .not.toBeChecked();
    expect(within(screen.getByRole('group', { name: 'Advanced Manual S.T.E.M. ASI' })).getAllByRole('checkbox'))
      .toHaveLength(2);
  });

  test('passes its description to the trigger and can be disabled', () => {
    render(
      <>
        <span id="modes-help">Which modes are translated.</span>
        <Harness describedBy="modes-help" disabled />
      </>,
    );
    expect(trigger()).toHaveAccessibleDescription('Which modes are translated.');
    expect(trigger()).toBeDisabled();
  });

  test('has no axe violations while open', async () => {
    const user = userEvent.setup();
    const { container } = render(<Harness developerModeEnabled />);
    await user.click(trigger());
    await expectNoAxeViolations(container);
  });
});
