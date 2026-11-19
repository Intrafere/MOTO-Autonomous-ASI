import React from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import AllowedOutputsRow from './AllowedOutputsRow';
import { PROOF_SETTINGS_CHANGED_EVENT } from '../utils/proofRuntimeSettings';

const baseStatus = {
  lean4_enabled: true,
  lean4_user_enabled: true,
  lean4_proof_timeout: 2000,
  lean4_lsp_enabled: false,
  lean4_lsp_idle_timeout: 600,
  proof_max_parallel_candidates: 6,
  proof_single_shot_enabled: true,
  proof_master_proofs_enabled: true,
  smt_enabled: false,
  smt_timeout: 300,
};

function renderRow({ proofsChecked = true, optionsLocked = false, status = baseStatus } = {}) {
  let current = { ...status };
  const getProofStatus = vi.fn(async () => current);
  const updateProofSettings = vi.fn(async (payload) => {
    current = {
      ...current,
      proof_max_parallel_candidates: payload.max_parallel_candidates,
      proof_single_shot_enabled: payload.single_shot_proofs_enabled ?? current.proof_single_shot_enabled,
      proof_master_proofs_enabled: payload.master_proofs_enabled ?? current.proof_master_proofs_enabled,
    };
    return current;
  });
  render(
    <AllowedOutputsRow
      rowTitle="Allowed Outputs"
      proofOutputsAvailable
      optionsLocked={optionsLocked}
      getProofStatus={getProofStatus}
      updateProofSettings={updateProofSettings}
      proofs={{ title: 'proofs', checked: proofsChecked, onChange: () => {}, disabled: false }}
      papers={{ title: 'papers', checked: true, onChange: () => {}, disabled: false }}
    />,
  );
  return { getProofStatus, updateProofSettings };
}

async function openPanel() {
  const toggle = screen.getByRole('button', { name: 'Proof options' });
  expect(toggle).toHaveAttribute('aria-expanded', 'false');
  fireEvent.click(toggle);
  expect(toggle).toHaveAttribute('aria-expanded', 'true');
  const panel = screen.getByRole('region', { name: 'Proof options' });
  await waitFor(() => expect(within(panel).getByLabelText('Concurrent proof attempts')).toHaveValue('6'));
  return panel;
}

const usageOf = (element) => document.getElementById(element.getAttribute('aria-describedby').split(' ').pop());

afterEach(() => {
  vi.restoreAllMocks();
});

test('proof options stay collapsed until the arrow is pressed and load lazily', async () => {
  const { getProofStatus } = renderRow();
  expect(screen.queryByRole('region', { name: 'Proof options' })).toBeNull();
  expect(getProofStatus).not.toHaveBeenCalled();

  const panel = await openPanel();
  const singleShot = within(panel).getByRole('checkbox', { name: 'Single-shot proofs' });
  const master = within(panel).getByRole('checkbox', { name: 'Master proofs' });
  expect(singleShot).toBeChecked();
  expect(master).toBeChecked();
  expect(usageOf(singleShot)).toHaveAttribute('data-usage-level', 'high');
  expect(usageOf(master)).toHaveAttribute('data-usage-level', 'very_high');
});

test('turning a proof type off saves only that change and keeps the rest of the settings', async () => {
  const { updateProofSettings } = renderRow();
  const panel = await openPanel();

  fireEvent.click(within(panel).getByRole('checkbox', { name: 'Master proofs' }));

  await waitFor(() => expect(updateProofSettings).toHaveBeenCalledTimes(1));
  expect(updateProofSettings).toHaveBeenCalledWith(expect.objectContaining({
    enabled: true,
    timeout: 2000,
    max_parallel_candidates: 6,
    master_proofs_enabled: false,
  }));
  expect(updateProofSettings.mock.calls[0][0]).not.toHaveProperty('single_shot_proofs_enabled');
  await waitFor(() => expect(within(panel).getByRole('checkbox', { name: 'Master proofs' })).not.toBeChecked());
});

test('the last proof type cannot be turned off', async () => {
  const { updateProofSettings } = renderRow({
    status: { ...baseStatus, proof_master_proofs_enabled: false },
  });
  const panel = await openPanel();

  fireEvent.click(within(panel).getByRole('checkbox', { name: 'Single-shot proofs' }));

  expect(within(panel).getByRole('alert')).toHaveTextContent('Keep at least one proof type on.');
  expect(updateProofSettings).not.toHaveBeenCalled();
});

test('concurrent proof attempts shows its usage level and saves on Enter and on the stepper', async () => {
  const { updateProofSettings } = renderRow();
  const panel = await openPanel();
  const input = within(panel).getByLabelText('Concurrent proof attempts');
  expect(usageOf(input)).toHaveAttribute('data-usage-level', 'high');

  fireEvent.change(input, { target: { value: '3' } });
  expect(usageOf(input)).toHaveAttribute('data-usage-level', 'medium');
  fireEvent.change(input, { target: { value: '1' } });
  expect(usageOf(input)).toHaveAttribute('data-usage-level', 'low');
  fireEvent.change(input, { target: { value: '7' } });
  expect(usageOf(input)).toHaveAttribute('data-usage-level', 'very_high');

  fireEvent.keyDown(input, { key: 'Enter' });
  await waitFor(() => expect(updateProofSettings).toHaveBeenLastCalledWith(
    expect.objectContaining({ max_parallel_candidates: 7 }),
  ));

  fireEvent.click(within(panel).getByRole('button', { name: 'Fewer concurrent proof attempts' }));
  await waitFor(() => expect(updateProofSettings).toHaveBeenLastCalledWith(
    expect.objectContaining({ max_parallel_candidates: 6 }),
  ));
});

test('an invalid count is rejected without saving', async () => {
  const { updateProofSettings } = renderRow();
  const panel = await openPanel();
  const input = within(panel).getByLabelText('Concurrent proof attempts');

  fireEvent.change(input, { target: { value: 'abc' } });
  fireEvent.blur(input);

  expect(within(panel).getByRole('alert')).toHaveTextContent('Enter a whole number from 0 to 1000.');
  expect(input).toHaveValue('6');
  expect(updateProofSettings).not.toHaveBeenCalled();
});

test('options are disabled while Mathematical Proofs is off or the run is locked', async () => {
  renderRow({ proofsChecked: false });
  const panel = await openPanel();
  expect(within(panel).getByRole('checkbox', { name: 'Master proofs' })).toBeDisabled();
  expect(within(panel).getByLabelText('Concurrent proof attempts')).toBeDisabled();
  expect(panel).toHaveTextContent('Turn on Mathematical Proofs to change these options.');
});

test('saves from other surfaces update the open panel, and Escape closes it', async () => {
  renderRow();
  const panel = await openPanel();

  window.dispatchEvent(new CustomEvent(PROOF_SETTINGS_CHANGED_EVENT, {
    detail: { ...baseStatus, proof_max_parallel_candidates: 2 },
  }));
  await waitFor(() => expect(within(panel).getByLabelText('Concurrent proof attempts')).toHaveValue('2'));

  fireEvent.keyDown(panel, { key: 'Escape' });
  expect(screen.queryByRole('region', { name: 'Proof options' })).toBeNull();
  expect(screen.getByRole('button', { name: 'Proof options' })).toHaveFocus();
});

test('typing a count and then pressing + saves once, from the typed value', async () => {
  const { updateProofSettings } = renderRow();
  const panel = await openPanel();
  const input = within(panel).getByLabelText('Concurrent proof attempts');
  const plus = within(panel).getByRole('button', { name: 'More concurrent proof attempts' });

  fireEvent.change(input, { target: { value: '3' } });
  fireEvent.blur(input, { relatedTarget: plus });
  fireEvent.click(plus);

  await waitFor(() => expect(updateProofSettings).toHaveBeenCalledTimes(1));
  expect(updateProofSettings).toHaveBeenCalledWith(expect.objectContaining({ max_parallel_candidates: 4 }));
  await waitFor(() => expect(input).toHaveValue('4'));
});

test('Enter followed by blur sends a single save', async () => {
  const { updateProofSettings } = renderRow();
  const panel = await openPanel();
  const input = within(panel).getByLabelText('Concurrent proof attempts');

  fireEvent.change(input, { target: { value: '2' } });
  fireEvent.keyDown(input, { key: 'Enter' });
  fireEvent.blur(input);

  await waitFor(() => expect(input).toHaveValue('2'));
  expect(updateProofSettings).toHaveBeenCalledTimes(1);
});

test('a failed save shows the error and the value still in effect', async () => {
  const { updateProofSettings } = renderRow();
  updateProofSettings.mockRejectedValueOnce(new Error('backend down'));
  const panel = await openPanel();
  const input = within(panel).getByLabelText('Concurrent proof attempts');

  fireEvent.change(input, { target: { value: '9' } });
  fireEvent.keyDown(input, { key: 'Enter' });

  await waitFor(() => expect(within(panel).getByRole('alert')).toHaveTextContent('backend down'));
  expect(input).toHaveValue('6');
});

test('pressing the arrow again closes the card', async () => {
  renderRow();
  await openPanel();
  const toggle = screen.getByRole('button', { name: 'Proof options' });
  fireEvent.click(toggle);
  expect(toggle).toHaveAttribute('aria-expanded', 'false');
  expect(screen.queryByRole('region', { name: 'Proof options' })).toBeNull();
});

// Lays the composer row out as the browser would: the pill is 560px wide with a
// 170px Clear button after it, the arrow's centre is 310px into the row, and
// the card is 460px wide.
function mockComposerLayout(rowWidth) {
  const widths = { 'allowed-outputs-row': 560, 'btn-clear': 170, 'allowed-output-panel': 460 };
  const widthOf = (node) => Object.entries(widths)
    .find(([name]) => node.classList?.contains(name))?.[1] ?? 0;
  vi.spyOn(HTMLElement.prototype, 'offsetWidth', 'get').mockImplementation(function offsetWidth() {
    return widthOf(this);
  });
  vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockImplementation(function clientWidth() {
    return this.classList.contains('prompt-composer-options') ? rowWidth : 0;
  });
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function rect() {
    if (this.classList.contains('allowed-output-panel-slot')) return { left: 100, width: rowWidth };
    if (this.classList.contains('allowed-output-expand')) return { left: 400, width: 20 };
    return { left: 0, width: 0 };
  });
}

function renderInComposer() {
  const status = { ...baseStatus };
  render(
    <div className="prompt-composer-options">
      <AllowedOutputsRow
        rowTitle="Allowed Outputs"
        proofOutputsAvailable
        getProofStatus={vi.fn(async () => status)}
        updateProofSettings={vi.fn(async () => status)}
        proofs={{ title: 'proofs', checked: true, onChange: () => {}, disabled: false }}
        papers={{ title: 'papers', checked: true, onChange: () => {}, disabled: false }}
      />
      <button type="button" className="btn-clear">Clear Research Run</button>
    </div>,
  );
}

test('the card is centred under its arrow, points at it, and stays below the buttons on their line', async () => {
  mockComposerLayout(1000);
  renderInComposer();
  const panel = await openPanel();

  await waitFor(() => expect(panel).toHaveClass('is-placed'));
  expect(panel.style.marginLeft).toBe('80px');
  expect(panel.style.getPropertyValue('--allowed-output-caret')).toBe('230px');
  expect(panel.parentElement.style.order).toBe('');
});

test('when the buttons have wrapped, the card goes right under the pill and stays inside the row', async () => {
  mockComposerLayout(500);
  renderInComposer();
  const panel = await openPanel();

  await waitFor(() => expect(panel).toHaveClass('is-placed'));
  expect(panel.parentElement.style.order).toBe('0');
  expect(panel.style.marginLeft).toBe('40px');
  expect(panel.style.getPropertyValue('--allowed-output-caret')).toBe('270px');
});
