import React, { useState } from 'react';
import { render, screen, fireEvent, within } from '@testing-library/react';
import { ProofCompetitionSettings } from './AutonomousResearchSettings';

function Harness({ model = 'primary', developerModeEnabled = false, isRunning = false, lmStudioEnabled = true }) {
  const [competition, setCompetition] = useState({ enabled: false, secondaries: [] });
  return <><ProofCompetitionSettings
    localConfig={{ high_param_provider: 'lm_studio', high_param_model: model,
      high_param_context_window: 64000, high_param_max_tokens: 8000, proof_competition: competition }}
    onChange={setCompetition} lmStudioModels={[{ id: 'primary' }, { id: 'changed' }]}
    openRouterModels={[]} openAICodexModels={[]} xaiGrokModels={[]} sakanaFuguModels={[]}
    modelProviders={{}} lmStudioEnabled={lmStudioEnabled} isRunning={isRunning} developerModeEnabled={developerModeEnabled}
  /><output data-testid="state">{JSON.stringify(competition)}</output></>;
}

test('Codex competition primary is locked and secondary effort uses its own catalog', () => {
  const onChange = vi.fn();
  const role = { provider: 'openai_codex_oauth', model_id: 'codex', openrouter_reasoning_effort: 'max', context_window: 400000, max_tokens: 32000 };
  render(<ProofCompetitionSettings localConfig={{ high_param_provider: role.provider, high_param_model: role.model_id, high_param_openrouter_reasoning_effort: 'max', proof_competition: { enabled: true, secondaries: [role] } }} onChange={onChange} lmStudioModels={[]} openRouterModels={[]} openAICodexModels={[{ id: 'codex', supported_reasoning_levels: [{ effort: 'medium' }, { effort: 'max' }] }]} xaiGrokModels={[]} sakanaFuguModels={[]} modelProviders={{}} lmStudioEnabled hasOpenAICodexLogin />);
  const controls = screen.getAllByRole('combobox', { name: 'Codex Reasoning Effort' });
  expect(controls[0]).toBeDisabled();
  expect(controls[1]).not.toBeDisabled();
  expect(controls[1]).toHaveValue('max');
  fireEvent.change(controls[1], { target: { value: 'medium' } });
  expect(onChange).toHaveBeenCalled();
});

test('opt-in reveals locked derived primary and ordered add/remove controls', () => {
  const view = render(<Harness />);
  expect(screen.queryByText('Secondary proof model 1')).toBeNull();
  fireEvent.click(screen.getByLabelText('Duplicate role to compete against other AI(s)'));
  const primary = screen.getByText('Primary proof model (mirrors Rigor & Proofs)').closest('.submitter-config-section');
  expect(within(primary).getAllByRole('spinbutton').every(input => input.disabled)).toBe(true);
  expect(screen.getByText('Secondary proof model 1')).toBeTruthy();
  fireEvent.click(screen.getByText('+ Add secondary proof model'));
  expect(screen.getByText('Secondary proof model 2')).toBeTruthy();
  fireEvent.click(screen.getByLabelText('Remove secondary proof model 1'));
  expect(screen.queryByText('Secondary proof model 2')).toBeNull();
  view.rerender(<Harness model="changed" />);
  expect(within(primary).getByRole('combobox').value).toBe('changed');
  expect(screen.getByTestId('state').textContent).not.toContain('"primary"');
});

test('competition help explains fallback and hosted controls omit local providers', () => {
  render(<Harness lmStudioEnabled={false} />);
  fireEvent.click(screen.getByLabelText('Duplicate role to compete against other AI(s)'));
  expect(screen.queryByText('LM Studio')).toBeNull();
  fireEvent.focus(screen.getByLabelText('Learn about proof model competition'));
  expect(screen.getByText(/Each model gets up to five proof attempts/).textContent).toMatch(/private failure feedback/);
});

test('running locks competition and Supercharge remains developer-only', () => {
  const view = render(<Harness />);
  fireEvent.click(screen.getByLabelText('Duplicate role to compete against other AI(s)'));
  expect(screen.queryByText('Supercharge')).toBeNull();
  view.rerender(<Harness developerModeEnabled isRunning />);
  expect(screen.getAllByText('Supercharge')).toHaveLength(2);
  expect(screen.getByLabelText('Duplicate role to compete against other AI(s)').disabled).toBe(true);
  expect(screen.getByText('+ Add secondary proof model').disabled).toBe(true);
});
