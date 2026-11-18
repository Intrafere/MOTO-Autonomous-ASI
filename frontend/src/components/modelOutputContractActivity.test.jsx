import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, test, vi } from 'vitest';

import { contractFailureActivityNotification, getScopedProviderActivityFeed } from '../App';
import AutonomousResearchLogs from './autonomous/AutonomousResearchLogs';
import { normalizeActivityEvent } from '../utils/activityNormalization';
import {
  MODEL_OUTPUT_CONTRACT_FAILED_EVENT,
  VALIDATOR_OUTPUT_CONTRACT_FAILED_EVENT,
  VALIDATOR_BATCH_INTERRUPTED_EVENT,
  formatModelOutputContractFailedMessage,
  formatValidatorBatchInterruptedMessage,
  formatProofFramingDecidedMessage,
  getActivityClass,
  getActivityIcon,
} from '../utils/activityStyles';

vi.mock('../services/api', async (importOriginal) => {
  const original = await importOriginal();
  return {
    ...original,
    autonomousAPI: {
      ...original.autonomousAPI,
      getProofCompetitionBenchmarks: vi.fn().mockResolvedValue({ reports: [] }),
    },
  };
});

vi.mock('../services/websocket', () => ({
  websocket: {
    on: vi.fn(() => () => {}),
    off: vi.fn(),
  },
}));

vi.mock('./ApiCallLogs', () => ({ default: () => <div>API logs</div> }));
vi.mock('./autonomous/ProofCompetitionBenchmarkReport', () => ({ default: () => <div>Benchmarks</div> }));

const ROUTE = '[Autonomous · Topic Validator · gemma-3 via LM Studio]';

const contractPayload = (overrides = {}) => ({
  activity_id: 'model_output_contract_failed:0001',
  role_id: 'autonomous_topic_validator',
  workflow_mode: 'autonomous',
  route_identity: ROUTE,
  contract_stage: 'missing_key',
  message: `${ROUTE} gemma-3 returned output that did not match the required JSON format after one retry (decision). The same proposal is validated again.`,
  activity_classification: 'operational',
  ...overrides,
});

// Mirrors App.addScopedProviderActivity for a contract-failure row.
const feedRow = (eventType, payload) => normalizeActivityEvent({
  event: eventType,
  type: eventType,
  timestamp: '2026-11-15T12:00:00Z',
  message: formatModelOutputContractFailedMessage(payload),
  data: payload,
});

afterEach(() => cleanup());

describe('model-output contract failure routing', () => {
  test('autonomous contract failures route to the Autonomous feed', () => {
    const notification = contractFailureActivityNotification(
      MODEL_OUTPUT_CONTRACT_FAILED_EVENT,
      contractPayload(),
    );
    expect(notification.event_type).toBe(MODEL_OUTPUT_CONTRACT_FAILED_EVENT);
    expect(getScopedProviderActivityFeed(notification)).toBe('autonomous');
  });

  test('child-aggregator validator requeues route to Autonomous; manual ones use the saved log', () => {
    const child = contractFailureActivityNotification(
      VALIDATOR_OUTPUT_CONTRACT_FAILED_EVENT,
      contractPayload({ role_id: 'aggregator_validator' }),
    );
    expect(getScopedProviderActivityFeed(child)).toBe('autonomous');
    expect(contractFailureActivityNotification(
      VALIDATOR_OUTPUT_CONTRACT_FAILED_EVENT,
      contractPayload({ workflow_mode: 'aggregator' }),
    )).toBeNull();
  });

  test('compiler warnings reach the Autonomous feed only from an Autonomous-owned compiler', () => {
    const warning = {
      workflow_mode: 'autonomous',
      warning: `${ROUTE} The reply did not match its JSON format after one retry (decision); validating the same submission again. This is not a rejection.`,
    };
    const routed = contractFailureActivityNotification('compiler_warning', warning);
    expect(routed.message).toBe(warning.warning);
    expect(getScopedProviderActivityFeed(routed)).toBe('autonomous');
    expect(contractFailureActivityNotification('compiler_warning', { ...warning, workflow_mode: 'compiler' })).toBeNull();
  });

  test('contract failures render as warnings with the route once', () => {
    const message = formatModelOutputContractFailedMessage(contractPayload());
    expect(message.split(ROUTE)).toHaveLength(2);
    expect(getActivityClass(MODEL_OUTPUT_CONTRACT_FAILED_EVENT, {})).toBe('activity-warning');
    expect(getActivityClass(VALIDATOR_OUTPUT_CONTRACT_FAILED_EVENT, {})).toBe('activity-warning');
    expect(getActivityIcon(MODEL_OUTPUT_CONTRACT_FAILED_EVENT)).toBe('⚠');
    expect(formatModelOutputContractFailedMessage({ role_id: 'x' })).toMatch(/no decision was recorded/);
  });
});

describe('validator batch interruption activity', () => {
  const interruption = (overrides = {}) => ({
    activity_id: 'validator_batch_interrupted:0001',
    event_id: 'validator_batch_interrupted:0001',
    role_id: 'aggregator_validator',
    workflow_mode: 'autonomous',
    route_identity: ROUTE,
    error_type: 'KeyError',
    batch_size: 3,
    message: `${ROUTE} Aggregator batch validation stopped on an internal error (KeyError) before any decision. This is not a model output format problem, a provider error, or a rejection.`,
    activity_classification: 'operational',
    activity_category: 'internal',
    ...overrides,
  });

  test('child-aggregator interruptions route to Autonomous; manual ones use the saved log', () => {
    const child = contractFailureActivityNotification(VALIDATOR_BATCH_INTERRUPTED_EVENT, interruption());
    expect(child.event_type).toBe(VALIDATOR_BATCH_INTERRUPTED_EVENT);
    expect(getScopedProviderActivityFeed(child)).toBe('autonomous');
    expect(contractFailureActivityNotification(
      VALIDATOR_BATCH_INTERRUPTED_EVENT,
      interruption({ workflow_mode: 'aggregator' }),
    )).toBeNull();
  });

  test('interruptions render as warnings that are not described as format failures', () => {
    const message = formatValidatorBatchInterruptedMessage(interruption());
    expect(message.split(ROUTE)).toHaveLength(2);
    const fallback = formatValidatorBatchInterruptedMessage({ error_type: 'KeyError' });
    expect(fallback).toMatch(/internal error \(KeyError\)/);
    expect(fallback).not.toMatch(/JSON format/);
    expect(getActivityClass(VALIDATOR_BATCH_INTERRUPTED_EVENT, {})).toBe('activity-warning');
    expect(getActivityIcon(VALIDATOR_BATCH_INTERRUPTED_EVENT)).toBe('⚠');
  });

  test('AutonomousResearchLogs shows an interruption as a warning', () => {
    const payload = interruption();
    render(
      <AutonomousResearchLogs
        stats={{}}
        events={[normalizeActivityEvent({
          event: VALIDATOR_BATCH_INTERRUPTED_EVENT,
          type: VALIDATOR_BATCH_INTERRUPTED_EVENT,
          timestamp: '2026-11-15T12:00:00Z',
          message: formatValidatorBatchInterruptedMessage(payload),
          data: payload,
        })]}
      />,
    );
    const row = screen.getByText(payload.message);
    expect(row.closest('.auto-log-entry')).toHaveClass('log-warning');
  });
});

describe('proof framing gate activity', () => {
  test('an unresolved gate never says the model chose framing', () => {
    const unresolved = {
      is_proof_amenable: true,
      unresolved: true,
      message: 'Proof framing decision could not be read; framing is on for this run and will be asked again on resume.',
    };
    const text = formatProofFramingDecidedMessage(unresolved);
    expect(text).toBe(unresolved.message);
    expect(text).not.toMatch(/chose proof framing/);
    expect(formatProofFramingDecidedMessage({ is_proof_amenable: true, unresolved: true }))
      .toMatch(/could not be read; proof framing is on/);
    expect(getActivityClass('proof_framing_decided', { data: unresolved })).toBe('activity-warning');
  });

  test('a decided gate keeps its existing wording', () => {
    expect(formatProofFramingDecidedMessage({ is_proof_amenable: true }))
      .toBe('Submitter 1 (Main Submitter) chose proof framing for this research run');
    expect(formatProofFramingDecidedMessage({ is_proof_amenable: false }, 'run'))
      .toBe('Submitter 1 (Main Submitter) did not choose proof framing for this run');
  });

  test('AutonomousResearchLogs shows contract failures and the unresolved gate as warnings', () => {
    const unresolvedMessage = 'Proof framing decision could not be read; framing is on for this run.';
    render(
      <AutonomousResearchLogs
        stats={{}}
        events={[
          feedRow(MODEL_OUTPUT_CONTRACT_FAILED_EVENT, contractPayload()),
          {
            event: 'proof_framing_decided',
            timestamp: '2026-11-15T12:01:00Z',
            data: { is_proof_amenable: true, unresolved: true, message: unresolvedMessage },
          },
        ]}
      />,
    );
    const failure = screen.getByText(contractPayload().message);
    expect(failure.closest('.auto-log-entry')).toHaveClass('log-warning');
    const gate = screen.getByText(unresolvedMessage);
    expect(gate.closest('.auto-log-entry')).toHaveClass('log-warning');
    expect(screen.queryByText(/chose proof framing/)).toBeNull();
  });
});
