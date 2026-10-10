import { describe, expect, test } from 'vitest';
import {
  boundedActivityDetail,
  formatValidatorDecisionActivity,
  getActivityIdentity,
  hasRecentActivityDuplicate,
  normalizeActivityEvent,
} from './activityNormalization';

describe('shared activity normalization', () => {
  test('normalizes legacy names and additive assurance fields', () => {
    const event = normalizeActivityEvent({
      type: 'reject',
      timestamp: '2026-10-05T12:00:00Z',
      data: {
        submission_id: 'sub-1',
        reasoning: 'Missing a required argument.',
      },
    });

    expect(event.event).toBe('submission_rejected');
    expect(event.type).toBe('submission_rejected');
    expect(event.category).toBe('validation');
    expect(event.severity).toBe('error');
    expect(event.outcome).toBe('rejected');
    expect(event.validatorFeedback).toBe('Missing a required argument.');
    expect(event.activityIdentity).toContain('sub-1');
  });

  test('redacts and bounds validator and error details', () => {
    const detail = boundedActivityDetail(`api_key=secret ${'x'.repeat(1300)}`);
    expect(detail).not.toContain('secret');
    expect(detail.length).toBeLessThanOrEqual(1200);

    const event = normalizeActivityEvent({
      event: 'compiler_error',
      data: { error_summary: 'Authorization: Bearer abc.def' },
    });
    expect(event.errorDetail).toContain('[redacted]');
    expect(event.errorDetail).not.toContain('abc.def');
  });

  test('uses stable identifiers to dedupe replay but preserves distinct attempts', () => {
    const first = {
      event: 'compiler_rejection',
      data: { submission_id: 'sub-1', attempt: 1, reasoning: 'first' },
    };
    const replay = {
      event: 'compiler_rejection',
      data: { submission_id: 'sub-1', attempt: 1, reasoning: 'replayed wording' },
    };
    const nextAttempt = {
      event: 'compiler_rejection',
      data: { submission_id: 'sub-1', attempt: 2, reasoning: 'second' },
    };

    expect(getActivityIdentity(first)).toBe(getActivityIdentity(replay));
    expect(hasRecentActivityDuplicate([first], replay)).toBe(true);
    expect(hasRecentActivityDuplicate([first], nextAttempt)).toBe(false);
  });

  test('does not collapse repeated lifecycle events that lack an instance identity', () => {
    const first = {
      event: 'phase_transition',
      timestamp: '2026-10-05T12:00:00Z',
      data: { workflow_mode: 'autonomous', from_phase: 'body', to_phase: 'conclusion' },
    };
    const later = {
      ...first,
      timestamp: '2026-10-05T13:00:00Z',
    };
    expect(getActivityIdentity(first)).not.toBe(getActivityIdentity(later));
    expect(hasRecentActivityDuplicate([first], later)).toBe(false);
  });

  test('formats validator decisions identically and safely with absent feedback', () => {
    expect(formatValidatorDecisionActivity({
      type: 'compiler_rejection',
      data: { mode: 'construction', reasoning: 'Use an exact edit anchor.' },
    })).toBe(
      'Submission REJECTED (construction mode). Validator feedback: Use an exact edit anchor.',
    );

    expect(formatValidatorDecisionActivity({
      type: 'submission_accepted',
      data: {},
    }, { subject: 'Submission from Submitter 2', includeMode: false })).toBe(
      'Submission from Submitter 2 ACCEPTED.',
    );
  });

  test('prefers explicit normalized envelope fields over legacy inference', () => {
    const event = normalizeActivityEvent({
      type: 'compiler_rejection',
      data: {
        activity_outcome: 'error',
        activity_category: 'workflow',
        activity_feedback: 'Structured validator feedback.',
        activity_detail: 'Structured operational detail.',
        activity_workflow: 'compiler',
        activity_run_id: 'manual-run-1',
        activity_generation: 4,
        activity_source_type: 'paper',
        activity_source_id: 'paper-1',
        activity_role_id: 'comp_val_001',
        retry_attempt: 2,
      },
    });

    expect(event.outcome).toBe('error');
    expect(event.category).toBe('workflow');
    expect(event.validatorFeedback).toBe('Structured validator feedback.');
    expect(event.errorDetail).toBe('Structured operational detail.');
    expect(event.activityIdentity).toContain('manual-run-1');
    expect(event.activityIdentity).toContain('paper-1');
  });

  test.each(['failed', 'failure', 'invalid'])(
    'keeps explicit operational outcome %s out of semantic rejection',
    (activityOutcome) => {
      const event = normalizeActivityEvent({
        type: 'validator_call_failed',
        data: { activity_outcome: activityOutcome, activity_category: 'workflow' },
      });
      expect(event.outcome).toBe('error');
      expect(event.severity).toBe('error');
    },
  );

  test.each(['paused', 'retrying'])(
    'preserves explicit transitional outcome %s',
    (activityOutcome) => {
      expect(normalizeActivityEvent({
        type: 'provider_retry_cooldown',
        data: { activity_outcome: activityOutcome },
      }).outcome).toBe(activityOutcome);
    },
  );
});
