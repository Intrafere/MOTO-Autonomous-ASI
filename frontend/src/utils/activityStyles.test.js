import {
  ASSISTANT_PROOF_PACK_EVENTS,
  REJECTION_FEEDBACK_NOTICE,
  buildRejectionFeedbackNoticeActivity,
  formatAssistantProofPackEventMessage,
  formatAssistantProofPackMessage,
  getAssistantProofPackDuplicateKey,
  shouldAddRejectionFeedbackNotice,
} from './activityStyles';

test('formats clean empty Assistant proof pack as info instead of warning', () => {
  const message = formatAssistantProofPackMessage({
    result_count: 0,
    max_result_count: 7,
    candidate_count: 64,
    local_result_count: 0,
    syntheticlib4_result_count: 0,
    target_kind: 'brainstorm_context',
    workflow_phase: 'brainstorm',
    assistant_role_id: 'autonomous_assistant',
    assistant_model_id: 'openai/gpt-oss-20b',
    warnings: [],
  });

  expect(message).toBe(
    'Assistant memory found no useful proofs from 64 candidates for brainstorm context during brainstorm via Assistant (openai/gpt-oss-20b): 0 local, 0 SyntheticLib4'
  );
  expect(message).not.toContain('warning');
});

test('keeps Assistant warning count when backend reports a real warning', () => {
  const message = formatAssistantProofPackMessage({
    result_count: 0,
    max_result_count: 7,
    candidate_count: 64,
    local_result_count: 0,
    syntheticlib4_result_count: 0,
    target_kind: 'brainstorm_context',
    workflow_phase: 'brainstorm',
    assistant_role_id: 'autonomous_assistant',
    assistant_model_id: 'openai/gpt-oss-20b',
    warnings: ['Assistant LLM selection failed: provider unavailable'],
  });

  expect(message).toContain('(1 warning)');
});

test('adds rejection feedback notice on first and tenth consecutive rejection only', () => {
  expect(shouldAddRejectionFeedbackNotice({ total_rejections: 1 })).toBe(true);
  expect(shouldAddRejectionFeedbackNotice({ total_rejections: 7, consecutive_rejections: 10 })).toBe(true);
  expect(shouldAddRejectionFeedbackNotice({ total_rejections: 7, consecutive_rejections: 2 })).toBe(false);
  expect(shouldAddRejectionFeedbackNotice({ total_rejections: 1 }, null, { first: true })).toBe(false);
  expect(shouldAddRejectionFeedbackNotice({}, 10, { tenth: true })).toBe(false);
});

test('builds a secondary rejection feedback activity after the rejection timestamp', () => {
  const activity = buildRejectionFeedbackNoticeActivity('2026-06-21T19:27:19.000Z', {
    total_rejections: 1,
  });

  expect(activity.event).toBe('rejection_feedback_notice');
  expect(activity.message).toBe(REJECTION_FEEDBACK_NOTICE);
  expect(activity.timestamp).toBe('2026-06-21T19:27:19.001Z');
  expect(activity.data.total_rejections).toBe(1);
});

test('does not treat Assistant skip or cooldown events as displayable live activity', () => {
  expect(ASSISTANT_PROOF_PACK_EVENTS.has('assistant_proof_memory_unavailable')).toBe(false);
  expect(ASSISTANT_PROOF_PACK_EVENTS.has('assistant_proof_memory_cooldown')).toBe(false);
  expect(ASSISTANT_PROOF_PACK_EVENTS.has('assistant_proof_memory_shutdown')).toBe(false);
  expect(getAssistantProofPackDuplicateKey('assistant_proof_memory_cooldown', {
    target_hash: 'target',
    cooldown_kind: 'zero_useful',
  })).toBe('');
});
