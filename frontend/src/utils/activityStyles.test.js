import {
  ASSISTANT_PROOF_PACK_EVENTS,
  REJECTION_FEEDBACK_NOTICE,
  buildAutonomousProofProviderPauseActivity,
  buildRejectionFeedbackNoticeActivity,
  formatAssistantProofPackEventMessage,
  formatAssistantProofPackMessage,
  formatContextOverflowActivityMessage,
  formatSolutionPathEventMessage,
  getActivityClass,
  getActivityIcon,
  getAssistantProofPackDuplicateKey,
  shouldAddRejectionFeedbackNotice,
} from './activityStyles';

test('context overflow activity identifies the effective or configured model', () => {
  expect(formatContextOverflowActivityMessage({
    message: 'Research stopped.',
    configured_model: 'configured/model',
    configured_provider: 'openrouter',
  })).toBe('Research stopped. Configured route: configured/model via openrouter.');

  expect(formatContextOverflowActivityMessage({
    message: 'Research stopped.',
    configured_model: 'configured/model',
    effective_model: 'fallback/model',
    effective_provider: 'lm_studio',
  })).toBe(
    'Research stopped. Effective route: fallback/model via lm_studio. '
    + 'Configured route: configured/model.'
  );

  expect(formatContextOverflowActivityMessage({
    message: 'Proof formalization skipped.',
    configured_model: 'configured/model',
    configured_provider: 'openrouter',
  })).toBe('Proof formalization skipped. Configured route: configured/model via openrouter.');
});

test('proof context overflow uses fatal activity styling without implying workflow stop', () => {
  expect(getActivityClass('proof_context_overflow')).toBe('activity-reject');
});

test('styles and formats every solution-path lifecycle event', () => {
  const expectations = {
    solution_path_activated: ['◇', 'activity-info'],
    solution_path_proposal_queued: ['+', 'activity-info'],
    solution_path_proposal_reviewing: ['◎', 'activity-info'],
    solution_path_updated: ['✓', 'activity-success'],
    solution_path_proposal_rejected: ['✗', 'activity-reject'],
    solution_path_proposal_retry_queued: ['↺', 'activity-warning'],
    solution_path_proposal_user_repair_required: ['⚠', 'activity-warning'],
    solution_path_proposal_resumed: ['▶', 'activity-info'],
  };

  Object.entries(expectations).forEach(([event, [icon, activityClass]]) => {
    expect(getActivityIcon(event)).toBe(icon);
    expect(getActivityClass(event)).toBe(activityClass);
    expect(formatSolutionPathEventMessage(event, {})).not.toBe('Solution path changed.');
  });
  expect(formatSolutionPathEventMessage('solution_path_proposal_queued', {
    queued_proposals: 2,
  })).toContain('(2 queued)');
  expect(formatSolutionPathEventMessage('solution_path_updated', {
    message: 'Engine supplied message.',
  })).toBe('Engine supplied message.');
});

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
    'Assistant memory found no useful proofs for brainstorm context during brainstorm via Assistant (openai/gpt-oss-20b): used 0 local and 0 SyntheticLib4'
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

test('formats Assistant model-output failure as an error activity', () => {
  const data = {
    candidate_count: 64,
    shortlist_count: 20,
    target_kind: 'brainstorm_context',
    workflow_phase: 'brainstorm',
    assistant_role_id: 'autonomous_assistant',
    assistant_model_id: 'google/gemma-4-26b-a4b',
    reason: 'assistant_llm_selection_failed',
    error_message: 'No JSON found in response',
  };

  expect(formatAssistantProofPackEventMessage('assistant_proof_pack_failed', data)).toBe(
    'Assistant memory model call failed for brainstorm context during brainstorm via Assistant (google/gemma-4-26b-a4b): No JSON found in response'
  );
  expect(getActivityClass('assistant_proof_pack_failed')).toBe('activity-reject');
});

test('formats federated Assistant lane counts without exposing proof content', () => {
  const message = formatAssistantProofPackEventMessage('assistant_proof_pack_updated', {
    result_count: 5,
    max_result_count: 7,
    candidate_count: 32,
    shortlist_count: 21,
    target_kind: 'paper',
    local_result_count: 4,
    syntheticlib4_result_count: 1,
    retrieval_observability: {
      raw_by_lane: {
        local: { total: 40 },
        duplicate_neighborhood: { total: 12 },
        syntheticlib4: { total: 8 },
      },
      deduped_distinct: {
        total: 48,
        by_corpus: { moto: 40, syntheticlib4: 8 },
      },
      fused_cap_64: { total: 32 },
      shortlist_21: { total: 21 },
      final_selected: {
        total: 5,
        by_corpus: { moto: 4, syntheticlib4: 1 },
      },
      matching_runs_examined: 9,
      matching_occurrences_examined: 60,
    },
  });

  expect(message).toContain('reviewed 40 local and 8 SyntheticLib4');
  expect(message).toContain('used 4 local and 1 SyntheticLib4');
  expect(message).not.toContain('64 candidates');
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
  expect(ASSISTANT_PROOF_PACK_EVENTS.has('assistant_proof_pack_failed')).toBe(true);
  expect(getAssistantProofPackDuplicateKey('assistant_proof_memory_cooldown', {
    target_hash: 'target',
    cooldown_kind: 'zero_useful',
  })).toBe('');
});

test('keeps distinct Assistant failure details out of duplicate suppression', () => {
  const base = {
    target_hash: 'target-1',
    workflow_mode: 'autonomous',
    target_kind: 'brainstorm_context',
    workflow_phase: 'brainstorm',
    source_type: 'brainstorm',
    source_id: 'topic_1',
    reason: 'assistant_llm_selection_failed',
  };

  expect(getAssistantProofPackDuplicateKey('assistant_proof_pack_failed', {
    ...base,
    error_message: 'No JSON found in response',
  })).not.toBe(getAssistantProofPackDuplicateKey('assistant_proof_pack_failed', {
    ...base,
    error_message: 'Response exceeded context window',
  }));
});

test('distinguishes autonomous proof transient provider pauses from credit pauses', () => {
  const transient = buildAutonomousProofProviderPauseActivity({
    reason: 'transient_provider_error',
    message: 'OpenRouter gateway timeout',
  });
  expect(transient.isCreditPause).toBe(false);
  expect(transient.message).toContain('will retry automatically');
  expect(transient.message).not.toContain('credits are reset');

  const credit = buildAutonomousProofProviderPauseActivity({
    reason: 'openrouter_credit_exhaustion',
    message: 'credits exhausted',
  });
  expect(credit.isCreditPause).toBe(true);
  expect(credit.message).toContain('credits are reset');
});
