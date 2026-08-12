import {
  ASSISTANT_PROOF_PACK_EVENTS,
  REJECTION_FEEDBACK_NOTICE,
  buildAutonomousProofProviderPauseActivity,
  buildRejectionFeedbackNoticeActivity,
  formatAssistantProofPackEventMessage,
  formatAssistantProofPackMessage,
  formatContextOverflowActivityMessage,
  formatEmptyProofDiscoveryMessage,
  formatProofRunEventMessage,
  formatProviderUsageLimitActivityMessage,
  formatProviderUsageLimitResumedMessage,
  formatSolutionPathEventMessage,
  getActivityClass,
  getActivityIcon,
  getAssistantProofPackDuplicateKey,
  getProofActivityIdentity,
  getProofActivityScope,
  hasRecentProofActivityDuplicate,
  isProviderUsageLimitActive,
  shouldShowProviderUsageLimitPopup,
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

test('proof truncation recovery exhaustion uses its dedicated warning activity style', () => {
  expect(getActivityIcon('proof_truncation_recovery_exhausted')).toBe('⚠');
  expect(getActivityClass('proof_truncation_recovery_exhausted')).toBe('activity-reject');
});

test('formats durable provider cooldown and confirmed resume activity', () => {
  const resetAt = 1_800_000_000;
  const fallbackMessage = formatProviderUsageLimitActivityMessage({
    provider_label: 'Sakana Fugu',
    role_id: 'proof_formalization',
    reason: 'usage_limit_reached',
    cooldown_until: resetAt,
    fallback_model: 'local-proof-model',
  });
  expect(fallbackMessage).toContain('Using LM Studio fallback (local-proof-model)');
  expect(fallbackMessage).toContain('Reset time:');

  const waitingMessage = formatProviderUsageLimitActivityMessage({
    provider_label: 'Sakana Fugu',
    role_id: 'proof_formalization',
    reason: 'usage_limit_reached',
    cooldown_until: resetAt,
  });
  expect(waitingMessage).toContain('waiting for the provider cooldown to end');
  expect(formatProviderUsageLimitResumedMessage({
    provider_label: 'Sakana Fugu',
    role_id: 'proof_formalization',
  })).toBe(
    'Sakana Fugu usage limit ended for proof_formalization; provider work resumed.'
  );
  expect(getActivityIcon('provider_usage_limit_resumed')).toBe('▶');
  expect(getActivityClass('provider_usage_limit_resumed')).toBe('activity-success');
});

test('shows usage-limit popup only for an active waiting cooldown', () => {
  const now = 1_700_000_000_000;
  const active = {
    reason: 'usage_limit_reached',
    cooldown_until: (now + 60_000) / 1000,
  };
  expect(isProviderUsageLimitActive(active, now)).toBe(true);
  expect(shouldShowProviderUsageLimitPopup(active, now)).toBe(true);
  expect(shouldShowProviderUsageLimitPopup({
    ...active,
    fallback_model: 'local-model',
  }, now)).toBe(false);
  expect(isProviderUsageLimitActive({
    ...active,
    cooldown_until: (now - 1) / 1000,
  }, now)).toBe(false);
  expect(shouldShowProviderUsageLimitPopup({
    ...active,
    cooldown_until: (now - 1) / 1000,
  }, now)).toBe(false);
});

test('formats proof-run lifecycle activity and source-specific state', () => {
  expect(formatProofRunEventMessage('proof_run_repair_required', {}))
    .toContain('start a new proof loop');
  expect(formatProofRunEventMessage('proof_run_provider_paused', {}))
    .toBe('Proof run paused for provider credits.');
  expect(getActivityClass('proof_run_repair_required')).toBe('activity-warning');
});

test('formats detailed continuous round activity without no-candidate exhaustion', () => {
  expect(formatEmptyProofDiscoveryMessage()).toBe(
    'Proof discovery: the model searched for useful novel proof candidates and found none, '
    + 'so no Lean proof attempts were needed.'
  );
  expect(formatProofRunEventMessage('proof_run_round_started', {
    run_mode: 'loop_with_pruning',
    current_round: 1,
  })).toContain('Proof discovery will identify prompt-relevant candidates');
  expect(formatProofRunEventMessage('proof_run_round_complete', {
    run_mode: 'loop_with_pruning',
    current_round: 1,
    candidate_count: 0,
    next_round_automatic: true,
  })).toBe(
    'Round 1 complete. Discovery: the model searched for useful novel proof candidates and found none, '
    + 'so no Lean proof attempts were needed. '
    + 'The next round will start automatically; the loop continues until you press Stop.'
  );
  expect(formatProofRunEventMessage('proof_run_round_complete', {
    run_mode: 'loop_with_pruning',
    current_round: 2,
    candidate_count: 2,
    next_round_automatic: true,
  })).toContain('Discovery found 2 candidates for this round');
});

test('dedupes proof activity by notification or run identity while preserving scope', () => {
  const autonomous = {
    run_id: 'run-1',
    proof_scope: 'autonomous',
    lifecycle_generation: 2,
  };
  const manual = {
    ...autonomous,
    proof_scope: 'manual',
  };
  expect(getProofActivityScope(autonomous)).toBe('autonomous');
  expect(getProofActivityScope(manual)).toBe('manual');
  expect(getProofActivityIdentity('proof_run_round_started', autonomous))
    .not.toBe(getProofActivityIdentity('proof_run_round_started', manual));
  expect(hasRecentProofActivityDuplicate([
    { event: 'proof_run_round_started', data: autonomous },
  ], 'proof_run_round_started', autonomous)).toBe(true);
  expect(hasRecentProofActivityDuplicate([
    { event: 'proof_run_round_started', data: autonomous },
  ], 'proof_run_round_started', manual)).toBe(false);
  expect(hasRecentProofActivityDuplicate([
    {
      event: 'proof_run_round_started',
      data: { ...autonomous, round_index: 1 },
    },
  ], 'proof_run_round_started', { ...autonomous, round_index: 2 })).toBe(false);
  expect(hasRecentProofActivityDuplicate([
    {
      event: 'proof_run_round_started',
      data: { ...autonomous, round_index: 2 },
    },
  ], 'proof_run_round_started', { ...autonomous, round_index: 2 })).toBe(true);
  expect(getProofActivityIdentity('proof_run_failed', {
    notification_key: 'proof-run:failed:1',
  })).toBe('proof-notification:proof-run:failed:1');
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
