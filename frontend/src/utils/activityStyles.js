export const CONTEXT_OVERFLOW_STOP_MESSAGE = 'Research stopped. Some required source content must be injected directly to preserve answer quality, and it reached the maximum context size for the selected model. Start a new session with a condensed prompt, or choose a model with a higher context limit.';

export const REJECTION_FEEDBACK_NOTICE = 'Rejections are normal and provide feedback to the model. Extended rejection streaks can be expected on difficult problems. Above is 10 submissions your validator thought were not worth your time!';

export const formatContextOverflowActivityMessage = (data = {}) => {
  const message = data.message || CONTEXT_OVERFLOW_STOP_MESSAGE;
  const configuredModel = data.configured_model || '';
  const configuredProvider = data.configured_provider || '';
  const effectiveModel = data.effective_model || data.model || '';
  const effectiveProvider = data.effective_provider || data.provider || '';
  const effectiveHost = data.effective_host_provider || data.host_provider || '';
  const configuredIdentity = [configuredModel, configuredProvider].filter(Boolean).join(' via ');
  const effectiveRoute = [
    [effectiveModel, effectiveProvider].filter(Boolean).join(' via '),
    effectiveHost ? `host ${effectiveHost}` : '',
  ].filter(Boolean).join(', ');
  const hasConfigured = Boolean(configuredIdentity);
  const hasEffective = Boolean(effectiveRoute);
  const routeChanged = hasConfigured && hasEffective && (
    (configuredModel && effectiveModel && configuredModel !== effectiveModel)
    || (configuredProvider && effectiveProvider && configuredProvider !== effectiveProvider)
  );
  let identity = '';
  if (routeChanged) {
    identity = `Effective route: ${effectiveRoute}. Configured route: ${configuredIdentity}.`;
  } else if (hasEffective) {
    identity = `Route: ${effectiveRoute}.`;
  } else if (hasConfigured) {
    identity = `Configured route: ${configuredIdentity}.`;
  }
  if (!identity) return message;
  const separator = /[.!?]$/.test(message.trim()) ? ' ' : '. ';
  return `${message}${separator}${identity}`;
};

export const shouldAddRejectionFeedbackNotice = (data = {}, observedConsecutiveRejections = null, shown = {}) => {
  const total = Number(data.total_rejections ?? data.total_rejection_count ?? data.rejection_count);
  const consecutive = Number(data.consecutive_rejections ?? data.consecutive);
  const observed = Number(observedConsecutiveRejections);
  const isFirstRejection = total === 1 || consecutive === 1 || observed === 1;
  const isTenthConsecutiveRejection = consecutive === 10 || observed === 10;
  return (isFirstRejection && !shown.first) || (isTenthConsecutiveRejection && !shown.tenth);
};

const timestampAfter = (timestamp) => {
  const parsed = new Date(timestamp || '').getTime();
  return Number.isNaN(parsed) ? timestamp : new Date(parsed + 1).toISOString();
};

export const buildRejectionFeedbackNoticeActivity = (timestamp, data = {}) => ({
  event: 'rejection_feedback_notice',
  type: 'rejection_feedback_notice',
  timestamp: timestampAfter(timestamp),
  message: REJECTION_FEEDBACK_NOTICE,
  data: {
    total_rejections: data.total_rejections,
    consecutive_rejections: data.consecutive_rejections ?? data.consecutive,
  },
});

export const formatSolutionPathEventMessage = (event = '', data = {}) => {
  if (data.message) return data.message;
  const queued = Number(data.queued_proposals || 0);
  switch (event) {
    case 'solution_path_activated':
      return 'Progressive solution-path tracking is now active.';
    case 'solution_path_proposal_queued':
      return `A solution-path update was queued for Main Submitter 1 review${queued ? ` (${queued} queued)` : ''}.`;
    case 'solution_path_proposal_reviewing':
      return 'Main Submitter 1 is reviewing a proposed solution-path update.';
    case 'solution_path_updated':
      return `Main Submitter 1 approved solution path revision ${Number(data.revision || 0)}.`;
    case 'solution_path_proposal_rejected':
      return 'Main Submitter 1 rejected a proposed solution-path update.';
    case 'solution_path_proposal_retry_queued':
      return `A solution-path update remains queued for retry${queued ? ` (${queued} queued)` : ''}.`;
    case 'solution_path_proposal_user_repair_required':
      return 'A solution-path update needs a provider, model, key, privacy, or context setting repaired before review can continue.';
    case 'solution_path_proposal_resumed':
      return 'The repaired solution-path update was explicitly returned to the Main Submitter 1 review queue.';
    default:
      return 'Solution path changed.';
  }
};

const PROOF_RUN_EVENT_PREFIXES = ['proof_run_', 'proof_prune_'];

export const isProofRunActivityEvent = (event = '') => (
  PROOF_RUN_EVENT_PREFIXES.some((prefix) => String(event).startsWith(prefix))
  || [
    'proof_check_started',
    'proof_check_no_candidates',
    'proof_check_candidates_found',
    'proof_check_complete',
  ].includes(event)
);

export const getProofActivityScope = (data = {}) => {
  const explicit = String(data.proof_scope || data.scope || data.workflow_mode || '').toLowerCase();
  if (explicit === 'manual' || explicit === 'manual_proof_check') return 'manual';
  if (explicit === 'autonomous') return 'autonomous';
  const sourceId = String(data.source_id || '').toLowerCase();
  const trigger = String(data.trigger || '').toLowerCase();
  if (
    sourceId === 'manual_aggregator'
    || sourceId === 'manual_compiler_current'
    || sourceId.startsWith('manual_compiler_')
    || trigger.startsWith('manual')
  ) {
    return 'manual';
  }
  return 'autonomous';
};

export const getProofActivityIdentity = (event = '', data = {}) => {
  if (!isProofRunActivityEvent(event)) return '';
  const notificationIdentity = data.notification_key || data.notification_id;
  if (notificationIdentity) return `proof-notification:${notificationIdentity}`;
  const runId = data.proof_run_id || data.run_id;
  if (!runId) return '';
  const round = data.proof_round_index || data.round_index || data.current_round || '';
  const generation = data.lifecycle_generation || '';
  const eventInstance = data.event_id || data.sequence || '';
  const subject = (
    data.proposal_id
    || data.candidate_id
    || data.proof_id
    || data.theorem_id
    || data.proof_label
    || ''
  );
  const attempt = data.attempt || data.attempt_index || '';
  return [
    'proof-run',
    getProofActivityScope(data),
    runId,
    event,
    `generation-${generation}`,
    `round-${round}`,
    `subject-${subject}`,
    `attempt-${attempt}`,
    `instance-${eventInstance}`,
  ].join(':');
};

export const hasRecentProofActivityDuplicate = (events = [], event = '', data = {}) => {
  const identity = getProofActivityIdentity(event, data);
  if (!identity) return false;
  return events.slice(-250).some((item) => (
    getProofActivityIdentity(item.event || item.type || '', item.data || item) === identity
  ));
};

export const formatEmptyProofDiscoveryMessage = (prefix = 'Proof discovery') => (
  `${prefix}: the model searched for useful novel proof candidates and found none, `
  + 'so no Lean proof attempts were needed.'
);

const parseProviderResetTime = (data = {}) => {
  const raw = data.cooldown_until ?? data.resets_at;
  const numeric = Number(raw);
  if (Number.isFinite(numeric) && numeric > 0) {
    return numeric > 1e12 ? numeric : numeric * 1000;
  }
  return null;
};

export const isProviderUsageLimitActive = (data = {}, nowMs = Date.now()) => {
  if ((data.reason || '') !== 'usage_limit_reached') return false;
  const resetTime = parseProviderResetTime(data);
  return resetTime === null || resetTime > nowMs;
};

export const shouldShowProviderUsageLimitPopup = (data = {}, nowMs = Date.now()) => (
  isProviderUsageLimitActive(data, nowMs) && !data.fallback_model
);

export const formatProviderUsageLimitActivityMessage = (
  data = {},
  fallbackProviderLabel = 'Provider',
) => {
  const providerLabel = data.provider_label || fallbackProviderLabel;
  const roleId = data.role_id || 'a role';
  const resetTime = parseProviderResetTime(data);
  const resetText = resetTime === null
    ? ''
    : ` Reset time: ${new Date(resetTime).toLocaleString()}.`;
  if (data.fallback_model) {
    return `${providerLabel} usage limit reached for ${roleId}. `
      + `Using LM Studio fallback (${data.fallback_model}) until reset.${resetText}`;
  }
  return `${providerLabel} usage limit reached for ${roleId}. `
    + `This role is waiting for the provider cooldown to end.${resetText}`;
};

export const formatProviderUsageLimitResumedMessage = (
  data = {},
  fallbackProviderLabel = 'Provider',
) => {
  const providerLabel = data.provider_label || fallbackProviderLabel;
  const roleId = data.role_id || 'a role';
  return `${providerLabel} usage limit ended for ${roleId}; provider work resumed.`;
};

export const formatProofRunEventMessage = (event = '', data = {}) => {
  if (data.message) return data.message;
  const round = Number(data.proof_round_index || data.round_index || data.current_round || 0);
  const roundLabel = round > 0 ? `Round ${round}` : 'Proof run';
  switch (event) {
    case 'proof_run_started':
    case 'proof_run_queued':
      return `${roundLabel} started${data.run_mode === 'loop_with_pruning' ? ' in continuous mode' : ''}.`;
    case 'proof_run_round_started':
      return `${roundLabel} started. Proof discovery will identify prompt-relevant candidates, Lean 4 will verify each attempted proof, and accepted proofs may trigger a non-blocking pruning review.`;
    case 'proof_run_round_complete': {
      const candidateCount = Number(data.candidate_count);
      const hasCandidateCount = Number.isFinite(candidateCount) && candidateCount >= 0;
      const candidateText = hasCandidateCount
        ? (candidateCount === 0
          ? formatEmptyProofDiscoveryMessage('Discovery')
          : `Discovery found ${candidateCount} ${candidateCount === 1 ? 'candidate' : 'candidates'} for this round.`)
        : 'The round finished its proof discovery and verification work.';
      const nextText = data.run_mode === 'loop_with_pruning' && data.next_round_automatic !== false
        ? ' The next round will start automatically; the loop continues until you press Stop.'
        : '';
      return `${roundLabel} complete. ${candidateText}${nextText}`;
    }
    case 'proof_run_provider_resumed':
      return `${roundLabel} resumed after the provider pause.`;
    case 'proof_run_stopping':
      return 'Proof run is stopping after its active work drains.';
    case 'proof_run_stopped':
      return 'Proof run stopped.';
    case 'proof_run_provider_paused':
      return 'Proof run paused for provider credits.';
    case 'proof_run_repair_required':
      return 'Proof run needs provider, model, source, or runtime repair. Repair settings, then start a new proof loop.';
    case 'proof_run_terminal':
    case 'proof_run_failed':
      return `Proof run ended${(data.terminal_reason || data.reason) ? `: ${data.terminal_reason || data.reason}` : '.'}`;
    case 'proof_prune_review_queued':
      return 'Proof solving continues while a non-destructive pruning review waits to start.';
    case 'proof_prune_review_started':
      return 'Proof solving continues while the pruning review examines the active proof context.';
    case 'proof_prune_proposed':
      return data.proposal?.action === 'no_prune'
        ? 'Rigor & Proofs proposed no pruning; keeping every proof is normal.'
        : `Rigor & Proofs proposed excluding ${data.proposal?.proof_id || 'one occurrence'} from this run’s later context.`;
    case 'proof_prune_validation_started':
      return 'Validator is independently reviewing the non-destructive pruning proposal.';
    case 'proof_prune_no_change':
      return `No proof was pruned; this is a normal review result${data.reason ? `: ${data.reason}` : '.'}`;
    case 'proof_prune_applied':
      return `Proof occurrence ${data.proof_id || ''} was excluded only from this owning run’s later context; its record and exports remain available.`;
    case 'proof_prune_rejected':
      return `Validator rejected the pruning proposal${data.reason ? `: ${data.reason}` : '.'}`;
    case 'proof_prune_stale':
      return `The pruning proposal became stale and made no change${data.reason ? `: ${data.reason}` : '.'}`;
    case 'proof_prune_provider_paused':
      return 'Pruning paused for provider credits; proof solving continues independently.';
    case 'proof_prune_repair_required':
      return 'Pruning needs provider or settings repair; healthy proof solving continues independently.';
    case 'proof_prune_error':
      return `Pruning failed without changing proof validity or storage${data.message ? `: ${data.message}` : '.'}`;
    default:
      return String(event || 'proof run').replaceAll('_', ' ');
  }
};

export const getActivityIcon = (event = '') => {
  switch (event) {
    case 'solution_path_activated':
      return '◇';
    case 'solution_path_proposal_queued':
      return '+';
    case 'solution_path_proposal_reviewing':
      return '◎';
    case 'solution_path_updated':
      return '✓';
    case 'solution_path_proposal_rejected':
      return '✗';
    case 'solution_path_proposal_retry_queued':
      return '↺';
    case 'solution_path_proposal_user_repair_required':
      return '⚠';
    case 'solution_path_proposal_resumed':
      return '▶';
    case 'assistant_proof_pack_updated':
      return 'A';
    case 'assistant_proof_pack_failed':
      return '!';
    case 'brainstorm_submission_accepted':
    case 'submission_accepted':
    case 'compiler_acceptance':
    case 'outline_locked':
      return '✓';
    case 'system_started':
      return '▶';
    case 'system_stopped':
      return '■';
    case 'system_reset':
      return '↻';
    case 'new_submission':
      return '+';
    case 'brainstorm_submission_rejected':
    case 'submission_rejected':
    case 'compiler_rejection':
      return '✗';
    case 'rejection_feedback_notice':
      return 'i';
    case 'topic_selected':
      return '»';
    case 'topic_selection_rejected':
      return '⚠';
    case 'topic_exploration_started':
      return '◉';
    case 'topic_exploration_progress':
      return '◈';
    case 'topic_exploration_rejected':
      return '⚠';
    case 'topic_exploration_complete':
      return '✓';
    case 'paper_title_exploration_started':
      return '◉';
    case 'paper_title_exploration_progress':
      return '◈';
    case 'paper_title_exploration_complete':
      return '✓';
    case 'completion_review_started':
      return '◎';
    case 'hung_connection_alert':
      return '⧗';
    case 'oauth_provider_usage_limited':
      return '⏳';
    case 'provider_usage_limit_resumed':
      return '▶';
    case 'openai_codex_oauth_error':
    case 'oauth_provider_error':
    case 'sakana_fugu_error':
      return '⚠';
    case 'completion_review_result':
      return '□';
    case 'manual_paper_writing_triggered':
      return '▶';
    case 'brainstorm_hard_limit_reached':
      return '⊘';
    case 'paper_writing_started':
    case 'paper_writing_resumed':
      return '▬';
    case 'critique_phase_started':
      return '◎';
    case 'critique_progress':
      return '⊟';
    case 'self_review_appended':
      return '◈';
    case 'critique_phase_ended':
      return '✓';
    case 'compiler_decline':
      return '↷';
    case 'phase_transition':
      return '□';
    case 'paper_completed':
      return '⊟';
    case 'paper_redundancy_review':
      return '◇';
    case 'brainstorm_continuation_started':
      return '◎';
    case 'brainstorm_continuation_decided':
      return '⊞';
    case 'brainstorm_paper_limit_reached':
      return '⊘';
    case 'reference_selection_started':
      return '▭';
    case 'reference_selection_complete':
      return '✓';
    case 'auto_research_resumed':
      return '↻';
    case 'tier3_started':
      return '★';
    case 'tier3_result':
      return '⊟';
    case 'tier3_format_selected':
      return '▬';
    case 'tier3_volume_organized':
      return '▭';
    case 'tier3_chapter_started':
      return '✎';
    case 'tier3_chapter_complete':
      return '✓';
    case 'tier3_complete':
      return '◆';
    case 'tier3_rejection':
      return '⚠';
    case 'tier3_forced':
      return '▶';
    case 'tier3_phase_changed':
      return '↻';
    case 'tier3_paper_started':
      return '▬';
    case 'tier3_short_form_complete':
    case 'tier3_long_form_complete':
      return '✓';
    case 'final_answer_complete':
      return '◆';
    case 'proof_framing_decided':
      return 'P';
    case 'proof_check_started':
    case 'proof_run_started':
    case 'proof_run_queued':
    case 'proof_run_round_started':
      return '◌';
    case 'proof_run_stopping':
    case 'proof_run_stopped':
      return '■';
    case 'proof_run_provider_paused':
      return '⏳';
    case 'proof_run_repair_required':
    case 'proof_run_terminal':
    case 'proof_run_failed':
      return '!';
    case 'proof_retry_scheduled':
      return '↺';
    case 'proof_retry_started':
      return '↻';
    case 'proof_check_candidates_found':
      return '#';
    case 'proof_check_no_candidates':
      return '-';
    case 'smt_check_started':
      return 'S';
    case 'smt_check_error':
      return '!';
    case 'smt_check_complete':
      return 'Z';
    case 'proof_attempt_started':
      return '>';
    case 'proof_lean_accepted':
      return '>';
    case 'proof_integrity_rejected':
      return '⚠';
    case 'proof_attempt_failed':
    case 'proof_attempts_exhausted':
    case 'proof_truncation_recovery_exhausted':
      return '⚠';
    case 'context_overflow_error':
    case 'proof_context_overflow':
      return '!';
    case 'proof_verified':
    case 'known_proof_verified':
    case 'proof_registration_duplicate':
    case 'proof_check_complete':
      return '✓';
    case 'novel_proof_discovered':
      return '◆';
    case 'proof_dependency_added':
      return '↗';
    case 'proof_prune_review_queued':
    case 'proof_prune_review_started':
    case 'proof_prune_proposed':
    case 'proof_prune_validation_started':
      return '◌';
    case 'proof_prune_no_change':
    case 'proof_prune_applied':
      return '✓';
    case 'proof_prune_rejected':
    case 'proof_prune_stale':
    case 'proof_prune_provider_paused':
    case 'proof_prune_repair_required':
    case 'proof_prune_error':
      return '!';
    case 'leanoj_started':
      return '▶';
    case 'leanoj_stopped':
      return '■';
    case 'leanoj_phase_changed':
    case 'leanoj_path_decided':
    case 'leanoj_path_validated':
    case 'leanoj_role_json_retrying':
    case 'leanoj_model_call_started':
    case 'leanoj_brainstorm_submitters_started':
    case 'leanoj_brainstorm_submission_queued':
    case 'leanoj_brainstorm_batch_validation_started':
    case 'leanoj_sufficiency_check_started':
    case 'leanoj_brainstorm_phase_limit_reached':
      return '□';
    case 'leanoj_skip_brainstorm_requested':
    case 'leanoj_brainstorm_skip_deferred':
    case 'leanoj_brainstorm_skipped':
    case 'leanoj_force_brainstorm_requested':
    case 'leanoj_brainstorm_forced':
      return '↷';
    case 'leanoj_recursive_brainstorm_started':
      return '◎';
    case 'leanoj_recursive_brainstorm_completed':
      return '✓';
    case 'leanoj_topic_validated':
    case 'leanoj_model_call_completed':
    case 'leanoj_brainstorm_accepted':
    case 'leanoj_sufficiency_checked':
    case 'leanoj_brainstorm_prune_applied':
    case 'leanoj_brainstorm_proof_verified':
    case 'leanoj_master_proof_edit_applied':
    case 'leanoj_final_verified':
      return '✓';
    case 'leanoj_brainstorm_rejected':
    case 'leanoj_brainstorm_submitter_failed':
    case 'leanoj_brainstorm_prune_rejected':
    case 'leanoj_brainstorm_prune_apply_failed':
    case 'leanoj_brainstorm_prune_error':
    case 'leanoj_brainstorm_proof_failed':
    case 'leanoj_brainstorm_proof_registration_failed':
    case 'leanoj_model_call_failed':
    case 'leanoj_master_proof_edit_rejected':
    case 'leanoj_final_attempt_failed':
    case 'leanoj_final_attempt_cycle_exhausted':
    case 'leanoj_master_proof_stuck':
    case 'leanoj_master_proof_progress_watchdog':
    case 'leanoj_error':
      return '✗';
    case 'leanoj_final_attempt_started':
      return '>';
    case 'leanoj_partial_proof_saved':
      return '▭';
    case 'leanoj_master_proof_initialized':
      return 'P';
    case 'leanoj_master_proof_edit_started':
    case 'leanoj_master_proof_edit_validation_started':
      return '✎';
    case 'leanoj_brainstorm_prune_review_complete':
      return '◇';
    default:
      return '•';
  }
};

export const getActivityClass = (event = '', item = {}) => {
  const data = item?.data || item || {};
  if (event === 'leanoj_path_validated') {
    return data.validated === false ? 'activity-reject' : 'activity-success';
  }

  if (event === 'leanoj_sufficiency_checked') {
    return data.enough ? 'activity-success' : 'activity-info';
  }

  if (event === 'tier3_complete' || event === 'final_answer_complete') {
    return 'activity-tier3-complete';
  }

  if (event === 'hung_connection_alert') {
    return 'activity-warning';
  }

  if (event === 'oauth_provider_usage_limited') {
    return 'activity-warning';
  }

  if (event === 'provider_usage_limit_resumed') {
    return 'activity-success';
  }

  if (
    event === 'proof_run_provider_paused'
    || event === 'proof_run_repair_required'
    || event === 'proof_run_terminal'
    || event === 'proof_run_failed'
  ) {
    return 'activity-warning';
  }

  if (
    event === 'proof_run_round_complete'
    || event === 'proof_run_stopping'
    || event === 'proof_run_stopped'
  ) {
    return 'activity-info';
  }

  if (event === 'openai_codex_oauth_error' || event === 'oauth_provider_error' || event === 'sakana_fugu_error') {
    return 'activity-warning';
  }

  if (
    event === 'assistant_proof_pack_updated' ||
    event === 'solution_path_activated' ||
    event === 'solution_path_proposal_queued' ||
    event === 'solution_path_proposal_reviewing'
  ) {
    return 'activity-info';
  }

  if (event === 'solution_path_updated') {
    return 'activity-success';
  }

  if (event === 'solution_path_proposal_rejected') {
    return 'activity-reject';
  }

  if (
    event === 'solution_path_proposal_retry_queued'
    || event === 'solution_path_proposal_user_repair_required'
  ) {
    return 'activity-warning';
  }

  if (event === 'solution_path_proposal_resumed') {
    return 'activity-info';
  }

  if (
    event.includes('accepted') ||
    event === 'compiler_acceptance' ||
    event === 'outline_locked' ||
    event === 'paper_completed' ||
    event === 'self_review_appended' ||
    event === 'topic_exploration_complete' ||
    event === 'paper_title_exploration_complete' ||
    event === 'tier3_chapter_complete' ||
    event === 'tier3_short_form_complete' ||
    event === 'tier3_long_form_complete' ||
    event === 'reference_selection_complete' ||
    event === 'proof_verified' ||
    event === 'proof_lean_accepted' ||
    event === 'novel_proof_discovered' ||
    event === 'known_proof_verified' ||
    event === 'proof_registration_duplicate' ||
    event === 'proof_check_complete' ||
    event === 'smt_check_complete' ||
    event === 'leanoj_model_call_completed' ||
    event === 'leanoj_recursive_brainstorm_completed' ||
    event === 'leanoj_topic_validated' ||
    event === 'leanoj_brainstorm_prune_applied' ||
    event === 'leanoj_brainstorm_proof_verified' ||
    event === 'leanoj_master_proof_edit_applied' ||
    event === 'leanoj_final_verified'
  ) {
    return 'activity-success';
  }

  if (
    event.includes('rejected') ||
    event === 'compiler_rejection' ||
    event === 'tier3_rejection' ||
    event === 'proof_attempt_failed' ||
    event === 'proof_attempts_exhausted' ||
    event === 'proof_truncation_recovery_exhausted' ||
    event === 'assistant_proof_pack_failed' ||
    event === 'proof_integrity_rejected' ||
    event === 'smt_check_error' ||
    event === 'context_overflow_error' ||
    event === 'proof_context_overflow' ||
    event === 'leanoj_brainstorm_rejected' ||
    event === 'leanoj_brainstorm_submitter_failed' ||
    event === 'leanoj_brainstorm_prune_rejected' ||
    event === 'leanoj_brainstorm_prune_apply_failed' ||
    event === 'leanoj_brainstorm_prune_error' ||
    event === 'leanoj_brainstorm_proof_failed' ||
    event === 'leanoj_brainstorm_proof_registration_failed' ||
    event === 'leanoj_model_call_failed' ||
    event === 'leanoj_master_proof_edit_rejected' ||
    event === 'leanoj_final_attempt_failed' ||
    event === 'leanoj_final_attempt_cycle_exhausted' ||
    event === 'leanoj_master_proof_stuck' ||
    event === 'leanoj_master_proof_progress_watchdog' ||
    event === 'leanoj_error'
  ) {
    return 'activity-reject';
  }

  if (event === 'rejection_feedback_notice') {
    return 'activity-info';
  }

  if (
    event.includes('review') ||
    event.includes('started') ||
    event.includes('resumed') ||
    event.includes('progress') ||
    event.includes('transition') ||
    event === 'new_submission' ||
    event === 'system_stopped' ||
    event === 'system_reset' ||
    event === 'manual_paper_writing_triggered' ||
    event === 'brainstorm_hard_limit_reached' ||
    event === 'tier3_forced' ||
    event === 'tier3_phase_changed' ||
    event === 'tier3_result' ||
    event === 'tier3_format_selected' ||
    event === 'tier3_volume_organized' ||
    event === 'topic_selected' ||
    event === 'reference_selection_started' ||
    event === 'compiler_decline' ||
    event === 'critique_phase_ended' ||
    event === 'brainstorm_continuation_decided' ||
    event === 'brainstorm_paper_limit_reached' ||
    event === 'proof_framing_decided' ||
    event === 'proof_retry_scheduled' ||
    event === 'proof_retry_started' ||
    event === 'proof_check_candidates_found' ||
    event === 'proof_check_no_candidates' ||
    event === 'proof_attempt_started' ||
    event === 'smt_check_started' ||
    event === 'leanoj_started' ||
    event === 'leanoj_stopped' ||
    event === 'leanoj_phase_changed' ||
    event === 'leanoj_model_call_started' ||
    event === 'leanoj_recursive_brainstorm_started' ||
    event === 'leanoj_brainstorm_submitters_started' ||
    event === 'leanoj_brainstorm_submission_queued' ||
    event === 'leanoj_brainstorm_batch_validation_started' ||
    event === 'leanoj_sufficiency_check_started' ||
    event === 'leanoj_brainstorm_phase_limit_reached' ||
    event === 'leanoj_role_json_retrying' ||
    event === 'leanoj_skip_brainstorm_requested' ||
    event === 'leanoj_brainstorm_skip_deferred' ||
    event === 'leanoj_brainstorm_skipped' ||
    event === 'leanoj_force_brainstorm_requested' ||
    event === 'leanoj_brainstorm_forced' ||
    event === 'leanoj_path_decided' ||
    event === 'leanoj_partial_proof_saved' ||
    event === 'leanoj_master_proof_initialized' ||
    event === 'leanoj_master_proof_edit_started' ||
    event === 'leanoj_brainstorm_prune_review_complete' ||
    event === 'leanoj_final_attempt_started'
  ) {
    return 'activity-info';
  }

  return 'activity-neutral';
};

export const formatAssistantProofPackMessage = (data = {}) => {
  const total = Number.isFinite(Number(data.result_count)) ? Number(data.result_count) : 0;
  const max = Number.isFinite(Number(data.max_result_count)) ? Number(data.max_result_count) : 7;
  const local = Number.isFinite(Number(data.local_result_count)) ? Number(data.local_result_count) : 0;
  const synthetic = Number.isFinite(Number(data.syntheticlib4_result_count))
    ? Number(data.syntheticlib4_result_count)
    : 0;
  const target = String(data.target_kind || '').replace(/_/g, ' ') || 'current target';
  const phase = String(data.workflow_phase || '').replace(/_/g, ' ').trim();
  const phaseText = phase ? ` during ${phase}` : '';
  const warningCount = Array.isArray(data.warnings) ? data.warnings.filter(Boolean).length : 0;
  const warningText = warningCount ? ` (${warningCount} warning${warningCount === 1 ? '' : 's'})` : '';
  const rawSelectionMode = String(data.selection_mode || '').trim();
  const selectionMode = rawSelectionMode.replace(/_/g, ' ').trim();
  const assistantModel = String(data.assistant_model_id || '').trim();
  const assistantSelected = Boolean(String(data.assistant_role_id || assistantModel || '').trim());
  const selectorText = assistantSelected
    ? ` via Assistant${assistantModel ? ` (${assistantModel})` : ''}`
    : (selectionMode ? ` via ${selectionMode}` : '');
  const reviewedByCorpus = data.retrieval_observability?.deduped_distinct?.by_corpus || {};
  const reviewedSynthetic = Number(reviewedByCorpus.syntheticlib4 || 0);
  const reviewedLocal = Object.entries(reviewedByCorpus)
    .filter(([corpus]) => corpus !== 'syntheticlib4')
    .reduce((sum, [, count]) => sum + Number(count || 0), 0);
  const countsText = Object.keys(reviewedByCorpus).length
    ? `: reviewed ${reviewedLocal} local and ${reviewedSynthetic} SyntheticLib4; used ${local} local and ${synthetic} SyntheticLib4`
    : `: used ${local} local and ${synthetic} SyntheticLib4`;

  if (total === 0 && warningCount === 0) {
    return `Assistant memory found no useful proofs for ${target}${phaseText}${selectorText}${countsText}`;
  }

  return `Assistant memory returned ${total}/${max} proofs for ${target}${phaseText}${selectorText}${countsText}${warningText}`;
};

export const ASSISTANT_PROOF_PACK_EVENTS = new Set([
  'assistant_proof_pack_updated',
  'assistant_proof_pack_failed',
]);

export const ASSISTANT_PROOF_PACK_DUPLICATE_WINDOW_MS = 15000;

export const getAssistantProofPackDuplicateKey = (event = '', data = {}) => {
  if (!ASSISTANT_PROOF_PACK_EVENTS.has(event)) {
    return '';
  }
  return [
    event,
    data.target_hash || '',
    data.workflow_mode || '',
    data.target_kind || '',
    data.workflow_phase || '',
    data.source_type || '',
    data.source_id || '',
    data.assistant_role_id || '',
    data.assistant_model_id || '',
    data.result_count ?? '',
    data.max_result_count ?? '',
    data.local_result_count ?? '',
    data.syntheticlib4_result_count ?? '',
    data.candidate_count ?? '',
    data.shortlist_count ?? '',
    data.selection_mode || '',
    data.cooldown_kind || '',
    data.cooldown_stage ?? '',
    data.eligible_turns_remaining ?? '',
    data.batch_attempts ?? '',
    data.batch_size ?? '',
    data.reason || '',
    data.error_message || '',
    Array.isArray(data.warnings) ? data.warnings.join('|') : '',
  ].join('::');
};

const parseActivityTimestamp = (...values) => {
  for (const value of values) {
    if (!value) {
      continue;
    }
    const parsed = new Date(value).getTime();
    if (!Number.isNaN(parsed)) {
      return parsed;
    }
  }
  return NaN;
};

export const hasRecentAssistantProofPackDuplicate = (
  events = [],
  event = '',
  data = {},
  timestamp = new Date().toISOString(),
  windowMs = ASSISTANT_PROOF_PACK_DUPLICATE_WINDOW_MS
) => {
  const key = getAssistantProofPackDuplicateKey(event, data);
  if (!key) {
    return false;
  }
  const eventTime = parseActivityTimestamp(timestamp);
  const safeEventTime = Number.isNaN(eventTime) ? Date.now() : eventTime;
  return events.some((existing) => {
    const existingType = existing.event || existing.type || '';
    const existingData = existing.data || {};
    if (getAssistantProofPackDuplicateKey(existingType, existingData) !== key) {
      return false;
    }
    const existingTime = parseActivityTimestamp(
      existing.fullTimestamp,
      existing.timestamp,
      existing.data?._serverTimestamp
    );
    if (Number.isNaN(existingTime)) {
      return false;
    }
    return Math.abs(safeEventTime - existingTime) <= windowMs;
  });
};

export const formatAssistantProofPackEventMessage = (event = '', data = {}) => {
  const target = String(data.target_kind || '').replace(/_/g, ' ') || 'current target';
  const phase = String(data.workflow_phase || '').replace(/_/g, ' ').trim();
  const phaseText = phase ? ` during ${phase}` : '';
  const assistantModel = String(data.assistant_model_id || '').trim();
  const assistantSelected = Boolean(String(data.assistant_role_id || assistantModel || '').trim());
  const modelText = assistantSelected ? ` via Assistant${assistantModel ? ` (${assistantModel})` : ''}` : '';
  const observability = data.retrieval_observability || {};
  const reviewedByCorpus = observability.deduped_distinct?.by_corpus || {};
  const usedByCorpus = observability.final_selected?.by_corpus || {};
  const sumLocal = (counts) => Object.entries(counts)
    .filter(([corpus]) => corpus !== 'syntheticlib4')
    .reduce((sum, [, count]) => sum + Number(count || 0), 0);
  const counterText = Object.keys(reviewedByCorpus).length
    ? `: reviewed ${sumLocal(reviewedByCorpus)} local and ${Number(reviewedByCorpus.syntheticlib4 || 0)} SyntheticLib4; used ${sumLocal(usedByCorpus)} local and ${Number(usedByCorpus.syntheticlib4 || 0)} SyntheticLib4`
    : '';
  if (event === 'assistant_proof_pack_refresh_started') {
    return `Assistant memory refresh started for ${target}${phaseText}${modelText}`;
  }
  if (event === 'assistant_proof_pack_warning') {
    const warnings = Array.isArray(data.warnings) ? data.warnings.filter(Boolean).join('; ') : '';
    return `Assistant memory refresh warning for ${target}${phaseText}: ${warnings || 'proof-search support could not be refreshed'}`;
  }
  if (event === 'assistant_proof_pack_failed') {
    const detail = String(data.error_message || data.reason || '').trim();
    const failureText = detail ? `: ${detail}` : '';
    return `Assistant memory model call failed for ${target}${phaseText}${modelText}${failureText}${counterText}`;
  }
  if (event === 'assistant_proof_pack_stopped') {
    return `Assistant memory stopped (${data.reason || 'parent stopped'})`;
  }
  return formatAssistantProofPackMessage(data);
};

export const buildAutonomousProofProviderPauseActivity = (data = {}) => {
  const isCreditPause = data.reason === 'openrouter_credit_exhaustion';
  return {
    isCreditPause,
    message: isCreditPause
      ? `Autonomous proof verification paused until OpenRouter credits are reset: ${data.message || data.source_id || 'provider credits exhausted'}`
      : `Autonomous proof verification paused for a transient provider error and will retry automatically: ${data.message || data.source_id || 'provider retry pending'}`,
  };
};
