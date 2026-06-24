export const CONTEXT_OVERFLOW_STOP_MESSAGE = 'Research stopped. Some required source content must be injected directly to preserve answer quality, and it reached the maximum context size for the selected model. Start a new session with a condensed prompt, or choose a model with a higher context limit.';

export const REJECTION_FEEDBACK_NOTICE = 'Rejections are normal and provide feedback to the model. Extended rejection streaks can be expected on difficult problems. Above is 10 submissions your validator thought were not worth your time!';

export const formatContextOverflowActivityMessage = (data = {}) => (
  data.message || CONTEXT_OVERFLOW_STOP_MESSAGE
);

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

export const getActivityIcon = (event = '') => {
  switch (event) {
    case 'assistant_proof_pack_updated':
      return 'A';
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
      return '◌';
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
      return '⚠';
    case 'context_overflow_error':
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

  if (event === 'openai_codex_oauth_error' || event === 'oauth_provider_error' || event === 'sakana_fugu_error') {
    return 'activity-warning';
  }

  if (
    event === 'assistant_proof_pack_updated'
  ) {
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
    event === 'proof_integrity_rejected' ||
    event === 'smt_check_error' ||
    event === 'context_overflow_error' ||
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
  const candidateCount = Number.isFinite(Number(data.candidate_count)) ? Number(data.candidate_count) : null;
  const candidateText = candidateCount !== null ? ` from ${candidateCount} candidates` : '';

  if (total === 0 && warningCount === 0) {
    return `Assistant memory found no useful proofs${candidateText} for ${target}${phaseText}${selectorText}: ${local} local, ${synthetic} SyntheticLib4`;
  }

  return `Assistant memory returned ${total}/${max} proofs${candidateText} for ${target}${phaseText}${selectorText}: ${local} local, ${synthetic} SyntheticLib4${warningText}`;
};

export const ASSISTANT_PROOF_PACK_EVENTS = new Set([
  'assistant_proof_pack_updated',
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
  const candidateCount = Number.isFinite(Number(data.candidate_count)) ? Number(data.candidate_count) : null;
  const shortlistCount = Number.isFinite(Number(data.shortlist_count)) ? Number(data.shortlist_count) : null;
  const candidateText = candidateCount !== null
    ? ` from ${candidateCount} candidates${shortlistCount !== null ? ` (${shortlistCount} shortlisted)` : ''}`
    : '';
  if (event === 'assistant_proof_pack_refresh_started') {
    if (candidateCount !== null && shortlistCount !== null && assistantSelected) {
      return `Assistant memory refresh started for ${target}${phaseText}: local proof-search ranking shortlisted ${shortlistCount} of ${candidateCount} candidates for Assistant review`;
    }
    return `Assistant memory refresh started${candidateText} for ${target}${phaseText}${modelText}`;
  }
  if (event === 'assistant_proof_pack_warning') {
    const warnings = Array.isArray(data.warnings) ? data.warnings.filter(Boolean).join('; ') : '';
    return `Assistant memory refresh warning for ${target}${phaseText}: ${warnings || 'proof-search support could not be refreshed'}`;
  }
  if (event === 'assistant_proof_pack_stopped') {
    return `Assistant memory stopped (${data.reason || 'parent stopped'})`;
  }
  return formatAssistantProofPackMessage(data);
};
