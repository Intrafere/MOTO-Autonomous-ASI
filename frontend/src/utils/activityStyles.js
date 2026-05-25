export const getActivityIcon = (event = '') => {
  switch (event) {
    case 'brainstorm_submission_accepted':
    case 'submission_accepted':
    case 'compiler_acceptance':
    case 'outline_locked':
      return '✓';
    case 'brainstorm_submission_rejected':
    case 'submission_rejected':
    case 'compiler_rejection':
      return '✗';
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

  if (
    event.includes('review') ||
    event.includes('started') ||
    event.includes('resumed') ||
    event.includes('progress') ||
    event.includes('transition') ||
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
