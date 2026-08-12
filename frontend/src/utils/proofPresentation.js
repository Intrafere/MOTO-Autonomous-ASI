export const STRICT_NOVELTY_CATEGORIES = Object.freeze([
  'major_mathematical_discovery',
  'mathematical_discovery',
  'novel_variant',
  'novel_formulation',
  'duplicate_novel',
  'not_novel',
]);

const NOVELTY_PRESENTATION = Object.freeze({
  major_mathematical_discovery: {
    label: 'Major Mathematical Discovery',
    shortLabel: 'Major Discovery',
    badgeClass: 'proof-badge--platinum',
    cardClass: 'proof-card--platinum',
    tileClass: 'assistant-proof-tile--platinum',
    group: 'novel',
  },
  mathematical_discovery: {
    label: 'Minor Mathematical Discovery',
    shortLabel: 'Discovery',
    badgeClass: 'proof-badge--gold',
    cardClass: 'proof-card--gold',
    tileClass: 'assistant-proof-tile--gold',
    group: 'novel',
  },
  novel_variant: {
    label: 'Novel Reformulation',
    shortLabel: 'Novel Variant',
    badgeClass: 'proof-badge--silver',
    cardClass: 'proof-card--silver',
    tileClass: 'assistant-proof-tile--silver',
    group: 'novel',
  },
  novel_formulation: {
    label: 'Novel Formalization',
    shortLabel: 'Novel Formalization',
    badgeClass: 'proof-badge--bronze',
    cardClass: 'proof-card--bronze',
    tileClass: 'assistant-proof-tile--bronze',
    group: 'novel',
  },
  duplicate_novel: {
    label: 'Duplicate Novel',
    shortLabel: 'Duplicate Novel',
    badgeClass: 'proof-badge--duplicate-novel',
    cardClass: 'proof-card--duplicate-novel',
    tileClass: 'assistant-proof-tile--duplicate-novel',
    group: 'duplicate_novel',
  },
  not_novel: {
    label: 'Not Novel',
    shortLabel: 'Not Novel',
    badgeClass: 'proof-badge--known',
    cardClass: 'proof-card--known',
    tileClass: 'assistant-proof-tile--known',
    group: 'not_novel',
  },
  unknown: {
    label: 'Unknown (Legacy)',
    shortLabel: 'Unknown',
    badgeClass: 'proof-badge--known',
    cardClass: 'proof-card--known',
    tileClass: 'assistant-proof-tile--known',
    group: 'unknown',
  },
});

export function classifyProofNovelty(proof = {}) {
  const category = String(proof.novelty_tier || '').trim().toLowerCase();
  return {
    category: STRICT_NOVELTY_CATEGORIES.includes(category) ? category : 'unknown',
    ...NOVELTY_PRESENTATION[STRICT_NOVELTY_CATEGORIES.includes(category) ? category : 'unknown'],
  };
}

const USER_PRUNE_ACTORS = new Set(['user']);

function firstNonEmpty(...values) {
  return values.find((value) => String(value ?? '').trim()) ?? '';
}

export function classifyProofLiveContext(proof = {}, options = {}) {
  const historical = Boolean(options.historical);
  const status = String(proof.live_context_status || 'active').trim().toLowerCase() === 'pruned'
    ? 'pruned'
    : 'active';
  const actor = String(proof.live_context_pruned_by || '').trim().toLowerCase();
  const ownerRunId = String(
    proof.live_context_owner_run_id || proof.run_id || proof.session_id || ''
  ).trim();
  const reason = String(firstNonEmpty(
    proof.live_context_prune_reason,
    proof.live_context_prune_validator_reasoning
  )).trim();

  return {
    status,
    isPruned: status === 'pruned',
    isActive: status !== 'pruned',
    historical,
    readOnly: historical,
    actor,
    actorLabel: actor === 'user'
      ? 'User'
      : actor === 'automatic_proof_pruning'
        ? 'Automatic proof-pruning review'
        : actor
          ? actor.replace(/_/g, ' ')
          : 'Unknown',
    ownerRunId,
    reason,
    prunedAt: proof.live_context_pruned_at || null,
    canUndo: !historical && status === 'pruned' && USER_PRUNE_ACTORS.has(actor),
    canPrune: !historical && status === 'active',
    badgeLabel: status === 'pruned' ? 'Pruned from live context' : 'Active in live context',
    badgeClass: status === 'pruned'
      ? 'proof-live-context-badge--pruned'
      : 'proof-live-context-badge--active',
  };
}

export function getProofLiveContextRiskWarnings(proof = {}, dependencyState = {}) {
  const warnings = [];
  const sourceType = String(proof.source_type || '').trim().toLowerCase();
  if (
    sourceType === 'leanoj_final'
    || proof.is_final_solution === true
    || proof.final_solution === true
    || proof.proof_kind === 'final'
  ) {
    warnings.push('This occurrence is marked as a verified final-solution proof.');
  }

  const dependents = dependencyState.dependedOnBy
    || dependencyState.depended_on_by
    || proof.depended_on_by
    || [];
  const dependentCount = Array.isArray(dependents)
    ? dependents.length
    : Math.max(0, Number(proof.dependent_count || proof.dependents_count || 0));
  if (dependentCount > 0) {
    warnings.push(
      `${dependentCount} stored proof${dependentCount === 1 ? '' : 's'} ${dependentCount === 1 ? 'depends' : 'depend'} on this occurrence.`
    );
  }

  const extractionStatus = String(
    proof.dependency_extraction_status || dependencyState.dependencyExtractionStatus || ''
  ).trim().toLowerCase();
  if (extractionStatus && extractionStatus !== 'complete') {
    warnings.push(
      `Dependency extraction is ${extractionStatus.replace(/_/g, ' ')}; downstream risk may be incomplete.`
    );
  }
  if (!extractionStatus && proof.dependencies_complete === false) {
    warnings.push('Dependency extraction is incomplete; downstream risk may be incomplete.');
  }
  return warnings;
}

export function buildProofLiveContextMutation(proof = {}, {
  status,
  reason,
  proofSetRevision,
} = {}) {
  const runId = String(
    proof.live_context_owner_run_id || proof.run_id || proof.session_id || ''
  ).trim();
  const revisionValue = proofSetRevision
    ?? proof.proof_set_revision
    ?? proof.live_context_prune_snapshot_revision;
  const revision = revisionValue === null || revisionValue === undefined || revisionValue === ''
    ? Number.NaN
    : Number(revisionValue);
  if (!runId) {
    throw new Error('Refresh required: this proof has no owning run identity.');
  }
  if (!Number.isInteger(revision) || revision < 0) {
    throw new Error('Refresh required: the current proof-set revision is unavailable.');
  }
  return {
    status,
    actor: 'user',
    expected_run_id: runId,
    expected_proof_set_revision: revision,
    reason: String(reason || '').trim(),
    expected_theorem_hash: String(
      proof.canonical_theorem_statement_hash || proof.theorem_statement_hash || ''
    ),
    expected_lean_hash: String(
      proof.canonical_lean_code_hash || proof.lean_code_hash || ''
    ),
  };
}

export function sanitizeDomId(value, prefix = 'proof') {
  const normalized = String(value ?? '')
    .normalize('NFKD')
    .replace(/[^A-Za-z0-9_-]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return `${prefix}-${normalized || 'unknown'}`;
}

export function getCanonicalProofIdentity(proof = {}, { includeIndex = false, index = 0 } = {}) {
  const canonical = String(proof.search_id || '').trim();
  if (canonical) return canonical;
  const corpus = String(proof.corpus || proof.scope || 'proof').trim();
  const run = String(proof.run_id || proof.session_id || 'legacy').trim();
  const proofId = String(proof.proof_id || proof.library_id || proof.lean_code_hash || proof.theorem_statement_hash || 'unknown').trim();
  return [corpus, run, proofId, includeIndex ? index : ''].filter((part) => part !== '').join(':');
}

export function getLeanOJProofPresentation(proof = {}) {
  if (proof.proof_kind === 'final') {
    return {
      badgeClass: 'proof-badge--gold',
      cardClass: 'proof-card--gold',
      label: 'Final Verified Submission',
    };
  }
  return {
    badgeClass: 'proof-badge--silver',
    cardClass: 'proof-card--silver',
    label: 'Verified Proof Fragment',
  };
}

export function formatProofProvenance(proof = {}) {
  const runId = proof.run_id || '';
  const sessionId = proof.session_id || '';
  const source = proof.source_type
    ? `${proof.corpus_scope ? `${proof.corpus_scope} · ` : ''}${proof.source_type}${proof.source_id ? `/${proof.source_id}` : ''}`
    : '';
  const lanes = Array.isArray(proof.retrieval_lanes) ? proof.retrieval_lanes : [];
  const omitted = Number(proof.occurrence_omitted ?? proof.omitted_total ?? 0);
  return {
    runId,
    sessionId,
    source,
    lanes,
    omitted: Number.isFinite(omitted) && omitted > 0 ? omitted : 0,
  };
}
