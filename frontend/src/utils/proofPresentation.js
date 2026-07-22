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
