export const SYNTHETICLIB_PRODUCT_STATES = Object.freeze({
  COMING_SOON: 'coming_soon',
  DEVELOPER_PREVIEW: 'developer_preview',
  LAUNCHED: 'launched',
});

const KNOWN_STATES = new Set(Object.values(SYNTHETICLIB_PRODUCT_STATES));

export function normalizeSyntheticLibProductState(value) {
  return typeof value === 'string' && KNOWN_STATES.has(value)
    ? value
    : SYNTHETICLIB_PRODUCT_STATES.COMING_SOON;
}

// One presentation policy for every SyntheticLib surface. Proof-trade and the
// developer workspace are desktop-only; the developer workspace stays behind
// browser developer mode even after launch (backend authority is separate).
export function getSyntheticLibPresentation(productState, developerModeEnabled, genericMode = false) {
  const rawStateKnown = typeof productState === 'string' && KNOWN_STATES.has(productState);
  const normalizedState = normalizeSyntheticLibProductState(productState);
  const launched = rawStateKnown && normalizedState === SYNTHETICLIB_PRODUCT_STATES.LAUNCHED;
  const desktop = !genericMode;
  const developerMode = Boolean(developerModeEnabled);
  const privatePreview = rawStateKnown && !launched && developerMode && desktop;
  const showIntegration = launched || privatePreview;

  return {
    productState: normalizedState,
    showIntegration,
    showTrade: desktop && showIntegration,
    showDeveloperWorkspace: rawStateKnown && developerMode && desktop,
    showComingSoon: !showIntegration,
    isPrivatePreview: privatePreview,
    isLaunched: launched,
  };
}

// Shared message for SyntheticLib requests the backend refused for lack of
// explicit local preview authority (browser developer mode cannot grant it).
export function previewErrorMessage(error, action, fallback) {
  if (error?.status === 403) {
    return 'The backend has not granted SyntheticLib local preview authority. '
      + 'Start the desktop backend with MOTO_SYNTHETICLIB_DEVELOPER_PREVIEW_AUTHORIZED=true '
      + `to ${action}. Browser developer mode cannot grant it.`;
  }
  return error?.message || fallback;
}

// Plain-language labels for the proof-trade reason codes users see most often.
// Unknown codes fall back to a readable form; the raw code is always kept beside it.
const TRADE_REASON_LABELS = {
  novelty_trade_mapping_unavailable:
    'SyntheticLib has not published how MOTO novelty ranks map to its trade ranks yet.',
  frozen_syntheticlib_contract_unpublished:
    'SyntheticLib has not published its frozen proof-trade contract yet.',
  source_local_rank_not_novel: 'MOTO ranked this proof as not novel.',
  source_local_rank_unavailable: 'This proof has no recorded MOTO novelty assessment.',
  source_duplicate_overlay_occurrence: 'This proof duplicates one from an earlier run.',
  source_local_duplicate_superseded: 'An earlier identical copy of this proof is used instead.',
  source_artifact_purpose_excluded: 'This proof is a supporting artifact, not a standalone result.',
  source_store_kind_unsupported: 'Proofs from this legacy store cannot be offered.',
  verification_evidence_incomplete: 'This proof lacks complete proof-time verification evidence.',
  compile_receipt_missing: 'This proof has no recorded Lean compile receipt.',
  novelty_receipt_missing: 'This proof has no recorded novelty receipt.',
};

export function describeTradeReasonCode(code) {
  const value = String(code || '');
  if (TRADE_REASON_LABELS[value]) return TRADE_REASON_LABELS[value];
  if (value.startsWith('lean_source_sensitive_value:')) {
    return 'The Lean source contains private data (for example a path, email, or credential).';
  }
  const readable = value.replace(/[:_]+/g, ' ').trim();
  return readable ? readable.charAt(0).toUpperCase() + readable.slice(1) + '.' : '';
}
