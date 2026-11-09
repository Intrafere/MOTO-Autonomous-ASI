import React, { useEffect, useMemo, useState } from 'react';
import { downloadTextFile } from '../../utils/downloadHelpers';
import { getNoveltyGroupAnchorIds, getProofTranslationParts } from '../../utils/proofPresentation';
import useTranslatedDocuments from '../../hooks/useTranslatedDocuments';
import TranslationLanguageSwitch from '../translation/TranslationLanguageSwitch';
import TranslationPendingNotice from '../translation/TranslationPendingNotice';
import '../autonomous/MathematicalProofs.css';

function formatDate(isoString) {
  if (!isoString) return 'Unknown';
  try {
    return new Date(isoString).toLocaleString();
  } catch {
    return isoString;
  }
}

function truncate(text, maxLength = 220) {
  if (!text) return '';
  return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
}

function getProofBadge(proof) {
  const tier = proof.novelty_tier;
  if (tier === 'major_mathematical_discovery') {
    return { cardClass: 'platinum', badgeClass: 'platinum', label: 'Major Mathematical Discovery' };
  }
  if (tier === 'mathematical_discovery') {
    return { cardClass: 'gold', badgeClass: 'gold', label: 'Minor Mathematical Discovery' };
  }
  if (tier === 'novel_variant') {
    return { cardClass: 'silver', badgeClass: 'silver', label: 'Novel Reformulation' };
  }
  if (tier === 'novel_formulation') {
    return { cardClass: 'bronze', badgeClass: 'bronze', label: 'Novel Formalization' };
  }
  if (proof.novel) {
    return { cardClass: 'gold', badgeClass: 'gold', label: 'Novel Proof' };
  }
  return { cardClass: 'known', badgeClass: 'known', label: 'Known Proof' };
}

function getStatusLabel(status) {
  if (!status?.session_id) return 'No Proof Solver Session Loaded';
  if (status.phase === 'verified') return 'Final Submission Verified';
  if (status.is_running) return `Proof Solver Running: ${status.phase || 'running'}`;
  return `Proof Solver ${status.phase || 'idle'}`;
}

function formatSolverName(solver) {
  return String(solver || 'Proof Solver').replace(/^LeanOJ\b/, 'Proof Solver');
}

function isStatementThePrompt(proof) {
  const prompt = String(proof.user_prompt || '').trim();
  return Boolean(prompt && String(proof.theorem_statement || '').trim() === prompt);
}

function getCurrentProofDisplay(proof, translatedStatement = null) {
  const prompt = String(proof.user_prompt || '').trim();
  const statement = String(translatedStatement ?? proof.theorem_statement ?? '').trim();
  const sourceTitle = String(proof.source_title || '').trim();
  const sourceIsPrompt = Boolean(prompt && sourceTitle === prompt);
  return {
    title: isStatementThePrompt(proof)
      ? (proof.theorem_name || (proof.proof_kind === 'final' ? 'Final verified submission' : 'Verified proof'))
      : (statement || proof.theorem_name || proof.proof_id),
    summary: sourceIsPrompt ? '' : sourceTitle,
  };
}

// Prose fields Auto Top-p Translate may translate here; Lean code, ids, theorem
// names, and metadata always stay original. A statement that repeats the run
// prompt is never shown, so it is never sent for translation either.
const LEANOJ_TRANSLATION_FIELDS = ['theorem_statement', 'novelty_reasoning', 'novelty_group_claim'];

function getProofTranslationKey(proof) {
  return `proof:leanoj:${proof.library_id || proof.proof_id}`;
}

function getLeanOJProofTranslationParts(proof) {
  const fields = isStatementThePrompt(proof)
    ? LEANOJ_TRANSLATION_FIELDS.filter((field) => field !== 'theorem_statement')
    : LEANOJ_TRANSLATION_FIELDS;
  return getProofTranslationParts(proof, fields);
}

export default function LeanOJMathematicalProofs({ api, status, refreshToken = 0 }) {
  const [proofs, setProofs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [filter, setFilter] = useState('all');
  const [expandedProofId, setExpandedProofId] = useState(null);

  const loadProofs = async () => {
    try {
      setLoading(true);
      setError('');
      const response = await api.getProofs();
      setProofs(response.proofs || []);
    } catch (err) {
      if (err.status === 404) {
        setProofs([]);
        setError('');
        return;
      }
      setError(err.message || 'Failed to load Proof Solver proofs');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadProofs();
  }, [refreshToken]);

  const counts = useMemo(() => ({
    total: proofs.length,
    final: proofs.filter((proof) => proof.proof_kind === 'final').length,
    subproof: proofs.filter((proof) => proof.proof_kind === 'subproof').length,
    novel: proofs.filter((proof) => proof.novel).length,
    majorDiscovery: proofs.filter((proof) => proof.novelty_tier === 'major_mathematical_discovery').length,
    discovery: proofs.filter((proof) => proof.novelty_tier === 'mathematical_discovery').length,
    variant: proofs.filter((proof) => proof.novelty_tier === 'novel_variant').length,
    formulation: proofs.filter((proof) => proof.novelty_tier === 'novel_formulation').length,
    known: proofs.filter((proof) => !proof.novel).length,
  }), [proofs]);

  const visibleProofs = useMemo(() => {
    if (filter === 'novel') {
      return proofs.filter((proof) => proof.novel);
    }
    if (filter === 'major_mathematical_discovery' || filter === 'mathematical_discovery' || filter === 'novel_variant' || filter === 'novel_formulation') {
      return proofs.filter((proof) => proof.novelty_tier === filter);
    }
    if (filter === 'known') {
      return proofs.filter((proof) => !proof.novel);
    }
    if (filter === 'final') {
      return proofs.filter((proof) => proof.proof_kind === 'final');
    }
    if (filter === 'subproof') {
      return proofs.filter((proof) => proof.proof_kind === 'subproof');
    }
    return proofs;
  }, [filter, proofs]);
  const translationDocuments = useMemo(
    () => proofs.map((proof) => ({
      docKey: getProofTranslationKey(proof),
      kind: 'proof',
      title: proof.theorem_name || proof.proof_id,
      profile: 'fields',
      parts: getLeanOJProofTranslationParts(proof),
    })),
    [proofs]
  );
  const translation = useTranslatedDocuments(translationDocuments);
  // Translated prose when this proof's translation is ready; otherwise the original.
  const getDisplayedProofText = (proof, field) => {
    const translated = translation.byKey[getProofTranslationKey(proof)]?.parts?.[field];
    return typeof translated === 'string' ? translated : proof[field];
  };
  const handleDownloadLean = (proof) => {
    if (!proof.lean_code) return;
    downloadTextFile(proof.lean_code, `${proof.theorem_name || proof.proof_id}.lean`);
  };
  const handleDownloadNoveltyGroup = async (proof) => {
    try {
      const anchorId = getNoveltyGroupAnchorIds(proof)[0] || proof.proof_id;
      const group = await api.getProofNoveltyGroup(anchorId);
      downloadTextFile(
        JSON.stringify(group, null, 2),
        `${proof.shared_proof_id || proof.proof_id}_novelty_group.json`,
        'application/json'
      );
    } catch (err) {
      setError(`Failed to download novelty-ranked proof group: ${err.message}`);
    }
  };

  return (
    <div className="math-proofs-view">
      <div className="math-proofs-header">
        <div>
          <h2>Mathematical Proofs</h2>
          <p>
            Verified Lean 4 proofs from the active Proof Solver run, including brainstorm proof fragments and the final solved submission.
          </p>
        </div>

        <div className="math-proofs-status-group">
          <span className={`math-proofs-status ${status?.phase === 'verified' ? 'ready' : 'pending'}`}>
            {getStatusLabel(status)}
          </span>
          <span className="math-proofs-count">
            {counts.novel} novel / {counts.total} total
          </span>
          <button className="math-proofs-refresh" onClick={loadProofs}>
            Refresh
          </button>
        </div>
      </div>

      <div className="math-proofs-toolbar">
        <div className="math-proofs-filters">
          <button
            className={`math-proofs-filter ${filter === 'novel' ? 'active' : ''}`}
            onClick={() => setFilter('novel')}
          >
            All Novel ({counts.novel})
          </button>
          <button
            className={`math-proofs-filter math-proofs-filter--platinum ${filter === 'major_mathematical_discovery' ? 'active' : ''}`}
            onClick={() => setFilter('major_mathematical_discovery')}
          >
            Major Discovery ({counts.majorDiscovery})
          </button>
          <button
            className={`math-proofs-filter math-proofs-filter--gold ${filter === 'mathematical_discovery' ? 'active' : ''}`}
            onClick={() => setFilter('mathematical_discovery')}
          >
            Minor Mathematical Discovery ({counts.discovery})
          </button>
          <button
            className={`math-proofs-filter math-proofs-filter--silver ${filter === 'novel_variant' ? 'active' : ''}`}
            onClick={() => setFilter('novel_variant')}
          >
            Reformulation ({counts.variant})
          </button>
          <button
            className={`math-proofs-filter math-proofs-filter--bronze ${filter === 'novel_formulation' ? 'active' : ''}`}
            onClick={() => setFilter('novel_formulation')}
          >
            Formalization ({counts.formulation})
          </button>
          <button
            className={`math-proofs-filter ${filter === 'known' ? 'active' : ''}`}
            onClick={() => setFilter('known')}
          >
            Known ({counts.known})
          </button>
          <button
            className={`math-proofs-filter ${filter === 'all' ? 'active' : ''}`}
            onClick={() => setFilter('all')}
          >
            All Verified ({counts.total})
          </button>
          <button
            className={`math-proofs-filter math-proofs-filter--gold ${filter === 'final' ? 'active' : ''}`}
            onClick={() => setFilter('final')}
          >
            Final Verified Submissions ({counts.final})
          </button>
          <button
            className={`math-proofs-filter math-proofs-filter--silver ${filter === 'subproof' ? 'active' : ''}`}
            onClick={() => setFilter('subproof')}
          >
            Verified Proof Fragments ({counts.subproof})
          </button>
        </div>
      </div>

      {loading && <div className="math-proofs-empty">Loading Proof Solver proofs...</div>}
      {!loading && error && <div className="math-proofs-error">{error}</div>}

      {!loading && !error && visibleProofs.length === 0 && (
        <div className="math-proofs-empty">
          {proofs.length === 0
            ? 'No Proof Solver proofs yet. This tab will populate as brainstorm proof fragments and final submissions pass Lean 4.'
            : 'No Proof Solver proofs match the selected filter.'}
        </div>
      )}

      {!loading && !error && visibleProofs.length > 0 && (
        <div className="math-proofs-list">
          {visibleProofs.map((proof) => {
            const isExpanded = expandedProofId === proof.library_id;
            const badge = getProofBadge(proof);
            const display = getCurrentProofDisplay(proof, getDisplayedProofText(proof, 'theorem_statement'));
            const noveltyReasoning = getDisplayedProofText(proof, 'novelty_reasoning');
            const noveltyGroupClaim = getDisplayedProofText(proof, 'novelty_group_claim');
            // Each proof has its own language choice; the default comes from the translation window.
            const proofTranslation = translation.forKey(getProofTranslationKey(proof));
            return (
              <article key={proof.library_id} className={`math-proof-card ${badge.cardClass}`}>
                <div className="math-proof-card-header">
                  <div>
                    <div className="math-proof-card-topline">
                      <span className={`math-proof-badge ${badge.badgeClass}`}>
                        {badge.label}
                      </span>
                      <span className="math-proof-source">
                        {proof.session_id}
                      </span>
                    </div>
                    <h3>{display.title}</h3>
                    <p className="math-proof-summary">
                      {truncate(display.summary || 'Lean 4 verified this Proof Solver proof.')}
                    </p>
                  </div>

                  <div className="math-proof-card-actions">
                    <TranslationLanguageSwitch translation={proofTranslation} className="math-proof-language" />
                    <button
                      type="button"
                      className="math-proof-download math-proof-download--compact"
                      onClick={() => handleDownloadLean(proof)}
                      disabled={!proof.lean_code}
                    >
                      Download .lean
                    </button>
                    {getNoveltyGroupAnchorIds(proof).length > 0 && (
                      <button
                        type="button"
                        className="math-proof-download math-proof-download--compact"
                        onClick={() => handleDownloadNoveltyGroup(proof)}
                      >
                        Download Ranked Group (JSON)
                      </button>
                    )}
                    <button
                      className="math-proof-expand"
                      onClick={() => setExpandedProofId(isExpanded ? null : proof.library_id)}
                    >
                      {isExpanded ? 'Hide Details' : 'View Details'}
                    </button>
                  </div>
                </div>

                <div className="math-proof-meta">
                  <span>Solver: {formatSolverName(proof.solver)}</span>
                  <span>Attempts: {proof.attempt_count || 0}</span>
                  <span>Verified: {formatDate(proof.created_at)}</span>
                </div>
                {isExpanded && <TranslationPendingNotice translation={proofTranslation} />}

                {isExpanded && (
                  <div className="math-proof-details">
                    <div className="math-proof-actions">
                      <button
                        type="button"
                        className="math-proof-download"
                        onClick={() => handleDownloadLean(proof)}
                        disabled={!proof.lean_code}
                      >
                        Download .lean
                      </button>
                      {getNoveltyGroupAnchorIds(proof).length > 0 && (
                        <button
                          type="button"
                          className="math-proof-download"
                          onClick={() => handleDownloadNoveltyGroup(proof)}
                        >
                          Download Ranked Group (JSON)
                        </button>
                      )}
                    </div>

                    <div className="math-proof-detail-block">
                      <strong>Proof Work</strong>
                      <div>{proof.theorem_name || proof.proof_id}</div>
                    </div>

                    {proof.shared_proof_id && (
                      <div className="math-proof-detail-block">
                        <strong>Shared Proof ID</strong>
                        <div>{proof.shared_proof_id}</div>
                      </div>
                    )}

                    {noveltyReasoning && (
                      <div className="math-proof-detail-block">
                        <strong>Novelty Reasoning</strong>
                        <div>{noveltyReasoning}</div>
                      </div>
                    )}

                    <div className="math-proof-detail-block">
                      <strong>Discovery Ranking Scope</strong>
                      <div>
                        {proof.novelty_scope === 'group'
                          ? 'Finite verified proof group'
                          : 'Accepted Lean statement alone'}
                      </div>
                      {noveltyGroupClaim && <div>{noveltyGroupClaim}</div>}
                      {proof.novelty_scope === 'group' && proof.novelty_group_proof_ids?.length > 0 && (
                        <div>Ranked with: {proof.novelty_group_proof_ids.join(', ')}</div>
                      )}
                    </div>

                    <div className="math-proof-detail-block">
                      <strong>Proof Solver Proof Kind</strong>
                      <div>{proof.proof_kind === 'final' ? 'Final verified submission' : 'Verified proof fragment'}</div>
                    </div>

                    {proof.role && (
                      <div className="math-proof-detail-block">
                        <strong>Proof Fragment Role</strong>
                        <div>{proof.role}</div>
                      </div>
                    )}

                    <div className="math-proof-detail-block">
                      <strong>Lean 4 Code</strong>
                      <pre className="math-proof-code">{proof.lean_code}</pre>
                    </div>
                  </div>
                )}
              </article>
            );
          })}
        </div>
      )}
    </div>
  );
}
