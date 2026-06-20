import React, { useState, useEffect, useMemo } from 'react';
import { autonomousAPI, proofSearchAPI } from '../../services/api';
import { buildResearchRunGroups, formatRunPromptPreview } from '../../utils/researchRunHistory';
import { downloadTextFile } from '../../utils/downloadHelpers';
import './FinalAnswerLibrary.css';
import './ProofLibrary.css';

function formatDate(isoString) {
  if (!isoString) return 'N/A';
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

function getTierBadge(proof) {
  const tier = proof.novelty_tier;
  if (tier === 'major_mathematical_discovery') {
    return { cssClass: 'proof-badge--platinum', label: 'Major Mathematical Discovery' };
  }
  if (tier === 'mathematical_discovery') {
    return { cssClass: 'proof-badge--gold', label: 'Minor Mathematical Discovery' };
  }
  if (tier === 'novel_variant') {
    return { cssClass: 'proof-badge--silver', label: 'Novel Reformulation' };
  }
  if (tier === 'novel_formulation') {
    return { cssClass: 'proof-badge--bronze', label: 'Novel Formalization' };
  }
  if (proof.novel) {
    return { cssClass: 'proof-badge--gold', label: 'Novel' };
  }
  return { cssClass: 'proof-badge--known', label: 'Known' };
}

function getCardClass(proof) {
  const tier = proof.novelty_tier;
  if (tier === 'major_mathematical_discovery') return 'proof-card--platinum';
  if (tier === 'mathematical_discovery') return 'proof-card--gold';
  if (tier === 'novel_variant') return 'proof-card--silver';
  if (tier === 'novel_formulation') return 'proof-card--bronze';
  if (proof.novel) return 'proof-card--novel';
  return 'proof-card--known';
}

const PROOF_SEARCH_CORPORA = [
  { id: 'moto', label: 'MOTO Autonomous' },
  { id: 'manual', label: 'MOTO Manual' },
  { id: 'leanoj', label: 'LeanOJ' },
  { id: 'syntheticlib4', label: 'SyntheticLib4' },
];

const PROOF_SEARCH_OPTIONS = [
  { id: 'verified_only', label: 'Verified only' },
  { id: 'include_partial', label: 'Include partial artifacts' },
  { id: 'include_failed', label: 'Include failed attempts' },
];

export default function ProofLibrary({
  proofScope = 'autonomous',
  title = 'Proof Library',
  description = 'All verified mathematical proofs generated across research sessions.',
}) {
  const [proofs, setProofs] = useState([]);
  const [sessionsResponse, setSessionsResponse] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedId, setExpandedId] = useState(null);
  const [expandedProof, setExpandedProof] = useState(null);
  const [loadingContentId, setLoadingContentId] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterNovelty, setFilterNovelty] = useState('novel');
  const [proofSearchOverview, setProofSearchOverview] = useState(null);
  const [proofSearchCorpora, setProofSearchCorpora] = useState(['moto', 'manual', 'leanoj', 'syntheticlib4']);
  const [proofSearchVerifiedOnly, setProofSearchVerifiedOnly] = useState(true);
  const [proofSearchIncludePartial, setProofSearchIncludePartial] = useState(false);
  const [proofSearchIncludeFailed, setProofSearchIncludeFailed] = useState(false);
  const [proofSearchResults, setProofSearchResults] = useState([]);
  const [proofSearchMessage, setProofSearchMessage] = useState('');
  const [proofSearchLoading, setProofSearchLoading] = useState(false);
  const [proofSearchExpandedId, setProofSearchExpandedId] = useState(null);
  const [proofSearchExpandedRecord, setProofSearchExpandedRecord] = useState(null);

  const loadProofLibrary = async () => {
    try {
      setLoading(true);
      setError(null);

      const novelOnly = filterNovelty === 'novel';
      const [proofsResult, sessionsResult] = await Promise.allSettled([
        autonomousAPI.getProofLibrary(novelOnly, proofScope),
        proofScope === 'manual' ? Promise.resolve(null) : autonomousAPI.getSessions(),
      ]);

      if (proofsResult.status !== 'fulfilled') {
        throw proofsResult.reason;
      }

      setProofs(proofsResult.value.proofs || []);

      if (sessionsResult.status === 'fulfilled') {
        setSessionsResponse(sessionsResult.value);
      } else {
        setSessionsResponse(null);
      }
    } catch (err) {
      setError(err.message || 'Failed to load proof library');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadProofLibrary();
  }, [filterNovelty, proofScope]);

  useEffect(() => {
    let cancelled = false;
    proofSearchAPI.getOverview()
      .then((overview) => {
        if (!cancelled) {
          setProofSearchOverview(overview);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setProofSearchMessage(err.message || 'Proof-search overview unavailable');
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const filteredProofs = useMemo(() => {
    if (!searchTerm.trim()) return proofs;
    const lower = searchTerm.toLowerCase();
    return proofs.filter(
      (p) =>
        (p.theorem_name || '').toLowerCase().includes(lower) ||
        (p.theorem_statement || '').toLowerCase().includes(lower) ||
        (p.source_title || '').toLowerCase().includes(lower) ||
        (p.user_prompt || '').toLowerCase().includes(lower) ||
        (p.novelty_reasoning || '').toLowerCase().includes(lower)
    );
  }, [proofs, searchTerm]);

  const runGroups = useMemo(() => {
    const pseudoPapers = filteredProofs.map((p) => ({
      session_id: p.session_id,
      paper_id: p.proof_id,
      created_at: p.created_at,
      user_prompt: p.user_prompt,
    }));

    return buildResearchRunGroups({
      sessionsResponse,
      stage2Papers: pseudoPapers,
      stage3Answers: [],
    });
  }, [filteredProofs, sessionsResponse]);

  const proofsBySession = useMemo(() => {
    const map = new Map();
    for (const proof of filteredProofs) {
      const sid = proof.session_id || 'unknown';
      if (!map.has(sid)) map.set(sid, []);
      map.get(sid).push(proof);
    }
    return map;
  }, [filteredProofs]);

  const handleExpand = async (proof) => {
    const id = proof.library_id || proof.proof_id;
    if (expandedId === id) {
      setExpandedId(null);
      setExpandedProof(null);
      return;
    }

    setExpandedId(id);
    setLoadingContentId(id);

    try {
      const fullProof = await autonomousAPI.getLibraryProof(proof.session_id, proof.proof_id, proofScope);
      setExpandedProof(fullProof);
    } catch {
      setExpandedProof(proof);
    } finally {
      setLoadingContentId(null);
    }
  };

  const handleDownloadLean = async (proof, event) => {
    event?.stopPropagation();

    let proofForDownload = proof;
    let leanCode = proof.lean_code || '';
    if (!leanCode && proof.session_id && proof.proof_id) {
      try {
        proofForDownload = await autonomousAPI.getLibraryProof(proof.session_id, proof.proof_id, proofScope);
        leanCode = proofForDownload.lean_code || '';
      } catch {
        return;
      }
    }

    if (!leanCode) return;
    const filename = `${proofForDownload.theorem_name || proof.theorem_name || proof.proof_id}.lean`;
    downloadTextFile(leanCode, filename);
  };

  const toggleProofSearchCorpus = (corpusId) => {
    setProofSearchCorpora((previous) => {
      if (previous.includes(corpusId)) {
        const next = previous.filter((id) => id !== corpusId);
        return next.length > 0 ? next : previous;
      }
      return [...previous, corpusId];
    });
  };

  const handleUnifiedProofSearch = async (event) => {
    event?.preventDefault();
    setProofSearchLoading(true);
    setProofSearchMessage('');
    setProofSearchExpandedId(null);
    setProofSearchExpandedRecord(null);
    try {
      const effectiveVerifiedOnly = proofSearchVerifiedOnly
        && !proofSearchIncludePartial
        && !proofSearchIncludeFailed;
      const response = await proofSearchAPI.search({
        query: '',
        corpora: proofSearchCorpora,
        verified_only: effectiveVerifiedOnly,
        include_partial: proofSearchIncludePartial,
        include_failed: proofSearchIncludeFailed,
        dependency_names: [],
        novelty_filters: [],
        module_filters: [],
        source_filters: [],
        limit: 7,
        hydrate_lean_code: false,
      });
      setProofSearchResults(response.results || []);
      setProofSearchMessage(response.weak_result_warning || response.ranking_notes || '');
    } catch (err) {
      setProofSearchResults([]);
      setProofSearchMessage(err.message || 'Proof search failed');
    } finally {
      setProofSearchLoading(false);
    }
  };

  const handleProofSearchReindex = async () => {
    setProofSearchLoading(true);
    setProofSearchMessage('');
    try {
      const response = await proofSearchAPI.reindex();
      setProofSearchOverview(response.overview || null);
      setProofSearchMessage('Unified proof-search index rebuilt.');
    } catch (err) {
      setProofSearchMessage(err.message || 'Proof-search reindex failed');
    } finally {
      setProofSearchLoading(false);
    }
  };

  const handleExpandProofSearchResult = async (record) => {
    const id = record.search_id || `${record.corpus}:${record.proof_id}`;
    if (proofSearchExpandedId === id) {
      setProofSearchExpandedId(null);
      setProofSearchExpandedRecord(null);
      return;
    }
    setProofSearchExpandedId(id);
    setProofSearchExpandedRecord(record);
    if (record.lean_code) {
      return;
    }
    try {
      const hydrated = await proofSearchAPI.getProof(record.corpus, record.proof_id, {
        sessionId: record.session_id || null,
      });
      setProofSearchExpandedRecord(hydrated);
    } catch {
      setProofSearchExpandedRecord(record);
    }
  };

  const novelCount = proofs.filter((p) => p.novel).length;
  const totalCount = proofs.length;

  if (loading) {
    return (
      <div className="final-answer-library">
        <div className="library-loading">
          <span className="library-loading__icon">&#x21BB;</span>
          <span className="library-loading__text">Loading proof library...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="final-answer-library">
        <div className="error-message">
          <span>&#x26A0;</span>
          <p>{error}</p>
          <button className="retry-button" onClick={loadProofLibrary}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="final-answer-library proof-library">
      <div className="library-header">
        <h2>{title}</h2>
        <p>{description}</p>
        <div className="library-stats">
          {filterNovelty === 'novel' ? (
            <span className="stat-badge">{novelCount} Novel Proof{novelCount !== 1 ? 's' : ''}</span>
          ) : (
            <>
              <span className="stat-badge">{totalCount} Total Proof{totalCount !== 1 ? 's' : ''}</span>
              <span className="stat-badge">{novelCount} Novel</span>
              <span className="stat-badge">{totalCount - novelCount} Known</span>
            </>
          )}
        </div>
      </div>

      <section className="proof-search-panel">
        <div className="proof-search-panel__header">
          <div>
            <h3>Unified Proof Search</h3>
            <p>
              Search MOTO proof history, LeanOJ artifacts, and SyntheticLib4 snapshot records through the backend index. Results are capped at 7 combined proofs.
            </p>
          </div>
          <div className="proof-search-panel__stats">
            <span>{proofSearchOverview?.total_records ?? 0} indexed</span>
            <span>{proofSearchOverview?.result_cap ?? 7} result cap</span>
          </div>
        </div>

        <form className="proof-search-form" onSubmit={handleUnifiedProofSearch}>
          <fieldset className="proof-search-checklist">
            <legend>Search Checklist</legend>
            <div className="proof-search-checklist__grid">
              {PROOF_SEARCH_CORPORA.map((corpus) => (
                <label key={corpus.id} className="proof-search-checkbox">
                  <input
                    type="checkbox"
                    checked={proofSearchCorpora.includes(corpus.id)}
                    onChange={() => toggleProofSearchCorpus(corpus.id)}
                  />
                  <span>{corpus.label}</span>
                </label>
              ))}
              {PROOF_SEARCH_OPTIONS.map((option) => {
                if (option.id === 'verified_only') {
                  return (
                    <label key={option.id} className="proof-search-checkbox">
                      <input
                        type="checkbox"
                        checked={proofSearchVerifiedOnly}
                        onChange={(event) => setProofSearchVerifiedOnly(event.target.checked)}
                      />
                      <span>{option.label}</span>
                    </label>
                  );
                }
                if (option.id === 'include_partial') {
                  return (
                    <label key={option.id} className="proof-search-checkbox">
                      <input
                        type="checkbox"
                        checked={proofSearchIncludePartial}
                        onChange={(event) => {
                          setProofSearchIncludePartial(event.target.checked);
                          if (event.target.checked) setProofSearchVerifiedOnly(false);
                        }}
                      />
                      <span>{option.label}</span>
                    </label>
                  );
                }
                return (
                  <label key={option.id} className="proof-search-checkbox">
                    <input
                      type="checkbox"
                      checked={proofSearchIncludeFailed}
                      onChange={(event) => {
                        setProofSearchIncludeFailed(event.target.checked);
                        if (event.target.checked) setProofSearchVerifiedOnly(false);
                      }}
                    />
                    <span>{option.label}</span>
                  </label>
                );
              })}
            </div>
            {(proofSearchIncludePartial || proofSearchIncludeFailed) && proofSearchVerifiedOnly && (
              <p className="proof-search-checklist__note">
                Partial or failed artifact searches automatically disable verified-only filtering for the request.
              </p>
            )}
          </fieldset>
          <div className="proof-search-actions">
            <button type="submit" className="refresh-button" disabled={proofSearchLoading}>
              {proofSearchLoading ? 'Searching...' : 'Search Proofs'}
            </button>
            <button
              type="button"
              className="refresh-button refresh-button--secondary"
              onClick={handleProofSearchReindex}
              disabled={proofSearchLoading}
            >
              Rebuild Index
            </button>
          </div>
        </form>

        {proofSearchMessage && (
          <div className="proof-search-message">
            {proofSearchMessage}
          </div>
        )}

        {proofSearchResults.length > 0 && (
          <div className="proof-search-results">
            {proofSearchResults.map((record) => {
              const id = record.search_id || `${record.corpus}:${record.proof_id}`;
              const isExpanded = proofSearchExpandedId === id;
              const expandedRecord = isExpanded ? (proofSearchExpandedRecord || record) : record;
              return (
                <div key={id} className="proof-search-result-card">
                  <button
                    type="button"
                    className="proof-search-result-card__summary"
                    onClick={() => handleExpandProofSearchResult(record)}
                  >
                    <div>
                      <h4>{record.theorem_name || record.display_title || record.proof_id}</h4>
                      <p>{truncate(record.theorem_statement, 260)}</p>
                    </div>
                    <span>{isExpanded ? '\u25B2' : '\u25BC'}</span>
                  </button>
                  <div className="proof-search-result-card__meta">
                    <span>{record.corpus}</span>
                    <span>{record.corpus_scope || record.release_id || 'current'}</span>
                    <span>{record.source_kind}</span>
                    {record.lean_code_hash && <span>Code hash: {record.lean_code_hash}</span>}
                  </div>
                  {isExpanded && (
                    <div className="proof-expanded-content proof-search-expanded">
                      <div className="proof-detail-section">
                        <h4>Description</h4>
                        <p>{expandedRecord.proof_description || expandedRecord.formal_sketch || 'No description available.'}</p>
                      </div>
                      <div className="proof-detail-section">
                        <h4>Imports</h4>
                        <p>{(expandedRecord.imports || []).join(', ') || 'None listed'}</p>
                      </div>
                      <div className="proof-detail-section">
                        <h4>Dependencies</h4>
                        <p>{(expandedRecord.dependency_names || []).join(', ') || 'None listed'}</p>
                      </div>
                      <div className="proof-detail-section">
                        <h4>Hashes</h4>
                        <p>
                          Statement: {expandedRecord.theorem_statement_hash || 'none'}
                          <br />
                          Lean code: {expandedRecord.lean_code_hash || 'none'}
                        </p>
                      </div>
                      {expandedRecord.lean_code ? (
                        <div className="proof-detail-section">
                          <h4>Lean 4 Source Code</h4>
                          <pre className="proof-code-block proof-lean-code">
                            {expandedRecord.lean_code}
                          </pre>
                        </div>
                      ) : (
                        <div className="proof-search-message">
                          This result is metadata-only. Hydration did not return full Lean code for this record.
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </section>

      <div className="library-controls">
        <input
          className="search-input"
          type="text"
          placeholder="Search by theorem name, statement, source, or research question..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
        <div className="filter-buttons">
          <button
            className={filterNovelty === 'novel' ? 'active' : ''}
            onClick={() => setFilterNovelty('novel')}
          >
            Novel Only
          </button>
          <button
            className={filterNovelty === 'all' ? 'active' : ''}
            onClick={() => setFilterNovelty('all')}
          >
            All Proofs
          </button>
        </div>
      </div>

      {filteredProofs.length === 0 ? (
        <div className="fal-empty-state">
          <span className="empty-icon">&#x1F9EE;</span>
          <h3>No Proofs Found</h3>
          <p>
            {proofs.length === 0
              ? (proofScope === 'manual'
                  ? 'No archived manual proof runs yet. Clear a manual run after generating proofs to move them into history.'
                  : 'No verified proofs have been generated yet. Run autonomous research with Lean 4 enabled to generate proofs.')
              : 'No proofs match your search criteria.'}
          </p>
        </div>
      ) : runGroups.length > 0 ? (
        <div className="run-history-groups">
          {runGroups.map((group) => {
            const sessionProofs = proofsBySession.get(group.sessionId) || [];
            if (sessionProofs.length === 0) return null;

            return (
              <div key={group.sessionId} className="run-history-group">
                <div className="run-history-group-header">
                  <div className="run-history-group-heading">
                    <h3 className="run-history-group-title">
                      {formatRunPromptPreview(group.userPrompt)}
                    </h3>
                    <p className="run-history-group-subtitle">
                      {sessionProofs.length} proof{sessionProofs.length !== 1 ? 's' : ''}
                      {group.createdAt && ` \u00B7 ${formatDate(group.createdAt)}`}
                    </p>
                  </div>
                  <div className="run-history-group-badges">
                    {group.isCurrent && (
                      <span className="run-history-group-badge run-history-group-badge--current">
                        Current Session
                      </span>
                    )}
                    {group.isLegacy && (
                      <span className="run-history-group-badge">Legacy</span>
                    )}
                  </div>
                </div>

                <div className="run-history-group-body">
                  <div className="answer-list">
                    {sessionProofs.map((proof) => {
                      const id = proof.library_id || proof.proof_id;
                      const isExpanded = expandedId === id;

                      return (
                        <div
                          key={id}
                          className={`answer-card proof-card ${isExpanded ? 'expanded' : ''} ${getCardClass(proof)}`}
                        >
                          <div
                            className="answer-header"
                            onClick={() => handleExpand(proof)}
                          >
                            <div className="answer-title-row">
                              <h4 className="answer-title proof-title">
                                {proof.theorem_name || proof.proof_id}
                              </h4>
                              <div className="proof-card-actions">
                                <button
                                  type="button"
                                  className="proof-header-download"
                                  onClick={(event) => handleDownloadLean(proof, event)}
                                >
                                  Download .lean
                                </button>
                                <button className="expand-button">
                                  {isExpanded ? '\u25B2' : '\u25BC'}
                                </button>
                              </div>
                            </div>

                            <div className="answer-metadata">
                              <span
                                className={`format-badge ${getTierBadge(proof).cssClass}`}
                              >
                                {getTierBadge(proof).label}
                              </span>
                              <span className="word-count">
                                {proof.solver || 'Lean 4'}
                              </span>
                              <span className="word-count">
                                {proof.attempt_count || 1} attempt{(proof.attempt_count || 1) !== 1 ? 's' : ''}
                              </span>
                              <span className="word-count">
                                Source: {proof.source_type === 'brainstorm' ? 'Brainstorm' : 'Paper'}
                              </span>
                            </div>

                            <p className="proof-statement">
                              {truncate(proof.theorem_statement, 300)}
                            </p>

                            {proof.source_title && (
                              <p className="answer-prompt">
                                <strong>Source:</strong> {proof.source_title}
                              </p>
                            )}

                            <div className="answer-footer-info">
                              <span className="completion-date">
                                Verified: {formatDate(proof.created_at)}
                              </span>
                            </div>
                          </div>

                          {isExpanded && (
                            <div className="answer-content">
                              {loadingContentId === id ? (
                                <div className="library-loading" style={{ padding: '20px' }}>
                                  <span className="library-loading__icon">&#x21BB;</span>
                                  <span className="library-loading__text">Loading proof details...</span>
                                </div>
                              ) : expandedProof ? (
                                <div className="proof-expanded-content">
                                  <div className="proof-detail-section">
                                    <h4>Theorem Statement</h4>
                                    <pre className="proof-code-block">
                                      {expandedProof.theorem_statement}
                                    </pre>
                                  </div>

                                  {expandedProof.novelty_reasoning && (
                                    <div className="proof-detail-section">
                                      <h4>Novelty Assessment</h4>
                                      <p>{expandedProof.novelty_reasoning}</p>
                                    </div>
                                  )}

                                  {expandedProof.verification_notes && (
                                    <div className="proof-detail-section">
                                      <h4>Verification Notes</h4>
                                      <p>{expandedProof.verification_notes}</p>
                                    </div>
                                  )}

                                  {expandedProof.formal_sketch && (
                                    <div className="proof-detail-section">
                                      <h4>Formal Sketch</h4>
                                      <pre className="proof-code-block">
                                        {expandedProof.formal_sketch}
                                      </pre>
                                    </div>
                                  )}

                                  {expandedProof.lean_code && (
                                    <div className="proof-detail-section">
                                      <h4>Lean 4 Source Code</h4>
                                      <pre className="proof-code-block proof-lean-code">
                                        {expandedProof.lean_code}
                                      </pre>
                                    </div>
                                  )}

                                  {expandedProof.dependencies && expandedProof.dependencies.length > 0 && (
                                    <div className="proof-detail-section">
                                      <h4>Dependencies</h4>
                                      <ul className="proof-dependencies-list">
                                        {expandedProof.dependencies.map((dep, i) => (
                                          <li key={i}>
                                            <span className={`proof-dep-kind proof-dep-kind--${dep.kind}`}>
                                              {dep.kind}
                                            </span>
                                            {' '}
                                            {dep.name}
                                            {dep.source_ref ? ` (${dep.source_ref})` : ''}
                                          </li>
                                        ))}
                                      </ul>
                                    </div>
                                  )}

                                  <div className="quick-download-buttons">
                                    {expandedProof.lean_code && (
                                      <button
                                        className="quick-download-raw"
                                        onClick={() => handleDownloadLean(expandedProof)}
                                      >
                                        Download .lean
                                      </button>
                                    )}
                                  </div>
                                </div>
                              ) : null}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="answer-list">
          {filteredProofs.map((proof) => {
            const id = proof.library_id || proof.proof_id;
            return (
              <div key={id} className="answer-card proof-card">
                <div className="answer-header" onClick={() => handleExpand(proof)}>
                  <div className="answer-title-row">
                    <h4 className="answer-title">{proof.theorem_name || proof.proof_id}</h4>
                    <button
                      type="button"
                      className="proof-header-download"
                      onClick={(event) => handleDownloadLean(proof, event)}
                    >
                      Download .lean
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      <div className="library-footer">
        <button className="refresh-button" onClick={loadProofLibrary}>
          Refresh Proof Library
        </button>
      </div>
    </div>
  );
}
