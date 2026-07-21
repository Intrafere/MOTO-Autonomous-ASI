import React, { useEffect, useMemo, useRef, useState } from 'react';
import { downloadTextFile } from '../../utils/downloadHelpers';
import {
  getCanonicalProofIdentity,
  getLeanOJProofPresentation,
  sanitizeDomId,
} from '../../utils/proofPresentation';
import { formatRunPromptPreview } from '../../utils/researchRunHistory';
import '../autonomous/FinalAnswerLibrary.css';
import '../autonomous/ProofLibrary.css';

function formatDate(isoString) {
  if (!isoString) return 'N/A';
  try {
    return new Date(isoString).toLocaleString();
  } catch {
    return isoString;
  }
}

function truncate(text, maxLength = 260) {
  if (!text) return '';
  return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
}

function formatSolverName(solver) {
  return String(solver || 'Proof Solver').replace(/^LeanOJ\b/, 'Proof Solver');
}

function getLeanOJProofCardId(proof) {
  return getCanonicalProofIdentity({ ...proof, corpus: proof.corpus || 'leanoj' });
}

function getLeanOJRunKey(proof) {
  return proof.run_id || proof.session_id || `orphan:${proof.proof_id}`;
}

function matchesSelectedLeanOJProof(proof, selectedProofId, selectedSessionId = '', selectedRunId = '') {
  if (!selectedProofId) return false;
  const proofIds = [
    proof.proof_id,
    proof.library_id,
    proof.search_id,
    proof.lean_code_hash,
    proof.theorem_statement_hash,
  ].filter(Boolean).map(String);
  if (!proofIds.includes(String(selectedProofId))) {
    return false;
  }
  if (selectedRunId) return getLeanOJRunKey(proof) === selectedRunId;
  return !selectedSessionId || !proof.session_id || proof.session_id === selectedSessionId;
}

export default function LeanOJProofLibrary({
  api,
  refreshToken = 0,
  selectedProofId = null,
  selectedSessionId = '',
  selectedRunId = '',
}) {
  const [proofs, setProofs] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [expandedId, setExpandedId] = useState(null);
  const [expandedProof, setExpandedProof] = useState(null);
  const [expandedSessions, setExpandedSessions] = useState(() => new Set());
  const [loadingContentId, setLoadingContentId] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterKind, setFilterKind] = useState('all');
  const detailGenerationRef = useRef(0);
  const libraryGenerationRef = useRef(0);

  const loadProofLibrary = async () => {
    const generation = ++libraryGenerationRef.current;
    detailGenerationRef.current += 1;
    setExpandedId(null);
    setExpandedProof(null);
    try {
      setLoading(true);
      setError('');
      const response = await api.getProofLibrary(true);
      if (generation !== libraryGenerationRef.current) return;
      setProofs(response.proofs || []);
      setSessions(response.sessions || []);
    } catch (err) {
      if (generation !== libraryGenerationRef.current) return;
      if (err.status === 404) {
        setProofs([]);
        setSessions([]);
        setError('');
        return;
      }
      setError(err.message || 'Failed to load Proof Solver proof works library');
    } finally {
      if (generation === libraryGenerationRef.current) setLoading(false);
    }
  };

  useEffect(() => {
    loadProofLibrary();
  }, [refreshToken]);

  useEffect(() => () => {
    libraryGenerationRef.current += 1;
    detailGenerationRef.current += 1;
  }, []);

  const filteredProofs = useMemo(() => {
    const lowerSearch = searchTerm.trim().toLowerCase();
    return proofs.filter((proof) => {
      const kindMatches = filterKind === 'all' || proof.proof_kind === filterKind;
      if (!kindMatches) return false;
      if (!lowerSearch) return true;
      return (
        (proof.theorem_name || '').toLowerCase().includes(lowerSearch) ||
        (proof.theorem_statement || '').toLowerCase().includes(lowerSearch) ||
        (proof.source_title || '').toLowerCase().includes(lowerSearch) ||
        (proof.user_prompt || '').toLowerCase().includes(lowerSearch) ||
        (proof.run_id || '').toLowerCase().includes(lowerSearch) ||
        (proof.session_id || '').toLowerCase().includes(lowerSearch)
      );
    });
  }, [filterKind, proofs, searchTerm]);

  const proofsBySession = useMemo(() => {
    const map = new Map();
    for (const proof of filteredProofs) {
      const sessionId = getLeanOJRunKey(proof);
      if (!map.has(sessionId)) map.set(sessionId, []);
      map.get(sessionId).push(proof);
    }
    return map;
  }, [filteredProofs]);

  const visibleSessions = useMemo(() => {
    const metadata = new Map();
    sessions.forEach((session) => {
      if (session.run_id) metadata.set(session.run_id, session);
      if (session.session_id && !metadata.has(session.session_id)) metadata.set(session.session_id, session);
    });
    return Array.from(proofsBySession.keys()).map((runKey) => {
      const matched = metadata.get(runKey);
      return matched ? { ...matched, group_key: runKey } : {
        session_id: proofsBySession.get(runKey)?.[0]?.session_id || '',
        run_id: proofsBySession.get(runKey)?.[0]?.run_id || '',
        group_key: runKey,
        user_prompt: proofsBySession.get(runKey)?.[0]?.user_prompt || 'Legacy Proof Solver run',
        updated_at: proofsBySession.get(runKey)?.[0]?.created_at,
        phase: 'legacy',
      };
    });
  }, [proofsBySession, sessions]);

  const counts = useMemo(() => ({
    total: proofs.length,
    final: proofs.filter((proof) => proof.proof_kind === 'final').length,
    subproof: proofs.filter((proof) => proof.proof_kind === 'subproof').length,
  }), [proofs]);

  const handleExpand = async (proof) => {
    const id = getLeanOJProofCardId(proof);
    if (expandedId === id) {
      detailGenerationRef.current += 1;
      setExpandedId(null);
      setExpandedProof(null);
      return;
    }

    setExpandedId(id);
    setLoadingContentId(id);
    const generation = ++detailGenerationRef.current;
    try {
      const fullProof = await api.getLibraryProof(proof.session_id, proof.proof_id);
      if (generation === detailGenerationRef.current) setExpandedProof(fullProof);
    } catch {
      if (generation === detailGenerationRef.current) setExpandedProof(proof);
    } finally {
      if (generation === detailGenerationRef.current) setLoadingContentId(null);
    }
  };

  useEffect(() => {
    if (!selectedProofId || loading) return;
    const visibleMatch = filteredProofs.find((proof) => matchesSelectedLeanOJProof(proof, selectedProofId, selectedSessionId, selectedRunId));
    if (visibleMatch) return;
    const searchHiddenMatch = proofs.find((proof) => matchesSelectedLeanOJProof(proof, selectedProofId, selectedSessionId, selectedRunId));
    if (searchHiddenMatch && searchTerm) {
      setSearchTerm('');
      return;
    }
    const hiddenKindMatch = proofs.find((proof) => matchesSelectedLeanOJProof(proof, selectedProofId, selectedSessionId, selectedRunId));
    if (hiddenKindMatch && filterKind !== 'all') {
      setFilterKind('all');
    }
  }, [filterKind, filteredProofs, loading, proofs, searchTerm, selectedProofId, selectedRunId, selectedSessionId]);

  useEffect(() => {
    if (!selectedProofId || loading) return;
    const match = filteredProofs.find((proof) => matchesSelectedLeanOJProof(proof, selectedProofId, selectedSessionId, selectedRunId));
    if (!match) return;
    const id = getLeanOJProofCardId(match);
    const sessionKey = getLeanOJRunKey(match);
    setExpandedSessions((previous) => new Set(previous).add(sessionKey));
    setExpandedId(id);
    setLoadingContentId(id);
    const generation = ++detailGenerationRef.current;
    let cancelled = false;
    api.getLibraryProof(match.session_id, match.proof_id)
      .then((fullProof) => {
        if (!cancelled && generation === detailGenerationRef.current) setExpandedProof(fullProof);
      })
      .catch(() => {
        if (!cancelled && generation === detailGenerationRef.current) setExpandedProof(match);
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingContentId(null);
          setTimeout(() => {
            document.getElementById(sanitizeDomId(id, 'leanoj-proof-card'))?.scrollIntoView({ behavior: 'smooth', block: 'center' });
          }, 0);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [api, filteredProofs, loading, selectedProofId, selectedRunId, selectedSessionId]);

  const toggleSession = (sessionId) => {
    setExpandedSessions((previous) => {
      const next = new Set(previous);
      if (next.has(sessionId)) next.delete(sessionId);
      else next.add(sessionId);
      return next;
    });
    detailGenerationRef.current += 1;
    setExpandedId(null);
    setExpandedProof(null);
  };

  const handleDownloadLean = async (proof, event) => {
    event?.stopPropagation();

    let proofForDownload = proof;
    let leanCode = proof.lean_code || '';
    if (!leanCode && proof.session_id && proof.proof_id) {
      try {
        proofForDownload = await api.getLibraryProof(proof.session_id, proof.proof_id);
        leanCode = proofForDownload.lean_code || '';
      } catch {
        return;
      }
    }

    if (!leanCode) return;
    downloadTextFile(leanCode, `${proofForDownload.theorem_name || proof.theorem_name || proof.proof_id}.lean`);
  };

  if (loading) {
    return (
      <div className="final-answer-library proof-library">
        <div className="library-loading">
          <span className="library-loading__icon">&#x21BB;</span>
          <span className="library-loading__text">Loading Proof Solver proof works...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="final-answer-library proof-library">
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
        <h2>Your Completed Proof Works Library</h2>
        <p>
          Browse verified Proof Solver final submissions and brainstorm proof fragments saved across proof-solver sessions.
        </p>
        <div className="library-stats">
          <span className="stat-badge">{counts.total} Proof Work{counts.total !== 1 ? 's' : ''}</span>
          <span className="stat-badge">{counts.final} Final Submission{counts.final !== 1 ? 's' : ''}</span>
          <span className="stat-badge">{counts.subproof} Proof Fragment{counts.subproof !== 1 ? 's' : ''}</span>
        </div>
      </div>

      <div className="library-controls">
        <input
          aria-label="Filter Proof Solver proof works"
          className="search-input"
          type="text"
          placeholder="Search by theorem, problem, session, or Proof Solver source..."
          value={searchTerm}
          onChange={(event) => setSearchTerm(event.target.value)}
        />
        <div className="filter-buttons">
          <button className={filterKind === 'all' ? 'active' : ''} aria-pressed={filterKind === 'all'} onClick={() => setFilterKind('all')}>
            All
          </button>
          <button className={filterKind === 'final' ? 'active' : ''} aria-pressed={filterKind === 'final'} onClick={() => setFilterKind('final')}>
            Final
          </button>
          <button className={filterKind === 'subproof' ? 'active' : ''} aria-pressed={filterKind === 'subproof'} onClick={() => setFilterKind('subproof')}>
            Proof Fragments
          </button>
        </div>
      </div>

      {filteredProofs.length === 0 ? (
        <div className="fal-empty-state">
          <span className="empty-icon">&#x1D7D9;</span>
          <h3>{proofs.length === 0 ? 'No Proof Solver Proofs Yet' : 'No Proof Solver Proof Works Found'}</h3>
          <p>
            {proofs.length === 0
              ? 'No Proof Solver proofs yet. Completed final submissions and verified brainstorm proof fragments will appear here.'
              : 'No proof works match your search criteria.'}
          </p>
        </div>
      ) : (
        <div className="run-history-groups">
          {visibleSessions.map((session) => {
            const groupKey = session.group_key || session.run_id || session.session_id;
            const sessionProofs = proofsBySession.get(groupKey) || [];
            const isSessionExpanded = expandedSessions.has(groupKey);
            const sessionRegionId = sanitizeDomId(groupKey, 'leanoj-proof-session');
            return (
              <div key={groupKey} className={`run-history-group proof-prompt-group ${isSessionExpanded ? 'expanded' : ''}`}>
                <button
                  type="button"
                  className="run-history-group-header proof-prompt-group-header"
                  onClick={() => toggleSession(groupKey)}
                  aria-expanded={isSessionExpanded}
                  aria-controls={sessionRegionId}
                >
                  <div className="run-history-group-heading">
                    <h3 className="run-history-group-title">
                      {formatRunPromptPreview(session.user_prompt || sessionProofs[0]?.user_prompt || 'Legacy Proof Solver run')}
                    </h3>
                    <p className="run-history-group-subtitle">
                      {sessionProofs.length} proof work{sessionProofs.length !== 1 ? 's' : ''} - {formatDate(session.updated_at)}
                    </p>
                  </div>
                  <div className="run-history-group-badges">
                    {session.is_current && (
                      <span className="run-history-group-badge run-history-group-badge--current">
                        Current Session
                      </span>
                    )}
                    {session.phase && (
                      <span className="run-history-group-badge">{session.phase}</span>
                    )}
                    <span className="proof-prompt-group-chevron" aria-hidden="true">
                      {isSessionExpanded ? '▲' : '▼'}
                    </span>
                  </div>
                </button>

                <div id={sessionRegionId} className="run-history-group-body" role="region" aria-label={`Proof works for ${formatRunPromptPreview(session.user_prompt || sessionProofs[0]?.user_prompt || groupKey)}`} hidden={!isSessionExpanded}>
                  {isSessionExpanded && (
                  <div className="answer-list">
                    {sessionProofs.map((proof) => {
                      const id = getLeanOJProofCardId(proof);
                      const cardDomId = sanitizeDomId(id, 'leanoj-proof-card');
                      const detailsDomId = sanitizeDomId(id, 'leanoj-proof-details');
                      const isExpanded = expandedId === id;
                      const badge = getLeanOJProofPresentation(proof);

                      return (
                        <div id={cardDomId} key={id} className={`answer-card proof-card ${isExpanded ? 'expanded' : ''} ${badge.cardClass}`}>
                          <div className="answer-header">
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
                                <button
                                  type="button"
                                  className="expand-button"
                                  onClick={() => handleExpand(proof)}
                                  aria-expanded={isExpanded}
                                  aria-controls={detailsDomId}
                                  aria-label={`${isExpanded ? 'Collapse' : 'Expand'} proof ${proof.theorem_name || proof.proof_id}`}
                                >
                                  {isExpanded ? 'Hide' : 'View'}
                                </button>
                              </div>
                            </div>

                            <div className="answer-metadata">
                              <span className={`format-badge ${badge.cssClass}`}>
                                {badge.label}
                              </span>
                              <span className="word-count">
                                {formatSolverName(proof.solver)}
                              </span>
                              <span className="word-count">
                                {proof.attempt_count || 0} attempt{(proof.attempt_count || 0) !== 1 ? 's' : ''}
                              </span>
                            </div>

                            <p className="proof-statement">
                              {truncate(proof.theorem_statement, 320)}
                            </p>

                            {proof.source_title && (
                              <p className="answer-prompt">
                                <strong>Source:</strong> {truncate(proof.source_title, 220)}
                              </p>
                            )}

                            <div className="answer-footer-info">
                              <span className="completion-date">
                                Verified: {formatDate(proof.created_at)}
                              </span>
                            </div>
                          </div>

                            <div id={detailsDomId} className="answer-content" role="region" aria-label={`Details for ${proof.theorem_name || proof.proof_id}`} hidden={!isExpanded}>
                            {isExpanded && (
                              loadingContentId === id ? (
                                <div className="library-loading" style={{ padding: '20px' }}>
                                  <span className="library-loading__icon">&#x21BB;</span>
                                  <span className="library-loading__text">Loading proof work details...</span>
                                </div>
                              ) : expandedProof ? (
                                <div className="proof-expanded-content">
                                  <div className="proof-detail-section">
                                    <h4>Proof Work</h4>
                                    <p>{expandedProof.theorem_statement}</p>
                                  </div>

                                  {expandedProof.lean_template && (
                                    <div className="proof-detail-section">
                                      <h4>Original Proof Solver Template</h4>
                                      <pre className="proof-code-block">{expandedProof.lean_template}</pre>
                                    </div>
                                  )}

                                  {expandedProof.lean_code && (
                                    <div className="proof-detail-section">
                                      <h4>Lean 4 Source Code</h4>
                                      <pre className="proof-code-block proof-lean-code">{expandedProof.lean_code}</pre>
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
                              ) : null
                            )}
                            </div>
                        </div>
                      );
                    })}
                  </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      <div className="library-footer">
        <button className="refresh-button" onClick={loadProofLibrary}>
          Refresh Proof Works Library
        </button>
      </div>
    </div>
  );
}
