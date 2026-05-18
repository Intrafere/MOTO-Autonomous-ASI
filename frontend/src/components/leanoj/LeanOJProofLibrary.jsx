import React, { useEffect, useMemo, useState } from 'react';
import { downloadTextFile } from '../../utils/downloadHelpers';
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

function getProofBadge(proof) {
  if (proof.proof_kind === 'final') {
    return { cssClass: 'proof-badge--gold', cardClass: 'proof-card--gold', label: 'Final Verified Submission' };
  }
  return { cssClass: 'proof-badge--silver', cardClass: 'proof-card--silver', label: 'Verified Proof Fragment' };
}

function formatSolverName(solver) {
  return String(solver || 'Proof Solver').replace(/^LeanOJ\b/, 'Proof Solver');
}

export default function LeanOJProofLibrary({ api, refreshToken = 0 }) {
  const [proofs, setProofs] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [expandedId, setExpandedId] = useState(null);
  const [expandedProof, setExpandedProof] = useState(null);
  const [loadingContentId, setLoadingContentId] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterKind, setFilterKind] = useState('all');

  const loadProofLibrary = async () => {
    try {
      setLoading(true);
      setError('');
      const response = await api.getProofLibrary(true);
      setProofs(response.proofs || []);
      setSessions(response.sessions || []);
    } catch (err) {
      if (err.status === 404) {
        setProofs([]);
        setSessions([]);
        setError('');
        return;
      }
      setError(err.message || 'Failed to load Proof Solver proof works library');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadProofLibrary();
  }, [refreshToken]);

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
        (proof.session_id || '').toLowerCase().includes(lowerSearch)
      );
    });
  }, [filterKind, proofs, searchTerm]);

  const proofsBySession = useMemo(() => {
    const map = new Map();
    for (const proof of filteredProofs) {
      const sessionId = proof.session_id || 'unknown';
      if (!map.has(sessionId)) map.set(sessionId, []);
      map.get(sessionId).push(proof);
    }
    return map;
  }, [filteredProofs]);

  const visibleSessions = useMemo(() => {
    return sessions.filter((session) => proofsBySession.has(session.session_id));
  }, [proofsBySession, sessions]);

  const counts = useMemo(() => ({
    total: proofs.length,
    final: proofs.filter((proof) => proof.proof_kind === 'final').length,
    subproof: proofs.filter((proof) => proof.proof_kind === 'subproof').length,
  }), [proofs]);

  const handleExpand = async (proof) => {
    const id = proof.library_id || `${proof.session_id}:${proof.proof_id}`;
    if (expandedId === id) {
      setExpandedId(null);
      setExpandedProof(null);
      return;
    }

    setExpandedId(id);
    setLoadingContentId(id);
    try {
      const fullProof = await api.getLibraryProof(proof.session_id, proof.proof_id);
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
          className="search-input"
          type="text"
          placeholder="Search by theorem, problem, session, or Proof Solver source..."
          value={searchTerm}
          onChange={(event) => setSearchTerm(event.target.value)}
        />
        <div className="filter-buttons">
          <button className={filterKind === 'all' ? 'active' : ''} onClick={() => setFilterKind('all')}>
            All
          </button>
          <button className={filterKind === 'final' ? 'active' : ''} onClick={() => setFilterKind('final')}>
            Final
          </button>
          <button className={filterKind === 'subproof' ? 'active' : ''} onClick={() => setFilterKind('subproof')}>
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
            const sessionProofs = proofsBySession.get(session.session_id) || [];
            return (
              <div key={session.session_id} className="run-history-group">
                <div className="run-history-group-header">
                  <div className="run-history-group-heading">
                    <h3 className="run-history-group-title">{session.user_prompt}</h3>
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
                  </div>
                </div>

                <div className="run-history-group-body">
                  <div className="answer-list">
                    {sessionProofs.map((proof) => {
                      const id = proof.library_id || `${proof.session_id}:${proof.proof_id}`;
                      const isExpanded = expandedId === id;
                      const badge = getProofBadge(proof);

                      return (
                        <div key={id} className={`answer-card proof-card ${isExpanded ? 'expanded' : ''} ${badge.cardClass}`}>
                          <div className="answer-header" onClick={() => handleExpand(proof)}>
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

                          {isExpanded && (
                            <div className="answer-content">
                              {loadingContentId === id ? (
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
      )}

      <div className="library-footer">
        <button className="refresh-button" onClick={loadProofLibrary}>
          Refresh Proof Works Library
        </button>
      </div>
    </div>
  );
}
