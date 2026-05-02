import React, { useState, useEffect, useMemo } from 'react';
import { autonomousAPI } from '../../services/api';
import { buildResearchRunGroups } from '../../utils/researchRunHistory';
import { downloadRawText } from '../../utils/downloadHelpers';
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

export default function ProofLibrary() {
  const [proofs, setProofs] = useState([]);
  const [sessionsResponse, setSessionsResponse] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedId, setExpandedId] = useState(null);
  const [expandedProof, setExpandedProof] = useState(null);
  const [loadingContentId, setLoadingContentId] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterNovelty, setFilterNovelty] = useState('novel');

  useEffect(() => {
    loadProofLibrary();
  }, [filterNovelty]);

  const loadProofLibrary = async () => {
    try {
      setLoading(true);
      setError(null);

      const novelOnly = filterNovelty === 'novel';
      const [proofsResult, sessionsResult] = await Promise.allSettled([
        autonomousAPI.getProofLibrary(novelOnly),
        autonomousAPI.getSessions(),
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
  }, [filterNovelty]);

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
      const fullProof = await autonomousAPI.getLibraryProof(proof.session_id, proof.proof_id);
      setExpandedProof(fullProof);
    } catch {
      setExpandedProof(proof);
    } finally {
      setLoadingContentId(null);
    }
  };

  const handleDownloadLean = (proof) => {
    const leanCode = proof.lean_code || '';
    if (!leanCode) return;
    const filename = `${proof.theorem_name || proof.proof_id}.lean`;
    downloadRawText(leanCode, filename);
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
        <h2>Proof Library</h2>
        <p>
          All verified mathematical proofs generated across research sessions.
        </p>
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
              ? 'No verified proofs have been generated yet. Run autonomous research with Lean 4 enabled to generate proofs.'
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
                    <h3 className="run-history-group-title">{group.userPrompt}</h3>
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
                          className={`answer-card proof-card ${isExpanded ? 'expanded' : ''} ${
                            proof.novel ? 'proof-card--novel' : 'proof-card--known'
                          }`}
                        >
                          <div
                            className="answer-header"
                            onClick={() => handleExpand(proof)}
                          >
                            <div className="answer-title-row">
                              <h4 className="answer-title proof-title">
                                {proof.theorem_name || proof.proof_id}
                              </h4>
                              <button className="expand-button">
                                {isExpanded ? '\u25B2' : '\u25BC'}
                              </button>
                            </div>

                            <div className="answer-metadata">
                              <span
                                className={`format-badge ${
                                  proof.novel ? 'proof-badge--novel' : 'proof-badge--known'
                                }`}
                              >
                                {proof.novel ? 'Novel' : 'Known'}
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
                  <h4 className="answer-title">{proof.theorem_name || proof.proof_id}</h4>
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
