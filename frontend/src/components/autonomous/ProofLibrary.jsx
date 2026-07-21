import React, { useState, useEffect, useMemo, useRef } from 'react';
import { autonomousAPI, proofSearchAPI } from '../../services/api';
import { buildResearchRunGroups, formatRunPromptPreview } from '../../utils/researchRunHistory';
import { downloadTextFile } from '../../utils/downloadHelpers';
import { classifyProofNovelty, getCanonicalProofIdentity, sanitizeDomId } from '../../utils/proofPresentation';
import { readBooleanStorage } from '../../utils/safeStorage';
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
  const presentation = classifyProofNovelty(proof);
  return { cssClass: presentation.badgeClass, label: presentation.label };
}

function getCardClass(proof) {
  return classifyProofNovelty(proof).cardClass;
}

function getProofCardId(proof) {
  return getCanonicalProofIdentity(proof);
}

function getProofRunKey(proof) {
  return proof.run_id || proof.session_id || `orphan:${proof.proof_id}`;
}

function matchesSelectedProof(proof, selectedProofId, selectedSessionId = '', selectedRunId = '') {
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
  if (selectedRunId) {
    return (proof.run_id || proof.session_id || '') === selectedRunId;
  }
  return !selectedSessionId || !proof.session_id || proof.session_id === selectedSessionId;
}

const FEDERATED_QUERY_STORAGE_KEY = 'proof_library_federated_query';
const FEDERATED_CORPORA_STORAGE_KEY = 'proof_library_federated_corpora';
const FEDERATED_VERIFIED_STORAGE_KEY = 'proof_library_federated_verified_only';
const LOCAL_QUERY_STORAGE_KEY = 'proof_library_local_query';
const LOCAL_CATEGORY_STORAGE_KEY = 'proof_library_local_category';
const DEFAULT_CORPORA = ['moto', 'manual', 'leanoj', 'syntheticlib4'];

function readStringStorage(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return value === null ? fallback : value;
  } catch {
    return fallback;
  }
}

function readCorporaStorage() {
  try {
    const parsed = JSON.parse(localStorage.getItem(FEDERATED_CORPORA_STORAGE_KEY));
    const valid = Array.isArray(parsed) ? parsed.filter((id) => DEFAULT_CORPORA.includes(id)) : [];
    return valid.length ? valid : DEFAULT_CORPORA;
  } catch {
    return DEFAULT_CORPORA;
  }
}

function writeStorage(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Storage is optional; keep the current in-memory controls usable.
  }
}

const PROOF_SEARCH_CORPORA = [
  { id: 'moto', label: 'MOTO Autonomous' },
  { id: 'manual', label: 'MOTO Manual' },
  { id: 'leanoj', label: 'LeanOJ' },
  { id: 'syntheticlib4', label: 'SyntheticLib4' },
];

const PROOF_SEARCH_OPTIONS = [
  { id: 'verified_only', label: 'Verified only' },
];

const PROOF_LIBRARY_FILTERS = [
  { id: 'novel', label: 'Novel Proofs' },
  { id: 'duplicate_novel', label: 'Duplicate Novel Proofs' },
  { id: 'not_novel', label: 'Not Novel Proofs' },
  { id: 'all', label: 'All Proofs' },
];

export default function ProofLibrary({
  proofScope = 'autonomous',
  title = 'Proof Library',
  description = 'All verified mathematical proofs generated across research sessions.',
  selectedProofId = null,
  selectedSessionId = '',
  selectedRunId = '',
}) {
  const [proofs, setProofs] = useState([]);
  const [proofCounts, setProofCounts] = useState({});
  const [sessionsResponse, setSessionsResponse] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedId, setExpandedId] = useState(null);
  const [expandedProof, setExpandedProof] = useState(null);
  const activeDetailRequestRef = useRef(null);
  const libraryRequestGenerationRef = useRef(0);
  const searchRequestGenerationRef = useRef(0);
  const searchDetailRequestGenerationRef = useRef(0);
  const [expandedRunGroups, setExpandedRunGroups] = useState(() => new Set());
  const [loadingContentId, setLoadingContentId] = useState(null);
  const [searchTerm, setSearchTerm] = useState(() => readStringStorage(LOCAL_QUERY_STORAGE_KEY, ''));
  const [filterNovelty, setFilterNovelty] = useState(() => {
    const value = readStringStorage(LOCAL_CATEGORY_STORAGE_KEY, 'novel');
    return PROOF_LIBRARY_FILTERS.some(({ id }) => id === value) ? value : 'novel';
  });
  const [proofSearchOverview, setProofSearchOverview] = useState(null);
  const [proofSearchCorpora, setProofSearchCorpora] = useState(readCorporaStorage);
  const [proofSearchVerifiedOnly, setProofSearchVerifiedOnly] = useState(() => readBooleanStorage(FEDERATED_VERIFIED_STORAGE_KEY, true));
  const [proofSearchQuery, setProofSearchQuery] = useState(() => readStringStorage(FEDERATED_QUERY_STORAGE_KEY, ''));
  const [proofSearchResults, setProofSearchResults] = useState([]);
  const [proofSearchMessage, setProofSearchMessage] = useState('');
  const [proofSearchLoading, setProofSearchLoading] = useState(false);
  const [proofSearchExpandedId, setProofSearchExpandedId] = useState(null);
  const [proofSearchExpandedRecord, setProofSearchExpandedRecord] = useState(null);

  const loadProofLibrary = async () => {
    const generation = ++libraryRequestGenerationRef.current;
    activeDetailRequestRef.current = null;
    setExpandedId(null);
    setExpandedProof(null);
    try {
      setLoading(true);
      setError(null);

      const [proofsResult, sessionsResult] = await Promise.allSettled([
        autonomousAPI.getProofLibrary(filterNovelty, proofScope),
        proofScope === 'manual' ? Promise.resolve(null) : autonomousAPI.getSessions(),
      ]);

      if (proofsResult.status !== 'fulfilled') {
        throw proofsResult.reason;
      }

      if (generation !== libraryRequestGenerationRef.current) return;
      setProofs((proofsResult.value.proofs || []).map((proof) => ({ ...proof, scope: proofScope })));
      setProofCounts(proofsResult.value.counts || {});

      if (sessionsResult.status === 'fulfilled') {
        setSessionsResponse(sessionsResult.value);
      } else {
        setSessionsResponse(null);
      }
    } catch (err) {
      if (generation !== libraryRequestGenerationRef.current) return;
      setError(err.message || 'Failed to load proof library');
    } finally {
      if (generation === libraryRequestGenerationRef.current) setLoading(false);
    }
  };

  useEffect(() => {
    loadProofLibrary();
  }, [filterNovelty, proofScope]);

  useEffect(() => {
    writeStorage(LOCAL_QUERY_STORAGE_KEY, searchTerm);
  }, [searchTerm]);
  useEffect(() => {
    writeStorage(LOCAL_CATEGORY_STORAGE_KEY, filterNovelty);
  }, [filterNovelty]);
  useEffect(() => {
    writeStorage(FEDERATED_QUERY_STORAGE_KEY, proofSearchQuery);
  }, [proofSearchQuery]);
  useEffect(() => {
    writeStorage(FEDERATED_CORPORA_STORAGE_KEY, JSON.stringify(proofSearchCorpora));
  }, [proofSearchCorpora]);
  useEffect(() => {
    writeStorage(FEDERATED_VERIFIED_STORAGE_KEY, JSON.stringify(proofSearchVerifiedOnly));
  }, [proofSearchVerifiedOnly]);
  useEffect(() => () => {
    libraryRequestGenerationRef.current += 1;
    searchRequestGenerationRef.current += 1;
    searchDetailRequestGenerationRef.current += 1;
    activeDetailRequestRef.current = null;
  }, []);

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
      session_id: p.run_id || p.session_id || `orphan:${p.proof_id}`,
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
      const sid = getProofRunKey(proof);
      if (!map.has(sid)) map.set(sid, []);
      map.get(sid).push(proof);
    }
    return map;
  }, [filteredProofs]);

  const handleExpand = async (proof) => {
    const id = getProofCardId(proof);
    if (expandedId === id) {
      activeDetailRequestRef.current = null;
      setExpandedId(null);
      setExpandedProof(null);
      return;
    }

    setExpandedId(id);
    activeDetailRequestRef.current = id;
    setLoadingContentId(id);

    try {
      let fullProof;
      try {
        fullProof = await proofSearchAPI.getProof(proof.corpus || proofScope, proof.proof_id, {
          searchId: proof.search_id || null,
          runId: proof.run_id || null,
          sessionId: proof.session_id || null,
        });
        if (!fullProof) throw new Error('Canonical proof hydration unavailable');
      } catch {
        fullProof = await autonomousAPI.getLibraryProof(proof.session_id, proof.proof_id, proofScope);
      }
      if (activeDetailRequestRef.current === id) setExpandedProof(fullProof);
    } catch {
      if (activeDetailRequestRef.current === id) setExpandedProof(proof);
    } finally {
      setLoadingContentId((current) => (current === id ? null : current));
    }
  };

  useEffect(() => {
    if (!selectedProofId || loading) return;
    const visibleMatch = filteredProofs.find((proof) => matchesSelectedProof(proof, selectedProofId, selectedSessionId, selectedRunId));
    if (visibleMatch) return;
    const hiddenNoveltyMatch = proofs.find((proof) => matchesSelectedProof(proof, selectedProofId, selectedSessionId, selectedRunId));
    if (hiddenNoveltyMatch && filterNovelty !== 'all') {
      setFilterNovelty('all');
    }
  }, [filterNovelty, filteredProofs, loading, proofs, selectedProofId, selectedRunId, selectedSessionId]);

  useEffect(() => {
    if (!selectedProofId || loading) return;
    const match = filteredProofs.find((proof) => matchesSelectedProof(proof, selectedProofId, selectedSessionId, selectedRunId));
    if (!match) return;
    const id = getProofCardId(match);
    const runKey = getProofRunKey(match);
    setExpandedRunGroups((previous) => new Set(previous).add(runKey));
    setExpandedId(id);
    setLoadingContentId(id);
    let cancelled = false;
    proofSearchAPI.getProof(match.corpus || proofScope, match.proof_id, {
      searchId: match.search_id || null,
      runId: match.run_id || null,
      sessionId: match.session_id || null,
    })
      .then((fullProof) => {
        if (!fullProof) throw new Error('Canonical proof hydration unavailable');
        return fullProof;
      })
      .catch(() => autonomousAPI.getLibraryProof(match.session_id, match.proof_id, proofScope))
      .then((fullProof) => {
        if (!cancelled) setExpandedProof(fullProof);
      })
      .catch(() => {
        if (!cancelled) setExpandedProof(match);
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingContentId(null);
          setTimeout(() => {
            document.getElementById(sanitizeDomId(id, 'proof-card'))?.scrollIntoView({ behavior: 'smooth', block: 'center' });
          }, 0);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [filteredProofs, loading, proofScope, selectedProofId, selectedRunId, selectedSessionId]);

  const toggleRunGroup = (runKey) => {
    setExpandedRunGroups((previous) => {
      const next = new Set(previous);
      if (next.has(runKey)) next.delete(runKey);
      else next.add(runKey);
      return next;
    });
    activeDetailRequestRef.current = null;
    setExpandedId(null);
    setExpandedProof(null);
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
    const generation = ++searchRequestGenerationRef.current;
    searchDetailRequestGenerationRef.current += 1;
    event?.preventDefault();
    setProofSearchLoading(true);
    setProofSearchMessage('');
    setProofSearchExpandedId(null);
    setProofSearchExpandedRecord(null);
    try {
      const response = await proofSearchAPI.search({
        query: proofSearchQuery,
        corpora: proofSearchCorpora,
        verified_only: proofSearchVerifiedOnly,
        include_partial: false,
        include_failed: false,
        dependency_names: [],
        novelty_filters: [],
        module_filters: [],
        source_filters: [],
        limit: 7,
        hydrate_lean_code: false,
      });
      if (generation !== searchRequestGenerationRef.current) return;
      searchDetailRequestGenerationRef.current += 1;
      setProofSearchResults(response.results || []);
      setProofSearchMessage(response.weak_result_warning || response.ranking_notes || '');
    } catch (err) {
      if (generation !== searchRequestGenerationRef.current) return;
      searchDetailRequestGenerationRef.current += 1;
      setProofSearchResults([]);
      setProofSearchMessage(err.message || 'Proof search failed');
    } finally {
      if (generation === searchRequestGenerationRef.current) setProofSearchLoading(false);
    }
  };

  const handleProofSearchReindex = async () => {
    searchDetailRequestGenerationRef.current += 1;
    setProofSearchExpandedId(null);
    setProofSearchExpandedRecord(null);
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
    const id = record.search_id || `${record.corpus}:${record.session_id || ''}:${record.proof_id}`;
    if (proofSearchExpandedId === id) {
      searchDetailRequestGenerationRef.current += 1;
      setProofSearchExpandedId(null);
      setProofSearchExpandedRecord(null);
      return;
    }
    setProofSearchExpandedId(id);
    setProofSearchExpandedRecord(record);
    const generation = ++searchDetailRequestGenerationRef.current;
    if (record.lean_code) {
      return;
    }
    try {
      const hydrated = await proofSearchAPI.getProof(record.corpus, record.proof_id, {
        searchId: record.search_id || null,
        runId: record.run_id || null,
        sessionId: record.session_id || null,
      });
      if (generation === searchDetailRequestGenerationRef.current) {
        setProofSearchExpandedRecord(hydrated);
      }
    } catch {
      if (generation === searchDetailRequestGenerationRef.current) setProofSearchExpandedRecord(record);
    }
  };

  const novelCount = proofCounts.novel ?? proofs.filter((p) => classifyProofNovelty(p).group === 'novel').length;
  const duplicateNovelCount = proofCounts.duplicate_novel ?? proofs.filter((p) => classifyProofNovelty(p).group === 'duplicate_novel').length;
  const notNovelCount = proofCounts.not_novel ?? proofs.filter((p) => classifyProofNovelty(p).group === 'not_novel').length;
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
          <span className="stat-badge">{totalCount} Listed Proof{totalCount !== 1 ? 's' : ''}</span>
          <span className="stat-badge">{novelCount} Novel</span>
          <span className="stat-badge">{duplicateNovelCount} Duplicate Novel</span>
          <span className="stat-badge">{notNovelCount} Not Novel</span>
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
          <label htmlFor="federated-proof-search-query">Federated proof query</label>
          <input
            id="federated-proof-search-query"
            className="search-input"
            type="search"
            value={proofSearchQuery}
            onChange={(event) => setProofSearchQuery(event.target.value)}
            placeholder="Search all enabled proof corpora..."
          />
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
                return null;
              })}
            </div>
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
              const detailsId = sanitizeDomId(id, 'proof-search-details');
              const isExpanded = proofSearchExpandedId === id;
              const expandedRecord = isExpanded ? (proofSearchExpandedRecord || record) : record;
              return (
                <div key={id} className="proof-search-result-card">
                  <button
                    type="button"
                    className="proof-search-result-card__summary"
                    onClick={() => handleExpandProofSearchResult(record)}
                    aria-expanded={isExpanded}
                    aria-controls={detailsId}
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
                  <div
                      id={detailsId}
                      className="proof-expanded-content proof-search-expanded"
                      role="region"
                      aria-label={`Details for ${record.theorem_name || record.proof_id}`}
                      hidden={!isExpanded}
                    >
                    {isExpanded && (
                      <>
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
                      </>
                    )}
                    </div>
                </div>
              );
            })}
          </div>
        )}
      </section>

      <div className="library-controls">
        <input
          aria-label="Filter this proof library"
          className="search-input"
          type="text"
          placeholder="Search by theorem name, statement, source, or research question..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
        <div className="filter-buttons">
          {PROOF_LIBRARY_FILTERS.map((filter) => (
            <button
              key={filter.id}
              className={filterNovelty === filter.id ? 'active' : ''}
              onClick={() => setFilterNovelty(filter.id)}
              aria-pressed={filterNovelty === filter.id}
            >
              {filter.label}
            </button>
          ))}
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
            const isRunExpanded = expandedRunGroups.has(group.sessionId);
            const runRegionId = sanitizeDomId(group.sessionId, 'proof-run');

            return (
              <div key={group.sessionId} className={`run-history-group proof-prompt-group ${isRunExpanded ? 'expanded' : ''}`}>
                <button
                  type="button"
                  className="run-history-group-header proof-prompt-group-header"
                  onClick={() => toggleRunGroup(group.sessionId)}
                  aria-expanded={isRunExpanded}
                  aria-controls={runRegionId}
                  aria-label={`${isRunExpanded ? 'Collapse' : 'Expand'} proof run ${formatRunPromptPreview(group.userPrompt)}`}
                >
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
                    <span className="proof-prompt-group-chevron" aria-hidden="true">
                      {isRunExpanded ? '▲' : '▼'}
                    </span>
                  </div>
                </button>

                <div
                  id={runRegionId}
                  className="run-history-group-body"
                  role="region"
                  aria-label={`Proofs for ${formatRunPromptPreview(group.userPrompt)}`}
                  hidden={!isRunExpanded}
                >
                  {isRunExpanded && (
                  <div className="answer-list">
                    {sessionProofs.map((proof) => {
                      const id = getProofCardId(proof);
                      const cardDomId = sanitizeDomId(id, 'proof-card');
                      const detailsDomId = sanitizeDomId(id, 'proof-library-details');
                      const isExpanded = expandedId === id;

                      return (
                        <div
                          id={cardDomId}
                          key={id}
                          className={`answer-card proof-card ${isExpanded ? 'expanded' : ''} ${getCardClass(proof)}`}
                        >
                          <div
                            className="answer-header"
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
                                <button
                                  type="button"
                                  className="expand-button"
                                  onClick={() => handleExpand(proof)}
                                  aria-expanded={isExpanded}
                                  aria-controls={detailsDomId}
                                  aria-label={`${isExpanded ? 'Collapse' : 'Expand'} proof ${proof.theorem_name || proof.proof_id}`}
                                >
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

                            <div
                              id={detailsDomId}
                              className="answer-content"
                              role="region"
                              aria-label={`Details for ${proof.theorem_name || proof.proof_id}`}
                              hidden={!isExpanded}
                            >
                            {isExpanded && (
                              loadingContentId === id ? (
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
      ) : (
        <div className="answer-list">
          {filteredProofs.map((proof) => {
            const id = getProofCardId(proof);
            const cardDomId = sanitizeDomId(id, 'proof-card');
            const detailsDomId = sanitizeDomId(id, 'proof-library-details');
            const isExpanded = expandedId === id;
            return (
              <div
                id={cardDomId}
                key={id}
                className={`answer-card proof-card ${isExpanded ? 'expanded' : ''} ${getCardClass(proof)}`}
              >
                <div className="answer-header">
                  <div className="answer-title-row">
                    <h4 className="answer-title proof-title">{proof.theorem_name || proof.proof_id}</h4>
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
                        {isExpanded ? '\u25B2' : '\u25BC'}
                      </button>
                    </div>
                  </div>
                  <div className="answer-metadata">
                    <span className={`format-badge ${getTierBadge(proof).cssClass}`}>
                      {getTierBadge(proof).label}
                    </span>
                    <span className="word-count">
                      {proof.solver || 'Lean 4'}
                    </span>
                  </div>
                  <p className="proof-statement">
                    {truncate(proof.theorem_statement, 300)}
                  </p>
                </div>
                  <div
                    id={detailsDomId}
                    className="answer-content"
                    role="region"
                    aria-label={`Details for ${proof.theorem_name || proof.proof_id}`}
                    hidden={!isExpanded}
                  >
                  {isExpanded && (
                    loadingContentId === id ? (
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
                        {expandedProof.lean_code && (
                          <div className="proof-detail-section">
                            <h4>Lean 4 Source Code</h4>
                            <pre className="proof-code-block proof-lean-code">
                              {expandedProof.lean_code}
                            </pre>
                          </div>
                        )}
                      </div>
                    ) : null
                  )}
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
