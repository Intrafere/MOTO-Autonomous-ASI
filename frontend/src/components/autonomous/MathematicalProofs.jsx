import React, { useEffect, useMemo, useState } from 'react';
import './MathematicalProofs.css';
import ProofGraph from './ProofGraph';

function formatDate(isoString) {
  if (!isoString) {
    return 'Unknown';
  }
  try {
    return new Date(isoString).toLocaleString();
  } catch {
    return isoString;
  }
}

function truncate(text, maxLength = 220) {
  if (!text) {
    return '';
  }
  return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
}

function getLeanStatusLabel(status) {
  if (!status?.lean4_enabled) {
    return 'Lean 4 Disabled';
  }
  if (status?.lsp_active) {
    return status.workspace_ready ? 'Lean 4 Ready (LSP)' : 'Lean 4 LSP Starting';
  }
  if (status.workspace_ready) {
    return 'Lean 4 Ready';
  }
  return 'Lean 4 Initializing';
}

function createEmptyGraphState() {
  return {
    loading: false,
    loaded: false,
    error: '',
    nodes: [],
    edgesMoto: [],
    edgesMathlib: [],
  };
}

function MathematicalProofs({ api, refreshToken = 0, selectedProofId = null, latestDependencyEvent = null }) {
  const [proofs, setProofs] = useState([]);
  const [proofStatus, setProofStatus] = useState(null);
  const [brainstorms, setBrainstorms] = useState([]);
  const [papers, setPapers] = useState([]);
  const [filter, setFilter] = useState('novel');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedProofId, setExpandedProofId] = useState(null);
  const [manualSourceType, setManualSourceType] = useState('brainstorm');
  const [manualSourceId, setManualSourceId] = useState('');
  const [manualCheckPending, setManualCheckPending] = useState(false);
  const [manualCheckMessage, setManualCheckMessage] = useState('');
  const [dependencyStateByProofId, setDependencyStateByProofId] = useState({});
  const [viewMode, setViewMode] = useState('list');
  const [proofGraphState, setProofGraphState] = useState(createEmptyGraphState);

  const loadProofs = async () => {
    try {
      setLoading(true);
      setError(null);
      setProofGraphState(createEmptyGraphState());

      const [proofsResponse, statusResponse, brainstormsResponse, papersResponse] = await Promise.all([
        api.getProofs(),
        api.getProofStatus(),
        api.getBrainstorms(),
        api.getPapers(),
      ]);

      setProofs(proofsResponse.proofs || []);
      setProofStatus(statusResponse);
      setBrainstorms(brainstormsResponse.brainstorms || []);
      setPapers(papersResponse.papers || []);
    } catch (err) {
      setError(`Failed to load proofs: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadProofs();
  }, [refreshToken]);

  useEffect(() => {
    if (!selectedProofId) {
      return;
    }
    setFilter('novel');
    setViewMode('list');
    setExpandedProofId(selectedProofId);
  }, [selectedProofId]);

  const expandedDependencyState = expandedProofId ? dependencyStateByProofId[expandedProofId] : null;

  useEffect(() => {
    if (!expandedProofId || !proofStatus?.lean4_enabled) {
      return;
    }
    if (expandedDependencyState?.loading || expandedDependencyState?.loaded) {
      return;
    }

    let cancelled = false;
    setDependencyStateByProofId((prev) => ({
      ...prev,
      [expandedProofId]: {
        loading: true,
        loaded: false,
        dependsOn: [],
        dependedOnBy: [],
        mathlibDependedOnBy: [],
      },
    }));

    api.getProofDependencies(expandedProofId)
      .then((response) => {
        if (cancelled) {
          return;
        }
        setDependencyStateByProofId((prev) => ({
          ...prev,
          [expandedProofId]: {
            loading: false,
            loaded: true,
            dependsOn: response.depends_on || [],
            dependedOnBy: response.depended_on_by || [],
            mathlibDependedOnBy: response.mathlib_depended_on_by || [],
          },
        }));
      })
      .catch(() => {
        if (cancelled) {
          return;
        }
        setDependencyStateByProofId((prev) => ({
          ...prev,
          [expandedProofId]: {
            loading: false,
            loaded: true,
            dependsOn: [],
            dependedOnBy: [],
            mathlibDependedOnBy: [],
          },
        }));
      });

    return () => {
      cancelled = true;
    };
  }, [api, expandedDependencyState, expandedProofId, proofStatus?.lean4_enabled]);

  useEffect(() => {
    if (!latestDependencyEvent || viewMode !== 'graph' || !proofGraphState.loaded) {
      return;
    }

    setProofGraphState((previous) => {
      if (!previous.loaded) {
        return previous;
      }

      const nodeIds = new Set(previous.nodes.map((node) => node.proof_id));
      const dependencyPayload = latestDependencyEvent.dependencies || [];
      const needsRefetch = !nodeIds.has(latestDependencyEvent.proof_id) || dependencyPayload.some(
        (dependency) => dependency.kind === 'moto' && dependency.source_ref && !nodeIds.has(dependency.source_ref)
      );

      if (needsRefetch) {
        return createEmptyGraphState();
      }

      const edgesMoto = [...previous.edgesMoto];
      const edgesMathlib = [...previous.edgesMathlib];
      const motoKeys = new Set(edgesMoto.map((edge) => `${edge.from}->${edge.to}:${edge.name || ''}`));
      const mathlibKeys = new Set(edgesMathlib.map((edge) => `${edge.from}->${edge.name}:${edge.source_ref || ''}`));

      dependencyPayload.forEach((dependency) => {
        if (dependency.kind === 'moto' && dependency.source_ref) {
          const key = `${latestDependencyEvent.proof_id}->${dependency.source_ref}:${dependency.name || ''}`;
          if (!motoKeys.has(key)) {
            motoKeys.add(key);
            edgesMoto.push({
              from: latestDependencyEvent.proof_id,
              to: dependency.source_ref,
              name: dependency.name,
            });
          }
          return;
        }

        if (dependency.kind === 'mathlib') {
          const key = `${latestDependencyEvent.proof_id}->${dependency.name}:${dependency.source_ref || ''}`;
          if (!mathlibKeys.has(key)) {
            mathlibKeys.add(key);
            edgesMathlib.push({
              from: latestDependencyEvent.proof_id,
              name: dependency.name,
              source_ref: dependency.source_ref,
            });
          }
        }
      });

      return {
        ...previous,
        edgesMoto,
        edgesMathlib,
      };
    });
  }, [latestDependencyEvent, proofGraphState.loaded, viewMode]);

  useEffect(() => {
    if (viewMode !== 'graph' || !proofStatus?.lean4_enabled) {
      return;
    }
    if (proofGraphState.loading || proofGraphState.loaded) {
      return;
    }

    let cancelled = false;
    setProofGraphState((previous) => ({
      ...previous,
      loading: true,
      error: '',
    }));

    api.getProofGraph()
      .then((response) => {
        if (cancelled) {
          return;
        }
        setProofGraphState({
          loading: false,
          loaded: true,
          error: '',
          nodes: response.nodes || [],
          edgesMoto: response.edges_moto || [],
          edgesMathlib: response.edges_mathlib || [],
        });
      })
      .catch((err) => {
        if (cancelled) {
          return;
        }
        setProofGraphState({
          loading: false,
          loaded: true,
          error: `Failed to load proof graph: ${err.message}`,
          nodes: [],
          edgesMoto: [],
          edgesMathlib: [],
        });
      });

    return () => {
      cancelled = true;
    };
  }, [api, proofGraphState.loaded, proofGraphState.loading, proofStatus?.lean4_enabled, viewMode]);

  const availableBrainstorms = useMemo(
    () => brainstorms.filter((brainstorm) => brainstorm.status === 'complete'),
    [brainstorms]
  );

  const availablePapers = useMemo(
    () => papers.filter((paper) => paper.status === 'complete'),
    [papers]
  );

  const availableSources = useMemo(
    () => (manualSourceType === 'brainstorm' ? availableBrainstorms : availablePapers),
    [manualSourceType, availableBrainstorms, availablePapers]
  );

  useEffect(() => {
    if (availableSources.length === 0) {
      setManualSourceId('');
      return;
    }

    const sourceIdKey = manualSourceType === 'brainstorm' ? 'topic_id' : 'paper_id';
    const hasSelectedSource = availableSources.some((source) => source[sourceIdKey] === manualSourceId);
    if (!hasSelectedSource) {
      setManualSourceId(availableSources[0][sourceIdKey]);
    }
  }, [availableSources, manualSourceId, manualSourceType]);

  const counts = useMemo(() => {
    if (proofStatus?.proof_counts) {
      return proofStatus.proof_counts;
    }
    const novel = proofs.filter((proof) => proof.novel).length;
    return {
      total: proofs.length,
      novel,
      known: proofs.length - novel,
    };
  }, [proofStatus, proofs]);

  const visibleProofs = useMemo(() => {
    if (filter === 'novel') {
      return proofs.filter((proof) => proof.novel);
    }
    return proofs;
  }, [proofs, filter]);
  const visibleProofIds = useMemo(
    () => visibleProofs.map((proof) => proof.proof_id),
    [visibleProofs]
  );
  const showManualPanel = Boolean(proofStatus?.lean4_path);
  const manualChecksDisabled = !proofStatus?.lean4_enabled || !proofStatus?.manual_check_ready || availableSources.length === 0;
  const manualChecksDisabledReason = !proofStatus
    ? 'Loading proof runtime status...'
    : !proofStatus?.lean4_enabled
      ? 'Lean 4 proof checks are disabled.'
    : !proofStatus?.manual_check_ready
      ? (proofStatus?.manual_check_message || 'Manual proof checks are not ready yet.')
      : availableSources.length === 0
        ? 'No completed sources are available yet.'
        : '';

  const handleSelectGraphProof = (proofId) => {
    setExpandedProofId(proofId);
    setViewMode('list');
  };

  const handleRunProofCheck = async () => {
    if (!manualSourceId) {
      return;
    }

    try {
      setManualCheckPending(true);
      setManualCheckMessage('');
      await api.runProofCheck({
        sourceType: manualSourceType,
        sourceId: manualSourceId,
      });
      setManualCheckMessage(`Queued proof check for ${manualSourceType} ${manualSourceId}.`);
    } catch (err) {
      setManualCheckMessage(`Failed to queue proof check: ${err.message}`);
    } finally {
      setManualCheckPending(false);
    }
  };

  return (
    <div className="math-proofs-view">
      <div className="math-proofs-header">
        <div>
          <h2>Mathematical Proofs</h2>
          <p>
            Lean 4 verification runs automatically after brainstorm and paper completion.
          </p>
        </div>

        <div className="math-proofs-status-group">
          <span className={`math-proofs-status ${proofStatus?.workspace_ready ? 'ready' : 'pending'} ${proofStatus?.lean4_enabled ? '' : 'disabled'}`}>
            {getLeanStatusLabel(proofStatus)}
          </span>
          <span className="math-proofs-count">
            {counts.novel || 0} novel / {counts.total || 0} total
          </span>
          <button className="math-proofs-refresh" onClick={loadProofs}>
            Refresh
          </button>
        </div>
      </div>

      <div className="math-proofs-toolbar">
        <div className="math-proofs-toolbar-groups">
          <div className="math-proofs-filters">
            <button
              className={`math-proofs-filter ${filter === 'novel' ? 'active' : ''}`}
              onClick={() => setFilter('novel')}
            >
              Novel Proofs
            </button>
            <button
              className={`math-proofs-filter ${filter === 'all' ? 'active' : ''}`}
              onClick={() => setFilter('all')}
            >
              All Verified Proofs
            </button>
          </div>

          <div className="math-proofs-filters">
            <button
              className={`math-proofs-filter ${viewMode === 'list' ? 'active' : ''}`}
              onClick={() => setViewMode('list')}
            >
              List
            </button>
            <button
              className={`math-proofs-filter ${viewMode === 'graph' ? 'active' : ''}`}
              onClick={() => setViewMode('graph')}
              disabled={!proofStatus?.lean4_enabled}
              title={!proofStatus?.lean4_enabled ? 'Graph view requires Lean 4 proof data.' : undefined}
            >
              Graph
            </button>
          </div>
        </div>

        <div className="math-proofs-version-group">
          <div className="math-proofs-version">
            {proofStatus?.lean4_version || 'Lean 4 version unavailable'}
          </div>
          {proofStatus?.lsp_available && (
            <div className="math-proofs-version">
              {proofStatus.lsp_active ? 'Persistent LSP Active' : 'Persistent LSP Ready'}
            </div>
          )}
          {proofStatus?.smt_enabled && (
            <div className="math-proofs-version">
              {proofStatus.smt_available ? 'Z3 Ready' : 'Z3 Unavailable'}
            </div>
          )}
        </div>
      </div>

      {showManualPanel && (
        <div className="math-proofs-manual-panel">
          <div className="math-proofs-manual-copy">
            <strong>Manual proof check</strong>
            <span>Queue a Lean 4 proof pass for any completed brainstorm or paper.</span>
          </div>
          <div className="math-proofs-manual-controls">
            <select
              value={manualSourceType}
              onChange={(event) => setManualSourceType(event.target.value)}
              disabled={manualCheckPending}
            >
              <option value="brainstorm">Brainstorm</option>
              <option value="paper">Paper</option>
            </select>
            <select
              value={manualSourceId}
              onChange={(event) => setManualSourceId(event.target.value)}
              disabled={manualCheckPending || availableSources.length === 0}
            >
              {availableSources.length === 0 && <option value="">No completed sources available</option>}
              {manualSourceType === 'brainstorm' &&
                availableBrainstorms.map((brainstorm) => (
                  <option key={brainstorm.topic_id} value={brainstorm.topic_id}>
                    {brainstorm.topic_id} - {truncate(brainstorm.topic_prompt, 80)}
                  </option>
                ))}
              {manualSourceType === 'paper' &&
                availablePapers.map((paper) => (
                  <option key={paper.paper_id} value={paper.paper_id}>
                    {paper.paper_id} - {truncate(paper.title, 80)}
                  </option>
                ))}
            </select>
            <button
              className="math-proofs-run-check"
              onClick={handleRunProofCheck}
              disabled={manualChecksDisabled || manualCheckPending}
              title={manualChecksDisabledReason || undefined}
            >
              {manualCheckPending ? 'Queueing...' : 'Run Proof Check'}
            </button>
          </div>
        </div>
      )}

      {manualCheckMessage && (
        <div className={`math-proofs-banner ${manualCheckMessage.startsWith('Failed') ? 'error' : 'success'}`}>
          {manualCheckMessage}
        </div>
      )}

      {loading && <div className="math-proofs-empty">Loading proof database...</div>}
      {!loading && error && <div className="math-proofs-error">{error}</div>}

      {!loading && !error && visibleProofs.length === 0 && (
        <div className="math-proofs-empty">
          No proofs verified yet. Proofs are automatically checked at brainstorm and paper completion.
        </div>
      )}

      {!loading && !error && visibleProofs.length > 0 && viewMode === 'graph' && (
        <>
          {!proofStatus?.lean4_enabled && (
            <div className="math-proofs-empty">
              Graph view is unavailable while Lean 4 proof support is disabled.
            </div>
          )}
          {proofStatus?.lean4_enabled && proofGraphState.loading && (
            <div className="math-proofs-empty">Loading proof dependency graph...</div>
          )}
          {proofStatus?.lean4_enabled && !proofGraphState.loading && proofGraphState.error && (
            <div className="math-proofs-error">{proofGraphState.error}</div>
          )}
          {proofStatus?.lean4_enabled && !proofGraphState.loading && !proofGraphState.error && (
            <ProofGraph
              nodes={proofGraphState.nodes}
              edgesMoto={proofGraphState.edgesMoto}
              edgesMathlib={proofGraphState.edgesMathlib}
              visibleProofIds={visibleProofIds}
              expandedProofId={expandedProofId}
              onSelectProof={handleSelectGraphProof}
            />
          )}
        </>
      )}

      {!loading && !error && visibleProofs.length > 0 && viewMode === 'list' && (
        <div className="math-proofs-list">
          {visibleProofs.map((proof) => {
            const isExpanded = expandedProofId === proof.proof_id;
            const dependencyState = dependencyStateByProofId[proof.proof_id];
            const dependsOn = dependencyState?.dependsOn || [];
            const dependedOnBy = dependencyState?.dependedOnBy || [];
            const mathlibDependedOnBy = dependencyState?.mathlibDependedOnBy || [];
            const showDependencyDetails = Boolean(
              proofStatus?.lean4_enabled &&
              (dependencyState?.loading || dependsOn.length > 0 || dependedOnBy.length > 0 || mathlibDependedOnBy.length > 0)
            );
            return (
              <article
                key={proof.proof_id}
                className={`math-proof-card ${proof.novel ? 'novel' : 'known'}`}
              >
                <div className="math-proof-card-header">
                  <div>
                    <div className="math-proof-card-topline">
                      <span className={`math-proof-badge ${proof.novel ? 'novel' : 'known'}`}>
                        {proof.novel ? 'Novel Proof' : 'Known Proof'}
                      </span>
                      <span className="math-proof-source">
                        {proof.source_type} {proof.source_id}
                      </span>
                    </div>
                    <h3>{proof.theorem_statement}</h3>
                    <p className="math-proof-summary">
                      {truncate(proof.novelty_reasoning || proof.formal_sketch || 'Lean 4 verified this proof.')}
                    </p>
                  </div>

                  <button
                    className="math-proof-expand"
                    onClick={() => setExpandedProofId(isExpanded ? null : proof.proof_id)}
                  >
                    {isExpanded ? 'Hide Details' : 'View Details'}
                  </button>
                </div>

                <div className="math-proof-meta">
                  <span>Solver: {proof.solver || 'Lean 4'}</span>
                  <span>Attempts: {proof.attempt_count || proof.attempts?.length || 0}</span>
                  <span>Created: {formatDate(proof.created_at)}</span>
                </div>

                {isExpanded && (
                  <div className="math-proof-details">
                    <div className="math-proof-actions">
                      <a
                        className="math-proof-download"
                        href={api.getProofCertificateUrl(proof.proof_id)}
                        download={`${proof.proof_id}_certificate.json`}
                      >
                        Download Certificate (JSON)
                      </a>
                      <a
                        className="math-proof-download"
                        href={api.getProofLeanDownloadUrl(proof.proof_id)}
                        download={`${proof.proof_id}.lean`}
                      >
                        Download .lean
                      </a>
                    </div>

                    {proof.theorem_name && (
                      <div className="math-proof-detail-block">
                        <strong>Theorem Name</strong>
                        <div>{proof.theorem_name}</div>
                      </div>
                    )}

                    {proof.source_title && (
                      <div className="math-proof-detail-block">
                        <strong>Source Title</strong>
                        <div>{proof.source_title}</div>
                      </div>
                    )}

                    {proof.formal_sketch && (
                      <div className="math-proof-detail-block">
                        <strong>Formal Sketch</strong>
                        <div>{proof.formal_sketch}</div>
                      </div>
                    )}

                    {proof.novelty_reasoning && (
                      <div className="math-proof-detail-block">
                        <strong>Novelty Review</strong>
                        <div>{proof.novelty_reasoning}</div>
                      </div>
                    )}

                    {proof.solver_hints?.length > 0 && (
                      <div className="math-proof-detail-block">
                        <strong>Solver Hints Used</strong>
                        <div>{proof.solver_hints.join(', ')}</div>
                      </div>
                    )}

                    {showDependencyDetails && (
                      <div className="math-proof-detail-block">
                        <strong>Proof Dependencies</strong>
                        {dependencyState?.loading ? (
                          <div className="math-proof-dependency-empty">Loading dependency graph...</div>
                        ) : (
                          <div className="math-proof-dependency-groups">
                            {dependsOn.length > 0 && (
                              <div>
                                <div className="math-proof-dependency-heading">Depends on</div>
                                <ul className="math-proof-dependency-list">
                                  {dependsOn.map((dependency, index) => (
                                    <li key={`${dependency.kind}-${dependency.name}-${index}`}>
                                      <span className="math-proof-dependency-kind">{dependency.kind}</span>
                                      <span>{dependency.name}</span>
                                      {dependency.source_ref && (
                                        <span className="math-proof-dependency-source">{dependency.source_ref}</span>
                                      )}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            {dependedOnBy.length > 0 && (
                              <div>
                                <div className="math-proof-dependency-heading">Depended on by</div>
                                <ul className="math-proof-dependency-list">
                                  {dependedOnBy.map((dependency) => (
                                    <li key={dependency.proof_id}>
                                      <span>{dependency.theorem_name || dependency.proof_id}</span>
                                      <span className="math-proof-dependency-source">
                                        {truncate(dependency.theorem_statement, 120)}
                                      </span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            {mathlibDependedOnBy.length > 0 && (
                              <div>
                                <div className="math-proof-dependency-heading">Shared Mathlib Usage</div>
                                <ul className="math-proof-dependency-list">
                                  {mathlibDependedOnBy.map((entry) => (
                                    <li key={entry.name}>
                                      <span className="math-proof-dependency-kind">mathlib</span>
                                      <span>{entry.name}</span>
                                      <span className="math-proof-dependency-source">
                                        {entry.dependents?.length || 0} proof(s) also use this lemma
                                      </span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </div>
                        )}
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

export default MathematicalProofs;
