import React, { useEffect, useMemo, useState } from 'react';
import { downloadTextFile } from '../../utils/downloadHelpers';
import './LeanOJMasterProof.css';

function formatDate(isoString) {
  if (!isoString) return 'Unknown';
  try {
    return new Date(isoString).toLocaleString();
  } catch {
    return isoString;
  }
}

function formatNumber(value) {
  const number = Number(value || 0);
  return Number.isFinite(number) ? number.toLocaleString() : '0';
}

function metadataValue(metadata, key, fallback = 'N/A') {
  const value = metadata?.[key];
  if (value === undefined || value === null || value === '') return fallback;
  return value;
}

export default function LeanOJMasterProof({ api, status, refreshToken = 0 }) {
  const [draft, setDraft] = useState(null);
  const [edits, setEdits] = useState([]);
  const [totalEdits, setTotalEdits] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [showFullProof, setShowFullProof] = useState(false);
  const [expandedEditIndex, setExpandedEditIndex] = useState(null);

  const loadMasterProof = async () => {
    try {
      setLoading(true);
      setError('');
      const [draftResponse, editsResponse] = await Promise.all([
        api.getMasterProof(),
        api.getMasterProofEdits(75),
      ]);
      setDraft(draftResponse || null);
      setEdits(editsResponse?.edits || []);
      setTotalEdits(editsResponse?.total_edits || 0);
    } catch (err) {
      setError(err.message || 'Failed to load Proof Solver master proof draft');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadMasterProof();
  }, [refreshToken, status?.master_proof_version, status?.master_proof_hash]);

  const metadata = draft?.metadata || {};
  const proofContent = draft?.content || '';
  const proofPreview = useMemo(() => {
    if (showFullProof || proofContent.length <= 20000) return proofContent;
    return `${proofContent.slice(0, 20000)}\n\n-- Preview truncated in UI. Download the .lean file or expand full proof to inspect all ${formatNumber(proofContent.length)} characters.`;
  }, [proofContent, showFullProof]);

  const handleDownload = () => {
    if (!proofContent) return;
    downloadTextFile(proofContent, `leanoj_master_proof_${metadataValue(metadata, 'version', 'draft')}.lean`);
  };

  return (
    <div className="leanoj-master-proof">
      <div className="leanoj-master-proof__header">
        <div>
          <h2>Master Proof Draft</h2>
          <p>
            Inspect the durable Proof Solver draft that the final solver edits with exact-string operations before Lean verification.
          </p>
        </div>
        <div className="leanoj-master-proof__actions">
          <button type="button" onClick={loadMasterProof} disabled={loading}>
            {loading ? 'Refreshing...' : 'Refresh'}
          </button>
          <button type="button" onClick={handleDownload} disabled={!proofContent}>
            Download .lean
          </button>
        </div>
      </div>

      {loading && <div className="leanoj-master-proof__empty">Loading master proof draft...</div>}
      {!loading && error && <div className="leanoj-master-proof__error">{error}</div>}

      {!loading && !error && (
        <>
          <div className="leanoj-master-proof__stats">
            <div className="leanoj-master-proof__stat">
              <span>{metadataValue(metadata, 'version', 0)}</span>
              <label>Version</label>
            </div>
            <div className="leanoj-master-proof__stat">
              <span>{formatNumber(metadata?.line_count)}</span>
              <label>Lines</label>
            </div>
            <div className="leanoj-master-proof__stat">
              <span>{formatNumber(metadata?.char_count)}</span>
              <label>Characters</label>
            </div>
            <div className="leanoj-master-proof__stat">
              <span>{formatNumber(totalEdits)}</span>
              <label>Edit Records</label>
            </div>
          </div>

          <div className="leanoj-master-proof__meta">
            <div>
              <strong>Session</strong>
              <span>{draft?.session_id || status?.session_id || 'No session loaded'}</span>
            </div>
            <div>
              <strong>Hash</strong>
              <span>{metadataValue(metadata, 'sha256')}</span>
            </div>
            <div>
              <strong>Last Edit</strong>
              <span>{metadataValue(metadata, 'last_edit_summary', 'No master proof edit recorded yet.')}</span>
            </div>
            {metadata?.last_stuck_reason && (
              <div>
                <strong>Last Stuck Reason</strong>
                <span>{metadata.last_stuck_reason}</span>
              </div>
            )}
          </div>

          {!draft?.exists ? (
            <div className="leanoj-master-proof__empty">
              No master proof draft exists yet. Start or resume Proof Solver and enter the final proof loop to initialize it.
            </div>
          ) : (
            <section className="leanoj-master-proof__panel">
              <div className="leanoj-master-proof__panel-header">
                <h3>Current Master Proof</h3>
                {proofContent.length > 20000 && (
                  <button type="button" onClick={() => setShowFullProof((value) => !value)}>
                    {showFullProof ? 'Show Preview' : 'Show Full Proof'}
                  </button>
                )}
              </div>
              <pre className="leanoj-master-proof__code">{proofPreview}</pre>
            </section>
          )}

          <section className="leanoj-master-proof__panel">
            <div className="leanoj-master-proof__panel-header">
              <h3>Recent Edit History</h3>
              <span>{edits.length} shown</span>
            </div>

            {edits.length === 0 ? (
              <div className="leanoj-master-proof__empty">No edit history recorded yet.</div>
            ) : (
              <div className="leanoj-master-proof__edits">
                {[...edits].reverse().map((edit, index) => {
                  const key = `${edit.master_proof_version || 'v'}-${edit.created_at || index}-${index}`;
                  const expanded = expandedEditIndex === index;
                  return (
                    <article key={key} className="leanoj-master-proof__edit-card">
                      <div className="leanoj-master-proof__edit-topline">
                        <span className="leanoj-master-proof__badge">{edit.action || 'edit'}</span>
                        {edit.operation && <span>{edit.operation}</span>}
                        <span>v{edit.master_proof_version || '?'}</span>
                        <span>{formatDate(edit.created_at)}</span>
                      </div>
                      <p>{edit.reasoning || edit.error_summary || edit.stuck_reason || 'No edit summary provided.'}</p>
                      <button type="button" onClick={() => setExpandedEditIndex(expanded ? null : index)}>
                        {expanded ? 'Hide Details' : 'Show Details'}
                      </button>
                      {expanded && (
                        <div className="leanoj-master-proof__edit-details">
                          {edit.old_string_preview && (
                            <div>
                              <strong>Old String Preview</strong>
                              <pre>{edit.old_string_preview}</pre>
                            </div>
                          )}
                          {edit.new_string_preview && (
                            <div>
                              <strong>New String Preview</strong>
                              <pre>{edit.new_string_preview}</pre>
                            </div>
                          )}
                          {edit.error_summary && (
                            <div>
                              <strong>Error</strong>
                              <pre>{edit.error_summary}</pre>
                            </div>
                          )}
                        </div>
                      )}
                    </article>
                  );
                })}
              </div>
            )}
          </section>
        </>
      )}
    </div>
  );
}
