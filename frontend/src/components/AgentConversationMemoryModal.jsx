import React, { useState } from 'react';
import { connectivityAPI } from '../services/api';
import './settings-common.css';

export default function AgentConversationMemoryModal({
  isOpen,
  onClose,
  connectivityStatus,
  onConnectivityChanged,
  anyWorkflowRunning = false,
}) {
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  if (!isOpen) return null;

  const memory = connectivityStatus?.skills?.agent_conversation_memory || {};
  const enabled = Boolean(memory.enabled);

  const handleToggle = async (nextEnabled) => {
    setLoading(true);
    setMessage('');
    try {
      const next = await connectivityAPI.updateToggles({
        agent_conversation_memory_enabled: nextEnabled,
      });
      onConnectivityChanged?.(next);
      setMessage(
        nextEnabled
          ? 'Session History Memory enabled for Assistant workflow-memory search.'
          : 'Session History Memory disabled for future Assistant workflow-memory search.'
      );
    } catch (error) {
      setMessage(error.message || 'Failed to update Session History Memory toggle');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="inline-modal-overlay" style={{ zIndex: 10000 }} onClick={(event) => event.target === event.currentTarget && onClose()}>
      <div className="inline-modal-content" style={{ width: '560px', maxWidth: '90vw', backgroundColor: '#1a1a2e', borderRadius: '12px' }}>
        <div className="settings-header-row" style={{ marginBottom: '1.5rem' }}>
          <h2 style={{ margin: 0, color: '#fff', fontSize: '1.4rem' }}>Session History Memory</h2>
          <button onClick={onClose} className="modal-close-btn" aria-label="Close Session History Memory">×</button>
        </div>

        <p className="settings-hint">
          Assistant runs in parallel during brainstorming, writing, and proof work.It retrieves up to 7 relevant records from local proof-history memory and SyntheticLib4 when enabled. It does not block workflows and is disabled during critique phases.
        </p>

        <label className="settings-checkbox-label settings-checkbox-label--stacked" style={{ marginBottom: '1rem' }}>
          <input
            type="checkbox"
            checked={enabled}
            onChange={(event) => handleToggle(event.target.checked)}
            disabled={loading || anyWorkflowRunning}
          />
          <span>Enable local proof-history memory for Assistant workflow-memory search</span>
        </label>
        {anyWorkflowRunning && (
          <div className="test-result-banner" style={{ marginBottom: '1rem' }}>
            Stop the active workflow before changing run-level Session History Memory availability.
          </div>
        )}

        <div className="test-result-banner" style={{ marginBottom: '1rem' }}>
          Status: <strong>{memory.status || 'disabled'}</strong>
          <br />
          <small>{memory.message || 'Local proof-history memory status is unavailable.'}</small>
          {memory.local_records !== undefined && (
            <>
              <br />
              <small>{memory.local_records} local proof/history record{memory.local_records === 1 ? '' : 's'} indexed.</small>
            </>
          )}
        </div>

        {message && (
          <div className="test-result-banner" style={{ marginBottom: '1rem' }}>
            {message}
          </div>
        )}

        <div className="submitter-config-section">
          <h3 style={{ marginTop: 0, color: '#fff', fontSize: '1rem' }}>Behavior</h3>
          <p className="settings-hint">
            This is not raw provider transcript storage and does not expose private chain-of-thought or retry scaffolding. Disabling it removes local MOTO/manual/LeanOJ proof-memory corpora from Assistant retrieval without deleting proof records, session history, rejection logs, or saved prompts.
          </p>
        </div>
      </div>
    </div>
  );
}

