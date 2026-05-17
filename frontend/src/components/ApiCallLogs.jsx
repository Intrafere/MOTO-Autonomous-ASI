import React, { useCallback, useEffect, useRef, useState } from 'react';
import './autonomous/AutonomousResearch.css';

const EMPTY_API_STATS = Object.freeze({
  total_calls: 0,
  successful_calls: 0,
  failed_calls: 0,
  success_rate: 0,
  boosted_calls: 0,
  by_phase: {},
  by_model: {},
  by_provider: {},
  by_source: {},
  by_boost_mode: {},
});

function formatDuration(ms) {
  if (ms === null || ms === undefined) return '-';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function formatTimestamp(timestamp) {
  try {
    return new Date(timestamp).toLocaleString();
  } catch {
    return timestamp;
  }
}

function getPhaseLabel(phase) {
  switch (phase) {
    case 'topic_selection': return 'Topic';
    case 'brainstorm': return 'Brainstorm';
    case 'paper_compilation': return 'Paper';
    case 'tier3': return 'Tier 3';
    case 'boost': return 'Boost';
    case 'initial_topic_candidates': return 'Initial Topics';
    case 'initial_brainstorm': return 'Initial Brainstorm';
    case 'recursive_brainstorm': return 'Recursive Brainstorm';
    case 'proof_storm': return 'Legacy Proof Storm';
    case 'path_decision': return 'Path Decision';
    case 'final_proof_loop': return 'Final Proof Loop';
    default: return phase || 'Unknown';
  }
}

function getSourceLabel(source) {
  switch (source) {
    case 'api+boost': return 'Boosted';
    case 'boost': return 'Boost Only';
    default: return 'Standard';
  }
}

function getBoostModeLabel(mode) {
  switch (mode) {
    case 'next_count': return 'Next X';
    case 'category': return 'Category';
    case 'task_id': return 'Task ID';
    default: return mode || 'Unknown';
  }
}

function getProviderLabel(provider) {
  switch (provider) {
    case 'openrouter': return 'OR';
    case 'lm_studio': return 'LMS';
    default: return provider || 'UNK';
  }
}

export default function ApiCallLogs({
  api,
  workflow = null,
  title = 'API Call Logs',
  emptyHint = 'Run a workflow and make API calls to see the combined logs here.',
  style,
}) {
  const [apiLogs, setApiLogs] = useState([]);
  const [apiStats, setApiStats] = useState(null);
  const [apiLogsLoading, setApiLogsLoading] = useState(true);
  const [expandedApiLogIdx, setExpandedApiLogIdx] = useState(null);
  const [apiAutoRefresh, setApiAutoRefresh] = useState(true);
  const [apiLogDetails, setApiLogDetails] = useState({});
  const abortControllerRef = useRef(null);

  const fetchApiLogs = useCallback(async () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await api.getApiLogs(100, { signal: controller.signal, workflow });
      if (abortControllerRef.current !== controller) {
        return;
      }

      if (response.success) {
        const logs = response.logs || [];
        setApiLogs(logs);
        setApiLogDetails((prev) => {
          const visibleKeys = new Set(logs.map((log) => log.log_key).filter(Boolean));
          return Object.fromEntries(
            Object.entries(prev).filter(([key]) => visibleKeys.has(key))
          );
        });
        setApiStats(response.stats || EMPTY_API_STATS);
      }
    } catch (error) {
      if (abortControllerRef.current !== controller) {
        return;
      }

      if (error.name !== 'AbortError') {
        console.error('Failed to fetch API logs:', error);
      }
    } finally {
      if (abortControllerRef.current === controller) {
        setApiLogsLoading(false);
      }
    }
  }, [api, workflow]);

  const fetchApiLogDetail = useCallback(async (log) => {
    const logKey = log?.log_key;
    if (!logKey || typeof api.getApiLogDetail !== 'function') {
      return log;
    }

    if (apiLogDetails[logKey]) {
      return apiLogDetails[logKey];
    }

    try {
      const response = await api.getApiLogDetail(logKey, { workflow });
      const detailedLog = response.log || log;
      setApiLogDetails((prev) => ({
        ...prev,
        [logKey]: detailedLog,
      }));
      return detailedLog;
    } catch (error) {
      console.error('Failed to fetch API log detail:', error);
      return log;
    }
  }, [api, apiLogDetails, workflow]);

  useEffect(() => {
    fetchApiLogs();

    let interval;
    if (apiAutoRefresh) {
      interval = setInterval(fetchApiLogs, 5000);
    }

    return () => {
      if (interval) clearInterval(interval);
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
    };
  }, [fetchApiLogs, apiAutoRefresh]);

  const handleClearApiLogs = async () => {
    if (!window.confirm('Are you sure you want to clear these API logs?')) {
      return;
    }

    try {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }

      await api.clearApiLogs({ workflow });
      setApiLogs([]);
      setApiLogDetails({});
      setApiStats(EMPTY_API_STATS);
      setExpandedApiLogIdx(null);
      setApiLogsLoading(false);
    } catch (error) {
      console.error('Failed to clear API logs:', error);
    }
  };

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
    }
  };

  const handleToggleApiLog = (log, index) => {
    const nextIndex = expandedApiLogIdx === index ? null : index;
    setExpandedApiLogIdx(nextIndex);
    if (nextIndex !== null) {
      fetchApiLogDetail(log);
    }
  };

  const handleCopyLogText = async (log, fullField, previewField) => {
    const detailedLog = await fetchApiLogDetail(log);
    copyToClipboard(detailedLog?.[fullField] || log?.[previewField] || '');
  };

  return (
    <div className="api-logs-section" style={style}>
      <div className="api-logs-header">
        <h3>{title}</h3>
        <div className="api-logs-actions">
          <label className="auto-refresh-toggle">
            <input
              type="checkbox"
              checked={apiAutoRefresh}
              onChange={(e) => setApiAutoRefresh(e.target.checked)}
            />
            Auto-refresh
          </label>
          <button onClick={fetchApiLogs} className="refresh-btn" title="Refresh now">
            Refresh
          </button>
          <button
            onClick={handleClearApiLogs}
            className="clear-btn"
            disabled={apiLogs.length === 0}
          >
            Clear Logs
          </button>
        </div>
      </div>

      {apiStats && (
        <div className="api-stats">
          <div className="stat-card">
            <span className="stat-value">{apiStats.total_calls}</span>
            <span className="stat-label">Total API Calls</span>
          </div>
          <div className="stat-card success">
            <span className="stat-value">{apiStats.successful_calls}</span>
            <span className="stat-label">Successful</span>
          </div>
          <div className="stat-card error">
            <span className="stat-value">{apiStats.failed_calls}</span>
            <span className="stat-label">Failed</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">
              {(apiStats.success_rate * 100).toFixed(1)}%
            </span>
            <span className="stat-label">Success Rate</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">{apiStats.boosted_calls || 0}</span>
            <span className="stat-label">Boosted Calls</span>
          </div>
        </div>
      )}

      {apiStats && apiStats.by_phase && Object.keys(apiStats.by_phase).length > 0 && (
        <div className="phase-stats">
          <span className="phase-stats-label">By Phase:</span>
          {Object.entries(apiStats.by_phase).map(([phase, count]) => (
            <span key={phase} className="phase-stat-badge">
              {getPhaseLabel(phase)}: {count}
            </span>
          ))}
        </div>
      )}

      {apiStats && apiStats.by_source && Object.keys(apiStats.by_source).length > 0 && (
        <div className="phase-stats">
          <span className="phase-stats-label">By Source:</span>
          {Object.entries(apiStats.by_source).map(([source, count]) => (
            <span key={source} className="phase-stat-badge">
              {getSourceLabel(source)}: {count}
            </span>
          ))}
        </div>
      )}

      {apiStats && apiStats.by_boost_mode && Object.keys(apiStats.by_boost_mode).length > 0 && (
        <div className="phase-stats">
          <span className="phase-stats-label">Boost Modes:</span>
          {Object.entries(apiStats.by_boost_mode).map(([mode, count]) => (
            <span key={mode} className="phase-stat-badge">
              {getBoostModeLabel(mode)}: {count}
            </span>
          ))}
        </div>
      )}

      <div className="api-logs-list">
        {apiLogsLoading ? (
          <div className="logs-loading">Loading API logs...</div>
        ) : apiLogs.length === 0 ? (
          <div className="logs-empty">
            <p>No API calls logged yet.</p>
            <p className="logs-empty-hint">{emptyHint}</p>
          </div>
        ) : (
          apiLogs.map((log, index) => {
            const detailedLog = log.log_key ? (apiLogDetails[log.log_key] || log) : log;
            return (
            <div
              key={log.log_key || `${log.timestamp || 'log'}-${log.task_id || index}`}
              className={`api-log-entry ${log.success ? 'success' : 'error'} ${expandedApiLogIdx === index ? 'expanded' : ''}`}
            >
              <div
                className="log-summary"
                onClick={() => handleToggleApiLog(log, index)}
              >
                <div className="log-status">
                  {log.success ? '✓' : '✗'}
                </div>
                <div className="log-info">
                  <div className="log-task">
                    <span className="log-task-id">{log.task_id}</span>
                    <span className="log-phase-badge">{getPhaseLabel(log.phase)}</span>
                    <span className={`log-source-badge ${log.boosted ? 'boosted' : 'standard'}`}>
                      {getSourceLabel(log.source)}
                    </span>
                    {log.boost_mode && (
                      <span className="log-boost-mode-badge">{getBoostModeLabel(log.boost_mode)}</span>
                    )}
                  </div>
                  <div className="log-meta">
                    <span className="log-model">{log.model}</span>
                    <span className="log-provider-badge">{getProviderLabel(log.provider)}</span>
                    <span className="log-duration">{formatDuration(log.duration_ms)}</span>
                    {log.tokens_used && (
                      <span className="log-tokens">{log.tokens_used} tokens</span>
                    )}
                  </div>
                </div>
                <div className="log-timestamp">{formatTimestamp(log.timestamp)}</div>
                <div className="log-expand-icon">{expandedApiLogIdx === index ? '▼' : '▶'}</div>
              </div>

              {expandedApiLogIdx === index && (
                <div className="log-details">
                  <div className="log-detail-section">
                    <h4>Role</h4>
                    <pre>{log.role_id}</pre>
                  </div>

                  <div className="log-detail-section">
                    <h4>Source</h4>
                    <pre>{getSourceLabel(log.source)}{log.boost_mode ? ` (${getBoostModeLabel(log.boost_mode)})` : ''}</pre>
                  </div>

                  {log.error && (
                    <div className="log-detail-section error">
                      <h4>Error</h4>
                      <pre>{log.error}</pre>
                    </div>
                  )}

                  <div className="log-detail-section">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <h4>Sent Prompt</h4>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleCopyLogText(log, 'prompt_full', 'prompt_preview');
                        }}
                        className="copy-btn"
                        title={detailedLog.has_full_prompt ? 'Copy full prompt to clipboard' : 'Copy prompt preview to clipboard'}
                      >
                        {detailedLog.has_full_prompt ? 'Copy Full' : 'Copy Preview'}
                      </button>
                    </div>
                    {detailedLog.prompt_redacted && (
                      <div className="settings-hint">Full prompt redacted; preview and size/hash metadata are retained.</div>
                    )}
                    <pre className="log-preview">{log.prompt_preview || '(empty)'}</pre>
                  </div>

                  <div className="log-detail-section">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <h4>Received Response</h4>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleCopyLogText(log, 'response_full', 'response_preview');
                        }}
                        className="copy-btn"
                        title={detailedLog.has_full_response ? 'Copy full response to clipboard' : 'Copy response preview to clipboard'}
                      >
                        {detailedLog.has_full_response ? 'Copy Full' : 'Copy Preview'}
                      </button>
                    </div>
                    {detailedLog.response_redacted && (
                      <div className="settings-hint">Full response redacted; preview and size/hash metadata are retained.</div>
                    )}
                    <pre className="log-response">{detailedLog.response_preview || log.response_preview || '(empty)'}</pre>
                  </div>
                </div>
              )}
            </div>
            );
          })
        )}
      </div>
    </div>
  );
}
