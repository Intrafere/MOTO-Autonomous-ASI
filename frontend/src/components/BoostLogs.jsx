import React, { useState, useEffect, useCallback } from 'react';
import { boostAPI } from '../services/api';
import { websocket } from '../services/websocket';
import './BoostLogs.css';

export default function BoostLogs() {
  const [logs, setLogs] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [expandedIndex, setExpandedIndex] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Fetch logs from API
  const fetchLogs = useCallback(async () => {
    try {
      const response = await boostAPI.getLogs(100);
      if (response.success) {
        setLogs(response.logs || []);
        setStats(response.stats || null);
      }
    } catch (error) {
      console.error('Failed to fetch boost logs:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial fetch and auto-refresh
  useEffect(() => {
    fetchLogs();

    let interval;
    if (autoRefresh) {
      interval = setInterval(fetchLogs, 5000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [fetchLogs, autoRefresh]);

  // Handle WebSocket events for real-time updates
  useEffect(() => {
    const handleBoostCallCompleted = () => {
      // Refresh logs when a boost call completes
      fetchLogs();
    };

    websocket.on('boost_call_completed', handleBoostCallCompleted);
    
    return () => {
      websocket.off('boost_call_completed', handleBoostCallCompleted);
    };
  }, [fetchLogs]);

  // Handle clear logs
  const handleClearLogs = async () => {
    if (!window.confirm('Are you sure you want to clear all boost logs?')) {
      return;
    }

    try {
      await boostAPI.clearLogs();
      setLogs([]);
      setStats(null);
      setExpandedIndex(null);
    } catch (error) {
      console.error('Failed to clear logs:', error);
    }
  };

  // Toggle log expansion
  const toggleExpand = (index) => {
    setExpandedIndex(expandedIndex === index ? null : index);
  };

  // Format timestamp
  const formatTimestamp = (timestamp) => {
    try {
      const date = new Date(timestamp);
      return date.toLocaleString();
    } catch {
      return timestamp;
    }
  };

  // Format duration
  const formatDuration = (ms) => {
    if (ms === null || ms === undefined) return '-';
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  // Get boost mode label
  const getBoostModeLabel = (mode) => {
    switch (mode) {
      case 'next_count': return 'Next X';
      case 'category': return 'Category';
      case 'task_id': return 'Task ID';
      default: return mode || 'Unknown';
    }
  };

  return (
    <div className="boost-logs-container">
      <div className="boost-logs-header">
        <h2>Boost API Logs</h2>
        <div className="boost-logs-actions">
          <label className="auto-refresh-toggle">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh
          </label>
          <button onClick={fetchLogs} className="refresh-btn" title="Refresh now">
            Refresh
          </button>
          <button 
            onClick={handleClearLogs} 
            className="clear-btn"
            disabled={logs.length === 0}
          >
            Clear Logs
          </button>
        </div>
      </div>

      {/* Stats Summary */}
      {stats && (
        <div className="boost-stats">
          <div className="stat-card">
            <span className="stat-value">{stats.total_calls}</span>
            <span className="stat-label">Total Calls</span>
          </div>
          <div className="stat-card success">
            <span className="stat-value">{stats.successful_calls}</span>
            <span className="stat-label">Successful</span>
          </div>
          <div className="stat-card error">
            <span className="stat-value">{stats.failed_calls}</span>
            <span className="stat-label">Failed</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">
              {(stats.success_rate * 100).toFixed(1)}%
            </span>
            <span className="stat-label">Success Rate</span>
          </div>
        </div>
      )}

      {/* Stats by Mode */}
      {stats && stats.by_mode && Object.keys(stats.by_mode).length > 0 && (
        <div className="boost-mode-stats">
          <span className="mode-stats-label">By Mode:</span>
          {Object.entries(stats.by_mode).map(([mode, count]) => (
            <span key={mode} className="mode-stat-badge">
              {getBoostModeLabel(mode)}: {count}
            </span>
          ))}
        </div>
      )}

      {/* Logs List */}
      <div className="boost-logs-list">
        {loading ? (
          <div className="logs-loading">Loading boost logs...</div>
        ) : logs.length === 0 ? (
          <div className="logs-empty">
            <p>No boost API calls logged yet.</p>
            <p className="logs-empty-hint">
              Enable boost and make API calls to see logs here.
            </p>
          </div>
        ) : (
          logs.map((log, index) => (
            <div 
              key={index} 
              className={`boost-log-entry ${log.success ? 'success' : 'error'} ${expandedIndex === index ? 'expanded' : ''}`}
            >
              <div 
                className="log-summary"
                onClick={() => toggleExpand(index)}
              >
                <div className="log-status">
                  {log.success ? '✓' : '✗'}
                </div>
                <div className="log-info">
                  <div className="log-task">
                    <span className="log-task-id">{log.task_id}</span>
                    <span className="log-mode-badge">{getBoostModeLabel(log.boost_mode)}</span>
                  </div>
                  <div className="log-meta">
                    <span className="log-model">{log.model}</span>
                    <span className="log-duration">{formatDuration(log.duration_ms)}</span>
                    {log.tokens_used && (
                      <span className="log-tokens">{log.tokens_used} tokens</span>
                    )}
                  </div>
                </div>
                <div className="log-timestamp">{formatTimestamp(log.timestamp)}</div>
                <div className="log-expand-icon">{expandedIndex === index ? '▼' : '▶'}</div>
              </div>

              {expandedIndex === index && (
                <div className="log-details">
                  <div className="log-detail-section">
                    <h4>Role</h4>
                    <pre>{log.role_id}</pre>
                  </div>

                  {log.error && (
                    <div className="log-detail-section error">
                      <h4>Error</h4>
                      <pre>{log.error}</pre>
                    </div>
                  )}

                  <div className="log-detail-section">
                    <h4>Prompt Preview</h4>
                    <pre className="log-preview">{log.prompt_preview || '(empty)'}</pre>
                  </div>

                  <div className="log-detail-section">
                    <h4>Response</h4>
                    <pre className="log-response">{log.response_full || log.response_preview || '(empty)'}</pre>
                  </div>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

