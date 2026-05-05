import React, { useState, useEffect, useCallback, useRef } from 'react';
import { websocket } from '../services/websocket';
import { boostAPI, workflowAPI } from '../services/api';
import './WorkflowPanel.css';

const HOURLY_AUTO_OPEN_INTERVAL_SECONDS = 3600;
const WORKFLOW_PANEL_AUTO_OPEN_HOUR_KEY = 'workflow_panel_last_auto_open_hour';

const formatNumber = (n) => n.toLocaleString();

const formatTime = (totalSeconds) => {
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = Math.floor(totalSeconds % 60);
  return `${String(h).padStart(2, '0')}h ${String(m).padStart(2, '0')}m ${String(s).padStart(2, '0')}s`;
};

export default function WorkflowPanel({ isRunning }) {
  const [collapsed, setCollapsed] = useState(() => {
    const savedState = localStorage.getItem('workflow_panel_collapsed');
    return savedState === 'true';
  });
  const [mode, setMode] = useState('idle');
  
  // Boost controls state
  const [boostNextCount, setBoostNextCount] = useState(0);
  const [boostNextInput, setBoostNextInput] = useState('');
  const [isEditingBoostNext, setIsEditingBoostNext] = useState(false);
  const [boostedCategories, setBoostedCategories] = useState([]);
  const [availableCategories, setAvailableCategories] = useState([]);
  const [boostEnabled, setBoostEnabled] = useState(false);
  const [boostAlwaysPrefer, setBoostAlwaysPrefer] = useState(false);

  // Token tracking & timer state
  const [tokenStats, setTokenStats] = useState({ total_input: 0, total_output: 0, by_model: {}, elapsed_seconds: 0 });
  const [showPerModel, setShowPerModel] = useState(false);
  const [localElapsed, setLocalElapsed] = useState(0);
  const lastSyncRef = useRef(Date.now());
  const hasElapsedSyncRef = useRef(false);
  const lastAutoOpenedHourRef = useRef(0);

  const expandPanel = useCallback(() => {
    setCollapsed(false);
    localStorage.setItem('workflow_panel_collapsed', 'false');
  }, []);

  // Fetch boost status and categories when running
  const fetchBoostStatus = useCallback(async () => {
    try {
      const statusResponse = await boostAPI.getStatus();
      if (statusResponse.success && statusResponse.status) {
        setBoostEnabled(statusResponse.status.enabled);
        setBoostNextCount(statusResponse.status.boost_next_count || 0);
        setBoostedCategories(statusResponse.status.boosted_categories || []);
        setBoostAlwaysPrefer(statusResponse.status.boost_always_prefer || false);
      }
      
      // Always fetch all categories (no mode filter)
      const categoriesResponse = await boostAPI.getCategories('all');
      if (categoriesResponse.success) {
        setAvailableCategories(categoriesResponse.categories || []);
      }
    } catch (error) {
      console.debug('Failed to fetch boost status:', error);
    }
  }, []);

  // Fetch boost status on mount and when running state changes
  // ETERNAL: Always fetch boost status, even when not running
  useEffect(() => {
    fetchBoostStatus();
    
    // Poll boost status periodically (every 5 seconds)
    const interval = setInterval(fetchBoostStatus, 5000);
    return () => clearInterval(interval);
  }, [fetchBoostStatus]);

  useEffect(() => {
    if (boostEnabled && isRunning) {
      expandPanel();
    }
  }, [boostEnabled, expandPanel, isRunning]);

  // Clear stale auto-open state when a new workflow session begins
  useEffect(() => {
    if (isRunning) {
      lastAutoOpenedHourRef.current = 0;
      localStorage.removeItem(WORKFLOW_PANEL_AUTO_OPEN_HOUR_KEY);
    }
  }, [isRunning]);

  useEffect(() => {
    if (!isRunning || !hasElapsedSyncRef.current) {
      return;
    }

    const elapsedHours = Math.floor(localElapsed / HOURLY_AUTO_OPEN_INTERVAL_SECONDS);
    if (elapsedHours < 1 || elapsedHours <= lastAutoOpenedHourRef.current) {
      return;
    }

    if (collapsed) {
      expandPanel();
    }

    lastAutoOpenedHourRef.current = elapsedHours;
    localStorage.setItem(WORKFLOW_PANEL_AUTO_OPEN_HOUR_KEY, elapsedHours.toString());
  }, [collapsed, expandPanel, isRunning, localElapsed]);

  useEffect(() => {
    if (!isEditingBoostNext) {
      setBoostNextInput(boostNextCount > 0 ? boostNextCount.toString() : '');
    }
  }, [boostNextCount, isEditingBoostNext]);

  useEffect(() => {
    if (!isEditingBoostNext) {
      setBoostNextInput(boostNextCount > 0 ? boostNextCount.toString() : '');
    }
  }, [boostNextCount, isEditingBoostNext]);

  // Handle setting boost next count
  const handleSetBoostNextCount = async () => {
    const count = parseInt(boostNextInput, 10);
    if (isNaN(count) || count < 0) {
      return;
    }
    
    try {
      await boostAPI.setNextCount(count);
      setBoostNextCount(count);
      setBoostNextInput(count > 0 ? count.toString() : '');
      setIsEditingBoostNext(false);
    } catch (error) {
      console.error('Failed to set boost count:', error);
    }
  };

  // Handle always-prefer toggle
  const handleAlwaysPreferToggle = async () => {
    try {
      const newValue = !boostAlwaysPrefer;
      await boostAPI.setAlwaysPrefer(newValue);
      setBoostAlwaysPrefer(newValue);
    } catch (error) {
      console.error('Failed to toggle always-prefer boost:', error);
    }
  };

  // Handle category toggle
  const handleCategoryToggle = async (categoryId) => {    try {
      const response = await boostAPI.toggleCategory(categoryId);
      if (response.success) {
        setBoostedCategories(response.all_boosted_categories || []);
      }
    } catch (error) {
      console.error('Failed to toggle category:', error);
    }
  };

  // Token stats: initial fetch on mount and when isRunning changes
  useEffect(() => {
    hasElapsedSyncRef.current = false;

    const fetchTokenStats = async () => {
      try {
        const resp = await workflowAPI.getTokenStats();
        if (resp.success) {
          hasElapsedSyncRef.current = true;
          setTokenStats(resp);
          setLocalElapsed(resp.elapsed_seconds || 0);
          lastSyncRef.current = Date.now();
        }
      } catch { /* ignore */ }
    };
    fetchTokenStats();
  }, [isRunning]);

  // Token stats: listen for real-time WebSocket updates
  useEffect(() => {
    const handleTokenUpdate = (data) => {
      hasElapsedSyncRef.current = true;
      setTokenStats(data);
      setLocalElapsed(data.elapsed_seconds || 0);
      lastSyncRef.current = Date.now();
    };
    websocket.on('token_usage_updated', handleTokenUpdate);
    return () => websocket.off('token_usage_updated', handleTokenUpdate);
  }, []);

  // Local 1-second timer tick for smooth elapsed display
  useEffect(() => {
    if (!isRunning) return;
    const interval = setInterval(() => {
      setLocalElapsed(prev => prev + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, [isRunning]);

  // Fetch current workflow mode when running
  useEffect(() => {
    if (!isRunning) {
      setMode('idle');
      return;
    }

    const fetchMode = async () => {
      try {
        const response = await workflowAPI.getPredictions();
        if (response.success) {
          setMode(response.mode || 'idle');
        }
      } catch (error) {
        console.debug('Failed to fetch workflow mode:', error);
      }
    };

    fetchMode();
    const interval = setInterval(fetchMode, 5000);
    return () => clearInterval(interval);
  }, [isRunning]);

  useEffect(() => {
    if (!isRunning) {
      return;
    }

    const handleBoostNextCountUpdated = (data) => {
      setBoostNextCount(data.count || 0);
    };

    const handleCategoryBoostToggled = (data) => {
      setBoostedCategories(data.all_categories || []);
    };

    const handleBoostEnabled = () => {
      setBoostEnabled(true);
      expandPanel();
      fetchBoostStatus();
    };

    const handleBoostDisabled = () => {
      setBoostEnabled(false);
      setBoostNextCount(0);
      setBoostedCategories([]);
      setBoostAlwaysPrefer(false);
    };

    const handleAlwaysPreferUpdated = (data) => {
      setBoostAlwaysPrefer(data.enabled || false);
    };

    websocket.on('boost_next_count_updated', handleBoostNextCountUpdated);
    websocket.on('category_boost_toggled', handleCategoryBoostToggled);
    websocket.on('boost_enabled', handleBoostEnabled);
    websocket.on('boost_disabled', handleBoostDisabled);
    websocket.on('boost_always_prefer_updated', handleAlwaysPreferUpdated);

    return () => {
      websocket.off('boost_next_count_updated', handleBoostNextCountUpdated);
      websocket.off('category_boost_toggled', handleCategoryBoostToggled);
      websocket.off('boost_enabled', handleBoostEnabled);
      websocket.off('boost_disabled', handleBoostDisabled);
      websocket.off('boost_always_prefer_updated', handleAlwaysPreferUpdated);
    };
  }, [isRunning, fetchBoostStatus, expandPanel]);

  const toggleCollapse = () => {
    const newState = !collapsed;
    setCollapsed(newState);
    localStorage.setItem('workflow_panel_collapsed', newState.toString());
  };

  // REMOVED: Conditional rendering that hid panel when no workflow running
  // WorkflowPanel is now ETERNAL - always visible for boost controls
  // User can access boost configuration at any time, not just during active research
  
  return (
    <div className={`workflow-panel ${collapsed ? 'collapsed' : ''}`}>
      <div className="workflow-header">
        <h3>MOTO Workflow</h3>
        <button onClick={toggleCollapse} className="collapse-btn">
          {collapsed ? '◀' : '▶'}
        </button>
      </div>

      {!collapsed && (
        <>
          <div className="workflow-mode">
            Mode: <span className="wf-mode-badge">{mode}</span>
          </div>

          {/* BOOST CONTROLS - ETERNAL (always visible, even when boost not enabled) */}
          <div className="boost-controls">
            {!boostEnabled && (
              <div className="boost-disabled-notice">
                Boost not enabled - Enable in API Boost button above. This is a great way to use your free, daily OpenRouter credits.
              </div>
            )}
            
            <div className={`boost-section ${boostedCategories.length > 0 || boostAlwaysPrefer ? 'boost-mode-inactive' : ''}`}>
              <label className="boost-label">Boost Next # of Tasks:</label>
              <div className="boost-next-row">
                <input
                  type="number"
                  min="0"
                  value={boostNextInput}
                  onChange={(e) => setBoostNextInput(e.target.value)}
                  onFocus={() => setIsEditingBoostNext(true)}
                  onBlur={() => setIsEditingBoostNext(false)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      handleSetBoostNextCount();
                    }
                  }}
                  placeholder="0"
                  className="boost-next-input"
                  disabled={!boostEnabled || boostedCategories.length > 0 || boostAlwaysPrefer}
                  title={boostedCategories.length > 0 ? 'Disable category boost first' : boostAlwaysPrefer ? 'Disable "always prefer" first' : 'Replace the remaining boosted-call count immediately'}
                />
                <button 
                  onClick={handleSetBoostNextCount}
                  className="boost-apply-btn"
                  disabled={!boostEnabled || boostNextInput.trim() === '' || boostedCategories.length > 0 || boostAlwaysPrefer}
                  title={boostedCategories.length > 0 ? 'Disable category boost first' : boostAlwaysPrefer ? 'Disable "always prefer" first' : 'Apply a new remaining count immediately'}
                >
                  Apply
                </button>
                {boostNextCount > 0 && (
                  <span className="boost-count-badge">{boostNextCount} left</span>
                )}
              </div>
            </div>

            <div className={`boost-section boost-always-prefer-row ${boostNextCount > 0 || boostedCategories.length > 0 ? 'boost-mode-inactive' : ''}`}>
              <label className="boost-always-prefer-label">
                <input
                  type="checkbox"
                  checked={boostAlwaysPrefer}
                  onChange={handleAlwaysPreferToggle}
                  disabled={!boostEnabled || boostNextCount > 0 || boostedCategories.length > 0}
                  className="boost-always-prefer-checkbox"
                />
                <span>Use boost as next API call when available</span>
              </label>
              {boostAlwaysPrefer && (
                <div className="boost-always-prefer-hint">Boost attempted first every call — falls back on failure</div>
              )}
            </div>

            {availableCategories.length > 0 && (
              <>
                <div className="boost-or-divider">— OR —</div>
                <div className={`boost-section ${boostNextCount > 0 || boostAlwaysPrefer ? 'boost-mode-inactive' : ''}`}>
                  <label className="boost-label">Boost by Category:</label>
                  <div className="boost-categories">
                    {['Aggregator', 'Compiler', 'Autonomous'].map(group => {
                      const groupCats = availableCategories.filter(cat => cat.group === group);
                      if (!groupCats.length) return null;
                      return (
                        <div key={group} className="boost-category-group">
                          <span className="boost-group-label">{group}</span>
                          <div className="boost-category-row">
                            {groupCats.map(cat => (
                              <button
                                key={cat.id}
                                className={`category-btn ${boostedCategories.includes(cat.id) ? 'active' : ''}`}
                                onClick={() => handleCategoryToggle(cat.id)}
                                disabled={!boostEnabled || boostNextCount > 0 || boostAlwaysPrefer}
                                title={boostNextCount > 0 ? 'Set Boost Next to 0 first' : boostAlwaysPrefer ? 'Disable "always prefer" first' : `Toggle boost for ${cat.label}`}
                              >
                                {cat.label}
                              </button>
                            ))}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </>
            )}
          </div>

          {/* RESEARCH TIMER & TOKEN STATS */}
          <div className="token-stats-section">
            <div className="research-timer">
              <span className="timer-label">Elapsed</span>
              <span className="timer-value">{formatTime(localElapsed)}</span>
            </div>

            <div className="token-totals">
              <div className="token-row">
                <span className="token-label">Input</span>
                <span className="token-value">{formatNumber(tokenStats.total_input)}</span>
              </div>
              <div className="token-row">
                <span className="token-label">Output</span>
                <span className="token-value">{formatNumber(tokenStats.total_output)}</span>
              </div>
              <div className="token-row token-total-row">
                <span className="token-label">Total</span>
                <span className="token-value">{formatNumber(tokenStats.total_input + tokenStats.total_output)}</span>
              </div>
            </div>

            {Object.keys(tokenStats.by_model || {}).length > 0 && (
              <div className="per-model-section">
                <button
                  className="per-model-toggle"
                  onClick={() => setShowPerModel(prev => !prev)}
                >
                  {showPerModel ? '▾' : '▸'} Per Model ({Object.keys(tokenStats.by_model).length})
                </button>
                {showPerModel && (
                  <div className="per-model-list">
                    {Object.entries(tokenStats.by_model)
                      .sort((a, b) => (b[1].input + b[1].output) - (a[1].input + a[1].output))
                      .map(([modelId, usage]) => (
                        <div key={modelId} className="model-row">
                          <div className="model-name" title={modelId}>{modelId}</div>
                          <div className="model-tokens">
                            <span className="model-in">In: {formatNumber(usage.input)}</span>
                            <span className="model-out">Out: {formatNumber(usage.output)}</span>
                          </div>
                        </div>
                      ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

