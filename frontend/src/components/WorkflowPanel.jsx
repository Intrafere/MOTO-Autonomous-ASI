import React, { useState, useEffect, useCallback, useRef } from 'react';
import { websocket } from '../services/websocket';
import { boostAPI, workflowAPI } from '../services/api';
import './WorkflowPanel.css';

const formatNumber = (n) => n.toLocaleString();

const formatTime = (totalSeconds) => {
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = Math.floor(totalSeconds % 60);
  return `${String(h).padStart(2, '0')}h ${String(m).padStart(2, '0')}m ${String(s).padStart(2, '0')}s`;
};

export default function WorkflowPanel({ isRunning }) {
  const [tasks, setTasks] = useState([]);
  const [collapsed, setCollapsed] = useState(false);
  const [mode, setMode] = useState('idle');
  
  // NEW: Boost controls state
  const [boostNextCount, setBoostNextCount] = useState(0);
  const [boostNextInput, setBoostNextInput] = useState('');
  const [boostedCategories, setBoostedCategories] = useState([]);
  const [availableCategories, setAvailableCategories] = useState([]);
  const [boostEnabled, setBoostEnabled] = useState(false);

  // Token tracking & timer state
  const [tokenStats, setTokenStats] = useState({ total_input: 0, total_output: 0, by_model: {}, elapsed_seconds: 0 });
  const [showPerModel, setShowPerModel] = useState(false);
  const [localElapsed, setLocalElapsed] = useState(0);
  const lastSyncRef = useRef(Date.now());

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
      }
      
      // Fetch categories for current mode
      const categoriesResponse = await boostAPI.getCategories(mode === 'idle' ? 'all' : mode);
      if (categoriesResponse.success) {
        setAvailableCategories(categoriesResponse.categories || []);
      }
    } catch (error) {
      console.debug('Failed to fetch boost status:', error);
    }
  }, [mode]);

  // Fetch boost status on mount and when running state changes
  // ETERNAL: Always fetch boost status, even when not running
  useEffect(() => {
    fetchBoostStatus();
    
    // Poll boost status periodically (every 5 seconds)
    const interval = setInterval(fetchBoostStatus, 5000);
    return () => clearInterval(interval);
  }, [fetchBoostStatus]);

  useEffect(() => {
    if (boostEnabled) {
      expandPanel();
    }
  }, [boostEnabled, expandPanel]);

  // Handle setting boost next count
  const handleSetBoostNextCount = async () => {
    const count = parseInt(boostNextInput, 10);
    if (isNaN(count) || count < 0) {
      return;
    }
    
    try {
      await boostAPI.setNextCount(count);
      setBoostNextCount(count);
      setBoostNextInput('');
    } catch (error) {
      console.error('Failed to set boost count:', error);
    }
  };

  // Handle category toggle
  const handleCategoryToggle = async (categoryId) => {
    try {
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
    const fetchTokenStats = async () => {
      try {
        const resp = await workflowAPI.getTokenStats();
        if (resp.success) {
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

  // Poll for workflow predictions when running
  useEffect(() => {
    if (!isRunning) {
      setTasks([]);
      setMode('idle');
      return;
    }

    // Helper to merge incoming tasks while preserving local boost state
    // (in case backend hasn't updated yet)
    const mergeTasksPreservingBoost = (incomingTasks, existingTasks) => {
      // Create a map of existing boost states
      const existingBoostStates = {};
      existingTasks.forEach(t => {
        existingBoostStates[t.task_id] = t.using_boost;
      });
      
      // Merge: Use backend boost state if set, otherwise preserve local state
      return incomingTasks.map(task => ({
        ...task,
        // Backend boost state takes precedence, but preserve local if backend shows false
        // (in case there's a race condition)
        using_boost: task.using_boost || existingBoostStates[task.task_id] || false
      }));
    };

    // Initial fetch
    const fetchWorkflow = async () => {
      try {
        const response = await workflowAPI.getPredictions();
        console.log('Workflow predictions response:', response);
        if (response.success) {
          if (response.tasks && response.tasks.length > 0) {
            // Merge incoming tasks with existing, preserving boost state
            setTasks(prevTasks => mergeTasksPreservingBoost(response.tasks, prevTasks));
            setMode(response.mode || 'unknown');
          } else {
            // No tasks yet - generate placeholder predictions
            console.log('No tasks from backend, generating placeholders');
            setTasks(prevTasks => {
              // Preserve any existing boost states when generating placeholders
              const placeholderTasks = generatePlaceholderTasks(response.mode || 'aggregator');
              return mergeTasksPreservingBoost(placeholderTasks, prevTasks);
            });
            setMode(response.mode || 'aggregator');
          }
        }
      } catch (error) {
        console.error('Failed to fetch workflow predictions:', error);
        // Generate placeholder tasks on error, preserving boost state
        setTasks(prevTasks => {
          const placeholderTasks = generatePlaceholderTasks('aggregator');
          return mergeTasksPreservingBoost(placeholderTasks, prevTasks);
        });
        setMode('aggregator');
      }
    };

    fetchWorkflow();

    // Poll every 3 seconds while running
    const interval = setInterval(fetchWorkflow, 3000);

    return () => clearInterval(interval);
  }, [isRunning]);

  // Generate placeholder tasks when backend hasn't generated them yet
  const generatePlaceholderTasks = (detectedMode) => {
    const tasks = [];
    const numSubmitters = 3; // Default
    
    if (detectedMode === 'aggregator') {
      // Generate aggregator pattern: S1, S2, S3, V, S1, S2, S3, V...
      for (let i = 0; i < 20; i++) {
        const cyclePos = i % 4;
        if (cyclePos < 3) {
          const submitterId = cyclePos + 1;
          tasks.push({
            task_id: `agg_sub${submitterId}_${String(i).padStart(3, '0')}`,
            sequence_number: i + 1,
            role: `Submitter ${submitterId}${submitterId === 1 ? ' (Main Submitter)' : ''}`,
            mode: null,
            provider: 'lm_studio',
            using_boost: false,
            completed: false,
            active: false
          });
        } else {
          tasks.push({
            task_id: `agg_val_${String(i).padStart(3, '0')}`,
            sequence_number: i + 1,
            role: 'Validator',
            mode: null,
            provider: 'lm_studio',
            using_boost: false,
            completed: false,
            active: false
          });
        }
      }
    } else if (detectedMode === 'compiler') {
      // Generate compiler pattern
      const pattern = ['High-Context', 'Validator', 'High-Context', 'Validator'];
      for (let i = 0; i < 20; i++) {
        const role = pattern[i % pattern.length];
        tasks.push({
          task_id: `comp_${role === 'Validator' ? 'val' : 'hc'}_${String(i).padStart(3, '0')}`,
          sequence_number: i + 1,
          role: role,
          mode: role === 'High-Context' ? 'Construction' : 'Review',
          provider: 'lm_studio',
          using_boost: false,
          completed: false,
          active: false
        });
      }
    }
    
    return tasks;
  };

  useEffect(() => {
    if (!isRunning) {
      return;
    }

    // Subscribe to workflow events
    const handleWorkflowUpdated = (data) => {
      if (data.tasks) {
        // Merge incoming tasks while preserving local boost state
        setTasks(prevTasks => {
          const existingBoostStates = {};
          prevTasks.forEach(t => {
            existingBoostStates[t.task_id] = t.using_boost;
          });
          
          return data.tasks.map(task => ({
            ...task,
            using_boost: task.using_boost || existingBoostStates[task.task_id] || false
          }));
        });
        setMode(data.mode || 'unknown');
      }
    };

    const handleTaskCompleted = async (data) => {
      // Mark task as completed locally
      setTasks(prevTasks => 
        prevTasks.map(task => 
          task.task_id === data.task_id 
            ? { ...task, completed: true, active: false }
            : task
        )
      );
      
      // Request fresh predictions from backend to get updated task list
      try {
        const response = await workflowAPI.getPredictions();
        if (response.success && response.tasks && response.tasks.length > 0) {
          setTasks(prevTasks => {
            // Preserve boost states when merging new predictions
            const existingBoostStates = {};
            prevTasks.forEach(t => {
              existingBoostStates[t.task_id] = t.using_boost;
            });
            
            return response.tasks.map(task => ({
              ...task,
              using_boost: task.using_boost || existingBoostStates[task.task_id] || false
            }));
          });
          setMode(response.mode || 'unknown');
        }
      } catch (error) {
        console.debug('Failed to fetch updated predictions after task completion:', error);
      }
    };

    const handleTaskStarted = (data) => {
      setTasks(prevTasks => 
        prevTasks.map(task => ({
          ...task,
          active: task.task_id === data.task_id
        }))
      );
    };

    const handleBoostToggled = (data) => {
      setTasks(prevTasks => 
        prevTasks.map(task => 
          task.task_id === data.task_id 
            ? { ...task, using_boost: data.boosted }
            : task
        )
      );
    };

    const handleFallback = (data) => {
      // Update tasks to show fallback state
      console.log('OpenRouter fallback:', data);
    };

    // NEW: Handle boost count updates
    const handleBoostNextCountUpdated = (data) => {
      setBoostNextCount(data.count || 0);
    };

    // NEW: Handle category boost toggle
    const handleCategoryBoostToggled = (data) => {
      setBoostedCategories(data.all_categories || []);
    };

    // NEW: Handle boost enabled/disabled
    const handleBoostEnabled = () => {
      setBoostEnabled(true);
      expandPanel();
      fetchBoostStatus();
    };

    const handleBoostDisabled = () => {
      setBoostEnabled(false);
      setBoostNextCount(0);
      setBoostedCategories([]);
    };

    websocket.on('workflow_updated', handleWorkflowUpdated);
    websocket.on('task_completed', handleTaskCompleted);
    websocket.on('task_started', handleTaskStarted);
    websocket.on('task_boost_toggled', handleBoostToggled);
    websocket.on('openrouter_fallback', handleFallback);
    websocket.on('boost_next_count_updated', handleBoostNextCountUpdated);
    websocket.on('category_boost_toggled', handleCategoryBoostToggled);
    websocket.on('boost_enabled', handleBoostEnabled);
    websocket.on('boost_disabled', handleBoostDisabled);

    return () => {
      websocket.off('workflow_updated', handleWorkflowUpdated);
      websocket.off('task_completed', handleTaskCompleted);
      websocket.off('task_started', handleTaskStarted);
      websocket.off('task_boost_toggled', handleBoostToggled);
      websocket.off('openrouter_fallback', handleFallback);
      websocket.off('boost_next_count_updated', handleBoostNextCountUpdated);
      websocket.off('category_boost_toggled', handleCategoryBoostToggled);
      websocket.off('boost_enabled', handleBoostEnabled);
      websocket.off('boost_disabled', handleBoostDisabled);
    };
  }, [isRunning, fetchBoostStatus, expandPanel]);

  const handleTaskClick = async (task) => {
    if (task.completed) return; // Can't toggle completed tasks

    try {
      // Optimistically update local state immediately for better UX
      const newBoostState = !task.using_boost;
      setTasks(prevTasks => 
        prevTasks.map(t => 
          t.task_id === task.task_id 
            ? { ...t, using_boost: newBoostState }
            : t
        )
      );
      
      // Call API (backend will broadcast event, but we already updated locally)
      await boostAPI.toggleTask(task.task_id);
    } catch (error) {
      console.error('Failed to toggle task boost:', error);
      // Revert on error
      setTasks(prevTasks => 
        prevTasks.map(t => 
          t.task_id === task.task_id 
            ? { ...t, using_boost: task.using_boost }
            : t
        )
      );
    }
  };

  // Load collapsed state from localStorage
  useEffect(() => {
    const savedState = localStorage.getItem('workflow_panel_collapsed');
    if (savedState !== null) {
      setCollapsed(savedState === 'true');
    }
  }, []);

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
                Boost not enabled - Enable in API Boost button above
              </div>
            )}
            
            <div className="boost-section">
              <label className="boost-label">Boost Next:</label>
              <div className="boost-next-row">
                <input
                  type="number"
                  min="0"
                  value={boostNextInput}
                  onChange={(e) => setBoostNextInput(e.target.value)}
                  placeholder={boostNextCount.toString()}
                  className="boost-next-input"
                  disabled={!boostEnabled}
                />
                <button 
                  onClick={handleSetBoostNextCount}
                  className="boost-apply-btn"
                  disabled={!boostEnabled || !boostNextInput}
                >
                  Apply
                </button>
                {boostNextCount > 0 && (
                  <span className="boost-count-badge">{boostNextCount} left</span>
                )}
              </div>
            </div>

            {availableCategories.length > 0 && (
              <div className="boost-section">
                <label className="boost-label">Categories:</label>
                <div className="boost-categories">
                  {availableCategories
                    .filter(cat => {
                      // Filter categories based on current mode
                      if (mode === 'aggregator') return cat.group === 'Aggregator';
                      if (mode === 'compiler') return cat.group === 'Compiler';
                      if (mode === 'autonomous') return true; // Show all for autonomous
                      return true;
                    })
                    .map(cat => (
                      <button
                        key={cat.id}
                        className={`category-btn ${boostedCategories.includes(cat.id) ? 'active' : ''}`}
                        onClick={() => handleCategoryToggle(cat.id)}
                        disabled={!boostEnabled}
                        title={`Toggle boost for ${cat.label}`}
                      >
                        {cat.label}
                      </button>
                    ))}
                </div>
              </div>
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

          <div className="workflow-tasks">
            {tasks.length === 0 ? (
              <div className="no-tasks">Loading workflow...</div>
            ) : (
              // Filter to show only non-completed tasks (up to 20)
              // This ensures the panel shows upcoming work, not past work
              tasks
                .filter(task => !task.completed)
                .slice(0, 20)
                .map((task, index) => (
                <div
                  key={task.task_id}
                  className={`task-card ${task.completed ? 'completed' : ''} ${task.active ? 'active' : ''} ${task.using_boost ? 'boosted' : ''}`}
                  onClick={() => handleTaskClick(task)}
                  style={{ cursor: task.completed ? 'default' : 'pointer' }}
                >
                  <div className="task-number">#{index + 1}</div>
                  <div className="task-info">
                    <div className="task-role">{task.role}</div>
                    {task.mode && <div className="task-mode">{task.mode}</div>}
                  </div>
                  <div className="task-badges">
                    {task.using_boost && <span className="boost-icon" title="Using API Boost">✨</span>}
                    {task.completed && <span className="completed-icon">✓</span>}
                    {task.active && <span className="active-icon">⟳</span>}
                  </div>
                  <div className="task-provider">{task.provider === 'openrouter' ? 'OR' : 'LMS'}</div>
                </div>
              ))
            )}
          </div>

          <div className="workflow-legend">
            <div className="legend-item">
              <span className="legend-color default"></span>
              <span>Pending</span>
            </div>
            <div className="legend-item">
              <span className="legend-color boosted"></span>
              <span>Boosted</span>
            </div>
            <div className="legend-item">
              <span className="legend-color active"></span>
              <span>Active</span>
            </div>
            <div className="legend-item">
              <span className="legend-color completed"></span>
              <span>Done</span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

