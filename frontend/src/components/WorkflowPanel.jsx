import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { websocket } from '../services/websocket';
import { boostAPI, proofSearchAPI, workflowAPI } from '../services/api';
import {
  classifyProofNovelty,
  formatProofProvenance,
  getCanonicalProofIdentity,
  sanitizeDomId,
} from '../utils/proofPresentation';
import {
  getSolutionPathEmptyLabel,
  isSolutionPathSnapshotAtLeast,
  solutionPathEventMatches,
} from '../utils/solutionPathPresentation';
import './WorkflowPanel.css';

const AUTO_OPEN_DELAY_SECONDS = 600;
const BOOST_CATEGORY_GROUPS = ['Aggregator', 'Compiler', 'Autonomous', 'Proof Solver'];
const MAX_SUBMITTERS = 10;
const WORKFLOW_PANEL_COLLAPSED_KEY = 'workflow_panel_collapsed';
const ASSISTANT_MAX_RESULTS = 7;

const readStoredCollapsedState = () => {
  if (typeof window === 'undefined') return true;
  return window.localStorage.getItem(WORKFLOW_PANEL_COLLAPSED_KEY) !== 'false';
};

const clampSubmitterCount = (value, fallback = 3) => {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return fallback;
  return Math.min(MAX_SUBMITTERS, Math.max(1, Math.floor(parsed)));
};

const readStoredJson = (key) => {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? JSON.parse(raw) : null;
  } catch (error) {
    console.debug(`Failed to read ${key}:`, error);
    return null;
  }
};

const extractCategoryPrefix = (taskId = '') => {
  const normalizedTaskId = String(taskId || '');
  const lastUnderscore = normalizedTaskId.lastIndexOf('_');
  if (lastUnderscore > 0) {
    return canonicalCategory(normalizedTaskId.slice(0, lastUnderscore));
  }
  return canonicalCategory(normalizedTaskId);
};

const canonicalCategory = (category) => {
  const aliases = {
    comp_hc: 'comp_writer',
    comp_crit: 'comp_hp',
    critique_val: 'comp_val',
    critique_cleanup: 'comp_val',
    leanoj_path: 'leanoj_final',
  };
  const normalized = String(category || '');
  if (/^critique_sub\d+$/.test(normalized)) return 'comp_hp';
  return aliases[normalized] || normalized;
};

const maxIndexedPrefixFromTasks = (tasks, prefix) => (
  tasks.reduce((maxIndex, task) => {
    const category = extractCategoryPrefix(task?.task_id);
    const match = category.match(new RegExp(`^${prefix}(\\d+)$`));
    return match ? Math.max(maxIndex, Number(match[1])) : maxIndex;
  }, 0)
);

const addIndexedCategories = (ids, prefix, count) => {
  const safeCount = clampSubmitterCount(count);
  for (let index = 1; index <= safeCount; index += 1) {
    ids.add(`${prefix}${index}`);
  }
};

const getAggregatorSubmitterCount = (tasks) => {
  const taskCount = maxIndexedPrefixFromTasks(tasks, 'agg_sub');
  if (taskCount > 0) return clampSubmitterCount(taskCount);

  const settings = readStoredJson('aggregator_settings') || {};
  const legacy = readStoredJson('aggregatorConfig') || {};
  const configs = settings.submitterConfigs || legacy.submitterConfigs || [];
  return clampSubmitterCount(settings.numSubmitters ?? legacy.numSubmitters ?? configs.length);
};

const getAutonomousSubmitterCount = (tasks) => {
  const taskCount = maxIndexedPrefixFromTasks(tasks, 'agg_sub');
  const settings = readStoredJson('autonomous_research_settings') || {};
  const configs = settings.submitterConfigs || settings.submitter_configs || [];
  const storedCount = clampSubmitterCount(settings.numSubmitters ?? configs.length);
  return taskCount > 1 ? clampSubmitterCount(taskCount) : storedCount;
};

const getLeanOJSubmitterCount = (tasks) => {
  const taskCount = Math.max(
    maxIndexedPrefixFromTasks(tasks, 'leanoj_topic_sub'),
    maxIndexedPrefixFromTasks(tasks, 'leanoj_brainstorm_sub')
  );
  if (taskCount > 0) return clampSubmitterCount(taskCount);

  const settings = readStoredJson('leanoj_solver_settings') || {};
  const configs = settings.submitterConfigs || settings.brainstorm_submitters || [];
  return clampSubmitterCount(settings.numSubmitters ?? configs.length);
};

const getAutonomousAllowsPapers = () => {
  const settings = readStoredJson('autonomous_research_settings') || {};
  return settings.allowResearchPapers ?? settings.allow_research_papers ?? true;
};

const getActiveCategoryIds = (mode, tasks = []) => {
  const ids = new Set();
  const normalizedMode = mode || 'idle';

  if (normalizedMode === 'aggregator') {
    addIndexedCategories(ids, 'agg_sub', getAggregatorSubmitterCount(tasks));
    ids.add('agg_val');
  } else if (normalizedMode === 'compiler') {
    ids.add('comp_val');
    ids.add('comp_writer');
    ids.add('comp_hp');
  } else if (normalizedMode === 'autonomous') {
    addIndexedCategories(ids, 'agg_sub', getAutonomousSubmitterCount(tasks));
    ids.add('agg_val');
    if (getAutonomousAllowsPapers()) {
      ids.add('comp_val');
      ids.add('comp_writer');
      ids.add('comp_hp');
    }
  } else if (normalizedMode === 'leanoj') {
    const submitterCount = getLeanOJSubmitterCount(tasks);
    ids.add('leanoj_topic');
    ids.add('leanoj_topic_val');
    ids.add('leanoj_brainstorm_val');
    ids.add('leanoj_sufficiency');
    ids.add('leanoj_path_val');
    ids.add('leanoj_final');
    addIndexedCategories(ids, 'leanoj_topic_sub', submitterCount);
    addIndexedCategories(ids, 'leanoj_brainstorm_sub', submitterCount);
  }

  tasks.forEach((task) => {
    const category = extractCategoryPrefix(task?.task_id);
    if (category) ids.add(category);
  });

  return ids;
};

const formatNumber = (n) => n.toLocaleString();

const formatTime = (totalSeconds) => {
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = Math.floor(totalSeconds % 60);
  return `${String(h).padStart(2, '0')}h ${String(m).padStart(2, '0')}m ${String(s).padStart(2, '0')}s`;
};

const truncateText = (text, maxLength = 140) => {
  const normalized = String(text || '').replace(/\s+/g, ' ').trim();
  if (normalized.length <= maxLength) return normalized;
  return `${normalized.slice(0, maxLength).trim()}...`;
};

const formatAssistantTierLabel = (support = {}) => classifyProofNovelty(support).shortLabel;

const getAssistantTileClass = (support = {}) => classifyProofNovelty(support).tileClass;

const formatAssistantSource = (support = {}) => (
  [
    support.corpus,
    support.corpus_scope,
    support.session_id ? `session ${support.session_id}` : '',
  ].filter(Boolean).join(' · ') || 'proof history'
);

const normalizeAssistantPack = (pack = {}) => ({
  ...pack,
  results: Array.isArray(pack.results) ? pack.results.slice(0, ASSISTANT_MAX_RESULTS) : [],
});

export default function WorkflowPanel({
  isRunning,
  onOpenBoostSettings,
  onOpenAssistantProof,
  onOpenSolutionPath,
  onSolutionPathSnapshotChange,
  collapsed: controlledCollapsed,
  onCollapseChange,
}) {
  const [internalCollapsed, setInternalCollapsed] = useState(readStoredCollapsedState);
  const isControlledCollapsed = typeof controlledCollapsed === 'boolean';
  const collapsed = isControlledCollapsed ? controlledCollapsed : internalCollapsed;
  const [mode, setMode] = useState('idle');
  const [workflowTasks, setWorkflowTasks] = useState([]);
  
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
  const [assistantMemoryPack, setAssistantMemoryPack] = useState(null);
  const [solutionPath, setSolutionPath] = useState(null);
  const solutionPathRef = useRef(null);
  const solutionPathRequestRef = useRef(0);
  const [assistantMemoryError, setAssistantMemoryError] = useState('');
  const [selectedAssistantProof, setSelectedAssistantProof] = useState(null);
  const [assistantProofDetail, setAssistantProofDetail] = useState(null);
  const [assistantProofLoading, setAssistantProofLoading] = useState(false);
  const assistantPackGenerationRef = useRef(0);
  const assistantDetailGenerationRef = useRef(0);
  const mountedRef = useRef(true);

  // Auto-open: pop open exactly once, 10 minutes after user presses Start.
  // No persistence. Resets every time isRunning goes true.
  const hasPoppedThisSession = useRef(false);

  const setPanelCollapsed = useCallback((nextCollapsed) => {
    if (!isControlledCollapsed) {
      setInternalCollapsed(nextCollapsed);
    }
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(WORKFLOW_PANEL_COLLAPSED_KEY, nextCollapsed.toString());
    }
    onCollapseChange?.(nextCollapsed);
  }, [isControlledCollapsed, onCollapseChange]);

  const expandPanel = useCallback(() => {
    setPanelCollapsed(false);
  }, [setPanelCollapsed]);

  useEffect(() => {
    if (isRunning) {
      hasPoppedThisSession.current = false;
    }
  }, [isRunning]);

  const fetchSolutionPath = useCallback(async () => {
    const requestSequence = ++solutionPathRequestRef.current;
    try {
      const snapshot = await workflowAPI.getSolutionPath();
      if (!mountedRef.current || requestSequence !== solutionPathRequestRef.current) return null;
      const current = solutionPathRef.current;
      if (isSolutionPathSnapshotAtLeast(snapshot, current)) {
        solutionPathRef.current = snapshot;
        setSolutionPath(snapshot);
        onSolutionPathSnapshotChange?.(snapshot);
      }
      return snapshot;
    } catch (error) {
      if (!mountedRef.current || requestSequence !== solutionPathRequestRef.current) return null;
      console.debug('Failed to fetch solution path:', error);
      return null;
    }
  }, [onSolutionPathSnapshotChange]);

  useEffect(() => {
    fetchSolutionPath();
  }, [fetchSolutionPath, isRunning]);

  useEffect(() => {
    const handleChanged = (data = {}) => {
      const current = solutionPathRef.current;
      if (!solutionPathEventMatches(data, current, current?.mode || '')) return;
      if (
        current?.run_id === data.run_id
        && Number(data.lifecycle_generation || 0)
          === Number(current?.lifecycle_generation || 0)
        && Number(data.revision || 0) < Number(current?.revision || 0)
      ) return;
      fetchSolutionPath();
    };
    const events = [
      'solution_path_activated',
      'solution_path_proposal_queued',
      'solution_path_proposal_reviewing',
      'solution_path_updated',
      'solution_path_proposal_rejected',
      'solution_path_proposal_retry_queued',
      'solution_path_proposal_user_repair_required',
      'solution_path_proposal_resumed',
    ];
    events.forEach((eventName) => websocket.on(eventName, handleChanged));
    return () => events.forEach((eventName) => websocket.off(eventName, handleChanged));
  }, [fetchSolutionPath]);

  useEffect(() => {
    if (!isRunning || hasPoppedThisSession.current) return;
    if (localElapsed >= AUTO_OPEN_DELAY_SECONDS) {
      expandPanel();
      hasPoppedThisSession.current = true;
    }
  }, [isRunning, localElapsed, expandPanel]);

  // Fetch boost status and categories
  const fetchBoostStatus = useCallback(async () => {
    try {
      const statusResponse = await boostAPI.getStatus();
      if (statusResponse.success && statusResponse.status) {
        setBoostEnabled(statusResponse.status.enabled);
        setBoostNextCount(statusResponse.status.boost_next_count || 0);
        setBoostedCategories(statusResponse.status.boosted_categories || []);
        setBoostAlwaysPrefer(statusResponse.status.boost_always_prefer || false);
      }
      const categoriesResponse = await boostAPI.getCategories('all');
      if (categoriesResponse.success) {
        setAvailableCategories(categoriesResponse.categories || []);
      }
    } catch (error) {
      console.debug('Failed to fetch boost status:', error);
    }
  }, []);

  useEffect(() => {
    fetchBoostStatus();
    const interval = setInterval(fetchBoostStatus, 5000);
    return () => clearInterval(interval);
  }, [fetchBoostStatus]);

  const fetchAssistantMemoryPack = useCallback(async () => {
    const generation = ++assistantPackGenerationRef.current;
    try {
      const pack = await proofSearchAPI.getAssistantLatestPack();
      if (!mountedRef.current || generation !== assistantPackGenerationRef.current) return;
      if (pack?.enabled === false) {
        setAssistantMemoryPack(normalizeAssistantPack({ ...pack, results: [] }));
        setSelectedAssistantProof(null);
        setAssistantProofDetail(null);
        setAssistantMemoryError(pack.disabled_reason || 'Session History Memory is disabled.');
        return;
      }
      setAssistantMemoryPack(normalizeAssistantPack(pack));
      setAssistantMemoryError(pack.disabled_reason || '');
    } catch (error) {
      if (!mountedRef.current || generation !== assistantPackGenerationRef.current) return;
      setAssistantMemoryError(error.message || 'Assistant memory unavailable');
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      assistantPackGenerationRef.current += 1;
      assistantDetailGenerationRef.current += 1;
      solutionPathRequestRef.current += 1;
    };
  }, []);

  useEffect(() => {
    fetchAssistantMemoryPack();
  }, [fetchAssistantMemoryPack]);

  useEffect(() => {
    if (!collapsed) {
      fetchAssistantMemoryPack();
    }
  }, [collapsed, fetchAssistantMemoryPack]);

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

  useEffect(() => {
    const handleAssistantPackUpdated = (data) => {
      if (data?.enabled === false) {
        setAssistantMemoryPack(normalizeAssistantPack({ ...data, results: [] }));
        setSelectedAssistantProof(null);
        setAssistantProofDetail(null);
        setAssistantMemoryError(data.disabled_reason || 'Session History Memory is disabled.');
      } else if (Array.isArray(data?.results)) {
        setAssistantMemoryPack(normalizeAssistantPack({ ...data, has_pack: true, enabled: true }));
        setAssistantMemoryError('');
      } else {
        fetchAssistantMemoryPack();
      }
    };
    const handleAssistantPackFailed = (data = {}) => {
      setAssistantMemoryError(data.error_message || data.reason || 'Assistant memory selection failed');
    };
    const handleAssistantMemoryUnavailable = (data = {}) => {
      setAssistantMemoryPack((previous) => previous ? { ...previous, results: [] } : previous);
      setSelectedAssistantProof(null);
      setAssistantProofDetail(null);
      setAssistantMemoryError(data.reason || 'Assistant memory found no external proof history yet.');
    };
    websocket.on('assistant_proof_pack_updated', handleAssistantPackUpdated);
    websocket.on('assistant_proof_pack_failed', handleAssistantPackFailed);
    websocket.on('assistant_proof_memory_unavailable', handleAssistantMemoryUnavailable);
    websocket.on('assistant_proof_memory_shutdown', handleAssistantMemoryUnavailable);
    return () => {
      websocket.off('assistant_proof_pack_updated', handleAssistantPackUpdated);
      websocket.off('assistant_proof_pack_failed', handleAssistantPackFailed);
      websocket.off('assistant_proof_memory_unavailable', handleAssistantMemoryUnavailable);
      websocket.off('assistant_proof_memory_shutdown', handleAssistantMemoryUnavailable);
    };
  }, [fetchAssistantMemoryPack]);

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
      setWorkflowTasks([]);
      return;
    }

    const fetchMode = async () => {
      try {
        const response = await workflowAPI.getPredictions();
        if (response.success) {
          setMode(response.mode || 'idle');
          setWorkflowTasks(response.tasks || []);
        }
      } catch (error) {
        console.debug('Failed to fetch workflow mode:', error);
      }
    };

    fetchMode();
    const interval = setInterval(fetchMode, 5000);
    return () => clearInterval(interval);
  }, [isRunning]);

  const activeCategoryIds = useMemo(
    () => (isRunning ? getActiveCategoryIds(mode, workflowTasks) : new Set()),
    [isRunning, mode, workflowTasks]
  );

  const visibleCategories = useMemo(
    () => availableCategories.filter((cat) => activeCategoryIds.has(cat.id)),
    [availableCategories, activeCategoryIds]
  );

  const visibleBoostedCategories = useMemo(
    () => boostedCategories.filter((categoryId) => activeCategoryIds.has(categoryId)),
    [boostedCategories, activeCategoryIds]
  );

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

    const handleWorkflowUpdated = (data) => {
      setMode(prevMode => data.mode || prevMode || 'idle');
      setWorkflowTasks(data.tasks || []);
    };

    websocket.on('boost_next_count_updated', handleBoostNextCountUpdated);
    websocket.on('category_boost_toggled', handleCategoryBoostToggled);
    websocket.on('boost_enabled', handleBoostEnabled);
    websocket.on('boost_disabled', handleBoostDisabled);
    websocket.on('boost_always_prefer_updated', handleAlwaysPreferUpdated);
    websocket.on('workflow_updated', handleWorkflowUpdated);

    return () => {
      websocket.off('boost_next_count_updated', handleBoostNextCountUpdated);
      websocket.off('category_boost_toggled', handleCategoryBoostToggled);
      websocket.off('boost_enabled', handleBoostEnabled);
      websocket.off('boost_disabled', handleBoostDisabled);
      websocket.off('boost_always_prefer_updated', handleAlwaysPreferUpdated);
      websocket.off('workflow_updated', handleWorkflowUpdated);
    };
  }, [isRunning, fetchBoostStatus]);

  const toggleCollapse = () => {
    setPanelCollapsed(!collapsed);
  };

  const handleAssistantProofClick = async (support) => {
    const identity = getCanonicalProofIdentity(support);
    const generation = ++assistantDetailGenerationRef.current;
    setSelectedAssistantProof(support);
    setAssistantProofDetail(null);
    setAssistantProofLoading(true);
    onOpenAssistantProof?.(support);
    try {
      const hydrated = await proofSearchAPI.getProof(support.corpus, support.proof_id, {
        searchId: support.search_id || null,
        runId: support.run_id || null,
        sessionId: support.session_id || null,
      });
      if (
        mountedRef.current
        && generation === assistantDetailGenerationRef.current
        && getCanonicalProofIdentity(support) === identity
      ) setAssistantProofDetail(hydrated);
    } catch (error) {
      if (!mountedRef.current || generation !== assistantDetailGenerationRef.current) return;
      setAssistantProofDetail({
        ...support,
        hydration_error: error.message || 'Proof history details could not be opened.',
      });
    } finally {
      if (mountedRef.current && generation === assistantDetailGenerationRef.current) {
        setAssistantProofLoading(false);
      }
    }
  };

  const assistantResults = assistantMemoryPack?.results || [];
  const selectedProofView = assistantProofDetail || selectedAssistantProof;

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
                Boost not enabled - open API Boost settings here to configure a boost model. This is a great way to use your free, daily OpenRouter credits.
              </div>
            )}

            <button
              type="button"
              className="boost-settings-open-btn"
              onClick={onOpenBoostSettings}
            >
              Open API Boost Settings
            </button>
            
            <div className={`boost-section ${visibleBoostedCategories.length > 0 || boostAlwaysPrefer ? 'boost-mode-inactive' : ''}`}>
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
                  disabled={!boostEnabled || visibleBoostedCategories.length > 0 || boostAlwaysPrefer}
                  title={visibleBoostedCategories.length > 0 ? 'Disable category boost first' : boostAlwaysPrefer ? 'Disable "always prefer" first' : 'Replace the remaining boosted-call count immediately'}
                />
                <button 
                  onClick={handleSetBoostNextCount}
                  className="boost-apply-btn"
                  disabled={!boostEnabled || boostNextInput.trim() === '' || visibleBoostedCategories.length > 0 || boostAlwaysPrefer}
                  title={visibleBoostedCategories.length > 0 ? 'Disable category boost first' : boostAlwaysPrefer ? 'Disable "always prefer" first' : 'Apply a new remaining count immediately'}
                >
                  Apply
                </button>
                {boostNextCount > 0 && (
                  <span className="boost-count-badge">{boostNextCount} left</span>
                )}
              </div>
            </div>

            <div className={`boost-section boost-always-prefer-row ${boostNextCount > 0 || visibleBoostedCategories.length > 0 ? 'boost-mode-inactive' : ''}`}>
              <label className="boost-always-prefer-label">
                <input
                  type="checkbox"
                  checked={boostAlwaysPrefer}
                  onChange={handleAlwaysPreferToggle}
                  disabled={!boostEnabled || boostNextCount > 0 || visibleBoostedCategories.length > 0}
                  className="boost-always-prefer-checkbox"
                />
                <span>Use boost as next API call when available</span>
              </label>
              {boostAlwaysPrefer && (
                <div className="boost-always-prefer-hint">Boost attempted first every call — falls back on failure</div>
              )}
            </div>

            {visibleCategories.length > 0 && (
              <>
                <div className="boost-or-divider">— OR —</div>
                <div className={`boost-section ${boostNextCount > 0 || boostAlwaysPrefer ? 'boost-mode-inactive' : ''}`}>
                  <label className="boost-label">Boost by Category:</label>
                  <div className="boost-categories">
                    {BOOST_CATEGORY_GROUPS.map(group => {
                      const groupCats = visibleCategories.filter(cat => cat.group === group);
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
            <div className="token-stats-heading">Token Usage</div>

            <div className="research-timer">
              <span className="timer-label">Elapsed</span>
              <span className="timer-value">{formatTime(localElapsed)}</span>
            </div>

            <div className="token-totals">
              <div className="token-row">
                <span className="token-label">Input tokens</span>
                <span className="token-value">{formatNumber(tokenStats.total_input)}</span>
              </div>
              <div className="token-row">
                <span className="token-label">Output tokens</span>
                <span className="token-value">{formatNumber(tokenStats.total_output)}</span>
              </div>
              <div className="token-row token-total-row">
                <span className="token-label">Total tokens</span>
                <span className="token-value">{formatNumber(tokenStats.total_input + tokenStats.total_output)}</span>
              </div>
            </div>

            {Object.keys(tokenStats.by_model || {}).length > 0 && (
              <div className="per-model-section">
                <button
                  className="per-model-toggle"
                  onClick={() => setShowPerModel(prev => !prev)}
                >
                  {showPerModel ? '▾' : '▸'} Per-model tokens ({Object.keys(tokenStats.by_model).length})
                </button>
                {showPerModel && (
                  <div className="per-model-list">
                    {Object.entries(tokenStats.by_model)
                      .sort((a, b) => (b[1].input + b[1].output) - (a[1].input + a[1].output))
                      .map(([modelId, usage]) => (
                        <div key={modelId} className="model-row">
                          <div className="model-name" title={modelId}>{modelId}</div>
                          <div className="model-tokens">
                            <span className="model-in">Input tokens: {formatNumber(usage.input)}</span>
                            <span className="model-out">Output tokens: {formatNumber(usage.output)}</span>
                          </div>
                        </div>
                      ))}
                  </div>
                )}
              </div>
            )}

            {(solutionPath?.enabled || solutionPath?.acceptance_count >= 5) && (
              <button
                type="button"
                className={`solution-path-card ${solutionPath?.enabled ? 'available' : ''}`}
                onClick={async () => {
                  const snapshot = await fetchSolutionPath();
                  onOpenSolutionPath?.(snapshot || solutionPath);
                }}
                aria-label="Open current solution path"
              >
                <span>
                  <strong>Solution Path</strong>
                  <small>
                    {solutionPath?.repair_required_proposals > 0
                      ? `${solutionPath.repair_required_proposals} solution-path update(s) need attention`
                      : solutionPath?.reviewing_proposals > 0
                      ? 'Solution-path update under review'
                      : solutionPath?.queued_proposals > 0
                      ? `${solutionPath.queued_proposals} solution-path update(s) queued`
                      : solutionPath?.enabled && (solutionPath.steps?.length || 0) > 0
                      ? `${solutionPath.steps?.length || 0} steps · revision ${solutionPath.revision || 1}`
                      : getSolutionPathEmptyLabel(solutionPath)}
                  </small>
                </span>
                <span aria-hidden="true">›</span>
              </button>
            )}

            <div className="assistant-memory-bank">
              <div className="assistant-memory-bank__header">
                <div>
                  <div className="assistant-memory-bank__title">Assistant Memory Bank</div>
                  <div className="assistant-memory-bank__subtitle">
                    Latest proof supports retrieved by Assistant memory.
                  </div>
                </div>
                {assistantMemoryPack?.has_pack && (
                  <span className="assistant-memory-bank__count">
                    {assistantResults.length}/{assistantMemoryPack.max_result_count || ASSISTANT_MAX_RESULTS}
                  </span>
                )}
              </div>

              {assistantMemoryError && assistantResults.length === 0 && (
                <div className="assistant-memory-bank__empty">{assistantMemoryError}</div>
              )}

              {!assistantMemoryError && assistantResults.length === 0 && (
                <div className="assistant-memory-bank__empty">
                  No Assistant proof pack has been retrieved yet.
                </div>
              )}

              {assistantResults.length > 0 && (
                <div className="assistant-proof-list">
                  {assistantResults.map((support, index) => {
                    const supportKey = getCanonicalProofIdentity(support, { includeIndex: true, index });
                    const supportIdentity = getCanonicalProofIdentity(support);
                    const selectedKey = selectedAssistantProof ? getCanonicalProofIdentity(selectedAssistantProof) : '';
                    const isSelected = selectedKey === supportIdentity;
                    const detailId = sanitizeDomId(supportIdentity, 'assistant-proof-details');
                    const provenance = formatProofProvenance(support);
                    return (
                      <button
                        type="button"
                        key={supportKey}
                        className={`assistant-proof-tile ${getAssistantTileClass(support)} ${isSelected ? 'selected' : ''}`}
                        onClick={() => handleAssistantProofClick(support)}
                        title="Open proof history preview"
                        aria-expanded={isSelected}
                        aria-controls={detailId}
                      >
                        <div className="assistant-proof-tile__topline">
                          <span className="assistant-proof-tile__tier">{formatAssistantTierLabel(support)}</span>
                          <span className="assistant-proof-tile__source">{support.corpus}</span>
                        </div>
                        <div className="assistant-proof-tile__name">
                          {support.theorem_name || support.proof_id || 'Untitled proof'}
                        </div>
                        <div className="assistant-proof-tile__statement">
                          {truncateText(support.theorem_statement, 128)}
                        </div>
                        <div className="assistant-proof-tile__hint">
                          {[
                            ...provenance.lanes.map((lane) => `Lane: ${lane}`),
                            provenance.runId && `Run: ${provenance.runId}`,
                            provenance.sessionId && `Session: ${provenance.sessionId}`,
                            provenance.source && `Source: ${provenance.source}`,
                            provenance.omitted > 0 && `+${provenance.omitted} omitted lineage records`,
                          ].filter(Boolean).join(' · ')}
                        </div>
                        {(support.relevance_reason || support.transfer_hint) && (
                          <div className="assistant-proof-tile__hint">
                            {truncateText(support.relevance_reason || support.transfer_hint, 110)}
                          </div>
                        )}
                      </button>
                    );
                  })}
                </div>
              )}

              <div
                id={sanitizeDomId(
                  selectedAssistantProof ? getCanonicalProofIdentity(selectedAssistantProof) : '',
                  'assistant-proof-details'
                )}
                className="assistant-proof-preview"
                hidden={!selectedAssistantProof}
              >
                {selectedAssistantProof && (
                  <>
                  <div className="assistant-proof-preview__header">
                    <div>
                      <div className="assistant-proof-preview__label">Proof History Preview</div>
                      <div className="assistant-proof-preview__title">
                        {selectedProofView?.theorem_name || selectedProofView?.display_title || selectedProofView?.proof_id}
                      </div>
                    </div>
                    <button
                      type="button"
                      className="assistant-proof-preview__close"
                      onClick={() => {
                        assistantDetailGenerationRef.current += 1;
                        setSelectedAssistantProof(null);
                        setAssistantProofDetail(null);
                        setAssistantProofLoading(false);
                      }}
                    >
                      Close
                    </button>
                  </div>
                  {assistantProofLoading ? (
                    <div className="assistant-proof-preview__loading">Opening proof history...</div>
                  ) : (
                    <>
                      <div className="assistant-proof-preview__meta">
                        <span>{formatAssistantSource(selectedProofView)}</span>
                        {selectedProofView?.proof_id && <span>{selectedProofView.proof_id}</span>}
                        <span>{formatAssistantTierLabel(selectedProofView)}</span>
                        {selectedProofView?.run_id && <span>Run: {selectedProofView.run_id}</span>}
                        {selectedProofView?.session_id && <span>Session: {selectedProofView.session_id}</span>}
                        {selectedProofView?.source_type && <span>Source: {selectedProofView.source_type}{selectedProofView.source_id ? `/${selectedProofView.source_id}` : ''}</span>}
                        {(selectedProofView?.retrieval_lanes || []).map((lane) => <span key={lane}>Lane: {lane}</span>)}
                        {formatProofProvenance(selectedProofView).omitted > 0 && <span>+{formatProofProvenance(selectedProofView).omitted} omitted lineage records</span>}
                      </div>
                      <p className="assistant-proof-preview__statement">
                        {selectedProofView?.theorem_statement || 'No theorem statement available.'}
                      </p>
                      {(selectedProofView?.proof_description || selectedProofView?.formal_sketch) && (
                        <p className="assistant-proof-preview__description">
                          {selectedProofView.proof_description || selectedProofView.formal_sketch}
                        </p>
                      )}
                      {selectedProofView?.hydration_error && (
                        <div className="assistant-proof-preview__error">{selectedProofView.hydration_error}</div>
                      )}
                      {selectedProofView?.lean_code && (
                        <pre className="assistant-proof-preview__code">
                          {truncateText(selectedProofView.lean_code, 900)}
                        </pre>
                      )}
                    </>
                  )}
                  </>
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

