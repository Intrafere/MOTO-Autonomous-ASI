import React, { useState } from 'react';
import { getSyntheticLibPresentation } from '../utils/syntheticLibPresentation';

function statusClass(status) {
  const normalized = String(status || '').toLowerCase();
  if (normalized === 'active' || normalized === 'ready') return 'connectivity-status--ready';
  if (normalized === 'error') return 'connectivity-status--error';
  if (normalized === 'disabled') return 'connectivity-status--disabled';
  if (normalized === 'starting' || normalized === 'not ready' || normalized === 'outdated' || normalized === 'coming soon') return 'connectivity-status--pending';
  return 'connectivity-status--inactive';
}

function StatusPill({ status }) {
  const label = status || 'inactive';
  return <span className={`connectivity-status ${statusClass(label)}`}>{label}</span>;
}

// Each row is a plain container: the label + status is one real button that
// opens the row's dialog, and any enable checkbox is a sibling control rather
// than a nested one. The Coming soon X is decorative and hidden from assistive
// technology; its meaning is in the button's name.
function ConnectivityRow({
  label,
  status,
  onOpen,
  checked,
  onToggle,
  checkbox = false,
  checkboxVariant = 'check',
  disabled = false,
  title = '',
}) {
  return (
    <div className="connectivity-row">
      {checkbox && checkboxVariant === 'x' && (
        <span
          aria-hidden="true"
          className="connectivity-checkbox connectivity-checkbox--x"
          data-testid="connectivity-coming-soon-indicator"
          onClick={onOpen}
        />
      )}
      {checkbox && checkboxVariant !== 'x' && (
        <input
          type="checkbox"
          checked={Boolean(checked)}
          disabled={disabled}
          onChange={() => onToggle?.(!checked)}
          className="connectivity-checkbox"
          aria-label={`Enable ${label}`}
        />
      )}
      <button
        type="button"
        className="connectivity-row__button"
        onClick={onOpen}
        aria-haspopup="dialog"
        title={title || label}
      >
        <span className="connectivity-label">{label}</span>
        <StatusPill status={status} />
      </button>
    </div>
  );
}

export default function ConnectivityPanel({
  appMode,
  developerModeEnabled,
  connectivityStatus,
  capabilities,
  onModeChange,
  onOpenOpenRouterOAuth,
  onOpenLmStudio,
  onOpenSyntheticLib4,
  onOpenAgentMemory,
  onOpenWolfram,
  onToggleAgentMemory,
  onToggleWolfram,
  anyWorkflowRunning = false,
}) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const isStarting = !connectivityStatus;
  const inference = connectivityStatus?.inference || {};
  const skills = connectivityStatus?.skills || {};
  const openRouterStatus = inference.openrouter_oauth?.status || (isStarting ? 'starting' : 'inactive');
  const lmStudioStatus = capabilities?.lmStudioEnabled === false
    ? 'inactive'
    : (inference.lm_studio?.status || (isStarting ? 'starting' : 'inactive'));
  const memoryStatus = skills.agent_conversation_memory || {};
  const wolframStatus = skills.wolfram_alpha || {};
  const skillFallbackStatus = isStarting ? 'starting' : 'inactive';
  const syntheticLibPresentation = getSyntheticLibPresentation(
    capabilities?.syntheticLibProductState,
    developerModeEnabled,
    capabilities?.genericMode,
  );

  return (
    <div className={`connectivity-panel ${isCollapsed ? 'connectivity-panel--collapsed' : ''}`}>
      <button
        type="button"
        className="connectivity-collapse-btn"
        onClick={() => setIsCollapsed((current) => !current)}
        aria-expanded={!isCollapsed}
        aria-label={isCollapsed ? 'Expand change mode and connectivity menu' : 'Collapse change mode and connectivity menu'}
        title={isCollapsed ? 'Expand menu' : 'Collapse menu'}
      >
        {isCollapsed ? '+' : '-'}
      </button>

      {!isCollapsed && (
        <>
          <div className="mode-switch-control mode-switch-control--compact">
            <label className="mode-switch-label" htmlFor="app-mode-select">
              Features
            </label>
            <select
              id="app-mode-select"
              className="mode-switch-select"
              value={appMode}
              onChange={(event) => onModeChange(event.target.value)}
            >
              <option value="autonomous">Autonomous S.T.E.M. ASI</option>
              <option value="manual">Advanced Manual S.T.E.M. ASI</option>
              {developerModeEnabled && (
                <option value="leanoj">LeanOJ Proof Solver</option>
              )}
            </select>
          </div>

          <section className="connectivity-section">
            <div className="connectivity-section__title">Inference Connectivity</div>
            <ConnectivityRow
              label="OpenRouter & Cloud Subscriptions"
              status={openRouterStatus}
              onOpen={onOpenOpenRouterOAuth}
              title="Configure OpenRouter, OAuth providers, and direct cloud provider API keys"
            />
            <ConnectivityRow
              label="LM Studio"
              status={lmStudioStatus}
              onOpen={onOpenLmStudio}
              title="Inspect LM Studio server and model readiness"
            />
          </section>

          <section className="connectivity-section">
            <div className="connectivity-section__title">Skills And Enhancements Connectivity</div>
            {syntheticLibPresentation.showComingSoon ? (
              <ConnectivityRow
                checkbox
                checkboxVariant="x"
                label="SyntheticLib4"
                status="Coming soon"
                onOpen={onOpenSyntheticLib4}
                title="SyntheticLib4 is coming soon"
              />
            ) : (
              <ConnectivityRow
                label={syntheticLibPresentation.isPrivatePreview
                  ? 'SyntheticLib4 — Private development'
                  : 'SyntheticLib4'}
                status={syntheticLibPresentation.isPrivatePreview ? 'Developer preview' : 'Integration'}
                onOpen={onOpenSyntheticLib4}
                title={syntheticLibPresentation.isPrivatePreview
                  ? 'Open the private SyntheticLib development workspace'
                  : 'Open SyntheticLib integration'}
              />
            )}
            <ConnectivityRow
              checkbox
              label="Wolfram Alpha"
              status={wolframStatus.status || skillFallbackStatus}
              checked={wolframStatus.enabled}
              disabled={anyWorkflowRunning}
              onToggle={onToggleWolfram}
              onOpen={onOpenWolfram}
              title={anyWorkflowRunning ? 'Stop the active workflow before changing run-level feature toggles' : 'Configure Wolfram Alpha App ID and tool availability'}
            />
            <ConnectivityRow
              checkbox
              label="Session History Memory"
              status={memoryStatus.status || (isStarting ? 'starting' : 'disabled')}
              checked={memoryStatus.enabled}
              disabled={anyWorkflowRunning}
              onToggle={onToggleAgentMemory}
              onOpen={onOpenAgentMemory}
              title={anyWorkflowRunning ? 'Stop the active workflow before changing run-level feature toggles' : 'Configure local proof-history memory for Assistant workflow-memory search'}
            />
          </section>

          {developerModeEnabled && (
            <div className="connectivity-dev-indicator" title="Developer mode settings are enabled.">
              Developer Mode
            </div>
          )}
        </>
      )}
    </div>
  );
}

