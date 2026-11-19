import React, { useId, useState } from 'react';
import { getSyntheticLibPresentation } from '../utils/syntheticLibPresentation';
import { AUTO_TRANSLATE_NAME, useAutoTranslate } from '../hooks/useAutoTranslate';
import UsageLevelBadge from './UsageLevelBadge';

const PENDING_STATUSES = new Set([
  'starting',
  'not ready',
  'outdated',
  'coming soon',
  'translating',
  'paused',
  'waiting',
]);

function statusClass(status) {
  const normalized = String(status || '').toLowerCase();
  if (normalized === 'active' || normalized === 'ready') return 'connectivity-status--ready';
  if (normalized === 'error' || normalized === 'repair required') return 'connectivity-status--error';
  if (normalized === 'disabled') return 'connectivity-status--disabled';
  if (PENDING_STATUSES.has(normalized)) return 'connectivity-status--pending';
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
  usageFeatureId = '',
  usageContext,
}) {
  const usageId = useId();
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
        aria-describedby={usageFeatureId ? usageId : undefined}
        title={title || label}
      >
        <span className="connectivity-label">{label}</span>
        {/* The usage badge sits left of the status pill so every row's pill
            stays right-aligned. It is hidden from the button's name and read
            as its description instead. */}
        <span className="connectivity-row__indicators">
          {usageFeatureId && (
            <span aria-hidden="true" className="connectivity-row__usage">
              <UsageLevelBadge id={usageId} compact featureId={usageFeatureId} context={usageContext} />
            </span>
          )}
          <StatusPill status={status} />
        </span>
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
  onOpenVllm,
  onOpenSyntheticLib4,
  onOpenAgentMemory,
  onOpenWolfram,
  onOpenAutoTranslate,
  onToggleAgentMemory,
  onToggleWolfram,
  onToggleAutoTranslate,
  anyWorkflowRunning = false,
}) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const { status: translateStatus } = useAutoTranslate();
  const isStarting = !connectivityStatus;
  const inference = connectivityStatus?.inference || {};
  const skills = connectivityStatus?.skills || {};
  const openRouterStatus = inference.openrouter_oauth?.status || (isStarting ? 'starting' : 'inactive');
  const lmStudioStatus = capabilities?.lmStudioEnabled === false
    ? 'inactive'
    : (inference.lm_studio?.status || (isStarting ? 'starting' : 'inactive'));
  const vllmEnabled = !capabilities?.genericMode && capabilities?.vllmAvailable !== false;
  const vllmStatus = inference.vllm?.status || (isStarting ? 'starting' : 'inactive');
  const memoryStatus = skills.agent_conversation_memory || {};
  const wolframStatus = skills.wolfram_alpha || {};
  const autoTranslateStatus = skills.auto_translate || {};
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
            {vllmEnabled && (
              <ConnectivityRow
                label="vLLM"
                status={vllmStatus}
                onOpen={onOpenVllm}
                title="Set up vLLM servers for chat models and RAG embeddings"
              />
            )}
          </section>

          <section className="connectivity-section">
            <div className="connectivity-section__title">Skills And Enhancements</div>
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
              usageFeatureId="skills.wolfram_alpha"
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
              usageFeatureId="skills.session_history_memory"
              title={anyWorkflowRunning ? 'Stop the active workflow before changing run-level feature toggles' : 'Configure local proof-history memory for Assistant workflow-memory search'}
            />
            {/* Translation runs beside workflows as a background service, so its
                toggle stays available while a workflow is running. */}
            <ConnectivityRow
              checkbox
              label={AUTO_TRANSLATE_NAME}
              status={autoTranslateStatus.status || (isStarting ? 'starting' : 'disabled')}
              checked={autoTranslateStatus.enabled}
              onToggle={onToggleAutoTranslate}
              onOpen={onOpenAutoTranslate}
              usageFeatureId="translation.auto_translate"
              usageContext={{
                batch: translateStatus?.batch_translation === true,
                translateAll: Boolean(translateStatus?.translate_all?.requested),
              }}
              title="Translate papers, proofs, and other outputs into your chosen languages with Top-p Exploration"
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

