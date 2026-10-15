import React from 'react';
import CodexReasoningControl from './CodexReasoningControl';
import { normalizeOpenRouterReasoningEffort } from '../utils/openRouterSelection';
import { getCloudAccessProvider } from '../utils/oauthProviders';

// Reasoning-effort selector for a cloud subscription provider, driven by its
// registry entry. Renders nothing for providers or models without the control.
export default function CloudReasoningControl({
  provider,
  modelId,
  models = [],
  value,
  disabled = false,
  onChange,
}) {
  const entry = getCloudAccessProvider(provider);
  const reasoning = entry?.reasoning;
  if (!reasoning || !modelId) {
    return null;
  }
  if (reasoning.control === 'codex') {
    return (
      <CodexReasoningControl
        model={models.find((item) => item.id === modelId)}
        modelId={modelId}
        value={value}
        disabled={disabled}
        onChange={onChange}
      />
    );
  }
  if (reasoning.noEffortModels?.includes(modelId)) {
    return (
      <div className="settings-row">
        <label>Reasoning Effort</label>
        <small className="settings-hint">This model always reasons; it has no effort setting.</small>
      </div>
    );
  }
  return (
    <div className="settings-row">
      <label>Reasoning Effort</label>
      <select
        aria-label={reasoning.ariaLabel || `${entry.label} Reasoning Effort`}
        value={normalizeOpenRouterReasoningEffort(value, provider)}
        onChange={(event) => onChange(event.target.value)}
        disabled={disabled}
      >
        {reasoning.options.map((option) => (
          <option key={option.value} value={option.value}>{option.label}</option>
        ))}
      </select>
      {reasoning.hint && <small className="settings-hint">{reasoning.hint}</small>}
    </div>
  );
}
