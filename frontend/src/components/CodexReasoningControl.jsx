import React from 'react';
import { getCodexReasoningInfo } from '../utils/codexReasoning';

export default function CodexReasoningControl({ model, modelId, value = 'auto', disabled = false, onChange }) {
  const info = getCodexReasoningInfo(model, modelId);
  const fixed = info.status === 'fixed';
  const unavailable = info.status === 'unknown' || info.status === 'unsupported';
  const selected = fixed ? 'high' : value;
  return <div className="settings-row">
    <label>Reasoning Effort</label>
    <select aria-label="Codex Reasoning Effort" value={selected} disabled={disabled || fixed || unavailable} onChange={event => onChange(event.target.value)}>
      {!fixed && <option value="auto">Auto{info.maximum ? ` (Maximum: ${info.maximum})` : ''}</option>}
      {info.levels.map(level => <option key={level} value={level}>{level === 'xhigh' ? 'Extra High' : level[0].toUpperCase() + level.slice(1)}</option>)}
      {!fixed && selected !== 'auto' && !info.levels.includes(selected) && <option value={selected}>{selected} (not supported by current catalog)</option>}
    </select>
    <small>{fixed ? 'This Spark High alias always uses High reasoning.' : info.status === 'unknown' ? 'Reasoning capabilities are unknown. Refresh the model catalog; saved settings are preserved.' : info.status === 'unsupported' ? 'This model does not support configurable reasoning effort.' : 'Auto uses the highest reasoning effort supported by this Codex model.'}</small>
  </div>;
}
