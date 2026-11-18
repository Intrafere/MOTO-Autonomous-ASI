import React from 'react';
import { findTokenCalibration, useTokenCalibration } from '../hooks/useTokenCalibration';

const formatTokens = (value) => Number(value).toLocaleString('en-US');

export const describeTokenCalibration = (entry, { adjustEnabled = true, contextWindow } = {}) => {
  if (!entry) return '';
  if (entry.state === 'measuring') {
    return `Checking this model's token count: ${entry.samples} of ${entry.samples_required} measurements.`;
  }
  if (entry.state === 'matches') {
    return "Token count checked: this model's count matches MOTO's estimate.";
  }
  const percent = Math.round((entry.ratio - 1) * 100);
  const lead = `Token count checked: this model counts about ${percent}% more tokens than MOTO's estimate.`;
  if (!adjustEnabled) return `${lead} The adjustment is turned off.`;
  const window = Number(contextWindow);
  if (!Number.isFinite(window) || window <= 0) return `${lead} MOTO fits prompts to a smaller window.`;
  return `${lead} MOTO fits prompts to ${formatTokens(Math.max(1, Math.floor(window / entry.ratio)))} tokens.`;
};

// Shown under a role's Context Window field once that model has been measured.
export default function TokenCountNote({ provider, modelId, contextWindow }) {
  const calibration = useTokenCalibration();
  const entry = findTokenCalibration(calibration, provider, modelId);
  if (!entry) return null;
  return (
    <small className="settings-hint" data-token-count-state={entry.state}>
      {describeTokenCalibration(entry, {
        adjustEnabled: calibration.adjust_enabled !== false,
        contextWindow,
      })}
    </small>
  );
}
