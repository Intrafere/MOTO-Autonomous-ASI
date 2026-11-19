import React from 'react';
import {
  LOCAL_SERVER_PROVIDERS,
  describeLocalFallback,
  localFallbackValue,
} from '../utils/oauthProviders';

// <option> entries for a role's local fallback: LM Studio models, then the chat
// models of each local server (vLLM). A local-server model is stored as
// "<provider>::<model>" so the backend knows which server to call. The role's
// own model is never offered as its fallback, and a saved choice stays listed
// even when its server's model list is not loaded.
export default function LocalFallbackOptions({
  lmStudioModels = [],
  modelsFor = () => [],
  provider = '',
  modelId = '',
  value = '',
}) {
  const options = lmStudioModels.map((model) => ({ value: model.id, label: model.id }));
  for (const entry of LOCAL_SERVER_PROVIDERS) {
    for (const model of modelsFor(entry.id) || []) {
      if (provider === entry.id && modelId === model.id) continue;
      options.push({
        value: localFallbackValue(entry.id, model.id),
        label: `${entry.label}: ${model.id}`,
      });
    }
  }
  const saved = value || '';
  const savedMissing = Boolean(saved) && !options.some((option) => option.value === saved);
  return (
    <>
      <option value="">No fallback</option>
      {savedMissing && (
        <option value={saved}>{describeLocalFallback(saved).text} (configured)</option>
      )}
      {options.map((option) => (
        <option key={option.value} value={option.value}>{option.label}</option>
      ))}
    </>
  );
}
