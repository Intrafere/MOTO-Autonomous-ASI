import React from 'react';
import { LOCAL_SERVER_PROVIDERS } from '../utils/oauthProviders';

// One provider button per local inference server in the registry (vLLM), shown
// beside LM Studio in every role's provider toggle. A server that is not set up
// yet stays disabled and says where to add it.
export default function LocalServerProviderButtons({
  provider,
  statusByProvider = {},
  disabled = false,
  onSelect,
}) {
  return LOCAL_SERVER_PROVIDERS.map((entry) => {
    const configured = Boolean(statusByProvider?.[entry.id]?.configured);
    return (
      <button
        key={entry.id}
        type="button"
        className={`provider-toggle-btn${provider === entry.id ? ' active-lm' : ''}`}
        disabled={disabled || !configured}
        style={!configured ? { color: '#666' } : undefined}
        onClick={() => {
          if (configured && provider !== entry.id) {
            onSelect(entry.id);
          }
        }}
        title={configured ? `Use your ${entry.label} server` : entry.unavailableTitle}
      >
        {entry.label}
      </button>
    );
  });
}
