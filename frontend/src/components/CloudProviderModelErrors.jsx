import React from 'react';

// One error banner per cloud provider whose credential or model list needs attention.
export default function CloudProviderModelErrors({ modelErrors = [] }) {
  if (modelErrors.length === 0) {
    return null;
  }
  return modelErrors.map((entry) => (
    <div
      key={entry.providerId}
      className="test-result-banner test-result-banner--error"
      style={{ marginBottom: '1rem' }}
      role="alert"
    >
      {entry.message}
    </div>
  ));
}
