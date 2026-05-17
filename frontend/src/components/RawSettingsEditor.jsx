import React from 'react';

export default function RawSettingsEditor({
  value,
  onChange,
  onSave,
  message,
  disabled = false,
}) {
  return (
    <div className="settings-group">
      <h4>Raw Settings JSON</h4>
      <p className="settings-info">
        Edit the full settings payload directly. Save only valid JSON.
      </p>
      <textarea
        className="textarea-dark-mono"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        disabled={disabled}
        spellCheck={false}
        style={{ minHeight: '440px' }}
      />
      <div className="actions-row">
        <span className={message?.startsWith('Saved') ? 'status-success-text' : 'error-text'}>
          {message}
        </span>
        <button
          type="button"
          className="btn-success-sm"
          onClick={onSave}
          disabled={disabled}
        >
          Save Raw Settings
        </button>
      </div>
    </div>
  );
}
