import React from 'react';

export default function ModelErrorNotificationStack({ notifications, onDismiss, embedded = false }) {
  if (!notifications?.length) return null;
  return (
    <div
      aria-live="polite"
      aria-atomic="false"
      style={{
        position: embedded ? 'static' : 'fixed',
        bottom: embedded ? undefined : 20,
        left: embedded ? undefined : 'clamp(12px, 25vw, 360px)',
        right: embedded ? undefined : 12,
        zIndex: embedded ? undefined : 999998,
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
        maxWidth: 380,
        pointerEvents: 'none',
      }}
    >
      {notifications.map(notification => (
        <div
          key={notification.notification_key}
          role="status"
          style={{
            width: '100%',
            boxSizing: 'border-box',
            padding: 16,
            borderRadius: 12,
            color: '#fff',
            background: 'linear-gradient(135deg, rgba(127, 29, 29, .98), rgba(69, 10, 10, .98))',
            border: '1px solid rgba(254, 202, 202, .55)',
            boxShadow: '0 12px 36px rgba(0,0,0,.55)',
            pointerEvents: 'auto',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
            <strong>{notification.title || 'Model error'}</strong>
            <button
              type="button"
              aria-label={`Dismiss ${notification.title || 'model error'}`}
              onClick={() => onDismiss(notification.notification_key)}
              style={{ border: 0, background: 'transparent', color: '#fff', cursor: 'pointer', fontSize: 18 }}
            >
              &times;
            </button>
          </div>
          <p style={{ fontSize: 12, lineHeight: 1.5, margin: '10px 0 6px' }}>{notification.message}</p>
          {notification.terminal_guidance && (
            <p style={{ fontSize: 11, lineHeight: 1.5, margin: 0, color: '#fecaca' }}>
              {notification.terminal_guidance}
            </p>
          )}
        </div>
      ))}
    </div>
  );
}
