import React from 'react';

const ALERT_SCALE = 1.4;

function scalePx(value) {
  return `${Math.round(value * ALERT_SCALE)}px`;
}

function truncate(text, maxLength = 120) {
  if (!text) {
    return '';
  }
  return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
}

export default function ProofNotificationStack({ notifications, onDismiss, onClickNotification }) {
  if (!notifications || notifications.length === 0) {
    return null;
  }

  return (
    <div
      style={{
        position: 'fixed',
        bottom: scalePx(116),
        right: '20px',
        zIndex: 999998,
        display: 'flex',
        flexDirection: 'column',
        gap: scalePx(8),
        pointerEvents: 'none',
      }}
    >
      {notifications.map((notification) => (
        <div
          key={notification.id}
          onClick={() => onClickNotification(notification.proof_id)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' || event.key === ' ') {
              event.preventDefault();
              onClickNotification(notification.proof_id);
            }
          }}
          role="button"
          tabIndex={0}
          style={{
            width: scalePx(320),
            textAlign: 'left',
            borderRadius: scalePx(14),
            border: '1.5px solid #ffd65c',
            background: 'linear-gradient(135deg, rgba(8, 35, 22, 0.96), rgba(15, 23, 42, 0.96))',
            boxShadow:
              '0 16px 36px rgba(0, 0, 0, 0.35), 0 0 12px rgba(255, 214, 92, 0.35), inset 0 0 0 1px rgba(255, 194, 57, 0.25)',
            padding: `${scalePx(14)} ${scalePx(14)} ${scalePx(12)} ${scalePx(14)}`,
            color: '#f8fafc',
            cursor: 'pointer',
            pointerEvents: 'auto',
          }}
        >
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              gap: scalePx(12),
              alignItems: 'flex-start',
            }}
          >
            <div>
              <div
                style={{
                  fontSize: scalePx(10),
                  letterSpacing: '0.08em',
                  textTransform: 'uppercase',
                  color: '#ffd65c',
                  marginBottom: scalePx(6),
                  fontWeight: 700,
                }}
              >
                Congratulations! Novel Mathematical Proof Discovered
              </div>
              <div
                style={{
                  fontSize: scalePx(13),
                  lineHeight: 1.45,
                  color: '#e2e8f0',
                  fontWeight: 500,
                }}
                title={notification.theorem_statement}
              >
                {truncate(notification.theorem_statement)}
              </div>
              <div
                style={{
                  marginTop: scalePx(8),
                  fontSize: scalePx(11),
                  lineHeight: 1.4,
                  color: '#1eff1c',
                  fontWeight: 600,
                }}
              >
                Verified by Lean 4. Click to open Mathematical Proofs.
              </div>
            </div>

            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation();
                onDismiss(notification.id);
              }}
              style={{
                border: 'none',
                background: 'transparent',
                color: '#94a3b8',
                cursor: 'pointer',
                fontSize: scalePx(16),
                lineHeight: 1,
                padding: 0,
              }}
            >
              x
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
