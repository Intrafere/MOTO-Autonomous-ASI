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

const TIER_STYLES = {
  novel_formulation: {
    borderColor: '#cd7f32',
    glowColor: 'rgba(205, 127, 50, 0.35)',
    glowInset: 'rgba(180, 100, 30, 0.25)',
    labelColor: '#e8a060',
    label: 'Novel Formalization Discovered',
    subLabel:
      'Your validator has determined this is the first-of-its-kind Lean 4 formalization for this historically known proof.',
  },
  novel_variant: {
    borderColor: '#c0c0c0',
    glowColor: 'rgba(192, 192, 192, 0.35)',
    glowInset: 'rgba(160, 160, 160, 0.25)',
    labelColor: '#d8d8d8',
    label: 'Novel Reformulation Discovered',
    subLabel:
      'Your validator has determined this proof is a novel reformulation of a historically known proof.',
  },
  mathematical_discovery: {
    borderColor: '#ffd65c',
    glowColor: 'rgba(255, 214, 92, 0.35)',
    glowInset: 'rgba(255, 194, 57, 0.25)',
    labelColor: '#ffd65c',
    label: 'Congratulations!\nMathematical Discovery Found!',
    subLabel:
      'Your validator has determined this proof is a mathematical discovery or a novel alternative proof that changes our understanding.',
  },
};

function getTierStyle(tier) {
  return TIER_STYLES[tier] || TIER_STYLES.mathematical_discovery;
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
      {notifications.map((notification) => {
        const tier = getTierStyle(notification.novelty_tier);
        return (
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
              border: `1.5px solid ${tier.borderColor}`,
              background: 'linear-gradient(135deg, rgba(8, 35, 22, 0.96), rgba(15, 23, 42, 0.96))',
              boxShadow: `0 16px 36px rgba(0, 0, 0, 0.35), 0 0 12px ${tier.glowColor}, inset 0 0 0 1px ${tier.glowInset}`,
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
                    color: tier.labelColor,
                    marginBottom: scalePx(4),
                    fontWeight: 700,
                    whiteSpace: 'pre-line',
                  }}
                >
                  {tier.label}
                </div>
                <div
                  style={{
                    fontSize: scalePx(10),
                    lineHeight: 1.4,
                    color: '#94a3b8',
                    marginBottom: scalePx(6),
                    fontStyle: 'italic',
                  }}
                >
                  {tier.subLabel}
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
        );
      })}
    </div>
  );
}
