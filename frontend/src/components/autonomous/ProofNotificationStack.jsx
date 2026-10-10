import React from 'react';
import { getNoveltyGroupSize } from '../../utils/proofPresentation';

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
  major_mathematical_discovery: {
    borderColor: '#f8fafc',
    glowColor: 'rgba(226, 232, 240, 0.55)',
    glowInset: 'rgba(255, 255, 255, 0.32)',
    labelColor: '#f8fafc',
    background: 'linear-gradient(135deg, rgb(30, 41, 59), rgb(15, 23, 42), rgb(51, 65, 85))',
    label: 'Congratulations!\nMajor Mathematical Discovery Found!',
    subLabel:
      'Your validator deemed this may be competitive for a major prize or medal in a related field.',
  },
  novel_formulation: {
    borderColor: '#cd7f32',
    glowColor: 'rgba(205, 127, 50, 0.35)',
    glowInset: 'rgba(180, 100, 30, 0.25)',
    labelColor: '#e8a060',
    label: 'Novel Formalization Discovered',
    subLabel:
      'Your validator deemed this a citable formulation or formalization absent from standard references or Mathlib.',
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
    label: 'Minor Mathematical Discovery Found!',
    subLabel:
      'Your validator has determined this proof is a mathematical discovery or a novel alternative proof that changes our understanding.',
  },
};

function getTierStyle(tier) {
  return TIER_STYLES[tier] || TIER_STYLES.mathematical_discovery;
}

export default function ProofNotificationStack({ notifications, onDismiss, onClickNotification, panelCollapsed }) {
  if (!notifications || notifications.length === 0) {
    return null;
  }

  return (
    <div
      style={{
        position: 'fixed',
        bottom: scalePx(116),
        right: panelCollapsed ? '20px' : '340px',
        zIndex: 999998,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'flex-end',
        gap: scalePx(8),
        maxHeight: `calc(100vh - ${scalePx(132)})`,
        overflowY: 'auto',
        overflowX: 'visible',
        padding: scalePx(4),
        overscrollBehavior: 'contain',
        pointerEvents: 'auto',
        transition: 'right 0.15s ease',
      }}
      aria-label="Proof discovery notifications"
      tabIndex={0}
    >
      {notifications.map((notification) => {
        const tier = getTierStyle(notification.novelty_tier);
        const isGroup = notification.novelty_scope === 'group';
        const groupSize = getNoveltyGroupSize(notification);
        return (
          <div
            key={notification.id}
            onClick={() => onClickNotification(notification)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                onClickNotification(notification);
              }
            }}
            role="button"
            tabIndex={0}
            style={{
              width: scalePx(320),
              textAlign: 'left',
              borderRadius: scalePx(14),
              border: `1.5px solid ${tier.borderColor}`,
              background: tier.background || 'linear-gradient(135deg, rgba(8, 35, 22, 0.96), rgba(15, 23, 42, 0.96))',
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
              <div
                style={{
                  flex: '1 1 auto',
                  minWidth: 0,
                }}
              >
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
                    color: isGroup ? '#c4b5fd' : '#bfdbfe',
                    marginBottom: scalePx(5),
                    fontWeight: 700,
                  }}
                >
                  {isGroup
                    ? `Ranking applies to a finite verified group${groupSize ? ` of ${groupSize} proofs` : ''}.`
                    : 'Ranking applies to this accepted Lean statement alone.'}
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
                  {isGroup
                    ? 'The validator assigned this tier to the listed proofs and their exact combined claim, not to this proof by itself.'
                    : tier.subLabel}
                </div>
                {isGroup && notification.novelty_group_claim && (
                  <div
                    style={{
                      fontSize: scalePx(10),
                      lineHeight: 1.4,
                      color: '#cbd5e1',
                      marginBottom: scalePx(6),
                    }}
                    title={notification.novelty_group_claim}
                  >
                    Combined claim: {truncate(notification.novelty_group_claim, 180)}
                  </div>
                )}
                <div
                  style={{
                    fontSize: scalePx(13),
                    lineHeight: 1.45,
                    color: '#e2e8f0',
                    fontWeight: 500,
                    overflowWrap: 'anywhere',
                    wordBreak: 'break-word',
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
                aria-label="Dismiss proof notification"
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
                  padding: scalePx(4),
                  margin: `-${scalePx(4)}`,
                  flex: '0 0 auto',
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
