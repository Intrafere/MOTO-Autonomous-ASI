import React from 'react';

const IconX = ({ className }) => (
  <svg className={className} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="18" y1="6" x2="6" y2="18"></line>
    <line x1="6" y1="6" x2="18" y2="18"></line>
  </svg>
);

const IconClock = ({ style }) => (
  <svg style={style} width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10"></circle>
    <polyline points="12 6 12 12 16 14"></polyline>
  </svg>
);

/**
 * Persistent notification stack for hung API connection alerts.
 * Amber-themed, stays visible until the user explicitly dismisses each notification.
 *
 * Props:
 * - notifications: Array of { id, role_id, model, provider, elapsed_minutes, message, timestamp }
 * - onDismiss: (id) => void
 */
export default function HungConnectionNotificationStack({ notifications, onDismiss }) {
  if (!notifications || notifications.length === 0) {
    return null;
  }

  return (
    <div
      style={{
        position: 'fixed',
        bottom: '20px',
        left: '360px',
        zIndex: 999999,
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
        pointerEvents: 'none',
      }}
    >
      {notifications.map((notification) => (
        <HungConnectionNotification
          key={notification.id}
          notification={notification}
          onDismiss={onDismiss}
        />
      ))}
    </div>
  );
}

function HungConnectionNotification({ notification, onDismiss }) {
  const [isHovered, setIsHovered] = React.useState(false);
  const [isExiting, setIsExiting] = React.useState(false);

  const handleDismiss = (e) => {
    e.stopPropagation();
    setIsExiting(true);
    setTimeout(() => {
      onDismiss(notification.id);
    }, 300);
  };

  const modelLabel = notification.model || 'Unknown Model';
  const providerLabel = notification.provider || 'Unknown Provider';

  return (
    <div
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{
        width: '320px',
        minHeight: '90px',
        background: `linear-gradient(135deg, ${isHovered ? 'rgba(180, 120, 20, 0.97)' : 'rgba(60, 40, 10, 0.96)'}, ${isHovered ? 'rgba(140, 90, 10, 0.97)' : 'rgba(40, 25, 5, 0.96)'})`,
        backdropFilter: 'blur(8px)',
        borderRadius: '12px',
        padding: '14px',
        boxShadow: isHovered
          ? '0 20px 40px -12px rgba(255, 165, 0, 0.6), 0 0 0 1px rgba(255, 165, 0, 0.5)'
          : '0 10px 30px -12px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(255, 165, 0, 0.4)',
        border: `1px solid ${isHovered ? 'rgba(255, 165, 0, 0.7)' : 'rgba(255, 165, 0, 0.5)'}`,
        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        transform: isExiting
          ? 'translateX(-360px) scale(0.8)'
          : `scale(${isHovered ? 1.02 : 1})`,
        opacity: isExiting ? 0 : 1,
        pointerEvents: 'auto',
        animation: isExiting ? 'none' : 'hungSlideIn 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div
            style={{
              padding: '6px',
              backgroundColor: 'rgba(255, 165, 0, 0.35)',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <IconClock style={{ color: '#ffb347' }} />
          </div>
          <div>
            <div style={{ fontSize: '10px', color: '#ffd699', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: '600' }}>
              Possible Hung Connection
            </div>
            <div style={{ fontSize: '14px', fontWeight: '700', lineHeight: '1.2', color: '#ffb347' }}>
              {notification.elapsed_minutes}+ Minutes
            </div>
          </div>
        </div>

        <button
          onClick={handleDismiss}
          style={{
            padding: '4px',
            backgroundColor: 'transparent',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            color: '#ffd699',
            transition: 'all 0.2s',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = 'rgba(255, 180, 50, 0.25)';
            e.currentTarget.style.color = '#ffffff';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'transparent';
            e.currentTarget.style.color = '#ffd699';
          }}
        >
          <IconX />
        </button>
      </div>

      {/* Model info */}
      <div
        style={{
          fontSize: '12px',
          fontWeight: '500',
          color: '#f3f4f6',
          lineHeight: '1.4',
          marginBottom: '4px',
        }}
      >
        {modelLabel} via {providerLabel}
      </div>

      {/* Message */}
      <div
        style={{
          fontSize: '11px',
          color: '#ffe0a3',
          lineHeight: '1.4',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          display: '-webkit-box',
          WebkitLineClamp: 3,
          WebkitBoxOrient: 'vertical',
        }}
      >
        Connection may be hung. Consider stopping and trying a different host/provider.
      </div>

      <style>{`
        @keyframes hungSlideIn {
          from {
            transform: translateX(-360px) scale(0.8);
            opacity: 0;
          }
          to {
            transform: translateX(0) scale(1);
            opacity: 1;
          }
        }
      `}</style>
    </div>
  );
}
