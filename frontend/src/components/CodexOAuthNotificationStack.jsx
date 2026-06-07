import React from 'react';

const IconX = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="18" y1="6" x2="6" y2="18" />
    <line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);

const IconKey = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="7.5" cy="15.5" r="5.5" />
    <path d="M12 12l8-8" />
    <path d="M15 4h5v5" />
  </svg>
);

export default function CodexOAuthNotificationStack({ notifications, onDismiss, onOpenCloudAccess }) {
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
        <CodexOAuthNotification
          key={notification.id}
          notification={notification}
          onDismiss={onDismiss}
          onOpenCloudAccess={onOpenCloudAccess}
        />
      ))}
    </div>
  );
}

function CodexOAuthNotification({ notification, onDismiss, onOpenCloudAccess }) {
  const [isHovered, setIsHovered] = React.useState(false);
  const [isExiting, setIsExiting] = React.useState(false);

  const handleDismiss = (event) => {
    event.stopPropagation();
    setIsExiting(true);
    setTimeout(() => {
      onDismiss(notification.id);
    }, 300);
  };

  const roleLabel = notification.role_id
    ? notification.role_id.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase())
    : 'OAuth provider';
  const providerLabel = notification.provider_label || 'OAuth';

  return (
    <div
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{
        width: '340px',
        minHeight: '116px',
        background: `linear-gradient(135deg, ${isHovered ? 'rgba(126, 58, 242, 0.98)' : 'rgba(46, 18, 82, 0.97)'}, ${isHovered ? 'rgba(88, 36, 180, 0.98)' : 'rgba(31, 14, 58, 0.97)'})`,
        backdropFilter: 'blur(8px)',
        borderRadius: '12px',
        padding: '14px',
        boxShadow: isHovered
          ? '0 20px 40px -12px rgba(126, 58, 242, 0.65), 0 0 0 1px rgba(196, 181, 253, 0.5)'
          : '0 10px 30px -12px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(196, 181, 253, 0.4)',
        border: `1px solid ${isHovered ? 'rgba(196, 181, 253, 0.75)' : 'rgba(196, 181, 253, 0.45)'}`,
        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        transform: isExiting
          ? 'translateX(-360px) scale(0.8)'
          : `scale(${isHovered ? 1.02 : 1})`,
        opacity: isExiting ? 0 : 1,
        pointerEvents: 'auto',
        animation: isExiting ? 'none' : 'codexOAuthSlideIn 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div
            style={{
              padding: '6px',
              backgroundColor: 'rgba(196, 181, 253, 0.24)',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#ddd6fe',
            }}
          >
            <IconKey />
          </div>
          <div>
            <div style={{ fontSize: '10px', color: '#ddd6fe', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: '700' }}>
              OAuth Needs Attention
            </div>
            <div style={{ fontSize: '14px', fontWeight: '700', lineHeight: '1.2', color: '#f5f3ff' }}>
              {providerLabel}
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
            color: '#ddd6fe',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <IconX />
        </button>
      </div>

      <div style={{ fontSize: '12px', fontWeight: '600', color: '#ffffff', lineHeight: '1.4', marginBottom: '6px' }}>
        {roleLabel}
      </div>
      <div style={{ fontSize: '11px', color: '#e9d5ff', lineHeight: '1.4', marginBottom: '10px' }}>
        {notification.message || `Check your ${providerLabel} OAuth connection, sign in again, and retry.`}
      </div>

      <button
        onClick={onOpenCloudAccess}
        style={{
          width: '100%',
          padding: '8px 12px',
          backgroundColor: 'rgba(255, 255, 255, 0.14)',
          border: '1px solid rgba(255, 255, 255, 0.28)',
          borderRadius: '8px',
          color: '#ffffff',
          fontSize: '12px',
          fontWeight: '700',
          cursor: 'pointer',
        }}
      >
        Open Cloud Access & Keys
      </button>

      <style>{`
        @keyframes codexOAuthSlideIn {
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
