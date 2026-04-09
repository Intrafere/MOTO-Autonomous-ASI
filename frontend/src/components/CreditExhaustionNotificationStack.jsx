import React from 'react';
import { openRouterAPI } from '../services/api';

const IconX = ({ className }) => (
  <svg className={className} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="18" y1="6" x2="6" y2="18"></line>
    <line x1="6" y1="6" x2="18" y2="18"></line>
  </svg>
);

const IconAlert = ({ className, style }) => (
  <svg className={className} style={style} width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
    <line x1="12" y1="9" x2="12" y2="13"></line>
    <line x1="12" y1="17" x2="12.01" y2="17"></line>
  </svg>
);

/**
 * Persistent notification stack for OpenRouter credit exhaustion alerts.
 * Red-themed, stays visible until the user explicitly dismisses each notification.
 * Includes a "Retry OpenRouter" button that resets exhaustion flags so roles resume.
 *
 * Props:
 * - notifications: Array of { id, role_id, message, reason, timestamp }
 * - onDismiss: (id) => void
 * - onDismissAll: () => void
 */
export default function CreditExhaustionNotificationStack({ notifications, onDismiss, onDismissAll }) {
  const [resetting, setResetting] = React.useState(false);
  const [resetResult, setResetResult] = React.useState(null);

  if (!notifications || notifications.length === 0) {
    return null;
  }

  const handleRetryOpenRouter = async () => {
    setResetting(true);
    setResetResult(null);
    try {
      const result = await openRouterAPI.resetCreditExhaustion();
      setResetResult({ success: true, message: result.message });
      setTimeout(() => {
        if (onDismissAll) onDismissAll();
        setResetResult(null);
      }, 2000);
    } catch (err) {
      setResetResult({ success: false, message: err.message || 'Reset failed' });
    } finally {
      setResetting(false);
    }
  };

  return (
    <div
      style={{
        position: 'fixed',
        bottom: '20px',
        left: '20px',
        zIndex: 999999,
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
        pointerEvents: 'none',
      }}
    >
      {notifications.map((notification) => (
        <CreditExhaustionNotification
          key={notification.id}
          notification={notification}
          onDismiss={onDismiss}
        />
      ))}

      {/* Retry OpenRouter button */}
      <div style={{ pointerEvents: 'auto' }}>
        {resetResult && (
          <div style={{
            padding: '8px 12px',
            borderRadius: '8px',
            fontSize: '11px',
            fontWeight: '500',
            textAlign: 'center',
            backgroundColor: resetResult.success ? 'rgba(76, 175, 80, 0.15)' : 'rgba(244, 67, 54, 0.15)',
            color: resetResult.success ? '#4CAF50' : '#f44336',
            border: `1px solid ${resetResult.success ? 'rgba(76, 175, 80, 0.4)' : 'rgba(244, 67, 54, 0.4)'}`,
          }}>
            {resetResult.message}
          </div>
        )}
        <button
          onClick={handleRetryOpenRouter}
          disabled={resetting}
          style={{
            width: '320px',
            padding: '10px 16px',
            backgroundColor: resetting ? '#333' : 'rgba(108, 92, 231, 0.9)',
            border: '1px solid rgba(108, 92, 231, 0.6)',
            borderRadius: '10px',
            color: '#fff',
            fontSize: '13px',
            fontWeight: '600',
            cursor: resetting ? 'not-allowed' : 'pointer',
            opacity: resetting ? 0.6 : 1,
            transition: 'all 0.2s',
          }}
          onMouseEnter={(e) => { if (!resetting) e.currentTarget.style.backgroundColor = 'rgba(108, 92, 231, 1)'; }}
          onMouseLeave={(e) => { if (!resetting) e.currentTarget.style.backgroundColor = 'rgba(108, 92, 231, 0.9)'; }}
        >
          {resetting ? 'Resetting...' : 'Retry OpenRouter (Credits Added)'}
        </button>
      </div>
    </div>
  );
}

function CreditExhaustionNotification({ notification, onDismiss }) {
  const [isHovered, setIsHovered] = React.useState(false);
  const [isExiting, setIsExiting] = React.useState(false);

  const handleDismiss = (e) => {
    e.stopPropagation();
    setIsExiting(true);
    setTimeout(() => {
      onDismiss(notification.id);
    }, 300);
  };

  const roleLabel = notification.role_id
    ? notification.role_id.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
    : 'Unknown Role';

  const isNoFallback = notification.reason === 'no_fallback_configured';

  return (
    <div
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{
        width: '320px',
        minHeight: '90px',
        background: `linear-gradient(135deg, ${isHovered ? 'rgba(180, 30, 30, 0.97)' : 'rgba(60, 15, 15, 0.96)'}, ${isHovered ? 'rgba(140, 20, 20, 0.97)' : 'rgba(40, 10, 10, 0.96)'})`,
        backdropFilter: 'blur(8px)',
        borderRadius: '12px',
        padding: '14px',
        boxShadow: isHovered
          ? '0 20px 40px -12px rgba(231, 76, 60, 0.6), 0 0 0 1px rgba(231, 76, 60, 0.5)'
          : '0 10px 30px -12px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(231, 76, 60, 0.4)',
        border: `1px solid ${isHovered ? 'rgba(231, 76, 60, 0.7)' : 'rgba(231, 76, 60, 0.5)'}`,
        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        transform: isExiting
          ? 'translateX(-360px) scale(0.8)'
          : `scale(${isHovered ? 1.02 : 1})`,
        opacity: isExiting ? 0 : 1,
        pointerEvents: 'auto',
        animation: isExiting ? 'none' : 'creditSlideIn 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div
            style={{
              padding: '6px',
              backgroundColor: 'rgba(231, 76, 60, 0.35)',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <IconAlert style={{ color: '#ff6b6b' }} />
          </div>
          <div>
            <div style={{ fontSize: '10px', color: '#ff9999', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: '600' }}>
              Credits Exhausted
            </div>
            <div style={{ fontSize: '14px', fontWeight: '700', lineHeight: '1.2', color: '#ff6b6b' }}>
              OpenRouter
            </div>
          </div>
        </div>

        {/* Dismiss button */}
        <button
          onClick={handleDismiss}
          style={{
            padding: '4px',
            backgroundColor: 'transparent',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            color: '#ff9999',
            transition: 'all 0.2s',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = 'rgba(255, 100, 100, 0.25)';
            e.currentTarget.style.color = '#ffffff';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'transparent';
            e.currentTarget.style.color = '#ff9999';
          }}
        >
          <IconX />
        </button>
      </div>

      {/* Role info */}
      <div
        style={{
          fontSize: '12px',
          fontWeight: '500',
          color: '#f3f4f6',
          lineHeight: '1.4',
          marginBottom: '4px',
        }}
      >
        {roleLabel}
      </div>

      {/* Message */}
      <div
        style={{
          fontSize: '11px',
          color: '#ffbbbb',
          lineHeight: '1.4',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          display: '-webkit-box',
          WebkitLineClamp: 3,
          WebkitBoxOrient: 'vertical',
        }}
      >
        {isNoFallback
          ? 'No LM Studio fallback configured. This role has stopped. Configure a fallback model or add credits.'
          : notification.fallback_model
            ? `Fell back to LM Studio model: ${notification.fallback_model}`
            : (notification.message || 'OpenRouter credits have been exhausted for this role.')}
      </div>

      <style>{`
        @keyframes creditSlideIn {
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
