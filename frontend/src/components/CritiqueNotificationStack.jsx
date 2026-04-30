import React from 'react';

const ALERT_SCALE = 1.4;

function scalePx(value) {
  return `${Math.round(value * ALERT_SCALE)}px`;
}

function scaleNumber(value) {
  return Math.round(value * ALERT_SCALE);
}

// Simple inline icon components
const IconX = ({ className, size = 16 }) => (
  <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="18" y1="6" x2="6" y2="18"></line>
    <line x1="6" y1="6" x2="18" y2="18"></line>
  </svg>
);

const IconStar = ({ className, size = 16 }) => (
  <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
  </svg>
);

/**
 * Get color classes based on average rating
 */
function getRatingColor(rating) {
  if (rating >= 8) return { text: '#34d399', bg: '#10b981' };
  if (rating >= 6.25) return { text: '#18cc17', bg: '#18cc17' };
  return { text: '#9ca3af', bg: '#6b7280' };
}

/**
 * Persistent notification stack for high-scoring paper critiques (>= 6.25).
 * 
 * Features:
 * - Fixed position (bottom-right corner)
 * - Max 3 notifications (FIFO queue)
 * - Click to open critique modal
 * - X button to dismiss
 * - Persists across screens (not localStorage)
 * 
 * Props:
 * - notifications: Array of notification objects { id, paper_id, paper_title, average_rating, timestamp }
 * - onDismiss: (id) => void - callback when notification is dismissed
 * - onClickNotification: (paper_id, paper_title) => void - callback when notification is clicked
 */
export default function CritiqueNotificationStack({ notifications, onDismiss, onClickNotification }) {
  if (!notifications || notifications.length === 0) {
    return null;
  }

  return (
    <div
      style={{
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        zIndex: 999999,
        display: 'flex',
        flexDirection: 'column',
        gap: scalePx(8),
        pointerEvents: 'none', // Allow clicks through container
      }}
    >
      {notifications.map((notification, index) => (
        <CritiqueNotification
          key={notification.id}
          notification={notification}
          index={index}
          onDismiss={onDismiss}
          onClickNotification={onClickNotification}
        />
      ))}
    </div>
  );
}

/**
 * Individual notification card
 */
function CritiqueNotification({ notification, index, onDismiss, onClickNotification }) {
  const colors = getRatingColor(notification.average_rating);
  const [isHovered, setIsHovered] = React.useState(false);
  const [isExiting, setIsExiting] = React.useState(false);

  const handleDismiss = (e) => {
    e.stopPropagation(); // Prevent triggering onClick
    setIsExiting(true);
    // Wait for animation to complete before calling onDismiss
    setTimeout(() => {
      onDismiss(notification.id);
    }, 300);
  };

  const handleClick = () => {
    onClickNotification(notification.paper_id, notification.paper_title);
  };

  return (
    <div
      onClick={handleClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{
        width: scalePx(280),
        minHeight: scalePx(80),
        background: `linear-gradient(135deg, ${isHovered ? 'rgba(20, 83, 45, 0.95)' : 'rgba(26, 26, 46, 0.95)'}, ${isHovered ? 'rgba(15, 60, 15, 0.95)' : 'rgba(17, 24, 39, 0.95)'})`,
        backdropFilter: 'blur(8px)',
        borderRadius: scalePx(12),
        padding: scalePx(12),
        boxShadow: isHovered 
          ? '0 20px 40px -12px rgba(30, 255, 28, 0.5), 0 0 0 1px rgba(30, 255, 28, 0.5)'
          : '0 10px 30px -12px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(30, 255, 28, 0.3)',
        border: `1px solid ${isHovered ? 'rgba(30, 255, 28, 0.6)' : 'rgba(30, 255, 28, 0.4)'}`,
        cursor: 'pointer',
        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        transform: isExiting 
          ? `translateX(${scalePx(320)}) scale(0.8)` 
          : `translateY(${index * 0}px) scale(${isHovered ? 1.02 : 1})`,
        opacity: isExiting ? 0 : 1,
        pointerEvents: 'auto', // Re-enable clicks for notification
        animation: isExiting ? 'none' : 'slideIn 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
      }}
    >
      {/* Header with star icon and rating */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: scalePx(8) }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: scalePx(8) }}>
          <div
            style={{
              padding: scalePx(6),
              backgroundColor: 'rgba(30, 255, 28, 0.2)',
              borderRadius: scalePx(8),
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <div style={{ color: colors.text, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <IconStar size={scaleNumber(16)} />
            </div>
          </div>
          <div>
            <div style={{ fontSize: scalePx(10), color: '#9ca3af', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              High Score
            </div>
            <div style={{ color: colors.text, fontSize: scalePx(18), fontWeight: '700', lineHeight: '1' }}>
              {notification.average_rating.toFixed(1)}/10
            </div>
          </div>
        </div>
        
        {/* Dismiss button */}
        <button
          onClick={handleDismiss}
          style={{
            padding: scalePx(4),
            backgroundColor: 'transparent',
            border: 'none',
            borderRadius: scalePx(6),
            cursor: 'pointer',
            color: '#9ca3af',
            transition: 'all 0.2s',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          onMouseEnter={(e) => {
            e.target.style.backgroundColor = 'rgba(75, 85, 99, 0.5)';
            e.target.style.color = '#f3f4f6';
          }}
          onMouseLeave={(e) => {
            e.target.style.backgroundColor = 'transparent';
            e.target.style.color = '#9ca3af';
          }}
        >
          <IconX size={scaleNumber(16)} />
        </button>
      </div>

      {/* Paper title */}
      <div
        style={{
          fontSize: scalePx(13),
          fontWeight: '500',
          color: '#f3f4f6',
          lineHeight: '1.4',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          display: '-webkit-box',
          WebkitLineClamp: 2,
          WebkitBoxOrient: 'vertical',
        }}
        title={notification.paper_title}
      >
        {notification.paper_title}
      </div>

      {/* Click hint */}
      <div
        style={{
          fontSize: scalePx(10),
          color: '#1eff1c',
          marginTop: scalePx(6),
          opacity: isHovered ? 1 : 0.7,
          transition: 'opacity 0.2s',
        }}
      >
        Click to view critique
      </div>

      {/* Keyframes for slide-in animation */}
      <style>{`
        @keyframes slideIn {
          from {
            transform: translateX(${scalePx(320)}) scale(0.8);
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

