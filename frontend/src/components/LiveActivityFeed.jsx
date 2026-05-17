import React, { useEffect, useRef } from 'react';
import { getActivityClass, getActivityIcon } from '../utils/activityStyles';
import './autonomous/AutonomousResearch.css';

const formatActivityTime = (timestamp) => {
  if (!timestamp) return '';

  const parsed = new Date(timestamp);
  if (!Number.isNaN(parsed.getTime())) {
    return parsed.toLocaleTimeString();
  }

  return timestamp;
};

export default function LiveActivityFeed({
  title = 'Live Activity',
  items = [],
  emptyMessage = 'No activity yet.',
  maxItems,
  getEventName = (item) => item?.event || item?.type || '',
  getMessage = (item) => item?.message || item?.data?.message || '',
  getTimestamp = (item) => item?.timestamp || item?.fullTimestamp || '',
  getIcon = getActivityIcon,
  getClassName = getActivityClass,
  headerAction = null,
}) {
  const feedRef = useRef(null);
  const prevLengthRef = useRef(0);
  const visibleItems = maxItems ? items.slice(-maxItems) : items;

  useEffect(() => {
    if (visibleItems.length > prevLengthRef.current && feedRef.current) {
      feedRef.current.scrollTop = feedRef.current.scrollHeight;
    }
    prevLengthRef.current = visibleItems.length;
  }, [visibleItems.length]);

  return (
    <div className="activity-section">
      <div className="activity-section-header">
        <h3>{title}</h3>
        {headerAction}
      </div>
      <div className="activity-feed" ref={feedRef}>
        {visibleItems.length === 0 ? (
          <div className="activity-empty">{emptyMessage}</div>
        ) : (
          visibleItems.map((item, index) => {
            const eventName = getEventName(item);
            const timestamp = getTimestamp(item);
            const message = getMessage(item) || eventName;

            return (
              <div
                key={`${eventName}-${timestamp}-${index}`}
                className={`activity-item ${getClassName(eventName, item)}`}
              >
                <span className="activity-icon">{getIcon(eventName, item)}</span>
                <span className="activity-time">{formatActivityTime(timestamp)}</span>
                <span className="activity-message">{message}</span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
