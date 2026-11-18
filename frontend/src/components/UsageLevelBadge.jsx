import React from 'react';
import {
  USAGE_LEVEL_MAX_RANK,
  describeUsage,
  getUsageLevel,
  resolveFeatureUsage,
} from '../utils/usageLevels';
import './UsageLevelBadge.css';

// Relative model-usage meter: ascending bars plus a text label, so the level
// never depends on color alone. Pass `featureId` for a registered feature, or
// `level` (and optional `reason`) for an ad-hoc indicator. Pass `id` so a
// control can reference the badge with aria-describedby. `compact` hides the
// visible text label for tight spaces; the bar count still shows the level and
// the hover text and screen-reader text still name it.
export default function UsageLevelBadge({
  featureId,
  level: levelId,
  reason,
  context,
  id,
  compact = false,
  className = '',
}) {
  const resolved = featureId
    ? resolveFeatureUsage(featureId, context)
    : (getUsageLevel(levelId) ? { level: getUsageLevel(levelId), reason } : null);
  if (!resolved) return null;

  const { level } = resolved;
  const detail = reason ?? resolved.reason ?? '';
  const description = describeUsage(level, detail);
  const classes = [
    'usage-level-badge',
    `usage-level-badge--${level.id}`,
    compact ? 'usage-level-badge--compact' : '',
    className,
  ].filter(Boolean).join(' ');

  return (
    <span id={id} className={classes} data-usage-level={level.id} title={description}>
      <span className="usage-level-bars" aria-hidden="true">
        {Array.from({ length: USAGE_LEVEL_MAX_RANK }, (_, index) => (
          <span
            key={index}
            className={`usage-level-bar${index < level.rank ? ' is-filled' : ''}`}
          />
        ))}
      </span>
      {!compact && <span className="usage-level-label" aria-hidden="true">{level.label}</span>}
      <span className="visually-hidden">{description}</span>
    </span>
  );
}
