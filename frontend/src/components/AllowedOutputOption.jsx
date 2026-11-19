import React, { useId } from 'react';
import UsageLevelBadge from './UsageLevelBadge';

// One Allowed Outputs checkbox with its relative usage-level badge. The
// checkbox keeps its plain output name as its accessible name and announces
// the usage level as its description.
export default function AllowedOutputOption({
  usageFeatureId,
  title,
  checked,
  onChange,
  disabled,
  children,
}) {
  const baseId = useId();
  const textId = `${baseId}-text`;
  const usageId = `${baseId}-usage`;
  return (
    <label className="allowed-output-option" title={title}>
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        disabled={disabled}
        aria-labelledby={textId}
        aria-describedby={usageFeatureId ? usageId : undefined}
      />
      <span id={textId} className="allowed-output-text">{children}</span>
      {usageFeatureId && <UsageLevelBadge id={usageId} featureId={usageFeatureId} />}
    </label>
  );
}
