import React from 'react';
import './settings-common.css';

const LEADERBOARD_TOOLTIP = 'This company\'s state-of-the-art model has been seen in MOTO testing to solve complex mathematical proofs and perform well in Submitter 1 (Main Submitter), High-Context Submitter, and High-Parameter Submitter, the three primary proof-creation roles.';

const ROLE_TOOLTIP = 'These are the three roles that submit proofs: Submitter 1 (Main Submitter), High-Context Submitter, and High-Parameter Submitter. For the best chance of creating novel proofs, use models comparable to those marked PS in the Highlighted Models list.';

export default function ProofStrengthBadge({ variant = 'role', className = '' }) {
  const tooltip = variant === 'leaderboard' ? LEADERBOARD_TOOLTIP : ROLE_TOOLTIP;
  const variantClass = variant === 'leaderboard' ? 'ps-badge-anchor--leaderboard' : 'ps-badge-anchor--role';

  return (
    <span className={`ps-badge-anchor ${variantClass} ${className}`.trim()} tabIndex={0}>
      <span className="ps-badge">PS</span>
      <span className="ps-badge-tooltip">{tooltip}</span>
    </span>
  );
}
