import React from 'react';
import './settings-common.css';

const LEADERBOARD_TOOLTIP = 'This company\'s state-of-the-art model has been seen in MOTO testing to solve complex mathematical proofs and perform well in brainstorm submitters, Writing Submitter, and Rigor & Proofs Submitter.';

const ROLE_TOOLTIP = 'Rigor & Proofs Submitter is the explicit proof-solving and formalization role. Brainstorm submitters may still produce proof candidates where applicable, and Writing Submitter builds the surrounding paper context.';

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
