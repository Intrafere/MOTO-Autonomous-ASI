import React, { useState } from 'react';

export const paperTargetKey = ({ session_id: sessionId, paper_id: paperId }) => (
  `${sessionId}\u0000${paperId}`
);

export const isPaperBatchPruneEligible = (paper) => (
  Boolean(
    paper?.paper_id
    && paper?.session_id
    && !paper.is_pruned
    && paper.status !== 'in_progress'
    && (!paper.status || paper.status === 'complete')
  )
);

export default function PaperBatchPruneControls({
  visibleTargets,
  selectedKeys,
  onSelectAllVisible,
  onClear,
  onPruneSelected,
  pruning = false,
  message = '',
}) {
  const [confirming, setConfirming] = useState(false);
  const selectedCount = selectedKeys.size;
  const visibleEligibleCount = visibleTargets.length;
  const allVisibleSelected = visibleEligibleCount > 0
    && visibleTargets.every((target) => selectedKeys.has(paperTargetKey(target)));

  const confirmPrune = async () => {
    const succeeded = await onPruneSelected();
    if (succeeded) setConfirming(false);
  };

  return (
    <div className="paper-batch-actions" aria-label="Paper bulk actions">
      <button
        type="button"
        className="paper-batch-action"
        onClick={onSelectAllVisible}
        disabled={visibleEligibleCount === 0 || allVisibleSelected || pruning}
      >
        Select all visible ({visibleEligibleCount})
      </button>
      <button
        type="button"
        className="paper-batch-action"
        onClick={onClear}
        disabled={selectedCount === 0 || pruning}
      >
        Clear selection
      </button>
      {confirming ? (
        <div className="paper-batch-confirm" role="group" aria-label="Confirm batch prune">
          <span>
            Prune {selectedCount} selected {selectedCount === 1 ? 'paper' : 'papers'} from model context?
            This batch is all-or-none.
          </span>
          <button
            type="button"
            className="btn-delete-confirm"
            onClick={confirmPrune}
            disabled={pruning}
          >
            {pruning ? 'Pruning...' : `Prune ${selectedCount}`}
          </button>
          <button
            type="button"
            className="btn-delete-cancel"
            onClick={() => setConfirming(false)}
            disabled={pruning}
          >
            Cancel
          </button>
        </div>
      ) : (
        <button
          type="button"
          className="btn-delete-paper"
          onClick={() => setConfirming(true)}
          disabled={selectedCount === 0 || pruning}
        >
          Prune selected ({selectedCount})
        </button>
      )}
      {message && (
        <span
          className={`paper-batch-message ${message.startsWith('Failed') ? 'paper-batch-message--error' : ''}`}
          role="status"
        >
          {message}
        </span>
      )}
    </div>
  );
}
