const asNumber = (value) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
};

export const isSameSolutionPathRun = (left, right) => (
  Boolean(left?.run_id)
  && Boolean(right?.run_id)
  && left.run_id === right.run_id
);

export const isSolutionPathSnapshotAtLeast = (candidate, current) => {
  if (!candidate) return false;
  if (!current) return true;
  if (!isSameSolutionPathRun(candidate, current)) return true;

  const candidateGeneration = asNumber(candidate.lifecycle_generation);
  const currentGeneration = asNumber(current.lifecycle_generation);
  if (candidateGeneration !== currentGeneration) {
    return candidateGeneration > currentGeneration;
  }
  return asNumber(candidate.revision) >= asNumber(current.revision);
};

export const solutionPathEventMatches = (event = {}, snapshot = {}, expectedMode = '') => {
  if (event.run_id && snapshot.run_id && event.run_id !== snapshot.run_id) return false;
  if (
    event.lifecycle_generation != null
    && snapshot.lifecycle_generation != null
    && asNumber(event.lifecycle_generation) !== asNumber(snapshot.lifecycle_generation)
  ) return false;

  const eventMode = String(event.workflow_mode || event.mode || '').toLowerCase();
  if (!eventMode || !expectedMode) return true;
  return eventMode === expectedMode;
};

export const getSolutionPathSettingsMode = (snapshot = {}) => (
  snapshot.mode === 'compiler' ? 'aggregator' : snapshot.mode
);

export const getSolutionPathEmptyLabel = (snapshot = {}) => {
  if (snapshot.ownership === 'resumable') {
    return 'Resumable run · no approved route yet';
  }
  if (snapshot.queued_proposals > 0 || snapshot.reviewing_proposals > 0) {
    return 'No approved route yet · update pending';
  }
  return 'No approved route yet';
};
