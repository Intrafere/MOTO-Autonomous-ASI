import {
  getSolutionPathEmptyLabel,
  getSolutionPathSettingsMode,
  isSolutionPathSnapshotAtLeast,
  solutionPathEventMatches,
} from './solutionPathPresentation';

test('fences snapshots by run generation and revision', () => {
  const current = { run_id: 'run-1', lifecycle_generation: 3, revision: 4 };
  expect(isSolutionPathSnapshotAtLeast({ ...current, revision: 5 }, current)).toBe(true);
  expect(isSolutionPathSnapshotAtLeast({ ...current, revision: 3 }, current)).toBe(false);
  expect(isSolutionPathSnapshotAtLeast({ ...current, lifecycle_generation: 2, revision: 99 }, current)).toBe(false);
  expect(isSolutionPathSnapshotAtLeast({ ...current, lifecycle_generation: 4, revision: 0 }, current)).toBe(true);
});

test('filters workflow events by run generation and mode', () => {
  const snapshot = { run_id: 'run-1', lifecycle_generation: 3 };
  expect(solutionPathEventMatches({
    run_id: 'run-1',
    lifecycle_generation: 3,
    workflow_mode: 'compiler',
  }, snapshot, 'compiler')).toBe(true);
  expect(solutionPathEventMatches({ run_id: 'run-2' }, snapshot, 'compiler')).toBe(false);
  expect(solutionPathEventMatches({ run_id: 'run-1', lifecycle_generation: 2 }, snapshot, 'compiler')).toBe(false);
  expect(solutionPathEventMatches({ workflow_mode: 'autonomous' }, snapshot, 'compiler')).toBe(false);
});

test('routes compiler repairs to aggregator settings and labels resumable no-plan state', () => {
  expect(getSolutionPathSettingsMode({ mode: 'compiler' })).toBe('aggregator');
  expect(getSolutionPathEmptyLabel({ ownership: 'resumable' })).toMatch(/Resumable run/);
});
