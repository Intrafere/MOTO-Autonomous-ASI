import { describe, expect, test } from 'vitest';
import {
  classifyProofNovelty,
  formatProofProvenance,
  sanitizeDomId,
} from './proofPresentation';

describe('proof presentation', () => {
  test.each([
    ['major_mathematical_discovery', 'novel'],
    ['mathematical_discovery', 'novel'],
    ['novel_variant', 'novel'],
    ['novel_formulation', 'novel'],
    ['duplicate_novel', 'duplicate_novel'],
    ['not_novel', 'not_novel'],
  ])('classifies strict backend category %s', (noveltyTier, group) => {
    expect(classifyProofNovelty({ novelty_tier: noveltyTier }).group).toBe(group);
  });

  test('does not infer novelty from legacy boolean fields', () => {
    expect(classifyProofNovelty({ novel: true })).toMatchObject({
      category: 'unknown',
      label: 'Unknown (Legacy)',
      group: 'unknown',
    });
  });

  test('sanitizes stable DOM ids and formats bounded lineage provenance', () => {
    expect(sanitizeDomId('manual:run/1 proof#2', 'details')).toBe('details-manual-run-1-proof-2');
    expect(formatProofProvenance({
      run_id: 'run-1',
      session_id: 'session-1',
      corpus_scope: 'manual',
      source_type: 'paper',
      source_id: 'paper-1',
      retrieval_lanes: ['exact', 'semantic'],
      occurrence_omitted: 4,
    })).toEqual({
      runId: 'run-1',
      sessionId: 'session-1',
      source: 'manual · paper/paper-1',
      lanes: ['exact', 'semantic'],
      omitted: 4,
    });
  });
});
