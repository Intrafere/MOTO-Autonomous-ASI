import { describe, expect, test } from 'vitest';
import {
  buildProofLiveContextMutation,
  classifyProofLiveContext,
  classifyProofNovelty,
  formatProofProvenance,
  getProofLiveContextRiskWarnings,
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

  test('classifies live-context state independently from novelty', () => {
    const proof = {
      novelty_tier: 'mathematical_discovery',
      live_context_status: 'pruned',
      live_context_owner_run_id: 'run-7',
      live_context_pruned_by: 'automatic_proof_pruning',
      live_context_prune_reason: 'Superseded in live solving context',
    };
    expect(classifyProofNovelty(proof).group).toBe('novel');
    expect(classifyProofLiveContext(proof)).toMatchObject({
      isPruned: true,
      ownerRunId: 'run-7',
      actorLabel: 'Automatic proof-pruning review',
      canUndo: false,
    });
    expect(classifyProofLiveContext(
      { ...proof, live_context_pruned_by: 'user' },
      { historical: true }
    )).toMatchObject({ readOnly: true, canUndo: false });
  });

  test('builds generation-fenced mutations and deterministic risk warnings', () => {
    const proof = {
      proof_id: 'proof-1',
      run_id: 'run-1',
      proof_set_revision: 9,
      source_type: 'leanoj_final',
      canonical_theorem_statement_hash: 'statement-hash',
      canonical_lean_code_hash: 'lean-hash',
      dependency_extraction_status: 'failed',
    };
    expect(buildProofLiveContextMutation(proof, {
      status: 'pruned',
      reason: 'Remove from current route',
    })).toEqual({
      status: 'pruned',
      actor: 'user',
      expected_run_id: 'run-1',
      expected_proof_set_revision: 9,
      reason: 'Remove from current route',
      expected_theorem_hash: 'statement-hash',
      expected_lean_hash: 'lean-hash',
    });
    expect(getProofLiveContextRiskWarnings(proof, {
      dependedOnBy: [{ proof_id: 'dependent-1' }],
    })).toEqual([
      'This occurrence is marked as a verified final-solution proof.',
      '1 stored proof depends on this occurrence.',
      'Dependency extraction is failed; downstream risk may be incomplete.',
    ]);
  });

  test('fails closed when mutation revision metadata is absent', () => {
    expect(() => buildProofLiveContextMutation(
      { run_id: 'run-1' },
      { status: 'pruned', reason: 'No longer useful' }
    )).toThrow(/proof-set revision is unavailable/i);
  });
});
