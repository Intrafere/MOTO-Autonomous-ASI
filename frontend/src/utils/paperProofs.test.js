import { describe, expect, test } from 'vitest';
import { parsePaperProofs, resolvePaperMetrics } from './paperProofs';

const START = '[HARD CODED THEOREMS APPENDIX START -- LEAN 4 VERIFIED THEOREMS BELOW]';
const END = '[HARD CODED THEOREMS APPENDIX END -- ALL APPENDIX CONTENT SHOULD BE ABOVE THIS LINE]';

describe('parsePaperProofs', () => {
  test('orders repeated compiler appendices and uses proof IDs as title fallback', () => {
    const source = [
      'Chapter One', START,
      'Theorem (proof_alpha) [Novel] - Alpha theorem',
      'Status: verified by Lean 4', 'Lean 4 proof:', 'by simp', '---', END,
      'Chapter Two', START,
      'Theorem (proof_beta) [Known] - proof_beta',
      'Status: verified by Lean 4', 'Lean 4 proof:', 'by omega', '---', END,
    ].join('\n');
    const parsed = parsePaperProofs(source);
    expect(parsed.proofs).toHaveLength(2);
    expect(parsed.proofs.map((proof) => proof.number)).toEqual([1, 2]);
    expect(parsed.proofs.map((proof) => proof.theoremName)).toEqual(['Alpha theorem', 'proof_beta']);
    expect(parsed.paperText).toContain('Chapter One');
    expect(parsed.paperText).toContain('Chapter Two');
  });

  test('supports legacy attached proof sections', () => {
    const source = [
      'Paper body.',
      '=== PROOFS ATTACHED TO THIS PAPER (Lean 4 Verified) ===',
      'Proof 4: Sum formula', 'Status: Verified (Known)', 'Proof ID: proof_sum',
      'Lean 4 Code:', 'by omega', '---',
    ].join('\n');
    expect(parsePaperProofs(source).proofs[0]).toMatchObject({
      proofId: 'proof_sum', theoremName: 'proof_sum', number: 1,
    });
  });

  test('prefers backend metric fields and falls back per missing field', () => {
    const parsed = parsePaperProofs('Paper text');
    const metrics = resolvePaperMetrics({
      paper_word_count: 40,
      proof_character_count: 12,
      total_word_count: 50,
      proof_count: 3,
    }, parsed);
    expect(metrics.paper.words).toBe(40);
    expect(metrics.paper.characters).toBe(parsed.fallbackMetrics.paper.characters);
    expect(metrics.proofs.characters).toBe(12);
    expect(metrics.total.words).toBe(50);
    expect(metrics.proofCount).toBe(3);
  });
});
