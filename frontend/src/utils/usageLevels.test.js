import { describe, expect, test } from 'vitest';
import {
  USAGE_FEATURES,
  USAGE_LEVEL_MAX_RANK,
  USAGE_LEVEL_SCALE,
  describeUsage,
  getUsageLevel,
  resolveFeatureUsage,
  usageLevelForParallelCount,
} from './usageLevels';

describe('usage level scale', () => {
  test('is ordered with unique ids and contiguous ranks', () => {
    expect(USAGE_LEVEL_SCALE.map((level) => level.id)).toEqual(['low', 'medium', 'high', 'very_high']);
    USAGE_LEVEL_SCALE.forEach((level, index) => expect(level.rank).toBe(index + 1));
    expect(USAGE_LEVEL_MAX_RANK).toBe(4);
  });

  test('unknown levels resolve to null', () => {
    expect(getUsageLevel('extreme')).toBeNull();
    expect(getUsageLevel(undefined)).toBeNull();
  });
});

describe('usage feature registry', () => {
  test('every registered feature has a label, reason, and valid static level', () => {
    Object.entries(USAGE_FEATURES).forEach(([featureId, feature]) => {
      expect(feature.label, featureId).toBeTruthy();
      expect(feature.reason, featureId).toBeTruthy();
      if (typeof feature.level !== 'function') {
        expect(getUsageLevel(feature.level), featureId).not.toBeNull();
      }
    });
  });

  test('parallel counts: 1 low, 2 medium, 3-4 high, 5+ very high', () => {
    expect([1, 2, 3, 4, 5, 10].map((count) => usageLevelForParallelCount(count)))
      .toEqual(['low', 'medium', 'high', 'high', 'very_high', 'very_high']);
    expect(usageLevelForParallelCount('2')).toBe('medium');
    expect(usageLevelForParallelCount(0)).toBeNull();
    [-1, 1.5, 'abc', '', null, undefined].forEach((value) => {
      expect(usageLevelForParallelCount(value)).toBeNull();
    });
  });

  test('submitter count drives the brainstorm submitter badge', () => {
    const level = (count) => resolveFeatureUsage('brainstorm.submitter_count', { count })?.level.id;
    expect([1, 2, 3, 5].map(level)).toEqual(['low', 'medium', 'high', 'very_high']);
    expect(resolveFeatureUsage('brainstorm.submitter_count', { count: 1 }).reason).toMatch(/^1 brainstorm submitter generate/);
    expect(resolveFeatureUsage('brainstorm.submitter_count', {})).toBeNull();
  });

  test('concurrent proof attempts treat 0 as unlimited and very high', () => {
    const resolve = (count) => resolveFeatureUsage('proofs.concurrent_attempts', { count });
    expect(resolve('1').level.id).toBe('low');
    expect(resolve('2').level.id).toBe('medium');
    expect(resolve('3').level.id).toBe('medium');
    expect(resolve('4').level.id).toBe('high');
    expect(resolve('6').level.id).toBe('high');
    expect(resolve('7').level.id).toBe('very_high');
    expect(resolve('40').level.id).toBe('very_high');
    expect(resolve('0').level.id).toBe('very_high');
    expect(resolve('0').reason).toMatch(/^Unlimited/);
    expect(resolve('')).toBeNull();
  });

  test('fixed levels for Supercharge, proof-check modes, and Stage 3', () => {
    const level = (featureId) => resolveFeatureUsage(featureId).level.id;
    expect(level('roles.supercharge')).toBe('very_high');
    expect(level('proofs.check_mode.one_round')).toBe('high');
    expect(level('proofs.check_mode.loop_with_pruning')).toBe('very_high');
    expect(level('stage3.final_answer')).toBe('very_high');
  });

  test('proof competition scales with the number of proof models', () => {
    const level = (models) => resolveFeatureUsage('proofs.competition', { models })?.level.id;
    expect([2, 3, 5].map(level)).toEqual(['medium', 'high', 'very_high']);
    expect(resolveFeatureUsage('proofs.competition', { models: 2 }).reason).toMatch(/^2 proof models/);
  });

  test('translation is high, very high during Translate all', () => {
    const level = (context) => resolveFeatureUsage('translation.auto_translate', context).level.id;
    expect(level({})).toBe('high');
    expect(level({ batch: true })).toBe('high');
    expect(level({ batch: true, translateAll: true })).toBe('very_high');
  });

  test('skills: Wolfram Alpha low, Session History Memory medium', () => {
    expect(resolveFeatureUsage('skills.wolfram_alpha').level.id).toBe('low');
    expect(resolveFeatureUsage('skills.session_history_memory').level.id).toBe('medium');
  });

  test('Allowed Outputs levels: proofs very high, papers medium', () => {
    expect(resolveFeatureUsage('allowed_outputs.mathematical_proofs').level.id).toBe('very_high');
    expect(resolveFeatureUsage('allowed_outputs.research_papers').level.id).toBe('medium');
    expect(resolveFeatureUsage('allowed_outputs.single_shot_proofs').level.id).toBe('high');
    expect(resolveFeatureUsage('allowed_outputs.master_proofs').level.id).toBe('very_high');
  });

  test('unknown features resolve to null instead of a misleading level', () => {
    expect(resolveFeatureUsage('not.registered')).toBeNull();
  });

  test('describeUsage composes the level label and reason', () => {
    expect(describeUsage(getUsageLevel('medium'))).toBe('Medium usage');
    expect(describeUsage(getUsageLevel('high'), 'Runs often.')).toBe('High usage. Runs often.');
    expect(describeUsage(null)).toBe('');
  });
});
