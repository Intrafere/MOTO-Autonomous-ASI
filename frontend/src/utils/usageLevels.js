// Shared usage-level scale and feature registry for MOTO's relative
// model-usage indicators. Any toggle, checkbox, or functionality check can
// show a UsageLevelBadge by adding one entry to USAGE_FEATURES; the badge and
// its accessible description come from this single source.
//
// Levels describe relative model-call/token consumption compared with other
// MOTO features, not an exact cost: provider pricing, model choice, and run
// length still decide the real spend.

export const USAGE_LEVEL_SCALE = Object.freeze([
  Object.freeze({ id: 'low', rank: 1, label: 'Low' }),
  Object.freeze({ id: 'medium', rank: 2, label: 'Medium' }),
  Object.freeze({ id: 'high', rank: 3, label: 'High' }),
  Object.freeze({ id: 'very_high', rank: 4, label: 'Very high' }),
]);

export const USAGE_LEVEL_MAX_RANK = USAGE_LEVEL_SCALE.length;

const LEVELS_BY_ID = Object.freeze(
  Object.fromEntries(USAGE_LEVEL_SCALE.map((level) => [level.id, level])),
);

// Shared rule for "how many run at once" controls: 1 is low, 2 is medium,
// 3-4 are high, 5 or more are very high. Anything else resolves to null (no badge).
export const PARALLEL_COUNT_THRESHOLDS = Object.freeze({ medium: 2, high: 3, veryHigh: 5 });

export function usageLevelForParallelCount(count) {
  if (count === null || count === undefined || count === '') return null;
  const value = Number(count);
  if (!Number.isInteger(value) || value < 1) return null;
  if (value >= PARALLEL_COUNT_THRESHOLDS.veryHigh) return 'very_high';
  if (value >= PARALLEL_COUNT_THRESHOLDS.high) return 'high';
  if (value >= PARALLEL_COUNT_THRESHOLDS.medium) return 'medium';
  return 'low';
}

// Concurrent Proof Attempts uses its own scale around the default of 6:
// 1 is low, 2-3 are medium, 4-6 are high, 7 or more are very high, and
// 0 (unlimited) is very high.
export const CONCURRENT_PROOF_ATTEMPT_THRESHOLDS = Object.freeze({ medium: 2, high: 4, veryHigh: 7 });

export function usageLevelForConcurrentProofAttempts(count) {
  if (count === null || count === undefined || count === '') return null;
  const value = Number(count);
  if (!Number.isInteger(value) || value < 0) return null;
  if (value === 0 || value >= CONCURRENT_PROOF_ATTEMPT_THRESHOLDS.veryHigh) return 'very_high';
  if (value >= CONCURRENT_PROOF_ATTEMPT_THRESHOLDS.high) return 'high';
  if (value >= CONCURRENT_PROOF_ATTEMPT_THRESHOLDS.medium) return 'medium';
  return 'low';
}

const plural = (count, word) => `${count} ${word}${Number(count) === 1 ? '' : 's'}`;

// Each entry: { label, level, reason }. `level` and `reason` are plain values,
// or functions of the badge's `context` for features whose usage depends on
// settings (for example a count of parallel submitters). Keep ids stable once
// shipped.
export const USAGE_FEATURES = Object.freeze({
  'allowed_outputs.mathematical_proofs': Object.freeze({
    label: 'Mathematical Proofs',
    level: 'very_high',
    reason:
      'Lean 4 proof search runs candidate discovery, Validator review, up to five '
      + 'formalization attempts per candidate, and novelty review, often across '
      + 'several rounds, so it uses the most model calls and tokens.',
  }),
  'allowed_outputs.single_shot_proofs': Object.freeze({
    label: 'Single-shot proofs',
    level: 'high',
    reason:
      'Each theorem gets one proof budget: up to five Lean formalization attempts, '
      + 'then novelty review.',
  }),
  'allowed_outputs.master_proofs': Object.freeze({
    label: 'Master proofs',
    level: 'very_high',
    reason:
      'A large theorem is split into a scaffold of steps solved in parallel. Every step '
      + 'gets up to 15 Lean attempts and can split again, so one master proof can make '
      + 'many model calls.',
  }),
  'allowed_outputs.research_papers': Object.freeze({
    label: 'Research Papers',
    level: 'medium',
    reason:
      'Paper writing runs one sequential outline, construction, review, critique, '
      + 'and validation loop per paper.',
  }),
  'brainstorm.submitter_count': Object.freeze({
    label: 'Number of Submitters',
    level: ({ count } = {}) => usageLevelForParallelCount(count),
    reason: ({ count } = {}) =>
      `${plural(count, 'brainstorm submitter')} generate in parallel, and the Validator `
      + 'reviews every submission. Each added submitter adds its own model calls.',
  }),
  'proofs.concurrent_attempts': Object.freeze({
    label: 'Concurrent Proof Attempts',
    level: ({ count } = {}) => usageLevelForConcurrentProofAttempts(count),
    reason: ({ count } = {}) => (
      Number(count) === 0
        ? 'Unlimited: every approved proof candidate is attempted at once, each with up to '
          + 'five Lean formalization attempts.'
        : `Up to ${plural(count, 'proof candidate')} are attempted at once, each with up to `
          + 'five Lean formalization attempts.'
    ),
  }),
  'roles.supercharge': Object.freeze({
    label: 'Supercharge',
    level: 'very_high',
    reason: 'Every call this role makes becomes five calls: four full answer attempts plus a '
      + 'synthesis call, so this role uses about five times the API calls.',
  }),
  'proofs.competition': Object.freeze({
    label: 'Proof model competition',
    // `models` counts the primary plus every secondary proof model.
    level: ({ models } = {}) => usageLevelForParallelCount(models),
    reason: ({ models } = {}) =>
      `${plural(models, 'proof model')} can each make up to five proof attempts per candidate. `
      + 'A secondary runs only after the models before it fail, while the primary moves on to new work.',
  }),
  'proofs.check_mode.one_round': Object.freeze({
    label: 'Run One Round',
    level: 'high',
    reason: 'One round finds candidates, has the Validator review them, then makes up to five Lean '
      + 'formalization attempts per approved candidate.',
  }),
  'proofs.check_mode.loop_with_pruning': Object.freeze({
    label: 'Run Rounds on Loop With Pruning',
    level: 'very_high',
    reason: 'Each round costs as much as a single round, rounds keep repeating while candidates are '
      + 'found, and pruning reviews add more calls.',
  }),
  'stage3.final_answer': Object.freeze({
    label: 'Stage 3 Final Answer Generation',
    level: 'very_high',
    reason: 'Synthesizes your completed papers into a final answer, often book-length, with its own '
      + 'assessment, organization, and full paper-writing passes.',
  }),
  'translation.auto_translate': Object.freeze({
    label: 'Auto Top-p Translate',
    level: ({ translateAll = false } = {}) => (translateAll ? 'very_high' : 'high'),
    reason: ({ batch = false, translateAll = false } = {}) => {
      const base = 'Every translated part runs a translator call and a validator check, per language.';
      if (translateAll) return `${base} Translate all older items is on, so your whole history is being translated.`;
      if (batch) return `${base} Batch translation runs several documents at once.`;
      return `${base} Documents translate one at a time.`;
    },
  }),
  'skills.wolfram_alpha': Object.freeze({
    label: 'Wolfram Alpha',
    level: 'low',
    reason: 'Paper construction may check facts with Wolfram Alpha queries; each query adds only a '
      + 'small tool reply to a call the writer already makes.',
  }),
  'skills.session_history_memory': Object.freeze({
    label: 'Session History Memory',
    level: 'medium',
    reason: 'Turns on the Assistant role, which runs proof-history selection calls beside the '
      + 'workflow and refreshes its support pack as work continues.',
  }),
});

export function getUsageLevel(levelId) {
  return LEVELS_BY_ID[levelId] ?? null;
}

// Resolves a registered feature (or an explicit level) to a usage level.
// Unknown features or levels resolve to null so callers render nothing rather
// than a misleading indicator.
export function resolveFeatureUsage(featureId, context = {}) {
  const feature = USAGE_FEATURES[featureId];
  if (!feature) return null;
  let levelId;
  let reason;
  try {
    levelId = typeof feature.level === 'function' ? feature.level(context) : feature.level;
    reason = typeof feature.reason === 'function' ? feature.reason(context) : feature.reason;
  } catch {
    return null;
  }
  const level = getUsageLevel(levelId);
  if (!level) return null;
  return { featureId, label: feature.label, reason, level };
}

export function describeUsage(level, reason = '') {
  if (!level) return '';
  const base = `${level.label} usage`;
  return reason ? `${base}. ${reason}` : base;
}
