import { enabledCompetitionSecondaries, normalizeProofCompetition, persistAutonomousSettings, getStoredAutonomousSettings,
  settingsToAutonomousConfig, publicAutonomousProfilesForStorage, applyAutonomousProfileSelection,
  RECOMMENDED_PROFILES, RECOMMENDED_PROFILE_KEYS } from './autonomousProfiles';

beforeEach(() => localStorage.clear());

test('disabled incomplete competition drafts are omitted without erasing stored routes', () => {
  const draft = { enabled: false, secondaries: [{ model_id: '', context_window: 0 }] };
  expect(enabledCompetitionSecondaries(draft)).toEqual([]);
  expect(draft.secondaries).toEqual([{ model_id: '', context_window: 0 }]);
  expect(enabledCompetitionSecondaries({ ...draft, enabled: true })).toBe(draft.secondaries);
});

const competition = { enabled: true, secondaries: [{ provider: 'openrouter', model_id: 'test/model',
  context_window: 64000, max_tokens: 8000, openrouter_provider: 'host',
  openrouter_reasoning_effort: 'high', lm_studio_fallback_id: 'local', supercharge_enabled: true }] };

test('legacy settings default off without persisting a primary', () => {
  expect(settingsToAutonomousConfig({}).proof_competition).toEqual({ enabled: false, secondaries: [] });
  expect(normalizeProofCompetition(null)).toEqual({ enabled: false, secondaries: [] });
});

test('settings and public profiles retain ordered runtime roles and exclude secrets', () => {
  const value = { ...competition, primary: { model_id: 'do-not-store' }, secondaries: [
    { ...competition.secondaries[0], api_key: 'secret' }, { ...competition.secondaries[0], model_id: 'second' },
  ] };
  persistAutonomousSettings({ localConfig: { proof_competition: value } });
  const restored = getStoredAutonomousSettings();
  expect(settingsToAutonomousConfig(restored).proof_competition).toEqual(normalizeProofCompetition(value));
  expect(JSON.stringify(restored.localConfig.proof_competition)).not.toMatch(/secret|do-not-store/);
  expect(publicAutonomousProfilesForStorage({ custom: { proof_competition: value } }).custom.proof_competition)
    .toEqual(normalizeProofCompetition(value));
});

test('profile application restores competition and legacy profiles reset it', async () => {
  const base = RECOMMENDED_PROFILES[RECOMMENDED_PROFILE_KEYS[0]];
  const result = await applyAutonomousProfileSelection('custom', { custom: { ...base, proof_competition: competition } });
  expect(result.config.proof_competition).toEqual(competition);
  const legacy = await applyAutonomousProfileSelection('legacy', { legacy: base });
  expect(legacy.config.proof_competition).toEqual({ enabled: false, secondaries: [] });
});
