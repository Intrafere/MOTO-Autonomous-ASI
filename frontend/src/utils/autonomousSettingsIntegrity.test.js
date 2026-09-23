import { describe, it, expect, beforeEach } from 'vitest';
import {
  AUTONOMOUS_SETTINGS_STORAGE_KEY, getStoredAutonomousSettings, persistAutonomousSettings,
  settingsToAutonomousConfig, validateAutonomousConfig, mergeAutonomousConfig,
  applyAutonomousProfileSelection, applyLmStudioStartupDefaults,
} from './autonomousProfiles';

beforeEach(() => localStorage.clear());

describe('Autonomous settings integrity', () => {
  it('offers valid recommended roles only when no settings exist', () => {
    const config = settingsToAutonomousConfig(getStoredAutonomousSettings());
    expect(() => validateAutonomousConfig(config)).not.toThrow();
    expect(config.writer_model).toBeTruthy();
  });

  it.each(['{}', 'null', '{broken', JSON.stringify({ localConfig: { validator_model: 'my-model' } })])(
    'does not restore recommended roles from invalid saved settings: %s', raw => {
      localStorage.setItem(AUTONOMOUS_SETTINGS_STORAGE_KEY, raw);
      const settings = getStoredAutonomousSettings();
      const config = settingsToAutonomousConfig(settings);
      expect(config.writer_model).toBe('');
      expect(config.submitter_configs).toEqual([]);
      expect(() => validateAutonomousConfig(config)).toThrow('Built-in defaults were not substituted');
      persistAutonomousSettings(settings);
      expect(settingsToAutonomousConfig(getStoredAutonomousSettings()).writer_model).toBe('');
    },
  );

  it('merges competition updates without losing selected roles through persistence', () => {
    const settings = getStoredAutonomousSettings();
    settings.localConfig.validator_model = 'chosen-validator';
    settings.localConfig.writer_model = 'chosen-writer';
    const current = settingsToAutonomousConfig(settings);
    const next = mergeAutonomousConfig(current, { proof_competition: { enabled: false, secondaries: [] } });
    persistAutonomousSettings({ ...settings, localConfig: next });
    const restored = settingsToAutonomousConfig(getStoredAutonomousSettings());
    expect(restored.validator_model).toBe('chosen-validator');
    expect(restored.writer_model).toBe('chosen-writer');
    expect(next.submitter_configs).toEqual(current.submitter_configs);
  });

  it('rejects missing role limits instead of supplying default token budgets', () => {
    const config = settingsToAutonomousConfig(getStoredAutonomousSettings());
    config.writer_max_tokens = undefined;
    expect(() => validateAutonomousConfig(config)).toThrow('writer: set positive');
  });

  it('rejects invalid user profiles before replacing saved selections', async () => {
    persistAutonomousSettings(getStoredAutonomousSettings());
    const before = localStorage.getItem(AUTONOMOUS_SETTINGS_STORAGE_KEY);
    await expect(applyAutonomousProfileSelection('user_broken', {
      user_broken: { numSubmitters: 1, submitters: [{ modelId: '' }], validator: {}, writer: {}, highParam: {} },
    })).rejects.toThrow('Invalid Autonomous settings');
    expect(localStorage.getItem(AUTONOMOUS_SETTINGS_STORAGE_KEY)).toBe(before);
  });

  it('keeps Assistant OpenRouter host and fallback independent of Validator', () => {
    const settings = getStoredAutonomousSettings();
    const config = settingsToAutonomousConfig({
      ...settings,
      localConfig: {
        ...settings.localConfig,
        validator_openrouter_provider: 'Anthropic',
        validator_lm_studio_fallback: 'local-validator',
        assistant_openrouter_provider: null,
        assistant_lm_studio_fallback: null,
      },
    });
    expect(config.assistant_openrouter_provider).toBeNull();
    expect(config.assistant_lm_studio_fallback).toBeNull();
  });

  it('rejects a saved submitter count that does not match configured roles', () => {
    const settings = getStoredAutonomousSettings();
    const config = settingsToAutonomousConfig({
      ...settings,
      numSubmitters: 3,
      submitterConfigs: [settings.submitterConfigs[0]],
    });
    expect(config.num_submitters).toBe(3);
    expect(config.submitter_configs).toHaveLength(1);
    expect(() => validateAutonomousConfig(config)).toThrow('Submitter count does not match configured roles');
  });

  it('applies LM Studio startup defaults only on recommended first-launch routes', () => {
    const firstLaunch = applyLmStudioStartupDefaults('local-chat');
    expect(firstLaunch.config.writer_model).toBe('local-chat');
    expect(firstLaunch.config.validator_provider).toBe('lm_studio');

    persistAutonomousSettings({
      ...getStoredAutonomousSettings(),
      selectedProfile: '',
      localConfig: {
        ...getStoredAutonomousSettings().localConfig,
        writer_model: 'chosen-writer',
        validator_model: 'chosen-validator',
      },
    });
    const custom = applyLmStudioStartupDefaults('should-not-replace');
    expect(custom.config.writer_model).toBe('chosen-writer');
    expect(custom.config.validator_model).toBe('chosen-validator');
  });
});
