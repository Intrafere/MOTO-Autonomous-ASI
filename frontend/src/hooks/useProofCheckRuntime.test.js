import { afterEach, describe, expect, test } from 'vitest';
import {
  buildCurrentProofRuntimeConfig,
  buildManualAggregatorProofRuntimeConfig,
  buildManualCompilerProofRuntimeConfig,
} from './useProofCheckRuntime';

afterEach(() => {
  localStorage.clear();
});

describe('proof check runtime Assistant snapshots', () => {
  test('manual Aggregator Try to Prove snapshot includes Aggregator Assistant settings', () => {
    localStorage.setItem('aggregator_settings', JSON.stringify({
      validatorProvider: 'openrouter',
      validatorModel: 'validator-model',
      validatorContextSize: 8192,
      validatorMaxOutput: 1024,
      assistantProvider: 'openrouter',
      assistantModel: 'assistant-model',
      assistantOpenrouterProvider: 'AssistantHost',
      assistantContextSize: 7777,
      assistantMaxOutput: 777,
      submitterConfigs: [{
        provider: 'openrouter',
        modelId: 'submitter-model',
        contextWindow: 8192,
        maxOutputTokens: 1024,
      }],
    }));

    const snapshot = buildManualAggregatorProofRuntimeConfig();

    expect(snapshot.assistant.model_id).toBe('assistant-model');
    expect(snapshot.assistant.openrouter_provider).toBe('AssistantHost');
    expect(snapshot.assistant.context_window).toBe(7777);
    expect(snapshot.assistant.max_output_tokens).toBe(777);
  });

  test('manual Aggregator Try to Prove snapshot does not borrow Compiler Rigor settings', () => {
    localStorage.setItem('aggregator_settings', JSON.stringify({
      validatorProvider: 'openrouter',
      validatorModel: 'aggregator-validator',
      validatorContextSize: 400000,
      validatorMaxOutput: 85000,
      submitterConfigs: [{
        provider: 'openrouter',
        modelId: 'aggregator-submitter',
        contextWindow: 400000,
        maxOutputTokens: 85000,
      }],
    }));
    localStorage.setItem('compiler_settings', JSON.stringify({
      validatorProvider: 'openrouter',
      validatorModel: 'compiler-validator',
      validatorContextSize: 131072,
      validatorMaxOutput: 25000,
      highParamProvider: 'openrouter',
      highParamModel: 'compiler-rigor',
      highParamContextSize: 131072,
      highParamMaxOutput: 25000,
    }));

    const snapshot = buildManualAggregatorProofRuntimeConfig();

    expect(snapshot.brainstorm.model_id).toBe('aggregator-submitter');
    expect(snapshot.brainstorm.context_window).toBe(400000);
    expect(snapshot.brainstorm.max_output_tokens).toBe(85000);
    expect(snapshot.validator.model_id).toBe('aggregator-validator');
  });

  test('manual Compiler Try to Prove snapshot includes Compiler Assistant settings', () => {
    localStorage.setItem('compiler_settings', JSON.stringify({
      validatorProvider: 'openrouter',
      validatorModel: 'compiler-validator',
      validatorContextSize: 9000,
      validatorMaxOutput: 900,
      highParamProvider: 'openrouter',
      highParamModel: 'rigor-model',
      highParamContextSize: 10000,
      highParamMaxOutput: 1000,
      assistantProvider: 'openrouter',
      assistantModel: 'compiler-assistant',
      assistantOpenrouterProvider: 'CompilerAssistantHost',
      assistantContextSize: 8888,
      assistantMaxOutput: 888,
    }));

    const snapshot = buildManualCompilerProofRuntimeConfig();

    expect(snapshot.assistant.model_id).toBe('compiler-assistant');
    expect(snapshot.assistant.openrouter_provider).toBe('CompilerAssistantHost');
    expect(snapshot.assistant.context_window).toBe(8888);
    expect(snapshot.assistant.max_output_tokens).toBe(888);
  });

  test('autonomous and history Try to Prove snapshot includes Autonomous Assistant settings', () => {
    localStorage.setItem('autonomous_research_settings', JSON.stringify({
      localConfig: {
        validator_provider: 'openrouter',
        validator_model: 'auto-validator',
        validator_context_window: 9000,
        validator_max_tokens: 900,
        high_param_provider: 'openrouter',
        high_param_model: 'auto-rigor',
        high_param_context_window: 10000,
        high_param_max_tokens: 1000,
        assistant_provider: 'openrouter',
        assistant_model: 'auto-assistant',
        assistant_openrouter_provider: 'AutoAssistantHost',
        assistant_context_window: 9999,
        assistant_max_tokens: 999,
      },
    }));

    const snapshot = buildCurrentProofRuntimeConfig();

    expect(snapshot.assistant.model_id).toBe('auto-assistant');
    expect(snapshot.assistant.openrouter_provider).toBe('AutoAssistantHost');
    expect(snapshot.assistant.context_window).toBe(9999);
    expect(snapshot.assistant.max_output_tokens).toBe(999);
  });
});

