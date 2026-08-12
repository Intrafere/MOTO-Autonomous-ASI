import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, test, vi } from 'vitest';
import {
  buildCurrentProofRuntimeConfig,
  buildManualAggregatorProofRuntimeConfig,
  buildManualCompilerProofRuntimeConfig,
  isProofRunBusy,
  mergeProofRun,
  proofRunStatusLabel,
  selectSourceProofRun,
  useProofCheckRuntime,
} from './useProofCheckRuntime';
import { API_ERROR_KINDS, autonomousAPI, MotoApiError } from '../services/api';

afterEach(() => {
  localStorage.clear();
  vi.restoreAllMocks();
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

  test('strips persisted Supercharge outside developer mode', () => {
    localStorage.setItem('aggregator_settings', JSON.stringify({
      validatorProvider: 'openrouter',
      validatorModel: 'validator',
      validatorContextSize: 8192,
      validatorMaxOutput: 1024,
      validatorSuperchargeEnabled: true,
      submitterConfigs: [{
        provider: 'openrouter',
        modelId: 'submitter',
        contextWindow: 8192,
        maxOutputTokens: 1024,
        superchargeEnabled: true,
      }],
    }));

    const snapshot = buildManualAggregatorProofRuntimeConfig();
    expect(snapshot.brainstorm.supercharge_enabled).toBe(false);
    expect(snapshot.validator.supercharge_enabled).toBe(false);
  });
});

describe('proof run lifecycle presentation', () => {
  test('ignores older lifecycle generations', () => {
    const current = { proof_run_id: 'run-1', lifecycle_generation: 4, status: 'running' };
    const stale = { proof_run_id: 'run-1', lifecycle_generation: 3, status: 'stopped' };
    expect(mergeProofRun(current, stale)).toBe(current);
  });

  test('describes running and pruning states', () => {
    expect(proofRunStatusLabel({ status: 'running', pruning_status: 'idle' }))
      .toBe('Running');
    expect(proofRunStatusLabel({ status: 'running', pruning_status: 'validating' }))
      .toBe('Running · pruning validating');
  });

  test('continuous no-candidate rounds remain running', () => {
    expect(proofRunStatusLabel({
      status: 'running',
      run_mode: 'loop_with_pruning',
    })).toBe('Running');
  });

  test('ended and repair-required runs are not busy', () => {
    expect(isProofRunBusy({ status: 'stopped', resumable: true })).toBe(false);
    expect(isProofRunBusy({ status: 'repair_required' })).toBe(false);
  });

  test('source selection honors the backend preferred run and otherwise uses the latest run', () => {
    const sourceKey = 'autonomous:paper:paper-1';
    const runs = {
      old: {
        proof_run_id: 'old',
        source_key: sourceKey,
        lifecycle_generation: 9,
        status: 'completed',
        updated_at: '2026-08-03T10:00:00Z',
      },
      resumable: {
        proof_run_id: 'resumable',
        source_key: sourceKey,
        lifecycle_generation: 3,
        status: 'stopped',
        resumable: true,
        updated_at: '2026-08-03T09:00:00Z',
      },
    };
    expect(selectSourceProofRun(runs, sourceKey)?.proof_run_id).toBe('old');
    expect(selectSourceProofRun(runs, sourceKey, 'old')?.proof_run_id).toBe('old');
  });
});

describe('proof check runtime recovery', () => {
  test('recovers a committed source run after an ambiguous queue transport failure', async () => {
    const recovered = {
      proof_run_id: 'recovered-run',
      run_mode: 'loop_with_pruning',
      scope: 'autonomous',
      source_type: 'paper',
      source_id: 'paper-1',
      source_key: 'autonomous:paper:paper-1',
      lifecycle_generation: 2,
      status: 'running',
      updated_at: '2026-08-03T10:00:00Z',
    };
    vi.spyOn(autonomousAPI, 'getProofStatus').mockResolvedValue({
      lean4_enabled: true,
      manual_check_ready: true,
      workspace_ready: true,
      lean4_version: '4.20',
    });
    vi.spyOn(autonomousAPI, 'runProofCheck').mockRejectedValue(new MotoApiError(
      'Queue response was lost',
      { kind: API_ERROR_KINDS.AMBIGUOUS_TRANSPORT },
    ));
    const listRuns = vi.spyOn(autonomousAPI, 'listProofRuns').mockImplementation(async (query = {}) => (
      query.sourceId
        ? {
            runs: [recovered],
            ambiguous: false,
            preferred_proof_run_id: 'recovered-run',
          }
        : { runs: [] }
    ));

    const { result } = renderHook(() => useProofCheckRuntime());
    await waitFor(() => expect(listRuns).toHaveBeenCalled());

    let queued;
    await act(async () => {
      queued = await result.current.queueManualProofCheck({
        sourceType: 'paper',
        sourceId: 'paper-1',
        scope: 'autonomous',
        runMode: 'loop_with_pruning',
      });
    });

    expect(queued.proof_run_id).toBe('recovered-run');
    expect(listRuns).toHaveBeenCalledWith({
      scope: 'autonomous',
      sourceType: 'paper',
      sourceId: 'paper-1',
    });
    await waitFor(() => {
      expect(result.current.getSourceState('paper', 'paper-1', 'autonomous')?.proof_run_id)
        .toBe('recovered-run');
    });
  });

  test('refreshes authoritative run state after a stale control conflict', async () => {
    const run = {
      proof_run_id: 'run-1',
      run_mode: 'loop_with_pruning',
      scope: 'manual',
      source_type: 'paper',
      source_id: 'manual_compiler_current',
      lifecycle_generation: 4,
      status: 'running',
      updated_at: '2026-08-03T10:00:00Z',
    };
    vi.spyOn(autonomousAPI, 'getProofStatus').mockResolvedValue({
      lean4_enabled: true,
      manual_check_ready: true,
      workspace_ready: true,
      lean4_version: '4.20',
    });
    vi.spyOn(autonomousAPI, 'listProofRuns').mockResolvedValue({ runs: [run] });
    vi.spyOn(autonomousAPI, 'stopProofRun').mockRejectedValue(new MotoApiError(
      'Lifecycle generation is stale',
      { kind: API_ERROR_KINDS.CONFLICT, status: 409 },
    ));
    const getRun = vi.spyOn(autonomousAPI, 'getProofRun').mockResolvedValue({
      ...run,
      lifecycle_generation: 5,
      status: 'stopped',
      resumable: true,
    });

    const { result } = renderHook(() => useProofCheckRuntime());
    await waitFor(() => expect(result.current.proofRuns).toHaveLength(1));
    let controlError;
    await act(async () => {
      try {
        await result.current.stopProofRun(run, 4);
      } catch (error) {
        controlError = error;
      }
    });
    expect(controlError?.message).toBe('Lifecycle generation is stale');
    expect(getRun).toHaveBeenCalledWith('run-1');
    await waitFor(() => {
      expect(result.current.proofRuns[0]).toEqual(expect.objectContaining({
        lifecycle_generation: 5,
        status: 'stopped',
      }));
    });
  });

  test('returns terminal source state so status controls can show completion', async () => {
    const run = {
      proof_run_id: 'completed-run',
      run_mode: 'one_round',
      scope: 'autonomous',
      source_type: 'paper',
      source_id: 'paper-1',
      source_key: 'autonomous:paper:paper-1',
      lifecycle_generation: 2,
      status: 'completed',
      terminal_reason: 'round_complete',
      updated_at: '2026-08-03T10:00:00Z',
    };
    vi.spyOn(autonomousAPI, 'getProofStatus').mockResolvedValue({
      lean4_enabled: true,
      manual_check_ready: true,
      workspace_ready: true,
      lean4_version: '4.20',
    });
    vi.spyOn(autonomousAPI, 'listProofRuns').mockResolvedValue({ runs: [run] });

    const { result } = renderHook(() => useProofCheckRuntime());

    await waitFor(() => {
      expect(result.current.getSourceState('paper', 'paper-1', 'autonomous')).toEqual(
        expect.objectContaining({
          proof_run_id: 'completed-run',
          status: 'completed',
          terminal_reason: 'round_complete',
          statusLabel: 'completed',
        }),
      );
    });
  });
});

