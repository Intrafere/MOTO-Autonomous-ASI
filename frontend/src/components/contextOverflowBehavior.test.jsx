import React from 'react';
import { act, cleanup, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import {
  readPersistedLiveActivity,
  shouldRecordWorkflowStoppedActivity,
} from '../App';
import {
  formatAggregatorPersistedOverflowMessage,
  shouldIncludeAggregatorProofContextOverflow,
  shouldIncludeAggregatorSolutionPathEvent,
} from './aggregator/AggregatorLogs';
import {
  compactCompilerActivityEvents,
  shouldIncludeCompilerContextOverflow,
  shouldIncludeCompilerProofContextOverflow,
  shouldIncludeCompilerSolutionPathEvent,
} from './compiler/CompilerLogs';
import { formatContextOverflowActivityMessage } from '../utils/activityStyles';
import AggregatorLogs from './aggregator/AggregatorLogs';
import CompilerLogs from './compiler/CompilerLogs';
import { api, compilerAPI } from '../services/api';
import { websocket } from '../services/websocket';

describe('context overflow activity behavior', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  test('solution path activity stays in its owning manual workflow log', () => {
    expect(shouldIncludeAggregatorSolutionPathEvent({ workflow_mode: 'aggregator' })).toBe(true);
    expect(shouldIncludeAggregatorSolutionPathEvent({ workflow_mode: 'compiler' })).toBe(false);
    expect(shouldIncludeAggregatorSolutionPathEvent({ workflow_mode: 'autonomous' })).toBe(false);
    expect(shouldIncludeCompilerSolutionPathEvent({ workflow_mode: 'compiler' })).toBe(true);
    expect(shouldIncludeCompilerSolutionPathEvent({ workflow_mode: 'aggregator' })).toBe(false);
    expect(shouldIncludeCompilerSolutionPathEvent({ workflow_mode: 'leanoj' })).toBe(false);
  });

  test('Compiler accepts its own events and rejects autonomous, LeanOJ, and Aggregator events', () => {
    expect(shouldIncludeCompilerContextOverflow({
      workflow_mode: 'compiler',
      role_id: 'compiler_writer',
    })).toBe(true);
    expect(shouldIncludeCompilerContextOverflow({
      role_id: 'compiler_validator',
    })).toBe(true);
    expect(shouldIncludeCompilerContextOverflow({
      workflow_mode: 'autonomous',
      role_id: 'compiler_writer',
    })).toBe(false);
    expect(shouldIncludeCompilerContextOverflow({
      workflow_mode: 'leanoj',
      role_id: 'leanoj_final_solver',
    })).toBe(false);
    expect(shouldIncludeCompilerContextOverflow({
      workflow_mode: 'aggregator',
      role_id: 'aggregator_validator',
    })).toBe(false);
    expect(shouldIncludeCompilerContextOverflow({})).toBe(false);
  });

  test('App suppresses duplicate overflow terminal entries but retains ordinary stops', () => {
    expect(shouldRecordWorkflowStoppedActivity(
      'auto_research_stopped',
      { reason: 'context_overflow' },
    )).toBe(false);
    expect(shouldRecordWorkflowStoppedActivity(
      'leanoj_stopped',
      { reason: 'context_overflow' },
    )).toBe(false);
    expect(shouldRecordWorkflowStoppedActivity(
      'auto_research_stopped',
      { reason: 'user_stop' },
    )).toBe(true);
  });

  test('App reformats persisted direct and terminal overflow records with model identity', () => {
    localStorage.setItem('activity', JSON.stringify([
      {
        event: 'context_overflow_error',
        message: 'stale message',
        data: {
          message: 'Research stopped.',
          configured_model: 'configured-model',
          configured_provider: 'openrouter',
        },
      },
      {
        event: 'auto_research_stopped',
        message: 'stale terminal message',
        data: {
          reason: 'context_overflow',
          message: 'Research stopped.',
          effective_model: 'fallback-model',
          effective_provider: 'lm_studio',
        },
      },
    ]));

    const restored = readPersistedLiveActivity('activity');
    expect(restored[0].message).toContain('Configured route: configured-model via openrouter');
    expect(restored[1].message).toContain('Route: fallback-model via lm_studio');
  });

  test('Aggregator persisted overflow display includes stored model and provider', () => {
    expect(formatAggregatorPersistedOverflowMessage({
      type: 'context_overflow_error',
      message: 'legacy text without identity',
      metadata: {
        message: 'Research stopped.',
        configured_model: 'aggregator-model',
        configured_provider: 'openrouter',
      },
    })).toBe('Research stopped. Configured route: aggregator-model via openrouter.');
  });

  test('manual proof overflow routing assigns each source to exactly one manual feed', () => {
    const aggregator = { source_type: 'brainstorm', source_id: 'manual_aggregator' };
    const compiler = { source_type: 'paper', source_id: 'manual_compiler_current' };
    const autonomous = { source_type: 'paper', source_id: 'paper_123', workflow_mode: 'autonomous' };
    const leanoj = { source_type: 'leanoj_final', source_id: 'leanoj_123' };

    expect([
      shouldIncludeAggregatorProofContextOverflow(aggregator),
      shouldIncludeCompilerProofContextOverflow(aggregator),
    ]).toEqual([true, false]);
    expect([
      shouldIncludeAggregatorProofContextOverflow(compiler),
      shouldIncludeCompilerProofContextOverflow(compiler),
    ]).toEqual([false, true]);
    expect(shouldIncludeAggregatorProofContextOverflow(autonomous)).toBe(false);
    expect(shouldIncludeCompilerProofContextOverflow(autonomous)).toBe(false);
    expect(shouldIncludeAggregatorProofContextOverflow(leanoj)).toBe(false);
    expect(shouldIncludeCompilerProofContextOverflow(leanoj)).toBe(false);
  });

  test('formatter distinguishes changed routes and tolerates partial route metadata', () => {
    expect(formatContextOverflowActivityMessage({
      message: 'Proof context overflow.',
      configured_model: 'primary-model',
      configured_provider: 'openrouter',
      effective_model: 'fallback-model',
      effective_provider: 'lm_studio',
      effective_host_provider: 'local-sibling',
    })).toContain(
      'Effective route: fallback-model via lm_studio, host local-sibling. '
      + 'Configured route: primary-model via openrouter.'
    );
    expect(formatContextOverflowActivityMessage({
      message: 'Proof context overflow.',
      effective_provider: 'openrouter',
      effective_host_provider: 'anthropic',
    })).toContain('Route: openrouter, host anthropic.');
  });

  test('compiler persistence compacts and bounds large event payloads', () => {
    const compacted = compactCompilerActivityEvents(Array.from({ length: 2100 }, (_, index) => ({
      type: 'proof_context_overflow',
      timestamp: `${index}`,
      fullTimestamp: `2026-07-13T00:00:${index}.000Z`,
      data: {
        configured_model: 'configured-model',
        effective_model: 'effective-model',
        effective_host_provider: 'anthropic',
        error_output: 'x'.repeat(5000),
      },
    })));
    expect(compacted).toHaveLength(2000);
    expect(compacted[0].data.configured_model).toBe('configured-model');
    expect(compacted[0].data.effective_host_provider).toBe('anthropic');
    expect(compacted[0].data.error_output.length).toBeLessThan(1300);
  });

  test('mounted manual logs route proof overflow to exactly one owning feed', async () => {
    const compilerMetrics = {
      construction: { acceptances: 0, rejections: 0, declines: 0, acceptance_rate: 0 },
      rigor: { acceptances: 0, rejections: 0, declines: 0, acceptance_rate: 0 },
      outline: { acceptances: 0, rejections: 0, declines: 0 },
      review: { acceptances: 0, rejections: 0, declines: 0 },
      minuscule_edit_count: 0,
      paper_word_count: 0,
      total_submissions: 0,
    };
    vi.spyOn(api, 'getStatus').mockResolvedValue({
      queue_size: 0,
      total_acceptances: 0,
      total_rejections: 0,
      submitter_statuses: [],
    });
    vi.spyOn(compilerAPI, 'getMetrics').mockResolvedValue({ data: compilerMetrics });
    vi.spyOn(compilerAPI, 'getStatus').mockResolvedValue({
      data: { current_mode: 'construction', is_running: true },
    });
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => ({ events: [] }),
    });

    render(
      <>
        <AggregatorLogs />
        <CompilerLogs />
      </>
    );

    await waitFor(() => {
      expect(api.getStatus).toHaveBeenCalled();
      expect(compilerAPI.getStatus).toHaveBeenCalled();
      expect(globalThis.fetch).toHaveBeenCalled();
    });
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
      websocket.emit('proof_context_overflow', {
        source_type: 'brainstorm',
        source_id: 'manual_aggregator',
        message: 'Aggregator-only overflow.',
        configured_model: 'agg-model',
      });
      websocket.emit('proof_context_overflow', {
        source_type: 'paper',
        source_id: 'manual_compiler_current',
        message: 'Compiler-only overflow.',
        configured_model: 'compiler-model',
      });
      websocket.emit('proof_context_overflow', {
        source_type: 'paper',
        source_id: 'autonomous-paper',
        workflow_mode: 'autonomous',
        message: 'App-owned overflow.',
      });
    });

    await waitFor(() => {
      expect(screen.getAllByText(/Aggregator-only overflow/)).toHaveLength(1);
      expect(screen.getAllByText(/Compiler-only overflow/)).toHaveLength(1);
    });
    expect(screen.queryByText(/App-owned overflow/)).toBeNull();
  });
});
