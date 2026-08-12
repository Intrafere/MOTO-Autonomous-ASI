import { MANUAL_AGGREGATOR_PROOF_SOURCE_ID, MANUAL_COMPILER_CURRENT_PROOF_SOURCE_ID } from './manualProofSources';
import { formatContextOverflowActivityMessage } from './activityStyles';

const MAX_COMPILER_ACTIVITY_EVENTS = 2000;
const MAX_PERSISTED_TEXT_LENGTH = 1200;

export const shouldIncludeAggregatorProofContextOverflow = (data = {}) => (
  data.source_type === 'brainstorm'
  && data.source_id === MANUAL_AGGREGATOR_PROOF_SOURCE_ID
);

export const shouldIncludeAggregatorSolutionPathEvent = (data = {}) => {
  const workflowMode = String(data.workflow_mode || data.mode || '').toLowerCase();
  return !workflowMode || workflowMode === 'aggregator';
};

export const formatAggregatorPersistedOverflowMessage = (event = {}) => (
  formatContextOverflowActivityMessage(event.metadata || {})
);

export const shouldIncludeCompilerContextOverflow = (data = {}) => {
  const roleId = String(data.role_id || '').toLowerCase();
  const workflowMode = String(data.workflow_mode || '').toLowerCase();
  return !(
    (workflowMode && workflowMode !== 'compiler')
    || (!workflowMode && !roleId.startsWith('compiler_'))
  );
};

export const shouldIncludeCompilerProofContextOverflow = (data = {}) => (
  data.source_type === 'paper'
  && (
    data.source_id === MANUAL_COMPILER_CURRENT_PROOF_SOURCE_ID
    || String(data.source_id || '').startsWith('manual_compiler_')
    || String(data.source_id || '').startsWith('compiler_manual_')
  )
);

export const shouldIncludeCompilerSolutionPathEvent = (data = {}) => {
  const workflowMode = String(data.workflow_mode || data.mode || '').toLowerCase();
  return !workflowMode || workflowMode === 'compiler';
};

const compactPersistedValue = (value) => {
  if (typeof value === 'string') {
    return value.length > MAX_PERSISTED_TEXT_LENGTH
      ? `${value.slice(0, MAX_PERSISTED_TEXT_LENGTH)}...`
      : value;
  }
  if (Array.isArray(value)) {
    return value.slice(0, 20).map(compactPersistedValue);
  }
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value)
        .slice(0, 40)
        .map(([key, nested]) => [key, compactPersistedValue(nested)])
    );
  }
  return value;
};

export const compactCompilerActivityEvents = (events = []) => (
  events.slice(0, MAX_COMPILER_ACTIVITY_EVENTS).map((event) => ({
    type: event.type,
    timestamp: event.timestamp,
    fullTimestamp: event.fullTimestamp,
    data: compactPersistedValue(event.data || {}),
  }))
);
