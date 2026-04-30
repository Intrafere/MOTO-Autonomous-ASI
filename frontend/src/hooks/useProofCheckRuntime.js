import { useCallback, useEffect, useMemo, useState } from 'react';
import { autonomousAPI } from '../services/api';
import { websocket } from '../services/websocket';

function buildSourceKey(sourceType, sourceId) {
  return `${sourceType}:${sourceId}`;
}

export function useProofCheckRuntime() {
  const [proofStatus, setProofStatus] = useState(null);
  const [runtimeError, setRuntimeError] = useState('');
  const [activeChecks, setActiveChecks] = useState({});
  const [queuedChecks, setQueuedChecks] = useState({});

  const refreshProofStatus = useCallback(async () => {
    try {
      const status = await autonomousAPI.getProofStatus();
      setProofStatus(status);
      setRuntimeError('');
      return status;
    } catch (err) {
      setRuntimeError(err.message || 'Failed to load proof status');
      return null;
    }
  }, []);

  useEffect(() => {
    refreshProofStatus();
  }, [refreshProofStatus]);

  useEffect(() => {
    const unsubscribeStarted = websocket.on('proof_check_started', (data) => {
      const sourceKey = buildSourceKey(data.source_type, data.source_id);
      setActiveChecks((prev) => ({
        ...prev,
        [sourceKey]: {
          status: 'running',
          candidateCount: prev[sourceKey]?.candidateCount ?? null,
        },
      }));
      setQueuedChecks((prev) => {
        if (!prev[sourceKey]) {
          return prev;
        }
        const next = { ...prev };
        delete next[sourceKey];
        return next;
      });
    });

    const unsubscribeCandidates = websocket.on('proof_check_candidates_found', (data) => {
      const sourceKey = buildSourceKey(data.source_type, data.source_id);
      setActiveChecks((prev) => ({
        ...prev,
        [sourceKey]: {
          status: 'running',
          candidateCount: data.count ?? null,
        },
      }));
    });

    const unsubscribeComplete = websocket.on('proof_check_complete', (data) => {
      const sourceKey = buildSourceKey(data.source_type, data.source_id);
      setActiveChecks((prev) => {
        if (!prev[sourceKey]) {
          return prev;
        }
        const next = { ...prev };
        delete next[sourceKey];
        return next;
      });
      setQueuedChecks((prev) => {
        if (!prev[sourceKey]) {
          return prev;
        }
        const next = { ...prev };
        delete next[sourceKey];
        return next;
      });
      refreshProofStatus();
    });

    return () => {
      unsubscribeStarted();
      unsubscribeCandidates();
      unsubscribeComplete();
    };
  }, [refreshProofStatus]);

  const queueManualProofCheck = useCallback(async ({ sourceType, sourceId }) => {
    const sourceKey = buildSourceKey(sourceType, sourceId);
    setQueuedChecks((prev) => ({
      ...prev,
      [sourceKey]: true,
    }));

    try {
      return await autonomousAPI.runProofCheck({ sourceType, sourceId });
    } catch (err) {
      setQueuedChecks((prev) => {
        if (!prev[sourceKey]) {
          return prev;
        }
        const next = { ...prev };
        delete next[sourceKey];
        return next;
      });
      throw err;
    }
  }, []);

  const getSourceState = useCallback((sourceType, sourceId) => {
    const sourceKey = buildSourceKey(sourceType, sourceId);
    if (activeChecks[sourceKey]) {
      return activeChecks[sourceKey];
    }
    if (queuedChecks[sourceKey]) {
      return {
        status: 'queued',
        candidateCount: null,
      };
    }
    return null;
  }, [activeChecks, queuedChecks]);

  const isSourceBusy = useCallback((sourceType, sourceId) => {
    return Boolean(getSourceState(sourceType, sourceId));
  }, [getSourceState]);

  const manualCheckReason = useMemo(() => {
    if (!proofStatus) {
      return 'Loading proof runtime status...';
    }
    if (!proofStatus.lean4_enabled) {
      return 'Lean 4 proof checks are disabled.';
    }
    if (!proofStatus.manual_check_ready) {
      return proofStatus.manual_check_message || 'Manual proof checks are not ready yet.';
    }
    return '';
  }, [proofStatus]);

  return {
    proofStatus,
    runtimeError,
    refreshProofStatus,
    queueManualProofCheck,
    getSourceState,
    isSourceBusy,
    manualCheckEnabled: Boolean(proofStatus?.lean4_enabled && proofStatus?.manual_check_ready),
    manualCheckReason,
  };
}
