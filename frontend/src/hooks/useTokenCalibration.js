import { useEffect, useSyncExternalStore } from 'react';
import { workflowAPI } from '../services/api';

// One shared copy of the backend's measured token-count ratios, so every
// Context Window field and the Token Usage panel read the same snapshot from a
// single request.
const EMPTY = Object.freeze({ adjust_enabled: true, samples_required: 10, models: [] });
const STALE_AFTER_MS = 30000;

let snapshot = EMPTY;
let fetchedAt = 0;
let inFlight = null;
const subscribers = new Set();

const publish = (next) => {
  if (!next || !Array.isArray(next.models)) return;
  snapshot = next;
  fetchedAt = Date.now();
  subscribers.forEach((notify) => notify());
};

export const refreshTokenCalibration = async () => {
  if (inFlight) return inFlight;
  inFlight = (async () => {
    try {
      publish(await workflowAPI.getTokenCalibration());
    } catch {
      // Measurements are informational; a failed read keeps the last snapshot.
    } finally {
      inFlight = null;
    }
  })();
  return inFlight;
};

export const setTokenCalibrationAdjustEnabled = async (enabled) => {
  publish(await workflowAPI.setTokenCalibrationAdjustEnabled(enabled));
};

export const resetTokenCalibration = async (provider, model) => {
  publish(await workflowAPI.resetTokenCalibration(provider, model));
};

const subscribe = (notify) => {
  subscribers.add(notify);
  return () => subscribers.delete(notify);
};

const getSnapshot = () => snapshot;

export const findTokenCalibration = (calibration, provider, model) => {
  if (!provider || !model) return null;
  // Loaded LM Studio sibling instances ("model:2") share one measurement.
  const modelId = provider === 'lm_studio' ? String(model).replace(/:\d+$/, '') : String(model);
  return (calibration?.models || []).find(
    (entry) => entry.provider === provider && entry.model === modelId,
  ) || null;
};

export function useTokenCalibration() {
  const calibration = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  useEffect(() => {
    if (Date.now() - fetchedAt > STALE_AFTER_MS) refreshTokenCalibration();
  }, []);
  return calibration;
}

// Test-only: forget the shared snapshot between cases.
export const resetTokenCalibrationStoreForTests = () => {
  snapshot = EMPTY;
  fetchedAt = 0;
  inFlight = null;
};
