import { useCallback, useEffect, useRef, useState } from 'react';
import { readPromptDraft, readPromptDraftSync, savePromptDraft } from '../utils/promptDraftStorage';

/**
 * Durable prompt draft for a prompt composer that unmounts on tab switches.
 *
 * Hydration fills an empty composer from the saved draft, then from the backend's
 * persisted prompt. User edits are saved immediately, even while hydration is still
 * pending, and hydration never replaces a prompt the user has already edited.
 * Until the first edit or the end of hydration nothing is written, so the initial
 * empty value cannot erase a large IndexedDB draft before it is read.
 */
export default function usePromptDraft(storageKey, loadPersistedPrompt) {
  const [prompt, setPromptState] = useState(() => readPromptDraftSync(storageKey));
  const [hydrated, setHydrated] = useState(false);
  const editedRef = useRef(false);
  const loadPersistedPromptRef = useRef(loadPersistedPrompt);
  loadPersistedPromptRef.current = loadPersistedPrompt;

  const setPrompt = useCallback((nextPrompt) => {
    editedRef.current = true;
    setPromptState(nextPrompt);
  }, []);

  useEffect(() => {
    if (hydrated || editedRef.current) {
      savePromptDraft(storageKey, prompt);
    }
  }, [storageKey, prompt, hydrated]);

  useEffect(() => {
    let cancelled = false;
    const fillIfUntouched = (candidate) => {
      if (!candidate || cancelled || editedRef.current) {
        return;
      }
      setPromptState((current) => (current.trim() ? current : candidate));
    };

    const hydrate = async () => {
      try {
        const savedDraft = await readPromptDraft(storageKey);
        fillIfUntouched(savedDraft);
        const loadPersisted = loadPersistedPromptRef.current;
        if (!savedDraft.trim() && loadPersisted && !cancelled && !editedRef.current) {
          fillIfUntouched(await loadPersisted());
        }
      } catch (error) {
        console.debug(`Could not hydrate prompt draft ${storageKey}:`, error);
      } finally {
        if (!cancelled) {
          setHydrated(true);
        }
      }
    };

    hydrate();
    return () => {
      cancelled = true;
    };
  }, [storageKey]);

  return [prompt, setPrompt];
}
