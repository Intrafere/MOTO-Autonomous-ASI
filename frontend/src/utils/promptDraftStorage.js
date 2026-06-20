const PROMPT_DRAFT_DB_NAME = 'moto_prompt_drafts';
const PROMPT_DRAFT_STORE_NAME = 'drafts';
const PROMPT_DRAFT_DB_VERSION = 1;
const LOCAL_STORAGE_PROMPT_CHAR_LIMIT = 250000;
const storagePrefix = (import.meta.env.VITE_MOTO_STORAGE_PREFIX || '').trim();

function getScopedPromptKey(key) {
  const normalizedKey = String(key || '');
  return storagePrefix ? `${storagePrefix}:${normalizedKey}` : normalizedKey;
}

function canUseBrowserStorage() {
  return typeof window !== 'undefined' && typeof window.localStorage !== 'undefined';
}

function canUseSessionStorage() {
  return typeof window !== 'undefined' && typeof window.sessionStorage !== 'undefined';
}

function canUseIndexedDb() {
  return typeof window !== 'undefined' && typeof window.indexedDB !== 'undefined';
}

function openPromptDraftDb() {
  if (!canUseIndexedDb()) {
    return Promise.resolve(null);
  }

  return new Promise((resolve, reject) => {
    const request = window.indexedDB.open(PROMPT_DRAFT_DB_NAME, PROMPT_DRAFT_DB_VERSION);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(PROMPT_DRAFT_STORE_NAME)) {
        db.createObjectStore(PROMPT_DRAFT_STORE_NAME);
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function withPromptDraftStore(mode, callback) {
  const db = await openPromptDraftDb();
  if (!db) {
    return null;
  }

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(PROMPT_DRAFT_STORE_NAME, mode);
    const store = transaction.objectStore(PROMPT_DRAFT_STORE_NAME);
    let callbackResult;

    transaction.oncomplete = () => {
      db.close();
      resolve(callbackResult);
    };
    transaction.onerror = () => {
      db.close();
      reject(transaction.error);
    };
    transaction.onabort = () => {
      db.close();
      reject(transaction.error);
    };

    callbackResult = callback(store);
  });
}

async function writeIndexedDraft(key, prompt) {
  await withPromptDraftStore('readwrite', (store) => {
    store.put(String(prompt || ''), getScopedPromptKey(key));
  });
}

async function readIndexedDraft(key) {
  const db = await openPromptDraftDb();
  if (!db) {
    return '';
  }

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(PROMPT_DRAFT_STORE_NAME, 'readonly');
    const store = transaction.objectStore(PROMPT_DRAFT_STORE_NAME);
    const request = store.get(getScopedPromptKey(key));
    let draft = '';

    request.onsuccess = () => {
      draft = typeof request.result === 'string' ? request.result : '';
    };
    request.onerror = () => {
      db.close();
      reject(request.error);
    };
    transaction.oncomplete = () => {
      db.close();
      resolve(draft);
    };
    transaction.onerror = () => {
      db.close();
      reject(transaction.error);
    };
    transaction.onabort = () => {
      db.close();
      reject(transaction.error);
    };
  });
}

async function removeIndexedDraft(key) {
  await withPromptDraftStore('readwrite', (store) => {
    store.delete(getScopedPromptKey(key));
  });
}

export function canStorePromptDraftInLocalStorage(prompt) {
  return String(prompt || '').length <= LOCAL_STORAGE_PROMPT_CHAR_LIMIT;
}

export function readPromptDraftSync(key) {
  if (canUseBrowserStorage()) {
    try {
      const localDraft = window.localStorage.getItem(key) || '';
      if (localDraft) {
        return localDraft;
      }
    } catch (error) {
      console.debug('Could not read prompt draft from localStorage:', error);
    }
  }

  if (canUseSessionStorage()) {
    try {
      return window.sessionStorage.getItem(key) || '';
    } catch (error) {
      console.debug('Could not read prompt draft from sessionStorage:', error);
    }
  }

  return '';
}

export async function readPromptDraft(key) {
  const localDraft = readPromptDraftSync(key);
  if (localDraft) {
    return localDraft;
  }

  try {
    return await readIndexedDraft(key);
  } catch (error) {
    console.debug('Could not read prompt draft from IndexedDB:', error);
    return '';
  }
}

export function savePromptDraft(key, prompt) {
  const normalizedPrompt = String(prompt || '');
  let syncStored = false;

  if (canUseBrowserStorage()) {
    try {
      if (canStorePromptDraftInLocalStorage(normalizedPrompt)) {
        window.localStorage.setItem(key, normalizedPrompt);
        syncStored = true;
      } else {
        window.localStorage.removeItem(key);
      }
    } catch (error) {
      console.debug('Could not save prompt draft to localStorage:', error);
      try {
        window.localStorage.removeItem(key);
      } catch {
        // Ignore cleanup failures; IndexedDB remains the larger-draft fallback.
      }
    }
  }

  if (canUseSessionStorage()) {
    try {
      if (!syncStored && canStorePromptDraftInLocalStorage(normalizedPrompt)) {
        window.sessionStorage.setItem(key, normalizedPrompt);
      } else if (syncStored || !canStorePromptDraftInLocalStorage(normalizedPrompt)) {
        window.sessionStorage.removeItem(key);
      }
    } catch (error) {
      console.debug('Could not save prompt draft to sessionStorage:', error);
      try {
        window.sessionStorage.removeItem(key);
      } catch {
        // Ignore cleanup failures; IndexedDB remains the larger-draft fallback.
      }
    }
  }

  void writeIndexedDraft(key, normalizedPrompt).catch((error) => {
    console.debug('Could not save prompt draft to IndexedDB:', error);
  });
}

export function removePromptDraft(key) {
  if (canUseBrowserStorage()) {
    try {
      window.localStorage.removeItem(key);
    } catch (error) {
      console.debug('Could not remove prompt draft from localStorage:', error);
    }
  }

  if (canUseSessionStorage()) {
    try {
      window.sessionStorage.removeItem(key);
    } catch (error) {
      console.debug('Could not remove prompt draft from sessionStorage:', error);
    }
  }

  void removeIndexedDraft(key).catch((error) => {
    console.debug('Could not remove prompt draft from IndexedDB:', error);
  });
}
