const storagePrefix = (import.meta.env.VITE_MOTO_STORAGE_PREFIX || '').trim();
const instanceId = (import.meta.env.VITE_MOTO_INSTANCE_ID || '').trim();
const dataRootDisplay = (import.meta.env.VITE_MOTO_DATA_ROOT_DISPLAY || '').trim();
const desktopApiToken = (import.meta.env.VITE_MOTO_DESKTOP_API_TOKEN || '').trim();
const backendUrl = (import.meta.env.VITE_MOTO_BACKEND_URL || '').trim();
const DESKTOP_TOKEN_HEADER = 'X-Moto-Desktop-Token';

function toScopedKey(key) {
  if (!storagePrefix || typeof key !== 'string' || key.length === 0) {
    return key;
  }
  if (key.startsWith(`${storagePrefix}:`)) {
    return key;
  }
  return `${storagePrefix}:${key}`;
}

export function getNamespacedStorageKey(key) {
  return toScopedKey(key);
}

export function installNamespacedLocalStorage() {
  if (typeof window === 'undefined' || !storagePrefix) {
    return;
  }

  if (window.__motoStorageNamespacePatched) {
    return;
  }

  const storageProto = Object.getPrototypeOf(window.localStorage);
  const originalGetItem = storageProto.getItem;
  const originalSetItem = storageProto.setItem;
  const originalRemoveItem = storageProto.removeItem;

  storageProto.getItem = function patchedGetItem(key) {
    if (this === window.localStorage) {
      return originalGetItem.call(this, toScopedKey(key));
    }
    return originalGetItem.call(this, key);
  };

  storageProto.setItem = function patchedSetItem(key, value) {
    if (this === window.localStorage) {
      return originalSetItem.call(this, toScopedKey(key), value);
    }
    return originalSetItem.call(this, key, value);
  };

  storageProto.removeItem = function patchedRemoveItem(key) {
    if (this === window.localStorage) {
      return originalRemoveItem.call(this, toScopedKey(key));
    }
    return originalRemoveItem.call(this, key);
  };

  window.__motoStorageNamespacePatched = true;
}

function shouldAttachDesktopToken(input) {
  if (!desktopApiToken || typeof window === 'undefined') {
    return false;
  }

  try {
    const rawUrl = typeof input === 'string'
      ? input
      : (input && typeof input.url === 'string' ? input.url : '');
    if (!rawUrl) {
      return false;
    }

    const requestUrl = new URL(rawUrl, window.location.origin);
    if (!requestUrl.pathname.startsWith('/api')) {
      return false;
    }

    if (requestUrl.origin === window.location.origin) {
      return true;
    }

    if (backendUrl) {
      return requestUrl.origin === new URL(backendUrl, window.location.origin).origin;
    }
  } catch {
    return false;
  }

  return false;
}

function withDesktopTokenHeaders(headers) {
  const nextHeaders = new Headers(headers || {});
  if (!nextHeaders.has(DESKTOP_TOKEN_HEADER)) {
    nextHeaders.set(DESKTOP_TOKEN_HEADER, desktopApiToken);
  }
  return nextHeaders;
}

export function installAuthenticatedFetch() {
  if (typeof window === 'undefined' || !desktopApiToken || window.__motoAuthFetchPatched) {
    return;
  }

  const originalFetch = window.fetch.bind(window);
  window.fetch = (input, init = {}) => {
    if (!shouldAttachDesktopToken(input)) {
      return originalFetch(input, init);
    }

    if (input instanceof Request) {
      const request = new Request(input, {
        ...init,
        headers: withDesktopTokenHeaders(init.headers || input.headers),
      });
      return originalFetch(request);
    }

    return originalFetch(input, {
      ...init,
      headers: withDesktopTokenHeaders(init.headers),
    });
  };
  window.__motoAuthFetchPatched = true;
}

export function getRuntimeDataPath(relativePath = '') {
  const normalizedRelativePath = String(relativePath || '').replace(/^[/\\]+/, '');
  const basePath = dataRootDisplay || 'this instance data root';
  return normalizedRelativePath ? `${basePath}/${normalizedRelativePath}` : basePath;
}

export function getRuntimeInstanceId() {
  return instanceId || 'default';
}

export function getDesktopApiToken() {
  return desktopApiToken;
}

