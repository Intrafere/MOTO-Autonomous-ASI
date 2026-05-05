const storagePrefix = (import.meta.env.VITE_MOTO_STORAGE_PREFIX || '').trim();
const instanceId = (import.meta.env.VITE_MOTO_INSTANCE_ID || '').trim();
const dataRootDisplay = (import.meta.env.VITE_MOTO_DATA_ROOT_DISPLAY || '').trim();

function toScopedKey(key) {
  if (!storagePrefix || typeof key !== 'string' || key.length === 0) {
    return key;
  }
  return `${storagePrefix}:${key}`;
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

export function getRuntimeDataPath(relativePath = '') {
  const normalizedRelativePath = String(relativePath || '').replace(/^[/\\]+/, '');
  const basePath = dataRootDisplay || 'this instance data root';
  return normalizedRelativePath ? `${basePath}/${normalizedRelativePath}` : basePath;
}

export function getRuntimeInstanceId() {
  return instanceId || 'default';
}

