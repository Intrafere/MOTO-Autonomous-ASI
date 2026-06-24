export function readBooleanStorage(key, fallback = false) {
  if (typeof window === 'undefined') {
    return fallback;
  }

  try {
    const raw = window.localStorage.getItem(key);
    if (raw === null) {
      return fallback;
    }

    if (raw === 'true') {
      return true;
    }
    if (raw === 'false') {
      return false;
    }

    const parsed = JSON.parse(raw);
    return typeof parsed === 'boolean' ? parsed : fallback;
  } catch (error) {
    console.warn(`Ignoring invalid boolean localStorage value for ${key}:`, error);
    return fallback;
  }
}
