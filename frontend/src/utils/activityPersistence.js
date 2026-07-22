const REDACTED = '[redacted]';

const SENSITIVE_FIELD_PATTERN = /^(?:access[_-]?token|api[_-]?key|app[_-]?id|authorization|bearer|client[_-]?secret|code[_-]?verifier|credential|id[_-]?token|password|refresh[_-]?token|secret|session[_-]?token|token)$/i;
const NAMED_SECRET_PATTERN = /((?:access[_-]?token|api[_-]?key|app[_-]?id|authorization|client[_-]?secret|code[_-]?verifier|id[_-]?token|password|refresh[_-]?token|secret|session[_-]?token|token)\s*["']?\s*[:=]\s*["']?)([^"',\s&#]+)/gi;
const BEARER_PATTERN = /\b(Bearer\s+)[A-Za-z0-9._~+/-]+=*/gi;
const OPENROUTER_KEY_PATTERN = /\bsk-or-v1-[A-Za-z0-9_-]+\b/gi;
const OAUTH_URL_PARAMETER_PATTERN = /([?&#](?:access_token|authorization|client_secret|code|code_verifier|id_token|refresh_token|token)=)([^&#\s]+)/gi;

export function redactPersistedActivityText(value) {
  return String(value ?? '')
    .replace(BEARER_PATTERN, `$1${REDACTED}`)
    .replace(OPENROUTER_KEY_PATTERN, REDACTED)
    .replace(OAUTH_URL_PARAMETER_PATTERN, `$1${REDACTED}`)
    .replace(NAMED_SECRET_PATTERN, `$1${REDACTED}`);
}

export function sanitizePersistedActivityValue(value, seen = new WeakSet()) {
  if (value == null || typeof value === 'number' || typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'string') {
    return redactPersistedActivityText(value);
  }
  if (typeof value !== 'object') {
    return redactPersistedActivityText(value);
  }
  if (seen.has(value)) {
    return '[omitted circular value]';
  }

  seen.add(value);
  let sanitized;
  if (Array.isArray(value)) {
    sanitized = value.map((item) => sanitizePersistedActivityValue(item, seen));
  } else {
    sanitized = Object.fromEntries(
      Object.entries(value).map(([key, nestedValue]) => [
        key,
        SENSITIVE_FIELD_PATTERN.test(key)
          ? REDACTED
          : sanitizePersistedActivityValue(nestedValue, seen),
      ]),
    );
  }
  seen.delete(value);
  return sanitized;
}

export function isSensitivePersistedActivityField(key) {
  return SENSITIVE_FIELD_PATTERN.test(String(key || ''));
}
