export const CODEX_PROVIDER = 'openai_codex_oauth';
export const CODEX_EFFORT_ORDER = ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'];

export function getCodexReasoningInfo(model, modelId = model?.id) {
  if (modelId === 'gpt-5.3-codex-spark-high') {
    return { status: 'fixed', levels: ['high'], maximum: 'high' };
  }
  const raw = model?.supported_reasoning_levels;
  if (!Array.isArray(raw)) return { status: 'unknown', levels: [], maximum: null };
  const levels = CODEX_EFFORT_ORDER.filter(level => raw.some(entry => (
    typeof entry === 'string' ? entry : entry?.effort
  )?.trim?.().toLowerCase() === level));
  return { status: levels.length ? 'supported' : (raw.length ? 'unknown' : 'unsupported'), levels, maximum: levels.at(-1) || null };
}
