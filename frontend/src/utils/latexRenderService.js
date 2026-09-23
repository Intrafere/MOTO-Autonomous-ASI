/** Shared bounded worker lane. Returned HTML is UNSANITIZED. No synchronous fallback. */
export const LATEX_RENDER_LIMITS = Object.freeze({ input: 64000, output: 1500000, queued: 48, queuedBytes: 1000000, timeoutMs: 8000 });
const abortError = () => new DOMException('Rendering cancelled', 'AbortError');
let worker = null, active = null, nextId = 0, queuedBytes = 0, screenStreak = 0;
const queue = [];
function destroyWorker() { worker?.terminate(); worker = null; }
function settle(job, error, html) {
  clearTimeout(job.timer);
  job.signal?.removeEventListener('abort', job.abort);
  if (active === job) active = null;
  if (error) job.reject(error); else job.resolve(html);
  queueMicrotask(pump);
}
function pump() {
  if (active || !queue.length) return;
  // Screen requests outrank exports; periodic export service prevents starvation.
  let index = queue.findIndex(job => job.priority === (screenStreak >= 4 ? 'export' : 'screen'));
  if (index < 0) index = 0;
  const job = queue.splice(index, 1)[0]; queuedBytes -= job.text.length;
  active = job;
  screenStreak = job.priority === 'screen' ? screenStreak + 1 : 0;
  try {
    if (!worker) {
      worker = new Worker(new URL('./latexRender.worker.js', import.meta.url), { type: 'module' });
      const ownedWorker = worker;
      worker.onmessage = ({ data }) => {
        if (worker !== ownedWorker || !active) return;
        if (!data || typeof data !== 'object') {
          const current = active;
          destroyWorker();
          settle(current, new Error('LaTeX worker returned an invalid response.'));
          return;
        }
        if (data.id !== active.id) return;
        const current = active;
        if (data.error || typeof data.html !== 'string' || data.html.length > LATEX_RENDER_LIMITS.output) {
          settle(current, new Error(data.error || 'Rendered block exceeds the safe HTML limit.'));
        } else settle(current, null, data.html);
      };
      worker.onerror = () => { if (worker !== ownedWorker) return; const current = active; destroyWorker(); if (current) settle(current, new Error('LaTeX worker failed.')); };
      worker.onmessageerror = worker.onerror;
    }
    job.timer = setTimeout(() => { if (active !== job) return; destroyWorker(); settle(job, new Error('LaTeX block exceeded the rendering time limit.')); }, LATEX_RENDER_LIMITS.timeoutMs);
    worker.postMessage({ id: job.id, text: job.text });
  } catch (error) { destroyWorker(); settle(job, error); }
}
export function renderLatexAsync(text, { signal, priority = 'screen' } = {}) {
  if (signal?.aborted) return Promise.reject(abortError());
  if (typeof text !== 'string' || text.length > LATEX_RENDER_LIMITS.input) return Promise.reject(new Error('Atomic LaTeX block exceeds the safe input limit.'));
  if (queue.length >= LATEX_RENDER_LIMITS.queued || queuedBytes + text.length > LATEX_RENDER_LIMITS.queuedBytes) return Promise.reject(new Error('LaTeX rendering queue is full.'));
  return new Promise((resolve, reject) => {
    const job = { id: ++nextId, text, priority, signal, resolve, reject };
    job.abort = () => {
      if (active === job) { destroyWorker(); settle(job, abortError()); }
      else {
        const index = queue.indexOf(job);
        if (index >= 0) { queue.splice(index, 1); queuedBytes -= job.text.length; settle(job, abortError()); }
      }
    };
    signal?.addEventListener('abort', job.abort, { once: true });
    queue.push(job); queuedBytes += text.length; queueMicrotask(pump);
  });
}
/** Yield before DOMPurify work; abortable and shared across screen/export consumers. */
export function scheduleLatexMainTask(task, { signal } = {}) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) { reject(abortError()); return; }
    const abort = () => { clearTimeout(timer); reject(abortError()); };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', abort);
      if (signal?.aborted) { reject(abortError()); return; }
      try { resolve(task()); } catch (error) { reject(error); }
    }, 0);
    signal?.addEventListener('abort', abort, { once: true });
  });
}
