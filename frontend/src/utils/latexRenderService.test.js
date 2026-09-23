import { afterEach, describe, expect, it, vi } from 'vitest';
class FakeWorker {
  static instances = [];
  constructor() { FakeWorker.instances.push(this); }
  postMessage = vi.fn();
  terminate = vi.fn();
  finish(html = '<b>ok</b>') { this.onmessage({ data: { id: this.postMessage.mock.lastCall[0].id, html } }); }
}
afterEach(() => { vi.unstubAllGlobals(); vi.resetModules(); FakeWorker.instances = []; });
describe('bounded shared renderer worker', () => {
  it('prioritizes screen work, reuses one worker, and returns unsanitized HTML', async () => {
    vi.stubGlobal('Worker', FakeWorker);
    const { renderLatexAsync } = await import('./latexRenderService');
    const exported = renderLatexAsync('export', { priority: 'export' });
    const screen = renderLatexAsync('screen');
    await Promise.resolve();
    const worker = FakeWorker.instances[0];
    expect(worker.postMessage.mock.lastCall[0].text).toBe('screen');
    worker.finish('<script>unsafe</script>');
    expect(await screen).toBe('<script>unsafe</script>');
    await Promise.resolve(); worker.finish(); await exported;
    expect(FakeWorker.instances).toHaveLength(1);
  });
  it('terminates cancelled work and ignores stale worker replies', async () => {
    vi.stubGlobal('Worker', FakeWorker);
    const { renderLatexAsync } = await import('./latexRenderService');
    const controller = new AbortController();
    const first = renderLatexAsync('old', { signal: controller.signal });
    const rejected = expect(first).rejects.toMatchObject({ name: 'AbortError' });
    await Promise.resolve(); const oldWorker = FakeWorker.instances[0];
    controller.abort(); await rejected;
    const next = renderLatexAsync('new'); await Promise.resolve();
    oldWorker.finish('stale');
    const worker = FakeWorker.instances.at(-1); worker.finish('fresh');
    expect(await next).toBe('fresh'); expect(oldWorker.terminate).toHaveBeenCalled();
  });
  it('times out a hung worker and continues queued work', async () => {
    vi.useFakeTimers(); vi.stubGlobal('Worker', FakeWorker);
    const { renderLatexAsync, LATEX_RENDER_LIMITS } = await import('./latexRenderService');
    const first = renderLatexAsync('hung');
    const rejected = expect(first).rejects.toThrow('time limit');
    await Promise.resolve();
    await vi.advanceTimersByTimeAsync(LATEX_RENDER_LIMITS.timeoutMs);
    await rejected; expect(FakeWorker.instances[0].terminate).toHaveBeenCalled();
    vi.useRealTimers();
  });
  it('bounds queued input and rejects oversized output', async () => {
    vi.stubGlobal('Worker', FakeWorker);
    const { renderLatexAsync, LATEX_RENDER_LIMITS } = await import('./latexRenderService');
    const controllers = Array.from({ length: LATEX_RENDER_LIMITS.queued }, () => new AbortController());
    const jobs = controllers.map(controller => renderLatexAsync('q', { signal: controller.signal }).catch(error => error));
    await expect(renderLatexAsync('overflow')).rejects.toThrow('queue is full');
    controllers.forEach(controller => controller.abort()); await Promise.all(jobs);
    const output = renderLatexAsync('small'); const rejection = expect(output).rejects.toThrow('HTML limit');
    await Promise.resolve(); FakeWorker.instances.at(-1).finish('x'.repeat(LATEX_RENDER_LIMITS.output + 1)); await rejection;
  });
  it('rejects malformed worker envelopes and recovers on the next request', async () => {
    vi.stubGlobal('Worker', FakeWorker);
    const { renderLatexAsync } = await import('./latexRenderService');
    const first = renderLatexAsync('first');
    const rejected = expect(first).rejects.toThrow('invalid response');
    await Promise.resolve();
    FakeWorker.instances[0].onmessage({ data: null });
    await rejected;
    const next = renderLatexAsync('next');
    await Promise.resolve();
    FakeWorker.instances.at(-1).finish('recovered');
    expect(await next).toBe('recovered');
  });
  it('rejects oversized atomic blocks and missing workers without sync conversion', async () => {
    const { renderLatexAsync, LATEX_RENDER_LIMITS } = await import('./latexRenderService');
    await expect(renderLatexAsync('x'.repeat(LATEX_RENDER_LIMITS.input + 1))).rejects.toThrow('input limit');
    vi.stubGlobal('Worker', undefined);
    await expect(renderLatexAsync('$x$')).rejects.toThrow();
  });
});
