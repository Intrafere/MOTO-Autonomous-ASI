import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import { chunkLatex } from './latexChunking';
import { renderLatexAsync } from './latexRenderService';
import { downloadPDFViaBackend, downloadRawText, PDF_EXPORT_LIMITS } from './downloadHelpers';

vi.mock('../components/LatexRenderer', () => ({ DOMPURIFY_CONFIG: { FORBID_TAGS: ['script'] } }));
vi.mock('./latexChunking', () => ({ chunkLatex: vi.fn() }));
vi.mock('./latexRenderService', () => ({ renderLatexAsync: vi.fn() }));

beforeEach(() => {
  vi.clearAllMocks();
  chunkLatex.mockImplementation((text) => [text]);
  renderLatexAsync.mockImplementation(async (text) => `<p>${text}</p>`);
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, blob: async () => new Blob(['pdf']) }));
  vi.stubGlobal('URL', { createObjectURL: vi.fn(() => 'blob:test'), revokeObjectURL: vi.fn() });
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
});
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('asynchronous complete PDF export', () => {
  test('renders all ordered chunks, sanitizes each, and preserves payload and callbacks', async () => {
    chunkLatex.mockReturnValue(['first\n\n', 'last']);
    renderLatexAsync.mockImplementation(async (text) => `<p onclick="bad()">${text}</p><script>bad()</script>`);
    const start = vi.fn(); const complete = vi.fn(); const error = vi.fn();
    await downloadPDFViaBackend('first\n\nlast', { title: 'Title', wordCount: 42, date: 'Today', models: 'Model' }, 'file', 'Outline', start, complete, error);
    expect(renderLatexAsync.mock.calls.map(([text]) => text)).toEqual(['first\n\n', 'last']);
    expect(renderLatexAsync.mock.calls[0][1].priority).toBe('export');
    const payload = JSON.parse(fetch.mock.calls[0][1].body);
    expect(payload).toEqual({ title: 'Title', word_count: 42, date: 'Today', models: 'Model', filename: 'file', outline: 'Outline', html_body: '<p>first\n\n</p><p>last</p>' });
    expect(start).toHaveBeenCalledOnce(); expect(complete).toHaveBeenCalledOnce(); expect(error).not.toHaveBeenCalled();
  });
  test('adds the disclaimer before chunking', async () => {
    await downloadPDFViaBackend('Body', {}, 'file', null, null, null, null, 'paper');
    expect(chunkLatex.mock.calls[0][0]).toContain('DISCLAIMER');
    expect(chunkLatex.mock.calls[0][0]).toContain('Body');
  });
  test('never exports a partial document after a worker failure', async () => {
    chunkLatex.mockReturnValue(['one', 'two']);
    renderLatexAsync.mockResolvedValueOnce('<p>one</p>').mockRejectedValueOnce(new Error('Worker failed'));
    const error = vi.fn(); const complete = vi.fn();
    await expect(downloadPDFViaBackend('onetwo', {}, 'file', null, null, complete, error)).rejects.toThrow('Worker failed');
    expect(fetch).not.toHaveBeenCalled(); expect(URL.createObjectURL).not.toHaveBeenCalled();
    expect(complete).not.toHaveBeenCalled(); expect(error).toHaveBeenCalledOnce();
  });
  test('refuses source loss in chunking', async () => {
    chunkLatex.mockReturnValue(['first']);
    await expect(downloadPDFViaBackend('first last', {}, 'file')).rejects.toThrow('complete source');
    expect(fetch).not.toHaveBeenCalled();
  });
  test('rejects huge source, outline and metadata before rendering', async () => {
    await expect(downloadPDFViaBackend('x'.repeat(PDF_EXPORT_LIMITS.htmlBytes + 1), {}, 'file')).rejects.toThrow('Source document');
    await expect(downloadPDFViaBackend('body', {}, 'file', 'x'.repeat(PDF_EXPORT_LIMITS.outlineBytes + 1))).rejects.toThrow('Outline');
    await expect(downloadPDFViaBackend('body', { title: 'x'.repeat(PDF_EXPORT_LIMITS.metadataBytes + 1) }, 'file')).rejects.toThrow('Metadata');
    expect(renderLatexAsync).not.toHaveBeenCalled(); expect(fetch).not.toHaveBeenCalled();
  });
  test('enforces cumulative HTML bytes before posting', async () => {
    chunkLatex.mockReturnValue(['one', 'two']);
    renderLatexAsync.mockResolvedValue(`<p>${'x'.repeat(PDF_EXPORT_LIMITS.htmlBytes / 2)}</p>`);
    await expect(downloadPDFViaBackend('onetwo', {}, 'file')).rejects.toThrow('Rendered document');
    expect(fetch).not.toHaveBeenCalled();
  });
  test('preserves hosted unavailability and backend errors', async () => {
    const error = vi.fn(); const start = vi.fn();
    await expect(downloadPDFViaBackend('body', {}, 'file', null, start, null, error, null, { pdfDownloadAvailable: false })).rejects.toThrow('hosted web mode');
    expect(start).not.toHaveBeenCalled(); expect(error).toHaveBeenCalledOnce(); expect(fetch).not.toHaveBeenCalled();
    fetch.mockResolvedValue({ ok: false, status: 413, json: async () => ({ detail: 'Server size limit' }) });
    await expect(downloadPDFViaBackend('body', {}, 'file')).rejects.toThrow('Server size limit');
  });
  test('cancels without downloading or synchronously rendering', async () => {
    const controller = new AbortController();
    renderLatexAsync.mockImplementation(async () => { controller.abort(); return '<p>body</p>'; });
    await expect(downloadPDFViaBackend('body', {}, 'file', null, null, null, null, null, { signal: controller.signal })).rejects.toMatchObject({ name: 'AbortError' });
    expect(fetch).not.toHaveBeenCalled(); expect(URL.createObjectURL).not.toHaveBeenCalled();
  });
  test('raw export remains full fidelity with outline and disclaimer', async () => {
    const content = `Body ${'full content '.repeat(1000)} THE END`;
    downloadRawText(content, 'file', 'Outline', 'brainstorm');
    const blob = URL.createObjectURL.mock.calls[0][0];
    const text = await new Promise((resolve) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.readAsText(blob); });
    expect(text).toContain('Outline'); expect(text).toContain('DISCLAIMER'); expect(text).toContain(content);
    expect(renderLatexAsync).not.toHaveBeenCalled();
  });
});
