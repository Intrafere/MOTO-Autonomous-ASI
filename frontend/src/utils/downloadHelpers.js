/**
 * Download helpers for papers and documents.
 *
 * PDF generation uses a Playwright (headless Chromium) backend endpoint for full
 * rendering fidelity — KaTeX math, theorem boxes, styled sections — without
 * whole-document synchronous frontend conversion. The frontend uses the same
 * pipeline as the screen renderer, then POSTs it to /api/download/pdf.
 */
import { DOMPURIFY_CONFIG } from '../components/LatexRenderer';
import { chunkLatex } from './latexChunking';
import { renderLatexAsync } from './latexRenderService';
import DOMPurify from 'dompurify';
import { prependDisclaimer } from './disclaimerHelper';

const API_BASE = import.meta.env.VITE_MOTO_API_BASE || '/api';
export const PDF_UNAVAILABLE_MESSAGE = 'PDF generation is unavailable in hosted web mode. Use raw text download instead.';

// Desktop backend defaults (config.py). Backend validation remains authoritative
// when an operator configures different server limits; no capability API is added.
export const PDF_EXPORT_LIMITS = Object.freeze({
  htmlBytes: 2 * 1024 * 1024,
  outlineBytes: 1024 * 1024,
  metadataBytes: 64 * 1024,
});
const byteSize = (text) => new Blob([text || '']).size;
const checkSize = (size, maximum, label) => {
  if (size > maximum) {
    throw new Error(`${label} exceeds the PDF export limit of ${maximum} bytes. Use raw text download for the complete document.`);
  }
};
const checkAborted = (signal) => {
  if (signal?.aborted) throw new DOMException('PDF export cancelled', 'AbortError');
};
const yieldToBrowser = () => new Promise((resolve) => setTimeout(resolve, 0));

async function renderPDFBody(body, signal, limits) {
  checkAborted(signal);
  // Refuse oversized source before chunking or creating expanded KaTeX HTML.
  checkSize(byteSize(body), limits.htmlBytes, 'Source document');
  await yieldToBrowser();
  checkAborted(signal);
  const chunks = chunkLatex(body);
  const sanitizedChunks = [];
  let htmlBytes = 0;
  let sourceOffset = 0;
  for (const chunk of chunks) {
    if (chunk === '' && body === '' && chunks.length === 1) continue;
    if (!chunk || !body.startsWith(chunk, sourceOffset)) {
      throw new Error('PDF export could not preserve the complete source document. Use raw text download instead.');
    }
    sourceOffset += chunk.length;
    checkAborted(signal);
    const rawHtml = await renderLatexAsync(chunk, { signal, priority: 'export' });
    checkAborted(signal);
    if (typeof rawHtml !== 'string' || (chunk.trim() && !rawHtml.trim())) {
      throw new Error('PDF rendering returned incomplete content. No PDF was exported.');
    }
    checkSize(byteSize(rawHtml), limits.htmlBytes, 'Rendered chunk');
    // Worker output is untrusted: sanitize on the main thread after a scheduled
    // yield, never passing worker HTML directly to the backend or a DOM sink.
    await yieldToBrowser();
    checkAborted(signal);
    const sanitizedHtml = DOMPurify.sanitize(rawHtml, DOMPURIFY_CONFIG);
    htmlBytes += byteSize(sanitizedHtml);
    checkSize(htmlBytes, limits.htmlBytes, 'Rendered document');
    sanitizedChunks.push(sanitizedHtml);
    await yieldToBrowser();
  }
  checkAborted(signal);
  if (sourceOffset !== body.length) {
    throw new Error('PDF export could not preserve the complete source document. Use raw text download instead.');
  }
  // Join only after every chunk succeeds and the accumulated UTF-8 size fits.
  return sanitizedChunks.join('');
}

export const isPDFDownloadAvailable = (capabilities = {}) => (
  capabilities?.pdfDownloadAvailable !== false
);

/**
 * Download raw text content as a .txt file.
 * @param {string} content - The text content
 * @param {string} filename - The filename (without extension)
 * @param {string|null} outline - Optional outline to prepend
 * @param {'paper'|'brainstorm'|null} disclaimerType - Prepend disclaimer if set
 */
export const downloadRawText = (content, filename, outline = null, disclaimerType = null) => {
  let fullContent = '';

  const body = disclaimerType ? prependDisclaimer(content, disclaimerType) : content;

  if (outline) {
    fullContent += 'OUTLINE\n';
    fullContent += '='.repeat(80) + '\n\n';
    fullContent += outline + '\n\n';
    fullContent += '='.repeat(80) + '\n\n';
  }

  fullContent += body;

  downloadTextFile(fullContent, `${filename}.txt`);
};

export const downloadTextFile = (content, filename, mimeType = 'text/plain') => {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

/**
 * Generate and download a PDF via the backend Playwright renderer.
 *
 * The content is rendered to HTML on the frontend (same pipeline as screen display),
 * then sent to POST /api/download/pdf where Playwright converts it to a proper PDF.
 * Backend printing runs in a thread pool; frontend chunk sanitization yields
 * between bounded results. Responsiveness still depends on device and content.
 *
 * @param {string} rawContent - Raw text content (LaTeX source)
 * @param {Object} metadata   - { title, wordCount, date, models }
 * @param {string} filename   - Filename without extension
 * @param {string|null} outline - Optional outline text to prepend
 * @param {Function|null} onStart    - Called immediately when request starts
 * @param {Function|null} onComplete - Called when PDF download begins
 * @param {Function|null} onError    - Called with Error on failure
 * @param {'paper'|'brainstorm'|null} disclaimerType - Prepend disclaimer if set
 */
export const downloadPDFViaBackend = async (
  rawContent,
  metadata,
  filename,
  outline = null,
  onStart = null,
  onComplete = null,
  onError = null,
  disclaimerType = null,
  options = {},
) => {
  if (options.pdfDownloadAvailable === false) {
    const error = new Error(PDF_UNAVAILABLE_MESSAGE);
    onError?.(error);
    throw error;
  }

  onStart?.();

  try {
    const signal = options.signal;
    checkAborted(signal);
    const limits = PDF_EXPORT_LIMITS;
    const payload = {
      title: metadata?.title || 'Document',
      word_count: metadata?.wordCount || null,
      date: metadata?.date || new Date().toLocaleDateString(),
      models: metadata?.models || null,
      outline: outline || null,
      filename: filename || 'document',
    };
    checkSize(byteSize(payload.outline), limits.outlineBytes, 'Outline');
    checkSize([payload.title, payload.date, payload.models, payload.filename]
      .reduce((sum, value) => sum + byteSize(value), 0), limits.metadataBytes, 'Metadata');
    const body = disclaimerType ? prependDisclaimer(rawContent, disclaimerType) : rawContent;
    payload.html_body = await renderPDFBody(body || '', signal, limits);
    checkAborted(signal);

    const response = await fetch(`${API_BASE}/download/pdf`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal,
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const err = await response.json();
        detail = err.detail || detail;
      } catch (_) { /* ignore */ }
      throw new Error(detail);
    }

    const blob = await response.blob();
    checkAborted(signal);
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${filename}.pdf`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    onComplete?.();
  } catch (error) {
    onError?.(error);
    throw error;
  }
};

/**
 * Sanitize a filename by removing special characters.
 * @param {string} filename
 * @returns {string}
 */
export const sanitizeFilename = (filename) => {
  return (filename || 'document')
    .replace(/[^a-z0-9_\-\s]/gi, '')
    .replace(/\s+/g, '_')
    .substring(0, 100);
};
