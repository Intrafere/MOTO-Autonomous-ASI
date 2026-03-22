/**
 * Download helpers for papers and documents.
 *
 * PDF generation uses a Playwright (headless Chromium) backend endpoint for full
 * rendering fidelity — KaTeX math, theorem boxes, styled sections — without
 * freezing the browser UI. The frontend renders the content to HTML using the same
 * pipeline as the screen renderer, then POSTs it to /api/download/pdf.
 */
import { renderLatexToHtml, DOMPURIFY_CONFIG } from '../components/LatexRenderer';
import DOMPurify from 'dompurify';

/**
 * Download raw text content as a .txt file.
 * @param {string} content - The text content
 * @param {string} filename - The filename (without extension)
 * @param {string|null} outline - Optional outline to prepend
 */
export const downloadRawText = (content, filename, outline = null) => {
  let fullContent = '';

  if (outline) {
    fullContent += 'OUTLINE\n';
    fullContent += '='.repeat(80) + '\n\n';
    fullContent += outline + '\n\n';
    fullContent += '='.repeat(80) + '\n\n';
  }

  fullContent += content;

  const blob = new Blob([fullContent], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${filename}.txt`;
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
 * This runs in a backend thread pool so the UI stays fully responsive.
 *
 * @param {string} rawContent - Raw text content (LaTeX source)
 * @param {Object} metadata   - { title, wordCount, date, models }
 * @param {string} filename   - Filename without extension
 * @param {string|null} outline - Optional outline text to prepend
 * @param {Function|null} onStart    - Called immediately when request starts
 * @param {Function|null} onComplete - Called when PDF download begins
 * @param {Function|null} onError    - Called with Error on failure
 */
export const downloadPDFViaBackend = async (
  rawContent,
  metadata,
  filename,
  outline = null,
  onStart = null,
  onComplete = null,
  onError = null,
) => {
  onStart?.();

  try {
    // Render LaTeX → HTML using the same pipeline as the screen renderer
    const rawHtml = renderLatexToHtml(rawContent);
    const sanitizedHtml = DOMPurify.sanitize(rawHtml, DOMPURIFY_CONFIG);

    const response = await fetch('/api/download/pdf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        html_body: sanitizedHtml,
        title: metadata.title || 'Document',
        word_count: metadata.wordCount || null,
        date: metadata.date || new Date().toLocaleDateString(),
        models: metadata.models || null,
        outline: outline || null,
        filename: filename || 'document',
      }),
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
