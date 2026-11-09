import { vi } from 'vitest';

function readBlob(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsText(blob);
  });
}

/**
 * Records every file the page saves through an object URL and a clicked link.
 * Returns `saved()`, which resolves to [{ filename, text }] in save order. jsdom has
 * no object URLs, so they are stubbed; vi.restoreAllMocks() undoes the link spy.
 */
export function captureDownloads() {
  const files = [];
  URL.createObjectURL = vi.fn((blob) => {
    files.push({ blob, filename: null });
    return `blob:test-${files.length}`;
  });
  URL.revokeObjectURL = vi.fn();
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function recordDownload() {
    const file = files[files.length - 1];
    if (file) {
      file.filename = this.download;
    }
  });
  return async () => Promise.all(files.map(async ({ blob, filename }) => ({ filename, text: await readBlob(blob) })));
}
