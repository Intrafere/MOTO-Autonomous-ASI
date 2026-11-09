/**
 * Downloads save the language a viewer shows. A viewer showing a complete translation
 * reports it as { source, language, content }: `source` is the original text it
 * translated. A download of that same original saves the translation instead, with the
 * language key added to the filename; any other download saves the original unchanged.
 */

const UNSAFE_FILENAME_CHARS = /[^A-Za-z0-9_-]/g;

/** `filename` (without extension) with the language key appended, e.g. `paper_zh-Hans`. */
export function languageFilename(filename, language) {
  const suffix = String(language ?? '').replace(UNSAFE_FILENAME_CHARS, '');
  const name = String(filename ?? '');
  return suffix ? `${name}_${suffix}` : name;
}

function isTranslatedView(view) {
  return Boolean(view && view.language && typeof view.content === 'string');
}

/**
 * What to download for `original`: the translation `view` shows of it, or the original.
 * `filename` has no extension. Returns { content, filename, language } (language null for
 * the original).
 */
export function viewedDownload(view, original, filename) {
  if (isTranslatedView(view) && view.source === original) {
    return {
      content: view.content,
      filename: languageFilename(filename, view.language),
      language: view.language,
    };
  }
  return { content: original, filename, language: null };
}

/** PDF metadata for a download; a stored word count describes the original, not a translation. */
export function viewedPdfMetadata(metadata, download) {
  return download?.language ? { ...metadata, wordCount: null } : metadata;
}

/** The view, among those reported, that translates exactly `original`. */
export function findViewOf(views, original) {
  return Object.values(views || {}).find((view) => isTranslatedView(view) && view.source === original) || null;
}
