import { describe, expect, test } from 'vitest';
import { findViewOf, languageFilename, viewedDownload, viewedPdfMetadata } from './translatedDownloads';

const VIEW = { source: 'Original body.', language: 'zh-Hans', content: '原始正文。' };

describe('languageFilename', () => {
  test('appends a filename-safe language key', () => {
    expect(languageFilename('paper_one', 'zh-Hans')).toBe('paper_one_zh-Hans');
    expect(languageFilename('paper_one', 'custom-1a2b/../x')).toBe('paper_one_custom-1a2bx');
  });

  test('leaves the filename alone without a usable language key', () => {
    expect(languageFilename('paper_one', null)).toBe('paper_one');
    expect(languageFilename('paper_one', '../')).toBe('paper_one');
  });
});

describe('viewedDownload', () => {
  test('saves the shown translation of the same original with a language-suffixed filename', () => {
    expect(viewedDownload(VIEW, 'Original body.', 'paper')).toEqual({
      content: '原始正文。',
      filename: 'paper_zh-Hans',
      language: 'zh-Hans',
    });
  });

  test('saves the original when nothing is translated or the view shows another document', () => {
    const original = { content: 'Original body.', filename: 'paper', language: null };
    expect(viewedDownload(null, 'Original body.', 'paper')).toEqual(original);
    expect(viewedDownload({ ...VIEW, content: null }, 'Original body.', 'paper')).toEqual(original);
    expect(viewedDownload({ ...VIEW, language: null }, 'Original body.', 'paper')).toEqual(original);
    expect(viewedDownload(VIEW, 'A newer original body.', 'paper')).toEqual({
      content: 'A newer original body.',
      filename: 'paper',
      language: null,
    });
  });
});

describe('findViewOf', () => {
  test('finds the view that translates exactly the given original', () => {
    const other = { source: 'Other body.', language: 'zh-Hans', content: '其他正文。' };
    expect(findViewOf({ a: other, b: VIEW }, 'Original body.')).toBe(VIEW);
    expect(findViewOf({ a: other }, 'Original body.')).toBeNull();
    expect(findViewOf(undefined, 'Original body.')).toBeNull();
  });
});

describe('viewedPdfMetadata', () => {
  test('drops the original word count from a translated PDF only', () => {
    const metadata = { title: 'Paper', wordCount: 1200, date: 'Today', models: null };
    expect(viewedPdfMetadata(metadata, { language: 'zh-Hans' })).toEqual({ ...metadata, wordCount: null });
    expect(viewedPdfMetadata(metadata, { language: null })).toBe(metadata);
  });
});
