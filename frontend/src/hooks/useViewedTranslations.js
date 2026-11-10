import { useCallback, useState } from 'react';
import { findViewOf, viewedDownload } from '../utils/translatedDownloads';

/**
 * Collects what a component's translation viewers show, so its downloads save the
 * language being read. Pass `reportView` to each PaperProofViewer as `onTranslationView`.
 * `downloadFor(original, filename)` returns { content, filename, language }: the shown
 * translation of `original` with a language-suffixed filename, or the original unchanged.
 */
export default function useViewedTranslations() {
  const [views, setViews] = useState({});

  // Viewers report under their own id; null means the viewer shows the original or unmounted.
  const reportView = useCallback((viewerId, view) => {
    setViews((current) => {
      const previous = current[viewerId];
      if (!view) {
        if (!previous) {
          return current;
        }
        const next = { ...current };
        delete next[viewerId];
        return next;
      }
      if (
        previous
        && previous.source === view.source
        && previous.language === view.language
        && previous.content === view.content
      ) {
        return current;
      }
      return { ...current, [viewerId]: view };
    });
  }, []);

  const downloadFor = useCallback(
    (original, filename) => viewedDownload(findViewOf(views, original), original, filename),
    [views],
  );

  return { reportView, downloadFor };
}
