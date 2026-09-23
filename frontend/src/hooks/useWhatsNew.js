import { useCallback, useEffect, useState } from 'react';
import { BUNDLED_VERSION, hasSeenRelease, markReleaseSeen } from '../utils/releaseNotes';

export default function useWhatsNew({ backendVersion, showDisclaimer }) {
  const [resolvedVersion, setResolvedVersion] = useState('');
  const [whatsNewVersion, setWhatsNewVersion] = useState(null);
  const [presentedVersion, setPresentedVersion] = useState(null);
  const confirmedVersion = backendVersion?.replace(/^v/i, '') || '';
  const currentVersion = confirmedVersion || resolvedVersion || BUNDLED_VERSION;

  useEffect(() => {
    if (confirmedVersion) setResolvedVersion(confirmedVersion);
  }, [confirmedVersion]);

  useEffect(() => {
    if (showDisclaimer || !(confirmedVersion || resolvedVersion) || whatsNewVersion) return;
    if (presentedVersion === currentVersion || hasSeenRelease(currentVersion)) return;
    setPresentedVersion(currentVersion);
    setWhatsNewVersion(currentVersion);
  }, [showDisclaimer, confirmedVersion, resolvedVersion, currentVersion, whatsNewVersion, presentedVersion]);

  const closeWhatsNew = useCallback(() => {
    if (whatsNewVersion) {
      markReleaseSeen(whatsNewVersion);
      setPresentedVersion(whatsNewVersion);
    }
    setWhatsNewVersion(null);
  }, [whatsNewVersion]);

  return {
    currentVersion,
    whatsNewVersion,
    closeWhatsNew,
    openWhatsNew: () => setWhatsNewVersion(currentVersion),
    startupBlocked: showDisclaimer || Boolean(whatsNewVersion),
  };
}
