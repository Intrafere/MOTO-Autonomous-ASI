import packageInfo from '../../package.json';

export const BUNDLED_VERSION = packageInfo.version;
export const MOTO_GITHUB_URL = 'https://github.com/Intrafere/MOTO-Autonomous-ASI';
export const releaseNotes = {
  '1.1.07': {
    summary: 'An updated model leaderboard and more reliable provider retries and Windows launches.',
    closing: "Stay tuned for SyntheticLib's imminent launch!",
    sections: [
      {
        title: 'Reliability improvements',
        items: [
          'The model leaderboard now names **GPT 6 Sol Maximum Thinking** King of the Hill, with **Grok 4.7** second and **MiniMax M3** third.',
          'Provider retry cooldowns now appear in **Live Activity**.',
          'Windows relaunches now safely recover verified orphaned backends while preserving active instances.',
          'Windows startup now has stronger data-root locking, process tracking, cleanup, and diagnostics.',
          'Fresh Windows installs now automatically repair the Microsoft runtime required by ChromaDB.',
        ],
      },
    ],
  },
};

export function releaseSeenKey(version) {
  return `moto_whats_new_seen:${version}`;
}

export function hasSeenRelease(version) {
  try {
    return localStorage.getItem(releaseSeenKey(version)) === 'true';
  } catch {
    return false;
  }
}

export function markReleaseSeen(version) {
  try {
    localStorage.setItem(releaseSeenKey(version), 'true');
  } catch {
    // Storage restrictions must never block startup or closing the dialog.
  }
}
