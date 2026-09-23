import packageInfo from '../../package.json';

export const BUNDLED_VERSION = packageInfo.version;
export const MOTO_GITHUB_URL = 'https://github.com/Intrafere/MOTO-Autonomous-ASI';
export const releaseNotes = {
  '1.1.06': {
    summary: 'Compare proof-solving models, control Codex reasoning, read research more easily, and keep long-running workflows on track.',
    closing: "Stay tuned for SyntheticLib's imminent launch!",
    sections: [
      {
        title: 'Let proof-solving models compete',
        items: [
          'Autonomous Rigor & Proofs now offers an optional **“Duplicate role to compete against other AI(s)”** setting. The primary model chooses what to prove; each secondary model gets its own five attempts only after the preceding model runs out of attempts, with separate feedback and overlapping work on different candidates.',
          'Compare results in API Call Logs or the new **Benchmarks tab** in Completed Works Library. Saved reports compare only problems both models actually attempted, distinguish interruptions from completed opportunities, link to the exact winning proof, and show observed Boost or fallback routes. Benchmarks survive API-log clearing.',
          'Secondary-model provider or malformed-response problems **no longer halt research**. Turning competition off ignores unfinished secondary settings, and changing the primary model preserves unchanged secondary attempt history.',
        ],
      },
      {
        title: 'More control over your models',
        items: [
          'Choose **Codex reasoning effort** in Aggregator, Compiler, Autonomous, and LeanOJ settings, including Assistant and proof-competition roles. Options follow the model’s supported levels; Auto selects the highest advertised effort, including Max when supported. Spark High stays fixed at High.',
          '**Codex Terra and Luna** now use the model you selected instead of being sent as a different model, and reasoning settings no longer force Auto or Max to XHigh regardless of model support.',
          'The model leaderboard now ranks **GPT 6 Astra first** with GPT 5.6’s merits and badges, followed by MiniMax M3 in second place.',
          'Added a **proof-strength (PS) badge to GPT OSS 120B** in Highlighted Models and renamed its tile from “OpenAI’s GPT OSS” to “GPT OSS 120b”.',
        ],
      },
      {
        title: 'Manage more with fewer clicks',
        items: [
          'Select multiple Stage 2 papers in your current library or history and **prune them together**—no need to open each paper.',
          '**Bulk-prune current proofs or restore user-pruned proofs** directly from the list. Adding a reason is now optional for user pruning; automatic pruning still requires validated reasoning.',
        ],
      },
      {
        title: 'Research that’s easier to read',
        items: [
          'Papers now open in **rendered view**. Embedded Lean proofs appear as **numbered, expandable sections** named after their theorems.',
          'Separate **paper, proof, and combined word counts** make document size clearer across live papers, history, final answers, and archives. Raw text and downloads still include everything.',
          'Find Intrafere on X and YouTube through the updated footer links.',
        ],
      },
      {
        title: 'Smoother, more reliable runs',
        items: [
          'When feedback gets too large for a model’s context window, MOTO drops the **oldest whole feedback entries** from the prompt first while preserving the newest feedback, required context, and saved history.',
          '**Force Paper Writing** now waits for brainstorming to shut down cleanly, preserving proof checks, title exploration, and resume progress. Its confirmation and Cancel buttons are also easier to use.',
          'Fatal Autonomous context overflows now reliably restore **repair notifications and stopped-run status**.',
          'Bulk cleanup is better protected against **interruptions and stale selections**. A failed display refresh no longer incorrectly reports that a completed prune was rolled back.',
          'Updated frontend dependencies address **two moderate security audit findings** and arrive through normal launcher updates.',
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
