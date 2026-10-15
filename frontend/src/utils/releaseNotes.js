import packageInfo from '../../package.json';

export const BUNDLED_VERSION = packageInfo.version;
export const MOTO_GITHUB_URL = 'https://github.com/Intrafere/MOTO-Autonomous-ASI';
export const releaseNotes = {
  '1.1.08': {
    summary: 'A new King of the Hill, clearer Live Activity, richer proof records, and more resilient providers and credentials.',
    closing: "Stay tuned for SyntheticLib's imminent launch!",
    sections: [
      {
        title: 'Known issue: OpenAI Codex',
        tone: 'warning',
        items: [
          '**OpenAI Codex may be experiencing issues with requests that run longer than 900 seconds (15 minutes).** Long-thinking Codex calls can be dropped and retried before they finish. The Codex team has been alerted; follow [the Codex GitHub issue](https://github.com/openai/codex/issues/51886) for updates.',
        ],
      },
      {
        title: 'Model leaderboard',
        items: [
          '**GPT 6.1 Sol** (Maximum Thinking) is the new King of the Hill, with **Grok 4.7** second and **MiniMax M3** third.',
          '**NVIDIA Nemotron Ultra** remains the top free pick, with 1,000 free calls per day and a 1 million token context window on OpenRouter.',
          'The recommended models list is reorganized and now includes **Alibaba\'s Qwen** and **Qwen Coder**.',
        ],
      },
      {
        title: 'Live Activity',
        items: [
          'Provider and API errors now name the exact route: workflow, role, model, provider, and OpenRouter host.',
          'Every retry wait shows its attempt and wait time, and timed usage limits on **Codex**, **Grok**, and **Sakana Fugu** now wait for the reset and confirm when work resumes.',
          'Validator accept and reject feedback is shown in full, and provider or formatting failures are no longer reported as validator rejections.',
          'Activity entries no longer duplicate after a page reload or reconnect.',
        ],
      },
      {
        title: 'Mathematical proofs',
        items: [
          'Proof discoveries now distinguish **standalone results** from **ranked proof groups**; each group bundle downloads separately and is linked from every member proof.',
          'Each newly verified proof keeps a permanent record of exactly what Lean checked, including its source, toolchain, and imports.',
          'Continuous proof checks now finish on their own after ten consecutive rounds find no new candidates.',
          'Proofs-only Autonomous runs now complete all four proof rounds for each brainstorm instead of stopping at the first empty round.',
          'Proof discovery still aims for **direct solutions first** whenever they have a reasonable chance of being fully proved within the 5 Lean attempts. When a target is out of reach, it picks the strongest **novel piece** that can be proved, and novel indirect steps still count.',
          'Failed proof targets are **no longer retried unchanged**. Discovery sees them as open targets and decides whether to retry with a new formalization, prove a novel piece, or move on, including after each completed Autonomous paper.',
        ],
      },
      {
        title: 'Providers, credentials, and setup',
        items: [
          '**OpenAI Codex OAuth** calls now stream over the same WebSocket connection the Codex CLI uses and automatically fall back if that connection is unavailable. See the known issue above for requests longer than 15 minutes.',
          'Grok roles now offer a **reasoning effort** selector.',
          'OpenRouter keys are now checked against your account, so settings show whether a saved key actually works.',
          'Free-model looping now moves past a failing model instead of stopping, and the still-thinking notice for long model calls now appears after 25 minutes.',
          'Key fields are protected from browser autofill mistakes, and saving or removing a key can no longer leave it half-updated.',
          'Windows can now install Git automatically for Lean 4 proofs; if Git is unavailable, MOTO still starts with proofs turned off.',
          'Startup and legal dialogs now adapt cleanly to smaller screens and stay accessible when scrolling is needed.',
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
