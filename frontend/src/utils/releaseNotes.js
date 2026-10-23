import packageInfo from '../../package.json';

export const BUNDLED_VERSION = packageInfo.version;
export const MOTO_GITHUB_URL = 'https://github.com/Intrafere/MOTO-Autonomous-ASI';
export const releaseNotes = {
  '1.1.09': {
    summary: 'OpenRouter streaming for long-thinking models, Kimi K3 on the leaderboard, smoother Kimi Code, clearer provider messages, and a desktop startup setup fix.',
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
          '**Kimi K3** is now third, behind King of the Hill **GPT 6.1 Sol** (Maximum Thinking) and **Grok 4.7**.',
        ],
      },
      {
        title: 'Providers and setup',
        items: [
          'On desktop, the **startup setup** chooser opens again after the disclaimer when no OpenRouter key or LM Studio setup is ready yet.',
          '**OpenRouter** calls now stream, so slow or long-thinking models such as free Nemotron Ultra no longer fail after about 5 minutes with an upstream timeout in place of the answer.',
          '**Kimi Code** requests now match Kimi\'s own client: Kimi chooses its own temperature, so roles no longer fail with an invalid temperature error, and repeated long prompts reuse Kimi\'s prompt cache to save plan quota.',
          'Tool-using roles on **Kimi Code**, including the HighSpeed model, no longer fail on follow-up turns.',
          'If **Kimi Code** rejects a request setting, MOTO now says this is a MOTO compatibility problem rather than a problem with your key or plan, and does not retry it.',
          'When an **OpenAI Codex** stream is cut early, Live Activity now explains it as a temporary fault that needs no settings change; only cuts near the 15-minute limit suggest lowering reasoning effort.',
          'When MOTO cannot reach **OpenAI Codex** at all (Wi-Fi down, DNS, VPN or proxy), Live Activity now says so and points at this computer\'s network instead of blaming OpenAI, and it no longer claims your network was healthy when a stream drops.',
          'A saved **OpenRouter** key is kept when OpenRouter cannot be reached to check it, instead of sending you back through setup.',
        ],
      },
      {
        title: 'Security',
        items: [
          'Frontend dependencies are updated to **DOMPurify 3.4.17**, **Vite 8.1.6**, and **Vitest 4.1.12** to close newly published advisories, so the launcher no longer warns about them.',
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
