import packageInfo from '../../package.json';

export const BUNDLED_VERSION = packageInfo.version;
export const MOTO_GITHUB_URL = 'https://github.com/Intrafere/MOTO-Autonomous-ASI';
export const releaseNotes = {
  '1.1.11': {
    summary: 'Introducing Master Proofs: MOTO can now plan a large theorem as a Lean 4 scaffold and prove it lemma by lemma. Plus a new moto command that lets you and AI agents run MOTO from any terminal, or hand it a hard question to think about.',
    sections: [
      {
        title: 'New: Master Proofs',
        tone: 'feature',
        graphic: 'master-proof',
        intro: 'Until now, every proof target had to fit into five Lean 4 attempts, so the largest results were cut down to whatever piece could be proved in one go. **Master proofs let MOTO go after the whole theorem.**',
        items: [
          'When a target is too large for a single attempt, proof discovery can now propose it as a **master proof**: a Lean 4 plan that states the full theorem and breaks it into smaller lemmas, each left open with a **sorry** placeholder. Lean checks that the plan fits together before any lemma is attempted.',
          'The open lemmas are then worked on **in parallel**. Each one gets 15 Lean 4 attempts and sees the feedback from its earlier attempts.',
          'A lemma that is still too hard can be **split again** into its own smaller lemmas, as many levels deep as it takes.',
          'If the plan itself turns out to be wrong, a solver can propose a new one. The **Validator** must approve the change, and the master theorem itself can never be changed or weakened.',
          'A master proof counts only when **Lean 4 accepts the complete proof with nothing left open**. If any lemma runs out of attempts, the master proof fails and its partial plan is kept on disk.',
          'Master proofs are available in **Autonomous Research** proof checks and from every **Try to prove** button, alongside normal proofs. Follow each lemma in Live Activity.',
          'Choose which proof types MOTO tries: the arrow beside **Mathematical Proofs** opens **Single-shot proofs**, **Master proofs**, and **Concurrent Proof Attempts**.',
        ],
      },
      {
        title: 'The moto command',
        items: [
          'You and AI agents such as **OpenClaw** and **Claude Code** can now run MOTO from any terminal: launch it, connect providers, start and stop research, wait for results, read papers, proofs, and final answers, and follow live activity. Open a new terminal after your next launch and type **moto --help**.',
          'Hand MOTO a hard question with **moto think**. It works on the question in depth, within a time limit you can set, while you or your agent carry on, then **moto think answer** returns the final answer or the best results so far.',
          'Start commands reuse the model settings you last chose in the app, so set up once in the window and then drive MOTO from the terminal.',
          'Every command has a **--json** mode for agents and never stops to ask a question; erasing progress needs **--yes**. Run **moto skill install** to give OpenClaw or Claude Code a ready-made skill for it.',
          'The launcher adds the command to the end of your PATH, so your existing commands keep priority. If another program already uses the name **moto**, MOTO leaves it alone and installs **moto-asi** instead.',
        ],
      },
      {
        title: 'More ways to run models',
        items: [
          '**vLLM**: add your own vLLM servers under **Inference Connectivity**. MOTO finds their models and context windows, can use their embedding models for research memory, and gives every role a **vLLM** button.',
          '**Anthropic Claude** and **Amazon Bedrock**: add a Claude API key, or a Bedrock API key with your AWS region, in the OpenRouter/OAuth window to use their models in any role.',
        ],
      },
      {
        title: 'Supercharge for everyone',
        items: [
          '**Supercharge** is now available to all users. Check it on any role in Settings: that role writes 4 full answers in parallel, then a 5th call from the same model picks or combines the best of them into one final answer. It uses 5x the API calls for that role, so it is slower and costs more, but it can give noticeably stronger answers.',
          'New silver pick under **Highlighted Models**: **Claude Haiku 5.5, Supercharged**. Low cost knowledge and a generous token allotment with a subscription make it a strong match for Supercharge. It replaces Grok 4.7.',
        ],
      },
      {
        title: 'Token count checks',
        items: [
          'MOTO now checks its estimate of each prompt\'s size against the token count your provider reports. If a model counts noticeably more, MOTO fits prompts for that model into a smaller budget so they stop overflowing its context window. A note under each role\'s **Context Window** shows the result.',
        ],
      },
      {
        title: 'Usage indicators',
        items: [
          'Settings that change how many model calls MOTO makes, such as Allowed Outputs, submitter counts, Supercharge, and the Skills And Enhancements toggles, now show a small **Low / Medium / High / Very high** meter. Hover it to see why. Levels compare MOTO features with each other; they are not exact costs.',
        ],
      },
      {
        title: 'Best free model',
        items: [
          'Our pick for the best free OpenRouter model is now **NVIDIA Nemotron 3.5 Lightning 1000K**, replacing NVIDIA Nemotron Ultra. It is free on OpenRouter with a 1 million token context window. Find it under **Highlighted Models** in Settings.',
        ],
      },
      {
        title: 'Fixes',
        items: [
          'Clearing the Single Paper Writer\'s paper, or running it in proofs-only mode, after an Autonomous Research run no longer empties your manual Aggregator database.',
          'A provider error during a brainstorm\'s completion review no longer leaves that brainstorm\'s aggregator running in the background.',
          'Ideas added to a brainstorm while its paper is being written now stay visible to the models when the brainstorm already has verified proofs.',
          'Pressing **Start** again after the manual Aggregator stopped for an error now runs exactly one validator.',
        ],
      },
    ],
    closing: 'Stay tuned for SyntheticLib\'s imminent launch!',
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
