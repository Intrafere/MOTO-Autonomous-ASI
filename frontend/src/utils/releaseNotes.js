import packageInfo from '../../package.json';

export const BUNDLED_VERSION = packageInfo.version;
export const MOTO_GITHUB_URL = 'https://github.com/Intrafere/MOTO-Autonomous-ASI';
export const releaseNotes = {
  '1.1.10': {
    summary: 'Introducing Foreign Language Translation w/ Top-p: MOTO now translates your papers, proofs, and brainstorms with the same validated Top-p Exploration behind its research. Plus crash-safe saves, clearer Lean memory messages, and steadier long runs.',
    closing: "Stay tuned for SyntheticLib's imminent launch!",
    sections: [
      {
        title: 'New: Foreign Language Translation w/ Top-p',
        tone: 'feature',
        graphic: 'top-p-translation',
        intro: 'Research and business move only as fast as ideas cross language barriers, and there translation has to be **flawless, not just fluent**. In academic research, one mistranslated definition, quantifier, or assumption can turn a correct result into a wrong one. In business, one slipped figure or clause in a contract, report, or specification can cost a deal or a customer\'s trust. That is why MOTO\'s new translator never takes a single model\'s word for it.',
        items: [
          '**Foreign Language Translation w/ Top-p** translates your papers, proofs, final answers, and brainstorms into the languages you choose, in the background while your research keeps running.',
          'It uses [Top-p Exploration](https://intrafere.com/structured-brainstorming-validated-feedback/), the structured brainstorming with validated feedback that powers MOTO\'s research. A translator model drafts each part, a separate validator model checks every attempt and explains exactly what to fix, and a part becomes final only after the validator has accepted three attempts, or the translator twice confirms an accepted attempt cannot be improved.',
          'The translator can go back and correct earlier parts, and can add notes that explain its choices or dispute feedback, so the most accurate reading wins on its merits.',
          '**Math, code, and Lean proofs stay exactly as written.** Numbers and formulas are never localized, and the original is always one click away.',
          'Choose from English, Mandarin Chinese, Hindi, Spanish, Modern Standard Arabic, French, and many more. MOTO asks your models which languages they handle well and warns you about any they may not translate reliably.',
          'Every paper, proof, brainstorm, and results view has its own **Language** switch, and downloads save the language you are viewing.',
          'Turn it on under **Skills And Enhancements** in the top-right menu, then pick your languages and your translator and validator models. Your current runs translate automatically, **Translate all older items** catches up everything else, and **Batch translation** runs several documents at once when your models support it.',
          'Translation runs beside any workflow without blocking or stopping it, and its model calls stay out of your research token totals and paper model credits.',
        ],
      },
      {
        title: 'Research quality',
        items: [
          'Brainstorm topics stay focused on your goal but open about method: validators now reject topics that assume their own answer or lock a brainstorm into one predetermined solution path.',
          'If the validator garbles its reply for a batch of submissions, MOTO now checks them again one at a time instead of losing them.',
          'Knowledge search (RAG) never mixes two embedding models in one index. If LM Studio\'s embedding model is briefly unavailable, MOTO retries it instead of switching models partway through a run.',
          '**Supercharge** stops its other attempts as soon as one fails, so no provider budget is spent on an answer that would be thrown away.',
        ],
      },
      {
        title: 'Reliability',
        items: [
          'Papers, outlines, brainstorm databases, accepted submissions, and run state are now saved **crash-safely**, so a crash or power cut during a save can no longer leave them cut short.',
          'Retrieval reranking now runs in the background and much faster, so the app stays responsive during big runs and long provider streams, such as **OpenAI Codex**, are no longer dropped by stalls.',
          'When **Lean 4** runs out of memory (RAM), MOTO now says so plainly, shows how much memory is free, and suggests what to close, instead of reporting a broken Mathlib setup.',
          'On **Windows**, the display now stays on during runs, because on Modern Standby PCs an idle screen-off starts standby and could pause a long run. Your monitor\'s own power button still works.',
          '**Stop** now cancels a forced Tier 3 final answer, and Start can no longer run on top of one.',
          'Live updates keep retrying after a failed reconnect attempt instead of going quiet until you reload.',
          'A prompt you type while MOTO is still loading is no longer replaced by your saved prompt.',
          '**Boost** now refuses a context window or output limit of zero, instead of turning on while every boosted call fails.',
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
