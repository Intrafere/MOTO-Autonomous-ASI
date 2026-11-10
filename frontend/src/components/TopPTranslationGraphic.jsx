import React, { useId } from 'react';
import './TopPTranslationGraphic.css';

// Release-highlights illustration of Foreign Language Translation w/ Top-p: an original
// document goes part by part through a translator/validator loop into the chosen languages.
const LANGUAGES = [
  { lang: 'zh', label: '中文', name: 'Mandarin Chinese' },
  { lang: 'hi', label: 'हिन्दी', name: 'Hindi' },
  { lang: 'es', label: 'Español', name: 'Spanish' },
  { lang: 'ar', label: 'العربية', name: 'Arabic', dir: 'rtl' },
  { lang: 'fr', label: 'Français', name: 'French' },
  { lang: 'en', label: 'English', name: 'English' },
];

function FlowArrow() {
  return (
    <span className="tpt-arrow" aria-hidden="true">
      <svg viewBox="0 0 48 24" focusable="false">
        <path d="M2 12h38" />
        <path d="M32 4l10 8-10 8" />
      </svg>
    </span>
  );
}

export default function TopPTranslationGraphic() {
  const id = useId();
  const captionId = `${id}-caption`;
  const loopTitleId = `${id}-loop-title`;
  // url(#...) references need plain characters; React's generated ids may not be.
  const markerPrefix = `tpt${id.replace(/[^a-zA-Z0-9_-]/g, '')}`;
  const attemptMarker = `${markerPrefix}-attempt`;
  const feedbackMarker = `${markerPrefix}-feedback`;

  return (
    <figure className="tpt-graphic" aria-labelledby={captionId}>
      <div className="tpt-flow">
        <div className="tpt-card tpt-card--source">
          <span className="tpt-card__label">Original</span>
          <div className="tpt-doc" aria-hidden="true">
            <span className="tpt-doc__title" />
            <span className="tpt-doc__line" />
            <span className="tpt-doc__line tpt-doc__line--part" />
            <span className="tpt-doc__line tpt-doc__line--part tpt-doc__line--short" />
            <span className="tpt-doc__math">∫₀¹ f(x) dx = π</span>
            <span className="tpt-doc__line" />
            <span className="tpt-doc__line tpt-doc__line--short" />
          </div>
          <p className="tpt-card__note">
            Papers, proofs, and brainstorms, one part at a time. <span className="tpt-exact">Math and Lean stay exact.</span>
          </p>
        </div>

        <FlowArrow />

        <div className="tpt-card tpt-card--loop">
          <svg className="tpt-loop" viewBox="22 0 316 224" role="img" aria-labelledby={loopTitleId} focusable="false">
            <title id={loopTitleId}>
              Top-p Exploration loop: the translator sends an attempt, the validator checks it and sends feedback
            </title>
            <defs>
              <marker id={attemptMarker} viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                <path d="M0 0L10 5L0 10z" className="tpt-loop__head tpt-loop__head--attempt" />
              </marker>
              <marker id={feedbackMarker} viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                <path d="M0 0L10 5L0 10z" className="tpt-loop__head tpt-loop__head--feedback" />
              </marker>
            </defs>

            <path className="tpt-loop__arc tpt-loop__arc--attempt" d="M118 72C160 20 200 20 242 72" markerEnd={`url(#${attemptMarker})`} />
            <path className="tpt-loop__flow tpt-loop__flow--attempt" d="M118 72C160 20 200 20 242 72" />
            <text className="tpt-loop__label tpt-loop__label--attempt" x="180" y="22">attempt</text>

            <path className="tpt-loop__arc tpt-loop__arc--feedback" d="M242 150C200 202 160 202 118 150" markerEnd={`url(#${feedbackMarker})`} />
            <path className="tpt-loop__flow tpt-loop__flow--feedback" d="M242 150C200 202 160 202 118 150" />
            <text className="tpt-loop__label tpt-loop__label--feedback" x="180" y="218">feedback</text>

            <text className="tpt-loop__center" x="180" y="108">Top-p</text>
            <text className="tpt-loop__center tpt-loop__center--sub" x="180" y="128">Exploration</text>

            <g className="tpt-node tpt-node--translator">
              <circle cx="76" cy="111" r="50" />
              <text className="tpt-node__glyph" x="76" y="111">文A</text>
              <text className="tpt-node__name" x="76" y="138">Translator</text>
            </g>
            <g className="tpt-node tpt-node--validator">
              <circle cx="284" cy="111" r="50" />
              <path className="tpt-node__check" d="M267 104l11 11 21-24" />
              <text className="tpt-node__name" x="284" y="138">Validator</text>
            </g>
          </svg>
          <div className="tpt-accepts" aria-hidden="true">
            <span className="tpt-accepts__check">✓</span>
            <span className="tpt-accepts__check">✓</span>
            <span className="tpt-accepts__check">✓</span>
            <span className="tpt-accepts__text">accepted again and again, then final</span>
          </div>
        </div>

        <FlowArrow />

        <div className="tpt-card tpt-card--targets">
          <span className="tpt-card__label">Your languages</span>
          <ul className="tpt-langs" aria-label="Example languages">
            {LANGUAGES.map((language) => (
              <li key={language.lang} lang={language.lang} dir={language.dir} title={language.name}>
                {language.label}
              </li>
            ))}
            <li className="tpt-langs__more">+ many more</li>
          </ul>
          <p className="tpt-card__note">Switch each item between the original and every language you pick.</p>
        </div>
      </div>
      <figcaption id={captionId} className="tpt-caption">
        Each part of a document goes back and forth between a translator model and a separate validator model,
        and becomes final only after the validator has accepted it again and again.
      </figcaption>
    </figure>
  );
}
