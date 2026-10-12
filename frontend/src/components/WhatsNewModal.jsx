import React, { useRef } from 'react';
import useModalDialogFocus from '../hooks/useModalDialogFocus';
import { MOTO_GITHUB_URL, releaseNotes } from '../utils/releaseNotes';
import './settings-common.css';
import './WhatsNewModal.css';

const RELEASE_NOTE_LINK = /^\[([^\]]+)\]\((https:\/\/[^\s)]+)\)$/;

function renderHighlightedText(text) {
  return text.split(/(\*\*[^*]+\*\*|\[[^\]]+\]\(https:\/\/[^\s)]+\))/g).filter(Boolean).map((part, index) => {
    const key = `${index}-${part}`;
    const link = part.match(RELEASE_NOTE_LINK);
    if (link) {
      return <a key={key} href={link[2]} target="_blank" rel="noopener noreferrer">{link[1]}</a>;
    }
    return part.startsWith('**') && part.endsWith('**')
      ? <strong key={key}>{part.slice(2, -2)}</strong>
      : <React.Fragment key={key}>{part}</React.Fragment>;
  });
}

export default function WhatsNewModal({ version, onClose }) {
  const dialogRef = useRef(null);
  const closeRef = useRef(null);
  const notes = releaseNotes[version];

  useModalDialogFocus({ dialogRef, initialFocusRef: closeRef, onEscape: onClose });

  return (
    <div className="inline-modal-overlay whats-new-overlay" onClick={(event) => event.target === event.currentTarget && onClose()}>
      <section ref={dialogRef} className="inline-modal-content whats-new-modal" role="dialog" aria-modal="true" aria-labelledby="whats-new-title" aria-describedby="whats-new-summary">
        <header className="whats-new-header">
          <span className="whats-new-eyebrow">RELEASE HIGHLIGHTS</span>
          <button ref={closeRef} className="modal-close-btn whats-new-close" onClick={onClose} aria-label="Close What's New">×</button>
          <h2 id="whats-new-title">What’s New with MOTO v{version}?</h2>
          <p id="whats-new-summary">{notes?.summary || 'You’re running a new version of MOTO. Visit the project on GitHub for the latest changes and release details.'}</p>
        </header>
        <div className="whats-new-body">
          {notes?.sections.map((section) => (
            <section className={`whats-new-section${section.tone === 'warning' ? ' whats-new-section--warning' : ''}`} key={section.title}>
              <h3>{section.title}</h3>
              <ul>{section.items.map((item) => <li key={item}>{renderHighlightedText(item)}</li>)}</ul>
            </section>
          ))}
        </div>
        <footer className="whats-new-footer">
          <a href={MOTO_GITHUB_URL} target="_blank" rel="noopener noreferrer">Explore MOTO on GitHub ↗</a>
          <button className="whats-new-done" onClick={onClose}>Got it</button>
          <p>Reopen these highlights anytime using the MOTO version in the top-right corner.</p>
          {notes?.closing && <p>{notes.closing}</p>}
        </footer>
      </section>
    </div>
  );
}
