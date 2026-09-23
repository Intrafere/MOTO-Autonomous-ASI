import React, { useEffect, useRef } from 'react';
import { MOTO_GITHUB_URL, releaseNotes } from '../utils/releaseNotes';
import './settings-common.css';
import './WhatsNewModal.css';

function renderHighlightedText(text) {
  return text.split(/(\*\*[^*]+\*\*)/g).filter(Boolean).map((part, index) => (
    part.startsWith('**') && part.endsWith('**')
      ? <strong key={`${index}-${part}`}>{part.slice(2, -2)}</strong>
      : <React.Fragment key={`${index}-${part}`}>{part}</React.Fragment>
  ));
}

export default function WhatsNewModal({ version, onClose }) {
  const dialogRef = useRef(null);
  const closeRef = useRef(null);
  const notes = releaseNotes[version];

  useEffect(() => {
    const previousFocus = document.activeElement;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    // Isolate siblings at every ancestor level, including app portals under body.
    const isolated = new Map();
    const isolateBackground = () => {
      let branch = dialogRef.current?.parentElement;
      while (branch && branch !== document.body) {
        for (const sibling of branch.parentElement?.children || []) {
          if (sibling !== branch && !isolated.has(sibling)) {
            isolated.set(sibling, sibling.inert);
            sibling.inert = true;
          }
        }
        branch = branch.parentElement;
      }
    };
    isolateBackground();
    const observer = new MutationObserver(isolateBackground);
    observer.observe(document.body, { childList: true, subtree: true });
    const containFocus = (event) => {
      if (!dialogRef.current?.contains(event.target)) closeRef.current?.focus();
    };
    document.addEventListener('focusin', containFocus);
    closeRef.current?.focus();
    const handleKey = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onClose();
      }
      if (event.key !== 'Tab') return;
      const controls = dialogRef.current?.querySelectorAll('button, a[href]');
      if (!controls?.length) return;
      const first = controls[0];
      const last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', handleKey);
    return () => {
      observer.disconnect();
      document.removeEventListener('focusin', containFocus);
      for (const [element, wasInert] of isolated) element.inert = wasInert;
      document.body.style.overflow = previousOverflow;
      document.removeEventListener('keydown', handleKey);
      if (previousFocus?.isConnected) previousFocus.focus();
    };
  }, [onClose]);

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
            <section className="whats-new-section" key={section.title}>
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
