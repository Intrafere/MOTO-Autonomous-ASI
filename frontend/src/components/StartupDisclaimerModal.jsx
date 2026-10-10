import React from 'react';
import useOverflowAssist from '../hooks/useOverflowAssist';
import './StartupModals.css';

export default function StartupDisclaimerModal({
  isOpen,
  lmStudioEnabled,
  onAcknowledge,
}) {
  const {
    scrollRef,
    hasOverflow,
    isAtEnd,
  } = useOverflowAssist(isOpen);

  if (!isOpen) return null;

  return (
    <div
      className="startup-modal-overlay startup-disclaimer-overlay"
      onClick={(event) => event.stopPropagation()}
    >
      <div
        className="startup-modal-shell startup-disclaimer-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="startup-disclaimer-title"
      >
        <header className="startup-modal-header startup-disclaimer-header">
          <h2 id="startup-disclaimer-title">Disclaimer &amp; Quickstart</h2>
        </header>

        <div className="startup-modal-scroll-shell">
          <div
            ref={scrollRef}
            className="startup-modal-scroll-region"
          >
            <div className="startup-modal-scroll-content startup-disclaimer-content">
            <p className="startup-disclaimer-quickstart">
              {lmStudioEnabled ? (
                <>
                  <strong>QUICKSTART:</strong> In LM Studio, load the embedding model <code>nomic-ai/nomic-embed-text-v1.5</code> by <strong>Nomic AI</strong> (optional but recommended), or use OpenRouter instead of LM Studio. You must leave your PC on and awake during runtime, the program will often run for days without interruption.
                </>
              ) : (
                <>
                  <strong>QUICKSTART:</strong> This hosted deployment uses OpenRouter-only inference. Set your OpenRouter API key, choose a profile or role models, and then begin your research run. LM Studio is intentionally disabled in this environment.
                </>
              )}
            </p>

            <div className="startup-disclaimer-legal">
              <p className="startup-disclaimer-legal-title">Legal Disclaimer</p>
              <p>
                MOTO is a prototype system under active development. It directs selected AI models to generate novel solution attempts in response to your prompt. Outputs may be incorrect, incomplete, misleading, fabricated, poorly reasoned, or otherwise unsuitable for reliance without independent review, especially for high-stakes, academic, financial, legal, medical, engineering, or operational use.
                <br />
                <br />
                This software and all generated content are provided as-is and at your own risk. By using MOTO, you acknowledge that you are solely responsible for reviewing, validating, and deciding how to use any output, and that the developers, operators, and contributors are not responsible or liable for incorrect solutions, hallucinations, omissions, formatting issues, infinite loops, wasted API calls, model or provider failures, data loss, third-party charges, or any direct or indirect loss, damage, cost, or liability resulting from use of the program or its outputs.
              </p>
            </div>
            </div>
          </div>
          {hasOverflow && !isAtEnd && (
            <div className="startup-modal-overflow-fade" aria-hidden="true" />
          )}
        </div>

        {hasOverflow && !isAtEnd && (
          <div className="startup-modal-more-indicator" aria-hidden="true">
            <span>More below</span>
            <span className="startup-modal-more-arrow">↓</span>
          </div>
        )}

        <footer className="startup-modal-footer">
          <button
            className="disclaimer-acknowledge-btn"
            onClick={onAcknowledge}
          >
            I Have Read and Acknowledge This Disclaimer
          </button>
        </footer>
      </div>
    </div>
  );
}
