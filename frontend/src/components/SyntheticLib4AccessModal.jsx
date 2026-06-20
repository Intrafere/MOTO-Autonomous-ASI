import React from 'react';
import HelpTooltip from './HelpTooltip';
import './settings-common.css';

export default function SyntheticLib4AccessModal({
  isOpen,
  onClose,
}) {
  if (!isOpen) return null;

  const bodyTextStyle = {
    color: 'rgba(244, 245, 255, 0.9)',
    fontSize: '1.02rem',
    fontStyle: 'normal',
    fontWeight: 400,
    lineHeight: 1.68,
    margin: '0 0 0.95rem',
  };

  const sectionStyle = {
    marginBottom: '1.15rem',
    padding: '1.25rem 1.35rem',
    borderRadius: '14px',
    background: 'rgba(255, 255, 255, 0.045)',
    border: '1px solid rgba(255, 255, 255, 0.085)',
  };

  const sectionHeadingStyle = {
    margin: '0 0 0.75rem',
    color: '#fff',
    fontSize: '1.13rem',
    fontWeight: 800,
    letterSpacing: '0.015em',
    textAlign: 'center',
  };

  const launchCardStyle = {
    maxWidth: '780px',
    margin: '1.45rem auto 0',
    padding: '1rem 1.25rem',
    borderRadius: '14px',
    textAlign: 'center',
    background: 'rgba(255, 255, 255, 0.035)',
    border: '1px solid rgba(255, 255, 255, 0.09)',
  };

  const xLinkStyle = {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: '0.65rem',
    padding: '0.58rem 1.05rem',
    borderRadius: '999px',
    color: '#f4f5ff',
    background: 'rgba(255, 255, 255, 0.075)',
    border: '1px solid rgba(255, 255, 255, 0.18)',
    fontSize: '0.95rem',
    fontWeight: 800,
    letterSpacing: '0.015em',
    textDecoration: 'none',
    boxShadow: '0 10px 24px rgba(0, 0, 0, 0.22)',
  };

  const didYouKnowDetails = (
    <>
      MOTO has approximately 100 weekly active users and the average user downloads us 4 times each to run in parallel (we can tell through the GitHub auto updater). Some users have hundreds of thousands of lines of Lean 4 code on their machines alone. Since Mathlib4 has only roughly 2 million lines of code, SyntheticLib4 could surpass it in only a few months as the largest Lean 4 code database in the world. Unlike Mathlib, which contains fundamental and known definitions, SyntheticLib4 is designed to be experimental and bleeding edge, containing useful proofs for inventors, business owners, and researchers working at the edge of human advancement.
    </>
  );

  return (
    <div className="inline-modal-overlay" style={{ zIndex: 10000 }} onClick={(event) => event.target === event.currentTarget && onClose()}>
      <div
        className="inline-modal-content"
        style={{
          width: '1173px',
          maxWidth: '96vw',
          maxHeight: '88vh',
          overflowY: 'auto',
          background: 'radial-gradient(circle at top, rgba(30, 255, 28, 0.08), transparent 34%), #1a1a2e',
          borderRadius: '16px',
          border: '1px solid rgba(30, 255, 28, 0.12)',
          padding: '2.35rem 2.6rem',
        }}
      >
        <div
          className="settings-header-row"
          style={{
            justifyContent: 'center',
            position: 'relative',
            textAlign: 'center',
            marginBottom: '1.25rem',
            padding: '0.65rem 2.5rem 1rem',
            borderBottom: '1px solid rgba(30, 255, 28, 0.18)',
          }}
        >
          <div>
            <h2
              style={{
                margin: 0,
                color: '#fff',
                fontSize: '2.05rem',
                fontWeight: 800,
                letterSpacing: '0.055em',
                textShadow: '0 0 18px rgba(30, 255, 28, 0.22)',
              }}
            >
              SyntheticLib4 <span aria-hidden="true" style={{ color: '#1eff1c', fontSize: '0.72em', verticalAlign: 'super' }}>™</span>
            </h2>
            <p style={{ margin: '0.25rem 0 0', color: '#d8d8f4', fontSize: '1rem', fontStyle: 'italic', letterSpacing: '0.045em' }}>
              The Inventor's Dictionary
            </p>
            <div
              aria-hidden="true"
              style={{
                width: '92px',
                height: '2px',
                margin: '0.75rem auto 0.7rem',
                borderRadius: '999px',
                background: 'linear-gradient(90deg, transparent, #1eff1c, transparent)',
                boxShadow: '0 0 16px rgba(30, 255, 28, 0.55)',
              }}
            />
            <div
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '0.28rem 0.85rem',
                border: '1px solid rgba(30, 255, 28, 0.38)',
                borderRadius: '999px',
                color: '#1eff1c',
                background: 'rgba(30, 255, 28, 0.08)',
                fontSize: '0.78rem',
                fontWeight: 800,
                letterSpacing: '0.12em',
                textTransform: 'uppercase',
              }}
            >
              Coming soon
            </div>
          </div>
          <button
            onClick={onClose}
            className="modal-close-btn"
            aria-label="Close SyntheticLib4 access"
            style={{ position: 'absolute', top: 0, right: 0 }}
          >
            ×
          </button>
        </div>

        <p style={{ ...bodyTextStyle, fontSize: '1.08rem', textAlign: 'center', maxWidth: '760px', margin: '0 auto 1.45rem' }}>
          SyntheticLib4 is a private, contribution-based Lean 4 proof ecosystem: users share novel verified discoveries for exclusive access to a relentlessly updated and pruned research memory, so every contribution strengthens the library that every participating MOTO can build from.
        </p>

        <div style={sectionStyle}>
          <h3 style={sectionHeadingStyle}>What It Is</h3>
          <p style={bodyTextStyle}>
            Your unused proofs now have value to provide you! We came up with a way for consenting users to trade discoveries with each other. SyntheticLib4 is a novel proof database: a super-MOTO brainstorm database of Lean 4 proofs that is relentlessly combed, vetted, updated, and pruned by state-of-the-art models. Old, weak, redundant, and newly outdated ideas are removed as stronger contributions arrive. Once filtered and processed, this information is organized into an easy-to-navigate library that your MOTO can read from autonomously while it researches your prompt! This gives users' MOTOs the ability to leapfrog off of each other's discoveries.
          </p>
          <p style={{ ...bodyTextStyle, marginBottom: 0 }}>
            Contributors receive value for their data with their consent. In exchange for contributing proof knowledge to the SyntheticLib4 ecosystem, users get shared insights from all contributors. Keeping the database private preserves that reciprocal exchange, so contributors are not asked to give without others returning value back into the ecosystem. SyntheticLib4 prevents MOTO users from retracing each other's steps. Users choose the proofs they want to contribute, and in exchange, those contributions keep the ecosystem alive.
          </p>
        </div>

        <div style={sectionStyle}>
          <h3 style={sectionHeadingStyle}>Contribution And Access</h3>
          <p style={bodyTextStyle}>
            Users will be able to select 30 novel proofs to contribute. Your local system must first deem them novel, then the SyntheticLib4 vetting system will double-check the results and decide whether they are contribution-worthy. If accepted, the contribution grants one month of access. Users can continue to contribute to extend their subscription beyond their initial month.
          </p>
          <p style={{ ...bodyTextStyle, marginBottom: 0 }}>
            Users who do not want to share proof data for access will be able to pay a small monthly fee instead. That fee supports ongoing model review, pruning, infrastructure, and ecosystem maintenance. The only data shared with our system is proof data manually selected by users; all other data remains private and local to users' computers.
          </p>
        </div>

        <div style={{ ...sectionStyle, marginBottom: 0 }}>
          <h3 style={sectionHeadingStyle}>Use And Redistribution</h3>
          <p style={bodyTextStyle}>
            Users will be able to cite, use, and reference individual proofs and groups of proofs. The ever-changing and growing Lean 4 database itself will not be redistributable as a whole, similar to how an academic journal publication can be cited and quoted but not redistributed as the entire journal archive.
          </p>
          <p style={{ ...bodyTextStyle, marginBottom: 0 }}>
            When accepted contributions update the database, all users with access will receive the improved corpus, with old and newly outdated ideas pruned as the shared library grows.
          </p>
        </div>

        <p
          style={{
            ...bodyTextStyle,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0.35rem',
            flexWrap: 'wrap',
            maxWidth: '780px',
            margin: '0.95rem auto 0.85rem',
            color: 'rgba(244, 245, 255, 0.88)',
            fontSize: '0.98rem',
            fontStyle: 'italic',
            textAlign: 'center',
          }}
        >
          <HelpTooltip
            label="Learn why SyntheticLib4 could grow quickly"
            buttonContent="Did you know MOTO has approximately 100 weekly active users?"
            buttonClassName="help-tooltip-btn--text"
            popupClassName="help-tooltip-popup--fixed"
            popupStyle={{ width: 'min(520px, calc(100vw - 96px))' }}
            useFixedPosition
          >
            {didYouKnowDetails}
          </HelpTooltip>
          <HelpTooltip
            label="Learn more about MOTO users and SyntheticLib4"
            anchorClassName="help-tooltip-anchor--inline"
            popupClassName="help-tooltip-popup--fixed"
            popupStyle={{ width: 'min(520px, calc(100vw - 96px))' }}
            useFixedPosition
          >
            {didYouKnowDetails}
          </HelpTooltip>
        </p>

        <div style={launchCardStyle}>
          <p
            style={{
              ...bodyTextStyle,
              margin: 0,
              color: 'rgba(244, 245, 255, 0.86)',
              fontSize: '1rem',
              fontWeight: 600,
              lineHeight: 1.45,
            }}
          >
            Stay tuned for the imminent SyntheticLib release...
          </p>
          <a
            href="https://x.com/IntrafereLLC"
            target="_blank"
            rel="noopener noreferrer"
            style={xLinkStyle}
          >
            Follow us on X for up-to-date information
          </a>
        </div>
      </div>
    </div>
  );
}

