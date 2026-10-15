import React from 'react';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import WhatsNewModal from './WhatsNewModal';
import { hasSeenRelease, markReleaseSeen, MOTO_GITHUB_URL } from '../utils/releaseNotes';

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.restoreAllMocks();
});

describe('release highlights', () => {
  it('remembers acknowledgement independently for each version', () => {
    expect(hasSeenRelease('1.1.08')).toBe(false);
    markReleaseSeen('1.1.08');
    expect(hasSeenRelease('1.1.08')).toBe(true);
    expect(hasSeenRelease('1.1.09')).toBe(false);
  });

  it('shows readable highlights, a safe GitHub link, and dismiss controls', () => {
    const onClose = vi.fn();
    render(<WhatsNewModal version="1.1.08" onClose={onClose} />);
    expect(screen.getByRole('dialog').getAttribute('aria-modal')).toBe('true');
    expect(screen.getByRole('heading', { name: 'What’s New with MOTO v1.1.08?' })).toBeTruthy();
    expect(screen.getByRole('link', { name: /Explore MOTO on GitHub/ }).getAttribute('href')).toBe(MOTO_GITHUB_URL);
    fireEvent.click(screen.getByRole('button', { name: 'Got it' }));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('covers user-facing pending changes and ends with the SyntheticLib announcement', () => {
    render(<WhatsNewModal version="1.1.08" onClose={() => {}} />);
    for (const name of ['Known issue: OpenAI Codex', 'Model leaderboard', 'Live Activity', 'Mathematical proofs', 'Providers, credentials, and setup']) {
      expect(screen.getByRole('heading', { name })).toBeTruthy();
    }
    expect(screen.getAllByRole('heading', { level: 3 })[0].textContent).toBe('Known issue: OpenAI Codex');
    expect(screen.getByText(/longer than 900 seconds/)).toBeTruthy();
    expect(screen.getByText(/The Codex team has been alerted/)).toBeTruthy();
    expect(screen.getByText(/GPT 6.1 Sol/)).toBeTruthy();
    expect(screen.getByText(/name the exact route/)).toBeTruthy();
    expect(screen.getByText(/standalone results/)).toBeTruthy();
    expect(screen.getByText(/direct solutions first/)).toBeTruthy();
    expect(screen.getByText(/no longer retried unchanged/)).toBeTruthy();
    expect(screen.getByText(/checked against your account/)).toBeTruthy();
    expect(screen.queryByText(/no longer cut off at about 15 minutes/)).toBeNull();
    expect(screen.queryByText(/GPT 6 Sol Maximum Thinking/)).toBeNull();
    expect(screen.queryByText(/\*\*/)).toBeNull();
    expect(screen.queryByText(/\]\(/)).toBeNull();
    expect(screen.queryByText(/deterministic workflow-test scenarios/)).toBeNull();
    expect(screen.getByRole('dialog').lastElementChild.lastElementChild.textContent)
      .toBe("Stay tuned for SyntheticLib's imminent launch!");
  });

  it('links the Codex known issue safely in a new tab', () => {
    render(<WhatsNewModal version="1.1.08" onClose={() => {}} />);
    const issueLink = screen.getByRole('link', { name: 'the Codex GitHub issue' });
    expect(issueLink.getAttribute('href')).toBe('https://github.com/openai/codex/issues/51886');
    expect(issueLink.getAttribute('target')).toBe('_blank');
    expect(issueLink.getAttribute('rel')).toBe('noopener noreferrer');
    expect(issueLink.closest('section').className).toContain('whats-new-section--warning');
  });

  it('traps keyboard focus, supports Escape, and restores focus', () => {
    const trigger = document.createElement('button');
    document.body.appendChild(trigger);
    trigger.focus();
    const onClose = vi.fn();
    const { unmount } = render(<WhatsNewModal version="1.1.08" onClose={onClose} />);
    const close = screen.getByRole('button', { name: "Close What's New" });
    expect(document.activeElement).toBe(close);
    fireEvent.keyDown(close, { key: 'Tab', shiftKey: true });
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Got it' }));
    fireEvent.keyDown(document.activeElement, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledOnce();
    unmount();
    expect(document.activeElement).toBe(trigger);
    trigger.remove();
  });

  it('isolates background controls, recovers escaped focus, and restores inert state', async () => {
    const background = document.createElement('button');
    const alreadyInert = document.createElement('div');
    alreadyInert.inert = true;
    document.body.append(background, alreadyInert);
    const { unmount } = render(<WhatsNewModal version="1.1.08" onClose={() => {}} />);
    expect(background.inert).toBe(true);
    background.focus();
    expect(document.activeElement).toBe(screen.getByRole('button', { name: "Close What's New" }));
    const portal = document.createElement('div');
    document.body.appendChild(portal);
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(portal.inert).toBe(true);
    unmount();
    expect(background.inert).toBeFalsy();
    expect(alreadyInert.inert).toBe(true);
    expect(portal.inert).toBeFalsy();
    background.remove();
    alreadyInert.remove();
    portal.remove();
  });

  it('does not show stale highlights for an unknown version', () => {
    render(<WhatsNewModal version="9.0.0" onClose={() => {}} />);
    expect(screen.queryByText('Manage more with fewer clicks')).toBeNull();
    expect(screen.queryByText("Stay tuned for SyntheticLib's imminent launch!")).toBeNull();
    expect(screen.getByText(/Visit the project on GitHub/)).toBeTruthy();
  });

  it('keeps storage failures nonfatal', () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked'); });
    expect(() => markReleaseSeen('1.1.08')).not.toThrow();
  });
});
