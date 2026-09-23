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
    expect(hasSeenRelease('1.1.06')).toBe(false);
    markReleaseSeen('1.1.06');
    expect(hasSeenRelease('1.1.06')).toBe(true);
    expect(hasSeenRelease('1.1.07')).toBe(false);
  });

  it('shows readable highlights, a safe GitHub link, and dismiss controls', () => {
    const onClose = vi.fn();
    render(<WhatsNewModal version="1.1.06" onClose={onClose} />);
    expect(screen.getByRole('dialog').getAttribute('aria-modal')).toBe('true');
    expect(screen.getByRole('heading', { name: 'What’s New with MOTO v1.1.06?' })).toBeTruthy();
    expect(screen.getByRole('link').getAttribute('href')).toBe(MOTO_GITHUB_URL);
    fireEvent.click(screen.getByRole('button', { name: 'Got it' }));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('covers user-facing pending changes and ends with the SyntheticLib announcement', () => {
    render(<WhatsNewModal version="1.1.06" onClose={() => {}} />);
    for (const title of [
      'Let proof-solving models compete',
      'More control over your models',
      'Manage more with fewer clicks',
      'Research that’s easier to read',
      'Smoother, more reliable runs',
    ]) {
      expect(screen.getByRole('heading', { name: title })).toBeTruthy();
    }
    expect(screen.getByText(/Benchmarks survive API-log clearing/)).toBeTruthy();
    expect(screen.getByText(/Auto selects the highest advertised effort/)).toBeTruthy();
    expect(screen.getByText(/GPT 6 Astra first/)).toBeTruthy();
    expect(screen.getByText('proof-strength (PS) badge to GPT OSS 120B').tagName).toBe('STRONG');
    expect(screen.getByText(/two moderate security audit findings/)).toBeTruthy();
    expect(screen.getByText('Force Paper Writing').tagName).toBe('STRONG');
    expect(screen.queryByText(/\*\*/)).toBeNull();
    expect(screen.queryByText(/deterministic workflow-test scenarios/)).toBeNull();
    expect(screen.getByRole('dialog').lastElementChild.lastElementChild.textContent)
      .toBe("Stay tuned for SyntheticLib's imminent launch!");
  });

  it('traps keyboard focus, supports Escape, and restores focus', () => {
    const trigger = document.createElement('button');
    document.body.appendChild(trigger);
    trigger.focus();
    const onClose = vi.fn();
    const { unmount } = render(<WhatsNewModal version="1.1.06" onClose={onClose} />);
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
    const { unmount } = render(<WhatsNewModal version="1.1.06" onClose={() => {}} />);
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
    expect(() => markReleaseSeen('1.1.06')).not.toThrow();
  });
});
