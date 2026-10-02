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
    expect(hasSeenRelease('1.1.07')).toBe(false);
    markReleaseSeen('1.1.07');
    expect(hasSeenRelease('1.1.07')).toBe(true);
    expect(hasSeenRelease('1.1.08')).toBe(false);
  });

  it('shows readable highlights, a safe GitHub link, and dismiss controls', () => {
    const onClose = vi.fn();
    render(<WhatsNewModal version="1.1.07" onClose={onClose} />);
    expect(screen.getByRole('dialog').getAttribute('aria-modal')).toBe('true');
    expect(screen.getByRole('heading', { name: 'What’s New with MOTO v1.1.07?' })).toBeTruthy();
    expect(screen.getByRole('link').getAttribute('href')).toBe(MOTO_GITHUB_URL);
    fireEvent.click(screen.getByRole('button', { name: 'Got it' }));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('covers user-facing pending changes and ends with the SyntheticLib announcement', () => {
    render(<WhatsNewModal version="1.1.07" onClose={() => {}} />);
    expect(screen.getByRole('heading', { name: 'Reliability improvements' })).toBeTruthy();
    expect(screen.getByText(/Provider retry cooldowns now appear/)).toBeTruthy();
    expect(screen.getByText(/safely recover verified orphaned backends/)).toBeTruthy();
    expect(screen.getByText(/stronger data-root locking/)).toBeTruthy();
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
    const { unmount } = render(<WhatsNewModal version="1.1.07" onClose={onClose} />);
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
    const { unmount } = render(<WhatsNewModal version="1.1.07" onClose={() => {}} />);
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
    expect(() => markReleaseSeen('1.1.07')).not.toThrow();
  });
});
