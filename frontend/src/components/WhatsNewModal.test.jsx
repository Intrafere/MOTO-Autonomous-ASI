import React from 'react';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import WhatsNewModal from './WhatsNewModal';
import { hasSeenRelease, markReleaseSeen, MOTO_GITHUB_URL, releaseNotes } from '../utils/releaseNotes';

const FIXTURE_VERSION = '0.0.0-fixture';

afterEach(() => {
  delete releaseNotes[FIXTURE_VERSION];
  cleanup();
  localStorage.clear();
  vi.restoreAllMocks();
});

describe('release highlights', () => {
  it('remembers acknowledgement independently for each version', () => {
    expect(hasSeenRelease('1.1.10')).toBe(false);
    markReleaseSeen('1.1.10');
    expect(hasSeenRelease('1.1.10')).toBe(true);
    expect(hasSeenRelease('1.1.11')).toBe(false);
  });

  it('shows readable highlights, a safe GitHub link, and dismiss controls', () => {
    const onClose = vi.fn();
    render(<WhatsNewModal version="1.1.11" onClose={onClose} />);
    expect(screen.getByRole('dialog').getAttribute('aria-modal')).toBe('true');
    expect(screen.getByRole('heading', { name: 'What’s New with MOTO v1.1.11?' })).toBeTruthy();
    expect(screen.getByRole('link', { name: /Explore MOTO on GitHub/ }).getAttribute('href')).toBe(MOTO_GITHUB_URL);
    fireEvent.click(screen.getByRole('button', { name: 'Got it' }));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('leads 1.1.11 with master proofs, illustrates the scaffold accessibly, then the moto command and fixes', () => {
    render(<WhatsNewModal version="1.1.11" onClose={() => {}} />);
    const headings = screen.getAllByRole('heading', { level: 3 });
    expect(headings.map((heading) => heading.textContent)).toEqual([
      'New: Master Proofs',
      'The moto command',
      'More ways to run models',
      'Supercharge for everyone',
      'Token count checks',
      'Usage indicators',
      'Best free model',
      'Fixes',
    ]);
    const headline = headings[0];
    const feature = headline.closest('section');
    expect(feature.className).toContain('whats-new-section--feature');
    // The moto command is the secondary feature: no highlight and no illustration.
    const secondary = headings[1].closest('section');
    expect(secondary.className).not.toContain('whats-new-section--feature');
    expect(within(secondary).queryByRole('figure')).toBeNull();
    expect(within(secondary).getByText('moto-asi').tagName).toBe('STRONG');
    expect(within(feature).getByText(/go after the whole theorem/).tagName).toBe('STRONG');
    const figure = within(feature).getByRole('figure', { name: /same Lean 4 proof system/ });
    expect(within(figure).getByText('Normal proof')).toBeTruthy();
    expect(within(figure).getByText('Master proof')).toBeTruthy();
    const steps = within(figure).getByRole('list', { name: 'Steps of the big theorem' });
    expect(within(steps).getAllByRole('listitem')).toHaveLength(3);
    // The same proof box appears once for the normal proof and once per step.
    expect(within(steps).getAllByText('Lean 4 proof')).toHaveLength(3);
    expect(within(figure).getAllByText('Lean 4 proof')).toHaveLength(4);
    expect(within(figure).getAllByText('✓ Verified')).toHaveLength(2);
    expect(document.getElementById('whats-new-summary').textContent).toMatch(/^Introducing Master Proofs/);
    expect(screen.getAllByRole('figure')).toHaveLength(1);
    expect(screen.queryByText(/\*\*/)).toBeNull();
    expect(screen.getByText("Stay tuned for SyntheticLib's imminent launch!")).toBeTruthy();
    expect(releaseNotes['1.1.10']).toBeUndefined();
  });

  it('renders a section intro and ignores an unknown graphic name', () => {
    releaseNotes[FIXTURE_VERSION] = {
      summary: 'Fixture summary.',
      sections: [{
        title: 'Fixture feature',
        tone: 'feature',
        graphic: 'no-such-graphic',
        intro: 'Intro with **bold** text.',
        items: ['Fixture item.'],
      }],
    };
    render(<WhatsNewModal version={FIXTURE_VERSION} onClose={() => {}} />);
    expect(screen.getByText('bold').tagName).toBe('STRONG');
    expect(screen.getByText(/Intro with/).className).toBe('whats-new-section__intro');
    expect(screen.queryByRole('figure')).toBeNull();
    expect(screen.getByText('Fixture item.')).toBeTruthy();
  });

  it('renders bold text and warning links safely without raw markup', () => {
    releaseNotes[FIXTURE_VERSION] = {
      summary: 'Fixture summary.',
      sections: [{
        title: 'Known issue: Fixture',
        tone: 'warning',
        items: ['**Bold fixture** item; follow [the fixture issue](https://example.com/issues/1) for updates.'],
      }],
    };
    render(<WhatsNewModal version={FIXTURE_VERSION} onClose={() => {}} />);
    expect(screen.getByText('Bold fixture').tagName).toBe('STRONG');
    const issueLink = screen.getByRole('link', { name: 'the fixture issue' });
    expect(issueLink.getAttribute('href')).toBe('https://example.com/issues/1');
    expect(issueLink.getAttribute('target')).toBe('_blank');
    expect(issueLink.getAttribute('rel')).toBe('noopener noreferrer');
    expect(issueLink.closest('section').className).toContain('whats-new-section--warning');
    expect(screen.queryByText(/\*\*/)).toBeNull();
    expect(screen.queryByText(/\]\(/)).toBeNull();
  });

  it('traps keyboard focus, supports Escape, and restores focus', () => {
    const trigger = document.createElement('button');
    document.body.appendChild(trigger);
    trigger.focus();
    const onClose = vi.fn();
    const { unmount } = render(<WhatsNewModal version="1.1.11" onClose={onClose} />);
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
    const { unmount } = render(<WhatsNewModal version="1.1.11" onClose={() => {}} />);
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
    expect(screen.queryByText('New: Master Proofs')).toBeNull();
    expect(screen.getByText(/Visit the project on GitHub/)).toBeTruthy();
  });

  it('keeps storage failures nonfatal', () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked'); });
    expect(() => markReleaseSeen('1.1.08')).not.toThrow();
  });
});
