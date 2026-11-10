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
    expect(hasSeenRelease('1.1.09')).toBe(false);
    markReleaseSeen('1.1.09');
    expect(hasSeenRelease('1.1.09')).toBe(true);
    expect(hasSeenRelease('1.1.10')).toBe(false);
  });

  it('shows readable highlights, a safe GitHub link, and dismiss controls', () => {
    const onClose = vi.fn();
    render(<WhatsNewModal version="1.1.10" onClose={onClose} />);
    expect(screen.getByRole('dialog').getAttribute('aria-modal')).toBe('true');
    expect(screen.getByRole('heading', { name: 'What’s New with MOTO v1.1.10?' })).toBeTruthy();
    expect(screen.getByRole('link', { name: /Explore MOTO on GitHub/ }).getAttribute('href')).toBe(MOTO_GITHUB_URL);
    fireEvent.click(screen.getByRole('button', { name: 'Got it' }));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('leads 1.1.10 with the translation feature and ends with the SyntheticLib announcement', () => {
    render(<WhatsNewModal version="1.1.10" onClose={() => {}} />);
    const headings = screen.getAllByRole('heading', { level: 3 }).map((heading) => heading.textContent);
    expect(headings).toEqual(['New: Foreign Language Translation w/ Top-p', 'Research quality', 'Reliability']);
    const feature = screen.getByRole('heading', { name: headings[0] }).closest('section');
    expect(feature.className).toContain('whats-new-section--feature');
    expect(within(feature).getByText(/flawless, not just fluent/).tagName).toBe('STRONG');
    expect(within(feature).getByText(/In academic research/)).toBeTruthy();
    expect(within(feature).getByText(/In business/)).toBeTruthy();
    expect(within(feature).getByRole('link', { name: 'Top-p Exploration' }).getAttribute('href'))
      .toBe('https://intrafere.com/structured-brainstorming-validated-feedback/');
    expect(screen.getByText('crash-safely').closest('li').textContent).toMatch(/^Papers, outlines/);
    expect(screen.getByText(/runs out of memory/)).toBeTruthy();
    expect(screen.queryByText(/is now third/)).toBeNull();
    expect(screen.queryByText(/\*\*/)).toBeNull();
    expect(screen.queryByText(/\]\(/)).toBeNull();
    expect(screen.getByRole('dialog').lastElementChild.lastElementChild.textContent)
      .toBe("Stay tuned for SyntheticLib's imminent launch!");
  });

  it('illustrates the Top-p translation loop accessibly inside the feature section', () => {
    render(<WhatsNewModal version="1.1.10" onClose={() => {}} />);
    const figure = screen.getByRole('figure', { name: /translator model and a separate validator model/ });
    expect(figure.closest('section').className).toContain('whats-new-section--feature');
    const loop = within(figure).getByRole('img', { name: /Top-p Exploration loop/ });
    const markerRefs = [...loop.querySelectorAll('[marker-end]')].map((path) => path.getAttribute('marker-end'));
    expect(markerRefs).toHaveLength(2);
    for (const ref of markerRefs) {
      const markerId = ref.match(/^url\(#([A-Za-z0-9_-]+)\)$/)[1];
      expect(loop.querySelector(`marker[id="${markerId}"]`)).not.toBeNull();
    }
    const languages = within(within(figure).getByRole('list', { name: 'Example languages' })).getAllByRole('listitem');
    expect(languages.map((item) => item.getAttribute('lang'))).toEqual(['zh', 'hi', 'es', 'ar', 'fr', 'en', null]);
    expect(within(figure).getByText('العربية').getAttribute('dir')).toBe('rtl');
    expect(within(figure).getByText(/Math and Lean stay exact/)).toBeTruthy();
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
    const { unmount } = render(<WhatsNewModal version="1.1.10" onClose={onClose} />);
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
    const { unmount } = render(<WhatsNewModal version="1.1.10" onClose={() => {}} />);
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
