import React, { useRef } from 'react';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, test, vi } from 'vitest';
import useModalDialogFocus from './useModalDialogFocus';

function Dialog({ onEscape, withHiddenContent = true }) {
  const dialogRef = useRef(null);
  const closeRef = useRef(null);
  useModalDialogFocus({ dialogRef, initialFocusRef: closeRef, onEscape });
  return (
    <div ref={dialogRef} role="dialog" aria-modal="true" aria-label="Test dialog">
      <button ref={closeRef} type="button">Close</button>
      <input aria-label="Search" />
      <button type="button">Last visible</button>
      {withHiddenContent && (
        // An inactive tab panel: neither it nor its controls can take focus.
        <div role="tabpanel" aria-label="Hidden panel" hidden tabIndex={0}>
          <button type="button">Hidden action</button>
        </div>
      )}
    </div>
  );
}

afterEach(() => {
  cleanup();
});

describe('useModalDialogFocus', () => {
  test('Tab and Shift+Tab wrap between the first and last visible controls, skipping hidden content', async () => {
    const user = userEvent.setup();
    render(<Dialog onEscape={vi.fn()} />);
    const close = screen.getByRole('button', { name: 'Close' });
    const last = screen.getByRole('button', { name: 'Last visible' });
    expect(close).toHaveFocus();

    await user.tab({ shift: true });
    expect(last).toHaveFocus();

    await user.tab();
    expect(close).toHaveFocus();
  });

  test('wraps without hidden content and routes Escape', async () => {
    const user = userEvent.setup();
    const onEscape = vi.fn();
    render(<Dialog onEscape={onEscape} withHiddenContent={false} />);
    const close = screen.getByRole('button', { name: 'Close' });
    const last = screen.getByRole('button', { name: 'Last visible' });

    await user.tab({ shift: true });
    expect(last).toHaveFocus();
    await user.tab();
    expect(close).toHaveFocus();
    await user.tab();
    expect(screen.getByRole('textbox', { name: 'Search' })).toHaveFocus();

    await user.keyboard('{Escape}');
    expect(onEscape).toHaveBeenCalledTimes(1);
  });
});
