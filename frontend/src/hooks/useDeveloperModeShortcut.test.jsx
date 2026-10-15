import { fireEvent, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import useDeveloperModeShortcut, { isEditableTarget } from './useDeveloperModeShortcut';

function pressChord(target = window) {
  fireEvent.keyDown(target, { key: 'Shift', code: 'ShiftLeft' });
  fireEvent.keyDown(target, { key: 'Z', code: 'KeyZ', shiftKey: true });
  const xEvent = fireEvent.keyDown(target, { key: 'X', code: 'KeyX', shiftKey: true });
  fireEvent.keyUp(target, { key: 'X', code: 'KeyX' });
  fireEvent.keyUp(target, { key: 'Z', code: 'KeyZ' });
  fireEvent.keyUp(target, { key: 'Shift', code: 'ShiftLeft' });
  return xEvent;
}

describe('useDeveloperModeShortcut', () => {
  afterEach(() => {
    document.body.innerHTML = '';
  });

  it('toggles once per Shift + Z + X chord outside text fields', () => {
    const onToggle = vi.fn();
    renderHook(() => useDeveloperModeShortcut(onToggle));

    const notPrevented = pressChord(document.body);
    expect(onToggle).toHaveBeenCalledTimes(1);
    expect(notPrevented).toBe(false);

    pressChord(document.body);
    expect(onToggle).toHaveBeenCalledTimes(2);
  });

  it.each([
    ['textarea', () => document.createElement('textarea')],
    ['text input', () => Object.assign(document.createElement('input'), { type: 'text' })],
    ['select', () => document.createElement('select')],
  ])('ignores the chord and keeps the keystroke while typing in a %s', (_label, make) => {
    const onToggle = vi.fn();
    renderHook(() => useDeveloperModeShortcut(onToggle));
    const field = make();
    document.body.appendChild(field);

    const notPrevented = pressChord(field);

    expect(onToggle).not.toHaveBeenCalled();
    expect(notPrevented).toBe(true);
  });

  it('still works when focus is on a checkbox or button', () => {
    const onToggle = vi.fn();
    renderHook(() => useDeveloperModeShortcut(onToggle));
    const checkbox = Object.assign(document.createElement('input'), { type: 'checkbox' });
    document.body.appendChild(checkbox);

    pressChord(checkbox);

    expect(onToggle).toHaveBeenCalledTimes(1);
  });

  it('classifies editable targets', () => {
    const editable = document.createElement('div');
    Object.defineProperty(editable, 'isContentEditable', { value: true });
    expect(isEditableTarget(editable)).toBe(true);
    expect(isEditableTarget(document.createElement('button'))).toBe(false);
    expect(isEditableTarget(null)).toBe(false);
  });
});
