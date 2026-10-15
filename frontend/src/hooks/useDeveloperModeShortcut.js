import { useEffect, useRef } from 'react';

const NON_TEXT_INPUT_TYPES = new Set([
  'button', 'checkbox', 'color', 'file', 'image', 'radio', 'range', 'reset', 'submit',
]);

// Typing a capital Z then X in a text field must stay ordinary input, so the
// chord is ignored (and never swallowed) while an editable control has focus.
export function isEditableTarget(target) {
  if (!target || typeof target !== 'object') return false;
  if (target.isContentEditable) return true;
  const tag = String(target.tagName || '').toLowerCase();
  if (tag === 'textarea' || tag === 'select') return true;
  if (tag === 'input') {
    return !NON_TEXT_INPUT_TYPES.has(String(target.type || 'text').toLowerCase());
  }
  return false;
}

function shortcutCode(event) {
  if (event.code?.startsWith('Shift') || event.key === 'Shift') return 'Shift';
  if (event.code === 'KeyZ' || event.key?.toLowerCase() === 'z') return 'KeyZ';
  if (event.code === 'KeyX' || event.key?.toLowerCase() === 'x') return 'KeyX';
  return null;
}

// Shift + Z + X toggles developer mode once per chord press.
export default function useDeveloperModeShortcut(onToggle) {
  const onToggleRef = useRef(onToggle);
  onToggleRef.current = onToggle;

  useEffect(() => {
    const pressedCodes = new Set();
    let chordActive = false;

    const chordHeld = () => (
      pressedCodes.has('Shift') && pressedCodes.has('KeyZ') && pressedCodes.has('KeyX')
    );

    const handleKeyDown = (event) => {
      const code = shortcutCode(event);
      if (!code || event.repeat) return;
      if (isEditableTarget(event.target)) {
        pressedCodes.clear();
        chordActive = false;
        return;
      }
      pressedCodes.add(code);
      if (chordHeld() && !chordActive) {
        chordActive = true;
        event.preventDefault();
        onToggleRef.current?.();
      }
    };

    const handleKeyUp = (event) => {
      const code = shortcutCode(event);
      if (code) pressedCodes.delete(code);
      if (!chordHeld()) chordActive = false;
    };

    const clearPressedCodes = () => {
      pressedCodes.clear();
      chordActive = false;
    };

    window.addEventListener('keydown', handleKeyDown, true);
    window.addEventListener('keyup', handleKeyUp, true);
    window.addEventListener('blur', clearPressedCodes);
    return () => {
      window.removeEventListener('keydown', handleKeyDown, true);
      window.removeEventListener('keyup', handleKeyUp, true);
      window.removeEventListener('blur', clearPressedCodes);
    };
  }, []);
}
