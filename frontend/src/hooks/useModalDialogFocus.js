import { useEffect, useRef } from 'react';

const FOCUSABLE = [
  'button:not([disabled])',
  'a[href]',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(', ');

// Modal dialog focus contract: move focus in on open, keep it (and assistive
// technology) inside by making the background inert, wrap Tab, route Escape to
// `onEscape`, and restore the previous focus and scroll state on close.
export default function useModalDialogFocus({ active = true, dialogRef, initialFocusRef, onEscape }) {
  const onEscapeRef = useRef(onEscape);
  onEscapeRef.current = onEscape;

  useEffect(() => {
    if (!active) return undefined;
    const previousFocus = document.activeElement;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    // Isolate siblings at every ancestor level, including app portals under body.
    const isolated = new Map();
    const isolateBackground = () => {
      let branch = dialogRef.current?.parentElement;
      while (branch && branch !== document.body) {
        for (const sibling of branch.parentElement?.children || []) {
          if (sibling !== branch && !isolated.has(sibling)) {
            isolated.set(sibling, sibling.inert);
            sibling.inert = true;
          }
        }
        branch = branch.parentElement;
      }
    };
    isolateBackground();
    const observer = new MutationObserver(isolateBackground);
    observer.observe(document.body, { childList: true, subtree: true });
    const containFocus = (event) => {
      if (!dialogRef.current?.contains(event.target)) initialFocusRef.current?.focus();
    };
    document.addEventListener('focusin', containFocus);
    initialFocusRef.current?.focus();
    const handleKey = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onEscapeRef.current?.();
      }
      if (event.key !== 'Tab') return;
      const controls = dialogRef.current?.querySelectorAll(FOCUSABLE);
      if (!controls?.length) return;
      const first = controls[0];
      const last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', handleKey);
    return () => {
      observer.disconnect();
      document.removeEventListener('focusin', containFocus);
      for (const [element, wasInert] of isolated) element.inert = wasInert;
      document.body.style.overflow = previousOverflow;
      document.removeEventListener('keydown', handleKey);
      if (previousFocus?.isConnected) previousFocus.focus();
    };
  }, [active, dialogRef, initialFocusRef]);
}
