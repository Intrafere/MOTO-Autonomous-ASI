import { useCallback, useEffect, useRef, useState } from 'react';

const OVERFLOW_TOLERANCE = 1;
const START_DELAY_MS = 1800;
const SCROLL_RATE_PX_PER_SECOND = 14;

export default function useOverflowAssist(active = true) {
  const scrollRef = useRef(null);
  const frameRef = useRef(null);
  const delayRef = useRef(null);
  const lastFrameRef = useRef(null);
  const programmaticTopRef = useRef(null);
  const cancelledRef = useRef(false);
  const [hasOverflow, setHasOverflow] = useState(false);
  const [isAtEnd, setIsAtEnd] = useState(true);

  const cancelAssist = useCallback(() => {
    cancelledRef.current = true;
    if (delayRef.current !== null) {
      window.clearTimeout(delayRef.current);
      delayRef.current = null;
    }
    if (frameRef.current !== null) {
      window.cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }
    lastFrameRef.current = null;
  }, []);

  const measure = useCallback(() => {
    const node = scrollRef.current;
    if (!node) return;
    const overflow = node.scrollHeight > node.clientHeight + OVERFLOW_TOLERANCE;
    const atEnd = !overflow
      || node.scrollTop + node.clientHeight >= node.scrollHeight - OVERFLOW_TOLERANCE;
    setHasOverflow(overflow);
    setIsAtEnd(atEnd);
  }, []);

  useEffect(() => {
    const node = scrollRef.current;
    if (!active || !node) return undefined;

    cancelledRef.current = false;
    programmaticTopRef.current = null;
    measure();

    const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)');
    const userEvents = ['wheel', 'touchstart', 'pointerdown', 'mousedown', 'click', 'keydown', 'focusin'];
    const cancelForUser = () => cancelAssist();
    const handleScroll = () => {
      const expectedTop = programmaticTopRef.current;
      if (
        expectedTop !== null
        && Math.abs(node.scrollTop - expectedTop) > OVERFLOW_TOLERANCE
      ) {
        cancelAssist();
      }
      measure();
    };
    const handleSelection = () => {
      const selection = document.getSelection?.();
      if (!selection || selection.isCollapsed || selection.rangeCount === 0) return;
      if (node.contains(selection.getRangeAt(0).commonAncestorContainer)) {
        cancelAssist();
      }
    };
    const handleVisibility = () => {
      if (document.hidden) cancelAssist();
    };
    const handleMotionChange = (event) => {
      if (event.matches) cancelAssist();
    };

    userEvents.forEach((eventName) => {
      document.addEventListener(eventName, cancelForUser, {
        capture: true,
        passive: true,
      });
    });
    node.addEventListener('scroll', handleScroll, { passive: true });
    document.addEventListener('selectionchange', handleSelection);
    document.addEventListener('visibilitychange', handleVisibility);
    window.addEventListener('resize', measure);
    reducedMotion?.addEventListener?.('change', handleMotionChange);

    const observer = typeof ResizeObserver === 'undefined'
      ? null
      : new ResizeObserver(measure);
    observer?.observe(node);
    if (node.firstElementChild) observer?.observe(node.firstElementChild);

    const step = (timestamp) => {
      if (cancelledRef.current || document.hidden) return;
      const currentNode = scrollRef.current;
      if (!currentNode) return;
      if (lastFrameRef.current === null) lastFrameRef.current = timestamp;
      const elapsed = Math.min(timestamp - lastFrameRef.current, 100);
      lastFrameRef.current = timestamp;
      const maxTop = Math.max(0, currentNode.scrollHeight - currentNode.clientHeight);
      const nextTop = Math.min(
        maxTop,
        currentNode.scrollTop + (elapsed * SCROLL_RATE_PX_PER_SECOND) / 1000,
      );
      programmaticTopRef.current = nextTop;
      currentNode.scrollTop = nextTop;
      measure();
      if (nextTop < maxTop - OVERFLOW_TOLERANCE) {
        frameRef.current = window.requestAnimationFrame(step);
      } else {
        frameRef.current = null;
      }
    };

    delayRef.current = window.setTimeout(() => {
      delayRef.current = null;
      measure();
      const currentNode = scrollRef.current;
      const overflows = currentNode
        && currentNode.scrollHeight > currentNode.clientHeight + OVERFLOW_TOLERANCE;
      if (
        !cancelledRef.current
        && overflows
        && !reducedMotion?.matches
        && !document.hidden
      ) {
        frameRef.current = window.requestAnimationFrame(step);
      }
    }, START_DELAY_MS);

    return () => {
      cancelAssist();
      observer?.disconnect();
      userEvents.forEach((eventName) => {
        document.removeEventListener(eventName, cancelForUser, true);
      });
      node.removeEventListener('scroll', handleScroll);
      document.removeEventListener('selectionchange', handleSelection);
      document.removeEventListener('visibilitychange', handleVisibility);
      window.removeEventListener('resize', measure);
      reducedMotion?.removeEventListener?.('change', handleMotionChange);
    };
  }, [active, cancelAssist, measure]);

  return {
    scrollRef,
    hasOverflow,
    isAtEnd,
    cancelAssist,
    measure,
  };
}
