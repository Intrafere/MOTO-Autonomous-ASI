import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

export default function HelpTooltip({
  label,
  children,
  anchorClassName = '',
  popupClassName = '',
  buttonClassName = '',
  popupStyle,
  buttonContent = '?',
  useFixedPosition = false,
  fixedPlacement = 'above-right',
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [fixedPopupStyle, setFixedPopupStyle] = useState(null);
  const buttonRef = useRef(null);
  const popupRef = useRef(null);

  const anchorClasses = ['help-tooltip-anchor', anchorClassName].filter(Boolean).join(' ');
  const buttonClasses = ['help-tooltip-btn', buttonClassName].filter(Boolean).join(' ');
  const popupClasses = ['help-tooltip-popup', popupClassName].filter(Boolean).join(' ');

  const updateFixedPosition = useCallback(() => {
    if (!useFixedPosition || !buttonRef.current || !popupRef.current || typeof window === 'undefined') {
      return;
    }

    const buttonRect = buttonRef.current.getBoundingClientRect();
    const popupRect = popupRef.current.getBoundingClientRect();
    const gap = 10;
    const viewportPadding = 16;

    let left = buttonRect.right + gap;
    let top = buttonRect.top - popupRect.height - gap;

    if (fixedPlacement === 'side-right') {
      if (left + popupRect.width > window.innerWidth - viewportPadding) {
        left = buttonRect.left - popupRect.width - gap;
      }
      left = Math.max(viewportPadding, left);
      top = buttonRect.top + (buttonRect.height - popupRect.height) / 2;
    } else if (left + popupRect.width > window.innerWidth - viewportPadding) {
      left = Math.max(viewportPadding, window.innerWidth - popupRect.width - viewportPadding);
    }

    top = Math.min(
      Math.max(viewportPadding, top),
      Math.max(viewportPadding, window.innerHeight - popupRect.height - viewportPadding)
    );

    setFixedPopupStyle({
      position: 'fixed',
      top: `${top}px`,
      left: `${left}px`,
      right: 'auto',
      bottom: 'auto',
      zIndex: 100000,
    });
  }, [fixedPlacement, useFixedPosition]);

  const showTooltip = () => {
    setIsOpen(true);
  };

  const hideTooltip = () => {
    setIsOpen(false);
    setFixedPopupStyle(null);
  };

  useLayoutEffect(() => {
    if (isOpen) {
      updateFixedPosition();
    }
  }, [isOpen, updateFixedPosition]);

  useEffect(() => {
    if (!isOpen || !useFixedPosition || typeof window === 'undefined') {
      return undefined;
    }

    const handleViewportChange = () => updateFixedPosition();
    window.addEventListener('resize', handleViewportChange);
    window.addEventListener('scroll', handleViewportChange, true);

    return () => {
      window.removeEventListener('resize', handleViewportChange);
      window.removeEventListener('scroll', handleViewportChange, true);
    };
  }, [isOpen, updateFixedPosition, useFixedPosition]);

  const handleClick = (event) => {
    // Prevent nested labels from toggling their checkbox when the help icon is clicked.
    event.preventDefault();
    event.stopPropagation();
  };

  const tooltipPopup = isOpen ? (
    <span
      ref={popupRef}
      className={popupClasses}
      style={
        useFixedPosition
          ? {
              position: 'fixed',
              top: 0,
              left: 0,
              visibility: fixedPopupStyle ? 'visible' : 'hidden',
              zIndex: 100000,
              ...fixedPopupStyle,
              ...popupStyle,
            }
          : popupStyle
      }
    >
      {children}
    </span>
  ) : null;

  return (
    <span className={anchorClasses}>
      <button
        ref={buttonRef}
        type="button"
        className={buttonClasses}
        aria-label={label}
        onMouseEnter={showTooltip}
        onMouseLeave={hideTooltip}
        onFocus={showTooltip}
        onBlur={hideTooltip}
        onClick={handleClick}
      >
        {buttonContent}
      </button>
      {!useFixedPosition && tooltipPopup}
      {useFixedPosition && tooltipPopup && typeof document !== 'undefined'
        ? createPortal(tooltipPopup, document.body)
        : null}
    </span>
  );
}
