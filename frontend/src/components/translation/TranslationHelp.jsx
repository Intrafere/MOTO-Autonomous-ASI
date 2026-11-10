import React from 'react';
import HelpTooltip from '../HelpTooltip';

// A "?" help button in the app's tooltip style, plus the same text as a hidden
// description. The tooltip popup is portaled outside the modal, which makes it
// unreachable to screen readers, so controls point aria-describedby at `id`.
export default function TranslationHelp({ id, label, children }) {
  return (
    <>
      <HelpTooltip
        label={label}
        anchorClassName="help-tooltip-anchor--inline"
        popupClassName="help-tooltip-popup--fixed"
        useFixedPosition
      >
        {children}
      </HelpTooltip>
      <span id={id} className="visually-hidden">{children}</span>
    </>
  );
}
