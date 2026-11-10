import React, { useEffect, useId, useRef, useState } from 'react';

// The modes whose current run is translated automatically, in the backend's canonical
// order. LeanOJ is a developer-only app mode, so it is offered (and saved) only in
// developer mode.
export const TRANSLATION_MODES = Object.freeze([
  {
    value: 'autonomous',
    label: 'Autonomous Research',
    shortLabel: 'Autonomous',
    description: 'The active research session: papers, final answer, proofs, and completed brainstorms.',
  },
  {
    value: 'aggregator',
    label: 'Aggregator',
    shortLabel: 'Aggregator',
    description: 'Live Results and proofs found in the Aggregator database (including Single Paper Writer proofs-only runs).',
  },
  {
    value: 'compiler',
    label: 'Single Paper Writer',
    shortLabel: 'Single Paper Writer',
    description: 'The Live Paper and the proofs found in it.',
  },
  {
    value: 'leanoj',
    label: 'LeanOJ Proof Solver',
    shortLabel: 'LeanOJ',
    description: 'Proofs from the loaded Proof Solver run.',
    developerOnly: true,
  },
]);

export const DEFAULT_TRANSLATION_MODES = Object.freeze(['autonomous', 'aggregator', 'compiler']);

const MODE_GROUPS = [
  { label: 'Autonomous S.T.E.M. ASI', values: ['autonomous'] },
  { label: 'Advanced Manual S.T.E.M. ASI', values: ['aggregator', 'compiler'] },
  { label: 'LeanOJ', values: ['leanoj'] },
];

export function visibleTranslationModes(developerModeEnabled) {
  return TRANSLATION_MODES.filter((mode) => developerModeEnabled || !mode.developerOnly);
}

// Whether saved modes include one that only developer mode shows.
export function hasDeveloperOnlyTranslationModes(modes) {
  return Array.isArray(modes)
    && TRANSLATION_MODES.some((mode) => mode.developerOnly && modes.includes(mode.value));
}

// The saved modes in canonical order, without the modes this user cannot see.
export function translationModesPayload(modes, developerModeEnabled) {
  const checked = new Set(Array.isArray(modes) ? modes : []);
  return visibleTranslationModes(developerModeEnabled)
    .filter((mode) => checked.has(mode.value))
    .map((mode) => mode.value);
}

// "All modes", "None — only items you open", or the checked short labels.
export function translationModesSummary(modes, developerModeEnabled) {
  const visible = visibleTranslationModes(developerModeEnabled);
  const checked = translationModesPayload(modes, developerModeEnabled);
  if (checked.length === visible.length) {
    return 'All modes';
  }
  if (checked.length === 0) {
    return 'None — only items you open';
  }
  return visible.filter((mode) => checked.includes(mode.value)).map((mode) => mode.shortLabel).join(', ');
}

// Disclosure checklist of the modes whose current run is translated. The panel renders
// inline because the modal makes portaled content inert; while closed it is hidden, so
// the modal's Tab trap skips it. Escape closes only the list, not the modal around it.
// unavailableReason disables the list and says why next to it.
export default function TranslationModesDropdown({
  modes = DEFAULT_TRANSLATION_MODES,
  developerModeEnabled = false,
  disabled = false,
  unavailableReason = '',
  describedBy,
  help = null,
  onChange,
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);
  const triggerRef = useRef(null);
  const idPrefix = useId();
  const panelId = `${idPrefix}-panel`;
  const unavailableId = `${idPrefix}-unavailable`;
  const visibleValues = new Set(visibleTranslationModes(developerModeEnabled).map((mode) => mode.value));
  const checked = new Set(Array.isArray(modes) ? modes : []);
  const triggerDescription = [unavailableReason ? unavailableId : '', describedBy].filter(Boolean).join(' ');

  // A list that becomes unavailable while open closes.
  useEffect(() => {
    if (unavailableReason) {
      setOpen(false);
    }
  }, [unavailableReason]);

  useEffect(() => {
    if (!open) {
      return undefined;
    }
    const closeOnOutsideMouseDown = (event) => {
      if (!rootRef.current?.contains(event.target)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', closeOnOutsideMouseDown);
    return () => document.removeEventListener('mousedown', closeOnOutsideMouseDown);
  }, [open]);

  const handleKeyDown = (event) => {
    if (!open || event.key !== 'Escape') {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    setOpen(false);
    triggerRef.current?.focus();
  };

  const toggleMode = (value, nextChecked) => {
    const next = new Set(checked);
    if (nextChecked) {
      next.add(value);
    } else {
      next.delete(value);
    }
    onChange?.(TRANSLATION_MODES.filter((mode) => next.has(mode.value)).map((mode) => mode.value));
  };

  // Escape is handled on the wrapper, which React reaches before the modal's document listener.
  return (
    <span ref={rootRef} className="auto-translate-modes" onKeyDown={handleKeyDown}>
      <button
        ref={triggerRef}
        type="button"
        className="secondary auto-translate-modes__trigger"
        aria-expanded={open}
        aria-controls={panelId}
        aria-describedby={triggerDescription || undefined}
        disabled={disabled || Boolean(unavailableReason)}
        onClick={() => setOpen((current) => !current)}
      >
        Current runs: {unavailableReason ? 'Unavailable' : translationModesSummary(modes, developerModeEnabled)}
        <span className="auto-translate-modes__arrow" aria-hidden="true">▾</span>
      </button>
      {help}
      {unavailableReason && (
        <span id={unavailableId} className="auto-translate-hint auto-translate-modes__unavailable">
          {unavailableReason}
        </span>
      )}
      <fieldset id={panelId} className="auto-translate-modes__panel" hidden={!open}>
        <legend>Translate the current run of</legend>
        {MODE_GROUPS.map((group) => {
          const groupModes = TRANSLATION_MODES.filter(
            (mode) => group.values.includes(mode.value) && visibleValues.has(mode.value),
          );
          if (groupModes.length === 0) {
            return null;
          }
          const groupLabelId = `${idPrefix}-${groupModes[0].value}-group`;
          return (
            <div key={group.label} className="auto-translate-modes__group" role="group" aria-labelledby={groupLabelId}>
              <span id={groupLabelId} className="auto-translate-modes__group-title">{group.label}</span>
              {groupModes.map((mode) => {
                const nameId = `${idPrefix}-${mode.value}-name`;
                const descriptionId = `${idPrefix}-${mode.value}-description`;
                return (
                  <label key={mode.value} className="auto-translate-modes__option">
                    <input
                      type="checkbox"
                      checked={checked.has(mode.value)}
                      aria-labelledby={nameId}
                      aria-describedby={descriptionId}
                      onChange={(event) => toggleMode(mode.value, event.target.checked)}
                    />
                    <span className="auto-translate-modes__option-text">
                      <span id={nameId} className="auto-translate-modes__option-name">{mode.label}</span>
                      <span id={descriptionId} className="auto-translate-modes__option-description">
                        {mode.description}
                      </span>
                    </span>
                  </label>
                );
              })}
            </div>
          );
        })}
      </fieldset>
    </span>
  );
}
