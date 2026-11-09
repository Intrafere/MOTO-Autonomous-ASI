import React, { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import { translationAPI } from '../../services/api';
import useExplicitRequest from '../../hooks/useExplicitRequest';
import { describeTranslationError, sanitizeLanguagePayload } from './translatorRoleConfig';
import TranslationHelp from './TranslationHelp';

export const MAX_SELECTED_LANGUAGES = 40;

export const SELECTED_LANGUAGES_HELP = 'Current-run items, and older items while Translate all is on, are translated '
  + 'into every language here; anything else is translated into the language you open it in. When several '
  + 'translations are waiting, items you are viewing go first, then current-run items, then older items, with '
  + 'proofs before papers before other outputs; after that, languages earlier in this list go first. Use ↑ to move '
  + 'a language earlier and × to remove it.';

export const LANGUAGE_CHECK_HELP = 'MOTO asks your saved translator and validator models which languages they '
  + 'translate well. Languages both models list show normally; the others are greyed out and tagged “May not be '
  + 'supported”, and MOTO still tries them. Languages outside the list that both models report are added to it. '
  + 'The check runs by itself when you open this tab after saving new translator models.';

export const ADD_LANGUAGE_HELP = 'Type any language name. A name that matches a language in the list, including '
  + 'one your models reported in the model language check, adds that language. Anything else is added with the tag '
  + '“Custom — may not be supported”, and MOTO still tries it.';
export const CUSTOM_PENDING_LANGUAGE_KEY = 'custom-pending';
export const UNCERTAIN_LANGUAGE_TAG = 'May not be supported';
export const CUSTOM_LANGUAGE_TAG = 'Custom — may not be supported';

export function uncertainLanguageWarning(name) {
  return `Your translator models did not list ${name} as a language they translate well. `
    + 'MOTO will still try, but the translation may be unreliable.';
}

export function normalizeLanguageText(value) {
  return String(value ?? '').trim().replace(/\s+/g, ' ').toLowerCase();
}

function catalogRow(entry) {
  return {
    key: entry.key,
    name: entry.name || entry.key,
    nativeName: entry.native_name && entry.native_name !== entry.name ? entry.native_name : '',
    aliases: Array.isArray(entry.aliases) ? entry.aliases : [],
    uncertain: entry.proficiency === 'uncertain',
    reported: false,
  };
}

function extraRow(extra) {
  return {
    key: extra.key,
    name: extra.name || extra.key,
    nativeName: '',
    aliases: [],
    uncertain: false,
    reported: true,
  };
}

function catalogLanguages(catalog) {
  return Array.isArray(catalog?.languages) ? catalog.languages.filter((entry) => entry?.key) : [];
}

function catalogExtras(catalog) {
  return Array.isArray(catalog?.extras) ? catalog.extras.filter((extra) => extra?.key) : [];
}

function rowMatchesQuery(row, query) {
  if (!query) {
    return true;
  }
  return [row.name, row.nativeName, ...row.aliases].some((value) => normalizeLanguageText(value).includes(query));
}

// "Most spoken" keeps the backend's pinned order; everything else (remaining
// catalog entries plus model-reported extras) is sorted A–Z by name.
export function buildLanguageGroups(catalog, query = '') {
  const languages = catalogLanguages(catalog);
  const byKey = new Map(languages.map((entry) => [entry.key, entry]));
  const pinnedKeys = Array.isArray(catalog?.pinned_keys) && catalog.pinned_keys.length > 0
    ? catalog.pinned_keys
    : languages.filter((entry) => entry.pinned).map((entry) => entry.key);
  const pinnedSet = new Set(pinnedKeys);
  const normalizedQuery = normalizeLanguageText(query);
  const pinned = pinnedKeys
    .map((key) => byKey.get(key))
    .filter(Boolean)
    .map(catalogRow)
    .filter((row) => rowMatchesQuery(row, normalizedQuery));
  const rest = [
    ...languages.filter((entry) => !pinnedSet.has(entry.key)).map(catalogRow),
    ...catalogExtras(catalog).filter((extra) => !byKey.has(extra.key)).map(extraRow),
  ]
    .filter((row) => rowMatchesQuery(row, normalizedQuery))
    .sort((left, right) => left.name.localeCompare(right.name));
  return { pinned, rest };
}

// Catalog entry (by name, native name, or alias) or model-reported extra (by name)
// matching freeform text, case-insensitively.
export function findLanguageMatch(catalog, text) {
  const target = normalizeLanguageText(text);
  if (!target) {
    return null;
  }
  const entry = catalogLanguages(catalog).find((item) => (
    [item.name, item.native_name, ...(Array.isArray(item.aliases) ? item.aliases : [])]
      .some((value) => normalizeLanguageText(value) === target)
  ));
  if (entry) {
    return catalogRow(entry);
  }
  const extra = catalogExtras(catalog).find((item) => normalizeLanguageText(item.name) === target);
  return extra ? extraRow(extra) : null;
}

// How a selected language displays: catalog proficiency wins, model-reported
// extras display as proficient, and freeform entries are always uncertain.
export function selectedLanguageDisplay(language, catalog) {
  const entry = catalogLanguages(catalog).find((item) => item.key === language.key);
  if (entry) {
    const uncertain = entry.proficiency === 'uncertain';
    return { name: entry.name || language.name, uncertain, tag: uncertain ? UNCERTAIN_LANGUAGE_TAG : '' };
  }
  const extra = catalogExtras(catalog).find((item) => item.key === language.key);
  if (extra) {
    return { name: extra.name || language.name, uncertain: false, tag: '' };
  }
  if (language.custom) {
    return { name: language.name || language.key, uncertain: true, tag: CUSTOM_LANGUAGE_TAG };
  }
  return { name: language.name || language.key, uncertain: false, tag: '' };
}

export function formatRelativeTime(value, now = Date.now()) {
  let time = Number.NaN;
  if (typeof value === 'number' && Number.isFinite(value)) {
    time = value < 1e12 ? value * 1000 : value;
  } else if (typeof value === 'string') {
    time = Date.parse(value);
  }
  if (!Number.isFinite(time)) {
    return 'earlier';
  }
  const seconds = Math.max(0, Math.round((now - time) / 1000));
  if (seconds < 45) {
    return 'just now';
  }
  const plural = (count, unit) => `${count} ${unit}${count === 1 ? '' : 's'} ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) {
    return plural(minutes, 'minute');
  }
  const hours = Math.round(minutes / 60);
  if (hours < 24) {
    return plural(hours, 'hour');
  }
  return plural(Math.round(hours / 24), 'day');
}

export function probeSummary(probe, now = Date.now()) {
  const stale = probe?.stale ? ' Models changed since the last check.' : '';
  switch (probe?.status) {
    case 'complete':
    case 'partial': {
      const models = Array.isArray(probe.models) ? probe.models : [];
      const answered = models.filter((model) => model?.ok !== false).length || models.length;
      const partial = probe.status === 'partial' ? ' Some models could not be checked.' : '';
      return `Checked ${formatRelativeTime(probe.probed_at, now)} with ${answered} model${answered === 1 ? '' : 's'}.${partial}${stale}`;
    }
    case 'failed':
      return `The last language check failed.${stale}`;
    default:
      return 'Check which languages your translator models handle well.';
  }
}

function LanguageRow({ row, checked, onToggle }) {
  return (
    <li className={`auto-translate-language-row${row.uncertain ? ' auto-translate-language-row--uncertain' : ''}`}>
      <label className="auto-translate-language-row__label">
        <input type="checkbox" checked={checked} onChange={() => onToggle(row)} />
        <span className="auto-translate-language-row__name">{row.name}</span>
        {row.nativeName && <span className="auto-translate-language-row__native">{row.nativeName}</span>}
        {row.uncertain && <span className="auto-translate-tag">{UNCERTAIN_LANGUAGE_TAG}</span>}
      </label>
    </li>
  );
}

// Languages tab: selected languages in priority order, the model language check,
// and the searchable catalog. Every change saves immediately with the saved roles.
export default function TranslationLanguagePicker({
  active = true,
  selectedLanguages = [],
  onSaveLanguages,
  modelsSaved = false,
  rolesSignature = '',
}) {
  const idPrefix = useId();
  const [catalog, setCatalog] = useState(null);
  const [catalogSignature, setCatalogSignature] = useState(null);
  const [catalogError, setCatalogError] = useState('');
  const [query, setQuery] = useState('');
  const [addText, setAddText] = useState('');
  const [pendingLanguages, setPendingLanguages] = useState(null);
  const [saveError, setSaveError] = useState('');
  const [notice, setNotice] = useState('');
  const [warning, setWarning] = useState(null);
  const mountedRef = useRef(false);
  const catalogGenerationRef = useRef(0);
  const signatureRef = useRef(rolesSignature);
  const saveChainRef = useRef(Promise.resolve());
  const saveSequenceRef = useRef(0);
  const autoProbedSignatureRef = useRef(null);
  signatureRef.current = rolesSignature;

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const loadCatalog = useCallback(async () => {
    catalogGenerationRef.current += 1;
    const generation = catalogGenerationRef.current;
    const requestSignature = signatureRef.current;
    setCatalogError('');
    try {
      const data = await translationAPI.getLanguages();
      if (!mountedRef.current || generation !== catalogGenerationRef.current) {
        return;
      }
      setCatalog(data && typeof data === 'object' ? data : null);
      setCatalogSignature(requestSignature);
    } catch (error) {
      if (!mountedRef.current || generation !== catalogGenerationRef.current) {
        return;
      }
      setCatalogError(describeTranslationError(error, 'The language list could not be loaded.'));
    }
  }, []);

  useEffect(() => {
    loadCatalog();
  }, [loadCatalog, rolesSignature]);

  const [probeState, runProbe] = useExplicitRequest(true, async ({ isCurrent }) => {
    catalogGenerationRef.current += 1;
    const generation = catalogGenerationRef.current;
    const requestSignature = signatureRef.current;
    try {
      const data = await translationAPI.probeLanguages();
      if (isCurrent() && mountedRef.current && generation === catalogGenerationRef.current) {
        setCatalog(data && typeof data === 'object' ? data : null);
        setCatalogSignature(requestSignature);
      }
      return data;
    } catch (error) {
      if (isCurrent() && mountedRef.current && generation === catalogGenerationRef.current) {
        loadCatalog();
      }
      throw error;
    }
  }, (error) => describeTranslationError(error, 'The model language check failed.'));
  const runProbeRef = useRef(runProbe);
  runProbeRef.current = runProbe;
  const probeRunning = probeState.status === 'running';

  // Check once per saved model set when the tab is open and the check is missing or stale.
  useEffect(() => {
    if (!active || !modelsSaved || !catalog || catalogSignature !== rolesSignature || probeRunning) {
      return;
    }
    const probe = catalog.probe;
    const needsProbe = probe?.status === 'never' || Boolean(probe?.stale);
    if (!needsProbe || autoProbedSignatureRef.current === rolesSignature) {
      return;
    }
    autoProbedSignatureRef.current = rolesSignature;
    runProbeRef.current();
  }, [active, catalog, catalogSignature, modelsSaved, probeRunning, rolesSignature]);

  const displayedLanguages = pendingLanguages ?? selectedLanguages;
  const selectedKeys = useMemo(
    () => new Set(displayedLanguages.map((language) => language.key)),
    [displayedLanguages],
  );
  const groups = useMemo(() => buildLanguageGroups(catalog, query), [catalog, query]);

  const commitLanguages = (nextLanguages) => {
    const payload = nextLanguages.map(sanitizeLanguagePayload);
    setPendingLanguages(payload);
    setSaveError('');
    saveSequenceRef.current += 1;
    const sequence = saveSequenceRef.current;
    const save = async () => {
      try {
        await onSaveLanguages(payload);
        if (mountedRef.current && sequence === saveSequenceRef.current) {
          setPendingLanguages(null);
        }
      } catch (error) {
        if (!mountedRef.current) {
          return;
        }
        if (sequence === saveSequenceRef.current) {
          setPendingLanguages(null);
        }
        setSaveError(describeTranslationError(error, 'Language changes could not be saved.'));
      }
    };
    const run = saveChainRef.current.then(save);
    saveChainRef.current = run.catch(() => {});
    return run;
  };

  const addLanguage = (language, display) => {
    if (displayedLanguages.length >= MAX_SELECTED_LANGUAGES) {
      setNotice(`You can select up to ${MAX_SELECTED_LANGUAGES} languages.`);
      return;
    }
    setNotice('');
    setWarning(display.uncertain ? { key: language.key, name: display.name } : null);
    commitLanguages([...displayedLanguages, language]);
  };

  const removeLanguage = (index) => {
    const removed = displayedLanguages[index];
    if (warning && removed && warning.key === removed.key) {
      setWarning(null);
    }
    setNotice('');
    commitLanguages(displayedLanguages.filter((_, itemIndex) => itemIndex !== index));
  };

  const moveLanguageEarlier = (index) => {
    if (index <= 0) {
      return;
    }
    const next = [...displayedLanguages];
    [next[index - 1], next[index]] = [next[index], next[index - 1]];
    commitLanguages(next);
  };

  const toggleRow = (row) => {
    const index = displayedLanguages.findIndex((language) => language.key === row.key);
    if (index >= 0) {
      removeLanguage(index);
      return;
    }
    addLanguage({ key: row.key, name: row.name, custom: row.reported }, row);
  };

  const submitFreeform = (event) => {
    event.preventDefault();
    const name = addText.trim().replace(/\s+/g, ' ');
    if (!name) {
      return;
    }
    const match = findLanguageMatch(catalog, name);
    if (match) {
      setAddText('');
      if (selectedKeys.has(match.key)) {
        setWarning(null);
        setNotice(`${match.name} is already selected.`);
        return;
      }
      addLanguage({ key: match.key, name: match.name, custom: match.reported }, match);
      return;
    }
    const normalizedName = normalizeLanguageText(name);
    const duplicate = displayedLanguages.find((language) => (
      language.custom && normalizeLanguageText(language.name) === normalizedName
    ));
    setAddText('');
    if (duplicate) {
      setWarning(null);
      setNotice(`${duplicate.name} is already selected.`);
      return;
    }
    addLanguage(
      { key: CUSTOM_PENDING_LANGUAGE_KEY, name, custom: true },
      { name, uncertain: true },
    );
  };

  const probe = catalog?.probe;
  const addId = `${idPrefix}-add`;
  const selectedHelpId = `${idPrefix}-selected-help`;
  const checkHelpId = `${idPrefix}-check-help`;
  const addHelpId = `${idPrefix}-add-help`;
  const hasRows = groups.pinned.length > 0 || groups.rest.length > 0;

  return (
    <div className="auto-translate-languages">
      <section
        className="auto-translate-section"
        aria-labelledby={`${idPrefix}-selected-heading`}
        aria-describedby={selectedHelpId}
      >
        <div className="auto-translate-section__head">
          <h3 id={`${idPrefix}-selected-heading`} className="auto-translate-section__title">Selected languages</h3>
          <TranslationHelp id={selectedHelpId} label="Learn about selected languages">
            {SELECTED_LANGUAGES_HELP}
          </TranslationHelp>
        </div>
        <p className="auto-translate-hint">Languages are translated in this order.</p>
        {displayedLanguages.length === 0 ? (
          <p className="auto-translate-empty">No languages selected yet. Choose languages below.</p>
        ) : (
          <ol className="auto-translate-chip-list" aria-label="Selected languages">
            {displayedLanguages.map((language, index) => {
              const display = selectedLanguageDisplay(language, catalog);
              return (
                <li
                  key={`${language.key}-${index}`}
                  className={`auto-translate-chip${display.uncertain ? ' auto-translate-chip--uncertain' : ''}`}
                >
                  <span className="auto-translate-chip__name">{display.name}</span>
                  {display.tag && <span className="auto-translate-tag">{display.tag}</span>}
                  {index > 0 && (
                    <button
                      type="button"
                      className="auto-translate-icon-btn"
                      aria-label={`Move ${display.name} earlier`}
                      title={`Move ${display.name} earlier`}
                      onClick={() => moveLanguageEarlier(index)}
                    >
                      <span aria-hidden="true">↑</span>
                    </button>
                  )}
                  <button
                    type="button"
                    className="auto-translate-icon-btn"
                    aria-label={`Remove ${display.name}`}
                    title={`Remove ${display.name}`}
                    onClick={() => removeLanguage(index)}
                  >
                    <span aria-hidden="true">×</span>
                  </button>
                </li>
              );
            })}
          </ol>
        )}
        {pendingLanguages && <p className="auto-translate-hint" role="status">Saving language changes…</p>}
        {saveError && <p className="settings-error" role="alert">{saveError}</p>}
        {warning && (
          <p className="auto-translate-warning" role="status">{uncertainLanguageWarning(warning.name)}</p>
        )}
        {notice && <p className="auto-translate-hint" role="status">{notice}</p>}
      </section>

      <section className="auto-translate-section auto-translate-probe" aria-label="Model language check">
        <div className="auto-translate-probe__text">
          <div className="auto-translate-section__head">
            <span className="auto-translate-probe__title">Model language check</span>
            <TranslationHelp id={checkHelpId} label="Learn about the model language check">
              {LANGUAGE_CHECK_HELP}
            </TranslationHelp>
          </div>
          <p className="auto-translate-probe__summary">{probeSummary(probe)}</p>
        </div>
        <div className="auto-translate-probe__actions">
          {!modelsSaved && (
            <span className="auto-translate-hint">Save translator models first to check their languages.</span>
          )}
          <button
            type="button"
            className="secondary"
            aria-describedby={checkHelpId}
            disabled={probeRunning || !modelsSaved}
            onClick={() => runProbe()}
          >
            {probeRunning ? 'Checking model languages…' : 'Check model languages'}
          </button>
        </div>
        {probeState.status === 'error' && (
          <p className="settings-error auto-translate-probe__error" role="alert">{probeState.error}</p>
        )}
      </section>

      <section className="auto-translate-section" aria-labelledby={`${idPrefix}-catalog-heading`}>
        <div className="auto-translate-section__head">
          <h3 id={`${idPrefix}-catalog-heading`} className="auto-translate-section__title">Choose languages</h3>
          <input
            type="search"
            className="input-dark auto-translate-search"
            aria-label="Search languages"
            placeholder="Search languages"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </div>
        {catalogError && (
          <div className="auto-translate-inline-error">
            <p className="settings-error" role="alert">{catalogError}</p>
            <button type="button" className="secondary" onClick={() => loadCatalog()}>Try again</button>
          </div>
        )}
        {!catalog && !catalogError && <p className="auto-translate-hint">Loading languages…</p>}
        {catalog && (
          <div className="auto-translate-language-groups">
            {groups.pinned.length > 0 && (
              <div className="auto-translate-language-group">
                <h4 className="auto-translate-language-group__title">Most spoken</h4>
                <ul className="auto-translate-language-list" aria-label="Most spoken">
                  {groups.pinned.map((row) => (
                    <LanguageRow key={row.key} row={row} checked={selectedKeys.has(row.key)} onToggle={toggleRow} />
                  ))}
                </ul>
              </div>
            )}
            {groups.rest.length > 0 && (
              <div className="auto-translate-language-group">
                <h4 className="auto-translate-language-group__title">All languages (A–Z)</h4>
                <ul className="auto-translate-language-list" aria-label="All languages (A–Z)">
                  {groups.rest.map((row) => (
                    <LanguageRow key={row.key} row={row} checked={selectedKeys.has(row.key)} onToggle={toggleRow} />
                  ))}
                </ul>
              </div>
            )}
            {!hasRows && (
              <p className="auto-translate-empty">No languages match “{query.trim()}”.</p>
            )}
          </div>
        )}
        <form className="auto-translate-add" onSubmit={submitFreeform}>
          <div className="auto-translate-add__head">
            <label htmlFor={addId}>Add another language</label>
            <TranslationHelp id={addHelpId} label="Learn about adding another language">
              {ADD_LANGUAGE_HELP}
            </TranslationHelp>
          </div>
          <div className="auto-translate-add__row">
            <input
              id={addId}
              type="text"
              className="input-dark"
              maxLength={60}
              autoComplete="off"
              aria-describedby={addHelpId}
              value={addText}
              onChange={(event) => setAddText(event.target.value)}
            />
            <button type="submit" className="secondary">Add</button>
          </div>
        </form>
      </section>
    </div>
  );
}
