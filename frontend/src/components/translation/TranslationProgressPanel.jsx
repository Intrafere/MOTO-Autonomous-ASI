import React, { useEffect, useId, useRef, useState } from 'react';
import {
  TRANSLATOR_SUBMITTER_TITLE,
  TRANSLATOR_VALIDATOR_TITLE,
  describeTranslationError,
} from './translatorRoleConfig';
import { withRouteIdentity } from '../../utils/activityStyles';
import { AUTO_TRANSLATE_NAME } from '../../hooks/useAutoTranslate';
import { formatPartRange, translationPercent } from './translationPercent';

export const MAX_ACTIVITY_SHOWN = 50;
export const ACCEPTS_TO_FINALIZE = 3;

export const QUEUE_KIND_LABELS = {
  proof: 'Proof',
  paper: 'Paper',
  final_answer: 'Final answer',
  brainstorm: 'Brainstorm',
  aggregator_results: 'Results',
  live_paper: 'Live paper',
  other: 'Other',
};

const RETRY_REASON_TEXT = {
  translation_validator_contract: 'the validator reply was not in the required format',
  free_models_exhausted: 'every free model is rate-limited',
  translation_loop_retry: 'an unexpected error interrupted the attempt',
};

const FINALIZE_REASON_TEXT = {
  three_accepted: 'three accepted attempts',
  kept_twice: 'kept twice',
};

// Translate all starting, being cancelled (by the user or a cache clear), and finishing;
// the text is used when an entry carries no message.
const TRANSLATE_ALL_ACTIVITY = {
  started: { tone: 'accepted', text: 'Translate all started.' },
  cancelled: { tone: 'kept', text: 'Translate all cancelled.' },
  finished: { tone: 'final', text: 'Translate all finished.' },
};

function count(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? Math.floor(number) : 0;
}

function formatCount(value) {
  return count(value).toLocaleString();
}

function routeIdentityText(value) {
  if (typeof value === 'string') {
    return value.trim();
  }
  if (value && typeof value === 'object' && typeof value.route_identity === 'string') {
    return value.route_identity.trim();
  }
  return '';
}

function activityTime(item) {
  const raw = item?.timestamp ?? item?._serverTimestamp ?? item?.received_at;
  if (typeof raw === 'number' && Number.isFinite(raw)) {
    return raw < 1e12 ? raw * 1000 : raw;
  }
  const parsed = Date.parse(typeof raw === 'string' ? raw : '');
  return Number.isFinite(parsed) ? parsed : null;
}

function translationRoleLabel(roleId) {
  const role = typeof roleId === 'string' ? roleId : '';
  if (role === 'translation_submitter') {
    return TRANSLATOR_SUBMITTER_TITLE;
  }
  if (role === 'translation_validator') {
    return TRANSLATOR_VALIDATOR_TITLE;
  }
  if (/^translation_language_probe_\d+$/.test(role)) {
    return 'Language check';
  }
  return 'Translation model';
}

// The attempt an activity item describes. Live events carry attempt_number; the
// status copy keeps candidate_id ("{target}:attempt{n}", ":rejudge{k}" when judged
// again), and round_index is the attempt number otherwise.
function attemptNumber(item) {
  if (count(item?.attempt_number)) {
    return count(item.attempt_number);
  }
  const match = /:attempt(\d+)(?::rejudge\d+)?$/.exec(typeof item?.candidate_id === 'string' ? item.candidate_id : '');
  if (match) {
    return Number(match[1]);
  }
  return count(item?.round_index) || '?';
}

function wasRejudged(item) {
  if (typeof item?.rejudged === 'boolean') {
    return item.rejudged;
  }
  return typeof item?.candidate_id === 'string' && /:rejudge\d+$/.test(item.candidate_id);
}

// A keep naming a rejected attempt is a dispute; it resets the keep streak, so a
// status copy without the dispute flag is a dispute when its streak is 0.
function wasDispute(item) {
  if (typeof item?.dispute === 'boolean') {
    return item.dispute;
  }
  return count(item?.keep_streak) === 0;
}

function attemptList(numbers) {
  const items = (Array.isArray(numbers) ? numbers : []).map(count).filter(Boolean);
  if (items.length === 0) {
    return 'an earlier attempt';
  }
  const label = items.length === 1 ? 'attempt' : 'attempts';
  return `${label} ${items.join(', ')}`;
}

function retryReasonText(reason) {
  const code = typeof reason === 'string' ? reason.trim() : '';
  if (!code) {
    return '';
  }
  return RETRY_REASON_TEXT[code] || code.replace(/_/g, ' ');
}

// The shared alert says to lower reasoning effort "in Settings", but translator models
// are set in this window; the route (or role) says which translator model is slow.
function describeHungConnection(item, route) {
  const minutes = count(item.elapsed_minutes);
  const message = `Possible hung model call${minutes ? ` (${minutes}+ min)` : ''}. It may still be thinking; `
    + 'you can keep waiting, or lower the reasoning effort for the translator models on the Translator models '
    + 'tab of this window if this repeats.';
  return route
    ? withRouteIdentity(message, { route_identity: route })
    : `${translationRoleLabel(item.role_id)}: ${message}`;
}

// A live retry wait from the shared seam; its message gets the route prefix once.
function describeProviderRetry(item, route) {
  if (item.message) {
    return { tone: 'provider', text: withRouteIdentity(item.message, { route_identity: route }), details: [] };
  }
  const source = route || 'Translation model';
  const seconds = count(item.retry_after_seconds);
  return { tone: 'provider', text: seconds ? `${source}: retrying in ${seconds}s` : `${source}: retrying`, details: [] };
}

// A retry wait the service recorded in status; its stored route (model, provider, host)
// prefixes the readable message once, and the provider's error detail follows it.
function describeTranslationRetry(item, route) {
  const details = item.error_summary ? [String(item.error_summary)] : [];
  if (item.message) {
    return { tone: 'provider', text: withRouteIdentity(item.message, { route_identity: route }), details };
  }
  const source = route || translationRoleLabel(item.role_id);
  const seconds = count(item.retry_after_seconds);
  const reason = retryReasonText(item.reason);
  const text = seconds ? `${source}: retrying in ${seconds}s` : `${source}: retrying`;
  return { tone: 'provider', text: reason ? `${text} (${reason})` : text, details };
}

// "[route] Repair required: …": the service stores the message with its route prefix,
// which moves to the front of the line once.
function describeRepair(item, route) {
  const stored = String(item.message || `${AUTO_TRANSLATE_NAME} needs its translator settings repaired.`).trim();
  const message = route && stored.startsWith(route) ? stored.slice(route.length).trim() : stored;
  return {
    tone: 'repair',
    text: withRouteIdentity(`Repair required: ${message}`, { route_identity: route }),
    details: item.guidance ? [String(item.guidance)] : [],
  };
}

// Plain-text description of one translation activity item (never HTML), or null for
// event types this panel does not show (progress, queue, and status notices, unknown types).
export function describeTranslationActivity(item, languageNames = new Map()) {
  const language = item?.language_name || languageNames.get(item?.language_key) || item?.language_key || 'Translation';
  const doc = item?.doc_title || 'untitled item';
  const where = `“${doc}”`;
  const route = routeIdentityText(item?.route_identity);
  switch (item?.event_type) {
    case 'translation_attempt_accepted':
      return {
        tone: 'accepted',
        text: `${language}: attempt ${attemptNumber(item)} for ${where} was ${wasRejudged(item) ? 'judged again and ' : ''}`
          + `accepted (${count(item.accepted_count)}/${ACCEPTS_TO_FINALIZE} accepted)`,
        details: [],
      };
    case 'translation_attempt_rejected':
      return {
        tone: 'rejected',
        text: `${language}: attempt ${attemptNumber(item)} for ${where} was ${wasRejudged(item) ? 'judged again and ' : ''}`
          + 'rejected',
        details: item.feedback ? [String(item.feedback)] : [],
      };
    case 'translation_attempt_kept':
      if (wasDispute(item)) {
        return {
          tone: 'kept',
          text: `${language}: the translator asked the validator to judge ${attemptList(item.keep_attempts)} for `
            + `${where} again`,
          details: [],
        };
      }
      return {
        tone: 'kept',
        text: `${language}: the translator kept ${attemptList(item.keep_attempts)} for ${where} `
          + `(keeps in a row: ${count(item.keep_streak)})`,
        details: [],
      };
    case 'translation_part_finalized': {
      const reason = FINALIZE_REASON_TEXT[item.reason];
      return {
        tone: 'final',
        text: `${language}: a part of ${where} is final with `
          + `attempt ${count(item.final_attempt) || '?'}${reason ? ` (${reason})` : ''}`,
        details: [],
      };
    }
    case 'translation_attempt_check_failed':
      return {
        tone: 'check',
        text: `${language}: attempt ${attemptNumber(item)} for ${where} failed the automatic checks`,
        details: (Array.isArray(item.problems) ? item.problems : []).filter(Boolean).slice(0, 5).map(String),
      };
    case 'translation_validator_contract_retry':
      return {
        tone: 'check',
        text: `${language}: the validator reply for ${where} was not in the required format; checking again`,
        details: item.problem ? [String(item.problem)] : [],
      };
    case 'translation_repair_required':
      return describeRepair(item, route);
    case 'translation_provider_retry':
      return describeTranslationRetry(item, route);
    case 'translation_waiting_for_credits':
      return {
        tone: 'provider',
        text: withRouteIdentity(item.message || 'Waiting for provider credits to be reset.', { route_identity: route }),
        details: [],
      };
    case 'provider_retry_cooldown':
      return describeProviderRetry(item, route);
    case 'hung_connection_alert':
      return { tone: 'provider', text: describeHungConnection(item, route), details: [] };
    case 'translation_translate_all': {
      const change = TRANSLATE_ALL_ACTIVITY[item.state];
      const text = String(item.message || change?.text || '').trim();
      return text ? { tone: change?.tone || 'kept', text, details: [] } : null;
    }
    default:
      return null;
  }
}

// "Part 12–15% · attempt 2 · 1/3 accepted · keeps in a row 0" for a running job; the range
// is where the part sits in its document of totalUnits units.
export function describeActivePart(active, totalUnits) {
  const range = formatPartRange(active?.part_start, active?.part_end, totalUnits);
  return [
    range ? `Part ${range}` : 'Current part',
    `attempt ${count(active?.attempt) || 1}`,
    `${count(active?.accepted_count)}/${ACCEPTS_TO_FINALIZE} accepted`,
    `keeps in a row ${count(active?.keep_streak)}`,
  ].join(' · ');
}

function jobKey(docHash, languageKey) {
  return `${docHash || ''}\u0001${languageKey || ''}`;
}

// Queue rows in position order with the running part attached to translating rows.
// A running job missing from the bounded queue list still gets a row.
export function buildQueueRows(queue, active) {
  const activeByJob = new Map(
    (Array.isArray(active) ? active : [])
      .filter((item) => item?.doc_hash)
      .map((item) => [jobKey(item.doc_hash, item.language_key), item]),
  );
  const rows = (Array.isArray(queue) ? queue : [])
    .filter((item) => item && typeof item === 'object')
    .map((item, index) => ({ item, index, position: count(item.position) }))
    .sort((left, right) => (left.position - right.position) || (left.index - right.index))
    .map(({ item, position }) => {
      const key = jobKey(item.doc_hash, item.language_key);
      const running = position === 0 ? activeByJob.get(key) || null : null;
      activeByJob.delete(key);
      return { key, item, position, running };
    });
  const missing = [...activeByJob.entries()].map(([key, running]) => ({
    key,
    item: { doc_hash: running.doc_hash, title: running.doc_title, language_key: running.language_key },
    position: 0,
    running,
  }));
  return [...missing, ...rows];
}

function LanguageProgress({ language }) {
  const translated = count(language.translated_units);
  const pending = count(language.pending_units);
  const total = translated + pending;
  const percent = total > 0 ? translationPercent(translated, total) : 0;
  const pendingDocuments = count(language.pending_documents);
  const backgroundDocuments = count(language.background_pending_documents);
  const name = language.name || language.key;
  const summary = total > 0 ? `${percent}% translated` : 'Nothing to translate';
  return (
    <li className="auto-translate-progress__language">
      <div className="auto-translate-progress__language-head">
        <span className="auto-translate-progress__language-name">{name}</span>
        <span className="auto-translate-progress__language-count">{summary}</span>
      </div>
      <div
        className="auto-translate-progress__bar"
        role="progressbar"
        aria-label={`${name} translation progress`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
        aria-valuetext={summary}
      >
        <div className="auto-translate-progress__bar-fill" style={{ width: `${percent}%` }} />
      </div>
      <span className="auto-translate-hint">
        {pendingDocuments} item{pendingDocuments === 1 ? '' : 's'} pending
      </span>
      {backgroundDocuments > 0 && (
        <span className="auto-translate-hint auto-translate-progress__background">
          + {formatCount(backgroundDocuments)} older item{backgroundDocuments === 1 ? '' : 's'} waiting in Translate all
        </span>
      )}
    </li>
  );
}

function QueueRow({ row, languageNames }) {
  const { item, position, running } = row;
  const translating = position === 0;
  return (
    <li className={`auto-translate-queue__item${translating ? ' auto-translate-queue__item--active' : ''}`}>
      <span className="auto-translate-queue__position">{translating ? 'Translating now' : `#${position}`}</span>
      <span className="auto-translate-queue__language">
        {languageNames.get(item.language_key) || item.language_key || 'Translation'}
      </span>
      {item.kind && (
        <span className="auto-translate-kind-badge">{QUEUE_KIND_LABELS[item.kind] || QUEUE_KIND_LABELS.other}</span>
      )}
      <span className="auto-translate-queue__title">{item.title || item.doc_key || 'Untitled item'}</span>
      {item.viewing && <span className="auto-translate-viewing-tag">Viewing</span>}
      {item.background && !item.viewing && <span className="auto-translate-background-tag">Background</span>}
      {item.total_units !== undefined && (
        <span className="auto-translate-queue__units">
          {translationPercent(item.translated_units, item.total_units)}% done
        </span>
      )}
      {running && <span className="auto-translate-queue__part">{describeActivePart(running, item.total_units)}</span>}
    </li>
  );
}

// Progress tab: per-language progress, the queue, live activity, usage, and cache clearing.
export default function TranslationProgressPanel({ status = null, recentActivity = [], onClearCache }) {
  const idPrefix = useId();
  const [confirmingClear, setConfirmingClear] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [clearMessage, setClearMessage] = useState('');
  const [clearError, setClearError] = useState('');
  const mountedRef = useRef(false);
  const clearButtonRef = useRef(null);
  const cancelButtonRef = useRef(null);
  const confirmWasOpenRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // Keep keyboard focus on the control that replaced the one just pressed.
  useEffect(() => {
    if (confirmingClear) {
      cancelButtonRef.current?.focus();
    } else if (confirmWasOpenRef.current) {
      clearButtonRef.current?.focus();
    }
    confirmWasOpenRef.current = confirmingClear;
  }, [confirmingClear]);

  const languages = Array.isArray(status?.languages) ? status.languages.filter((item) => item?.key) : [];
  const languageNames = new Map(languages.map((language) => [language.key, language.name || language.key]));
  const queueRows = buildQueueRows(status?.queue, status?.active);
  const activity = (Array.isArray(recentActivity) ? recentActivity : [])
    .map((item) => ({ item, description: describeTranslationActivity(item, languageNames) }))
    .filter((entry) => entry.description)
    .slice(0, MAX_ACTIVITY_SHOWN);
  const usage = status?.usage || {};

  const confirmClear = async () => {
    setClearing(true);
    setClearError('');
    try {
      await onClearCache();
      if (mountedRef.current) {
        setClearMessage('Stored translations cleared.');
        setConfirmingClear(false);
      }
    } catch (error) {
      if (mountedRef.current) {
        setClearError(describeTranslationError(error, 'Stored translations could not be cleared.'));
      }
    } finally {
      if (mountedRef.current) {
        setClearing(false);
      }
    }
  };

  return (
    <div className="auto-translate-progress">
      <section className="auto-translate-section" aria-labelledby={`${idPrefix}-languages`}>
        <h3 id={`${idPrefix}-languages`} className="auto-translate-section__title">Languages</h3>
        {languages.length === 0 ? (
          <p className="auto-translate-empty">Select languages on the Languages tab to see translation progress.</p>
        ) : (
          <ul className="auto-translate-progress__languages">
            {languages.map((language) => <LanguageProgress key={language.key} language={language} />)}
          </ul>
        )}
      </section>

      <section className="auto-translate-section" aria-labelledby={`${idPrefix}-queue`}>
        <h3 id={`${idPrefix}-queue`} className="auto-translate-section__title">Queue</h3>
        {queueRows.length === 0 ? (
          <p className="auto-translate-empty">Nothing is waiting to be translated.</p>
        ) : (
          <ul className="auto-translate-queue" aria-label="Translation queue">
            {queueRows.map((row) => <QueueRow key={row.key} row={row} languageNames={languageNames} />)}
          </ul>
        )}
      </section>

      <section className="auto-translate-section" aria-labelledby={`${idPrefix}-activity`}>
        <h3 id={`${idPrefix}-activity`} className="auto-translate-section__title">Activity</h3>
        {activity.length === 0 ? (
          <p className="auto-translate-empty">No translation activity yet.</p>
        ) : (
          <ul className="auto-translate-activity" aria-label="Translation activity">
            {activity.map(({ item, description }, index) => {
              const time = activityTime(item);
              return (
                <li
                  key={item.activity_id || `${item.event_type}-${index}`}
                  className={`auto-translate-activity__item auto-translate-activity__item--${description.tone}`}
                >
                  {time !== null && (
                    <time className="auto-translate-activity__time" dateTime={new Date(time).toISOString()}>
                      {new Date(time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </time>
                  )}
                  <span className="auto-translate-activity__text">{description.text}</span>
                  {description.details.length > 0 && (
                    <ul className="auto-translate-activity__details">
                      {description.details.map((detail, detailIndex) => (
                        <li key={`${detailIndex}-${detail}`}>{detail}</li>
                      ))}
                    </ul>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </section>

      <section className="auto-translate-section" aria-labelledby={`${idPrefix}-usage`}>
        <h3 id={`${idPrefix}-usage`} className="auto-translate-section__title">Usage</h3>
        <p className="auto-translate-usage">
          <span>{formatCount(usage.calls)} model calls</span>
          <span>{formatCount(usage.prompt_tokens)} prompt tokens</span>
          <span>{formatCount(usage.completion_tokens)} completion tokens</span>
          <span>{formatCount(usage.accepted_attempts)} accepted attempts</span>
          <span>{formatCount(usage.rejected_attempts)} rejected attempts</span>
          <span>{formatCount(usage.finalized_parts)} finished parts</span>
          <span>{formatCount(usage.check_failures)} automatic check failures</span>
        </p>
        <p className="auto-translate-hint">
          New papers, proofs, final answers, completed brainstorms, and results from the current run of each mode you
          checked are translated automatically. Anything else translates while you have it open in a selected
          language, and Translate all older items translates the older runs of the checked modes in the background.
        </p>
      </section>

      <section className="auto-translate-section auto-translate-clear" aria-label="Stored translations">
        {confirmingClear ? (
          <div className="auto-translate-confirm" role="group" aria-label="Confirm clearing stored translations">
            <p>Delete every stored translation? Translations are made again from the start.</p>
            <div className="auto-translate-confirm__actions">
              <button type="button" className="auto-translate-danger-btn" onClick={confirmClear} disabled={clearing}>
                {clearing ? 'Clearing…' : 'Clear translations'}
              </button>
              <button
                ref={cancelButtonRef}
                type="button"
                className="secondary"
                onClick={() => setConfirmingClear(false)}
                disabled={clearing}
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <button
            ref={clearButtonRef}
            type="button"
            className="secondary"
            onClick={() => {
              setClearMessage('');
              setClearError('');
              setConfirmingClear(true);
            }}
          >
            Clear stored translations
          </button>
        )}
        {clearMessage && <p className="auto-translate-hint" role="status">{clearMessage}</p>}
        {clearError && <p className="settings-error" role="alert">{clearError}</p>}
      </section>
    </div>
  );
}
