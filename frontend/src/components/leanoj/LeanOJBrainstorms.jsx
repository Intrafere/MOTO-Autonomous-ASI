import React, { useMemo, useState } from 'react';
import '../autonomous/AutonomousResearch.css';

const BRAINSTORM_PHASES = [
  {
    key: 'initial_brainstorm',
    title: 'Initial Brainstorm',
    description: 'Accepted proof ideas gathered before the first path decision.',
  },
  {
    key: 'recursive_brainstorm',
    title: 'Recursive Brainstorm',
    description: 'Accepted proof ideas gathered when the solver returns for more context.',
  },
];

function toNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function formatDate(isoString) {
  if (!isoString) return '';
  try {
    return new Date(isoString).toLocaleString();
  } catch {
    return isoString;
  }
}

function hasValue(value) {
  return value !== undefined && value !== null && value !== '';
}

function dateValue(value) {
  const timestamp = Date.parse(value || '');
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function sortEvent(record) {
  const number = Number(record.acceptanceEvent);
  return Number.isFinite(number) && number > 0 ? number : record.fallbackIndex;
}

function compareRecordsChronologically(a, b) {
  const eventA = sortEvent(a);
  const eventB = sortEvent(b);
  if (eventA !== eventB) return eventA - eventB;

  const createdA = dateValue(a.createdAt);
  const createdB = dateValue(b.createdAt);
  if (createdA !== createdB) return createdA - createdB;

  return a.fallbackIndex - b.fallbackIndex;
}

function pluralize(count, singular, plural = `${singular}s`) {
  return `${count} ${count === 1 ? singular : plural}`;
}

function normalizeRecords(status) {
  const acceptedIdeas = Array.isArray(status?.accepted_ideas) ? status.accepted_ideas : [];
  const records = Array.isArray(status?.accepted_idea_records) ? status.accepted_idea_records : [];

  if (records.length > 0) {
    return records
      .map((record, index) => ({
        content: record.content || acceptedIdeas[index] || '',
        phase: record.phase || 'initial_brainstorm',
        submitterIndex: record.submitter_index,
        createdAt: record.created_at,
        editedAt: record.edited_at,
        acceptanceEvent: record.acceptance_event,
        pruneAdd: Boolean(record.prune_add),
        reasoning: record.reasoning || record.edit_reasoning || '',
        fallbackIndex: index + 1,
      }))
      .sort(compareRecordsChronologically);
  }

  return acceptedIdeas
    .map((content, index) => ({
      content,
      phase: 'initial_brainstorm',
      submitterIndex: null,
      createdAt: '',
      acceptanceEvent: index + 1,
      pruneAdd: false,
      reasoning: '',
      fallbackIndex: index + 1,
    }))
    .sort(compareRecordsChronologically);
}

export default function LeanOJBrainstorms({ status }) {
  const [expandedPhase, setExpandedPhase] = useState('initial_brainstorm');

  const summary = useMemo(() => {
    const records = normalizeRecords(status);
    const acceptedCount = toNumber(status?.accepted_brainstorm_count, records.length);
    const acceptanceEvents = toNumber(status?.brainstorm_acceptance_events, Math.max(acceptedCount, records.length));
    const prunedCount = Math.max(0, acceptanceEvents - acceptedCount);

    const grouped = BRAINSTORM_PHASES.reduce((acc, phase) => {
      acc[phase.key] = [];
      return acc;
    }, {});

    records.forEach((record) => {
      const phaseKey = grouped[record.phase] ? record.phase : 'initial_brainstorm';
      grouped[phaseKey].push(record);
    });

    Object.values(grouped).forEach((phaseRecords) => {
      phaseRecords.sort(compareRecordsChronologically);
    });

    return {
      records,
      grouped,
      acceptedCount,
      acceptanceEvents,
      prunedCount,
      pruneOperations: toNumber(status?.brainstorm_prune_operations_applied),
    };
  }, [status]);

  return (
    <div className="brainstorm-list leanoj-brainstorms">
      <div className="brainstorm-list-header">
        <div>
          <h3>Proof Solver Brainstorms ({summary.records.length})</h3>
          <p className="settings-hint">
            Review the two Proof Solver brainstorm memories and track how many accepted ideas were pruned from the current working context.
          </p>
        </div>
      </div>

      <div className="brainstorm-list-warning">
        (WARNING: Any given brainstorm idea may be pruned/deleted if the AI deems it to hurt the collective database quality. These Lean brainstorm memories are the working context used to build the final proof.)
      </div>

      <div className="logs-metrics leanoj-brainstorms__metrics">
        <div className="metric-card">
          <span className="metric-value">{summary.acceptedCount}</span>
          <span className="metric-label">Current Ideas</span>
        </div>
        <div className="metric-card">
          <span className="metric-value">{summary.acceptanceEvents}</span>
          <span className="metric-label">Accepted Events</span>
        </div>
        <div className="metric-card">
          <span className="metric-value">{summary.prunedCount}</span>
          <span className="metric-label">Pruned</span>
        </div>
        <div className="metric-card">
          <span className="metric-value">{summary.pruneOperations}</span>
          <span className="metric-label">Prune Operations</span>
        </div>
      </div>

      {BRAINSTORM_PHASES.map((phase) => {
        const records = summary.grouped[phase.key] || [];
        const isExpanded = expandedPhase === phase.key;
        const pruneAddedCount = records.filter((record) => record.pruneAdd).length;
        const latestActivityTime = Math.max(
          0,
          ...records.map((record) => Math.max(dateValue(record.createdAt), dateValue(record.editedAt))),
        );
        const latestActivity = latestActivityTime ? formatDate(new Date(latestActivityTime).toISOString()) : '';

        return (
          <div
            key={phase.key}
            className={`brainstorm-card ${isExpanded ? 'expanded' : ''}`}
          >
            <div
              className="brainstorm-card-clickable"
              onClick={() => setExpandedPhase(isExpanded ? '' : phase.key)}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault();
                  setExpandedPhase(isExpanded ? '' : phase.key);
                }
              }}
            >
              <div className="brainstorm-card-header">
                <span className="brainstorm-card-id">{phase.title}</span>
                <span className={`brainstorm-status ${records.length > 0 ? 'complete' : 'in-progress'}`}>
                  {isExpanded ? '▼' : '▶'} {pluralize(records.length, 'Idea')}
                </span>
              </div>

              <div className="brainstorm-card-prompt">
                {phase.description}
              </div>

              <div className="brainstorm-card-meta">
                <span>{pluralize(records.length, 'current idea')}</span>
                <span>{pluralize(pruneAddedCount, 'prune-added idea')}</span>
                <span>Last: {latestActivity || 'N/A'}</span>
              </div>
            </div>

            {isExpanded && (
              <div className="brainstorm-submissions-container">
                <div className="file-content">
                  <div className="submissions-list-header">
                    <h4>Accepted Ideas</h4>
                  </div>

                  {records.length === 0 ? (
                    <div className="auto-empty-state">
                      No accepted ideas in this brainstorm yet.
                    </div>
                  ) : (
                    <div className="submissions-list leanoj-brainstorms__ideas">
                      {records.map((record) => {
                        const ideaNumber = record.acceptanceEvent || record.fallbackIndex;
                        const createdAt = formatDate(record.createdAt);
                        const editedAt = formatDate(record.editedAt);
                        const meta = [
                          hasValue(record.submitterIndex) ? `Submitter ${record.submitterIndex}` : '',
                          editedAt ? `Edited: ${editedAt}` : createdAt,
                        ].filter(Boolean);

                        return (
                          <article
                            key={`${phase.key}-${ideaNumber}-${record.fallbackIndex}`}
                            className="submission-item leanoj-brainstorms__idea-card"
                          >
                            <div className="submission-header leanoj-brainstorms__idea-header">
                              <span className="submission-number">Idea #{ideaNumber}</span>
                              <span className="submission-timestamp">
                                {meta.join(' | ') || phase.title}
                              </span>
                              {record.pruneAdd && (
                                <span className="brainstorm-status complete">Prune Added</span>
                              )}
                            </div>
                            <div className="submission-content">
                              <pre>{record.content || 'No brainstorm content recorded.'}</pre>
                              {record.reasoning && (
                                <div className="leanoj-brainstorms__reasoning">
                                  <strong>Reasoning</strong>
                                  <span>{record.reasoning}</span>
                                </div>
                              )}
                            </div>
                          </article>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
