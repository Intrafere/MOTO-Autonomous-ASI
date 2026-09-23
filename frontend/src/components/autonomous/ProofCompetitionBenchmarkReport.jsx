import React, { useEffect, useState } from 'react';
import { proofCompetitionBenchmarksAPI } from '../../services/api';
import HelpTooltip from '../HelpTooltip';

const routeLabel = (route) => route
  ? `${route.provider} / ${route.model_id}${route.host ? ` (${route.host})` : ''}${route.boosted ? ' · Boost' : ''}${route.fallback ? ' · fallback' : ''}${route.supercharge ? ' · Supercharge' : ''}`
  : 'Unknown';
const usage = (value) => value == null ? 'Unknown' : value.toLocaleString();

/** Shared live and historical report reader; never starts competition work. */
export default function ProofCompetitionBenchmarkReport({ current = false, api = proofCompetitionBenchmarksAPI }) {
  const [page, setPage] = useState({ reports: [], total: 0 });
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState('');
  const [report, setReport] = useState(null);
  const [pairIndex, setPairIndex] = useState(0);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    let alive = true;
    let timer;
    const load = async () => {
      try {
        const next = await api.list({ current, offset, limit: 25 });
        if (!alive) return;
        setPage(next);
        setSelected((previous) => next.reports.some((row) => `${row.session_id}/${row.report_id}` === previous)
          ? previous : next.reports[0] ? `${next.reports[0].session_id}/${next.reports[0].report_id}` : '');
        setError('');
      } catch (err) {
        if (alive) setError(err.message || 'Unable to load benchmarks');
      } finally {
        if (alive) {
          setLoading(false);
          if (current) timer = setTimeout(load, 10000);
        }
      }
    };
    setLoading(true);
    load();
    return () => { alive = false; clearTimeout(timer); };
  }, [api, current, offset, refresh]);

  useEffect(() => {
    let alive = true;
    let timer;
    setReport(null);
    setPairIndex(0);
    if (!selected) return () => { alive = false; };
    const row = page.reports.find((item) => `${item.session_id}/${item.report_id}` === selected);
    if (!row) return () => { alive = false; };
    const load = async () => {
      try {
        const next = await api.get(row.session_id, row.report_id);
        if (alive) { setReport(next); setError(''); }
      } catch (err) {
        if (alive) setError(err.message || 'Unable to load report');
      } finally {
        if (alive && current) timer = setTimeout(load, 10000);
      }
    };
    load();
    return () => { alive = false; clearTimeout(timer); };
    // Identity, not polling-created row objects, owns the mounted detail reader.
  }, [api, selected, current, refresh]); // eslint-disable-line react-hooks/exhaustive-deps

  const pair = report?.pairwise?.[pairIndex];
  return <section className="proof-competition-benchmarks" aria-label={current ? 'Current proof competition benchmarks' : 'Proof competition benchmark library'}>
    <h3>{current ? 'Current-run Benchmarks' : 'Benchmarks'}</h3>
    <button type="button" onClick={() => setRefresh((value) => value + 1)}>Refresh benchmarks</button>
    {error && <p role="alert">{error}</p>}
    {loading && <p role="status">Loading benchmarks…</p>}
    {!loading && !error && page.reports.length === 0 && <p>No competition benchmark reports. Reports are created only for enabled proof competitions.</p>}
    {page.reports.length > 0 && <label>Report <select aria-label="Benchmark report" value={selected} onChange={(event) => setSelected(event.target.value)}>
      {page.reports.map((row) => <option key={`${row.session_id}/${row.report_id}`} value={`${row.session_id}/${row.report_id}`}>{row.session_id} · {row.report_id} · {row.record_count} outcomes</option>)}
    </select></label>}
    {!current && <nav aria-label="Benchmark pages">
      <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 25))}>Previous reports</button>
      <span> {page.total} reports </span>
      <button disabled={offset + 25 >= page.total} onClick={() => setOffset(offset + 25)}>Next reports</button>
    </nav>}
    {report && <>
      <p>How to interpret these benchmarks <HelpTooltip label="Benchmark comparison limitations">{report.caveat} Context differences may be larger when comparing Model 1 with Model 3 or later.</HelpTooltip></p>
      <p>Operational totals: {report.attempted} started · {report.verified} verified · {report.secondary_rescues} secondary rescues · {report.unavailable} unavailable · {report.interrupted} interrupted</p>
      <p>Comparisons score configured lanes, not individual effective models. Boost or fallback may change the model used; inspect observed effective routes below.</p>
      {report.pairwise.length > 0 ? <label>Compare configured lanes <select aria-label="Compare benchmark pair" value={pairIndex} onChange={(event) => setPairIndex(Number(event.target.value))}>
        {report.pairwise.map((item, index) => <option key={`${item.left.competitor_key}/${item.right.competitor_key}`} value={index}>{item.left.competitor_id} ({routeLabel(item.left.configured_route)}) vs {item.right.competitor_id} ({routeLabel(item.right.configured_route)})</option>)}
      </select></label> : <p>No competitor pair recorded yet.</p>}
      {pair && (pair.shared_problem_count === 0 ? <p>No shared attempted problems.</p> : <>
        <h4>{pair.shared_problem_count} shared attempted problems</h4>
        <p>Conditional rescue: {pair.conditional_rescue_rate == null ? 'No completed shared rescue opportunities' : `${pair.conditional_rescue_count} of ${pair.conditional_rescue_opportunities} completed opportunities (${(pair.conditional_rescue_rate * 100).toFixed(1)}%)`}. Interrupted opportunities are excluded from this rate.</p>
        <details><summary>Shared-problem details</summary>
          {(pair.shared_records || []).map((row, index) => <p key={index}>{row.source_id} · round {row.round_index} · {row.candidate_id} · {row.competitor_id}: {row.outcome} · context {row.context_revision || 'Unknown'}</p>)}
        </details>
        {[pair.left, pair.right].map((score) => <p key={score.competitor_key}>{score.competitor_id}: {score.verified} verified · {score.completed_losses} completed losses · {score.incomplete} incomplete/interrupted · {score.solve_rate == null ? 'No completed opportunities' : `${(score.solve_rate * 100).toFixed(1)}% verified of completed shared opportunities`}</p>)}
      </>)}
      <details><summary>Problem outcomes, routes, context and error breakdown</summary>
        {report.records.map((row, index) => <article key={`${row.execution_id}/${row.candidate_id}/${row.competitor_id}/${row.route_revision}/${index}`}>
          <h4>{row.candidate_id} · {row.competitor_id} · {row.outcome}</h4>
          <p>{row.source_type}: {row.source_id} · round {row.round_index} · execution {row.execution_id} · route revision {row.route_revision}</p>
          <p>Started: {row.started_at || 'Not started'} · Finished: {row.finished_at || 'Not finished'} · Attempts consumed: {row.attempts_consumed} · Elapsed: {row.elapsed_seconds == null ? 'Unknown' : `${row.elapsed_seconds}s`}</p>
          <p>Configured: {routeLabel(row.configured_route)}</p>
          <p>Effective: {row.effective_routes.length ? row.effective_routes.map(routeLabel).join('; ') : 'Unknown — not attributed to configured model'}</p>
          <p>Tokens: input {usage(row.input_tokens)} · output {usage(row.output_tokens)}</p>
          <p>Context revision: {row.context_revision || 'Unknown'} · Supports: {row.support_ids.length ? row.support_ids.join(', ') : 'Unknown / not recorded'}</p>
          {row.interruption_kind && <p>Error category: {row.interruption_kind}</p>}
          {row.proof_id && <p>Winning proof: {row.proof_id}</p>}
        </article>)}
      </details>
    </>}
  </section>;
}
