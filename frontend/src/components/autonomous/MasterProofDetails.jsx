import React, { useId, useState } from 'react';
import { downloadTextFile } from '../../utils/downloadHelpers';
import { isMasterProof, sanitizeDomId } from '../../utils/proofPresentation';
import './MasterProofDetails.css';

/**
 * Shows how one verified master proof was built: its solved steps, the solved
 * template models see, and each step's verified Lean proof. The full export is
 * loaded only when the reader opens a part that needs it or downloads it.
 */
export default function MasterProofDetails({ proof, loadBundle }) {
  const panelId = sanitizeDomId(useId(), 'master-proof');
  const stepPanelId = (name) => sanitizeDomId(name, `${panelId}-step`);
  const [bundle, setBundle] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showTemplate, setShowTemplate] = useState(false);
  const [openSteps, setOpenSteps] = useState(() => new Set());

  if (!isMasterProof(proof)) return null;

  const summary = proof.master_proof || {};
  const steps = Array.isArray(summary.steps) ? summary.steps : [];
  const stepCode = new Map((bundle?.steps || []).map((step) => [step.name, step.lean_code || '']));

  const ensureBundle = async () => {
    if (bundle) return bundle;
    setLoading(true);
    setError('');
    try {
      const loaded = await loadBundle(proof);
      setBundle(loaded);
      return loaded;
    } catch (err) {
      setError(err?.message || 'Could not load this master proof.');
      return null;
    } finally {
      setLoading(false);
    }
  };

  const toggleTemplate = async () => {
    const next = !showTemplate;
    setShowTemplate(next);
    if (next) await ensureBundle();
  };

  const toggleStep = async (name) => {
    const next = new Set(openSteps);
    if (next.has(name)) {
      next.delete(name);
    } else {
      next.add(name);
    }
    setOpenSteps(next);
    if (next.has(name)) await ensureBundle();
  };

  const downloadBundle = async () => {
    const loaded = await ensureBundle();
    if (!loaded) return;
    downloadTextFile(
      JSON.stringify(loaded, null, 2),
      `${proof.proof_id}_master_proof.json`,
      'application/json'
    );
  };

  const downloadTemplate = async () => {
    const loaded = await ensureBundle();
    if (!loaded) return;
    downloadTextFile(
      loaded.solved_template || '',
      `${proof.theorem_name || proof.proof_id}_solved_template.lean`
    );
  };

  const stepCount = steps.length;
  const submissions = Number(summary.total_attempts || proof.attempt_count || 0);

  return (
    <div className="master-proof-details">
      <strong>Master Proof</strong>
      <p className="master-proof-details__summary">
        Built from {stepCount} solved {stepCount === 1 ? 'step' : 'steps'} in {submissions} Lean 4
        {submissions === 1 ? ' submission' : ' submissions'}. Lean 4 verified the whole proof, and the
        novelty rating above is for the master theorem as a whole.
      </p>

      <div className="master-proof-details__actions">
        <button type="button" className="math-proof-download" onClick={downloadBundle} disabled={loading}>
          Download Master Proof Bundle (JSON)
        </button>
        <button type="button" className="math-proof-download" onClick={downloadTemplate} disabled={loading}>
          Download Solved Template (.lean)
        </button>
        <button
          type="button"
          className="math-proof-download"
          onClick={toggleTemplate}
          aria-expanded={showTemplate}
          aria-controls={`${panelId}-template`}
        >
          {showTemplate ? 'Hide Solved Template' : 'View Solved Template'}
        </button>
      </div>

      {loading && <p className="master-proof-details__status" role="status">Loading the master proof…</p>}
      {error && <p className="master-proof-details__error" role="alert">{error}</p>}

      {showTemplate && bundle && (
        <div id={`${panelId}-template`} className="master-proof-details__template">
          <p className="master-proof-details__hint">
            This is how the master proof is shown to models in later work: the statement and its solved
            steps, with the proofs hidden until a model asks to view them.
          </p>
          <pre>{bundle.solved_template}</pre>
        </div>
      )}

      {stepCount > 0 && (
        <ol className="master-proof-details__steps" aria-label="Solved steps of this master proof">
          {steps.map((step) => {
            const isOpen = openSteps.has(step.name);
            const failed = Number(step.failed_attempts || 0);
            return (
              <li
                key={step.name}
                className="master-proof-details__step"
                style={{ marginLeft: `${Math.min(Number(step.depth || 0), 6) * 1.1}rem` }}
              >
                <div className="master-proof-details__step-head">
                  <span className="master-proof-details__solved" aria-hidden="true">✓</span>
                  <code className="master-proof-details__step-name">{step.name}</code>
                  <span className="master-proof-details__step-meta">
                    Solved{failed > 0 ? ` after ${failed} failed ${failed === 1 ? 'attempt' : 'attempts'}` : ' first try'}
                    {step.parent ? ` · expanded from ${step.parent}` : ''}
                  </span>
                  <button
                    type="button"
                    className="master-proof-details__toggle"
                    onClick={() => toggleStep(step.name)}
                    aria-expanded={isOpen}
                    aria-controls={stepPanelId(step.name)}
                  >
                    {isOpen ? 'Hide proof' : 'View proof'}
                  </button>
                </div>
                {step.signature && <pre className="master-proof-details__signature">{step.signature}</pre>}
                {isOpen && bundle && (
                  <pre id={stepPanelId(step.name)} className="master-proof-details__code">
                    {stepCode.get(step.name) || 'This step is no longer a separate declaration in the final proof.'}
                  </pre>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
