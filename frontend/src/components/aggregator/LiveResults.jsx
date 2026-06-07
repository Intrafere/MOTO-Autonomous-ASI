/**
 * LiveResults - Displays accepted aggregator submissions with LaTeX rendering.
 */
import React, { useState, useEffect, useRef } from 'react';
import { api } from '../../services/api';
import { websocket } from '../../services/websocket';
import LatexRenderer from '../LatexRenderer';
import { prependDisclaimer } from '../../utils/disclaimerHelper';
import {
  MANUAL_AGGREGATOR_PROOF_SOURCE_ID,
  useProofCheckRuntime,
} from '../../hooks/useProofCheckRuntime';

export default function LiveResults() {
  const [results, setResults] = useState('');
  const [autoScroll, setAutoScroll] = useState(true);
  const [showLatex, setShowLatex] = useState(false); // Raw text by default for performance with large docs
  const [proofActionMessage, setProofActionMessage] = useState('');
  const resultsRef = useRef(null);
  const {
    canQueueManualProofCheck,
    getManualCheckReason,
    getSourceState,
    queueManualProofCheck,
  } = useProofCheckRuntime();

  useEffect(() => {
    fetchResults();
    const interval = setInterval(fetchResults, 5000);

    const unsubscribeSubmissionAccepted = websocket.on('submission_accepted', () => {
      // Fetch new results when a submission is accepted
      setTimeout(fetchResults, 500);
    });
    const refreshManualAggregatorProofs = (data = {}) => {
      if (data.source_type === 'brainstorm' && data.source_id === MANUAL_AGGREGATOR_PROOF_SOURCE_ID) {
        setTimeout(fetchResults, 500);
      }
    };
    const proofRefreshUnsubscribers = [
      websocket.on('proof_check_complete', refreshManualAggregatorProofs),
      websocket.on('proof_verified', refreshManualAggregatorProofs),
      websocket.on('known_proof_verified', refreshManualAggregatorProofs),
      websocket.on('proof_registration_duplicate', refreshManualAggregatorProofs),
      websocket.on('novel_proof_discovered', refreshManualAggregatorProofs),
    ];

    return () => {
      clearInterval(interval);
      unsubscribeSubmissionAccepted();
      proofRefreshUnsubscribers.forEach(unsubscribe => unsubscribe());
    };
  }, []);

  useEffect(() => {
    if (autoScroll && resultsRef.current) {
      resultsRef.current.scrollTop = resultsRef.current.scrollHeight;
    }
  }, [results, autoScroll]);

  const fetchResults = async () => {
    try {
      const data = await api.getResults();
      setResults(data.results || 'No accepted submissions yet.');
    } catch (error) {
      console.error('Failed to fetch results:', error);
    }
  };

  const handleSave = async () => {
    try {
      const data = await api.saveResults();
      alert(`Results saved to ${data.path}`);
    } catch (error) {
      console.error('Failed to save results:', error);
      alert('Failed to save results');
    }
  };

  const handleClearAll = async () => {
    if (!confirm('Are you sure you want to clear ALL accepted submissions and start a fresh manual run?\n\nCurrent manual proofs will be archived to history and removed from future prompt context. This cannot be undone!')) {
      return;
    }
    
    try {
      const data = await api.clearAllSubmissions();
      alert(data.message);
      // Refresh results to show empty state
      fetchResults();
    } catch (error) {
      console.error('Failed to clear submissions:', error);
      alert('Failed to clear submissions: ' + error.message);
    }
  };

  const handleDownload = () => {
    if (!results || results === 'No accepted submissions yet.') {
      alert('No results to download');
      return;
    }
    
    const blob = new Blob([prependDisclaimer(results, 'brainstorm')], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `aggregator_results_${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleProofCheck = async () => {
    try {
      setProofActionMessage('');
      await queueManualProofCheck({
        sourceType: 'brainstorm',
        sourceId: MANUAL_AGGREGATOR_PROOF_SOURCE_ID,
      });
      setProofActionMessage('Queued proof check for the manual Aggregator database.');
    } catch (error) {
      setProofActionMessage(`Failed to queue proof check: ${error.message}`);
    }
  };

  const proofCheckState = getSourceState('brainstorm', MANUAL_AGGREGATOR_PROOF_SOURCE_ID);
  const proofCheckLabel = proofCheckState?.status === 'queued'
    ? 'Queueing Proof Check...'
    : proofCheckState?.status === 'running'
      ? `Proof Check Running${proofCheckState.candidateCount ? ` (${proofCheckState.candidateCount})` : '...'}`
      : 'Try to Prove This';
  const hasResults = Boolean(results && results !== 'No accepted submissions yet.');
  const proofCheckEnabled = hasResults && canQueueManualProofCheck('brainstorm', MANUAL_AGGREGATOR_PROOF_SOURCE_ID);
  const proofCheckTitle = proofCheckState?.status === 'running'
    ? 'A proof verification is already running for the manual Aggregator database.'
    : getManualCheckReason('brainstorm', MANUAL_AGGREGATOR_PROOF_SOURCE_ID) || 'Queue a manual proof check for this brainstorm database.';

  return (
    <div className="live-results">
      <h1>Live Results</h1>

      <div className="button-group results-controls">
        <button onClick={handleSave} className="secondary">
          💾 Save to Server
        </button>
        <button onClick={handleDownload} className="secondary">
          Download
        </button>
        <button
          onClick={handleProofCheck}
          className="secondary"
          disabled={!proofCheckEnabled || Boolean(proofCheckState)}
          title={hasResults ? proofCheckTitle : 'No accepted submissions to prove yet.'}
        >
          {proofCheckLabel}
        </button>
        <button 
          onClick={() => setAutoScroll(!autoScroll)}
          className={autoScroll ? '' : 'secondary'}
        >
          Auto-scroll: {autoScroll ? 'ON' : 'OFF'}
        </button>
        <button 
          onClick={() => setShowLatex(!showLatex)}
          className={showLatex ? '' : 'secondary'}
        >
          LaTeX: {showLatex ? 'ON' : 'OFF'}
        </button>
        <button onClick={fetchResults} className="secondary">
          Refresh
        </button>
        <button onClick={handleClearAll} className="danger">
          Clear All
        </button>
      </div>

      {proofActionMessage && (
        <div className={`test-result-banner ${proofActionMessage.startsWith('Failed') ? 'test-result-banner--error' : 'test-result-banner--success'}`}>
          {proofActionMessage}
        </div>
      )}

      <div className="results-container" ref={resultsRef}>
        <LatexRenderer
          content={prependDisclaimer(results, 'brainstorm')}
          className="results-latex-renderer"
          showToggle={false}
          showLatex={showLatex}
        />
      </div>
    </div>
  );
}
