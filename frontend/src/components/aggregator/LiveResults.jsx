/**
 * LiveResults - Displays accepted aggregator submissions with LaTeX rendering.
 */
import React, { useState, useEffect, useMemo, useRef } from 'react';
import { api } from '../../services/api';
import { websocket } from '../../services/websocket';
import LatexRenderer from '../LatexRenderer';
import ProofCheckModeModal from '../autonomous/ProofCheckModeModal';
import ProofRunStatusControls from '../autonomous/ProofRunStatusControls';
import TranslationLanguageSwitch from '../translation/TranslationLanguageSwitch';
import TranslationPendingNotice from '../translation/TranslationPendingNotice';
import { prependDisclaimer } from '../../utils/disclaimerHelper';
import { downloadRawText } from '../../utils/downloadHelpers';
import { viewedDownload } from '../../utils/translatedDownloads';
import {
  isProofRunBusy,
  MANUAL_AGGREGATOR_PROOF_SOURCE_ID,
  useProofCheckRuntime,
} from '../../hooks/useProofCheckRuntime';
import useTranslatedDocuments from '../../hooks/useTranslatedDocuments';

const NO_RESULTS_TEXT = 'No accepted submissions yet.';
const RESULTS_TRANSLATION_KEY = 'aggregator:results';
const NO_TRANSLATION_DOCUMENTS = [];

export default function LiveResults({ onClearPrompt = null }) {
  const [results, setResults] = useState('');
  // False while an Autonomous run shows its own brainstorm database here; that content
  // is not the manual Aggregator results, so it is not registered under their key.
  const [manualDatabase, setManualDatabase] = useState(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [showLatex, setShowLatex] = useState(false); // Raw text by default for performance with large docs
  const [proofActionMessage, setProofActionMessage] = useState('');
  const [proofCheckModalOpen, setProofCheckModalOpen] = useState(false);
  const [proofCheckStarting, setProofCheckStarting] = useState(false);
  const resultsRef = useRef(null);
  const {
    canQueueManualProofCheck,
    getManualCheckReason,
    getSourceState,
    queueManualProofCheck,
    stopProofRun,
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

  const hasResults = Boolean(results && results !== NO_RESULTS_TEXT);
  const translationDocuments = useMemo(
    () => (hasResults && manualDatabase !== false
      ? [{
        docKey: RESULTS_TRANSLATION_KEY,
        kind: 'aggregator_results',
        title: 'Live Results',
        profile: 'document',
        parts: { content: results },
      }]
      : NO_TRANSLATION_DOCUMENTS),
    [hasResults, manualDatabase, results]
  );
  const translation = useTranslatedDocuments(translationDocuments);
  // The results' own language choice, kept even before there are results to translate.
  const resultsTranslation = translation.forKey(RESULTS_TRANSLATION_KEY);
  const translationEntry = translation.byKey[RESULTS_TRANSLATION_KEY];
  const translatedResults = translationEntry?.status === 'complete' && typeof translationEntry.parts?.content === 'string'
    ? translationEntry.parts.content
    : null;
  const displayedResults = translatedResults ?? results;

  useEffect(() => {
    if (autoScroll && resultsRef.current) {
      resultsRef.current.scrollTop = resultsRef.current.scrollHeight;
    }
  }, [displayedResults, autoScroll]);

  const fetchResults = async () => {
    try {
      const data = await api.getResults();
      setResults(data.results || NO_RESULTS_TEXT);
      setManualDatabase(typeof data.manual_database === 'boolean' ? data.manual_database : null);
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
      onClearPrompt?.();
      alert(data.message);
      // Refresh results to show empty state
      fetchResults();
    } catch (error) {
      console.error('Failed to clear submissions:', error);
      alert('Failed to clear submissions: ' + error.message);
    }
  };

  // Saves the language shown: the translation once it is complete, otherwise the original.
  const handleDownload = () => {
    if (!hasResults) {
      alert('No results to download');
      return;
    }

    const shownTranslation = translatedResults !== null
      ? { source: results, language: translation.language, content: translatedResults }
      : null;
    const download = viewedDownload(
      shownTranslation,
      results,
      `aggregator_results_${new Date().toISOString().slice(0, 10)}`,
    );
    downloadRawText(download.content, download.filename, null, 'brainstorm');
  };

  const handleProofCheck = () => {
    setProofActionMessage('');
    setProofCheckModalOpen(true);
  };

  const startProofCheck = async (runMode) => {
    try {
      setProofCheckStarting(true);
      await queueManualProofCheck({
        sourceType: 'brainstorm',
        sourceId: MANUAL_AGGREGATOR_PROOF_SOURCE_ID,
        scope: 'manual',
        runMode,
      });
      setProofActionMessage(`Queued ${runMode === 'loop_with_pruning' ? 'continuous' : 'one-round'} proof check for the manual Aggregator database.`);
      setProofCheckModalOpen(false);
    } catch (error) {
      setProofActionMessage(`Failed to queue proof check: ${error.message}`);
    } finally {
      setProofCheckStarting(false);
    }
  };

  const proofCheckState = getSourceState('brainstorm', MANUAL_AGGREGATOR_PROOF_SOURCE_ID, 'manual');
  const proofCheckLabel = proofCheckState?.status === 'queued'
    ? 'Queueing Proof Check...'
    : proofCheckState?.status === 'running'
      ? `Proof Check Running${proofCheckState.candidateCount ? ` (${proofCheckState.candidateCount})` : '...'}`
      : 'Search For More Mathematical Proofs';
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
          disabled={!proofCheckEnabled || isProofRunBusy(proofCheckState)}
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
        <TranslationLanguageSwitch translation={resultsTranslation} />
      </div>
      <TranslationPendingNotice translation={resultsTranslation} />

      {proofActionMessage && (
        <div className={`test-result-banner ${proofActionMessage.startsWith('Failed') ? 'test-result-banner--error' : 'test-result-banner--success'}`}>
          {proofActionMessage}
        </div>
      )}
      {proofCheckModalOpen && (
        <ProofCheckModeModal
          sourceTitle="the manual Aggregator database"
          busy={proofCheckStarting}
          error={proofActionMessage.startsWith('Failed') ? proofActionMessage : ''}
          onClose={() => setProofCheckModalOpen(false)}
          onConfirm={startProofCheck}
        />
      )}
      <ProofRunStatusControls
        run={proofCheckState}
        sourceLabel="Manual Aggregator database"
        onStop={stopProofRun}
      />

      <div className="results-container" ref={resultsRef}>
        <LatexRenderer
          documentId={translatedResults !== null ? `${RESULTS_TRANSLATION_KEY}:lang:${translation.language}` : undefined}
          content={prependDisclaimer(displayedResults, 'brainstorm')}
          className="results-latex-renderer"
          showToggle={false}
          showLatex={showLatex}
        />
      </div>
    </div>
  );
}
