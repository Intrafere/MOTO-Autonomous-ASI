/**
 * LiveResults - Displays accepted aggregator submissions with LaTeX rendering.
 */
import React, { useState, useEffect, useRef } from 'react';
import { api } from '../../services/api';
import { websocket } from '../../services/websocket';
import LatexRenderer from '../LatexRenderer';
import { prependDisclaimer } from '../../utils/disclaimerHelper';

export default function LiveResults() {
  const [results, setResults] = useState('');
  const [autoScroll, setAutoScroll] = useState(true);
  const [showLatex, setShowLatex] = useState(false); // Raw text by default for performance with large docs
  const resultsRef = useRef(null);

  useEffect(() => {
    fetchResults();
    const interval = setInterval(fetchResults, 5000);

    const unsubscribe = websocket.on('submission_accepted', () => {
      // Fetch new results when a submission is accepted
      setTimeout(fetchResults, 500);
    });

    return () => {
      clearInterval(interval);
      unsubscribe();
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
    if (!confirm('Are you sure you want to clear ALL accepted submissions? This cannot be undone!')) {
      return;
    }
    
    try {
      const data = await api.clearAllSubmissions();
      alert(data.message);
      // Refresh results to show empty state
      fetchResults();
    } catch (error) {
      console.error('Failed to clear submissions:', error);
      alert('Failed to clear submissions');
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
