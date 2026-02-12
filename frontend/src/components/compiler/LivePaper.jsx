import React, { useState, useEffect, useRef } from 'react';
import { compilerAPI } from '../../services/api';
import { websocket } from '../../services/websocket';
import LatexRenderer from '../LatexRenderer';
import { downloadRawText, downloadPDF, sanitizeFilename } from '../../utils/downloadHelpers';
import PaperCritiqueModal from '../PaperCritiqueModal';

function LivePaper() {
  const [paper, setPaper] = useState('');
  const [outline, setOutline] = useState('');
  const [wordCount, setWordCount] = useState(0);
  const [version, setVersion] = useState(0);
  const [autoScroll, setAutoScroll] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [previousVersions, setPreviousVersions] = useState([]);
  const [showVersions, setShowVersions] = useState(false);
  const [showLatex, setShowLatex] = useState(true);
  const [isGeneratingPDF, setIsGeneratingPDF] = useState(false);
  const paperContainerRef = useRef(null);
  
  // Critique modal state
  const [critiqueModalOpen, setCritiqueModalOpen] = useState(false);

  useEffect(() => {
    loadPaper();
    loadStatus();
    loadPreviousVersions();
    
    // Poll paper and status every 10 seconds
    const interval = setInterval(() => {
      loadPaper();
      loadStatus();
      loadPreviousVersions();
    }, 10000);

    // Listen for paper updates via WebSocket
    websocket.on('paper_updated', handlePaperUpdate);
    websocket.on('body_rewrite_started', handleBodyRewrite);

    return () => {
      clearInterval(interval);
      websocket.off('paper_updated', handlePaperUpdate);
      websocket.off('body_rewrite_started', handleBodyRewrite);
    };
  }, []);

  useEffect(() => {
    if (autoScroll && paperContainerRef.current) {
      paperContainerRef.current.scrollTop = paperContainerRef.current.scrollHeight;
    }
  }, [paper, autoScroll]);

  const [isRunning, setIsRunning] = useState(false);

  const loadPaper = async () => {
    try {
      const response = await compilerAPI.getPaper();
      setPaper(response.data.paper);
      setWordCount(response.data.word_count);
      setVersion(response.data.version);
      
      // Also load outline for downloads
      try {
        const outlineResponse = await compilerAPI.getOutline();
        setOutline(outlineResponse.data.outline || '');
      } catch (err) {
        console.error('Failed to load outline:', err);
        setOutline('');
      }
    } catch (error) {
      console.error('Failed to load paper:', error);
    }
  };

  const loadStatus = async () => {
    try {
      const response = await compilerAPI.getStatus();
      setIsRunning(response.data.is_running);
    } catch (error) {
      console.error('Failed to load status:', error);
    }
  };

  const handlePaperUpdate = (event) => {
    loadPaper(); // Refresh paper on update
  };

  const handleBodyRewrite = (event) => {
    // Reload previous versions when rewrite starts
    loadPreviousVersions();
    loadPaper();
  };

  const loadPreviousVersions = async () => {
    try {
      const response = await compilerAPI.getPreviousVersions();
      setPreviousVersions(response.data.previous_versions || []);
    } catch (error) {
      console.error('Failed to load previous versions:', error);
    }
  };

  const handleSaveDraft = async () => {
    setIsSaving(true);
    try {
      const response = await compilerAPI.savePaper();
      alert(response.data.message);
    } catch (error) {
      console.error('Failed to save paper:', error);
      alert('Failed to save paper: ' + error.message);
    } finally {
      setIsSaving(false);
    }
  };

  const handleGenerateFinal = async () => {
    if (!confirm('This will stop the compiler and save the final paper. Continue?')) {
      return;
    }

    try {
      // Stop compiler
      await compilerAPI.stop();
      
      // Save paper
      await handleSaveDraft();
      
      alert('Final paper generated and saved!');
    } catch (error) {
      console.error('Failed to generate final:', error);
      alert('Failed to generate final: ' + error.message);
    }
  };

  const handleClearPaper = async () => {
    const message = 'Are you sure you want to DELETE this paper and RESTART from scratch?\n\n' +
      'This will:\n' +
      '• Delete the current paper content\n' +
      '• Delete the current outline\n' +
      '• Clear all rejection/acceptance logs\n' +
      '• Restart from outline creation\n\n' +
      'This action CANNOT be undone!';
    
    if (!confirm(message)) {
      return;
    }

    try {
      const response = await compilerAPI.clearPaper();
      alert('Paper reset successfully. Starting fresh from outline creation.');
      // Refresh paper to show empty state
      loadPaper();
      loadStatus();
    } catch (error) {
      console.error('Failed to clear paper:', error);
      alert('Failed to clear paper: ' + error.message);
    }
  };

  const handleDownloadPDF = async () => {
    if (!paper || !paperContainerRef.current) {
      alert('No paper content available to download');
      return;
    }

    setIsGeneratingPDF(true);
    try {
      const element = paperContainerRef.current.querySelector('.latex-rendered-content') || 
                      paperContainerRef.current.querySelector('.paper-content-renderer');
      
      if (!element) {
        throw new Error('Could not find paper content element');
      }

      const metadata = {
        title: 'Compiler Paper',
        wordCount: wordCount,
        date: new Date().toLocaleDateString(),
        models: null // Single paper writer doesn't track this currently
      };
      
      const filename = sanitizeFilename('compiler_paper');
      await downloadPDF(element, metadata, filename, outline);
    } catch (error) {
      console.error('PDF generation error:', error);
      alert('Failed to generate PDF: ' + error.message);
    } finally {
      setIsGeneratingPDF(false);
    }
  };

  const handleDownloadRawText = () => {
    if (!paper) {
      alert('No paper content available to download');
      return;
    }
    
    const filename = sanitizeFilename('compiler_paper');
    downloadRawText(paper, filename, outline);
  };

  return (
    <div className="live-paper">
      <div className="paper-header">
        <h2>Live Paper</h2>
        <div className="paper-meta">
          <span className="word-count">{wordCount.toLocaleString()} words</span>
          <span className="version">v{version}</span>
        </div>
      </div>

      <div className="paper-controls">
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={autoScroll}
            onChange={(e) => setAutoScroll(e.target.checked)}
          />
          Auto-scroll
        </label>

        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={showLatex}
            onChange={(e) => setShowLatex(e.target.checked)}
          />
          LaTeX Rendering
        </label>

        <div className="button-group">
          <button
            onClick={handleDownloadPDF}
            className="btn btn-secondary"
            disabled={!paper || isGeneratingPDF}
            title="Download as PDF"
          >
            {isGeneratingPDF ? 'Generating...' : 'PDF'}
          </button>

          <button
            onClick={handleDownloadRawText}
            className="btn btn-secondary"
            disabled={!paper}
            title="Download as text"
          >
            Text
          </button>

          <button
            onClick={handleSaveDraft}
            className="btn btn-secondary"
            disabled={isSaving}
          >
            {isSaving ? 'Saving...' : 'Save Draft'}
          </button>

          <button
            onClick={handleGenerateFinal}
            className="btn btn-primary"
          >
            Generate Final
          </button>

          <button
            onClick={handleClearPaper}
            className="btn btn-danger"
            disabled={isRunning}
            title="Delete paper and restart from scratch"
          >
            🗑️ Delete & Retry
          </button>

          <button
            onClick={() => setCritiqueModalOpen(true)}
            className="btn-critique"
            disabled={!paper}
            title="Ask validator to critique this paper"
            style={{
              background: 'linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%)',
              border: 'none',
              color: '#fff',
              padding: '0.5rem 1rem',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: '500',
              fontSize: '0.85rem'
            }}
          >
            ⭐ Ask Validator to Critique
          </button>
        </div>
      </div>

      <div className="paper-container" ref={paperContainerRef}>
        {paper ? (
          <LatexRenderer 
            content={paper}
            className="paper-content-renderer"
            defaultRaw={!showLatex}
            showToggle={true}
          />
        ) : (
          <div className="paper-empty">
            {!isRunning ? (
              <>
                <h3>Compiler Not Started</h3>
                <p>The compiler has not been started yet.</p>
                <p>Go to the <strong>Compiler Interface</strong> tab to configure and start the compiler.</p>
                <p style={{marginTop: '1rem', fontSize: '0.9rem', color: '#666'}}>
                  The compiler will transform the aggregator's accepted submissions into a cohesive academic paper.
                </p>
              </>
            ) : (
              <>
                <h3>Paper Generation In Progress</h3>
                <p>The compiler is running and will begin generating paper content soon...</p>
                <p style={{marginTop: '1rem', fontSize: '0.9rem', color: '#666'}}>
                  Current mode: {wordCount === 0 ? 'Initializing outline' : 'Building paper'}
                </p>
              </>
            )}
          </div>
        )}
      </div>

      {previousVersions.length > 0 && (
        <div className="previous-versions-section">
          <button 
            className="btn btn-secondary"
            onClick={() => setShowVersions(!showVersions)}
            style={{ marginTop: '1rem', marginBottom: '0.5rem' }}
          >
            📜 Previous Versions ({previousVersions.length})
          </button>
          
          {showVersions && (
            <div className="versions-list">
              {previousVersions.map(v => (
                <div key={v.version} className="version-card" style={{
                  border: '1px solid #444',
                  borderRadius: '4px',
                  padding: '1rem',
                  marginBottom: '1rem',
                  backgroundColor: '#2a2a2a'
                }}>
                  <h3 style={{ color: '#ffd700', marginBottom: '0.5rem' }}>
                    Version {v.version}: {v.title}
                  </h3>
                  
                  <div className="version-body" style={{ marginBottom: '1rem' }}>
                    <h4 style={{ color: '#4CAF50', marginBottom: '0.5rem' }}>Body Section:</h4>
                    <pre style={{
                      backgroundColor: '#1a1a1a',
                      padding: '1rem',
                      borderRadius: '4px',
                      maxHeight: '400px',
                      overflow: 'auto',
                      fontSize: '0.85rem',
                      whiteSpace: 'pre-wrap'
                    }}>{v.body}</pre>
                  </div>
                  
                  <div className="version-critique">
                    <h4 style={{ color: '#ff6b6b', marginBottom: '0.5rem' }}>Critique Feedback (triggered rewrite):</h4>
                    <pre style={{
                      backgroundColor: '#1a1a1a',
                      padding: '1rem',
                      borderRadius: '4px',
                      maxHeight: '300px',
                      overflow: 'auto',
                      fontSize: '0.85rem',
                      whiteSpace: 'pre-wrap'
                    }}>{v.critique_feedback}</pre>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="paper-footer">
        <p className="info-text">
          This paper is being constructed in real-time from the aggregator database.
          The compiler follows a sequential workflow: outline creation → paper construction → review → rigor enhancement.
        </p>
      </div>

      {/* Critique Modal */}
      <PaperCritiqueModal
        isOpen={critiqueModalOpen}
        onClose={() => setCritiqueModalOpen(false)}
        paperType="compiler_paper"
        paperId={null}
        paperTitle="Compiler Paper"
        onGenerateCritique={(customPrompt, validatorConfig) => compilerAPI.generateCritique(customPrompt, validatorConfig)}
        onGetCritiques={() => compilerAPI.getCritiques()}
      />
    </div>
  );
}

export default LivePaper;
