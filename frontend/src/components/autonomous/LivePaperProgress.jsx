/**
 * LivePaperProgress - Displays paper being compiled in real-time during Tier 2.
 */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { websocket } from '../../services/websocket';
import LatexRenderer from '../LatexRenderer';
import { downloadRawText, downloadPDF, sanitizeFilename } from '../../utils/downloadHelpers';

const LivePaperProgress = ({ api, isCompiling }) => {
  const [paperData, setPaperData] = useState(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [isExpanded, setIsExpanded] = useState(true);
  const [isDownloadingPdf, setIsDownloadingPdf] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const containerRef = useRef(null);
  const contentRef = useRef(null);

  // Memoize loadPaperProgress with useCallback
  const loadPaperProgress = useCallback(async () => {
    if (!api || !isCompiling) return;
    
    try {
      const data = await api.getCurrentPaperProgress();
      setPaperData(data);
    } catch (error) {
      console.error('Failed to load paper progress:', error);
    }
  }, [api, isCompiling]);  // Dependencies: api and isCompiling

  useEffect(() => {
    if (!isCompiling) {
      setPaperData(null);
      return;
    }

    // Initial load
    loadPaperProgress();

    // Poll every 5 seconds
    const interval = setInterval(loadPaperProgress, 5000);

    // Listen for paper updates and reset events
    const unsubscribers = [
      websocket.on('paper_updated', loadPaperProgress),
      websocket.on('tier2_paper_reset', loadPaperProgress),
      websocket.on('paper_cleared', loadPaperProgress),
    ];

    return () => {
      clearInterval(interval);
      unsubscribers.forEach(unsub => unsub());
    };
  }, [isCompiling, loadPaperProgress]);  // Now includes loadPaperProgress

  useEffect(() => {
    if (autoScroll && containerRef.current && paperData?.content) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [paperData?.content, autoScroll]);

  const handleDownloadRaw = () => {
    if (!paperData?.content) return;
    const filename = sanitizeFilename(paperData.title || paperData.paper_id || 'paper');
    downloadRawText(paperData.content, filename, paperData.outline);
  };

  const handleDownloadPdf = async () => {
    if (!paperData?.content || !contentRef.current) return;
    
    setIsDownloadingPdf(true);
    try {
      const filename = sanitizeFilename(paperData.title || paperData.paper_id || 'paper');
      const metadata = {
        title: paperData.title || 'Untitled Paper',
        wordCount: paperData.word_count,
        date: new Date().toLocaleDateString(),
        models: 'Autonomous Research'
      };
      
      // Find the rendered content element
      const renderedElement = contentRef.current.querySelector('.latex-rendered-content') || 
                             contentRef.current.querySelector('.latex-raw-content') ||
                             contentRef.current;
      
      await downloadPDF(renderedElement, metadata, filename, paperData.outline);
    } catch (error) {
      console.error('PDF download failed:', error);
      alert('Failed to generate PDF. Please try downloading as raw text instead.');
    } finally {
      setIsDownloadingPdf(false);
    }
  };

  const handleResetPaper = async () => {
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
    
    setIsResetting(true);
    try {
      const response = await api.resetAutonomousPaper();
      alert(`Paper reset successfully. ${response.data.message}`);
      await loadPaperProgress();
    } catch (error) {
      console.error('Failed to reset paper:', error);
      alert('Failed to reset paper: ' + error.message);
    } finally {
      setIsResetting(false);
    }
  };

  if (!isCompiling || !paperData) {
    return null;
  }

  return (
    <div className="live-paper-progress">
      <div className="live-paper-header" onClick={() => setIsExpanded(!isExpanded)}>
        <h3>
          Paper In Progress: {paperData.title}
          <span className="toggle-icon">{isExpanded ? '▼' : '▶'}</span>
        </h3>
        <div className="paper-meta">
          <span className="paper-id">{paperData.paper_id}</span>
          <span className="word-count">{paperData.word_count?.toLocaleString()} words</span>
        </div>
      </div>

      {isExpanded && (
        <>
          <div className="live-paper-controls">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={autoScroll}
                onChange={(e) => setAutoScroll(e.target.checked)}
              />
              Auto-scroll
            </label>
            
            <div className="download-buttons">
              <button
                className="btn-download-raw"
                onClick={handleDownloadRaw}
                disabled={!paperData?.content}
                title="Download as plain text"
              >
                Raw Text
              </button>
              <button
                className="btn-download-pdf"
                onClick={handleDownloadPdf}
                disabled={!paperData?.content || isDownloadingPdf}
                title="Download as PDF"
              >
                {isDownloadingPdf ? 'Generating...' : 'PDF'}
              </button>
              <button
                className="tier3-btn-reset"
                onClick={handleResetPaper}
                disabled={isResetting || !isCompiling}
                title="Delete paper and restart from scratch"
              >
                {isResetting ? '🔄 Resetting...' : '🗑️ Delete & Retry'}
              </button>
            </div>
          </div>

          <div className="live-paper-container" ref={containerRef}>
            {paperData.content ? (
              <div className="paper-section" ref={contentRef}>
                <h4>Paper Content</h4>
                <LatexRenderer 
                  content={
                    paperData.outline
                      ? `${paperData.outline}\n\n${'='.repeat(80)}\n\n${paperData.content}`
                      : paperData.content
                  }
                  className="live-paper-latex-renderer"
                  defaultRaw={true}
                  showToggle={true}
                />
              </div>
            ) : (
              <div className="paper-empty">
                Initializing paper compilation...
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default LivePaperProgress;

