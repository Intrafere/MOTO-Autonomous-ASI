/**
 * LiveTier3Progress - Displays Tier 3 final answer being generated in real-time.
 * 
 * Shows:
 * - Overall Tier 3 status (phase, format, certainty)
 * - For long-form: chapter list with progress
 * - Current paper/chapter content being written
 * - Real-time updates via WebSocket events
 */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { websocket } from '../../services/websocket';
import LatexRenderer from '../LatexRenderer';
import { downloadRawText, downloadPDF, sanitizeFilename } from '../../utils/downloadHelpers';

const LiveTier3Progress = ({ api, status }) => {
  const [paperData, setPaperData] = useState(null);
  const [volumeProgress, setVolumeProgress] = useState(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [isExpanded, setIsExpanded] = useState(true);
  const [isDownloadingPdf, setIsDownloadingPdf] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const containerRef = useRef(null);
  const contentRef = useRef(null);

  // Check banner shimmer setting from localStorage
  const getBannerShimmerEnabled = () => {
    const saved = localStorage.getItem('banner_shimmer_enabled');
    return saved !== null ? JSON.parse(saved) : true;
  };

  // Load paper progress from API
  const loadPaperProgress = useCallback(async () => {
    if (!api || status?.current_tier !== 'tier3_final_answer') return;
    
    try {
      const data = await api.getCurrentPaperProgress();
      if (data.tier === 'tier3') {
        setPaperData(data);
      }
    } catch (error) {
      console.error('Failed to load Tier 3 paper progress:', error);
    }
  }, [api, status?.current_tier]);

  // Load volume progress for long-form answers
  const loadVolumeProgress = useCallback(async () => {
    if (!api || status?.current_tier !== 'tier3_final_answer') return;
    
    try {
      const data = await api.getVolumeProgress();
      if (data.is_long_form) {
        setVolumeProgress(data);
      }
    } catch (error) {
      console.error('Failed to load volume progress:', error);
    }
  }, [api, status?.current_tier]);

  // Load all data
  const loadAllData = useCallback(async () => {
    await Promise.all([loadPaperProgress(), loadVolumeProgress()]);
  }, [loadPaperProgress, loadVolumeProgress]);

  useEffect(() => {
    if (status?.current_tier !== 'tier3_final_answer') {
      setPaperData(null);
      setVolumeProgress(null);
      return;
    }

    // Initial load
    loadAllData();

    // Poll every 5 seconds
    const interval = setInterval(loadAllData, 5000);

    // Listen for tier3 WebSocket events
    const unsubscribers = [
      websocket.on('tier3_chapter_started', loadAllData),
      websocket.on('tier3_chapter_complete', loadAllData),
      websocket.on('tier3_paper_started', loadAllData),
      websocket.on('tier3_phase_changed', loadAllData),
      websocket.on('tier3_format_selected', loadAllData),
      websocket.on('tier3_volume_organized', loadAllData),
      websocket.on('paper_updated', loadPaperProgress),
      websocket.on('tier3_paper_reset', loadAllData),
      websocket.on('tier3_chapter_reset', loadAllData),
      websocket.on('tier2_paper_reset', loadAllData),
    ];

    return () => {
      clearInterval(interval);
      unsubscribers.forEach(unsub => unsub());
    };
  }, [status?.current_tier, loadAllData, loadPaperProgress]);

  // Auto-scroll effect
  useEffect(() => {
    if (autoScroll && containerRef.current && paperData?.content) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [paperData?.content, autoScroll]);

  const handleDownloadRaw = () => {
    if (!paperData?.content) return;
    const filename = sanitizeFilename(paperData.title || 'tier3_final_answer');
    downloadRawText(paperData.content, filename, paperData.outline);
  };

  const handleDownloadPdf = async () => {
    if (!paperData?.content || !contentRef.current) return;
    
    setIsDownloadingPdf(true);
    try {
      const filename = sanitizeFilename(paperData.title || 'tier3_final_answer');
      const metadata = {
        title: paperData.title || 'Final Answer',
        wordCount: paperData.word_count,
        date: new Date().toLocaleDateString(),
        models: 'Tier 3 Final Answer'
      };
      
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
    // Determine what will be reset based on mode
    const isShortForm = volumeProgress?.is_long_form === false;
    const isLongForm = volumeProgress?.is_long_form === true;
    
    let message = 'Are you sure you want to DELETE this paper and RESTART?\n\n';
    
    if (isShortForm) {
      message += 'This will:\n' +
        '• Delete the current paper content\n' +
        '• Delete the current outline\n' +
        '• Clear the selected title\n' +
        '• Restart from title selection\n\n';
    } else if (isLongForm) {
      const currentChapter = volumeProgress?.current_writing_chapter || 0;
      message += 'This will:\n' +
        `• Delete the current chapter (Chapter ${currentChapter}) content\n` +
        '• Delete the chapter outline\n' +
        '• Reset chapter status to pending\n' +
        '• Keep all other chapters intact\n' +
        '• Restart writing this chapter\n\n';
    } else {
      message += 'This will:\n' +
        '• Delete the current paper content\n' +
        '• Delete the current outline\n' +
        '• Restart from outline creation\n\n';
    }
    
    message += 'This action CANNOT be undone!';
    
    if (!confirm(message)) {
      return;
    }
    
    setIsResetting(true);
    try {
      const response = await api.resetAutonomousPaper();
      alert(`Paper reset successfully. ${response.data.message}`);
      // Trigger reload of current paper progress
      window.dispatchEvent(new CustomEvent('autonomous-paper-reset'));
      await loadAllData();
    } catch (error) {
      console.error('Failed to reset paper:', error);
      alert('Failed to reset paper: ' + error.message);
    } finally {
      setIsResetting(false);
    }
  };

  // Get status badge
  const getStatusBadge = () => {
    const tierStatus = paperData?.tier3_status || status?.tier3?.status;
    const statusMap = {
      'idle': { label: 'Initializing', color: 'gray' },
      'assessing': { label: 'Assessing Certainty', color: 'blue' },
      'format_selecting': { label: 'Selecting Format', color: 'blue' },
      'organizing_volume': { label: 'Organizing Volume', color: 'purple' },
      'writing': { label: 'Writing', color: 'green' },
      'complete': { label: 'Complete', color: 'gold' }
    };
    
    const info = statusMap[tierStatus] || { label: tierStatus || 'Unknown', color: 'gray' };
    return (
      <span className={`tier3-status-badge tier3-status-${info.color}`}>
        {info.label}
      </span>
    );
  };

  // Get format badge
  const getFormatBadge = () => {
    const format = paperData?.tier3_format || status?.tier3?.format;
    if (!format) return null;
    
    return (
      <span className={`tier3-format-badge tier3-format-${format}`}>
        {format === 'short_form' ? '📄 Short Form' : '📚 Long Form Volume'}
      </span>
    );
  };

  // Render chapter list for long-form
  const renderChapterList = () => {
    if (!volumeProgress?.is_long_form || !volumeProgress.chapters) return null;

    return (
      <div className="tier3-chapter-list">
        <h4>
          Volume Chapters ({volumeProgress.completed_chapters}/{volumeProgress.total_chapters} complete)
        </h4>
        <div className="chapter-items">
          {volumeProgress.chapters.map(ch => (
            <div 
              key={ch.order} 
              className={`chapter-item chapter-${ch.status} ${ch.order === paperData?.tier3_chapter?.order ? 'chapter-active' : ''}`}
            >
              <span className="chapter-order">{ch.order}</span>
              <span className="chapter-type">{ch.type.replace('_', ' ')}</span>
              <span className="chapter-title">{ch.title}</span>
              <span className={`chapter-status chapter-status-${ch.status}`}>
                {ch.status === 'complete' ? '✓' : ch.status === 'writing' ? '✍' : '○'}
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // Don't render if not in tier3 or no data
  if (status?.current_tier !== 'tier3_final_answer') {
    return null;
  }

  return (
    <div className="live-tier3-progress">
      <div className="live-tier3-header" onClick={() => setIsExpanded(!isExpanded)}>
        <h3>
          🎯 Tier 3 Final Answer In Progress
          <span className="toggle-icon">{isExpanded ? '▼' : '▶'}</span>
        </h3>
        <div className="tier3-meta">
          {getStatusBadge()}
          {getFormatBadge()}
          {paperData?.word_count > 0 && (
            <span className="word-count">{paperData.word_count?.toLocaleString()} words</span>
          )}
        </div>
      </div>

      {isExpanded && (
        <>
          {/* Chapter Progress (Long Form) */}
          {renderChapterList()}

          {/* Current Writing Status */}
          {paperData?.tier3_chapter && (
            <div className="tier3-current-writing">
              <span className="writing-indicator">✍ Currently Writing:</span>
              <span className="writing-chapter">
                Chapter {paperData.tier3_chapter.order}: {paperData.tier3_chapter.title}
              </span>
              <span className="writing-type">({paperData.tier3_chapter.type.replace('_', ' ')})</span>
            </div>
          )}

          {/* Controls */}
          {paperData?.content && (
            <div className="live-tier3-controls">
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
                  Download Raw Text
                </button>
                <button
                  className="btn-download-pdf"
                  onClick={handleDownloadPdf}
                  disabled={!paperData?.content || isDownloadingPdf}
                  title="Download as PDF"
                >
                  {isDownloadingPdf ? 'Generating...' : 'Download PDF'}
                </button>
                <button
                  className="tier3-btn-reset"
                  onClick={handleResetPaper}
                  disabled={isResetting || !status.is_active}
                  title="Delete current paper and restart"
                >
                  {isResetting ? '🔄 Resetting...' : '🗑️ Delete & Retry'}
                </button>
              </div>
            </div>
          )}

          {/* Paper Content */}
          <div className="live-tier3-container" ref={containerRef}>
            {paperData?.content ? (
              <div className="paper-section" ref={contentRef}>
                <h4>{paperData.title || 'Final Answer Content'}</h4>
                <LatexRenderer 
                  content={
                    paperData.outline
                      ? `${paperData.outline}\n\n${'='.repeat(80)}\n\n${paperData.content}`
                      : paperData.content
                  }
                  className="live-tier3-latex-renderer"
                  defaultRaw={true}
                  showToggle={true}
                />
              </div>
            ) : (
              <div className="tier3-empty">
                <div className="tier3-loading">
                  <span className="loading-spinner">⟳</span>
                  {paperData?.tier3_status === 'assessing' && 'Assessing certainty from existing papers...'}
                  {paperData?.tier3_status === 'format_selecting' && 'Selecting answer format...'}
                  {paperData?.tier3_status === 'organizing_volume' && 'Organizing volume structure...'}
                  {(!paperData?.tier3_status || paperData?.tier3_status === 'writing') && 'Preparing to write final answer...'}
                </div>
              </div>
            )}
          </div>
        </>
      )}

      <style>{`
        .live-tier3-progress {
          background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
          border: 1px solid #ffd700;
          border-radius: 8px;
          margin: 1rem 0;
          overflow: hidden;
        }

        .live-tier3-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.75rem 1rem;
          background: linear-gradient(90deg, rgba(255, 215, 0, 0.15) 0%, rgba(255, 215, 0, 0.05) 100%);
          cursor: pointer;
          border-bottom: 1px solid rgba(255, 215, 0, 0.3);
        }

        .live-tier3-header h3 {
          margin: 0;
          color: #ffd700;
          font-size: 1rem;
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .toggle-icon {
          font-size: 0.75rem;
          opacity: 0.7;
          margin-left: 0.5rem;
        }

        .tier3-meta {
          display: flex;
          gap: 0.75rem;
          align-items: center;
        }

        .tier3-status-badge {
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          font-size: 0.75rem;
          font-weight: 600;
          text-transform: uppercase;
        }

        .tier3-status-gray { background: #4a4a4a; color: #ccc; }
        .tier3-status-blue { background: #1e40af; color: #93c5fd; }
        .tier3-status-purple { background: #6b21a8; color: #d8b4fe; }
        .tier3-status-green { background: #166534; color: #86efac; }
        .tier3-status-gold { background: #854d0e; color: #fcd34d; }

        .tier3-format-badge {
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          font-size: 0.75rem;
          background: rgba(255, 255, 255, 0.1);
          color: #d4d4d4;
        }

        .word-count {
          font-size: 0.75rem;
          color: #9ca3af;
        }

        .tier3-chapter-list {
          padding: 0.75rem 1rem;
          border-bottom: 1px solid rgba(255, 215, 0, 0.2);
          background: rgba(0, 0, 0, 0.2);
        }

        .tier3-chapter-list h4 {
          margin: 0 0 0.5rem 0;
          color: #d4d4d4;
          font-size: 0.85rem;
        }

        .chapter-items {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }

        .chapter-item {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.35rem 0.5rem;
          border-radius: 4px;
          font-size: 0.8rem;
          background: rgba(255, 255, 255, 0.05);
          transition: all 0.2s;
        }

        .chapter-item.chapter-active {
          background: rgba(255, 215, 0, 0.2);
          border: 1px solid rgba(255, 215, 0, 0.4);
        }

        .chapter-item.chapter-complete {
          opacity: 0.7;
        }

        .chapter-order {
          width: 1.5rem;
          height: 1.5rem;
          display: flex;
          align-items: center;
          justify-content: center;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 50%;
          font-weight: 600;
          color: #ffd700;
        }

        .chapter-type {
          color: #9ca3af;
          font-size: 0.7rem;
          text-transform: uppercase;
          min-width: 80px;
        }

        .chapter-title {
          flex: 1;
          color: #e5e5e5;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .chapter-status {
          width: 1.25rem;
          text-align: center;
        }

        .chapter-status-complete { color: #86efac; }
        .chapter-status-writing { color: #fcd34d; ${getBannerShimmerEnabled() ? 'animation: pulse 1s infinite;' : ''} }
        .chapter-status-pending { color: #6b7280; }

        .tier3-current-writing {
          padding: 0.5rem 1rem;
          background: rgba(255, 215, 0, 0.1);
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.85rem;
        }

        .writing-indicator {
          color: #fcd34d;
          ${getBannerShimmerEnabled() ? 'animation: pulse 1.5s infinite;' : ''}
        }

        .writing-chapter {
          color: #e5e5e5;
          font-weight: 500;
        }

        .writing-type {
          color: #9ca3af;
          font-size: 0.75rem;
        }

        .live-tier3-controls {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.5rem 1rem;
          background: rgba(0, 0, 0, 0.2);
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .checkbox-label {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          color: #d4d4d4;
          font-size: 0.85rem;
          cursor: pointer;
        }

        .download-buttons {
          display: flex;
          gap: 0.5rem;
        }

        .btn-download-raw, .btn-download-pdf {
          padding: 0.35rem 0.75rem;
          border-radius: 4px;
          font-size: 0.75rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .btn-download-raw {
          background: transparent;
          border: 1px solid #6b7280;
          color: #d4d4d4;
        }

        .btn-download-raw:hover:not(:disabled) {
          background: rgba(107, 114, 128, 0.2);
        }

        .btn-download-pdf {
          background: #ffd700;
          border: none;
          color: #1a1a2e;
          font-weight: 600;
        }

        .btn-download-pdf:hover:not(:disabled) {
          background: #ffed4e;
        }

        .btn-download-raw:disabled, .btn-download-pdf:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .live-tier3-container {
          max-height: 600px;
          overflow-y: auto;
          padding: 1rem;
        }

        .paper-section h4 {
          margin: 0 0 1rem 0;
          color: #ffd700;
          font-size: 0.9rem;
          border-bottom: 1px solid rgba(255, 215, 0, 0.2);
          padding-bottom: 0.5rem;
        }

        .tier3-empty {
          padding: 2rem;
          text-align: center;
        }

        .tier3-loading {
          color: #9ca3af;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.5rem;
        }

        .loading-spinner {
          animation: spin 1s linear infinite;
          font-size: 1.25rem;
        }

        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
};

export default LiveTier3Progress;

