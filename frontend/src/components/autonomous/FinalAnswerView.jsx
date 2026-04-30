/**
 * FinalAnswerView - Displays the Tier 3 Final Answer progress and content.
 * Shows both short-form (single paper) and long-form (volume) answers.
 */
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { websocket } from '../../services/websocket';
import ArchiveViewerModal from './ArchiveViewerModal';
import LatexRenderer from '../LatexRenderer';
import { downloadRawText, downloadPDFViaBackend, sanitizeFilename } from '../../utils/downloadHelpers';
import PaperCritiqueModal from '../PaperCritiqueModal';
import { autonomousAPI } from '../../services/api';
import { getRuntimeDataPath } from '../../utils/runtimeConfig';
import './AutonomousResearch.css';

const FinalAnswerView = ({ api, isRunning, status }) => {
  const [finalAnswerData, setFinalAnswerData] = useState(null);
  const [volumeContent, setVolumeContent] = useState(null);
  const [shortFormPaper, setShortFormPaper] = useState(null);
  const [isExpanded, setIsExpanded] = useState(true);
  const [autoScroll, setAutoScroll] = useState(true);
  const [activeSection, setActiveSection] = useState('overview');
  const [isRegenerating, setIsRegenerating] = useState(false);
  const [showArchiveModal, setShowArchiveModal] = useState(false);
  const [showLatex, setShowLatex] = useState(false); // Raw text by default for performance with large docs
  const [isGeneratingPDF, setIsGeneratingPDF] = useState(false);
  const containerRef = useRef(null);
  
  // Critique modal state
  const [critiqueModalOpen, setCritiqueModalOpen] = useState(false);

  // Check banner shimmer setting from localStorage
  const getBannerShimmerEnabled = () => {
    const saved = localStorage.getItem('banner_shimmer_enabled');
    return saved !== null ? JSON.parse(saved) : true;
  };

  // Handle regenerate final answer
  const handleRegenerate = async (e) => {
    e.stopPropagation(); // Prevent header collapse
    
    if (!window.confirm('Regenerate Final Answer?\n\nThis will delete the current final answer and trigger a new Tier 3 generation with all completed papers.\n\nAre you sure?')) {
      return;
    }
    
    setIsRegenerating(true);
    try {
      // Call force tier 3 with skip_incomplete mode
      const response = await api.forceTier3('skip_incomplete');
      if (response.success) {
        // Success - reload status to show progress
        setTimeout(() => {
          loadFinalAnswerStatus();
          setIsRegenerating(false);
        }, 1000);
      }
    } catch (error) {
      alert(`Failed to regenerate final answer: ${error.details || error.message}`);
      setIsRegenerating(false);
    }
  };

  // Handle download raw text
  const handleDownloadRaw = (e) => {
    e.stopPropagation();
    
    let content = '';
    let title = '';
    
    if (finalAnswerData?.answer_format === 'short_form' && shortFormPaper) {
      content = shortFormPaper.content || '';
      title = shortFormPaper.title || 'Final_Answer_Short_Form';
    } else if (finalAnswerData?.answer_format === 'long_form' && volumeContent) {
      content = volumeContent.content || '';
      title = volumeContent.title || finalAnswerData?.volume?.volume_title || 'Final_Answer_Volume';
    } else {
      alert('Content not loaded yet. Please wait for the final answer to be generated.');
      return;
    }
    
    const filename = sanitizeFilename(`Final_Answer_${title}`);
    downloadRawText(content, filename, null);
  };

  // Handle download PDF
  const handleDownloadPDF = async (e) => {
    e.stopPropagation();

    // Use already-loaded state, or fetch now and use the returned data directly
    // (can't rely on React state after await since state updates are async)
    let resolvedShortForm = shortFormPaper;
    let resolvedVolume = volumeContent;

    if (!resolvedVolume && !resolvedShortForm) {
      const loaded = await loadFinalAnswerContent();
      if (loaded?.type === 'short_form') resolvedShortForm = loaded.data;
      if (loaded?.type === 'long_form') resolvedVolume = loaded.data;
    }

    let rawContent = '';
    let metadata = {};
    let filename = '';

    if (finalAnswerData?.answer_format === 'short_form' && resolvedShortForm) {
      rawContent = resolvedShortForm.content || '';
      metadata = {
        title: resolvedShortForm.title || 'Final Answer - Short Form',
        wordCount: resolvedShortForm.word_count,
        date: new Date().toLocaleDateString(),
        models: resolvedShortForm.model_usage ? Object.keys(resolvedShortForm.model_usage).join(', ') : null,
      };
      filename = sanitizeFilename(`Final_Answer_${resolvedShortForm.title}`);
    } else if (finalAnswerData?.answer_format === 'long_form' && resolvedVolume) {
      rawContent = resolvedVolume.content || '';
      metadata = {
        title: resolvedVolume.title || finalAnswerData?.volume?.volume_title || 'Final Answer - Volume',
        wordCount: resolvedVolume.word_count,
        date: new Date().toLocaleDateString(),
        models: null,
      };
      filename = sanitizeFilename(`Final_Answer_${resolvedVolume.title || 'Volume'}`);
    } else {
      alert('Content not available yet. Please wait for the final answer to be generated.');
      return;
    }

    await downloadPDFViaBackend(
      rawContent,
      metadata,
      filename,
      null,
      () => setIsGeneratingPDF(true),
      () => setIsGeneratingPDF(false),
      (error) => {
        setIsGeneratingPDF(false);
        console.error('PDF generation error:', error);
        alert(`PDF generation failed: ${error.message}`);
      },
    );
  };

  // Load final answer status (metadata only - NOT content)
  const loadFinalAnswerStatus = useCallback(async () => {
    if (!api) return;
    
    try {
      const data = await api.getFinalAnswerStatus();
      setFinalAnswerData(data);
    } catch (error) {
      console.error('Failed to load final answer status:', error);
    }
  }, [api]);
  
  // Load content on demand (only when needed). Returns the loaded content for immediate use.
  const loadFinalAnswerContent = useCallback(async () => {
    if (!api || !finalAnswerData) return null;
    
    try {
      if (finalAnswerData.answer_format === 'long_form' && finalAnswerData.volume && !volumeContent) {
        console.log('Loading volume content on demand...');
        const volume = await api.getFinalAnswerVolume();
        setVolumeContent(volume);
        return { type: 'long_form', data: volume };
      } else if (finalAnswerData.answer_format === 'short_form' && finalAnswerData.short_form_paper_id && !shortFormPaper) {
        console.log('Loading short form paper content on demand...');
        const paper = await api.getFinalAnswerPaper();
        setShortFormPaper(paper);
        return { type: 'short_form', data: paper };
      }
    } catch (error) {
      console.error('Failed to load content:', error);
    }
  }, [api, finalAnswerData, volumeContent, shortFormPaper]);

  // Initial load and polling
  useEffect(() => {
    loadFinalAnswerStatus();
    
    // Only poll when NOT complete - stop wasting resources on finished answers!
    if (finalAnswerData?.status !== 'complete') {
      const interval = setInterval(loadFinalAnswerStatus, 5000);
      return () => clearInterval(interval);
    }
  }, [loadFinalAnswerStatus, finalAnswerData?.status]);

  // WebSocket event listeners for Tier 3 events
  useEffect(() => {
    const unsubscribers = [];
    
    unsubscribers.push(websocket.on('tier3_started', () => {
      loadFinalAnswerStatus();
    }));
    
    unsubscribers.push(websocket.on('tier3_certainty_assessed', () => {
      loadFinalAnswerStatus();
    }));
    
    unsubscribers.push(websocket.on('tier3_format_selected', () => {
      loadFinalAnswerStatus();
    }));
    
    unsubscribers.push(websocket.on('tier3_volume_organized', () => {
      loadFinalAnswerStatus();
    }));
    
    unsubscribers.push(websocket.on('tier3_chapter_complete', () => {
      loadFinalAnswerStatus();
    }));
    
    unsubscribers.push(websocket.on('tier3_complete', () => {
      loadFinalAnswerStatus();
    }));
    
    unsubscribers.push(websocket.on('paper_updated', () => {
      // Refresh if tier 3 is active
      if (finalAnswerData?.is_active) {
        loadFinalAnswerStatus();
      }
    }));
    
    return () => {
      unsubscribers.forEach(unsub => unsub());
    };
  }, [loadFinalAnswerStatus, finalAnswerData?.is_active]);

  // Auto-scroll effect
  useEffect(() => {
    if (autoScroll && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [volumeContent, shortFormPaper, autoScroll]);
  
  // Load content on demand when user clicks "Content" tab
  useEffect(() => {
    if (activeSection === 'content' && finalAnswerData && (finalAnswerData.is_active || finalAnswerData.status === 'complete')) {
      loadFinalAnswerContent();
    }
  }, [activeSection, loadFinalAnswerContent, finalAnswerData]);

  // Get status badge class and text
  const getStatusBadge = () => {
    if (!finalAnswerData) return { class: 'status-idle', text: 'Pending' };
    
    if (finalAnswerData.status === 'complete') {
      return { class: 'status-complete', text: 'FINAL ANSWER' };
    }
    if (finalAnswerData.is_active) {
      return { class: 'status-active', text: 'FINAL ANSWER IN PROGRESS' };
    }
    return { class: 'status-pending', text: 'Awaiting Tier 3 Trigger' };
  };

  const statusBadge = getStatusBadge();

  // Get certainty level display
  const getCertaintyDisplay = (level) => {
    const displays = {
      'totally_answered': { icon: '✓', color: '#2ecc71', text: 'Can Be Totally Answered' },
      'partially_answered': { icon: '◐', color: '#f39c12', text: 'Partially Answerable' },
      'no_answer_known': { icon: '?', color: '#e74c3c', text: 'No Answer Known' },
      'appears_impossible': { icon: '✗', color: '#c0392b', text: 'Appears Impossible' },
      'other': { icon: '○', color: '#95a5a6', text: 'Other' }
    };
    return displays[level] || displays['other'];
  };

  // Get chapter status badge
  const getChapterStatusBadge = (status) => {
    switch (status) {
      case 'complete':
        return <span className="chapter-status chapter-complete">✓ Complete</span>;
      case 'writing':
        return <span className="chapter-status chapter-writing">Writing</span>;
      case 'pending':
      default:
        return <span className="chapter-status chapter-pending">○ Pending</span>;
    }
  };

  // Render certainty assessment section
  const renderCertaintyAssessment = () => {
    if (!finalAnswerData?.certainty_assessment) return null;
    
    const assessment = finalAnswerData.certainty_assessment;
    const certaintyDisplay = getCertaintyDisplay(assessment.certainty_level);
    
    return (
      <div className="tier3-section certainty-section">
        <h4>Certainty Assessment</h4>
        <div className="certainty-display">
          <div className="certainty-level" style={{ borderColor: certaintyDisplay.color }}>
            <span className="certainty-icon" style={{ color: certaintyDisplay.color }}>
              {certaintyDisplay.icon}
            </span>
            <span className="certainty-text">{certaintyDisplay.text}</span>
          </div>
          <div className="certainty-details">
            <p><strong>Assessment:</strong></p>
            <pre className="certainty-reasoning">{assessment.reasoning}</pre>
            {assessment.known_certainties && assessment.known_certainties.length > 0 && (
              <div className="known-certainties">
                <p><strong>Known Certainties:</strong></p>
                <ul>
                  {assessment.known_certainties.map((cert, idx) => (
                    <li key={idx}>{cert}</li>
                  ))}
                </ul>
              </div>
            )}
            {assessment.knowledge_gaps && assessment.knowledge_gaps.length > 0 && (
              <div className="knowledge-gaps">
                <p><strong>Knowledge Gaps:</strong></p>
                <ul>
                  {assessment.knowledge_gaps.map((gap, idx) => (
                    <li key={idx}>{gap}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  // Render format selection section
  const renderFormatSelection = () => {
    if (!finalAnswerData?.answer_format) return null;
    
    return (
      <div className="tier3-section format-section">
        <h4>Answer Format</h4>
        <div className={`format-badge ${finalAnswerData.answer_format}`}>
          {finalAnswerData.answer_format === 'short_form' ? (
            <>Short Form (Single Paper)</>
          ) : (
            <>Long Form (Volume Collection)</>
          )}
        </div>
      </div>
    );
  };

  // Render volume organization section (long form)
  const renderVolumeOrganization = () => {
    if (finalAnswerData?.answer_format !== 'long_form' || !finalAnswerData?.volume) {
      return null;
    }
    
    const volume = finalAnswerData.volume;
    
    return (
      <div className="tier3-section volume-section">
        <h4>Volume Organization</h4>
        <div className="volume-header">
          <h3 className="volume-title">{volume.volume_title}</h3>
          {finalAnswerData?.sample_label && (
            <div className="sample-label" style={{ color: '#FF6700', fontSize: '0.9em', fontWeight: 'bold', marginTop: '0.5em' }}>
              {finalAnswerData.sample_label}
            </div>
          )}
        </div>
        
        <div className="chapters-list">
          <h5>Chapters ({volume.chapters?.length || 0})</h5>
          {volume.chapters?.map((chapter, idx) => (
            <div key={idx} className={`chapter-item ${chapter.status}`}>
              <div className="chapter-header">
                <span className="chapter-number">Ch. {chapter.order}</span>
                <span className="chapter-type">{chapter.type}</span>
                {getChapterStatusBadge(chapter.status)}
              </div>
              <div className="chapter-title">{chapter.title}</div>
              {chapter.paper_id && (
                <div className="chapter-source">
                  Source: {chapter.paper_id}
                </div>
              )}
            </div>
          ))}
        </div>
        
        {volume.writing_order && (
          <div className="writing-order">
            <h5>Writing Order</h5>
            <ol>
              {volume.writing_order.map((chapterNum, idx) => (
                <li key={idx}>Chapter {chapterNum}</li>
              ))}
            </ol>
          </div>
        )}
      </div>
    );
  };

  // Render short form paper content
  const renderShortFormContent = () => {
    if (finalAnswerData?.answer_format !== 'short_form') return null;
    
    return (
      <div className="tier3-section content-section">
        <h4>Final Answer Paper</h4>
        <div className="paper-library-file-location" style={{ fontSize: '0.75em', color: '#aaa', marginBottom: '0.75em', lineHeight: '1.5' }}>
          📁 For manual file retrieval, the short-form final answer is saved at: <code>{getRuntimeDataPath('auto_sessions/[session_folder]/final_answer/final_short_form_paper.txt')}</code>. Session folders are named after your research prompt and timestamp (e.g. <code>solve_riemann_hypothesis_2026-03-20_14-30/</code>).
        </div>
        {shortFormPaper ? (
          <div className="paper-content-container" ref={containerRef}>
            <div className="paper-meta">
              <span className="paper-id">{shortFormPaper.paper_id}</span>
              <span className="word-count">
                {shortFormPaper.word_count?.toLocaleString()} words
              </span>
            </div>
            <h3 className="paper-title-display">{shortFormPaper.title}</h3>
            {finalAnswerData?.sample_label && (
              <div className="sample-label" style={{ color: '#FF6700', fontSize: '0.9em', fontWeight: 'bold', marginTop: '0.5em' }}>
                {finalAnswerData.sample_label}
              </div>
            )}
            <LatexRenderer content={shortFormPaper.content} showLatex={showLatex} />
          </div>
        ) : (
          <div className="loading-content">
            Compiling final answer paper...
          </div>
        )}
      </div>
    );
  };

  // Render volume content (long form)
  const renderVolumeContent = () => {
    if (finalAnswerData?.answer_format !== 'long_form') return null;
    
    return (
      <div className="tier3-section content-section">
        <h4>Volume Content</h4>
        <div className="paper-library-file-location" style={{ fontSize: '0.75em', color: '#aaa', marginBottom: '0.75em', lineHeight: '1.5' }}>
          📁 For manual file retrieval, the long-form volume is saved at: <code>{getRuntimeDataPath('auto_sessions/[session_folder]/final_answer/final_volume.txt')}</code>. Individual chapter papers are stored as <code>chapter_[index]_paper.txt</code> in the same directory. Session folders are named after your research prompt and timestamp (e.g. <code>solve_riemann_hypothesis_2026-03-20_14-30/</code>).
        </div>
        {volumeContent && volumeContent.content ? (
          <div className="volume-content-container" ref={containerRef}>
            <div className="volume-meta">
              <span className="word-count">
                {volumeContent.word_count?.toLocaleString()} words total
              </span>
            </div>
            <LatexRenderer content={volumeContent.content} showLatex={showLatex} />
          </div>
        ) : (
          <div className="loading-content">
            {finalAnswerData?.volume ? (
              'Volume chapters in progress...'
            ) : (
              'Organizing volume structure...'
            )}
          </div>
        )}
      </div>
    );
  };

  // Check if there's anything to show
  const hasContent = finalAnswerData && (
    finalAnswerData.is_active || 
    finalAnswerData.status === 'complete' || 
    finalAnswerData.certainty_assessment
  );

  return (
    <div className={`final-answer-view ${statusBadge.class} ${getBannerShimmerEnabled() ? '' : 'no-shimmer'}`}>
      {/* Header */}
      <div className="final-answer-header" onClick={() => setIsExpanded(!isExpanded)}>
        <div className="header-left">
          <h2>
            🎯 Final Answer
            <span className="toggle-icon">{isExpanded ? '▼' : '▶'}</span>
          </h2>
          <span className={`final-status-badge ${statusBadge.class}`}>
            {statusBadge.text}
          </span>
        </div>
        {hasContent && (
          <div className="header-controls">
            <button
              className="archive-button"
              onClick={(e) => {
                e.stopPropagation();
                setShowArchiveModal(true);
              }}
              title="View archived papers and brainstorms used to create this answer"
              style={{
                fontSize: '0.875rem',
                padding: '0.375rem 0.75rem',
                backgroundColor: '#374151',
                color: '#d1d5db',
                border: 'none',
                borderRadius: '0.375rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                transition: 'background-color 0.2s'
              }}
              onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#4b5563'}
              onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#374151'}
            >
              📦 View Research Archive
            </button>
            <button
              className="regenerate-button"
              onClick={handleRegenerate}
              disabled={isRegenerating || (finalAnswerData?.is_active && finalAnswerData?.status !== 'complete')}
              title="Delete current final answer and regenerate with all papers"
            >
              {isRegenerating ? 'Regenerating...' : 'Regenerate'}
            </button>
            <label className="checkbox-label" onClick={e => e.stopPropagation()}>
              <input
                type="checkbox"
                checked={autoScroll}
                onChange={(e) => setAutoScroll(e.target.checked)}
              />
              Auto-scroll
            </label>
          </div>
        )}
      </div>

      {isExpanded && (
        <div className="final-answer-content">
          {!hasContent ? (
            <div className="no-content">
              <p className="no-content-message">
                Tier 3 Final Answer generation will trigger after every 5 completed papers.
              </p>
              <p className="no-content-stats">
                Papers completed: {status?.stats?.total_papers_completed || 0}
              </p>
              {status?.stats?.total_papers_completed > 0 && (
                <p className="next-trigger">
                  Next trigger at: {Math.ceil((status?.stats?.total_papers_completed + 1) / 5) * 5} papers
                </p>
              )}
            </div>
          ) : (
            <>
              {/* Navigation tabs for different sections */}
              <div className="section-tabs">
                <button
                  className={`section-tab ${activeSection === 'overview' ? 'active' : ''}`}
                  onClick={() => setActiveSection('overview')}
                >
                  Overview
                </button>
                {(finalAnswerData?.is_active || finalAnswerData?.status === 'complete') && (
                  <button
                    className={`section-tab ${activeSection === 'content' ? 'active' : ''}`}
                    onClick={() => setActiveSection('content')}
                  >
                    {finalAnswerData?.answer_format === 'long_form' ? 'Volume' : 'Paper'}
                  </button>
                )}
              </div>

              {activeSection === 'overview' && (
                <div className="overview-section">
                  {renderCertaintyAssessment()}
                  {renderFormatSelection()}
                  {renderVolumeOrganization()}
                </div>
              )}

              {activeSection === 'content' && (
                <div className="content-display-section">
                  {/* View toggle and download buttons */}
                  <div className="content-controls">
                    <div className="view-toggle">
                      <button
                        className={`btn ${showLatex ? '' : 'btn-secondary'}`}
                        onClick={() => setShowLatex(true)}
                      >
                        Rendered View
                      </button>
                      <button
                        className={`btn ${!showLatex ? '' : 'btn-secondary'}`}
                        onClick={() => setShowLatex(false)}
                      >
                        Raw Text
                      </button>
                    </div>
                    <div className="download-buttons">
                      <button
                        className="btn-download"
                        onClick={handleDownloadRaw}
                        disabled={(!shortFormPaper && !volumeContent)}
                        title="Download as raw text file"
                      >
                        Download Raw
                      </button>
                      <button
                        className="btn-download-pdf"
                        onClick={handleDownloadPDF}
                        disabled={isGeneratingPDF || (!shortFormPaper && !volumeContent)}
                        title="Download as PDF"
                      >
                        {isGeneratingPDF ? 'Preparing PDF...' : 'Download PDF'}
                      </button>
                      <button
                        className="btn-critique"
                        onClick={() => setCritiqueModalOpen(true)}
                        disabled={(!shortFormPaper && !volumeContent)}
                        title="Ask validator to critique this final answer"
                        style={{
                          background: 'linear-gradient(135deg, #18cc17 0%, #0f9110 100%)',
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
                  {finalAnswerData?.answer_format === 'short_form' 
                    ? renderShortFormContent()
                    : renderVolumeContent()
                  }
                </div>
              )}
            </>
          )}
        </div>
      )}
      
      {/* Archive Viewer Modal */}
      {showArchiveModal && finalAnswerData?.metadata?.answer_id && (
        <ArchiveViewerModal
          answerId={finalAnswerData.metadata.answer_id}
          onClose={() => setShowArchiveModal(false)}
        />
      )}

      {/* Critique Modal */}
      <PaperCritiqueModal
        isOpen={critiqueModalOpen}
        onClose={() => setCritiqueModalOpen(false)}
        paperType="final_answer"
        paperId={finalAnswerData?.metadata?.answer_id || finalAnswerData?.short_form_paper_id}
        paperTitle={shortFormPaper?.title || volumeContent?.title || finalAnswerData?.volume?.volume_title || 'Final Answer'}
        onGenerateCritique={(customPrompt, validatorConfig) => 
          autonomousAPI.generateFinalAnswerCritique(
            finalAnswerData?.metadata?.answer_id || finalAnswerData?.short_form_paper_id,
            customPrompt,
            validatorConfig
          )
        }
        onGetCritiques={() => 
          autonomousAPI.getFinalAnswerCritiques(
            finalAnswerData?.metadata?.answer_id || finalAnswerData?.short_form_paper_id
          )
        }
      />
    </div>
  );
};

export default FinalAnswerView;

