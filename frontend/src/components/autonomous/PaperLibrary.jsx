/**
 * PaperLibrary - Displays grid of completed papers.
 */
import React, { useState, useRef } from 'react';
import './AutonomousResearch.css';
import LatexRenderer from '../LatexRenderer';
import { downloadRawText, downloadPDF, sanitizeFilename } from '../../utils/downloadHelpers';
import PaperCritiqueModal from '../PaperCritiqueModal';
import { autonomousAPI } from '../../services/api';

const PaperLibrary = ({ papers, onRefresh, api, archivedCount = 0 }) => {
  const [expandedId, setExpandedId] = useState(null);
  const [expandedContent, setExpandedContent] = useState(null);
  const [loading, setLoading] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const [isGeneratingPDF, setIsGeneratingPDF] = useState(false);
  const paperContainerRef = useRef(null);
  
  // Critique modal state
  const [critiqueModalOpen, setCritiqueModalOpen] = useState(false);
  const [critiquePaper, setCritiquePaper] = useState(null);

  const handleCardClick = async (paperId) => {
    if (expandedId === paperId) {
      setExpandedId(null);
      setExpandedContent(null);
      return;
    }

    setExpandedId(paperId);
    setLoading(true);

    try {
      const data = await api.getAutonomousPaper(paperId);
      setExpandedContent({
        content: data.content,
        outline: data.outline,
        title: data.title
      });
    } catch (error) {
      console.error('Failed to load paper content:', error);
      setExpandedContent({ content: 'Failed to load content', outline: '', title: '' });
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteClick = (e, paperId) => {
    e.stopPropagation();
    setDeleteConfirm(paperId);
  };

  const handleDeleteConfirm = async (paperId) => {
    setDeleting(true);
    try {
      await api.deletePaper(paperId);
      setDeleteConfirm(null);
      onRefresh();
    } catch (error) {
      console.error('Failed to delete paper:', error);
      alert(`Failed to delete paper: ${error.message}`);
    } finally {
      setDeleting(false);
    }
  };

  const handleDeleteCancel = (e) => {
    e.stopPropagation();
    setDeleteConfirm(null);
  };

  const handleDownloadRaw = (e, paper) => {
    e.stopPropagation();
    
    if (!expandedContent || typeof expandedContent !== 'object') {
      alert('Paper content not loaded. Please expand the paper first.');
      return;
    }
    
    const filename = sanitizeFilename(`${paper.paper_id}_${paper.title}`);
    const content = expandedContent.content || '';
    const outline = expandedContent.outline || '';
    
    downloadRawText(content, filename, outline);
  };

  const handleDownloadPDF = async (e, paper) => {
    e.stopPropagation();
    
    if (!expandedContent || typeof expandedContent !== 'object' || !paperContainerRef.current) {
      alert('Paper content not loaded. Please expand the paper first.');
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
        title: expandedContent.title || paper.title,
        wordCount: paper.word_count,
        date: paper.created_at ? new Date(paper.created_at).toLocaleDateString() : new Date().toLocaleDateString(),
        models: paper.model_usage ? Object.keys(paper.model_usage).join(', ') : null
      };
      
      const filename = sanitizeFilename(`${paper.paper_id}_${paper.title}`);
      // Don't pass outline - it's already included in the rendered content (line 237-238)
      await downloadPDF(element, metadata, filename, null);
    } catch (error) {
      console.error('PDF generation error:', error);
      alert('Failed to generate PDF: ' + error.message);
    } finally {
      setIsGeneratingPDF(false);
    }
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return 'N/A';
    return new Date(dateStr).toLocaleDateString();
  };

  const truncateAbstract = (abstract, maxLength = 200) => {
    if (!abstract || abstract.length <= maxLength) return abstract;
    return abstract.substring(0, maxLength) + '...';
  };

  // Get color for critique rating badge
  const getCritiqueColor = (rating) => {
    if (rating >= 8) return '#10b981'; // Green
    if (rating >= 6.25) return '#3b82f6'; // Blue
    if (rating >= 4) return '#eab308'; // Yellow
    if (rating >= 2) return '#f97316'; // Orange
    return '#ef4444'; // Red
  };

  // Open critique modal for a paper
  const handleOpenCritique = (e, paper) => {
    e.stopPropagation();
    setCritiquePaper(paper);
    setCritiqueModalOpen(true);
  };

  if (!papers || papers.length === 0) {
    return (
      <div className="paper-library">
        <div className="paper-library-header">
          <h3>Paper Library (0 Papers)</h3>
          <button onClick={onRefresh} className="btn-refresh">
            Refresh
          </button>
        </div>
        <div className="paper-library-warning">
          (WARNING: these may get auto pruned - back up anything you like)
        </div>
        <div className="paper-library-pruned-counter">
          Pruned Papers: {archivedCount}
        </div>
        <div className="empty-state">
          No papers completed yet. Autonomous research will generate papers from brainstorm databases.
        </div>
      </div>
    );
  }

  return (
    <div className="paper-library">
      <div className="paper-library-header">
        <h3>Paper Library ({papers.length} Papers)</h3>
        <button onClick={onRefresh} className="btn-refresh">
          Refresh
        </button>
      </div>
      <div className="paper-library-warning">
        (HOW THIS PAGE WORKS: This paper database will continue to accumulate until the AI harness autonomously decides to generate the final answer or until the user forces final answer generation. Papers utilize their respective brainstorm topics during writing and may undergo critique-revision before final appearance on this page. Papers may start off mediocre, however will improve over time as the AI selects internal papers for future reference or removal, if you are unhappy with your paper quality try a higher parameter model. Paper quality greatly improves with higher parameter models. Any given paper may be pruned/deleted if the AI deems it to hurt the collective database quality - back up any paper you certainly want to save. Accumulating a large amount of papers before final answer generation is normal (i.e. 10 to 20 papers with several pruned/deleted. When forcing final answer generation the AI will decide either: 1.) not enough info, brainstorm more, 2.) write answer - new short form paper, 3.) write answer, longform volume - organize select accepted papers into a longform volume with chapters, write gap papers (if applicable), conclusion chapter then introduction chapter).)
      </div>
      <div className="paper-library-pruned-counter">
        Pruned Papers: {archivedCount}
      </div>

      <div className="paper-grid">
        {papers.map((paper) => (
          <div
            key={paper.paper_id}
            className={`paper-card ${expandedId === paper.paper_id ? 'expanded' : ''}`}
            onClick={() => handleCardClick(paper.paper_id)}
          >
            <div className="paper-card-header">
              <span className="paper-card-id">{paper.paper_id}</span>
              <span className="paper-word-count">{paper.word_count?.toLocaleString()} words</span>
            </div>

            <div className="paper-card-title">
              {paper.title}
              {paper.critique_avg !== null && paper.critique_avg !== undefined && (
                <span
                  style={{
                    display: 'inline-block',
                    marginLeft: '8px',
                    padding: '2px 8px',
                    borderRadius: '4px',
                    fontSize: '0.75rem',
                    fontWeight: '600',
                    backgroundColor: getCritiqueColor(paper.critique_avg),
                    color: '#fff',
                    verticalAlign: 'middle'
                  }}
                  title={`Auto-critique rating: ${paper.critique_avg}/10`}
                >
                  ⭐ {paper.critique_avg}
                </span>
              )}
            </div>

            <div className="paper-card-abstract">
              {truncateAbstract(paper.abstract)}
            </div>

            <div className="paper-card-meta">
              <span>Source: {paper.source_brainstorm_ids?.join(', ') || 'N/A'}</span>
              <span>{formatDate(paper.created_at)}</span>
            </div>

            {paper.referenced_papers?.length > 0 && (
              <div className="paper-references">
                References: {paper.referenced_papers.join(', ')}
              </div>
            )}

            {expandedId === paper.paper_id && (
              <>
                <div className="paper-actions">
                  <button
                    className="btn-download"
                    onClick={(e) => handleDownloadPDF(e, paper)}
                    disabled={isGeneratingPDF || !expandedContent}
                    title="Download as PDF"
                  >
                    {isGeneratingPDF ? 'Generating...' : 'Download PDF'}
                  </button>
                  
                  <button
                    className="btn-download"
                    onClick={(e) => handleDownloadRaw(e, paper)}
                    disabled={!expandedContent}
                    title="Download as raw text"
                  >
                    Download Raw
                  </button>

                  <button
                    className="btn-critique"
                    onClick={(e) => handleOpenCritique(e, paper)}
                    title="Ask validator to critique this paper"
                    style={{
                      background: 'linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%)',
                      border: 'none',
                      color: '#fff',
                      padding: '0.35rem 0.7rem',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontWeight: '500',
                      fontSize: '0.75rem'
                    }}
                  >
                    ⭐ Critique
                  </button>
                  
                  {deleteConfirm === paper.paper_id ? (
                    <div className="delete-confirm-inline" onClick={(e) => e.stopPropagation()}>
                      <span>Delete this paper?</span>
                      <button 
                        className="btn-delete-confirm" 
                        onClick={() => handleDeleteConfirm(paper.paper_id)}
                        disabled={deleting}
                      >
                        {deleting ? 'Deleting...' : 'Yes'}
                      </button>
                      <button 
                        className="btn-delete-cancel" 
                        onClick={handleDeleteCancel}
                        disabled={deleting}
                      >
                        Cancel
                      </button>
                    </div>
                  ) : (
                    <button
                      className="btn-delete-paper"
                      onClick={(e) => handleDeleteClick(e, paper.paper_id)}
                      title="Delete this paper"
                    >
                      Delete
                    </button>
                  )}
                </div>
                <div className="paper-full-content" ref={paperContainerRef}>
                  {loading ? (
                    <div className="loading">Loading content...</div>
                  ) : expandedContent && typeof expandedContent === 'object' ? (
                    <div className="paper-section">
                      <h4>Paper Content</h4>
                      <LatexRenderer
                        content={
                          expandedContent.outline
                            ? `${expandedContent.outline}\n\n${'='.repeat(80)}\n\n${expandedContent.content || 'No content available'}`
                            : expandedContent.content || 'No content available'
                        }
                        className="paper-content-renderer"
                        showToggle={true}
                        defaultRaw={false}
                      />
                    </div>
                  ) : (
                    <pre className="paper-content">{expandedContent || 'No content available'}</pre>
                  )}
                </div>
              </>
            )}
          </div>
        ))}
      </div>

      {/* Critique Modal */}
      <PaperCritiqueModal
        isOpen={critiqueModalOpen}
        onClose={() => {
          setCritiqueModalOpen(false);
          setCritiquePaper(null);
        }}
        paperType="autonomous_paper"
        paperId={critiquePaper?.paper_id}
        paperTitle={critiquePaper?.title}
        onGenerateCritique={(customPrompt, validatorConfig) => 
          autonomousAPI.generatePaperCritique(critiquePaper?.paper_id, customPrompt, validatorConfig)
        }
        onGetCritiques={() => 
          autonomousAPI.getPaperCritiques(critiquePaper?.paper_id)
        }
      />
    </div>
  );
};

export default PaperLibrary;

