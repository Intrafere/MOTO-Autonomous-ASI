import React, { useState, useEffect } from 'react';
import LatexRenderer from '../LatexRenderer';
import PaperCritiqueModal from '../PaperCritiqueModal';
import { autonomousAPI } from '../../services/api';
import { downloadRawText, downloadPDFViaBackend, sanitizeFilename } from '../../utils/downloadHelpers';
import './FinalAnswerLibrary.css';

/**
 * FinalAnswerLibrary Component
 * 
 * Displays a library of ALL completed final answers (volumes and papers)
 * from all research sessions. Allows browsing, viewing, and downloading.
 * 
 * Features:
 * - List view with metadata cards
 * - Expandable to show full content
 * - Filter by format (short/long form)
 * - Search by title or prompt
 * - Download individual answers
 * - Shows certainty level and word count
 */
function FinalAnswerLibrary() {
  const [finalAnswers, setFinalAnswers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedId, setExpandedId] = useState(null);
  const [expandedContent, setExpandedContent] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterFormat, setFilterFormat] = useState('all'); // 'all', 'short_form', 'long_form'
  const [showLatex, setShowLatex] = useState(false); // Raw text by default for performance with large docs
  const [downloadingPDF, setDownloadingPDF] = useState(null); // Track which answer is generating PDF
  
  // Critique modal state
  const [critiqueModalOpen, setCritiqueModalOpen] = useState(false);
  const [selectedAnswerForCritique, setSelectedAnswerForCritique] = useState(null);

  useEffect(() => {
    loadFinalAnswers();
  }, []);

  const loadFinalAnswers = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch('/api/auto-research/final-answer-library');
      const data = await response.json();
      
      if (data.success) {
        setFinalAnswers(data.final_answers || []);
      } else {
        setError('Failed to load final answer library');
      }
    } catch (err) {
      setError(`Error loading library: ${err.message}`);
      console.error('Failed to load final answer library:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadFullContent = async (answerId) => {
    if (expandedId === answerId) {
      // Collapse if already expanded
      setExpandedId(null);
      setExpandedContent(null);
      return;
    }

    try {
      const response = await fetch(`/api/auto-research/final-answer-library/${answerId}`);
      const data = await response.json();
      
      if (data.success) {
        setExpandedId(answerId);
        setExpandedContent(data);
      } else {
        setError(`Failed to load content for ${answerId}`);
      }
    } catch (err) {
      setError(`Error loading content: ${err.message}`);
      console.error(`Failed to load content for ${answerId}:`, err);
    }
  };

  const downloadAnswer = (answer) => {
    if (expandedId !== answer.answer_id || !expandedContent) {
      alert('Please expand the answer first to download it');
      return;
    }

    const blob = new Blob([expandedContent.content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `final_answer_${answer.answer_id}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };
  
  // Download raw text without expanding card first
  const downloadAnswerRaw = async (e, answer) => {
    e.stopPropagation(); // Prevent card expansion
    
    try {
      const response = await fetch(`/api/auto-research/final-answer-library/${answer.answer_id}`);
      const data = await response.json();
      
      if (data.success && data.content) {
        const filename = sanitizeFilename(`Final_Answer_${answer.title}`);
        downloadRawText(data.content, filename, null);
      } else {
        alert('Failed to load content for download');
      }
    } catch (err) {
      console.error('Download failed:', err);
      alert(`Download failed: ${err.message}`);
    }
  };
  
  // Download PDF without expanding card first (loads content on demand)
  const downloadAnswerPDF = async (e, answer) => {
    e.stopPropagation();

    if (downloadingPDF) {
      alert('Already preparing a PDF, please wait...');
      return;
    }

    try {
      const response = await fetch(`/api/auto-research/final-answer-library/${answer.answer_id}`);
      const data = await response.json();

      if (!data.success || !data.content) {
        throw new Error('Failed to load content');
      }

      const filename = sanitizeFilename(`Final_Answer_${answer.title}`);
      const metadata = {
        title: answer.title,
        wordCount: answer.word_count,
        date: formatDate(answer.completion_date),
        models: null,
      };

      await downloadPDFViaBackend(
        data.content,
        metadata,
        filename,
        null,
        () => setDownloadingPDF(answer.answer_id),
        () => setDownloadingPDF(null),
        (error) => {
          setDownloadingPDF(null);
          console.error('PDF generation failed:', error);
          alert(`PDF generation failed: ${error.message}`);
        },
      );
    } catch (error) {
      setDownloadingPDF(null);
      console.error('PDF generation failed:', error);
      alert(`Failed to generate PDF: ${error.message}`);
    }
  };

  const getCertaintyBadgeColor = (level) => {
    switch (level) {
      case 'total_answer': return '#2d5f2d';
      case 'partial_answer': return '#5a6e2d';
      case 'no_answer_known': return '#6e542d';
      case 'appears_impossible': return '#6e2d2d';
      default: return '#555';
    }
  };

  const formatCertaintyLevel = (level) => {
    return level.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  };

  const formatDate = (isoString) => {
    try {
      const date = new Date(isoString);
      return date.toLocaleString();
    } catch {
      return isoString;
    }
  };

  // Filter final answers
  const filteredAnswers = finalAnswers.filter(answer => {
    // Format filter
    if (filterFormat !== 'all' && answer.format !== filterFormat) {
      return false;
    }

    // Search filter
    if (searchTerm) {
      const searchLower = searchTerm.toLowerCase();
      const matchesTitle = answer.title.toLowerCase().includes(searchLower);
      const matchesPrompt = answer.user_prompt.toLowerCase().includes(searchLower);
      return matchesTitle || matchesPrompt;
    }

    return true;
  });

  if (loading) {
    return (
      <div className="final-answer-library">
        <div className="library-header">
          <h2>Final Answer Library</h2>
          <p>Loading your completed research volumes...</p>
        </div>
        <div className="loading-spinner">⟳ Loading...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="final-answer-library">
        <div className="library-header">
          <h2>Final Answer Library</h2>
        </div>
        <div className="error-message">
          <span>⚠</span>
          <p>{error}</p>
          <button onClick={loadFinalAnswers} className="retry-button">
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="final-answer-library">
      {/* Header */}
      <div className="library-header">
        <h2>📚 Final Answer Library</h2>
        <p>If you have enabled Tier 3 experimental final answer generation, any completed answers will appear here. Browse all completed research volumes and papers from your autonomous research sessions.</p>
        <div className="library-stats">
          <span className="stat-badge">
            {finalAnswers.length} {finalAnswers.length === 1 ? 'Answer' : 'Answers'}
          </span>
          <span className="stat-badge">
            {finalAnswers.filter(a => a.format === 'long_form').length} Volumes
          </span>
          <span className="stat-badge">
            {finalAnswers.filter(a => a.format === 'short_form').length} Papers
          </span>
        </div>
      </div>

      {/* Controls */}
      <div className="library-controls">
        <input
          type="text"
          placeholder="Search by title or prompt..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="search-input"
        />
        
        <div className="filter-buttons">
          <button
            className={filterFormat === 'all' ? 'active' : ''}
            onClick={() => setFilterFormat('all')}
          >
            All
          </button>
          <button
            className={filterFormat === 'long_form' ? 'active' : ''}
            onClick={() => setFilterFormat('long_form')}
          >
            Volumes
          </button>
          <button
            className={filterFormat === 'short_form' ? 'active' : ''}
            onClick={() => setFilterFormat('short_form')}
          >
            Papers
          </button>
        </div>
      </div>

      {/* List */}
      {filteredAnswers.length === 0 ? (
        <div className="empty-state">
          <span className="empty-icon">📭</span>
          <h3>No final answers found</h3>
          <p>
            {searchTerm || filterFormat !== 'all'
              ? 'Try adjusting your search or filters'
              : 'Complete a Tier 3 final answer to see it here'}
          </p>
        </div>
      ) : (
        <div className="answer-list">
          {filteredAnswers.map(answer => (
            <div
              key={answer.answer_id}
              className={`answer-card ${expandedId === answer.answer_id ? 'expanded' : ''}`}
            >
              {/* Header */}
              <div className="answer-header" onClick={() => loadFullContent(answer.answer_id)}>
                <div className="answer-title-row">
                  <h3 className="answer-title">
                    {answer.format === 'long_form' ? '▭' : '⊟'} {answer.title}
                  </h3>
                  <button className="expand-button">
                    {expandedId === answer.answer_id ? '▼' : '▶'}
                  </button>
                </div>
                
                <div className="answer-metadata">
                  <span className="format-badge">
                    {answer.format === 'long_form' ? 'Volume' : 'Paper'}
                  </span>
                  <span
                    className="certainty-badge"
                    style={{ backgroundColor: getCertaintyBadgeColor(answer.certainty_level) }}
                  >
                    {formatCertaintyLevel(answer.certainty_level)}
                  </span>
                  <span className="word-count">{answer.word_count.toLocaleString()} words</span>
                  {answer.format === 'long_form' && (
                    <span className="chapter-count">{answer.chapter_count} chapters</span>
                  )}
                </div>

                <div className="answer-prompt">
                  <strong>Research Question:</strong> {answer.user_prompt}
                </div>

                <div className="answer-footer-info">
                  <span className="completion-date">
                    Completed: {formatDate(answer.completion_date)}
                  </span>
                  <span className="session-id">
                    Session: {answer.session_id === 'legacy' ? 'Legacy' : answer.session_id}
                  </span>
                </div>
                
                {/* Quick Download Buttons (no expand needed) */}
                <div className="quick-download-buttons" onClick={(e) => e.stopPropagation()}>
                  <button
                    className="quick-download-raw"
                    onClick={(e) => downloadAnswerRaw(e, answer)}
                    title="Download raw text immediately"
                  >
                    📄 Download Raw
                  </button>
                  <button
                    className="quick-download-pdf"
                    onClick={(e) => downloadAnswerPDF(e, answer)}
                    disabled={downloadingPDF === answer.answer_id}
                    title="Generate and download PDF"
                  >
                    {downloadingPDF === answer.answer_id ? '⏳ Preparing PDF...' : '📑 Download PDF'}
                  </button>
                </div>
              </div>

              {/* Expanded Content */}
              {expandedId === answer.answer_id && expandedContent && (
                <div className="answer-content">
                  <div className="content-actions">
                    <button onClick={() => downloadAnswer(answer)} className="download-button">
                      💾 Download
                    </button>
                    <button
                      className="critique-button"
                      onClick={() => {
                        setSelectedAnswerForCritique(answer);
                        setCritiqueModalOpen(true);
                      }}
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
                      title="Ask validator to critique this final answer"
                    >
                      ⭐ Ask Validator to Critique
                    </button>
                    {/* View toggle for LaTeX rendering */}
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
                  </div>

                  {/* Chapter list (for volumes) */}
                  {answer.format === 'long_form' && expandedContent.chapters && (
                    <div className="chapter-list">
                      <h4>Chapters:</h4>
                      <ol>
                        {expandedContent.chapters.map(ch => (
                          <li key={ch.order}>
                            <strong>{ch.title}</strong>
                            <span className="chapter-type">
                              [{ch.chapter_type.replace(/_/g, ' ')}]
                            </span>
                          </li>
                        ))}
                      </ol>
                    </div>
                  )}

                  {/* Full content */}
                  <div className="full-content">
                    <LatexRenderer content={expandedContent.content} showLatex={showLatex} />
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Refresh Button */}
      <div className="library-footer">
        <button onClick={loadFinalAnswers} className="refresh-button">
          Refresh Library
        </button>
      </div>

      {/* Critique Modal */}
      {selectedAnswerForCritique && (
        <PaperCritiqueModal
          isOpen={critiqueModalOpen}
          onClose={() => {
            setCritiqueModalOpen(false);
            setSelectedAnswerForCritique(null);
          }}
          paperType="final_answer"
          paperId={selectedAnswerForCritique.answer_id}
          paperTitle={selectedAnswerForCritique.title || 'Final Answer'}
          onGenerateCritique={(customPrompt, validatorConfig) => 
            autonomousAPI.generateFinalAnswerCritique(
              selectedAnswerForCritique.answer_id,
              customPrompt,
              validatorConfig
            )
          }
          onGetCritiques={() => 
            autonomousAPI.getFinalAnswerCritiques(selectedAnswerForCritique.answer_id)
          }
        />
      )}
    </div>
  );
}

export default FinalAnswerLibrary;

