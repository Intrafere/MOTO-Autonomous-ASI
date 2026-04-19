import React, { useState, useEffect, useMemo } from 'react';
import LatexRenderer from '../LatexRenderer';
import PaperCritiqueModal from '../PaperCritiqueModal';
import { autonomousAPI } from '../../services/api';
import { downloadRawText, downloadPDFViaBackend, sanitizeFilename } from '../../utils/downloadHelpers';
import { prependDisclaimer } from '../../utils/disclaimerHelper';
import { buildResearchRunGroups } from '../../utils/researchRunHistory';
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
  const [stage2Papers, setStage2Papers] = useState([]);
  const [sessionsResponse, setSessionsResponse] = useState(null);
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

      const [answersResult, sessionsResult, papersResult] = await Promise.allSettled([
        fetch('/api/auto-research/final-answer-library').then(async (response) => {
          if (!response.ok) {
            throw new Error('Failed to load final answer library');
          }
          return response.json();
        }),
        autonomousAPI.getSessions(),
        autonomousAPI.getPaperHistory(),
      ]);

      if (answersResult.status !== 'fulfilled') {
        throw answersResult.reason;
      }

      if (answersResult.value.success) {
        setFinalAnswers(answersResult.value.final_answers || []);
      } else {
        setError('Failed to load final answer library');
      }

      if (sessionsResult.status === 'fulfilled') {
        setSessionsResponse(sessionsResult.value);
      } else {
        setSessionsResponse(null);
        console.warn('Stage 3 history: failed to load sessions metadata', sessionsResult.reason);
      }

      if (papersResult.status === 'fulfilled') {
        setStage2Papers(papersResult.value.papers || []);
      } else {
        setStage2Papers([]);
        console.warn('Stage 3 history: failed to load Stage 2 paper metadata', papersResult.reason);
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

    const blob = new Blob([prependDisclaimer(expandedContent.content, 'paper')], { type: 'text/plain' });
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
        downloadRawText(data.content, filename, null, 'paper');
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
        'paper',
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

  const runGroups = useMemo(() => (
    buildResearchRunGroups({
      sessionsResponse,
      stage2Papers,
      stage3Answers: finalAnswers,
    })
  ), [sessionsResponse, stage2Papers, finalAnswers]);

  const visibleRunGroups = useMemo(() => {
    const searchLower = searchTerm.trim().toLowerCase();

    const matchesAnswer = (answer) => {
      if (filterFormat !== 'all' && answer.format !== filterFormat) {
        return false;
      }

      if (!searchLower) {
        return true;
      }

      return (
        answer.title?.toLowerCase().includes(searchLower) ||
        answer.user_prompt?.toLowerCase().includes(searchLower) ||
        answer.session_id?.toLowerCase().includes(searchLower)
      );
    };

    return runGroups
      .map((runGroup) => ({
        ...runGroup,
        visibleStage3Answers: runGroup.stage3Answers.filter(matchesAnswer),
      }))
      .filter((runGroup) => runGroup.visibleStage3Answers.length > 0);
  }, [runGroups, filterFormat, searchTerm]);

  if (loading) {
    return (
      <div className="final-answer-library">
        <div className="library-header">
          <h2>Stage 3 Final Answers History</h2>
          <p>Loading completed Stage 3 final answers...</p>
        </div>
        <div className="library-loading">
          <span className="library-loading__icon" aria-hidden="true">⟳</span>
          <span className="library-loading__text">Loading...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="final-answer-library">
        <div className="library-header">
          <h2>Stage 3 Final Answers History</h2>
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
        <h2>📚 Stage 3 Final Answers History</h2>
        <p>If you have enabled Tier 3 experimental final answer generation, completed Stage 3 answers will appear here. Browse all completed research volumes and short-form answers from your autonomous research sessions.</p>
        <div className="library-stats">
          <span className="stat-badge">
            {finalAnswers.length} {finalAnswers.length === 1 ? 'Answer' : 'Answers'}
          </span>
          <span className="stat-badge">
            {runGroups.filter((runGroup) => runGroup.stage3AnswerCount > 0).length} Research Runs
          </span>
          <span className="stat-badge">
            {runGroups
              .filter((runGroup) => runGroup.stage3AnswerCount > 0)
              .reduce((total, runGroup) => total + runGroup.stage2PaperCount, 0)} Matching Stage 2 Papers
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
      {visibleRunGroups.length === 0 ? (
        <div className="fal-empty-state">
          <span className="empty-icon">📭</span>
          <h3>No Stage 3 final answers found</h3>
          <p>
            {searchTerm || filterFormat !== 'all'
              ? 'Try adjusting your search or filters'
              : 'Complete a Tier 3 final answer to see it here'}
          </p>
        </div>
      ) : (
        <div className="run-history-groups">
          {visibleRunGroups.map((runGroup) => (
            <section key={runGroup.sessionId} className="run-history-group">
              <div className="run-history-group-header">
                <div className="run-history-group-heading">
                  <h3 className="run-history-group-title">{runGroup.userPrompt}</h3>
                  <p className="run-history-group-subtitle">
                    Research Run: {runGroup.displaySessionId}
                  </p>
                  {runGroup.createdAt && (
                    <p className="run-history-group-subtitle">
                      Started: {formatDate(runGroup.createdAt)}
                    </p>
                  )}
                </div>
                <div className="run-history-group-badges">
                  {runGroup.isCurrent && (
                    <span className="run-history-group-badge run-history-group-badge--current">
                      Current Run
                    </span>
                  )}
                  <span className="run-history-group-badge">
                    Stage 2 Papers: {runGroup.stage2PaperCount}
                  </span>
                  <span className="run-history-group-badge run-history-group-badge--linked">
                    Stage 3 Answers: {runGroup.stage3AnswerCount}
                  </span>
                  {runGroup.brainstormCount !== null && runGroup.brainstormCount !== undefined && (
                    <span className="run-history-group-badge">
                      Brainstorms: {runGroup.brainstormCount}
                    </span>
                  )}
                </div>
              </div>

              <div className="run-history-group-body">
                <div className="answer-list">
                  {runGroup.visibleStage3Answers.map(answer => (
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
                                background: 'linear-gradient(135deg, #1eff1c 0%, #0fcc0d 100%)',
                                border: 'none',
                                color: '#0b2e0b',
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
                            <LatexRenderer content={prependDisclaimer(expandedContent.content, 'paper')} showLatex={showLatex} />
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </section>
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

