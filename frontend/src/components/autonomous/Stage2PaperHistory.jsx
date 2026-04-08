import React, { useEffect, useMemo, useState } from 'react';
import LatexRenderer from '../LatexRenderer';
import PaperCritiqueModal from '../PaperCritiqueModal';
import { autonomousAPI } from '../../services/api';
import { downloadRawText, downloadPDFViaBackend, sanitizeFilename } from '../../utils/downloadHelpers';
import { buildResearchRunGroups } from '../../utils/researchRunHistory';
import './FinalAnswerLibrary.css';
import './AutonomousResearch.css';
import './Stage2PaperHistory.css';

function getCritiqueColor(rating) {
  if (rating >= 8) return '#10b981';
  if (rating >= 6.25) return '#3b82f6';
  if (rating >= 4) return '#eab308';
  if (rating >= 2) return '#f97316';
  return '#ef4444';
}

function formatDate(dateStr) {
  if (!dateStr) return 'N/A';
  return new Date(dateStr).toLocaleString();
}

function truncateAbstract(abstract, maxLength = 220) {
  if (!abstract || abstract.length <= maxLength) return abstract;
  return `${abstract.substring(0, maxLength)}...`;
}

export default function Stage2PaperHistory({ onCurrentSessionDataChanged }) {
  const [papers, setPapers] = useState([]);
  const [finalAnswers, setFinalAnswers] = useState([]);
  const [sessionsResponse, setSessionsResponse] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedId, setExpandedId] = useState(null);
  const [expandedContent, setExpandedContent] = useState(null);
  const [loadingContentId, setLoadingContentId] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [deleteConfirmId, setDeleteConfirmId] = useState(null);
  const [deletingId, setDeletingId] = useState(null);
  const [generatingPdfId, setGeneratingPdfId] = useState(null);
  const [critiqueModalOpen, setCritiqueModalOpen] = useState(false);
  const [critiquePaper, setCritiquePaper] = useState(null);

  useEffect(() => {
    loadPaperHistory();
  }, []);

  const loadPaperHistory = async () => {
    try {
      setLoading(true);
      setError(null);

      const [papersResult, sessionsResult, finalAnswersResult] = await Promise.allSettled([
        autonomousAPI.getPaperHistory(),
        autonomousAPI.getSessions(),
        fetch('/api/auto-research/final-answer-library').then(async (response) => {
          if (!response.ok) {
            throw new Error('Failed to load Stage 3 final answer history');
          }
          return response.json();
        }),
      ]);

      if (papersResult.status !== 'fulfilled') {
        throw papersResult.reason;
      }

      setPapers(papersResult.value.papers || []);

      if (sessionsResult.status === 'fulfilled') {
        setSessionsResponse(sessionsResult.value);
      } else {
        setSessionsResponse(null);
        console.warn('Stage 2 history: failed to load sessions metadata', sessionsResult.reason);
      }

      if (finalAnswersResult.status === 'fulfilled' && finalAnswersResult.value.success) {
        setFinalAnswers(finalAnswersResult.value.final_answers || []);
      } else {
        setFinalAnswers([]);
        if (finalAnswersResult.status === 'rejected') {
          console.warn('Stage 2 history: failed to load Stage 3 final answer metadata', finalAnswersResult.reason);
        }
      }
    } catch (err) {
      setError(`Error loading Stage 2 history: ${err.message}`);
      console.error('Failed to load Stage 2 paper history:', err);
    } finally {
      setLoading(false);
    }
  };

  const runGroups = useMemo(() => (
    buildResearchRunGroups({
      sessionsResponse,
      stage2Papers: papers,
      stage3Answers: finalAnswers,
    })
  ), [sessionsResponse, papers, finalAnswers]);

  const visibleRunGroups = useMemo(() => {
    const searchLower = searchTerm.trim().toLowerCase();
    const matchesPaper = (paper) => {
      const sources = (paper.source_brainstorm_ids || []).join(' ').toLowerCase();
      return (
        paper.title?.toLowerCase().includes(searchLower) ||
        paper.abstract?.toLowerCase().includes(searchLower) ||
        paper.user_prompt?.toLowerCase().includes(searchLower) ||
        paper.paper_id?.toLowerCase().includes(searchLower) ||
        paper.session_id?.toLowerCase().includes(searchLower) ||
        sources.includes(searchLower)
      );
    };

    return runGroups
      .map((runGroup) => ({
        ...runGroup,
        visibleStage2Papers: searchLower
          ? runGroup.stage2Papers.filter(matchesPaper)
          : runGroup.stage2Papers,
      }))
      .filter((runGroup) => runGroup.visibleStage2Papers.length > 0);
  }, [runGroups, searchTerm]);

  const handleCardClick = async (paper) => {
    if (expandedId === paper.history_id) {
      setExpandedId(null);
      setExpandedContent(null);
      setLoadingContentId(null);
      return;
    }

    setExpandedId(paper.history_id);
    setExpandedContent(null);
    setLoadingContentId(paper.history_id);

    try {
      const data = await autonomousAPI.getHistoryPaper(paper.session_id, paper.paper_id);
      setExpandedContent(data);
    } catch (err) {
      console.error('Failed to load history paper content:', err);
      setExpandedContent({
        history_id: paper.history_id,
        content: 'Failed to load content',
        outline: '',
        title: paper.title,
      });
    } finally {
      setLoadingContentId(null);
    }
  };

  const handleDownloadRaw = (e, paper) => {
    e.stopPropagation();

    if (expandedId !== paper.history_id || !expandedContent) {
      alert('Please expand the paper first.');
      return;
    }

    const filename = sanitizeFilename(`${paper.session_id}_${paper.paper_id}_${paper.title}`);
    downloadRawText(
      expandedContent.content || '',
      filename,
      expandedContent.outline || ''
    );
  };

  const handleDownloadPDF = async (e, paper) => {
    e.stopPropagation();

    if (expandedId !== paper.history_id || !expandedContent) {
      alert('Please expand the paper first.');
      return;
    }

    const filename = sanitizeFilename(`${paper.session_id}_${paper.paper_id}_${paper.title}`);
    const metadata = {
      title: expandedContent.title || paper.title,
      wordCount: paper.word_count,
      date: paper.created_at ? new Date(paper.created_at).toLocaleDateString() : new Date().toLocaleDateString(),
      models: paper.model_usage ? Object.keys(paper.model_usage).join(', ') : null,
    };

    await downloadPDFViaBackend(
      expandedContent.content || '',
      metadata,
      filename,
      expandedContent.outline || null,
      () => setGeneratingPdfId(paper.history_id),
      () => setGeneratingPdfId(null),
      (downloadError) => {
        setGeneratingPdfId(null);
        console.error('PDF generation error:', downloadError);
        alert(`PDF generation failed: ${downloadError.message}`);
      },
    );
  };

  const handleOpenCritique = (e, paper) => {
    e.stopPropagation();
    setCritiquePaper(paper);
    setCritiqueModalOpen(true);
  };

  const handleDeleteClick = (e, paper) => {
    e.stopPropagation();
    setDeleteConfirmId(paper.history_id);
  };

  const handleDeleteCancel = (e) => {
    e.stopPropagation();
    setDeleteConfirmId(null);
  };

  const handleDeleteConfirm = async (paper) => {
    setDeletingId(paper.history_id);
    try {
      await autonomousAPI.deleteHistoryPaper(paper.session_id, paper.paper_id);
      if (expandedId === paper.history_id) {
        setExpandedId(null);
        setExpandedContent(null);
      }
      if (critiquePaper?.history_id === paper.history_id) {
        setCritiqueModalOpen(false);
        setCritiquePaper(null);
      }
      setDeleteConfirmId(null);
      await loadPaperHistory();
      if (onCurrentSessionDataChanged) {
        await onCurrentSessionDataChanged();
      }
    } catch (err) {
      console.error('Failed to delete history paper:', err);
      alert(`Failed to delete paper: ${err.message}`);
    } finally {
      setDeletingId(null);
    }
  };

  if (loading) {
    return (
      <div className="final-answer-library stage2-paper-history">
        <div className="library-header">
          <h2>Stage 2 Final Answer History</h2>
          <p>Loading completed Stage 2 papers from all sessions...</p>
        </div>
        <div className="loading-spinner">⟳ Loading...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="final-answer-library stage2-paper-history">
        <div className="library-header">
          <h2>Stage 2 Final Answer History</h2>
        </div>
        <div className="error-message">
          <span>⚠</span>
          <p>{error}</p>
          <button onClick={loadPaperHistory} className="retry-button">
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="final-answer-library stage2-paper-history">
      <div className="library-header">
        <h2>Stage 2 Final Answer History</h2>
        <p>Browse completed Stage 2 papers from all autonomous research sessions. This history excludes pruned and archived papers.</p>
        <div className="library-stats">
          <span className="stat-badge">
            {papers.length} {papers.length === 1 ? 'Paper' : 'Papers'}
          </span>
          <span className="stat-badge">
            {runGroups.length} {runGroups.length === 1 ? 'Research Run' : 'Research Runs'}
          </span>
          <span className="stat-badge">
            {runGroups.filter((runGroup) => runGroup.hasStage3Answer).length} Runs With Stage 3 Answer
          </span>
        </div>
      </div>

      <div className="library-controls">
        <input
          type="text"
          placeholder="Search by title, prompt, session, paper ID, or source brainstorm..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="search-input"
        />
      </div>

      {visibleRunGroups.length === 0 ? (
        <div className="fal-empty-state">
          <span className="empty-icon">📭</span>
          <h3>No Stage 2 history papers found</h3>
          <p>
            {searchTerm
              ? 'Try adjusting your search.'
              : 'Completed non-archived Stage 2 papers will appear here.'}
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
                  <span
                    className={`run-history-group-badge ${runGroup.hasStage3Answer ? 'run-history-group-badge--linked' : ''}`}
                  >
                    {runGroup.hasStage3Answer ? 'Stage 3 Final Answer Available' : 'No Stage 3 Final Answer Yet'}
                  </span>
                  {runGroup.brainstormCount !== null && runGroup.brainstormCount !== undefined && (
                    <span className="run-history-group-badge">
                      Brainstorms: {runGroup.brainstormCount}
                    </span>
                  )}
                </div>
              </div>

              <div className="run-history-group-body">
                <div className="paper-grid">
                  {runGroup.visibleStage2Papers.map((paper) => (
                    <div
                      key={paper.history_id}
                      className={`paper-card stage2-history-card ${expandedId === paper.history_id ? 'expanded' : ''}`}
                      onClick={() => handleCardClick(paper)}
                    >
                      <div className="paper-card-header">
                        <div className="stage2-history-card-identifiers">
                          <span className="paper-card-id">{paper.paper_id}</span>
                          <span className="stage2-history-session-badge">
                            {paper.session_id === 'legacy' ? 'Legacy' : paper.session_id}
                          </span>
                        </div>
                        <span className="paper-word-count">{paper.word_count?.toLocaleString()} words</span>
                      </div>

                      <div className="paper-card-title">
                        {paper.title}
                        {paper.critique_avg !== null && paper.critique_avg !== undefined && (
                          <span
                            className="stage2-history-critique-badge"
                            style={{ backgroundColor: getCritiqueColor(paper.critique_avg) }}
                            title={`Critique rating: ${paper.critique_avg}/10`}
                          >
                            ⭐ {paper.critique_avg}
                          </span>
                        )}
                      </div>

                      <div className="paper-card-abstract">
                        {truncateAbstract(paper.abstract)}
                      </div>

                      <div className="stage2-history-prompt">
                        <strong>Research Question:</strong> {paper.user_prompt}
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

                      {expandedId === paper.history_id && (
                        <>
                          <div className="paper-actions">
                            <button
                              className="btn-download"
                              onClick={(e) => handleDownloadPDF(e, paper)}
                              disabled={generatingPdfId === paper.history_id || !expandedContent}
                              title="Download as PDF"
                            >
                              {generatingPdfId === paper.history_id ? 'Preparing PDF...' : 'Download PDF'}
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
                                background: 'linear-gradient(135deg, #1eff1c 0%, #0fcc0d 100%)',
                                border: 'none',
                                color: '#0b2e0b',
                                padding: '0.35rem 0.7rem',
                                borderRadius: '4px',
                                cursor: 'pointer',
                                fontWeight: '500',
                                fontSize: '0.75rem',
                              }}
                            >
                              ⭐ Critique
                            </button>

                            {deleteConfirmId === paper.history_id ? (
                              <div className="delete-confirm-inline" onClick={(e) => e.stopPropagation()}>
                                <span>Delete this paper?</span>
                                <button
                                  className="btn-delete-confirm"
                                  onClick={() => handleDeleteConfirm(paper)}
                                  disabled={deletingId === paper.history_id}
                                >
                                  {deletingId === paper.history_id ? 'Deleting...' : 'Yes'}
                                </button>
                                <button
                                  className="btn-delete-cancel"
                                  onClick={handleDeleteCancel}
                                  disabled={deletingId === paper.history_id}
                                >
                                  Cancel
                                </button>
                              </div>
                            ) : (
                              <button
                                className="btn-delete-paper"
                                onClick={(e) => handleDeleteClick(e, paper)}
                                title="Delete this paper"
                              >
                                Delete
                              </button>
                            )}
                          </div>

                          <div className="paper-full-content">
                            {loadingContentId === paper.history_id ? (
                              <div className="loading">Loading content...</div>
                            ) : expandedContent && expandedContent.history_id === paper.history_id ? (
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
                              <div className="loading">Loading content...</div>
                            )}
                          </div>
                        </>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </section>
          ))}
        </div>
      )}

      <div className="library-footer">
        <button onClick={loadPaperHistory} className="refresh-button">
          Refresh History
        </button>
      </div>

      <PaperCritiqueModal
        isOpen={critiqueModalOpen}
        onClose={() => {
          setCritiqueModalOpen(false);
          setCritiquePaper(null);
        }}
        paperType="autonomous_paper"
        paperId={critiquePaper?.paper_id}
        paperTitle={critiquePaper?.title}
        onGenerateCritique={async (customPrompt, validatorConfig) => {
          const result = await autonomousAPI.generateHistoryPaperCritique(
            critiquePaper?.session_id,
            critiquePaper?.paper_id,
            customPrompt,
            validatorConfig
          );
          await loadPaperHistory();
          if (onCurrentSessionDataChanged) {
            await onCurrentSessionDataChanged();
          }
          return result;
        }}
        onGetCritiques={() =>
          autonomousAPI.getHistoryPaperCritiques(
            critiquePaper?.session_id,
            critiquePaper?.paper_id
          )
        }
      />
    </div>
  );
}
