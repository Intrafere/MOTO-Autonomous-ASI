/**
 * PaperLibrary - Displays grid of completed papers.
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import './AutonomousResearch.css';
import LatexRenderer from '../LatexRenderer';
import {
  PDF_UNAVAILABLE_MESSAGE,
  downloadRawText,
  downloadPDFViaBackend,
  isPDFDownloadAvailable,
  sanitizeFilename,
} from '../../utils/downloadHelpers';
import PaperCritiqueModal from '../PaperCritiqueModal';
import { autonomousAPI } from '../../services/api';
import { useProofCheckRuntime } from '../../hooks/useProofCheckRuntime';
import { getRuntimeDataPath } from '../../utils/runtimeConfig';
import { websocket } from '../../services/websocket';

const PaperLibrary = ({ papers, onRefresh, api, archivedCount = 0, capabilities }) => {
  const [expandedId, setExpandedId] = useState(null);
  const [expandedContent, setExpandedContent] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showLibraryTooltip, setShowLibraryTooltip] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteAllPrunedConfirm, setDeleteAllPrunedConfirm] = useState(false);
  const [deletingAllPruned, setDeletingAllPruned] = useState(false);
  const [isGeneratingPDF, setIsGeneratingPDF] = useState(false);
  const [currentPrunedPapers, setCurrentPrunedPapers] = useState([]);
  const pdfDownloadAvailable = isPDFDownloadAvailable(capabilities);
  const getAutonomousPaper = api?.getAutonomousPaper;
  const getCurrentSession = api?.getCurrentSession;
  const getPrunedPaperHistory = api?.getPrunedPaperHistory;
  const getPrunedHistoryPaper = api?.getPrunedHistoryPaper;
  
  // Critique modal state
  const [critiqueModalOpen, setCritiqueModalOpen] = useState(false);
  const [critiquePaper, setCritiquePaper] = useState(null);
  const [proofActionMessage, setProofActionMessage] = useState('');
  const {
    getSourceState,
    manualCheckEnabled,
    manualCheckReason,
    queueManualProofCheck,
  } = useProofCheckRuntime();

  const loadCurrentPrunedPapers = useCallback(async () => {
    if (!getPrunedPaperHistory) {
      setCurrentPrunedPapers([]);
      return;
    }

    try {
      const sessionInfo = getCurrentSession
        ? await getCurrentSession()
        : { is_active: false, session_id: null };
      const activeSessionId = sessionInfo?.is_active && sessionInfo?.session_id
        ? sessionInfo.session_id
        : 'legacy';
      const prunedHistory = await getPrunedPaperHistory();
      const currentSessionPruned = (prunedHistory.papers || [])
        .filter((paper) => paper.session_id === activeSessionId);

      setCurrentPrunedPapers(currentSessionPruned);
    } catch (error) {
      console.error('Failed to load current pruned papers:', error);
      setCurrentPrunedPapers([]);
    }
  }, [getCurrentSession, getPrunedPaperHistory]);

  useEffect(() => {
    loadCurrentPrunedPapers();
  }, [archivedCount, loadCurrentPrunedPapers]);

  const visiblePapers = useMemo(() => {
    const activePapers = (papers || []).map((paper) => ({ ...paper, is_pruned: false }));
    const activeIds = new Set(activePapers.map((paper) => paper.paper_id));
    const prunedPapers = currentPrunedPapers
      .filter((paper) => !activeIds.has(paper.paper_id))
      .map((paper) => ({ ...paper, is_pruned: true }));

    return [...activePapers, ...prunedPapers].sort((a, b) => {
      const aTime = new Date(a.created_at || a.pruned_at || 0).getTime();
      const bTime = new Date(b.created_at || b.pruned_at || 0).getTime();
      return bTime - aTime;
    });
  }, [papers, currentPrunedPapers]);

  useEffect(() => {
    const unsubscribeNovelProof = websocket.on('novel_proof_discovered', async (data) => {
      if (
        !expandedId ||
        data.source_type !== 'paper' ||
        data.source_id !== expandedId
      ) {
        return;
      }

      try {
        const refreshed = await getAutonomousPaper(expandedId);
        setExpandedContent({
          content: refreshed.content,
          outline: refreshed.outline,
          title: refreshed.title,
        });
      } catch (error) {
        console.error('Failed to refresh paper after proof append:', error);
      }
    });

    return () => unsubscribeNovelProof();
  }, [expandedId, getAutonomousPaper]);

  const handleCardClick = async (paper) => {
    const paperId = paper.paper_id;
    if (expandedId === paperId) {
      setExpandedId(null);
      setExpandedContent(null);
      return;
    }

    setExpandedId(paperId);
    setLoading(true);

    try {
      const data = paper.is_pruned && getPrunedHistoryPaper
        ? await getPrunedHistoryPaper(paper.session_id || 'legacy', paperId)
        : await getAutonomousPaper(paperId);
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
      await onRefresh();
      await loadCurrentPrunedPapers();
    } catch (error) {
      console.error('Failed to delete paper:', error);
      alert(`Failed to delete paper: ${error.message}`);
    } finally {
      setDeleting(false);
    }
  };

  const handleDeleteAllPrunedConfirm = async () => {
    if (!api.deleteAllPrunedPapers) return;
    setDeletingAllPruned(true);
    try {
      await api.deleteAllPrunedPapers();
      setDeleteAllPrunedConfirm(false);
      setCurrentPrunedPapers([]);
      await onRefresh();
    } catch (error) {
      console.error('Failed to delete pruned papers:', error);
      alert(`Failed to delete pruned papers: ${error.message}`);
    } finally {
      setDeletingAllPruned(false);
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
    
    const filename = sanitizeFilename(`${paper.is_pruned ? 'pruned_' : ''}${paper.paper_id}_${paper.title}`);
    const content = expandedContent.content || '';
    const outline = expandedContent.outline || '';
    
    downloadRawText(content, filename, outline);
  };

  const handleProofCheck = async (e, paperId) => {
    e.stopPropagation();
    try {
      setProofActionMessage('');
      await queueManualProofCheck({
        sourceType: 'paper',
        sourceId: paperId,
      });
      setProofActionMessage(`Queued proof check for paper ${paperId}.`);
    } catch (error) {
      setProofActionMessage(`Failed to queue proof check: ${error.message}`);
    }
  };

  const handleDownloadPDF = async (e, paper) => {
    e.stopPropagation();

    if (!pdfDownloadAvailable) {
      alert(PDF_UNAVAILABLE_MESSAGE);
      return;
    }

    if (!expandedContent || typeof expandedContent !== 'object') {
      alert('Paper content not loaded. Please expand the paper first.');
      return;
    }

    const filename = sanitizeFilename(`${paper.is_pruned ? 'pruned_' : ''}${paper.paper_id}_${paper.title}`);
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
      () => setIsGeneratingPDF(true),
      () => setIsGeneratingPDF(false),
      (error) => {
        setIsGeneratingPDF(false);
        console.error('PDF generation error:', error);
        alert('PDF generation failed: ' + error.message);
      },
      null,
      { pdfDownloadAvailable },
    );
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
    if (rating >= 6.25) return '#18cc17'; // Green
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

  const activePaperCount = visiblePapers.filter((paper) => !paper.is_pruned).length;
  const prunedPaperCount = visiblePapers.filter((paper) => paper.is_pruned).length;

  if (visiblePapers.length === 0) {
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
        {archivedCount > 0 && api.deleteAllPrunedPapers && (
          <div className="paper-library-pruned-actions">
            {deleteAllPrunedConfirm ? (
              <div className="delete-confirm-inline">
                <span>Delete all pruned papers permanently?</span>
                <button
                  className="btn-delete-confirm"
                  onClick={handleDeleteAllPrunedConfirm}
                  disabled={deletingAllPruned}
                >
                  {deletingAllPruned ? 'Deleting...' : 'Yes'}
                </button>
                <button
                  className="btn-delete-cancel"
                  onClick={() => setDeleteAllPrunedConfirm(false)}
                  disabled={deletingAllPruned}
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button className="btn-delete-paper" onClick={() => setDeleteAllPrunedConfirm(true)}>
                Delete All Pruned Papers
              </button>
            )}
          </div>
        )}
        <div className="auto-empty-state">
          No papers completed yet. Autonomous research will generate papers from brainstorm databases.
        </div>
      </div>
    );
  }

  return (
    <div className="paper-library">
      <div className="paper-library-header">
        <h3>
          Paper Library ({activePaperCount} Active{prunedPaperCount > 0 ? `, ${prunedPaperCount} Pruned` : ''})
          <span className="help-tooltip-anchor help-tooltip-anchor--inline">
            <button
              type="button"
              className="help-tooltip-btn"
              aria-label="Learn how the paper library works"
              onMouseEnter={() => setShowLibraryTooltip(true)}
              onMouseLeave={() => setShowLibraryTooltip(false)}
              onFocus={() => setShowLibraryTooltip(true)}
              onBlur={() => setShowLibraryTooltip(false)}
            >
              ?
            </button>
            {showLibraryTooltip && (
              <span className="help-tooltip-popup help-tooltip-popup--center">
                <strong>HOW THIS PAGE WORKS</strong>
                <br /><br />
                This paper database will continue to accumulate until the AI harness autonomously decides to generate the final answer or until the user forces final answer generation. Papers utilize their respective brainstorm topics during writing and may undergo critique-revision before final appearance on this page.
                <br /><br />
                Papers may start off mediocre, however will improve over time as the AI selects internal papers for future reference or removal. Paper quality greatly improves with higher parameter models.
                <br /><br />
                Accumulating a large amount of papers before final answer generation is normal (i.e. 10 to 20 papers with several pruned/deleted). When forcing final answer generation the AI will decide either: 1.) not enough info — brainstorm more, 2.) write answer — new short form paper, 3.) write answer, longform volume — organize select accepted papers into a longform volume with chapters, write gap papers (if applicable), conclusion chapter then introduction chapter.
                <br /><br />
                <span style={{ color: '#f0a' }}>📁 Manual file retrieval:</span> Paper files are saved at <code>{getRuntimeDataPath('auto_sessions/[session_folder]/papers')}</code> — each paper is stored as <code>paper_[id].txt</code> with a matching <code>paper_[id]_abstract.txt</code> and <code>paper_[id]_outline.txt</code>. Session folders are named after your research prompt and timestamp (e.g. <code>solve_riemann_hypothesis_2026-03-20_14-30/</code>).
              </span>
            )}
          </span>
        </h3>
        <button onClick={onRefresh} className="btn-refresh">
          Refresh
        </button>
      </div>
      <div className="paper-library-persist-warning">
        (WARNING: Any given paper may be pruned/deleted if the AI deems it to hurt the collective database quality — back up any paper you certainly want to save.)
      </div>
      <div className="paper-library-pruned-counter">
        Pruned Papers: {archivedCount}
      </div>
      {archivedCount > 0 && api.deleteAllPrunedPapers && (
        <div className="paper-library-pruned-actions">
          {deleteAllPrunedConfirm ? (
            <div className="delete-confirm-inline">
              <span>Delete all pruned papers permanently?</span>
              <button
                className="btn-delete-confirm"
                onClick={handleDeleteAllPrunedConfirm}
                disabled={deletingAllPruned}
              >
                {deletingAllPruned ? 'Deleting...' : 'Yes'}
              </button>
              <button
                className="btn-delete-cancel"
                onClick={() => setDeleteAllPrunedConfirm(false)}
                disabled={deletingAllPruned}
              >
                Cancel
              </button>
            </div>
          ) : (
            <button className="btn-delete-paper" onClick={() => setDeleteAllPrunedConfirm(true)}>
              Delete All Pruned Papers
            </button>
          )}
        </div>
      )}

      {proofActionMessage && (
        <div className={`test-result-banner ${proofActionMessage.startsWith('Failed') ? 'test-result-banner--error' : 'test-result-banner--success'}`}>
          {proofActionMessage}
        </div>
      )}

      {proofActionMessage && (
        <div className={`test-result-banner ${proofActionMessage.startsWith('Failed') ? 'test-result-banner--error' : 'test-result-banner--success'}`}>
          {proofActionMessage}
        </div>
      )}

      <div className="paper-grid">
        {visiblePapers.map((paper) => (
          <div
            key={paper.paper_id}
            className={`paper-card ${paper.is_pruned ? 'paper-card--pruned' : ''} ${expandedId === paper.paper_id ? 'expanded' : ''}`}
            onClick={() => handleCardClick(paper)}
          >
            <div className="paper-card-header">
              <div className="paper-card-identifiers">
                <span className="paper-card-id">{paper.paper_id}</span>
                {paper.is_pruned && (
                  <span className="paper-pruned-badge">Pruned Paper</span>
                )}
              </div>
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

            {paper.is_pruned && (
              <div className="paper-pruned-note">
                {paper.pruned_note || 'This paper was removed from model context and preserved for download.'}
              </div>
            )}

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
                  {!paper.is_pruned && (() => {
                    const proofCheckState = getSourceState('paper', paper.paper_id);
                    const proofCheckLabel = proofCheckState?.status === 'queued'
                      ? 'Queueing Proof Check...'
                      : proofCheckState?.status === 'running'
                        ? `Proof Check Running${proofCheckState.candidateCount ? ` (${proofCheckState.candidateCount})` : '...'}`
                        : 'Try to prove with Lean 4 theorem prover';
                    const proofCheckTitle = proofCheckState?.status === 'running'
                      ? 'A proof verification is already running for this paper.'
                      : manualCheckReason || 'Queue a manual proof check for this paper.';
                    return (
                      <button
                        className="btn-download"
                        onClick={(e) => handleProofCheck(e, paper.paper_id)}
                        disabled={!manualCheckEnabled || Boolean(proofCheckState)}
                        title={proofCheckTitle}
                      >
                        {proofCheckLabel}
                      </button>
                    );
                  })()}

                  <button
                    className="btn-download"
                    onClick={(e) => handleDownloadPDF(e, paper)}
                    disabled={isGeneratingPDF || !expandedContent || !pdfDownloadAvailable}
                    title={pdfDownloadAvailable ? 'Download as PDF' : PDF_UNAVAILABLE_MESSAGE}
                  >
                    {isGeneratingPDF ? 'Preparing PDF...' : 'Download PDF'}
                  </button>
                  
                  <button
                    className="btn-download"
                    onClick={(e) => handleDownloadRaw(e, paper)}
                    disabled={!expandedContent}
                    title="Download as raw text"
                  >
                    Download Raw
                  </button>

                  {!paper.is_pruned && (
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
                        fontSize: '0.75rem'
                      }}
                    >
                      ⭐ Critique
                    </button>
                  )}
                  
                  {!paper.is_pruned && (deleteConfirm === paper.paper_id ? (
                    <div className="delete-confirm-inline" onClick={(e) => e.stopPropagation()}>
                      <span>Prune this paper from model context?</span>
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
                      title="Prune this paper from future model context"
                    >
                      Prune
                    </button>
                  ))}
                </div>
                <div className="paper-full-content">
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

