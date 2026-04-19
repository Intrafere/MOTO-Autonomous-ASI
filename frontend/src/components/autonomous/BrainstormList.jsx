/**
 * BrainstormList - Displays list of all brainstorm topics with rendered content.
 * Now uses LatexRenderer for proper mathematical notation display.
 */
import React, { useState, useEffect, useRef } from 'react';
import './AutonomousResearch.css';
import { websocket } from '../../services/websocket';
import LatexRenderer from '../LatexRenderer';
import { prependDisclaimer } from '../../utils/disclaimerHelper';

const BrainstormList = ({ brainstorms, onRefresh, api }) => {
  const [expandedId, setExpandedId] = useState(null);
  const [fileContent, setFileContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [showLatex, setShowLatex] = useState(true);
  const [userChoseLatex, setUserChoseLatex] = useState(false);
  const unsubscribeRef = useRef(null);

  // Auto-disable LaTeX rendering when brainstorm grows large (>50k chars).
  // Only fires if the user has not explicitly toggled the LaTeX checkbox.
  useEffect(() => {
    if (!userChoseLatex && fileContent && fileContent.length > 50000) {
      setShowLatex(false);
    }
  }, [fileContent, userChoseLatex]);

  // Auto-refresh expanded brainstorm every 2 seconds
  useEffect(() => {
    if (!expandedId || !autoRefresh) return;

    const refreshContent = async () => {
      try {
        const data = await api.getBrainstorm(expandedId);
        setFileContent(data.content || 'No content yet...');
      } catch (error) {
        console.error('Auto-refresh failed:', error);
      }
    };

    // Initial load
    refreshContent();

    // Set up interval
    const interval = setInterval(refreshContent, 5000);
    return () => clearInterval(interval);
  }, [expandedId, autoRefresh, api]);

  // Subscribe to WebSocket events for immediate updates
  useEffect(() => {
    const handleSubmissionAccepted = async (data) => {
      if (expandedId && data.topic_id === expandedId) {
        try {
          const refreshedData = await api.getBrainstorm(expandedId);
          setFileContent(refreshedData.content || 'No content yet...');
        } catch (error) {
          console.error('Failed to refresh submissions:', error);
        }
      }
    };

    unsubscribeRef.current = websocket.on('brainstorm_submission_accepted', handleSubmissionAccepted);
    return () => {
      if (unsubscribeRef.current) {
        unsubscribeRef.current();
      }
    };
  }, [expandedId, api]);

  const handleCardClick = async (topicId) => {
    if (expandedId === topicId) {
      setExpandedId(null);
      setFileContent('');
      return;
    }

    setExpandedId(topicId);
    setLoading(true);

    try {
      const data = await api.getBrainstorm(topicId);
      setFileContent(data.content || 'No content yet...');
    } catch (error) {
      console.error('Failed to load brainstorm content:', error);
      setFileContent('Error loading brainstorm');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteClick = (e, topicId) => {
    e.stopPropagation();
    setDeleteConfirm(topicId);
  };

  const handleDeleteConfirm = async (topicId) => {
    setDeleting(true);
    try {
      await api.deleteBrainstorm(topicId);
      setDeleteConfirm(null);
      onRefresh();
    } catch (error) {
      console.error('Failed to delete brainstorm:', error);
      alert(`Failed to delete brainstorm: ${error.message}`);
    } finally {
      setDeleting(false);
    }
  };

  const handleDeleteCancel = (e) => {
    e.stopPropagation();
    setDeleteConfirm(null);
  };

  const handleDownload = (e, brainstorm) => {
    e.stopPropagation();
    if (!fileContent) return;
    
    const blob = new Blob([prependDisclaimer(fileContent, 'brainstorm')], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${brainstorm.topic_id}_brainstorm.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return 'N/A';
    return new Date(dateStr).toLocaleString();
  };

  if (!brainstorms || brainstorms.length === 0) {
    return (
      <div className="brainstorm-list">
        <div className="brainstorm-list-header">
          <h3>Brainstorm Topics</h3>
          <button onClick={onRefresh} className="btn-refresh">
            Refresh
          </button>
        </div>
        <div className="brainstorm-list-warning">
          (WARNING: Any given brainstorm idea may be pruned/deleted if the AI deems it to hurt the collective database quality)
        </div>
        <div className="auto-empty-state">
          No brainstorm topics yet. Start autonomous research to create brainstorms.
        </div>
      </div>
    );
  }

  return (
    <div className="brainstorm-list">
      <div className="brainstorm-list-header">
        <h3>Brainstorm Topics ({brainstorms.length})</h3>
        <button onClick={onRefresh} className="btn-refresh">
          Refresh
        </button>
      </div>
      <div className="brainstorm-list-warning">
        (WARNING: Any given brainstorm idea may be pruned/deleted if the AI deems it to hurt the collective database quality. These brainstorms are the real powerhouse behind the ASI creativity! The brainstorms themselves often contain many great ideas that get turned into the stage 2 papers.)
      </div>

      {brainstorms.map((brainstorm) => (
        <div
          key={brainstorm.topic_id}
          className={`brainstorm-card ${expandedId === brainstorm.topic_id ? 'expanded' : ''}`}
        >
          <div 
            className="brainstorm-card-clickable"
            onClick={() => handleCardClick(brainstorm.topic_id)}
          >
            <div className="brainstorm-card-header">
              <span className="brainstorm-card-id">{brainstorm.topic_id}</span>
              <span className={`brainstorm-status ${brainstorm.status === 'complete' ? 'complete' : 'in-progress'}`}>
                {brainstorm.status === 'complete' ? '✓ Complete' : '↻ In Progress'}
              </span>
            </div>

            <div className="brainstorm-card-prompt">
              {brainstorm.topic_prompt}
            </div>

            <div className="brainstorm-card-meta">
              <span>{brainstorm.submission_count} submissions</span>
              <span>{brainstorm.papers_generated?.length || 0}/3 papers</span>
              <span>Last: {formatDate(brainstorm.last_activity)}</span>
            </div>

            {deleteConfirm === brainstorm.topic_id ? (
              <div className="delete-confirm" onClick={(e) => e.stopPropagation()}>
                <p>Delete this brainstorm{brainstorm.papers_generated?.length > 0 ? ` and ${brainstorm.papers_generated.length} paper(s)` : ''}?</p>
                <div className="delete-confirm-buttons">
                  <button 
                    className="btn-delete-confirm" 
                    onClick={() => handleDeleteConfirm(brainstorm.topic_id)}
                    disabled={deleting}
                  >
                    {deleting ? 'Deleting...' : 'Delete'}
                  </button>
                  <button 
                    className="btn-delete-cancel" 
                    onClick={handleDeleteCancel}
                    disabled={deleting}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <button 
                className="btn-delete-brainstorm"
                onClick={(e) => handleDeleteClick(e, brainstorm.topic_id)}
                title="Delete brainstorm and associated papers"
              >
                Delete
              </button>
            )}
          </div>

          {expandedId === brainstorm.topic_id && (
            <div className="brainstorm-submissions-container">
              {loading ? (
                <div className="loading">Loading...</div>
              ) : (
                <div className="file-content">
                  <div className="submissions-list-header">
                    <h4>Database Content</h4>
                    <div className="brainstorm-content-controls">
                      <label className="toggle-label" onClick={(e) => e.stopPropagation()}>
                        <input
                          type="checkbox"
                          checked={showLatex}
                          onChange={(e) => { setUserChoseLatex(true); setShowLatex(e.target.checked); }}
                        />
                        LaTeX Rendering
                      </label>
                      <button
                        className="btn-download-small"
                        onClick={(e) => handleDownload(e, brainstorm)}
                        title="Download brainstorm content"
                      >
                        Download
                      </button>
                    </div>
                  </div>
                  <div className="brainstorm-content-viewer" onClick={(e) => e.stopPropagation()}>
                    <LatexRenderer
                      content={prependDisclaimer(fileContent, 'brainstorm')}
                      className="brainstorm-latex-renderer"
                      showToggle={false}
                      showLatex={showLatex}
                    />
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default BrainstormList;
