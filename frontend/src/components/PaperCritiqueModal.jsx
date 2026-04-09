import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import './critique-modal.css';

// Simple inline icon components
const IconX = ({ className }) => (
  <svg className={className} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="18" y1="6" x2="6" y2="18"></line>
    <line x1="6" y1="6" x2="18" y2="18"></line>
  </svg>
);
const IconRefresh = ({ className }) => (
  <svg className={className} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M23 4v6h-6"></path>
    <path d="M1 20v-6h6"></path>
    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
  </svg>
);
const IconClock = ({ className }) => (
  <svg className={className} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10"></circle>
    <polyline points="12 6 12 12 16 14"></polyline>
  </svg>
);
const IconStar = ({ className }) => (
  <svg className={className} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
  </svg>
);
const IconChevronDown = ({ className }) => (
  <svg className={className} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="6 9 12 15 18 9"></polyline>
  </svg>
);
const IconAlertCircle = ({ className }) => (
  <svg className={className} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10"></circle>
    <line x1="12" y1="8" x2="12" y2="12"></line>
    <line x1="12" y1="16" x2="12.01" y2="16"></line>
  </svg>
);

function getRatingColor(rating) {
  if (rating >= 8) return 'critique-color--emerald';
  if (rating >= 6) return 'critique-color--blue';
  if (rating >= 4) return 'critique-color--yellow';
  if (rating >= 2) return 'critique-color--orange';
  return 'critique-color--red';
}

function getRatingBgColor(rating) {
  if (rating >= 8) return 'critique-bg--emerald';
  if (rating >= 6) return 'critique-bg--blue';
  if (rating >= 4) return 'critique-bg--yellow';
  if (rating >= 2) return 'critique-bg--orange';
  return 'critique-bg--red';
}

/**
 * Modal for displaying paper critiques from the validator model.
 * 
 * Props:
 * - isOpen: boolean - whether the modal is visible
 * - onClose: function - callback when modal is closed
 * - paperType: 'autonomous_paper' | 'final_answer' | 'compiler_paper'
 * - paperId: string - the paper ID (for autonomous papers) or answer ID (for final answers)
 * - paperTitle: string - title of the paper being critiqued
 * - onGenerateCritique: async function(customPrompt) - callback to generate a new critique
 * - onGetCritiques: async function() - callback to get critique history
 */
export default function PaperCritiqueModal({
  isOpen,
  onClose,
  paperType,
  paperId,
  paperTitle,
  onGenerateCritique,
  onGetCritiques,
}) {
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState(null);
  const [critiques, setCritiques] = useState([]);
  const [selectedCritique, setSelectedCritique] = useState(null);
  const [historyOpen, setHistoryOpen] = useState(false);

  // Load critiques when modal opens
  useEffect(() => {
    if (isOpen && onGetCritiques) {
      loadCritiques();
    }
  }, [isOpen, paperId]);

  const loadCritiques = async () => {
    try {
      setLoading(true);
      setError(null);
      const result = await onGetCritiques();
      const critiqueList = result.critiques || [];
      setCritiques(critiqueList);
      
      // Select the most recent critique if available
      if (critiqueList.length > 0) {
        setSelectedCritique(critiqueList[0]);
      } else {
        setSelectedCritique(null);
      }
    } catch (err) {
      console.error('Failed to load critiques:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateCritique = async () => {
    try {
      setGenerating(true);
      setError(null);
      
      // Get custom prompt from localStorage if available
      const storageKey = paperType === 'compiler_paper' 
        ? 'compiler_critique_custom_prompt'
        : 'autonomous_critique_custom_prompt';
      const customPrompt = localStorage.getItem(storageKey);
      
      // Get validator config from localStorage (allows critiques without starting research)
      let validatorConfig = null;
      try {
        const configKey = paperType === 'compiler_paper' ? 'compiler_settings' : 'autonomousConfig';
        const configStr = localStorage.getItem(configKey);
        if (configStr) {
          const config = JSON.parse(configStr);
          // Extract validator config fields
          validatorConfig = {
            validator_model: config.validator_model,
            validator_context_window: config.validator_context_window,
            validator_max_tokens: config.validator_max_tokens,
            validator_provider: config.validator_provider,
            validator_openrouter_provider: config.validator_openrouter_provider,
          };
        }
      } catch (e) {
        console.warn('Could not read validator config from localStorage:', e);
      }
      
      const result = await onGenerateCritique(customPrompt, validatorConfig);
      
      // Reload critiques to get the updated list
      await loadCritiques();
    } catch (err) {
      console.error('Failed to generate critique:', err);
      setError(err.message);
    } finally {
      setGenerating(false);
    }
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return 'Unknown date';
    const date = new Date(dateStr);
    return date.toLocaleString();
  };

  if (!isOpen) return null;

  // Use createPortal to render at document.body level, bypassing any parent stacking contexts
  const modalContent = (
    <div 
      className="critique-modal-overlay"
      onClick={(e) => {
        // Close when clicking the backdrop
        if (e.target === e.currentTarget) {
          onClose();
        }
      }}
    >
      <div 
        className="critique-modal-panel"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header - Compact */}
        <div className="critique-modal-header">
          <div className="critique-header-left">
            <div className="critique-header-icon">
              <IconStar className="critique-icon--green" />
            </div>
            <div>
              <h2 className="critique-modal-title">Validator Critique</h2>
              <p className="critique-modal-subtitle" title={paperTitle}>
                {paperTitle || 'Paper'}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="critique-close-btn"
          >
            <IconX className="critique-icon--close" />
          </button>
        </div>

        {/* Content - Scrollable */}
        <div className="critique-modal-body">
          {loading ? (
            <div className="critique-loading-wrapper">
              <div className="critique-loading-inner">
                <div className="critique-spinner"></div>
                <p className="critique-loading-text">Loading critiques...</p>
              </div>
            </div>
          ) : error ? (
            <div className="critique-error-box">
              <div className="critique-error-row">
                <IconAlertCircle className="critique-icon--red" style={{ flexShrink: 0, marginTop: '2px' }} />
                <div>
                  <h4 className="critique-error-title">Error</h4>
                  <p className="critique-error-message">{error}</p>
                </div>
              </div>
            </div>
          ) : selectedCritique ? (
            <div className="critique-content-layout">
              {/* Critic Identity - Compact */}
              <div className="critique-identity-card">
                <div className="critique-identity-row">
                  <div>
                    <p className="critique-identity-label">Critique by</p>
                    <p className="critique-model-name">{selectedCritique.model_id}</p>
                    {selectedCritique.host_provider && (
                      <p className="critique-host-provider">via {selectedCritique.host_provider}</p>
                    )}
                  </div>
                  <div className="critique-date-area">
                    <div className="critique-date-row">
                      <IconClock className="critique-icon--sm" />
                      {formatDate(selectedCritique.date)}
                    </div>
                  </div>
                </div>
              </div>

              {/* Ratings - Compact Grid */}
              <div className="critique-ratings-grid">
                <CompactRating label="Novelty" rating={selectedCritique.novelty_rating} feedback={selectedCritique.novelty_feedback} />
                <CompactRating label="Correctness" rating={selectedCritique.correctness_rating} feedback={selectedCritique.correctness_feedback} />
                <CompactRating label="Impact" rating={selectedCritique.impact_rating} feedback={selectedCritique.impact_feedback} />
              </div>

              {/* Full Critique - Expanded to fill space */}
              {selectedCritique.full_critique && (
                <div className="critique-full-box">
                  <h3 className="critique-section-label">Full Critique</h3>
                  <p className="critique-full-text">
                    {selectedCritique.full_critique}
                  </p>
                </div>
              )}

              {/* History - Compact */}
              {critiques.length > 1 && (
                <div className="critique-history-container">
                  <button
                    onClick={() => setHistoryOpen(!historyOpen)}
                    className="critique-history-toggle"
                  >
                    <span className="critique-history-label">
                      History ({critiques.length})
                    </span>
                    <IconChevronDown 
                      className={`critique-icon--sm critique-icon--muted critique-history-chevron ${historyOpen ? 'critique-history-chevron--open' : ''}`}
                    />
                  </button>
                  
                  {historyOpen && (
                    <div className="critique-history-list">
                      {critiques.map((critique, idx) => (
                        <button
                          key={critique.critique_id || idx}
                          onClick={() => {
                            setSelectedCritique(critique);
                            setHistoryOpen(false);
                          }}
                          className={`critique-history-item ${selectedCritique?.critique_id === critique.critique_id ? 'critique-history-item--selected' : ''}`}
                        >
                          <div className="critique-history-item-row">
                            <span className="critique-history-model">{critique.model_id}</span>
                            <span className="critique-history-date">{formatDate(critique.date)}</span>
                          </div>
                          <div className="critique-history-ratings">
                            <span className={getRatingColor(critique.novelty_rating)}>N: {critique.novelty_rating}</span>
                            <span className={getRatingColor(critique.correctness_rating)}>C: {critique.correctness_rating}</span>
                            <span className={getRatingColor(critique.impact_rating)}>I: {critique.impact_rating}</span>
                          </div>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div className="critique-empty-state">
              <div className="critique-empty-icon">
                <IconStar className="critique-icon--lg critique-icon--muted" />
              </div>
              <h3 className="critique-empty-title">No Critique Yet</h3>
              <p className="critique-empty-desc">
                Click "Generate Critique" to have your validator model provide an honest assessment of this paper.
              </p>
            </div>
          )}
        </div>

        {/* Footer - Compact */}
        <div className="critique-modal-footer">
          <p className="critique-footer-note">
            {critiques.length > 0 && 'Up to 10 critiques saved'}
          </p>
          <div className="critique-footer-actions">
            <button
              onClick={onClose}
              className="critique-btn-secondary"
            >
              Close
            </button>
            <button
              onClick={handleGenerateCritique}
              disabled={generating}
              className="critique-btn-primary"
            >
              {generating ? (
                <>
                  <div className="critique-spinner--sm"></div>
                  Generating...
                </>
              ) : (
                <>
                  <IconRefresh className="critique-icon--sm" />
                  {selectedCritique ? 'Regenerate' : 'Generate Critique'}
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  // Render via portal to document.body to bypass any parent stacking contexts
  return createPortal(modalContent, document.body);
}

/**
 * Compact rating display component for the modal
 */
function CompactRating({ label, rating, feedback }) {
  const percentage = (rating / 10) * 100;
  
  return (
    <div className="critique-compact-card">
      <div className="critique-compact-header">
        <span className="critique-compact-label">{label}</span>
        <span className={`${getRatingColor(rating)} critique-compact-value`}>
          {rating > 0 ? rating : '—'}
        </span>
      </div>
      
      {/* Progress bar */}
      <div className="critique-compact-track" style={{ marginBottom: feedback ? '6px' : '0' }}>
        <div 
          className={`${getRatingBgColor(rating)} critique-compact-fill`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      
      {/* Feedback text - full display */}
      {feedback && (
        <p className="critique-compact-feedback">{feedback}</p>
      )}
    </div>
  );
}

