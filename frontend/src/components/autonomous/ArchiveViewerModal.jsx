import React, { useState, useEffect } from 'react';
import { api } from '../../services/api';
import LatexRenderer from '../LatexRenderer';
import './ArchiveViewerModal.css';

const IconX = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="18" y1="6" x2="6" y2="18"></line>
    <line x1="6" y1="6" x2="18" y2="18"></line>
  </svg>
);
const IconFileText = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
    <polyline points="14 2 14 8 20 8"></polyline>
    <line x1="16" y1="13" x2="8" y2="13"></line>
    <line x1="16" y1="17" x2="8" y2="17"></line>
  </svg>
);
const IconDatabase = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <ellipse cx="12" cy="5" rx="9" ry="3"></ellipse>
    <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path>
    <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path>
  </svg>
);
const IconChevronRight = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="9 18 15 12 9 6"></polyline>
  </svg>
);

export default function ArchiveViewerModal({ answerId, onClose }) {
  const [activeTab, setActiveTab] = useState('papers');
  const [papers, setPapers] = useState([]);
  const [brainstorms, setBrainstorms] = useState([]);
  const [selectedPaper, setSelectedPaper] = useState(null);
  const [selectedBrainstorm, setSelectedBrainstorm] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadArchive();
  }, [answerId]);

  const loadArchive = async () => {
    try {
      setLoading(true);
      const papersRes = await api.get(`/api/auto-research/final-answer/${answerId}/archive/papers`);
      setPapers(papersRes.papers);
      const brainstormsRes = await api.get(`/api/auto-research/final-answer/${answerId}/archive/brainstorms`);
      setBrainstorms(brainstormsRes.brainstorms);
    } catch (error) {
      console.error('Failed to load archive:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadPaperDetails = async (paperId) => {
    try {
      const res = await api.get(`/api/auto-research/final-answer/${answerId}/archive/papers/${paperId}`);
      setSelectedPaper(res);
    } catch (error) {
      console.error('Failed to load paper:', error);
    }
  };

  const loadBrainstormDetails = async (topicId) => {
    try {
      const res = await api.get(`/api/auto-research/final-answer/${answerId}/archive/brainstorms/${topicId}`);
      setSelectedBrainstorm(res);
    } catch (error) {
      console.error('Failed to load brainstorm:', error);
    }
  };

  return (
    <div className="archive-overlay">
      <div className="archive-panel">
        <div className="archive-header">
          <h2 className="archive-title">
            <IconDatabase className="archive-icon-header" />
            Research Archive (Read-Only)
          </h2>
          <button onClick={onClose} className="archive-close-btn">
            <IconX className="archive-icon-close" />
          </button>
        </div>

        <div className="archive-tabs">
          <button
            onClick={() => { setActiveTab('papers'); setSelectedPaper(null); }}
            className={`archive-tab ${activeTab === 'papers' ? 'archive-tab--active' : ''}`}
          >
            <IconFileText className="archive-tab-icon" />
            Papers ({papers.length})
          </button>
          <button
            onClick={() => { setActiveTab('brainstorms'); setSelectedBrainstorm(null); }}
            className={`archive-tab ${activeTab === 'brainstorms' ? 'archive-tab--active' : ''}`}
          >
            <IconDatabase className="archive-tab-icon" />
            Brainstorms ({brainstorms.length})
          </button>
        </div>

        <div className="archive-content">
          {loading ? (
            <div className="archive-placeholder">Loading archive...</div>
          ) : (
            <>
              {activeTab === 'papers' && (
                selectedPaper ? (
                  <PaperDetailView paper={selectedPaper} onBack={() => setSelectedPaper(null)} />
                ) : (
                  <PapersListView papers={papers} onSelectPaper={loadPaperDetails} />
                )
              )}
              {activeTab === 'brainstorms' && (
                selectedBrainstorm ? (
                  <BrainstormDetailView brainstorm={selectedBrainstorm} onBack={() => setSelectedBrainstorm(null)} />
                ) : (
                  <BrainstormsListView brainstorms={brainstorms} onSelectBrainstorm={loadBrainstormDetails} />
                )
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function PapersListView({ papers, onSelectPaper }) {
  if (papers.length === 0) {
    return <div className="archive-placeholder">No papers in archive</div>;
  }

  return (
    <div className="archive-list">
      {papers.map((paper) => (
        <div key={paper.paper_id} onClick={() => onSelectPaper(paper.paper_id)} className="archive-card">
          <div className="archive-card-row">
            <div className="archive-card-body">
              <h3 className="archive-card-title">{paper.title}</h3>
              <p className="archive-card-desc">{paper.abstract}</p>
              <div className="archive-card-meta">
                {paper.word_count} words &bull; Paper ID: {paper.paper_id}
              </div>
            </div>
            <IconChevronRight className="archive-card-chevron" />
          </div>
        </div>
      ))}
    </div>
  );
}

function PaperDetailView({ paper, onBack }) {
  return (
    <div>
      <button onClick={onBack} className="archive-back-btn">
        &larr; Back to Papers
      </button>
      <div className="archive-detail">
        <div className="archive-detail-divider">
          <span className="archive-badge-readonly">ARCHIVED - READ ONLY</span>
        </div>
        <h2 className="archive-detail-title">{paper.metadata.title}</h2>
        <div className="archive-section">
          <h3 className="archive-section-heading">Abstract</h3>
          <p className="archive-section-text">{paper.abstract}</p>
        </div>
        <div className="archive-section">
          <h3 className="archive-section-heading">Paper Content</h3>
          <div className="archive-content-viewer">
            <LatexRenderer
              content={
                paper.outline
                  ? `${paper.outline}\n\n${'='.repeat(80)}\n\n${paper.content}`
                  : paper.content
              }
              className="archive-paper-renderer"
              showToggle={true}
              defaultRaw={false}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function BrainstormsListView({ brainstorms, onSelectBrainstorm }) {
  if (brainstorms.length === 0) {
    return <div className="archive-placeholder">No brainstorms in archive</div>;
  }

  return (
    <div className="archive-list">
      {brainstorms.map((brainstorm) => (
        <div key={brainstorm.topic_id} onClick={() => onSelectBrainstorm(brainstorm.topic_id)} className="archive-card">
          <div className="archive-card-row">
            <div className="archive-card-body">
              <h3 className="archive-card-title">{brainstorm.topic_prompt}</h3>
              <div className="archive-card-desc">
                {brainstorm.submission_count} submissions &bull; Status: {brainstorm.status}
              </div>
              <div className="archive-card-meta">
                Topic ID: {brainstorm.topic_id}
              </div>
            </div>
            <IconChevronRight className="archive-card-chevron" />
          </div>
        </div>
      ))}
    </div>
  );
}

function BrainstormDetailView({ brainstorm, onBack }) {
  return (
    <div>
      <button onClick={onBack} className="archive-back-btn">
        &larr; Back to Brainstorms
      </button>
      <div className="archive-detail">
        <div className="archive-detail-divider">
          <span className="archive-badge-readonly">ARCHIVED - READ ONLY</span>
        </div>
        <h2 className="archive-detail-title">{brainstorm.metadata.topic_prompt}</h2>
        <div className="archive-section archive-section-meta">
          <div>Status: {brainstorm.metadata.status}</div>
          <div>Submissions: {brainstorm.metadata.submission_count}</div>
          <div>Topic ID: {brainstorm.topic_id}</div>
        </div>
        <div>
          <h3 className="archive-section-heading">Brainstorm Database</h3>
          <pre className="archive-pre-content">{brainstorm.content}</pre>
        </div>
      </div>
    </div>
  );
}
