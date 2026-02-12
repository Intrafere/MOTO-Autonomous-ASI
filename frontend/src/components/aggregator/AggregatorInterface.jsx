import React, { useState, useEffect } from 'react';
import { api } from '../../services/api';
import TextFileUploader from '../TextFileUploader';

export default function AggregatorInterface({ config, setConfig }) {
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState(null);
  const [uploadedFiles, setUploadedFiles] = useState([]);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  const fetchStatus = async () => {
    try {
      const data = await api.getStatus();
      setStatus(data);
      setIsRunning(data.is_running);
    } catch (error) {
      console.error('Failed to fetch status:', error);
    }
  };

  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files);
    
    for (const file of files) {
      try {
        const result = await api.uploadFile(file);
        setUploadedFiles(prev => [...prev, result.path]);
        setConfig(prev => ({
          ...prev,
          uploadedFiles: [...prev.uploadedFiles, result.path]
        }));
      } catch (error) {
        console.error('Failed to upload file:', error);
        alert(`Failed to upload ${file.name}`);
      }
    }
  };

  const handleTextFileLoaded = (content) => {
    // Append to existing prompt with separator
    const separator = config.userPrompt.trim() ? '\n\n' : '';
    const newPrompt = config.userPrompt + separator + content;
    setConfig({ ...config, userPrompt: newPrompt });
  };

  const handleStart = async () => {
    if (!config.userPrompt.trim()) {
      alert('Please enter a user prompt');
      return;
    }
    
    // Validate submitter configs
    const submitterConfigs = config.submitterConfigs || [];
    if (submitterConfigs.length === 0) {
      alert('Please configure at least one submitter in Settings tab');
      return;
    }
    const hasInvalidSubmitter = submitterConfigs.some(s => !s.modelId);
    if (hasInvalidSubmitter) {
      alert('Please select a model for all submitters in Settings tab');
      return;
    }
    if (!config.validatorModel) {
      alert('Please select a validator model in Settings tab');
      return;
    }

    try {
      // Format submitter configs for backend (includes OpenRouter provider fields)
      const formattedConfigs = submitterConfigs.map(s => ({
        submitter_id: s.submitterId,
        provider: s.provider || 'lm_studio',
        model_id: s.modelId,
        openrouter_provider: s.openrouterProvider || null,
        lm_studio_fallback_id: s.lmStudioFallbackId || null,
        context_window: s.contextWindow,
        max_output_tokens: s.maxOutputTokens
      }));

      await api.startAggregator({
        user_prompt: config.userPrompt,
        submitter_configs: formattedConfigs,
        // Validator config with OpenRouter support
        validator_provider: config.validatorProvider || 'lm_studio',
        validator_model: config.validatorModel,
        validator_openrouter_provider: config.validatorOpenrouterProvider || null,
        validator_lm_studio_fallback: config.validatorLmStudioFallback || null,
        validator_context_size: config.validatorContextSize,
        validator_max_output_tokens: config.validatorMaxOutput || 25000,
        uploaded_files: config.uploadedFiles,
      });
      setIsRunning(true);
    } catch (error) {
      console.error('Failed to start aggregator:', error);
      alert('Failed to start aggregator. Check console for details.');
    }
  };

  const handleStop = async () => {
    try {
      await api.stopAggregator();
      setIsRunning(false);
    } catch (error) {
      console.error('Failed to stop aggregator:', error);
      alert('Failed to stop aggregator');
    }
  };

  return (
    <div>
      <h1>Aggregator Interface</h1>
      
      <div className="metric-card">
        <div className="metric-label">Status</div>
        <div className="metric-value">
          <span className={`status-badge ${isRunning ? 'status-running' : 'status-stopped'}`}>
            {isRunning ? 'Running' : 'Stopped'}
          </span>
        </div>
      </div>

      <div className="form-group">
        <label>User Prompt *</label>
        <textarea
          value={config.userPrompt}
          onChange={(e) => setConfig({ ...config, userPrompt: e.target.value })}
          placeholder='Be descriptive, this prompt should direct an open-ended brainstorming question, i.e. "Tell me all of the reasons we both should or should not be able to mathematically \"square the circle\"."'
          disabled={isRunning}
        />
        <TextFileUploader 
          onFileLoaded={handleTextFileLoaded}
          disabled={isRunning}
          maxSizeMB={5}
          showCharCount={true}
          confirmIfNotEmpty={true}
          existingPromptLength={config.userPrompt.length}
        />
      </div>

      <div className="form-group">
        <label>Upload Files (optional)</label>
        <input
          type="file"
          onChange={handleFileUpload}
          multiple
          disabled={isRunning}
        />
        {uploadedFiles.length > 0 && (
          <div style={{ marginTop: '0.5rem', color: '#4CAF50' }}>
            Uploaded {uploadedFiles.length} file(s)
          </div>
        )}
      </div>

      <div className="button-group">
        {!isRunning ? (
          <button onClick={handleStart}>Start Aggregator</button>
        ) : (
          <button onClick={handleStop} className="danger">Stop Aggregator</button>
        )}
      </div>

      {status && (
        <div className="grid-3">
          <div className="metric-card">
            <div className="metric-label">Total Submissions</div>
            <div className="metric-value">{status.total_submissions}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Accepted</div>
            <div className="metric-value" style={{ color: '#4CAF50' }}>
              {status.total_acceptances}
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Rejected</div>
            <div className="metric-value" style={{ color: '#f44336' }}>
              {status.total_rejections}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

