/**
 * API service for backend communication
 */

const API_BASE = '/api';

// Aggregator API
export const api = {
  // Get available models from LM Studio
  async getModels() {
    const response = await fetch(`${API_BASE}/aggregator/models`);
    if (!response.ok) throw new Error('Failed to fetch models');
    return response.json();
  },

  // Start aggregator
  async startAggregator(config) {
    const response = await fetch(`${API_BASE}/aggregator/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (!response.ok) throw new Error('Failed to start aggregator');
    return response.json();
  },

  // Stop aggregator
  async stopAggregator() {
    const response = await fetch(`${API_BASE}/aggregator/stop`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error('Failed to stop aggregator');
    return response.json();
  },

  // Get status
  async getStatus() {
    const response = await fetch(`${API_BASE}/aggregator/status`);
    if (!response.ok) throw new Error('Failed to get status');
    return response.json();
  },

  // Get results
  async getResults() {
    const response = await fetch(`${API_BASE}/aggregator/results`);
    if (!response.ok) throw new Error('Failed to get results');
    return response.json();
  },

  // Save results
  async saveResults() {
    const response = await fetch(`${API_BASE}/aggregator/save-results`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error('Failed to save results');
    return response.json();
  },

  // Clear all submissions
  async clearAllSubmissions() {
    const response = await fetch(`${API_BASE}/aggregator/clear-all`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error('Failed to clear submissions');
    return response.json();
  },

  // Upload file
  async uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${API_BASE}/aggregator/upload-file`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) throw new Error('Failed to upload file');
    return response.json();
  },

  // Get aggregator settings
  async getSettings() {
    const response = await fetch(`${API_BASE}/aggregator/settings`);
    if (!response.ok) throw new Error('Failed to fetch settings');
    return { data: await response.json() };
  },
  
  // Wolfram Alpha API
  async getWolframStatus() {
    const response = await fetch(`${API_BASE}/compiler/wolfram/status`);
    if (!response.ok) throw new Error('Failed to get Wolfram Alpha status');
    return response.json();
  },
  
  async setWolframApiKey(apiKey) {
    const response = await fetch(`${API_BASE}/compiler/wolfram/set-api-key`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: apiKey }),
    });
    if (!response.ok) throw new Error('Failed to set Wolfram Alpha API key');
    return response.json();
  },
  
  async clearWolframApiKey() {
    const response = await fetch(`${API_BASE}/compiler/wolfram/api-key`, {
      method: 'DELETE',
    });
    if (!response.ok) throw new Error('Failed to clear Wolfram Alpha API key');
    return response.json();
  },
  
  async testWolframQuery(request) {
    const response = await fetch(`${API_BASE}/compiler/wolfram/test-query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    if (!response.ok) throw new Error('Failed to test Wolfram Alpha query');
    return response.json();
  },
};

// Export aggregatorAPI as alias for backward compatibility
export const aggregatorAPI = api;

// Compiler API
export const compilerAPI = {
  // Get available models (reuse from main API)
  async getModels() {
    const response = await fetch(`${API_BASE}/aggregator/models`);
    if (!response.ok) throw new Error('Failed to fetch models');
    return { data: await response.json() };
  },

  // Start compiler
  async start(config) {
    const response = await fetch(`${API_BASE}/compiler/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (!response.ok) {
      const errorData = await response.json();
      const error = new Error('Failed to start compiler');
      error.details = errorData.detail;
      throw error;
    }
    return { data: await response.json() };
  },

  // Test models compatibility
  async testModels(config) {
    const response = await fetch(`${API_BASE}/compiler/test-models`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (!response.ok) throw new Error('Failed to test models');
    return { data: await response.json() };
  },

  // Stop compiler
  async stop() {
    const response = await fetch(`${API_BASE}/compiler/stop`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error('Failed to stop compiler');
    return { data: await response.json() };
  },

  // Get status
  async getStatus() {
    const response = await fetch(`${API_BASE}/compiler/status`);
    if (!response.ok) throw new Error('Failed to get status');
    return { data: await response.json() };
  },

  // Get paper
  async getPaper() {
    const response = await fetch(`${API_BASE}/compiler/paper`);
    if (!response.ok) throw new Error('Failed to get paper');
    return { data: await response.json() };
  },

  // Get outline
  async getOutline() {
    const response = await fetch(`${API_BASE}/compiler/outline`);
    if (!response.ok) throw new Error('Failed to get outline');
    return { data: await response.json() };
  },

  // Get previous versions
  async getPreviousVersions() {
    const response = await fetch(`${API_BASE}/compiler/previous-versions`);
    if (!response.ok) throw new Error('Failed to get previous versions');
    return { data: await response.json() };
  },

  // Save paper
  async savePaper() {
    const response = await fetch(`${API_BASE}/compiler/save-paper`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error('Failed to save paper');
    return { data: await response.json() };
  },

  // Get metrics
  async getMetrics() {
    const response = await fetch(`${API_BASE}/compiler/metrics`);
    if (!response.ok) throw new Error('Failed to get metrics');
    return { data: await response.json() };
  },

  // Clear paper
  async clearPaper() {
    const response = await fetch(`${API_BASE}/compiler/clear-paper?confirm=true`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error('Failed to clear paper');
    return { data: await response.json() };
  },

  // Skip critique phase
  async skipCritique() {
    const response = await fetch(`${API_BASE}/compiler/skip-critique`, {
      method: 'POST',
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to skip critique');
    }
    return { data: await response.json() };
  },

  // ============================================================
  // Paper Critique API (Validator Critique Feature)
  // ============================================================

  // Generate a critique for the current compiler paper
  async generateCritique(customPrompt = null, validatorConfig = null) {
    const body = { custom_prompt: customPrompt };
    if (validatorConfig) {
      body.validator_model = validatorConfig.validator_model;
      body.validator_context_window = validatorConfig.validator_context_window;
      body.validator_max_tokens = validatorConfig.validator_max_tokens;
      body.validator_provider = validatorConfig.validator_provider;
      body.validator_openrouter_provider = validatorConfig.validator_openrouter_provider;
    }

    const response = await fetch(`${API_BASE}/compiler/critique-paper`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to generate critique');
    }
    return { data: await response.json() };
  },

  // Get all critiques for the current compiler paper
  async getCritiques() {
    const response = await fetch(`${API_BASE}/compiler/critiques`);
    if (!response.ok) throw new Error('Failed to get critiques');
    return { data: await response.json() };
  },

  // Clear all critiques for the current compiler paper
  async clearCritiques() {
    const response = await fetch(`${API_BASE}/compiler/critiques?confirm=true`, {
      method: 'DELETE',
    });
    if (!response.ok) throw new Error('Failed to clear critiques');
    return { data: await response.json() };
  },

  // Get default critique prompt
  async getDefaultCritiquePrompt() {
    const response = await fetch(`${API_BASE}/compiler/default-critique-prompt`);
    if (!response.ok) throw new Error('Failed to get default critique prompt');
    return { data: await response.json() };
  },
};

// Autonomous Research API
export const autonomousAPI = {
  // Start autonomous research
  async start(config) {
    const response = await fetch(`${API_BASE}/auto-research/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (!response.ok) {
      const errorData = await response.json();
      const error = new Error('Failed to start autonomous research');
      error.details = errorData.detail;
      throw error;
    }
    return response.json();
  },

  // Stop autonomous research
  async stop() {
    const response = await fetch(`${API_BASE}/auto-research/stop`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error('Failed to stop autonomous research');
    return response.json();
  },

  // Clear all autonomous research data
  async clear() {
    const response = await fetch(`${API_BASE}/auto-research/clear?confirm=true`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error('Failed to clear autonomous research data');
    return response.json();
  },

  // Get status
  async getStatus() {
    const response = await fetch(`${API_BASE}/auto-research/status`);
    if (!response.ok) throw new Error('Failed to get autonomous status');
    return response.json();
  },

  // Get all brainstorms
  async getBrainstorms() {
    const response = await fetch(`${API_BASE}/auto-research/brainstorms`);
    if (!response.ok) throw new Error('Failed to get brainstorms');
    return response.json();
  },

  // Get all papers
  async getPapers() {
    const response = await fetch(`${API_BASE}/auto-research/papers`);
    if (!response.ok) throw new Error('Failed to get papers');
    return response.json();
  },

  // Get specific brainstorm
  async getBrainstorm(topicId) {
    const response = await fetch(`${API_BASE}/auto-research/brainstorm/${topicId}`);
    if (!response.ok) throw new Error(`Failed to get brainstorm ${topicId}`);
    return response.json();
  },

  // Get specific paper (alias for getAutonomousPaper)
  async getPaper(paperId) {
    return this.getAutonomousPaper(paperId);
  },

  // Get specific autonomous paper with outline
  async getAutonomousPaper(paperId) {
    const response = await fetch(`${API_BASE}/auto-research/paper/${paperId}`);
    if (!response.ok) throw new Error(`Failed to get paper ${paperId}`);
    return response.json();
  },

  // Get statistics
  async getStats() {
    const response = await fetch(`${API_BASE}/auto-research/stats`);
    if (!response.ok) throw new Error('Failed to get autonomous stats');
    return response.json();
  },

  // Get current paper progress (in-progress paper during Tier 2)
  async getCurrentPaperProgress() {
    const response = await fetch(`${API_BASE}/auto-research/current-paper-progress`);
    if (!response.ok) throw new Error('Failed to get current paper progress');
    return response.json();
  },

  // Force paper writing (manual override)
  async forcePaperWriting() {
    const response = await fetch(`${API_BASE}/auto-research/force-paper-writing`, {
      method: 'POST',
    });
    if (!response.ok) {
      const errorData = await response.json();
      const error = new Error('Failed to force paper writing');
      error.details = errorData.detail;
      throw error;
    }
    return response.json();
  },

  // Skip critique phase (manual override)
  async skipCritique() {
    const response = await fetch(`${API_BASE}/auto-research/skip-critique`, {
      method: 'POST',
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to skip critique');
    }
    return { data: await response.json() };
  },

  // Reset current paper (delete and retry from scratch)
  async resetAutonomousPaper() {
    const response = await fetch(`${API_BASE}/auto-research/reset-current-paper?confirm=true`, {
      method: 'POST',
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to reset paper');
    }
    return { data: await response.json() };
  },

  // Force Tier 3 Final Answer (manual override)
  async forceTier3(mode = 'complete_current') {
    const response = await fetch(`${API_BASE}/auto-research/force-tier3?mode=${mode}`, {
      method: 'POST',
    });
    if (!response.ok) {
      const errorData = await response.json();
      const error = new Error('Failed to force Tier 3');
      error.details = errorData.detail;
      throw error;
    }
    return response.json();
  },

  // Get all research sessions
  async getSessions() {
    const response = await fetch(`${API_BASE}/auto-research/sessions`);
    if (!response.ok) throw new Error('Failed to get sessions');
    return response.json();
  },

  // Get current session info
  async getCurrentSession() {
    const response = await fetch(`${API_BASE}/auto-research/current-session`);
    if (!response.ok) throw new Error('Failed to get current session');
    return response.json();
  },

  // Delete brainstorm
  async deleteBrainstorm(topicId) {
    const response = await fetch(`${API_BASE}/auto-research/brainstorm/${topicId}?confirm=true`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `Failed to delete brainstorm ${topicId}`);
    }
    return response.json();
  },

  // Delete paper
  async deletePaper(paperId) {
    const response = await fetch(`${API_BASE}/auto-research/paper/${paperId}?confirm=true`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `Failed to delete paper ${paperId}`);
    }
    return response.json();
  },

  // ============================================================
  // Tier 3 - Final Answer API
  // ============================================================

  // Get final answer status
  async getFinalAnswerStatus() {
    const response = await fetch(`${API_BASE}/auto-research/tier3/status`);
    if (!response.ok) throw new Error('Failed to get final answer status');
    return response.json();
  },

  // Get final answer content (for both long-form volumes and short-form papers)
  async getFinalAnswerVolume() {
    const response = await fetch(`${API_BASE}/auto-research/tier3/final-answer`);
    if (!response.ok) throw new Error('Failed to get final answer');
    return response.json();
  },

  // Get final answer paper (for short-form answers) - same endpoint handles both
  async getFinalAnswerPaper() {
    const response = await fetch(`${API_BASE}/auto-research/tier3/final-answer`);
    if (!response.ok) throw new Error('Failed to get final answer');
    return response.json();
  },

  // Get volume progress (detailed chapter status for long-form)
  async getVolumeProgress() {
    const response = await fetch(`${API_BASE}/auto-research/tier3/volume-progress`);
    if (!response.ok) throw new Error('Failed to get volume progress');
    return response.json();
  },

  // Get Tier 3 rejections
  async getTier3Rejections(phase = null) {
    const url = phase 
      ? `${API_BASE}/auto-research/tier3/rejections?phase=${phase}`
      : `${API_BASE}/auto-research/tier3/rejections`;
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to get Tier 3 rejections');
    return response.json();
  },

  // Clear Tier 3 data
  async clearTier3Data() {
    const response = await fetch(`${API_BASE}/auto-research/tier3/clear?confirm=true`, {
      method: 'POST',
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to clear Tier 3 data');
    }
    return response.json();
  },

  // ============================================================
  // API CALL LOGGING
  // ============================================================

  // Get API logs
  async getApiLogs(limit = 100) {
    const response = await fetch(`${API_BASE}/auto-research/api-logs?limit=${limit}`);
    if (!response.ok) throw new Error('Failed to get API logs');
    return response.json();
  },

  // Clear API logs
  async clearApiLogs() {
    const response = await fetch(`${API_BASE}/auto-research/api-logs/clear`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error('Failed to clear API logs');
    return response.json();
  },

  // Get API stats
  async getApiStats() {
    const response = await fetch(`${API_BASE}/auto-research/api-logs/stats`);
    if (!response.ok) throw new Error('Failed to get API stats');
    return response.json();
  },

  // ============================================================
  // Paper Critique API (Validator Critique Feature)
  // ============================================================

  // Generate a critique for an autonomous paper
  // validatorConfig is optional - if not provided, will try to use coordinator's stored config
  async generatePaperCritique(paperId, customPrompt = null, validatorConfig = null) {
    const body = {};
    if (customPrompt) body.custom_prompt = customPrompt;
    
    // Include validator config if provided AND validator_model is non-empty
    // (allows critiques without starting research)
    if (validatorConfig && validatorConfig.validator_model) {
      body.validator_model = validatorConfig.validator_model;
      body.validator_context_window = validatorConfig.validator_context_window;
      body.validator_max_tokens = validatorConfig.validator_max_tokens;
      body.validator_provider = validatorConfig.validator_provider;
      body.validator_openrouter_provider = validatorConfig.validator_openrouter_provider;
    }
    
    const response = await fetch(`${API_BASE}/auto-research/paper/${paperId}/critique`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to generate critique');
    }
    return response.json();
  },

  // Get all critiques for an autonomous paper
  async getPaperCritiques(paperId) {
    const response = await fetch(`${API_BASE}/auto-research/paper/${paperId}/critiques`);
    if (!response.ok) throw new Error('Failed to get paper critiques');
    return response.json();
  },

  // Clear all critiques for an autonomous paper
  async clearPaperCritiques(paperId) {
    const response = await fetch(`${API_BASE}/auto-research/paper/${paperId}/critiques?confirm=true`, {
      method: 'DELETE',
    });
    if (!response.ok) throw new Error('Failed to clear paper critiques');
    return response.json();
  },

  // Generate a critique for a final answer
  // validatorConfig is optional - if not provided, will try to use coordinator's stored config
  async generateFinalAnswerCritique(answerId, customPrompt = null, validatorConfig = null) {
    const body = {};
    if (customPrompt) body.custom_prompt = customPrompt;
    
    // Include validator config if provided AND validator_model is non-empty
    // (allows critiques without starting research)
    if (validatorConfig && validatorConfig.validator_model) {
      body.validator_model = validatorConfig.validator_model;
      body.validator_context_window = validatorConfig.validator_context_window;
      body.validator_max_tokens = validatorConfig.validator_max_tokens;
      body.validator_provider = validatorConfig.validator_provider;
      body.validator_openrouter_provider = validatorConfig.validator_openrouter_provider;
    }
    
    const response = await fetch(`${API_BASE}/auto-research/final-answer-library/${answerId}/critique`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to generate critique');
    }
    return response.json();
  },

  // Get all critiques for a final answer
  async getFinalAnswerCritiques(answerId) {
    const response = await fetch(`${API_BASE}/auto-research/final-answer-library/${answerId}/critiques`);
    if (!response.ok) throw new Error('Failed to get final answer critiques');
    return response.json();
  },

  // Clear all critiques for a final answer
  async clearFinalAnswerCritiques(answerId) {
    const response = await fetch(`${API_BASE}/auto-research/final-answer-library/${answerId}/critiques?confirm=true`, {
      method: 'DELETE',
    });
    if (!response.ok) throw new Error('Failed to clear final answer critiques');
    return response.json();
  },

  // Get default critique prompt
  async getDefaultCritiquePrompt() {
    const response = await fetch(`${API_BASE}/auto-research/default-critique-prompt`);
    if (!response.ok) throw new Error('Failed to get default critique prompt');
    return response.json();
  },
};

// Boost API
export const boostAPI = {
  // Enable boost
  async enable(config) {
    const response = await fetch(`${API_BASE}/boost/enable`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to enable boost');
    }
    return response.json();
  },

  // Update boost model/config without clearing boost state
  async updateModel(config) {
    const response = await fetch(`${API_BASE}/boost/update-model`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to update boost model');
    }
    return response.json();
  },

  // Disable boost
  async disable() {
    const response = await fetch(`${API_BASE}/boost/disable`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error('Failed to disable boost');
    return response.json();
  },

  // Get boost status
  async getStatus() {
    const response = await fetch(`${API_BASE}/boost/status`);
    if (!response.ok) throw new Error('Failed to get boost status');
    return response.json();
  },

  // Toggle task boost (per-task mode - legacy)
  async toggleTask(taskId) {
    const response = await fetch(`${API_BASE}/boost/toggle-task/${taskId}`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error('Failed to toggle task boost');
    return response.json();
  },

  // Get OpenRouter models
  async getOpenRouterModels(apiKey) {
    const response = await fetch(`${API_BASE}/boost/openrouter-models`, {
      headers: apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {}
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to fetch OpenRouter models');
    }
    return response.json();
  },

  // Get providers for a specific model
  async getModelProviders(apiKey, modelId) {
    const response = await fetch(
      `${API_BASE}/boost/model-providers?model_id=${encodeURIComponent(modelId)}`,
      { headers: apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {} }
    );
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to fetch model providers');
    }
    return response.json();
  },

  // ============================================================
  // NEW: Boost Next X Calls (Counter-based mode)
  // ============================================================
  
  // Set the number of next API calls to boost
  async setNextCount(count) {
    const response = await fetch(`${API_BASE}/boost/set-next-count`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ count }),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to set boost count');
    }
    return response.json();
  },

  // ============================================================
  // NEW: Category Boost (Role-based mode)
  // ============================================================

  // Toggle boost for a category (role prefix)
  async toggleCategory(category) {
    const response = await fetch(`${API_BASE}/boost/toggle-category/${category}`, {
      method: 'POST',
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to toggle category boost');
    }
    return response.json();
  },

  // Get available categories for the current mode
  async getCategories(mode = 'all') {
    const response = await fetch(`${API_BASE}/boost/categories?mode=${mode}`);
    if (!response.ok) throw new Error('Failed to get boost categories');
    return response.json();
  },

  // ============================================================
  // NEW: Boost Logs
  // ============================================================

  // Get boost API call logs
  async getLogs(limit = 100) {
    const response = await fetch(`${API_BASE}/boost/logs?limit=${limit}`);
    if (!response.ok) throw new Error('Failed to get boost logs');
    return response.json();
  },

  // Get a specific log entry with full response
  async getLogEntry(index) {
    const response = await fetch(`${API_BASE}/boost/logs/${index}`);
    if (!response.ok) throw new Error('Failed to get log entry');
    return response.json();
  },

  // Clear all boost logs
  async clearLogs() {
    const response = await fetch(`${API_BASE}/boost/clear-logs`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error('Failed to clear boost logs');
    return response.json();
  },
};

// Workflow API
export const workflowAPI = {
  // Get workflow predictions
  async getPredictions() {
    const response = await fetch(`${API_BASE}/workflow/predictions`);
    if (!response.ok) throw new Error('Failed to get workflow predictions');
    return response.json();
  },

  // Get workflow history
  async getHistory(limit = 50) {
    const response = await fetch(`${API_BASE}/workflow/history?limit=${limit}`);
    if (!response.ok) throw new Error('Failed to get workflow history');
    return response.json();
  },
};

// OpenRouter API (for per-role model selection)
export const openRouterAPI = {
  // Check if LM Studio is available
  async checkLMStudioAvailability() {
    const response = await fetch(`${API_BASE}/openrouter/lm-studio-availability`);
    if (!response.ok) throw new Error('Failed to check LM Studio availability');
    return response.json();
  },

  // Get API key status (has_key, enabled)
  async getApiKeyStatus() {
    const response = await fetch(`${API_BASE}/openrouter/api-key-status`);
    if (!response.ok) throw new Error('Failed to get API key status');
    return response.json();
  },

  // Set the global OpenRouter API key
  async setApiKey(apiKey) {
    const response = await fetch(`${API_BASE}/openrouter/set-api-key`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: apiKey }),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to set API key');
    }
    return response.json();
  },

  // Clear the global OpenRouter API key
  async clearApiKey() {
    const response = await fetch(`${API_BASE}/openrouter/api-key`, {
      method: 'DELETE',
    });
    if (!response.ok) throw new Error('Failed to clear API key');
    return response.json();
  },

  // Test connection with an API key (doesn't save it)
  async testConnection(apiKey) {
    const response = await fetch(`${API_BASE}/openrouter/test-connection`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: apiKey }),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to test connection');
    }
    return response.json();
  },

  // Get available OpenRouter models (uses stored key or provided key)
  async getModels(apiKey = null, freeOnly = false) {
    const params = new URLSearchParams();
    if (apiKey) params.append('api_key', apiKey);
    if (freeOnly) params.append('free_only', 'true');
    
    const url = `${API_BASE}/openrouter/models${params.toString() ? '?' + params.toString() : ''}`;
    const response = await fetch(url);
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to fetch models');
    }
    return response.json();
  },

  // Get providers for a specific model
  async getProviders(modelId, apiKey = null) {
    const url = `${API_BASE}/openrouter/providers/${encodeURIComponent(modelId)}`;
    const response = await fetch(url, {
      headers: apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {}
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to fetch providers');
    }
    return response.json();
  },
};

// Add helper methods to main api object
api.post = async (url, data) => {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: data ? JSON.stringify(data) : undefined,
  });
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || 'Request failed');
  }
  return response.json();
};

api.get = async (url) => {
  const response = await fetch(url);
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || 'Request failed');
  }
  return response.json();
};

