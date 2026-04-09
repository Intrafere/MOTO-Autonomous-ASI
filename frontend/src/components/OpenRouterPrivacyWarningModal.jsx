import React from 'react';
import './settings-common.css';

/**
 * Modal that displays OpenRouter privacy policy warning.
 * 
 * Shows when user's OpenRouter privacy settings block access to free models.
 * Provides clear instructions on how to fix the issue.
 */
function OpenRouterPrivacyWarningModal({ isOpen, onClose, errorData }) {
  if (!isOpen) return null;

  const handleOverlayClick = (e) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  const handleOpenSettings = () => {
    window.open('https://openrouter.ai/settings/privacy', '_blank');
  };

  return (
    <div 
      className="inline-modal-overlay"
      onClick={handleOverlayClick}
      style={{
        padding: '20px',
        zIndex: 10000
      }}
    >
      <div 
        className="privacy-warning-modal"
        style={{
          backgroundColor: '#1e1e1e',
          borderRadius: '12px',
          padding: '32px',
          maxWidth: '600px',
          width: '100%',
          maxHeight: '90vh',
          overflow: 'auto',
          border: '2px solid #ff6b6b',
          boxShadow: '0 8px 32px rgba(255, 107, 107, 0.3)'
        }}
      >
        {/* Header */}
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          marginBottom: '24px',
          gap: '12px'
        }}>
          <div style={{
            fontSize: '40px',
            lineHeight: 1
          }}>⚠️</div>
          <h2 style={{ 
            margin: 0,
            fontSize: '24px',
            fontWeight: '600',
            color: '#ff6b6b'
          }}>
            OpenRouter Privacy Settings Required
          </h2>
        </div>

        {/* Error Details */}
        {errorData && (
          <div style={{
            backgroundColor: '#2a2a2a',
            borderRadius: '8px',
            padding: '16px',
            marginBottom: '24px',
            border: '1px solid #404040'
          }}>
            <div style={{ 
              fontSize: '13px',
              color: '#999',
              marginBottom: '8px',
              fontWeight: '500'
            }}>
              Model: <span style={{ color: '#61dafb' }}>{errorData.model}</span>
            </div>
            <div style={{ 
              fontSize: '13px',
              color: '#999',
              fontWeight: '500'
            }}>
              Role: <span style={{ color: '#61dafb' }}>{errorData.role_id}</span>
            </div>
          </div>
        )}

        {/* Main Message */}
        <div style={{
          marginBottom: '24px',
          lineHeight: '1.6',
          color: '#e0e0e0'
        }}>
          <p style={{ marginTop: 0, marginBottom: '16px', fontSize: '15px' }}>
            Your OpenRouter account's privacy settings are blocking access to free models.
          </p>
          <p style={{ marginBottom: '16px', fontSize: '14px', color: '#b0b0b0' }}>
            <strong style={{ color: '#fff' }}>Why this happens:</strong> Free models on OpenRouter 
            are subsidized through training data collection. You must opt-in to data sharing to use them.
          </p>
        </div>

        {/* Solution Steps */}
        <div style={{
          backgroundColor: '#2a2a2a',
          borderRadius: '8px',
          padding: '20px',
          marginBottom: '24px',
          border: '1px solid #404040'
        }}>
          <h3 style={{ 
            margin: '0 0 16px 0',
            fontSize: '16px',
            fontWeight: '600',
            color: '#4caf50'
          }}>
            ✓ How to Fix
          </h3>
          <ol style={{
            margin: 0,
            paddingLeft: '20px',
            lineHeight: '1.8',
            color: '#e0e0e0'
          }}>
            <li style={{ marginBottom: '12px' }}>
              Go to <strong style={{ color: '#61dafb' }}>https://openrouter.ai/settings/privacy</strong>
            </li>
            <li style={{ marginBottom: '12px' }}>
              Enable the option: <strong style={{ color: '#61dafb' }}>"Allow my data to be used for model training"</strong>
            </li>
            <li style={{ marginBottom: '12px' }}>
              Save your settings
            </li>
            <li>
              Come back and start your research again
            </li>
          </ol>
        </div>

        {/* Alternative Solutions */}
        <div style={{
          backgroundColor: '#2a2a2a',
          borderRadius: '8px',
          padding: '20px',
          marginBottom: '24px',
          border: '1px solid #404040'
        }}>
          <h3 style={{ 
            margin: '0 0 16px 0',
            fontSize: '16px',
            fontWeight: '600',
            color: '#ffa726'
          }}>
            Alternative Solutions
          </h3>
          <ul style={{
            margin: 0,
            paddingLeft: '20px',
            lineHeight: '1.8',
            color: '#e0e0e0'
          }}>
            <li style={{ marginBottom: '8px' }}>
              Use a paid OpenRouter model instead of a free model
            </li>
            <li>
              Configure an LM Studio fallback model in Settings
            </li>
          </ul>
        </div>

        {/* Action Buttons */}
        <div style={{
          display: 'flex',
          gap: '12px',
          justifyContent: 'flex-end'
        }}>
          <button
            onClick={handleOpenSettings}
            className="btn-success-sm"
            style={{
              padding: '12px 24px',
              fontSize: '14px',
              fontWeight: '600'
            }}
          >
            Open Privacy Settings
          </button>
          <button
            onClick={onClose}
            className="btn-ghost"
            style={{
              padding: '12px 24px',
              fontSize: '14px',
              fontWeight: '600',
              backgroundColor: '#555',
              color: 'white',
              border: 'none'
            }}
          >
            OK, I Understand
          </button>
        </div>

        {/* Footer Note */}
        <div style={{
          marginTop: '20px',
          paddingTop: '20px',
          borderTop: '1px solid #404040',
          fontSize: '12px',
          color: '#888',
          lineHeight: '1.5'
        }}>
          <strong>Note:</strong> This is an OpenRouter account setting, not a MOTO setting. 
          You only need to configure this once on OpenRouter's website.
        </div>
      </div>
    </div>
  );
}

export default OpenRouterPrivacyWarningModal;

