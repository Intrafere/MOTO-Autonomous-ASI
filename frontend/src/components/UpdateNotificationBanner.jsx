import React, { useState, useEffect, useRef } from 'react';
import { api } from '../services/api';

const POLL_INTERVAL_MS = 1000;

export default function UpdateNotificationBanner({ notice, onDismiss }) {
  const [phase, setPhase] = useState('idle');
  const [outputLines, setOutputLines] = useState([]);
  const [errorMessage, setErrorMessage] = useState('');
  const logRef = useRef(null);
  const pollRef = useRef(null);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [outputLines]);

  const startPolling = () => {
    pollRef.current = setInterval(async () => {
      try {
        const status = await api.getPullStatus();
        setOutputLines(status.output_lines || []);
        if (status.status === 'done') {
          clearInterval(pollRef.current);
          pollRef.current = null;
          setPhase('success');
        } else if (status.status === 'error') {
          clearInterval(pollRef.current);
          pollRef.current = null;
          setErrorMessage(
            (status.output_lines || []).slice(-5).join('\n') || 'Unknown error'
          );
          setPhase('error');
        }
      } catch {
        clearInterval(pollRef.current);
        pollRef.current = null;
        setErrorMessage('Lost connection to backend while pulling.');
        setPhase('error');
      }
    }, POLL_INTERVAL_MS);
  };

  const handlePull = async () => {
    setPhase('pulling');
    setOutputLines([]);
    setErrorMessage('');
    try {
      const resp = await api.startPull();
      if (!resp.started) {
        setErrorMessage(resp.reason || 'Pull rejected by server.');
        setPhase('error');
        return;
      }
      startPolling();
    } catch (err) {
      setErrorMessage(err.message || 'Failed to start pull.');
      setPhase('error');
    }
  };

  const handleRetry = () => {
    handlePull();
  };

  if (phase === 'idle') {
    return (
      <div className="update-notice-banner">
        <div className="update-notice-content">
          <span className="update-notice-icon">&#9432;</span>
          <span className="update-notice-text">
            <strong>Update available:</strong>{' '}
            {notice.installed_version} ({notice.installed_commit})
            {' '}&rarr;{' '}
            {notice.available_version} ({notice.available_commit})
          </span>
          <div className="update-notice-actions">
            <button
              className="update-notice-pull-btn"
              onClick={handlePull}
            >
              Update
            </button>
            <button
              className="update-notice-dismiss"
              onClick={onDismiss}
              aria-label="Dismiss update notice"
              title="Dismiss"
            >
              Dismiss
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (phase === 'pulling') {
    return (
      <div className="update-notice-banner update-notice-pulling">
        <div className="update-notice-content">
          <span className="update-notice-spinner" />
          <span className="update-notice-text">
            <strong>Pulling update...</strong>
          </span>
        </div>
        <pre className="update-notice-log" ref={logRef}>
          {outputLines.length > 0
            ? outputLines.join('\n')
            : 'Waiting for output...'}
        </pre>
      </div>
    );
  }

  if (phase === 'success') {
    return (
      <div className="update-notice-banner update-notice-success">
        <div className="update-notice-content">
          <span className="update-notice-icon update-notice-icon-success">&#10003;</span>
          <span className="update-notice-text">
            <strong>Update applied!</strong>{' '}
            Restart the backend server to take effect.
          </span>
          <button
            className="update-notice-dismiss"
            onClick={onDismiss}
            aria-label="Dismiss update notice"
            title="Dismiss"
          >
            Dismiss
          </button>
        </div>
        <pre className="update-notice-log" ref={logRef}>
          {outputLines.join('\n')}
        </pre>
      </div>
    );
  }

  // phase === 'error'
  return (
    <div className="update-notice-banner update-notice-error">
      <div className="update-notice-content">
        <span className="update-notice-icon update-notice-icon-error">&#10007;</span>
        <span className="update-notice-text">
          <strong>Update failed.</strong>{' '}
          {errorMessage}
        </span>
        <div className="update-notice-actions">
          <button
            className="update-notice-pull-btn"
            onClick={handleRetry}
          >
            Retry
          </button>
          <button
            className="update-notice-dismiss"
            onClick={onDismiss}
            aria-label="Dismiss update notice"
            title="Dismiss"
          >
            Dismiss
          </button>
        </div>
      </div>
      {outputLines.length > 0 && (
        <pre className="update-notice-log" ref={logRef}>
          {outputLines.join('\n')}
        </pre>
      )}
    </div>
  );
}
