import React from 'react';

export default function StatusBadge({ status, docCount, onRetry }) {
  let isHealthy = status === 'healthy';
  let isChecking = status === 'checking';

  return (
    <div className="status-badge-container">
      <div className={`status-pill ${status}`} onClick={onRetry} title="Click to refresh connection">
        <span className="status-dot"></span>
        <span className="status-text">
          {isChecking && 'CONNECTING...'}
          {isHealthy && `BACKEND READY (${docCount ?? 0} DOCS)`}
          {status === 'offline' && 'BACKEND OFFLINE'}
        </span>
      </div>
    </div>
  );
}
