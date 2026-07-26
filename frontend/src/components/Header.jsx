import React from 'react';
import StatusBadge from './StatusBadge';

export default function Header({ status, docCount, onRetryHealth, onClearHistory, hasHistory }) {
  return (
    <header className="app-header">
      <div className="header-container">
        <div className="header-brand">
          <img
            src="/Logo maker project - 20 July 2026 at 22.29.12.png"
            alt="Intelli Docs Logo"
            className="brand-logo-img"
          />
        </div>

        <div className="header-actions">
          <StatusBadge status={status} docCount={docCount} onRetry={onRetryHealth} />

          {hasHistory && (
            <button
              type="button"
              className="ghost-button clear-btn"
              onClick={onClearHistory}
              title="Clear conversation history"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="3 6 5 6 21 6"></polyline>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
              </svg>
              <span>Clear Session</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
