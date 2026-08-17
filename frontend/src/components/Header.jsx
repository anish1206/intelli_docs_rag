import React from 'react';
import StatusBadge from './StatusBadge';

export default function Header({ status, docCount, onRetryHealth, onToggleSidebar }) {
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
        </div>
      </div>
    </header>
  );
}
