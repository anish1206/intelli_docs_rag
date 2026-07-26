import React, { useState } from 'react';

export default function SourceCard({ source, index }) {
  const [isExpanded, setIsExpanded] = useState(false);

  const metadata = source.metadata || {};
  const filename = metadata.filename || metadata.source_file || 'Document';
  const pageNo = metadata.page_no ?? metadata.source_page_range ?? null;
  const headings = metadata.headings || null;
  const fileType = metadata.file_type || (filename.includes('.') ? filename.split('.').pop() : 'txt');
  
  // Format similarity score to percentage
  const rawSim = source.similarity;
  let matchPercentage = null;
  if (typeof rawSim === 'number' && !isNaN(rawSim)) {
    // If cosine similarity is between 0 and 1
    const pct = Math.max(0, Math.min(100, Math.round(rawSim * 100)));
    matchPercentage = `${pct}% Match`;
  }

  return (
    <div className={`source-card ${isExpanded ? 'expanded' : ''}`}>
      <div className="source-card-header" onClick={() => setIsExpanded(!isExpanded)}>
        <div className="source-card-title-group">
          <span className="source-index-badge">#{index + 1}</span>
          <span className="source-file-icon">📄</span>
          <span className="source-filename" title={filename}>{filename}</span>
          
          {pageNo !== null && pageNo !== undefined && pageNo !== 'N/A' && (
            <span className="source-meta-tag">Page {pageNo}</span>
          )}

          {fileType && (
            <span className="source-filetype-badge">{fileType.toUpperCase()}</span>
          )}
        </div>

        <div className="source-card-controls">
          {matchPercentage && (
            <span className="source-match-badge">{matchPercentage}</span>
          )}
          
          <button type="button" className="source-toggle-btn" aria-label="Toggle excerpt view">
            <svg 
              className={`chevron-icon ${isExpanded ? 'open' : ''}`} 
              width="16" 
              height="16" 
              viewBox="0 0 24 24" 
              fill="none" 
              stroke="currentColor" 
              strokeWidth="2"
            >
              <polyline points="6 9 12 15 18 9"></polyline>
            </svg>
          </button>
        </div>
      </div>

      {headings && (
        <div className="source-section-header">
          <span className="section-label">SECTION:</span> {headings}
        </div>
      )}

      {isExpanded && (
        <div className="source-card-body">
          <div className="source-excerpt">
            <pre>{source.content}</pre>
          </div>
        </div>
      )}
    </div>
  );
}
