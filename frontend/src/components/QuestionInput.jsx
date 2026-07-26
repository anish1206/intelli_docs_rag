import React, { useRef, useEffect } from 'react';

export default function QuestionInput({ question, setQuestion, onSubmit, isLoading, disabled }) {
  const textareaRef = useRef(null);

  // Auto-resize textarea height as user types
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
    }
  }, [question]);

  const handleKeyDown = (e) => {
    // Enter without Shift triggers submit
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isLoading && !disabled && question.trim()) {
        onSubmit(e);
      }
    }
  };

  return (
    <form className="question-input-form" onSubmit={onSubmit}>
      <div className={`input-pill-container ${isLoading ? 'loading' : ''}`}>
        <textarea
          ref={textareaRef}
          className="interactive-textarea"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything about your personal documents & notes..."
          rows={1}
          disabled={isLoading || disabled}
          aria-label="Ask a question about your documents"
        />

        <div className="input-controls">
          <span className="input-hint">Enter ↵</span>
          
          <button
            type="submit"
            className="submit-pill-button"
            disabled={!question.trim() || isLoading || disabled}
            title={isLoading ? 'RAG pipeline is running...' : 'Send Query'}
          >
            {isLoading ? (
              <div className="button-spinner"></div>
            ) : (
              <>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="22" y1="2" x2="11" y2="13"></line>
                  <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                </svg>
              </>
            )}
          </button>
        </div>
      </div>
    </form>
  );
}
