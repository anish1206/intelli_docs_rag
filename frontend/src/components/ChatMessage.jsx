import React, { useState } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import SourceCard from './SourceCard';

// Helper to safely render KaTeX math HTML
function renderKaTeX(tex, displayMode = false) {
  try {
    // Clean up excessive whitespace in LaTeX tokens (Docling artifact fix)
    const cleaned = tex
      .replace(/\\\s+/g, '\\')
      .replace(/\s*_\s*/g, '_')
      .replace(/\s*\^\s*/g, '^');
    
    return katex.renderToString(cleaned, {
      displayMode,
      throwOnError: false,
    });
  } catch (err) {
    console.warn('KaTeX render error:', err);
    return tex;
  }
}

// Component to parse text with inline math ($...$ or \(...\)), bold, italic, code
function InlineFormattedContent({ text }) {
  if (!text) return null;

  // Split by inline math ($...$ or \(...\)) or inline formatting (**...**, *...*, `...`)
  const parts = text.split(/(\$\$.*?\$\$|\$\b.*?\b\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]|\*\*.*?\*\*|\*.*?\*|`.*?`)/g);

  return parts.map((part, idx) => {
    if (!part) return null;

    // Display Block Math ($$ ... $$ or \[ ... \])
    if ((part.startsWith('$$') && part.endsWith('$$')) || (part.startsWith('\\[') && part.endsWith('\\]'))) {
      const math = part.startsWith('$$') ? part.slice(2, -2) : part.slice(2, -2);
      return (
        <span
          key={idx}
          className="katex-display-wrapper"
          dangerouslySetInnerHTML={{ __html: renderKaTeX(math, true) }}
        />
      );
    }

    // Inline Math ($ ... $ or \( ... \))
    if ((part.startsWith('$') && part.endsWith('$')) || (part.startsWith('\\(') && part.endsWith('\\)'))) {
      const math = part.startsWith('$') ? part.slice(1, -1) : part.slice(2, -2);
      return (
        <span
          key={idx}
          className="katex-inline-wrapper"
          dangerouslySetInnerHTML={{ __html: renderKaTeX(math, false) }}
        />
      );
    }

    // Bold text
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={idx}>{part.slice(2, -2)}</strong>;
    }

    // Italic text
    if (part.startsWith('*') && part.endsWith('*')) {
      return <em key={idx}>{part.slice(1, -1)}</em>;
    }

    // Inline Code
    if (part.startsWith('`') && part.endsWith('`')) {
      return <code key={idx} className="inline-code">{part.slice(1, -1)}</code>;
    }

    return part;
  });
}

// Paragraph & Block level formatter
function FormattedText({ content }) {
  if (!content) return null;

  // Split by double newline for paragraphs or math blocks
  const blocks = content.split(/(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\n\n+)/g);

  return (
    <div className="formatted-text-content">
      {blocks.map((block, bIdx) => {
        if (!block || block === '\n\n') return null;

        // Block display math ($$ ... $$ or \[ ... \])
        if ((block.startsWith('$$') && block.endsWith('$$')) || (block.startsWith('\\[') && block.endsWith('\\]'))) {
          const math = block.startsWith('$$') ? block.slice(2, -2) : block.slice(2, -2);
          return (
            <div key={bIdx} className="math-block-container">
              <span
                dangerouslySetInnerHTML={{ __html: renderKaTeX(math, true) }}
              />
            </div>
          );
        }

        // Code block (``` ... ```)
        if (block.startsWith('```')) {
          const cleanCode = block.replace(/^```[a-z]*\n?/, '').replace(/\n?```$/, '');
          return (
            <div key={bIdx} className="code-block-wrapper">
              <pre className="code-block"><code>{cleanCode}</code></pre>
            </div>
          );
        }

        const lines = block.split('\n');

        // List items
        const isList = lines.every((line) => /^\s*([-*]|\d+\.)\s+/.test(line));
        if (isList && lines.length > 0) {
          return (
            <ul key={bIdx} className="formatted-list">
              {lines.map((line, lIdx) => {
                const cleanLine = line.replace(/^\s*([-*]|\d+\.)\s+/, '');
                return (
                  <li key={lIdx}>
                    <InlineFormattedContent text={cleanLine} />
                  </li>
                );
              })}
            </ul>
          );
        }

        // Standard paragraph
        return (
          <p key={bIdx} className="formatted-paragraph">
            {lines.map((line, lIdx) => (
              <React.Fragment key={lIdx}>
                <InlineFormattedContent text={line} />
                {lIdx < lines.length - 1 && <br />}
              </React.Fragment>
            ))}
          </p>
        );
      })}
    </div>
  );
}

export default function ChatMessage({ message }) {
  const isUser = message.role === 'user';
  const [copied, setCopied] = useState(false);
  const [showSources, setShowSources] = useState(true);

  const handleCopy = () => {
    if (message.content) {
      navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (isUser) {
    return (
      <div className="message-row user-row">
        <div className="message-bubble user-bubble">
          <div className="user-avatar">YOU</div>
          <div className="message-body">
            <p>{message.content}</p>
          </div>
        </div>
      </div>
    );
  }

  const hasSources = message.sources && Array.isArray(message.sources) && message.sources.length > 0;

  return (
    <div className="message-row assistant-row">
      <div className="message-bubble assistant-bubble">
        <div className="assistant-header">
          <div className="assistant-avatar">
            LLM Response
          </div>

          <div className="message-actions">
            {message.content && (
              <button
                type="button"
                className="action-icon-btn"
                onClick={handleCopy}
                title="Copy answer"
              >
                {copied ? '✓ Copied' : 'Copy'}
              </button>
            )}
          </div>
        </div>

        <div className="message-body">
          <FormattedText content={message.content} />
        </div>

        {hasSources && (
          <div className="message-sources-wrapper">
            <div
              className="sources-toggle-header"
              onClick={() => setShowSources(!showSources)}
            >
              <div className="sources-title">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                  <polyline points="14 2 14 8 20 8"></polyline>
                </svg>
                <span>RETRIEVED EVIDENCE ({message.sources.length} SOURCES)</span>
              </div>
              <span className="sources-toggle-indicator">
                {showSources ? 'Hide' : 'Show'}
              </span>
            </div>

            {showSources && (
              <div className="sources-grid">
                {message.sources.map((src, index) => (
                  <SourceCard key={index} source={src} index={index} />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
