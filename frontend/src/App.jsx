import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import ChatMessage from './components/ChatMessage';
import QuestionInput from './components/QuestionInput';
import { checkBackendHealth, askQuestion } from './services/api';
import './App.css';

export default function App() {
  const [healthStatus, setHealthStatus] = useState('checking'); // 'checking' | 'healthy' | 'offline'
  const [docCount, setDocCount] = useState(0);
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  
  const chatBottomRef = useRef(null);

  // Check backend health on mount
  const runHealthCheck = async () => {
    setHealthStatus('checking');
    try {
      const data = await checkBackendHealth();
      setHealthStatus('healthy');
      setDocCount(data.vector_store_documents ?? 0);
    } catch (err) {
      console.warn('Backend server is offline or unreachable:', err);
      setHealthStatus('offline');
    }
  };

  useEffect(() => {
    runHealthCheck();
  }, []);

  // Auto-scroll chat to bottom
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading, errorMsg]);

  // Handle submitting query
  const handleSubmit = async (e) => {
    if (e) e.preventDefault();

    const queryText = question.trim();
    if (!queryText || isLoading) return;

    // Reset error state
    setErrorMsg(null);

    // Create user message object
    const userMsg = {
      id: Date.now().toString(),
      role: 'user',
      content: queryText,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setQuestion('');
    setIsLoading(true);

    try {
      const responseData = await askQuestion(queryText);

      const assistantMsg = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: responseData.answer || 'No response text returned.',
        sources: responseData.sources || [],
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMsg]);
      setHealthStatus('healthy');
    } catch (err) {
      console.error('Query execution failed:', err);
      const userFriendlyErr = healthStatus === 'offline' 
        ? 'Unable to connect to the knowledge assistant backend. Please ensure uvicorn backend.server:app is running.'
        : (err.message || 'An error occurred while retrieving answer from the RAG pipeline.');
      
      setErrorMsg(userFriendlyErr);
    } finally {
      setIsLoading(false);
    }
  };

  // Clear conversation history
  const handleClearHistory = () => {
    setMessages([]);
    setErrorMsg(null);
  };

  const isLanding = messages.length === 0;

  return (
    <div className={`app-layout ${isLanding ? 'landing-mode' : 'chat-mode'}`}>
      {/* Cosmic background radial glow */}
      {isLanding && <div className="hero-glow-effect" />}

      {/* Header */}
      <Header
        status={healthStatus}
        docCount={docCount}
        onRetryHealth={runHealthCheck}
        onClearHistory={handleClearHistory}
        hasHistory={messages.length > 0}
      />

      {/* Main Content Area */}
      <main className="main-content">
        <div className="content-container">
          
          {/* Centered Landing Hero View */}
          {isLanding ? (
            <div className="hero-landing-wrapper">
              <div className="hero-section">
                <div className="hero-badge">INTELLECTUAL VECTOR SEARCH & RAG</div>
                <h1 className="hero-headline">Document Intelligence System</h1>
                
                {healthStatus === 'offline' && (
                  <div className="system-warning-callout">
                    <div className="warning-icon">⚠️</div>
                    <div className="warning-text">
                      <strong>Backend Offline:</strong> Start the FastAPI server using{' '}
                      <code>uvicorn backend.server:app --reload</code> on <code>http://localhost:8000</code>.
                    </div>
                    <button type="button" className="ghost-button" onClick={runHealthCheck}>
                      Retry Connection
                    </button>
                  </div>
                )}

                <div className="hero-input-container">
                  <QuestionInput
                    question={question}
                    setQuestion={setQuestion}
                    onSubmit={handleSubmit}
                    isLoading={isLoading}
                    disabled={healthStatus === 'offline'}
                  />
                </div>
              </div>
            </div>
          ) : (
            /* Conversation Stream View */
            <div className="chat-stream">
              {messages.map((msg) => (
                <ChatMessage key={msg.id} message={msg} />
              ))}

              {/* Loading Indicator */}
              {isLoading && (
                <div className="message-row assistant-row loading-row">
                  <div className="message-bubble assistant-bubble loading-bubble">
                    <div className="assistant-avatar">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="12" cy="12" r="10" />
                      </svg>
                      <span>INTELLI RAG</span>
                    </div>
                    <div className="loading-animation">
                      <div className="typing-dot"></div>
                      <div className="typing-dot"></div>
                      <div className="typing-dot"></div>
                      <span className="loading-status-text">Searching documents & generating answer...</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Error Callout */}
              {errorMsg && (
                <div className="error-callout-card">
                  <div className="error-header">
                    <span className="error-icon">✕</span>
                    <span className="error-title">Query Execution Failed</span>
                  </div>
                  <p className="error-description">{errorMsg}</p>
                  <button 
                    type="button" 
                    className="ghost-button error-retry-btn"
                    onClick={runHealthCheck}
                  >
                    Check Backend Status
                  </button>
                </div>
              )}

              <div ref={chatBottomRef} />
            </div>
          )}
        </div>
      </main>

      {/* Sticky Bottom Input Bar (Chat mode only) */}
      {!isLanding && (
        <footer className="app-footer-input-bar">
          <div className="input-bar-container">
            <QuestionInput
              question={question}
              setQuestion={setQuestion}
              onSubmit={handleSubmit}
              isLoading={isLoading}
              disabled={healthStatus === 'offline'}
            />
            <div className="footer-meta-info">
              <span>Intelli Docs RAG • Fast Vector Search • Docling Parser</span>
              <span>ChromaDB Persistent Store</span>
            </div>
          </div>
        </footer>
      )}
    </div>
  );
}
