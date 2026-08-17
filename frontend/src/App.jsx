import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import ChatMessage from './components/ChatMessage';
import QuestionInput from './components/QuestionInput';
import { checkBackendHealth, askQuestion, getSessions, getChatHistory } from './services/api';
import './App.css';

export default function App() {
  const [healthStatus, setHealthStatus] = useState('checking'); // 'checking' | 'healthy' | 'offline'
  const [docCount, setDocCount] = useState(0);
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  
  // Session states
  const [sessionId, setSessionId] = useState(Date.now().toString());
  const [sessions, setSessions] = useState([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  
  const chatBottomRef = useRef(null);

  // Check backend health and load sessions on mount
  const initializeApp = async () => {
    setHealthStatus('checking');
    try {
      const data = await checkBackendHealth();
      setHealthStatus('healthy');
      setDocCount(data.vector_store_documents ?? 0);
      
      // Load sessions
      const loadedSessions = await getSessions();
      setSessions(loadedSessions);
      
      // If there are previous sessions, load the latest one
      if (loadedSessions.length > 0) {
        const latestSessionId = loadedSessions[0].id;
        setSessionId(latestSessionId);
        loadSessionHistory(latestSessionId);
      }
    } catch (err) {
      console.warn('Backend server is offline or unreachable:', err);
      setHealthStatus('offline');
    }
  };

  useEffect(() => {
    initializeApp();
  }, []);

  const loadSessionHistory = async (sid) => {
    try {
      const history = await getChatHistory(sid);
      // Map history format to messages state
      const mappedMessages = history.map((msg, idx) => ({
        id: `hist-${idx}`,
        role: msg.role,
        content: msg.content,
        sources: [], // We don't save sources in history currently to save space, but it's ok
        timestamp: new Date()
      }));
      setMessages(mappedMessages);
    } catch (error) {
      console.error('Error loading history:', error);
    }
  };

  const handleSelectSession = (sid) => {
    setSessionId(sid);
    loadSessionHistory(sid);
    if (window.innerWidth < 768) {
        setIsSidebarOpen(false); // Auto-close on mobile
    }
  };

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
      const responseData = await askQuestion(queryText, sessionId);

      const assistantMsg = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: responseData.answer || 'No response text returned.',
        sources: responseData.sources || [],
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMsg]);
      setHealthStatus('healthy');
      
      // Refresh sessions list silently
      const refreshedSessions = await getSessions();
      setSessions(refreshedSessions);
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

  // Clear conversation history -> New Chat
  const handleNewChat = () => {
    setMessages([]);
    setErrorMsg(null);
    setSessionId(Date.now().toString());
    if (window.innerWidth < 768) {
        setIsSidebarOpen(false);
    }
  };

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  const isLanding = messages.length === 0;

  return (
    <div className={`app-layout ${isLanding ? 'landing-mode' : 'chat-mode'} ${isSidebarOpen ? 'sidebar-open' : ''}`}>
      {/* Cosmic background radial glow */}
      {isLanding && <div className="hero-glow-effect" />}

      {/* Floating Toggle Button (visible when sidebar is closed) */}
      {!isSidebarOpen && (
        <button 
          className="floating-sidebar-toggle" 
          onClick={toggleSidebar} 
          title="Open Sidebar"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="m9 18 6-6-6-6"/>
          </svg>
        </button>
      )}

      {/* Sidebar */}
      <aside className={`sidebar ${isSidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
            <h3>Recent Chats</h3>
            <button className="icon-button" onClick={toggleSidebar}>✕</button>
        </div>
        <button className="new-chat-btn" onClick={handleNewChat}>
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-message-square-plus">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
              <path d="M12 7v6"/>
              <path d="M9 10h6"/>
            </svg>
            New Chat
        </button>
        <div className="sessions-list">
            {sessions.length === 0 && <p className="no-sessions">No previous chats.</p>}
            {sessions.map(s => (
                <div 
                    key={s.id} 
                    className={`session-item ${s.id === sessionId ? 'active' : ''}`}
                    onClick={() => handleSelectSession(s.id)}
                >
                    <div className="session-title">{s.title}</div>
                </div>
            ))}
        </div>
      </aside>

      <div className="main-wrapper">
          {/* Header */}
          <Header
            status={healthStatus}
            docCount={docCount}
            onRetryHealth={initializeApp}
            onNewChat={handleNewChat}
            onToggleSidebar={toggleSidebar}
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
                        <button type="button" className="ghost-button" onClick={initializeApp}>
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
                        onClick={initializeApp}
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
    </div>
  );
}
