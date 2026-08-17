/**
 * frontend/src/services/api.js
 * API client service for Intelli Docs RAG backend.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Check backend health and get indexed document count
 * @returns {Promise<{status: string, vector_store_documents: number}>}
 */
export async function checkBackendHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Failed to connect to backend health check:', error);
    throw error;
  }
}

/**
 * Send user query to RAG pipeline via backend API
 * @param {string} question - The query string from user
 * @param {string} sessionId - The ID of the current chat session
 * @returns {Promise<{question: string, answer: string, sources: Array<{content: string, metadata: object, similarity: number}>}>}
 */
export async function askQuestion(question, sessionId = "default") {
  const trimmed = question.trim();
  if (!trimmed) {
    throw new Error('Question cannot be empty.');
  }

  try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify({ question: trimmed, session_id: sessionId }),
    });

    const data = await response.json();

    if (!response.ok || data.error) {
      throw new Error(data.error || `Server responded with status ${response.status}`);
    }

    return data;
  } catch (error) {
    console.error('Error calling /chat endpoint:', error);
    throw error;
  }
}

/**
 * Fetch all chat sessions
 * @returns {Promise<Array<{id: string, title: string, timestamp: number}>>}
 */
export async function getSessions() {
  try {
    const response = await fetch(`${API_BASE_URL}/sessions`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    if (!response.ok) throw new Error('Failed to fetch sessions');
    return await response.json();
  } catch (error) {
    console.error('Error fetching sessions:', error);
    return [];
  }
}

/**
 * Fetch chat history for a specific session
 * @param {string} sessionId 
 * @returns {Promise<Array<{role: string, content: string}>>}
 */
export async function getChatHistory(sessionId) {
  try {
    const response = await fetch(`${API_BASE_URL}/chat/${sessionId}/history`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    if (!response.ok) throw new Error('Failed to fetch chat history');
    const data = await response.json();
    return data.history || [];
  } catch (error) {
    console.error('Error fetching chat history:', error);
    return [];
  }
}
