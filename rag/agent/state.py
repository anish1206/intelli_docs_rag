from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

class AgentState(TypedDict):
    question: str                      # Raw question from user
    session_id: str                    # Chat session UUID
    history: str                       # Short-term formatted chat history
    standalone_query: str              # Context-resolved query
    tool_choice: Optional[str]         # 'search', 'calculate', 'list_docs', 'direct'
    tool_output: Optional[Any]         # Output from tool execution
    documents: List[Dict[str, Any]]    # Retrieved chunks from ChromaDB
    retrieval_retry_count: int         # CRAG retry counter
    hallucination_retry_count: int     # Guard retry counter
    is_relevant: bool                  # CRAG grade status
    is_grounded: bool                  # Guard check status
    final_answer: str                  # Generated or tool answer
    sources: List[Dict[str, Any]]      # Source citation metadata
