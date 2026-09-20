# `plan.md` — Implementation Plan: Agentic Intelli Docs RAG

## Executive Summary
This document outlines the step-by-step engineering plan to upgrade **Intelli Docs RAG** from a linear pipeline (`Query → Retrieve → Generate`) to an **Autonomous Agentic RAG** system using **LangGraph**, **ChromaDB**, **Redis**, and **Qwen 2.5:7b** (via Ollama on Colab).

### Target Capabilities
1. **Multi-Turn Contextual Query Rewriter** (Resolves pronouns and conversational context from Redis history).
2. **Corrective RAG (CRAG)** (Grades retrieved chunks; reformulates search query if chunks are irrelevant).
3. **Native Tool Calling** (`search_notes`, `calculate` for tables/formulas, `list_available_notes`).
4. **In-Loop Hallucination Guard** (Verifies that generated answers are grounded in context before sending to user).

---

## Architectural Workflow (LangGraph State Graph)

```
                       [User Query + Redis History]
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Query Rewriter    │  (Contextualizes follow-ups)
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Agent Router     │  (Decides Tool / Path)
                         └────┬─────────────┬──┘
             Direct / Math    │             │  Requires Doc Search
       ┌──────────────────────┘             ▼
       │                          ┌─────────────────────┐
       │                          │    search_notes     │  (ChromaDB Retrieval)
       │                          └──────────┬──────────┘
       │                                     │
       │                                     ▼
       │                          ┌─────────────────────┐
       │                          │   Retrieval Grader  │
       │                          └─────┬─────────┬─────┘
       │                 Relevant Docs  │         │  Irrelevant (Loop ≤ 1)
       │                                │         └──────────────┐
       │                                ▼                        ▼
       │                     ┌─────────────────────┐   ┌───────────────────┐
       │                     │  Synthesizer Node   │   │  Query Rewriter   │
       │                     └──────────┬──────────┘   │   (Reformulate)   │
       │                                │              └─────────┬─────────┘
       │                                ▼                        │
       │                     ┌─────────────────────┐             │
       │                     │ Hallucination Guard │             │
       │                     └─────┬─────────┬─────┘             │
       │             Pass / Grounded│        │ Ungrounded (≤ 1)  │
       │                            │        └─────────┐         │
       │                            ▼                  ▼         │
       │                       [Final Answer]    [Regenerate]    │
       │                            ▲                  │         │
       └────────────────────────────┴──────────────────┴─────────┘
```

---

## Latency Budget & Guardrails
Because LLM calls run over a Cloudflare tunnel to Google Colab, minimizing sequential round-trips is critical:
* **Max Iteration Limit:** Strict `max_retries = 1` for both CRAG query rewrites and hallucination retries.
* **Deterministic Fast Paths:** Greetings and exact SHA-256 Redis answer cache hits remain in FastAPI *before* entering the graph.
* **Compact Prompts:** Evaluator prompts use strict binary flags (`"yes"` / `"no"` or single JSON tokens) to minimize generation tokens.

---

## Phase 1: Dependencies & Environment Setup

### Step 1.1: Install Open-Source Packages
Add the following free libraries to `requirements.txt`:
```txt
langgraph>=0.2.0
langchain-core>=0.3.0
simpleeval>=0.9.13
```
* **`langgraph`**: Graph-based state machine framework.
* **`simpleeval`**: Safe, sandboxed mathematical expression evaluator (zero external runtime risk for calculator tools).

### Step 1.2: Update `.env.example` & `rag/config.py`
Add agent-specific parameters to configuration:
```python
# rag/config.py additions
AGENT_MAX_RETRIES = int(os.getenv("AGENT_MAX_RETRIES", 1))
CRAG_RELEVANCE_THRESHOLD = float(os.getenv("CRAG_RELEVANCE_THRESHOLD", 0.5))
HISTORY_WINDOW_TURNS = int(os.getenv("HISTORY_WINDOW_TURNS", 3))
```

---

## Phase 2: Redis Memory Enhancements (`backend/chat_memory.py`)

### Step 2.1: Add Context Window Helper
Add a helper to retrieve only the last $N$ conversational turns for the query rewriter without loading large histories into prompt context:

```python
def get_recent_history_formatted(self, session_id: str, max_turns: int = 3) -> str:
    """Returns the last `max_turns` messages formatted as User/Assistant dialogues."""
    history = self.get_history(session_id)
    if not history:
        return ""
    recent = history[- (max_turns * 2):]  # pairs of user & assistant
    lines = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent]
    return "\n".join(lines)
```

---

## Phase 3: Agent State & Node Implementation (`rag/agent/`)

Create a dedicated directory `rag/agent/` to keep agent logic modular:
```
rag/
└── agent/
    ├── __init__.py
    ├── state.py         # Agent state definitions
    ├── tools.py         # Tool registry (Search, Calc, Doc List)
    ├── prompts.py       # Rewriting, Grading, and Guard prompts
    └── nodes.py         # Graph execution nodes
```

### Step 3.1: Define Graph State (`rag/agent/state.py`)
```python
from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
import operator

class AgentState(TypedDict):
    question: str                      # Original user query
    session_id: str                    # Chat session UUID
    history: str                       # Formatted short-term history
    standalone_query: str              # Context-resolved query
    tool_choice: Optional[str]         # 'search', 'calculate', 'list_docs', 'direct'
    tool_output: Optional[Any]         # Output from tool execution
    documents: List[Dict[str, Any]]    # Retrieved chunks from ChromaDB
    retrieval_retry_count: int         # Track CRAG loops (cap at 1)
    hallucination_retry_count: int     # Track Guard loops (cap at 1)
    is_relevant: bool                  # CRAG grade
    is_grounded: bool                  # Guard grade
    final_answer: str                  # Generated response
    sources: List[Dict[str, Any]]      # Citations
```

---

### Step 3.2: Tool Registry (`rag/agent/tools.py`)
Wrap system tools using `langchain_core.tools`:

1. **`search_notes(query: str, filter_metadata: Optional[dict] = None)`**:
   * Invokes your existing `RAGRetriever.retrieve()`.
   * Returns top matching chunks with scores and page numbers.
2. **`calculate(expression: str)`**:
   * Evaluates expressions like `"15 * 0.4 + 20 * 0.6"` using `simpleeval`.
   * Safe for formulas, GPA calculations, grading schemes, and numerical course tables.
3. **`list_available_notes()`**:
   * Directly queries ChromaDB metadata or `data/` folder for document titles and page ranges.
   * Answers queries like *"What syllabus notes do I have for Semester 4?"* without semantic search.

---

### Step 3.3: Node Definitions (`rag/agent/nodes.py`)

#### 1. Contextual Query Rewriter Node (`rewrite_query_node`)
* **Trigger:** Runs if `history` is non-empty.
* **Prompt:** Given chat history and the current user input, reformulate any pronoun or missing subject into a standalone sentence.
* *Example:*
  * History: `User: What is PageRank? Assistant: An algorithm to rank web pages...`
  * Input: `How does it handle dead ends?`
  * Output: `How does the PageRank algorithm handle dead ends and spider traps?`

#### 2. Agent Router Node (`router_node`)
* Uses Qwen's tool-calling format or a constrained JSON classification prompt to choose one branch:
  * `call_search`
  * `call_calculator`
  * `call_list_docs`
  * `direct_chat`

#### 3. CRAG Retrieval Grader Node (`grade_documents_node`)
* Inspects retrieved chunks against `standalone_query`.
* Fast prompt: Evaluates if at least one chunk contains sufficient signal to answer the question (`{"relevant": true/false}`).
* If `relevant is False` and `retrieval_retry_count < 1`:
  * Routes to a `reformulate_query_node` which extracts key entities and synonyms.
  * Re-queries `search_notes` with broader retrieval thresholds.
* Otherwise, proceeds directly to generation.

#### 4. Synthesis / Generation Node (`generate_answer_node`)
* Uses your existing `RAG_PROMPT_TEMPLATE` with context citation headers.
* Injects formatted context or calculator outputs.
* Generates drafted answer.

#### 5. In-Loop Hallucination Guard Node (`hallucination_guard_node`)
* Adapted from your proven `eval/eval_rag_updated.py` Groundedness metric.
* Fast verification prompt:
  > *"Analyze the context and drafted answer. Are all factual assertions in the drafted answer directly derived from the context? Answer strictly JSON: {\"grounded\": true|false}"*
* If `grounded is False` and `hallucination_retry_count < 1`:
  * Updates state `hallucination_retry_count += 1`.
  * Calls `generate_answer_node` with a strict temperature `0.0` and a constrained "state only what is explicitly in context" directive.
* If `grounded is True` (or retry cap reached), commits to `final_answer`.

---

## Phase 4: LangGraph Workflow Compilation (`rag/agent/graph.py`)

Assemble the nodes into a compiled `StateGraph`:

```python
from langgraph.graph import StateGraph, END
from rag.agent.state import AgentState
from rag.agent import nodes

def build_agentic_rag_graph(retriever, llm):
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("rewrite_query", nodes.create_rewriter_node(llm))
    workflow.add_node("router", nodes.create_router_node(llm))
    workflow.add_node("search_notes", nodes.create_search_node(retriever))
    workflow.add_node("calculate", nodes.calculate_node)
    workflow.add_node("list_docs", nodes.list_docs_node)
    workflow.add_node("grade_retrieval", nodes.create_grade_node(llm))
    workflow.add_node("reformulate_query", nodes.create_reformulate_node(llm))
    workflow.add_node("generate_answer", nodes.create_generator_node(llm))
    workflow.add_node("hallucination_guard", nodes.create_guard_node(llm))

    # 2. Add Edges & Conditional Routes
    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "router")

    workflow.add_conditional_edges(
        "router",
        nodes.route_decision,
        {
            "search": "search_notes",
            "calculate": "calculate",
            "list_docs": "list_docs",
            "direct": "generate_answer",
        }
    )

    workflow.add_edge("search_notes", "grade_retrieval")

    workflow.add_conditional_edges(
        "grade_retrieval",
        nodes.crag_decision,
        {
            "proceed": "generate_answer",
            "retry": "reformulate_query",
        }
    )
    workflow.add_edge("reformulate_query", "search_notes")

    workflow.add_edge("calculate", "generate_answer")
    workflow.add_edge("list_docs", "generate_answer")

    workflow.add_edge("generate_answer", "hallucination_guard")

    workflow.add_conditional_edges(
        "hallucination_guard",
        nodes.guard_decision,
        {
            "pass": END,
            "regenerate": "generate_answer",
        }
    )

    return workflow.compile()
```

---

## Phase 5: FastAPI Backend Integration (`backend/server.py`)

### Step 5.1: Initialize Compiled Graph on Startup
Update startup lifecycle in `server.py`:
```python
# Initialize LangGraph alongside retriever and chat memory
agent_app = build_agentic_rag_graph(retriever=retriever, llm=llm)
```

### Step 5.2: Refactor `POST /chat` Route
Maintain performance by preserving pre-graph quick exits:
1. **Validate question** (empty string check).
2. **Greeting Interceptor** (instant return, zero LLM calls).
3. **Redis Cache Check** (instant return on duplicate queries).
4. **Invoke Agent Graph**:
   ```python
   # Load last 3 turns from Redis
   recent_history = memory.get_recent_history_formatted(session_id, max_turns=3)

   initial_state = {
       "question": question,
       "session_id": session_id,
       "history": recent_history,
       "retrieval_retry_count": 0,
       "hallucination_retry_count": 0,
       "sources": [],
   }

   result_state = await agent_app.ainvoke(initial_state)

   answer = result_state["final_answer"]
   sources = result_state.get("sources", [])
   ```
5. **Persist Messages & Update Cache** in Redis as usual.
6. **Return Payload** matching existing contract:
   ```json
   {
     "question": "How does it handle dead ends?",
     "answer": "...",
     "sources": [...],
     "from_cache": false,
     "agent_metadata": {
       "rewritten_query": "How does the PageRank algorithm handle dead ends?",
       "tool_used": "search_notes",
       "retrieval_retried": false,
       "guard_passed": true
     }
   }
   ```

---

## Phase 6: Frontend Backward Compatibility (`frontend/`)

* **No Breaking Changes:** The existing React frontend expects `{ question, answer, sources, from_cache }`, which continues to work seamlessly.
* **Optional Visual Enhancement (`ChatMessage.jsx`):**
  * If `message.agent_metadata?.rewritten_query` exists, display a subtle subtext tag:
    * *🔍 Searched for: "How does the PageRank algorithm handle dead ends?"*
  * If `message.agent_metadata?.tool_used === 'calculate'`, display a small calculator badge (`🧮 Math evaluated`).

---

## Phase 7: Verification & Testing Playbook

Create `eval/test_agent.py` to test each agentic capability systematically:

| Test Case | Prompt Sequence | Expected Agent Behavior |
|---|---|---|
| **Multi-Turn Context** | Turn 1: *"What is an AVL Tree?"*<br>Turn 2: *"What is its worst-case search time complexity?"* | Rewriter passes *"What is the worst-case search time complexity of an AVL tree?"* to ChromaDB. |
| **Tool Calling (Calc)** | *"If Assignment 1 is 20% (scored 85) and Exam is 80% (scored 90), what is my weighted grade?"* | Agent routes to `calculate` tool; evaluates `(85 * 0.2) + (90 * 0.8)` directly without hallucinating math. |
| **Tool Calling (Metadata)** | *"What lecture notes or PDFs are loaded in the system?"* | Agent calls `list_available_notes` and lists indexed files without doing vector semantic search. |
| **CRAG Self-Correction** | Query using unfamiliar jargon or slight typo: *"Explain convolutional kernal pooling layers"* | Grader marks initial poor chunks, rewrites terms to standard keywords, and successfully retrieves correct CNN chunks. |
| **Hallucination Guard** | Asking a trick question not present in notes (e.g., *"What is the professor's personal phone number?"*) | Guard blocks ungrounded answer drafts and outputs standard safe refusal without hallucination. |

---

## Rollout Sequence (Completed)

- [x] **Step 1:** Add `langgraph` and `simpleeval` to `requirements.txt` and install.
- [x] **Step 2:** Add `get_recent_history_formatted()` helper to `backend/chat_memory.py`.
- [x] **Step 3:** Implement `rag/agent/state.py` and `rag/agent/tools.py`.
- [x] **Step 4:** Implement `rag/agent/prompts.py` and `rag/agent/nodes.py`.
- [x] **Step 5:** Assemble graph in `rag/agent/graph.py` with unit tests for each conditional branch.
- [x] **Step 6:** Hook graph invocation into `backend/server.py` `POST /chat`.
- [x] **Step 7:** Run `eval/test_agent.py` live verification CLI script and `tests/test_e2e_agent.py` pytest suite.
- [x] **Step 8:** Audit and verify frontend backward compatibility (UI kept visually identical per strict UI constraints).