from typing import Callable, Any
from .state import AgentState
from .tools import calculate, search_notes, list_available_notes
from .prompts import (
    REWRITE_PROMPT,
    ROUTER_PROMPT,
    CRAG_GRADER_PROMPT,
    REFORMULATE_PROMPT,
    GENERATE_PROMPT,
    GUARD_PROMPT,
    clean_json_response
)
import os

AGENT_MAX_RETRIES = int(os.getenv("AGENT_MAX_RETRIES", "1"))

def create_rewriter_node(llm: Any) -> Callable[[AgentState], dict]:
    def rewrite_query_node(state: AgentState) -> dict:
        if not state.get("history"):
            return {"standalone_query": state["question"]}
        
        prompt = REWRITE_PROMPT.format(
            history=state["history"], 
            question=state["question"]
        )
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        return {"standalone_query": content.strip()}
    return rewrite_query_node

def create_router_node(llm: Any) -> Callable[[AgentState], dict]:
    def router_node(state: AgentState) -> dict:
        prompt = ROUTER_PROMPT.format(question=state["standalone_query"])
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        parsed = clean_json_response(content)
        tool = parsed.get("tool", "direct")
        return {"tool_choice": tool}
    return router_node

def create_search_node(retriever: Any) -> Callable[[AgentState], dict]:
    def search_node(state: AgentState) -> dict:
        query = state["standalone_query"]
        docs = search_notes(retriever, query)
        
        # Format sources
        sources = []
        for doc in docs:
            # Depending on how retriever returns documents, extract metadata
            meta = doc.metadata if hasattr(doc, 'metadata') else doc.get('metadata', {})
            if meta:
                sources.append(meta)
                
        return {"documents": docs, "sources": sources}
    return search_node

def calculate_node(state: AgentState) -> dict:
    query = state["standalone_query"]
    # We could optionally ask LLM to extract the exact math expression,
    # but simpleeval evaluates what we pass. Let's assume the router
    # or the user passes a reasonably clean math expression, or the query itself is math.
    result = calculate(query)
    return {"tool_output": result, "final_answer": result}

def create_list_docs_node(retriever: Any) -> Callable[[AgentState], dict]:
    def list_docs_node(state: AgentState) -> dict:
        docs = list_available_notes(retriever)
        if not docs:
            answer = "There are currently no documents indexed."
        else:
            answer = "Here are the available documents:\n" + "\n".join(f"- {d}" for d in docs)
        return {"tool_output": docs, "final_answer": answer}
    return list_docs_node

def create_grade_node(llm: Any) -> Callable[[AgentState], dict]:
    def grade_documents_node(state: AgentState) -> dict:
        docs = state.get("documents", [])
        if not docs:
            return {"is_relevant": False}
        
        # Combine doc texts for grading
        doc_texts = []
        for doc in docs:
            text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            doc_texts.append(text)
        combined_docs = "\n\n".join(doc_texts)
        
        prompt = CRAG_GRADER_PROMPT.format(
            document=combined_docs, 
            question=state["standalone_query"]
        )
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        parsed = clean_json_response(content)
        
        return {"is_relevant": parsed.get("relevant", False)}
    return grade_documents_node

def create_reformulate_node(llm: Any) -> Callable[[AgentState], dict]:
    def reformulate_query_node(state: AgentState) -> dict:
        prompt = REFORMULATE_PROMPT.format(question=state["standalone_query"])
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        retries = state.get("retrieval_retry_count", 0)
        return {
            "standalone_query": content.strip(),
            "retrieval_retry_count": retries + 1
        }
    return reformulate_query_node

def create_generator_node(llm: Any, retriever: Any) -> Callable[[AgentState], dict]:
    def generate_answer_node(state: AgentState) -> dict:
        # Format context
        docs = state.get("documents", [])
        
        if hasattr(retriever, 'format_context_for_llm'):
            context = retriever.format_context_for_llm(docs)
        else:
            doc_texts = []
            for doc in docs:
                text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                doc_texts.append(text)
            context = "\n\n".join(doc_texts)
            
        tool_out = state.get("tool_output")
        if tool_out:
            context += f"\n\nTool Output:\n{tool_out}"
            
        prompt = GENERATE_PROMPT.format(
            context=context,
            question=state["standalone_query"]
        )
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        return {"final_answer": content.strip()}
    return generate_answer_node

def create_guard_node(llm: Any) -> Callable[[AgentState], dict]:
    def hallucination_guard_node(state: AgentState) -> dict:
        docs = state.get("documents", [])
        
        doc_texts = []
        for doc in docs:
            text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            doc_texts.append(text)
        context = "\n\n".join(doc_texts)
        
        prompt = GUARD_PROMPT.format(
            context=context,
            answer=state["final_answer"]
        )
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        parsed = clean_json_response(content)
        
        is_grounded = parsed.get("grounded", False)
        retries = state.get("hallucination_retry_count", 0)
        
        if not is_grounded and retries < AGENT_MAX_RETRIES:
            return {
                "is_grounded": False,
                "hallucination_retry_count": retries + 1
            }
        
        return {"is_grounded": True}
    return hallucination_guard_node

# Conditional routing helpers
def route_decision(state: AgentState) -> str:
    choice = state.get("tool_choice")
    if choice in ["search", "calculate", "list_docs"]:
        return choice
    return "direct"

def crag_decision(state: AgentState) -> str:
    if state.get("is_relevant"):
        return "proceed"
    if state.get("retrieval_retry_count", 0) >= AGENT_MAX_RETRIES:
        return "proceed"
    return "retry"

def guard_decision(state: AgentState) -> str:
    if state.get("is_grounded"):
        return "pass"
    if state.get("hallucination_retry_count", 0) >= AGENT_MAX_RETRIES:
        return "pass"
    return "regenerate"
