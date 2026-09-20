from langgraph.graph import StateGraph, END
from rag.agent.state import AgentState
from rag.agent import nodes

def build_agentic_rag_graph(retriever, llm):
    """Assembles and compiles the StateGraph for Agentic RAG."""
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("rewrite_query", nodes.create_rewriter_node(llm))
    workflow.add_node("router", nodes.create_router_node(llm))
    workflow.add_node("search_notes", nodes.create_search_node(retriever))
    workflow.add_node("calculate", nodes.calculate_node)
    workflow.add_node("list_docs", nodes.create_list_docs_node(retriever))
    workflow.add_node("grade_retrieval", nodes.create_grade_node(llm))
    workflow.add_node("reformulate_query", nodes.create_reformulate_node(llm))
    workflow.add_node("generate_answer", nodes.create_generator_node(llm, retriever))
    workflow.add_node("hallucination_guard", nodes.create_guard_node(llm))

    # 2. Add Edges & Conditional Routes
    
    # Entry point
    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "router")

    # Routing
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

    # Retrieval path
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

    # Direct tool edges to generation
    workflow.add_edge("calculate", "generate_answer")
    workflow.add_edge("list_docs", "generate_answer")

    # Hallucination Guard loop
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
