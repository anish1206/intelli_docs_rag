from typing import List, Dict, Any
import simpleeval

def calculate(expression: str) -> str:
    """Safely evaluates math expressions using simpleeval.simple_eval."""
    try:
        # simple_eval only allows basic math operators by default
        result = simpleeval.simple_eval(expression)
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except SyntaxError:
        return "Error: Invalid math expression syntax."
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

def search_notes(retriever: Any, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Calls retriever to get context."""
    try:
        # Assuming retriever has a retrieve method
        chunks = retriever.retrieve(query=query, top_k=top_k)
        # Fallback if standard retriever interface is different
        return chunks
    except Exception as e:
        return []

def list_available_notes(retriever: Any) -> List[str]:
    """Extracts and returns a sorted list of unique document filenames/titles currently indexed."""
    try:
        # If retriever uses ChromaDB natively, access vector_store
        if hasattr(retriever, "vector_store") and hasattr(retriever.vector_store, "get"):
            metadata_list = retriever.vector_store.get(include=["metadatas"])["metadatas"]
            titles = set()
            for meta in metadata_list:
                if meta and "source" in meta:
                    titles.add(meta["source"])
            return sorted(list(titles))
        return []
    except Exception as e:
        return []
