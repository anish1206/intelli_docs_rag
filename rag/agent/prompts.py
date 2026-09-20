import json
import re

REWRITE_PROMPT = """Given the following conversation history and the user's latest question, reformulate the question to be a standalone query.
If the latest question contains pronouns or implicit references, resolve them using the conversation history.
If there is no history or the question is already standalone, return the question exactly as is without adding anything.

History:
{history}

Question:
{question}

Standalone question:"""

ROUTER_PROMPT = """You are a router. Based on the user's question, determine the most appropriate tool to handle it.
The available tools are:
- "search": use this to answer questions about the course material, syllabus, or any knowledge queries that would require retrieving documents.
- "calculate": use this to evaluate mathematical expressions, grades, formulas, etc.
- "list_docs": use this if the user wants to know what documents, notes, or PDFs are indexed/available in the system.
- "direct": use this for casual greetings, simple conversational chat, or questions that need no external information.

Question: {question}

You MUST output ONLY a valid JSON object in the following format:
{{"tool": "choice"}}
where choice is one of: search, calculate, list_docs, direct.
"""

CRAG_GRADER_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.

Retrieved document: 
{document}

User question: {question}

You MUST output ONLY a valid JSON object with a single boolean key "relevant":
{{"relevant": true}} or {{"relevant": false}}
"""

REFORMULATE_PROMPT = """The original query failed to retrieve relevant documents. 
Provide a reformulated version of the query using alternative keywords or synonyms to improve search results.
Just output the reformulated query and nothing else.

Original query: {question}

Reformulated query:"""

GENERATE_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
The context contains information from various documents with source citations.

Context:
{context}

Question: {question}

Instructions:
- Answer the question using only the provided context.
- If a tool provided output (e.g., calculator or doc lister), incorporate that smoothly into the answer.
- If the answer is not in the context, say you don't have enough information.
- Include source citations in your answer when relevant (e.g. [Source: filename, Page N]).
- Be specific and accurate.
- If multiple documents provide information, synthesize them coherently.

Answer:"""

GUARD_PROMPT = """You are a hallucination guard assessing whether an answer is grounded in and supported by a set of facts.
Are all factual claims in the drafted answer directly supported by the context?

Context:
{context}

Drafted Answer:
{answer}

You MUST output ONLY a valid JSON object with a single boolean key "grounded":
{{"grounded": true}} or {{"grounded": false}}
"""

def clean_json_response(raw_text: str) -> dict:
    """Strips markdown fences and parses JSON safely."""
    text = raw_text.strip()
    # Remove markdown code block backticks if present
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text).strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}
