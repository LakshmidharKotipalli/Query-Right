from routing.classifier import classify_query
from routing.web_search import web_search_query
from retrieval.chain import rag_query


def route_query(question: str) -> dict:
    """Route a query to either local RAG retrieval or web search."""
    route = classify_query(question)

    if route == "WEB":
        result = web_search_query(question)
    else:
        result = rag_query(question)

    result["route"] = route
    return result
