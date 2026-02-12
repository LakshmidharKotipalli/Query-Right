from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.llm import get_llm
import config


def web_search_query(question: str) -> dict:
    """Search the web via Tavily and synthesize an answer with the LLM."""
    from tavily import TavilyClient

    client = TavilyClient(api_key=config.TAVILY_API_KEY)
    search_results = client.search(
        query=question,
        max_results=config.TAVILY_MAX_RESULTS,
        include_answer=True,
    )

    # Format search results as context
    context_parts = []
    web_citations = []
    results = search_results.get("results", [])

    for i, result in enumerate(results, 1):
        title = result.get("title", "Web Source")
        url = result.get("url", "")
        content = result.get("content", "")
        context_parts.append(f"[Web Source {i}: {title}]\n{content}")
        web_citations.append({
            "source": title,
            "page": url,
            "file_type": "web",
            "preview": content[:150] + "..." if content else "",
        })

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No search results found."

    # Generate answer from web context
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful legal research assistant. Answer the question based on the web search results provided. Cite the web sources by their [Web Source N] markers. If the search results don't contain relevant information, say so clearly."""),
        ("human", """Web search results:
{context}

Question: {question}"""),
    ])

    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return {
        "answer": answer,
        "citations": web_citations,
        "source_documents": [],
    }
