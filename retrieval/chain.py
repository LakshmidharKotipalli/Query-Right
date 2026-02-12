from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.llm import get_llm
from retrieval.retriever import retrieve_with_sources, format_context, extract_citations

LEGAL_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a legal document assistant. Answer questions based ONLY on the provided context from legal/policy documents. Follow these rules strictly:

1. If the answer is found in the context, provide it with specific references to the source documents (e.g., "According to [Source 1: contract.pdf, Page 3]...").
2. If the context does not contain enough information to answer, say: "I could not find sufficient information in the uploaded documents to answer this question."
3. Do NOT make up or infer legal conclusions beyond what the documents state.
4. Use precise legal language when the source documents use it.
5. If multiple sources address the question, synthesize them and cite each."""),
    ("human", """Context from documents:
{context}

Question: {question}

Provide a thorough answer with source citations."""),
])


def rag_query(question: str) -> dict:
    """Execute a full RAG query: retrieve -> format context -> generate answer."""
    docs = retrieve_with_sources(question)

    if not docs:
        return {
            "answer": "No relevant documents found. Please upload documents first.",
            "citations": [],
            "source_documents": [],
        }

    context = format_context(docs)

    llm = get_llm()
    chain = LEGAL_QA_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    citations = extract_citations(docs)

    return {
        "answer": answer,
        "citations": citations,
        "source_documents": docs,
    }
