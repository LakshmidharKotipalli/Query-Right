from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.llm import get_llm

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query classifier. Determine whether a user's question should be answered from LOCAL documents or from the WEB.

Classify as LOCAL if the question:
- Asks about specific contracts, policies, agreements, or legal documents
- References particular clauses, sections, or terms from documents
- Asks about document-specific definitions, obligations, or rights
- Can likely be answered from uploaded legal/policy documents

Classify as WEB if the question:
- Asks about current events, recent news, or real-time information
- Asks about general legal concepts, case law, or statutory references NOT in uploaded docs
- Asks about publicly available regulations or government policies
- Requires information that changes frequently (dates, prices, current status)

Respond with EXACTLY one word: LOCAL or WEB"""),
    ("human", "{question}"),
])


def classify_query(question: str) -> str:
    """Classify a query as needing 'LOCAL' retrieval or 'WEB' search."""
    llm = get_llm(temperature=0.0)
    chain = CLASSIFICATION_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"question": question}).strip().upper()

    if result not in ("LOCAL", "WEB"):
        return "LOCAL"
    return result
