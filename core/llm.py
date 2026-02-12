from langchain_ollama import ChatOllama

import config


def get_llm(
    temperature: float = config.LLM_TEMPERATURE,
    num_ctx: int = config.LLM_NUM_CTX,
) -> ChatOllama:
    """Returns a ChatOllama instance configured for Gemma 3 4B."""
    return ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=2048,
    )
