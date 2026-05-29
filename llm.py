"""LLM factory (Groq / Mistral / OpenAI compatible)."""
from langchain_openai import ChatOpenAI

from config import config


def get_chat_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
    )
