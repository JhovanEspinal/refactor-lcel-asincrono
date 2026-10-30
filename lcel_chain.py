import asyncio
import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODELS = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-5-5"}
API_KEY_VARS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1024
SAMPLE_QUESTION = "¿Qué es la entropía?"

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un asistente técnico. Responde en español, de forma clara y breve.",
        ),
        ("human", "{pregunta}"),
    ]
)


def build_model(provider: str | None = None) -> BaseChatModel:
    provider = provider or os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
    if provider not in API_KEY_VARS:
        raise ValueError(
            f"Proveedor desconocido: {provider!r}. Opciones: {', '.join(API_KEY_VARS)}"
        )
    key_var = API_KEY_VARS[provider]
    api_key = os.getenv(key_var)
    if not api_key:
        raise ValueError(f"Falta la variable de entorno {key_var} para {provider}.")

    model_name = os.getenv(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider]
    base_url = os.getenv(f"{provider.upper()}_BASE_URL") or None

    if provider == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=float(os.getenv("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)),
            max_tokens=DEFAULT_MAX_TOKENS,
            api_key=api_key,
            base_url=base_url,
        )
    return ChatAnthropic(
        model=model_name,
        max_tokens=DEFAULT_MAX_TOKENS,
        api_key=api_key,
        base_url=base_url,
    )


def build_chain(model: BaseChatModel) -> Runnable[dict, str]:
    return PROMPT | model | StrOutputParser()


async def responder(chain: Runnable[dict, str], pregunta: str) -> str:
    return await chain.ainvoke({"pregunta": pregunta})


async def main() -> None:
    load_dotenv()
    try:
        chain = build_chain(build_model())
    except ValueError as exc:
        print(f"Configuracion invalida: {exc}")
        return

    print(f"Pregunta: {SAMPLE_QUESTION}\n")
    print(await responder(chain, SAMPLE_QUESTION))


if __name__ == "__main__":
    asyncio.run(main())
