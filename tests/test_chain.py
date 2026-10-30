import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

import lcel_chain
from fake_api import CHAT_ANSWER, FakeAPI
from lcel_chain import PROMPT, build_chain, build_model, main, responder


@pytest.fixture
def api():
    server = FakeAPI()
    yield server
    server.close()


@pytest.fixture
def fake_model():
    return GenericFakeChatModel(messages=iter([AIMessage(content="Respuesta de prueba")]))


def anthropic_model(api):
    return ChatAnthropic(model="claude-haiku-5-5", max_tokens=300, api_key="k", base_url=api.url)


def openai_model(api):
    return ChatOpenAI(
        model="gpt-4o-mini", temperature=0.2, max_tokens=300, api_key="k", base_url=f"{api.url}/v1"
    )


def test_la_cadena_es_prompt_pipe_modelo_pipe_parser(fake_model):
    chain = build_chain(fake_model)
    assert isinstance(chain, RunnableSequence)
    assert chain.first == PROMPT
    assert chain.middle == [fake_model]
    assert isinstance(chain.last, StrOutputParser)


def test_el_prompt_define_roles_system_y_human():
    messages = PROMPT.format_messages(pregunta="¿Qué es la entropía?")
    assert [type(m) for m in messages] == [SystemMessage, HumanMessage]
    assert messages[1].content == "¿Qué es la entropía?"


async def test_ainvoke_devuelve_texto_plano_y_no_un_mensaje(fake_model):
    result = await build_chain(fake_model).ainvoke({"pregunta": "hola"})
    assert isinstance(result, str) and not isinstance(result, BaseMessage)
    assert result == "Respuesta de prueba"


async def test_responder_pasa_la_pregunta_como_diccionario(fake_model):
    assert await responder(build_chain(fake_model), "hola") == "Respuesta de prueba"


async def test_variable_con_otro_nombre_falla(fake_model):
    with pytest.raises(KeyError, match="pregunta"):
        await build_chain(fake_model).ainvoke({"question": "hola"})


async def test_anthropic_real_sdk_texto_plano_y_payload(api):
    result = await build_chain(anthropic_model(api)).ainvoke({"pregunta": "¿Qué es la entropía?"})
    assert result == CHAT_ANSWER and isinstance(result, str)
    sent = api.requests[0]["body"]
    assert sent["system"].startswith("Eres un asistente técnico")
    assert sent["messages"] == [{"role": "user", "content": "¿Qué es la entropía?"}]
    assert "temperature" not in sent


async def test_openai_real_sdk_texto_plano_y_payload(api):
    result = await build_chain(openai_model(api)).ainvoke({"pregunta": "¿Qué es la entropía?"})
    assert result == CHAT_ANSWER and isinstance(result, str)
    sent = api.requests[0]["body"]
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert sent["temperature"] == 0.2


def test_build_model_anthropic_desde_entorno(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    model = build_model()
    assert isinstance(model, ChatAnthropic) and model.model == "claude-haiku-5-5"


def test_build_model_openai_desde_entorno(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.5")
    model = build_model()
    assert isinstance(model, ChatOpenAI) and model.model_name == "gpt-4o"
    assert model.temperature == 0.5


def test_build_model_sin_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_model("anthropic")


def test_build_model_proveedor_desconocido():
    with pytest.raises(ValueError, match="desconocido"):
        build_model("gemini")


async def test_main_imprime_la_respuesta(api, monkeypatch, capsys):
    monkeypatch.setattr(lcel_chain, "load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", api.url)
    await main()
    assert CHAT_ANSWER in capsys.readouterr().out


async def test_main_con_configuracion_invalida_no_se_cae(monkeypatch, capsys):
    monkeypatch.setattr(lcel_chain, "load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    await main()
    assert "Configuracion invalida" in capsys.readouterr().out
