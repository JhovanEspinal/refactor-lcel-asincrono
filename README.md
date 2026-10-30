# Refactorización a LCEL asíncrono

Ejercicio del Módulo 2 (AI Engineering): migrar de un cliente imperativo basado en SDKs crudos (Módulo 1) a una cadena **LCEL** declarativa y asíncrona con LangChain.

```
{"pregunta": "..."}  →  ChatPromptTemplate  →  ChatAnthropic / ChatOpenAI  →  StrOutputParser  →  str
```

## Antes y después

**Antes (imperativo, Módulo 1):** el código armaba la lista de mensajes, llamaba al SDK, y extraía el texto de `choices[0].message.content` (OpenAI) o `content[0].text` (Anthropic), con una rama por proveedor.

**Después (declarativo, este repo):**

```python
chain = PROMPT | model | StrOutputParser()
respuesta = await chain.ainvoke({"pregunta": "¿Qué es la entropía?"})
```

El prompt, el modelo y el parser son `Runnable`s intercambiables: cambiar de proveedor solo cambia el objeto `model`; la cadena no se toca.

## Criterios de aceptación

| Criterio | Dónde |
|---|---|
| Flujo principal solo con `\|` | `build_chain()` en `lcel_chain.py` |
| Ejecución asíncrona con `await chain.ainvoke(...)` | `responder()` |
| Salida en texto plano con `StrOutputParser` (no `AIMessage`) | último eslabón de la cadena |
| `ChatPromptTemplate` con roles System/Human | `PROMPT` |
| Variables del prompt = claves del diccionario (`pregunta`) | `PROMPT` / `responder()` |

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # luego completa tu API key
```

## Variables de entorno

| Variable | Descripción |
|---|---|
| `LLM_PROVIDER` | `anthropic` u `openai` (por defecto `anthropic`) |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Clave del proveedor elegido |
| `ANTHROPIC_MODEL` / `OPENAI_MODEL` | Opcionales. Por defecto `claude-haiku-5-5` / `gpt-4o-mini` |
| `LLM_TEMPERATURE` | Opcional, por defecto `0.2`. Solo se aplica a OpenAI |

## Ejecutar

```bash
python lcel_chain.py     # imprime la respuesta en texto plano
python -m pytest         # tests sin API keys ni red externa
```

## Notas

- **Temperatura en Anthropic:** la API actual de Anthropic (`anthropic-version: 2026-10-21`) rechaza `temperature`, `top_p` y `top_k`. Por eso `ChatAnthropic` se crea sin `temperature` y el parámetro solo se configura en `ChatOpenAI`. Los tests comprueban que no se envía.
- **Errores de configuración:** una API key faltante o un proveedor desconocido producen un mensaje claro (`ValueError`) en lugar de un fallo dentro de la cadena.
- **Pruebas:** `tests/test_chain.py` usa un modelo falso de LangChain para la lógica de la cadena y, además, ejecuta `ChatAnthropic` y `ChatOpenAI` **reales** contra un servidor HTTP local, verificando el request que realmente sale. Aún no se ha ejecutado contra los servidores reales de OpenAI/Anthropic: para eso se necesita una API key en `.env`.
