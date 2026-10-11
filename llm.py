from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

try:
    from .config import LLM_MAX_TOKENS, LLM_MODEL, LLM_PROVIDER, OLLAMA_BASE_URL
except ImportError:
    from config import LLM_MAX_TOKENS, LLM_MODEL, LLM_PROVIDER, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

ANTHROPIC_DEFAULT_MODEL = "claude-opus-5-5"
ANTHROPIC_MODELS_WITH_SERVER_FALLBACK = {
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
}

SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages provided by the user. "
    "Cite the passages you rely on with their numbers, like [1] or [2]. "
    "If the context does not contain the answer, say that you could not find it in the documents "
    "instead of guessing. Reply in the same language as the question."
)


class LLMConfigError(RuntimeError):
    pass


class LLMRefusalError(RuntimeError):
    pass


@dataclass
class LLMAnswer:
    text: str
    provider: str
    model: str


def build_context(docs: list[dict[str, Any]]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        metadata = doc.get("metadata") or {}
        source = metadata.get("source") or "unknown"
        parts.append(f"[{i}] (source: {source})\n{doc.get('text', '')}")
    return "\n\n".join(parts)


def build_user_message(question: str, docs: list[dict[str, Any]]) -> str:
    return f"<context>\n{build_context(docs)}\n</context>\n\nQuestion: {question}"


def _require_model(provider: str) -> str:
    if not LLM_MODEL:
        raise LLMConfigError(f"LLM_MODEL is required when LLM_PROVIDER={provider}")
    return LLM_MODEL


def _generate_anthropic(system: str, user: str) -> LLMAnswer:
    try:
        import anthropic
    except ImportError as exc:
        raise LLMConfigError("Install the 'anthropic' package to use LLM_PROVIDER=anthropic") from exc

    model = LLM_MODEL or ANTHROPIC_DEFAULT_MODEL
    client = anthropic.Anthropic()
    request: dict[str, Any] = {
        "model": model,
        "max_tokens": LLM_MAX_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if model in ANTHROPIC_MODELS_WITH_SERVER_FALLBACK:
        response = client.beta.messages.create(
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            **request,
        )
    else:
        response = client.messages.create(**request)

    if response.stop_reason == "refusal":
        category = response.stop_details.category if response.stop_details else None
        raise LLMRefusalError(f"Model declined to answer (category: {category})")

    text = "".join(block.text for block in response.content if block.type == "text")
    return LLMAnswer(text=text, provider="anthropic", model=response.model)


def _generate_openai_compatible(
    system: str,
    user: str,
    provider: str,
    model: str,
    token_limit_param: str,
    **client_kwargs: Any,
) -> LLMAnswer:
    try:
        import openai
    except ImportError as exc:
        raise LLMConfigError(f"Install the 'openai' package to use LLM_PROVIDER={provider}") from exc

    client = openai.OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **{token_limit_param: LLM_MAX_TOKENS},
    )
    return LLMAnswer(text=response.choices[0].message.content or "", provider=provider, model=response.model)


def _generate_openai(system: str, user: str) -> LLMAnswer:
    return _generate_openai_compatible(system, user, "openai", _require_model("openai"), "max_completion_tokens")


def _generate_ollama(system: str, user: str) -> LLMAnswer:
    return _generate_openai_compatible(
        system,
        user,
        "ollama",
        _require_model("ollama"),
        "max_tokens",
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
    )


def _generate_gemini(system: str, user: str) -> LLMAnswer:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise LLMConfigError("Install the 'google-genai' package to use LLM_PROVIDER=gemini") from exc

    model = _require_model("gemini")
    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=LLM_MAX_TOKENS,
        ),
    )
    return LLMAnswer(text=response.text or "", provider="gemini", model=model)


PROVIDERS: dict[str, Callable[[str, str], LLMAnswer]] = {
    "anthropic": _generate_anthropic,
    "openai": _generate_openai,
    "gemini": _generate_gemini,
    "ollama": _generate_ollama,
}


def generate_answer(question: str, docs: list[dict[str, Any]]) -> LLMAnswer:
    generate = PROVIDERS.get(LLM_PROVIDER)
    if generate is None:
        raise LLMConfigError(
            f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}. Choose one of: {', '.join(sorted(PROVIDERS))}"
        )
    answer = generate(SYSTEM_PROMPT, build_user_message(question, docs))
    logger.info("[LLM] %s/%s answered with %d chars", answer.provider, answer.model, len(answer.text))
    return answer
