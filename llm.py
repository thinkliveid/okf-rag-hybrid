from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

try:
    from .config import LLM_MAX_TOKENS, LLM_MODEL, LLM_PROVIDER, OLLAMA_BASE_URL
    from .knowledge import Concept, render_listing
except ImportError:
    from config import LLM_MAX_TOKENS, LLM_MODEL, LLM_PROVIDER, OLLAMA_BASE_URL
    from knowledge import Concept, render_listing

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

KNOWLEDGE_SYSTEM_PROMPT = SYSTEM_PROMPT + (
    " Some passages are curated knowledge concepts numbered [K1], [K2]; cite them the same way. "
    "Prefer a concept over a numbered passage when they disagree."
)

SELECT_SYSTEM_PROMPT = (
    "You route questions to a curated knowledge base. You receive an index of concepts, one per line as "
    "'- <path> | <title> | <description>', followed by a question. Pick the concepts whose content is needed "
    "to answer the question, most relevant first, at most {max_concepts}. Reply with JSON only, no prose: "
    '{{"concepts": ["<path>", ...]}}. Reply {{"concepts": []}} when no concept applies.'
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


def build_concept_context(concepts: Sequence[Concept]) -> str:
    return "\n\n".join(
        f"[K{i}] (concept: {c.path}, verified: {c.verified_tier})\n# {c.title}\n\n{c.body}"
        for i, c in enumerate(concepts, 1)
    )


def build_user_message(question: str, docs: list[dict[str, Any]], concepts: Sequence[Concept] = ()) -> str:
    message = f"<context>\n{build_context(docs)}\n</context>\n\nQuestion: {question}"
    if not concepts:
        return message
    return f"<concepts>\n{build_concept_context(concepts)}\n</concepts>\n\n{message}"


def parse_selection(text: str, known_paths: Iterable[str], max_concepts: int) -> list[str]:
    candidates = [text.strip()] + re.findall(r"\{.*\}", text, re.S)
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("concepts"), list):
            break
    else:
        raise ValueError(f"Selection reply is not a JSON object with a 'concepts' list: {text[:200]!r}")

    known = set(known_paths)
    selected: list[str] = []
    for path in data["concepts"]:
        if isinstance(path, str) and path in known and path not in selected:
            selected.append(path)
    return selected[:max_concepts]


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


def complete(system: str, user: str) -> LLMAnswer:
    generate = PROVIDERS.get(LLM_PROVIDER)
    if generate is None:
        raise LLMConfigError(
            f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}. Choose one of: {', '.join(sorted(PROVIDERS))}"
        )
    return generate(system, user)


def select_concepts(question: str, concepts: Sequence[Concept], max_concepts: int) -> list[Concept]:
    by_path = {c.path: c for c in concepts}
    user = f"<index>\n{render_listing(list(concepts))}\n</index>\n\nQuestion: {question}"
    reply = complete(SELECT_SYSTEM_PROMPT.format(max_concepts=max_concepts), user)
    paths = parse_selection(reply.text, by_path, max_concepts)
    logger.info("[ROUTER] Selected %d/%d concepts: %s", len(paths), len(concepts), paths)
    return [by_path[p] for p in paths]


def generate_answer(question: str, docs: list[dict[str, Any]], concepts: Sequence[Concept] = ()) -> LLMAnswer:
    system = KNOWLEDGE_SYSTEM_PROMPT if concepts else SYSTEM_PROMPT
    answer = complete(system, build_user_message(question, docs, concepts))
    logger.info("[LLM] %s/%s answered with %d chars", answer.provider, answer.model, len(answer.text))
    return answer
