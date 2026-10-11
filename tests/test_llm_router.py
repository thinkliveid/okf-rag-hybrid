import pytest

import llm
from knowledge import Concept

KNOWN = ["app/database.md", "app/stack.md", "other/deploy.md"]


def concept(path: str, tier: str = "unverified") -> Concept:
    return Concept(path=path, type="fact", title=path.title(), description=f"About {path}", body=f"Body of {path}", verified_tier=tier)


@pytest.mark.parametrize(
    "reply",
    [
        '{"concepts": ["app/database.md"]}',
        'Sure:\n```json\n{"concepts": ["app/database.md"]}\n```',
        '  {"concepts": ["app/database.md", "app/database.md"]}  ',
    ],
)
def test_parse_selection_accepts_wrapped_json_and_dedupes(reply):
    assert llm.parse_selection(reply, KNOWN, 3) == ["app/database.md"]


def test_parse_selection_drops_unknown_paths_and_caps_count():
    reply = '{"concepts": ["nope.md", "app/stack.md", 7, "app/database.md", "other/deploy.md"]}'
    assert llm.parse_selection(reply, KNOWN, 2) == ["app/stack.md", "app/database.md"]


def test_parse_selection_allows_empty_list():
    assert llm.parse_selection('{"concepts": []}', KNOWN, 3) == []


@pytest.mark.parametrize("reply", ["no idea", '{"paths": ["app/stack.md"]}', '["app/stack.md"]', ""])
def test_parse_selection_rejects_replies_without_concepts_object(reply):
    with pytest.raises(ValueError):
        llm.parse_selection(reply, KNOWN, 3)


def test_user_message_without_concepts_is_unchanged():
    docs = [{"text": "chunk", "metadata": {"source": "a.md"}}]
    assert llm.build_user_message("Q?", docs) == "<context>\n[1] (source: a.md)\nchunk\n</context>\n\nQuestion: Q?"


def test_user_message_with_concepts_puts_numbered_concepts_first():
    docs = [{"text": "chunk", "metadata": {"source": "a.md"}}]
    message = llm.build_user_message("Q?", docs, [concept("app/stack.md", "human")])

    assert message.startswith("<concepts>\n[K1] (concept: app/stack.md, verified: human)\n# App/Stack.Md\n\nBody of app/stack.md\n</concepts>")
    assert message.endswith("<context>\n[1] (source: a.md)\nchunk\n</context>\n\nQuestion: Q?")


def test_select_concepts_sends_listing_only_and_maps_paths(monkeypatch):
    calls = []

    def fake_complete(system, user):
        calls.append((system, user))
        return llm.LLMAnswer(text='{"concepts": ["other/deploy.md"]}', provider="fake", model="fake")

    monkeypatch.setattr(llm, "complete", fake_complete)
    concepts = [concept(p) for p in KNOWN]

    selected = llm.select_concepts("How to deploy?", concepts, 3)

    assert [c.path for c in selected] == ["other/deploy.md"]
    system, user = calls[0]
    assert "at most 3" in system
    assert "- other/deploy.md | Other/Deploy.Md | About other/deploy.md" in user
    assert "Body of" not in user
    assert user.endswith("Question: How to deploy?")


def test_generate_answer_switches_system_prompt_only_with_concepts(monkeypatch):
    systems = []
    monkeypatch.setattr(llm, "complete", lambda system, user: systems.append(system) or llm.LLMAnswer("ok", "fake", "fake"))

    llm.generate_answer("Q?", [])
    llm.generate_answer("Q?", [], [concept("app/stack.md")])

    assert systems == [llm.SYSTEM_PROMPT, llm.KNOWLEDGE_SYSTEM_PROMPT]
