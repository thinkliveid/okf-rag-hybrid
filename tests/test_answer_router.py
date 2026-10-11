from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api
import llm

DOCS = [{"text": "rag chunk", "metadata": {"source": "doc.md"}, "score": 0.5}]


class FakeDoc:
    def __init__(self, d):
        self.text, self.metadata, self.score = d["text"], d["metadata"], d["score"]


def write_concept(root: Path, rel: str, status: str = "active") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\ntype: fact\ntitle: {rel}\ndescription: about {rel}\nstatus: {status}\n---\n\nbody of {rel}\n",
        encoding="utf-8",
    )


class FakeLLM:
    def __init__(self, selection_reply="", selection_error=None):
        self.selection_reply = selection_reply
        self.selection_error = selection_error
        self.calls = []

    def __call__(self, system, user):
        is_selection = system.startswith("You route questions")
        self.calls.append(("select" if is_selection else "answer", user))
        if is_selection:
            if self.selection_error:
                raise self.selection_error
            return llm.LLMAnswer(self.selection_reply, "fake", "fake-model")
        return llm.LLMAnswer("final answer", "fake", "fake-model")

    def kinds(self):
        return [kind for kind, _ in self.calls]

    def answer_message(self):
        return next(user for kind, user in self.calls if kind == "answer")


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "KNOWLEDGE_DIR", tmp_path)
    monkeypatch.setattr(api, "advanced_search", lambda query, top_k: [FakeDoc(d) for d in DOCS])
    return TestClient(api.app)


def use_llm(monkeypatch, fake):
    monkeypatch.setattr(llm, "complete", fake)
    return fake


def test_without_knowledge_dir_behaves_like_plain_rag(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "KNOWLEDGE_DIR", tmp_path / "missing")
    fake = use_llm(monkeypatch, FakeLLM())

    body = client.post("/api/v1/answer", json={"query": "Q?"}).json()

    assert fake.kinds() == ["answer"]
    assert "<concepts>" not in fake.answer_message()
    assert body["answer"] == "final answer"
    assert body["count"] == 1
    assert body["concepts"] == []
    assert body["router"] == {"enabled": False, "selected": [], "error": None}


def test_selected_concepts_reach_the_answer_and_the_response(client, monkeypatch, tmp_path):
    write_concept(tmp_path, "app/db.md")
    write_concept(tmp_path, "app/stack.md")
    fake = use_llm(monkeypatch, FakeLLM('{"concepts": ["app/db.md"]}'))

    body = client.post("/api/v1/answer", json={"query": "Which database?"}).json()

    assert fake.kinds() == ["select", "answer"]
    message = fake.answer_message()
    assert "[K1] (concept: app/db.md, verified: unverified)" in message
    assert "body of app/db.md" in message
    assert "body of app/stack.md" not in message
    assert "rag chunk" in message
    assert body["concepts"] == [{"path": "app/db.md", "title": "app/db.md", "verified_tier": "unverified"}]
    assert body["router"] == {"enabled": True, "selected": ["app/db.md"], "error": None}


def test_use_knowledge_false_skips_selection(client, monkeypatch, tmp_path):
    write_concept(tmp_path, "app/db.md")
    fake = use_llm(monkeypatch, FakeLLM('{"concepts": ["app/db.md"]}'))

    body = client.post("/api/v1/answer", json={"query": "Q?", "use_knowledge": False}).json()

    assert fake.kinds() == ["answer"]
    assert body["router"]["enabled"] is False


@pytest.mark.parametrize(
    "fake",
    [FakeLLM(selection_reply="I cannot decide"), FakeLLM(selection_error=RuntimeError("provider down")), FakeLLM(selection_error=llm.LLMRefusalError("declined"))],
)
def test_selection_failures_fall_back_to_rag_only(client, monkeypatch, tmp_path, fake):
    write_concept(tmp_path, "app/db.md")
    use_llm(monkeypatch, fake)

    response = client.post("/api/v1/answer", json={"query": "Q?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "final answer"
    assert body["concepts"] == []
    assert body["router"]["enabled"] is True
    assert body["router"]["error"]
    assert "<concepts>" not in fake.answer_message()


def test_unknown_selected_paths_are_ignored(client, monkeypatch, tmp_path):
    write_concept(tmp_path, "app/db.md")
    use_llm(monkeypatch, FakeLLM('{"concepts": ["made/up.md"]}'))

    body = client.post("/api/v1/answer", json={"query": "Q?"}).json()

    assert body["concepts"] == []
    assert body["router"] == {"enabled": True, "selected": [], "error": None}


def test_deprecated_concepts_are_never_listed(client, monkeypatch, tmp_path):
    write_concept(tmp_path, "app/old.md", status="deprecated")
    write_concept(tmp_path, "app/new.md")
    fake = use_llm(monkeypatch, FakeLLM('{"concepts": ["app/old.md"]}'))

    body = client.post("/api/v1/answer", json={"query": "Q?"}).json()

    selection_message = fake.calls[0][1]
    assert "app/old.md" not in selection_message
    assert body["concepts"] == []


def test_misconfigured_provider_still_returns_503(client, monkeypatch, tmp_path):
    write_concept(tmp_path, "app/db.md")
    use_llm(monkeypatch, FakeLLM(selection_error=llm.LLMConfigError("LLM_MODEL is required")))

    response = client.post("/api/v1/answer", json={"query": "Q?"})

    assert response.status_code == 503
