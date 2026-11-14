"""Shared fixtures. All model/LLM calls are replaced by small deterministic fakes (tests only)."""
from __future__ import annotations

import pymupdf
import pytest
from app.config import Settings
from app.main import create_app
from app.services import build_services
from evaluation.mocks import HashedEmbedder, MockNLI
from fastapi.testclient import TestClient

# Deterministic stand-ins shared with the evaluation harness (tests only; they are NOT the real models).
FakeEmbedder = HashedEmbedder
FakeNLI = MockNLI


class ScriptedLLM:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0
        self.prompts: list[str] = []

    def complete(self, system, user, max_tokens):
        self.prompts.append(user)
        self.calls += 1
        return self.replies[min(self.calls - 1, len(self.replies) - 1)]


def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


REPORT_PAGES = [
    "Annual Report\nThe company opened its Lisbon office in 2019. The Lisbon office employs 120 people.",
    "Financial Results\nRevenue increased to 45 million dollars in 2023. Operating costs were stable across the year.",
    "Sustainability\nThe company reduced carbon emissions by 30 percent between 2020 and 2023.",
]


@pytest.fixture
def llm():
    return ScriptedLLM(["The company opened its Lisbon office in 2019. [S1]"])


@pytest.fixture
def settings(tmp_path):
    return Settings(
        data_dir=tmp_path, qdrant_url=":memory:", secret_key="x" * 40, llm_provider="openai", llm_model="test-model",
        llm_api_key="test-key", min_extracted_chars=20, chunk_words=40, chunk_overlap_words=5, rate_limit_per_minute=1000,
        min_dense_score=0.2, cors_origins="http://allowed.example",
    )


@pytest.fixture
def svc(settings, llm):
    s = build_services(settings, embedder=FakeEmbedder(), nli=FakeNLI(), llm_factory=lambda: llm)
    yield s
    s.db.close()


@pytest.fixture
def client(svc, settings):
    with TestClient(create_app(services=svc, settings=settings)) as c:
        yield c


def register(client, email="alice@example.com"):
    r = client.post("/api/auth/register", json={"email": email, "password": "correct-horse-battery"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


@pytest.fixture
def auth(client):
    return register(client)


def upload(client, headers, pages=None, name="report.pdf"):
    r = client.post("/api/documents", headers=headers, files={"file": (name, make_pdf(pages or REPORT_PAGES), "application/pdf")})
    assert r.status_code in (200, 202), r.text
    return r.json()
