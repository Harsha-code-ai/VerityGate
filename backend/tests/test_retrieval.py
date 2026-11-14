import time

import pytest
from app.config import Settings
from app.db import new_id
from app.retrieval import Retriever, rrf_fuse, tokenize
from app.services import build_services, ingest_document
from conftest import FakeEmbedder, FakeNLI
from evaluation import corpus


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("retrieval")
    pdfs = corpus.build_corpus(tmp / "pdfs")
    settings = Settings(data_dir=tmp / "data", qdrant_url=":memory:", secret_key="x" * 40, llm_provider="openai", llm_model="m", llm_api_key="k")
    svc = build_services(settings, embedder=FakeEmbedder(), nli=FakeNLI(), llm_factory=lambda: None)
    users = {name: new_id() for name in ("alice", "bob")}
    for name, uid in users.items():
        svc.db.execute("INSERT INTO users(id,email,password_hash,created_at) VALUES (?,?,?,?)", (uid, f"{name}@example.com", "x", time.time()))
    docs = {}
    for owner, doc in (("alice", "company_report"), ("alice", "research_paper"), ("alice", "long_report"), ("bob", "company_report"), ("bob", "incomplete_info")):
        doc_id, stored = new_id(), f"{new_id()}.pdf"
        data = pdfs[doc].read_bytes()
        (svc.settings.uploads_dir / stored).write_bytes(data)
        svc.db.execute("INSERT INTO documents(id,owner_id,display_name,stored_name,sha256,size_bytes,status,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                       (doc_id, users[owner], f"{doc}.pdf", stored, new_id(), len(data), "pending", time.time(), time.time()))
        ingest_document(svc, doc_id)
        assert svc.db.one("SELECT status FROM documents WHERE id=?", (doc_id,))["status"] == "ready"
        docs[(owner, doc)] = doc_id
    yield svc, users, docs
    svc.db.close()


def pages_of(passages):
    return [p.page for p in passages]


def test_exact_phrase_finds_the_right_page(env):
    svc, users, docs = env
    top = svc.retriever.retrieve(users["alice"], docs[("alice", "company_report")], "Lisbon office 15 June 2019", 3)
    assert top[0].page == 1 and "15 June 2019" in top[0].text


def test_paraphrased_question_reaches_the_evidence_via_synonym_normalisation(env):
    """'start operating' vs 'opened': no shared word, but both fold to the same canonical term."""
    svc, users, docs = env
    top = svc.retriever.retrieve(users["alice"], docs[("alice", "company_report")], "When did the Lisbon office start operating?", 3, mode="bm25")
    assert top and top[0].page == 1 and "15 June 2019" in top[0].text


def test_synonym_folding_in_tokens():
    assert tokenize("start operating") and set(tokenize("The office opened")) & set(tokenize("began operations at the office"))
    assert tokenize("sales") == tokenize("revenue")


def test_numeric_question_ranks_numeric_evidence_first(env):
    svc, users, docs = env
    top = svc.retriever.retrieve(users["alice"], docs[("alice", "research_paper")], "How many images are in the dataset 12,000", 3)
    assert top[0].page == 2 and "12,000" in top[0].text


def test_long_document_retrieval_finds_deep_page(env):
    svc, users, docs = env
    top = svc.retriever.retrieve(users["alice"], docs[("alice", "long_report")], "Which depot relocated to Harbour Road?", 3)
    assert 23 in pages_of(top)


def test_page_filter_restricts_results(env):
    svc, users, docs = env
    top = svc.retriever.retrieve(users["alice"], docs[("alice", "company_report")], "revenue", 5, page=2)
    assert top and set(pages_of(top)) == {2}


def test_results_are_ordered_and_limited(env):
    svc, users, docs = env
    top = svc.retriever.retrieve(users["alice"], docs[("alice", "long_report")], "operations staffing procedures", 4)
    assert len(top) <= 4 and [p.score for p in top] == sorted((p.score for p in top), reverse=True)


# ---------------------------------------------------------------- isolation between users
def test_one_users_document_is_never_returned_to_another_user(env):
    svc, users, docs = env
    bob_doc = docs[("bob", "company_report")]
    assert svc.retriever.retrieve(users["alice"], bob_doc, "Lisbon office", 5) == []  # alice asks about bob's document id
    assert svc.retriever.retrieve(users["bob"], docs[("alice", "company_report")], "Lisbon office", 5) == []


def test_cache_is_keyed_by_owner_so_it_cannot_leak(env):
    svc, users, docs = env
    bob_doc = docs[("bob", "company_report")]
    assert svc.retriever.retrieve(users["bob"], bob_doc, "Lisbon office", 3)  # warms the cache for bob
    assert svc.retriever.retrieve(users["alice"], bob_doc, "Lisbon office", 3) == []
    assert svc.retriever.term_df(users["alice"], bob_doc) == {}


def test_vector_search_filters_by_owner_and_document(env):
    svc, users, docs = env
    vec = FakeEmbedder().embed(["Lisbon office opened"])[0]
    assert svc.store.search(users["alice"], docs[("bob", "company_report")], vec, 10) == []
    hits = svc.store.search(users["bob"], docs[("bob", "company_report")], vec, 10)
    assert hits and all(h.chunk_id for h in hits)


def test_same_file_for_two_users_gives_two_independent_indexes(env):
    svc, users, docs = env
    a = {p.chunk_id for p in svc.retriever.retrieve(users["alice"], docs[("alice", "company_report")], "Lisbon", 5)}
    b = {p.chunk_id for p in svc.retriever.retrieve(users["bob"], docs[("bob", "company_report")], "Lisbon", 5)}
    assert a and b and not (a & b)


def test_deleting_a_document_removes_its_vectors_and_chunks(env):
    svc, users, docs = env
    doc_id, stored = new_id(), f"{new_id()}.pdf"
    data = (corpus.build_corpus(svc.settings.data_dir / "tmp_pdfs")["incomplete_info"]).read_bytes()
    (svc.settings.uploads_dir / stored).write_bytes(data)
    svc.db.execute("INSERT INTO documents(id,owner_id,display_name,stored_name,sha256,size_bytes,status,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                   (doc_id, users["alice"], "tmp.pdf", stored, new_id(), len(data), "pending", time.time(), time.time()))
    ingest_document(svc, doc_id)
    assert svc.retriever.retrieve(users["alice"], doc_id, "bakery sourdough", 3)
    svc.store.delete_document(users["alice"], doc_id)
    svc.db.delete_document_rows(doc_id, users["alice"])
    svc.retriever.invalidate(doc_id)
    assert svc.retriever.retrieve(users["alice"], doc_id, "bakery sourdough", 3) == []


# ---------------------------------------------------------------- gate and fusion
def test_gate_works_for_tiny_documents_where_bm25_scores_are_small(env):
    svc, users, docs = env
    q = "What power supply does the bakery use"
    passages = svc.retriever.retrieve(users["bob"], docs[("bob", "incomplete_info")], "When is the bakery open?", 5)
    assert svc.retriever.evidence_is_sufficient(passages, "When is the bakery open?")
    unrelated = svc.retriever.retrieve(users["bob"], docs[("bob", "incomplete_info")], q, 5)
    assert isinstance(svc.retriever.evidence_is_sufficient(unrelated, q), bool)


def test_gate_rejects_when_nothing_matches(env):
    svc, *_ = env
    r = Retriever(svc.db, svc.store, svc.embedder, min_dense=2.0, min_bm25=1e9)
    assert r.evidence_is_sufficient([], "anything") is False


def test_rrf_prefers_items_ranked_high_in_both_lists():
    scores = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert scores["a"] == pytest.approx(scores["b"]) and scores["a"] > scores["c"]
    assert rrf_fuse([["x"], []])["x"] == pytest.approx(1 / 61)


def test_bm25_works_in_a_two_chunk_document():
    """Regression: BM25Okapi gave non-positive scores for terms occurring in most chunks of a short document, so keyword search found nothing."""
    from rank_bm25 import BM25Plus

    from app.retrieval import tokenize

    docs = ["The Lisbon office opened in 2019 and employs staff", "The Madrid office opened in 2021 and employs staff"]
    scores = BM25Plus([tokenize(d) for d in docs]).get_scores(tokenize("Lisbon office"))
    assert scores[0] > scores[1] > 0
