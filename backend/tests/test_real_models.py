"""Integration tests with the REAL embedding and NLI models. Skipped automatically when the models cannot be loaded
(no internet / not installed). Run them with:  pytest -m real_models   (see docs in README, section Tests)."""
import pytest
from app.claims import ParsedClaim
from app.config import Settings
from app.retrieval import Passage
from app.verifier import CONTRADICTED, NOT_ESTABLISHED, SUPPORTED, ClaimVerifier, TransformersNLI

pytestmark = pytest.mark.real_models


@pytest.fixture(scope="module")
def nli():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    model = TransformersNLI(Settings().nli_model, Settings().nli_label_order)
    try:
        model._load()  # includes the label self-test
    except Exception as exc:  # offline, no cache, blocked download...
        pytest.skip(f"NLI model unavailable: {str(exc)[:120]}")
    return model


def P(i, page, text):
    return Passage(f"c{i}", "d", page, None, text)


def test_label_mapping_is_correct_on_clear_pairs(nli):
    ent, con, neu = nli.predict([
        ("The company opened its Lisbon office on 15 June 2019.", "The Lisbon office opened in 2019."),
        ("The company opened its Lisbon office on 15 June 2019.", "The company has never had an office in Lisbon."),
        ("The company opened its Lisbon office on 15 June 2019.", "The Madrid office employs 95 people."),
    ])
    assert ent.entailment > 0.7 and con.contradiction > 0.7 and neu.neutral > max(neu.entailment, neu.contradiction)


def test_lisbon_regression_with_the_real_model(nli):
    lisbon = P(1, 1, "The company opened its Lisbon office on 15 June 2019. The Lisbon office employs 120 people.")
    madrid = P(2, 2, "The Madrid office opened in 2021. Warehouse maintenance is scheduled for March.")
    r = ClaimVerifier(nli).verify("The Lisbon office opened on 15 June 2019.", ["S1"], {"S1": lisbon, "S2": madrid}, [lisbon, madrid])
    assert r.status == SUPPORTED


def test_revenue_regression_with_the_real_model(nli):
    p = P(3, 3, "Revenue increased by 12.5 percent from 2022 to 2023, driven by strong demand.")
    r = ClaimVerifier(nli).verify("Revenue increased by 12.5% from 2022 to 2023.", ["S1"], {"S1": p}, [p])
    assert r.status == SUPPORTED


def test_real_model_contradiction_and_missing_evidence(nli):
    p = P(1, 1, "The company opened its Lisbon office on 15 June 2019.")
    v = ClaimVerifier(nli)
    assert v.verify("The Lisbon office opened on 16 June 2019.", [], {}, [p]).status == CONTRADICTED
    assert v.verify("The company opened an office in Tokyo.", [], {}, [p]).status == NOT_ESTABLISHED
    claims = [ParsedClaim("The Lisbon office opened in 2019.", [])]
    assert v.verify_claims(claims, {}, [p])[0].status == SUPPORTED


def test_real_embedder_retrieves_paraphrases():
    pytest.importorskip("sentence_transformers")
    from app.embeddings import SentenceTransformerEmbedder

    emb = SentenceTransformerEmbedder(Settings().embedding_model)
    try:
        q, good, bad = emb.embed_query("When did the Lisbon office start operating?"), *emb.embed(
            ["The company opened its Lisbon office on 15 June 2019.", "Warehouse maintenance is scheduled for March."])
    except Exception as exc:
        pytest.skip(f"embedding model unavailable: {str(exc)[:120]}")
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731
    assert dot(q, good) > dot(q, bad)
