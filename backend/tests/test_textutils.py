import pytest
from app.textutils import canon, content_terms, entity_anchors, normalize_text, split_sentences, stem, term_document_frequency


@pytest.mark.parametrize("text,expected", [
    ("Revenue grew 12.5 percent. Costs fell.", ["Revenue grew 12.5 percent.", "Costs fell."]),
    ("Dr. Smith joined Acme Inc. The firm grew.", ["Dr. Smith joined Acme Inc.", "The firm grew."]),
    ("See Fig. 3 for details. It rose.", ["See Fig. 3 for details.", "It rose."]),
    ("It opened in 2019. 45 people work there.", ["It opened in 2019.", "45 people work there."]),
    ("Version 3.2.1 shipped on 15 June 2019. Next.", ["Version 3.2.1 shipped on 15 June 2019.", "Next."]),
    ("J. Smith wrote it. He left.", ["J. Smith wrote it.", "He left."]),
])
def test_sentence_splitting(text, expected):
    assert split_sentences(text) == expected


def test_newline_is_a_boundary():
    assert split_sentences("Heading\nThe office opened.") == ["Heading", "The office opened."]


def test_normalize_text_repairs_ligatures_and_invisible_chars():
    assert normalize_text("o\ufb03ce\u00ad  a\u200bb") == "office ab"


def test_stemming_is_consistent_between_question_and_document():
    assert canon("opened") == canon("start") == canon("operating")
    assert stem("offices") == stem("office")
    assert canon("sales") == canon("revenue")


def test_content_terms_drop_numbers_by_default():
    assert "2019" not in content_terms("opened in 2019") and "2019" in content_terms("opened in 2019", keep_numbers=True)


def test_entity_anchors_ignore_numbers_months_and_sentence_start():
    anchors = entity_anchors("The Lisbon office opened on 15 June 2008 under Northstar. In 2008 it grew.")
    assert "lisbon" in anchors and "northstar" in anchors and "2008" not in anchors and "june" not in anchors


def test_term_document_frequency():
    df = term_document_frequency(["Lisbon office", "Madrid office", "Madrid warehouse"], min_docs=1)
    assert df[canon("office")] == pytest.approx(2 / 3) and df[canon("lisbon")] == pytest.approx(1 / 3)


def test_document_frequency_is_unknown_for_tiny_documents():
    assert term_document_frequency(["Lisbon office", "Madrid office"]) == {}
