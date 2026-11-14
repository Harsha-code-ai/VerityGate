"""Regression gate: the evaluation harness (mock models) must keep meeting minimum bars.

These bars protect the pipeline LOGIC (retrieval fusion, relevance gating, quantity checks, aggregation, abstention).
They say nothing about real-model accuracy - see TESTING_REPORT.md and `python -m evaluation.run --mode real`."""
import pytest
from evaluation.run import evaluate


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return evaluate("mock", tmp_path_factory.mktemp("pdfs"))


def test_retrieval_floors(result):
    h = result["retrieval"]["hybrid"]
    assert h["recall@3"] >= 0.95 and h["recall@5"] >= 0.95 and h["mrr"] >= 0.85
    assert h["recall@5"] >= result["retrieval"]["bm25"]["recall@5"]  # fusion never hurts


def test_verification_never_supports_something_it_should_not(result):
    v = result["verification"]
    assert v["all"]["false_support_rate"] == 0.0
    assert v["all"]["supported"]["precision"] == 1.0


def test_verification_logic_floors_on_cases_that_do_not_need_language_understanding(result):
    v = result["verification"]["non_semantic"]
    assert v["accuracy"] >= 0.95 and v["contradicted"]["precision"] == 1.0 and v["supported"]["recall"] >= 0.95
    assert v["conflicts_reported"] == "1/1"


def test_no_unanswerable_question_is_answered_as_verified(result):
    assert result["end_to_end"]["unanswerable_false_support"].startswith("0/")


def test_citations_point_to_the_right_pages(result):
    e = result["end_to_end"]
    assert e["citation_accuracy"] == 1.0 and e["citations_checked"] >= 20


def test_retries_are_rare_and_bounded(result):
    assert result["end_to_end"]["avg_llm_calls"] <= 1.5




def test_known_answerable_questions_are_not_abstained(result):
    failures = {
        item["id"]: item
        for item in result["end_to_end"]["failures"]
    }

    known_questions = {"q08", "q09", "q12", "q13", "q19"}

    failed_questions = known_questions & failures.keys()

    assert not failed_questions, (
        f"Known answerable questions failed: "
        f"{sorted(failed_questions)}"
    )



