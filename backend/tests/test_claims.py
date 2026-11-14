import pytest
from app.claims import INSUFFICIENT, completeness_problem, parse_answer, split_claims


@pytest.mark.parametrize("fragment", ["15 June 2019.", "45 million dollars.", "Lisbon.", "Revenue 12.5%", "12.5 percent", "In 2019"])
def test_fragments_are_flagged_incomplete(fragment):
    assert completeness_problem(fragment)


@pytest.mark.parametrize("good", [
    "The Lisbon office opened on 15 June 2019.",
    "The company's 2023 revenue was 45 million US dollars.",
    "Revenue was 45 million US dollars in 2023.",
    "This assessment is worth 40 percent of the final mark.",
    "The Router X200 uses a 12 V power supply.",
    "Employees may work remotely 3 days per week.",
])
def test_complete_claims_pass(good):
    assert completeness_problem(good) is None


@pytest.mark.parametrize("pronoun", ["It employs 120 people in Lisbon.", "They opened the office in 2019.", "This is worth 40 percent of the mark."])
def test_pronoun_led_claims_cannot_stand_alone(pronoun):
    assert "pronoun" in completeness_problem(pronoun)


def test_json_answer_with_sources_and_fences():
    raw = '```json\n{"insufficient": false, "claims": [{"text": "The Lisbon office opened on 15 June 2019.", "sources": ["S1", "s2"]}, {"text": "It has 120 staff.", "sources": ["S1"]}]}\n```'
    a = parse_answer(raw)
    assert a.fmt == "json" and not a.insufficient and len(a.claims) == 2
    assert a.claims[0].cited == ["S1", "S2"] and a.claims[0].problem is None
    assert a.claims[1].problem  # starts with a pronoun


def test_json_insufficient_flag():
    assert parse_answer('{"insufficient": true, "claims": []}').insufficient
    # claims present => the flag is ignored, they are verified
    a = parse_answer('{"insufficient": true, "claims": [{"text": "The office opened in 2019.", "sources": []}]}')
    assert not a.insufficient and len(a.claims) == 1


def test_citation_markers_inside_json_text_are_removed_and_read():
    a = parse_answer('{"claims": [{"text": "The office opened in 2019. [S3]", "sources": []}]}')
    assert a.claims[0].text == "The office opened in 2019." and a.claims[0].cited == ["S3"]


def test_plain_text_fallback_reads_citations_and_splits_clauses():
    a = parse_answer("Revenue grew 12.5% in 2023. [S2] The office opened in 2019; the site employs 120 staff. [S1, S3]")
    assert a.fmt == "text"
    assert [c.text for c in a.claims] == ["Revenue grew 12.5% in 2023.", "The office opened in 2019", "the site employs 120 staff."]
    assert a.claims[0].cited == ["S2"] and a.claims[2].cited == ["S1", "S3"]


def test_insufficient_marker_variants():
    assert parse_answer(INSUFFICIENT).insufficient and parse_answer(INSUFFICIENT + ".").insufficient
    # the phrase inside a real answer must not wipe out the answer
    a = parse_answer("The Lisbon office opened in 2019. [S1] The rest is INSUFFICIENT_EVIDENCE.")
    assert not a.insufficient and a.claims


def test_empty_and_garbage_replies():
    assert parse_answer("").fmt == "empty" and not parse_answer("   ").claims
    assert parse_answer("{not json").fmt == "text"


def test_claim_count_is_capped():
    raw = "{\"claims\": [" + ",".join('{"text": "The office %d opened in 2019.", "sources": []}' % i for i in range(40)) + "]}"
    assert len(parse_answer(raw).claims) == 12


def test_split_claims_keeps_decimals_and_abbreviations_together():
    cs = split_claims("Acme Inc. reported 12.5 percent growth. The Lisbon office opened in 2019.")
    assert [c.text for c in cs] == ["Acme Inc. reported 12.5 percent growth.", "The Lisbon office opened in 2019."]
