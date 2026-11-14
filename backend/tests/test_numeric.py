"""Numerical regression tests: typed comparison of quantities in a claim against one evidence sentence."""
import pytest
from app.numeric import compare_quantities, extract_quantities


def outcome(claim: str, evidence: str, context: str = "") -> str:
    r = compare_quantities(claim, evidence, context)
    if r.conflicts:
        return "conflict"
    return "missing" if r.missing else "ok"


@pytest.mark.parametrize("claim,evidence,expected", [
    # currencies
    ("Revenue was 82 million dollars.", "Revenue was 82 million euros.", "conflict"),
    ("Revenue was 82 million euros.", "Revenue was 82 million euros.", "ok"),
    ("Revenue was $82 million.", "Revenue was 82 million US dollars.", "ok"),
    ("Revenue was €82m.", "Revenue was EUR 82 million.", "ok"),
    ("Revenue was 82 million.", "Revenue was 82 million euros.", "ok"),  # a claim less specific than the evidence is fine
    ("Revenue was 82 million euros.", "Revenue was 82 million.", "missing"),  # more specific than evidence: unproven
    # percent vs plain numbers
    ("Margin was 50 units.", "Operating margin was 50%.", "missing"),
    ("Margin was 50%.", "Operating margin was 50 units.", "missing"),
    ("Growth was 125%.", "Growth was 12.5%.", "conflict"),
    ("Growth was 12.5%.", "Growth was 12.5 percent.", "ok"),
    ("Growth was 12.5%.", "Growth was 12.5 per cent.", "ok"),
    ("Growth was 5 percentage points.", "Growth was 5%.", "missing"),
    # magnitudes and formats
    ("Sales were 45 million.", "Sales were 45,000,000.", "ok"),
    ("Sales were 4.5 billion.", "Sales were 4,500 million.", "ok"),
    ("Users numbered 1,234,567.", "Users numbered 1234567.", "ok"),
    ("The fee is 2.5 lakh.", "The fee is 250,000.", "ok"),
    ("The share is -5%.", "The share is -5 percent.", "ok"),
    ("The change was -5%.", "The change was 5%.", "conflict"),
    # rounding of the claim is tolerated, a different value is not
    ("Revenue was $45 million.", "Revenue was 45.2 million dollars.", "ok"),
    ("Revenue was $45 million.", "Revenue was 45.9 million dollars.", "conflict"),
    ("Revenue was 45.0 million dollars.", "Revenue was 45.2 million dollars.", "conflict"),
    # units and dimensions
    ("The unit weighs 3.2 tonnes.", "The unit weighs 3,200 kilograms.", "ok"),
    ("The build takes 11 weeks.", "The build takes 11 days.", "conflict"),
    ("The warranty lasts 24 months.", "The warranty lasts 2 years.", "ok"),
    ("It employs 120 people.", "The office employs 120 employees.", "ok"),
    # dates and years
    ("It opened on 15 June 2019.", "It opened on 15 June 2019.", "ok"),
    ("It opened on 15 June 2019.", "It opened on 16 June 2019.", "conflict"),
    ("It opened on June 15, 2019.", "It opened on 15 June 2019.", "ok"),
    ("It opened on 2019-06-15.", "It opened on 15 June 2019.", "ok"),
    ("It opened in 2019.", "It opened on 15 June 2019.", "ok"),
    ("It opened on 15 June 2019.", "It opened in 2019.", "missing"),  # day not stated in the evidence
    ("It opened in 2021.", "It opened on 15 June 2019.", "conflict"),
    ("It opened in September 2021.", "It opened in October 2021.", "conflict"),
    ("The deadline is 3/4/2025.", "The deadline is 3 April 2025.", "ok"),  # day/month order is ambiguous: either reading accepted
    # ranges
    ("Emissions fell in 2021.", "Emissions fell between 2020 and 2023.", "ok"),
    ("Emissions fell in 2020.", "Emissions fell between 2020 and 2023.", "ok"),
    # identifiers: versions and addresses are compared exactly
    ("The address is 192.168.1.1.", "The address is 192.168.10.1.", "conflict"),
    ("The firmware is version 3.2.1.", "The firmware is version 3.2.1.", "ok"),
    ("The firmware is version 3.2.1.", "The firmware is version 4.0.0.", "conflict"),
    # identifiers that are not quantities
    ("See Table 2 on page 4.", "Table 3 is on page 9.", "ok"),
    ("The 3rd quarter was strong.", "The fourth quarter was weak.", "ok"),
])
def test_quantity_matrix(claim, evidence, expected):
    assert outcome(claim, evidence) == expected


def test_unrelated_number_in_evidence_is_not_a_conflict_when_claim_has_none():
    assert outcome("The Lisbon office is large.", "The Madrid office employs 95 people.") == "ok"


def test_several_candidates_means_missing_not_conflict():
    # Two amounts in the evidence: a differing claim cannot be pinned on either, so it is simply unproven.
    assert outcome("Revenue was 54 million dollars.", "Revenue was 45 million dollars, up from 40 million dollars.") == "missing"


def test_period_mismatch_never_becomes_a_conflict():
    r = compare_quantities("Revenue was 95 million euros in 2024.", "Revenue was 82 million euros in 2023.")
    assert not r.conflicts and r.period_mismatch and r.missing


def test_year_in_context_covers_claim_year():
    assert outcome("Revenue grew 12.5% from 2022 to 2023.", "Revenue grew 12.5 percent year over year.") == "missing"
    assert outcome("Revenue grew 12.5% from 2022 to 2023.", "Revenue grew 12.5 percent year over year.", "Financial Results 2022 2023") == "ok"


def test_extraction_types():
    kinds = {q.raw: q.kind for q in extract_quantities("On 15 June 2019 revenue rose 12.5% to $45m in 2023, 3 offices, v2.1")}
    assert kinds["15 June 2019"] == "date" and kinds["12.5%"] == "percent" and kinds["$45m"] == "money"
    assert kinds["2023"] == "year" and kinds["v2.1"] == "code"


def test_year_modifier_is_not_a_count():
    qs = extract_quantities("The company's 2023 revenue was 45 million US dollars.")
    assert [q.kind for q in qs] == ["year", "money"]
