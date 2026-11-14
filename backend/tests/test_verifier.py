"""Regression tests for the verification decision (the Lisbon / revenue audit findings) using scripted NLI scores.

The scripted NLI returns chosen scores per evidence sentence, so these tests pin down the DECISION LOGIC independent of
any model. Real-model behaviour is covered separately in test_real_models.py (skipped unless models are installed)."""
import pytest
from app.claims import ParsedClaim
from app.retrieval import Passage
from app.textutils import term_document_frequency
from app.verifier import (
    CONTRADICTED,
    NOT_ESTABLISHED,
    SUPPORTED,
    ClaimVerifier,
    NLIScores,
    TransformersNLI,
    VerifierConfig,
    resolve_nli_labels,
)

NEUTRAL = (0.05, 0.05, 0.90)


class ScriptedNLI:
    """table: {substring of premise: (entailment, contradiction, neutral)}; records every pair it receives."""

    def __init__(self, table):
        self.table, self.pairs = table, []

    def predict(self, pairs):
        self.pairs.extend(pairs)
        out = []
        for premise, _ in pairs:
            scores = next((v for k, v in self.table.items() if k in premise), NEUTRAL)
            out.append(NLIScores(*scores))
        return out


def P(i, page, text, section=None):
    return Passage(f"c{i}", "d", page, section, text)


def verify(nli, claim, passages, cited=("S1",), **kw):
    id_map = {f"S{i}": p for i, p in enumerate(passages, 1)}
    df = term_document_frequency([p.text for p in passages])
    return ClaimVerifier(nli, **kw).verify(claim, list(cited), id_map, passages, df)


LISBON = P(1, 1, "The company opened its Lisbon office on 15 June 2019. The Lisbon office employs 120 people.")
MADRID = P(2, 2, "The Madrid office opened in 2021. Warehouse maintenance is scheduled for March.")
CLAIM = "The Lisbon office opened on 15 June 2019."


def test_unrelated_contradiction_never_overrides_supporting_evidence():
    """THE reported bug: page 1 entails the claim; unrelated Madrid/warehouse sentences score a high contradiction."""
    nli = ScriptedNLI({"Lisbon office on": (0.95, 0.02, 0.03), "Madrid": (0.01, 0.97, 0.02), "Warehouse": (0.01, 0.98, 0.01)})
    r = verify(nli, CLAIM, [LISBON, MADRID])
    assert r.status == SUPPORTED and not r.conflict
    assert [e.page for e in r.evidence if e.role == "supporting"] == [1]
    assert all(e.role != "contradicting" for e in r.evidence)


def test_contradiction_from_unrelated_sentence_alone_is_not_a_contradiction():
    nli = ScriptedNLI({"Madrid": (0.01, 0.97, 0.02), "Warehouse": (0.01, 0.98, 0.01)})
    r = verify(nli, CLAIM, [MADRID])
    assert r.status == NOT_ESTABLISHED


def test_nli_receives_one_sentence_as_premise_and_the_claim_as_hypothesis():
    nli = ScriptedNLI({"Lisbon office on": (0.95, 0.02, 0.03)})
    verify(nli, CLAIM, [LISBON, MADRID])
    assert nli.pairs, "NLI was never called"
    for premise, hypothesis in nli.pairs:
        assert hypothesis == CLAIM
        assert premise.count(".") <= 2 and len(premise.split()) < 25, f"premise is not a single sentence: {premise!r}"
    assert ("The company opened its Lisbon office on 15 June 2019.", CLAIM) in nli.pairs


def test_revenue_increase_is_supported_when_years_are_in_the_heading():
    """Second reported bug: correct claim, evidence says '12.5 percent' without repeating the years."""
    p = P(3, 3, "Revenue grew 12.5 percent year over year, driven by strong demand.", "Financial Results 2022 2023")
    r = verify(ScriptedNLI({"Revenue grew": (0.97, 0.01, 0.02)}), "Revenue increased by 12.5% from 2022 to 2023.", [p])
    assert r.status == SUPPORTED


def test_revenue_increase_supported_when_years_are_in_the_sentence():
    p = P(3, 3, "Revenue increased by 12.5 percent from 2022 to 2023.")
    assert verify(ScriptedNLI({"Revenue": (0.97, 0.01, 0.02)}), "Revenue increased by 12.5% from 2022 to 2023.", [p]).status == SUPPORTED


def test_percent_written_differently_is_still_a_match():
    p = P(3, 3, "Margins improved to 12.5 per cent.")
    assert verify(ScriptedNLI({"Margins": (0.96, 0.01, 0.03)}), "Margins improved to 12.5%.", [p]).status == SUPPORTED


def test_year_missing_everywhere_is_not_established_even_if_nli_entails():
    p = P(3, 3, "Revenue grew 12.5 percent year over year.")
    r = verify(ScriptedNLI({"Revenue grew": (0.97, 0.01, 0.02)}), "Revenue increased by 12.5% from 2022 to 2023.", [p])
    assert r.status == NOT_ESTABLISHED and r.reason_code == "quantity_not_stated"


def test_currency_error_is_contradicted_even_when_nli_entails():
    p = P(5, 4, "Revenue was 82 million euros in 2023.")
    nli = ScriptedNLI({"Revenue was": (0.96, 0.01, 0.03)})  # NLI models are often blind to currencies
    assert verify(nli, "Revenue was 82 million dollars in 2023.", [p]).status == CONTRADICTED
    assert verify(nli, "Revenue was 82 million euros in 2023.", [p]).status == SUPPORTED


def test_unit_confusion_is_not_supported():
    p = P(5, 4, "The operating margin was 50%.")
    r = verify(ScriptedNLI({"margin": (0.96, 0.01, 0.03)}), "The operating margin was 50 units.", [p])
    assert r.status != SUPPORTED


def test_percent_magnitude_error_is_contradicted():
    p = P(5, 4, "Growth in the region was 12.5%.")
    assert verify(ScriptedNLI({"Growth": (0.6, 0.3, 0.1)}), "Growth in the region was 125%.", [p]).status == CONTRADICTED


def test_number_from_unrelated_topic_does_not_contradict():
    passages = [P(1, 1, "The Madrid office employs 95 people."), P(2, 1, "The Lisbon office opened on 15 June 2019.")]
    nli = ScriptedNLI({"Lisbon office opened": (0.95, 0.02, 0.03)})
    assert verify(nli, "The Lisbon office opened on 15 June 2019.", passages).status == SUPPORTED


def test_other_period_is_not_a_contradiction():
    p = P(1, 1, "Group revenue was 82 million euros in 2023.")
    r = verify(ScriptedNLI({"Group revenue": (0.1, 0.9, 0.0)}), "Group revenue was 95 million euros in 2024.", [p])
    assert r.status == NOT_ESTABLISHED  # about 2024; the sentence is about 2023


def test_genuine_conflict_between_relevant_pages_is_reported_not_verified():
    a = P(6, 1, "Northwind was founded in 2008 in Rotterdam.")
    b = P(7, 5, "Founded in 2012, Northwind has grown quickly.")
    nli = ScriptedNLI({"in 2008": (0.95, 0.02, 0.03), "Founded in 2012": (0.02, 0.96, 0.02)})
    r = verify(nli, "Northwind was founded in 2008.", [a, b], cited=("S1", "S2"))
    assert r.status == NOT_ESTABLISHED and r.conflict and r.reason_code == "conflict"
    roles = {e.page: e.role for e in r.evidence}
    assert roles[1] == "supporting" and roles[5] == "contradicting"  # both are shown to the user


def test_only_contradicting_page_gives_contradicted():
    b = P(7, 5, "Founded in 2012, Northwind has grown quickly.")
    r = verify(ScriptedNLI({"Founded in 2012": (0.02, 0.96, 0.02)}), "Northwind was founded in 2008.", [b])
    assert r.status == CONTRADICTED and r.evidence[0].role == "contradicting"


def test_weak_contradiction_does_not_override_strong_support():
    a = P(1, 1, "The plant employs 300 people at its main site.")
    b = P(2, 2, "The plant employs staff at its main site only part of the year.")
    nli = ScriptedNLI({"300 people": (0.97, 0.01, 0.02), "part of the year": (0.1, 0.86, 0.04)})
    r = verify(nli, "The plant employs 300 people at its main site.", [a, b])
    assert r.status == SUPPORTED


def test_different_identifier_cannot_support_the_claim():
    a = P(1, 1, "The Router X300 uses a 24 V power supply.")
    b = P(1, 1, "The Router X200 uses a 12 V power supply.")
    nli = ScriptedNLI({"X300": (0.95, 0.02, 0.03), "X200": (0.03, 0.92, 0.05)})
    r = verify(nli, "The Router X200 uses a 24 V power supply.", [a, b])
    assert r.status == CONTRADICTED


def test_version_mismatch_is_contradicted():
    p = P(1, 1, "The Router X300 ships with firmware version 4.0.0.")
    assert verify(ScriptedNLI({"X300": (0.9, 0.05, 0.05)}), "The Router X300 ships with firmware version 3.2.1.", [p]).status == CONTRADICTED


def test_direction_reversal_with_same_number_is_not_supported():
    p = P(1, 1, "Operating costs rose by 5% in 2023.")
    r = verify(ScriptedNLI({"costs": (0.9, 0.05, 0.05)}), "Operating costs fell by 5% in 2023.", [p])
    assert r.status == CONTRADICTED and r.reason_code == "contradicted_direction"


def test_missing_evidence_is_not_established_not_contradicted():
    r = verify(ScriptedNLI({}), "The company opened a Tokyo office in 2020.", [LISBON, MADRID])
    assert r.status == NOT_ESTABLISHED


def test_incomplete_claim_is_flagged_without_calling_nli():
    nli = ScriptedNLI({})
    v = ClaimVerifier(nli)
    res = v.verify_claims([ParsedClaim("15 June 2019.", ["S1"], problem="too short")], {"S1": LISBON}, [LISBON])
    assert res[0].status == NOT_ESTABLISHED and res[0].reason_code == "incomplete_claim" and not nli.pairs


def test_multiple_claims_are_judged_independently():
    nli = ScriptedNLI({"Lisbon office on": (0.95, 0.02, 0.03), "employs 120": (0.95, 0.02, 0.03)})
    claims = [ParsedClaim(CLAIM, ["S1"]), ParsedClaim("The Lisbon office employs 150 people.", ["S1"]), ParsedClaim("The Lisbon office employs 120 people.", ["S1"])]
    res = ClaimVerifier(nli).verify_claims(claims, {"S1": LISBON}, [LISBON], term_document_frequency([LISBON.text]))
    assert [r.status for r in res] == [SUPPORTED, CONTRADICTED, SUPPORTED]


def test_coreference_sentence_gets_previous_sentence_as_premise():
    p = P(1, 1, "The Lisbon office opened in 2019. It employs 120 people.")
    nli = ScriptedNLI({"employs 120": (0.95, 0.02, 0.03)})
    r = verify(nli, "The Lisbon office employs 120 people.", [p])
    assert r.status == SUPPORTED
    assert any(prem.startswith("The Lisbon office opened in 2019. It employs") for prem, _ in nli.pairs)


def test_truncated_premise_scores_are_ignored():
    class Truncating(ScriptedNLI):
        def predict(self, pairs):
            return [NLIScores(0.0, 0.99, 0.01, truncated=True) for _ in pairs]

    r = verify(Truncating({}), CLAIM, [LISBON])
    assert r.status == NOT_ESTABLISHED  # an unreliable score must not create a contradiction


def test_wrong_citation_is_flagged_but_real_source_reported():
    nli = ScriptedNLI({"Lisbon office on": (0.95, 0.02, 0.03)})
    r = verify(nli, CLAIM, [LISBON, MADRID], cited=("S2", "S9"))
    assert r.status == SUPPORTED and not r.citation_valid
    assert any("not provided" in w for w in r.warnings) and any("did not cite" in w for w in r.warnings)


def test_table_row_sentence_supports_claim():
    p = P(1, 1, "Revenue: 2023 is 45.0; 2022 is 40.0.", "Revenue by Region")
    nli = ScriptedNLI({"Revenue:": (0.9, 0.02, 0.08)})
    assert verify(nli, "Revenue in 2023 was 45.0.", [p]).status == SUPPORTED


def test_thresholds_are_respected():
    nli = ScriptedNLI({"Lisbon office on": (0.75, 0.02, 0.23)})
    assert verify(nli, CLAIM, [LISBON]).status == NOT_ESTABLISHED
    assert verify(nli, CLAIM, [LISBON], config=VerifierConfig(entail_thr=0.7)).status == SUPPORTED


# --------------------------------------------------------------------------- NLI label mapping
@pytest.mark.parametrize("labels,name,expected", [
    ({0: "contradiction", 1: "entailment", 2: "neutral"}, "cross-encoder/nli-deberta-v3-small", {"contradiction": 0, "entailment": 1, "neutral": 2}),
    ({0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}, "x", {"entailment": 0, "neutral": 1, "contradiction": 2}),
    ({0: "entailment", 1: "not_entailment"}, "x-2c", {"entailment": 0, "neutral": 1}),
])
def test_label_mapping_from_config(labels, name, expected):
    assert resolve_nli_labels(labels, name) == expected


def test_generic_labels_use_known_model_family_or_fail():
    generic = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
    assert resolve_nli_labels(generic, "cross-encoder/nli-roberta-base") == {"contradiction": 0, "entailment": 1, "neutral": 2}
    assert resolve_nli_labels(generic, "facebook/bart-large-mnli") == {"contradiction": 0, "neutral": 1, "entailment": 2}
    with pytest.raises(RuntimeError, match="NLI_LABEL_ORDER"):
        resolve_nli_labels(generic, "some/unknown-model")


def test_explicit_label_order_override_wins():
    generic = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
    assert resolve_nli_labels(generic, "some/unknown-model", "neutral,entailment,contradiction") == {"neutral": 0, "entailment": 1, "contradiction": 2}
    with pytest.raises(RuntimeError):
        resolve_nli_labels(generic, "x", "entailment,entailment,neutral")


class _FakeHF(TransformersNLI):
    """TransformersNLI with the torch parts replaced so the label self-test can run without any model."""

    def __init__(self, probs_by_probe, labels):
        super().__init__("fake/model")
        self._labels, self._probs = labels, probs_by_probe

    def _probabilities(self, pairs):
        return [self._probs[i] for i in range(len(pairs))], [False] * len(pairs)


def test_self_test_passes_for_correct_mapping_and_fails_for_swapped_one():
    labels = {"contradiction": 0, "entailment": 1, "neutral": 2}
    good = _FakeHF([[0.01, 0.98, 0.01], [0.97, 0.02, 0.01]], labels)  # probe 1 entails, probe 2 contradicts
    good.self_test()
    swapped = _FakeHF([[0.98, 0.01, 0.01], [0.01, 0.97, 0.02]], labels)  # logits order differs from the mapping
    with pytest.raises(RuntimeError, match="self-test failed"):
        swapped.self_test()


def test_scores_use_the_resolved_label_indices():
    m = _FakeHF([], {"contradiction": 2, "entailment": 0, "neutral": 1})
    s = m._to_scores([0.7, 0.2, 0.1], False)
    assert (s.entailment, s.neutral, s.contradiction) == (0.7, 0.2, 0.1)
