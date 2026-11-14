
"""Deterministic stand-ins for the ML models.

These mocks are ONLY for fast regression testing. They do not measure
real embedding, NLI, or LLM accuracy.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

from app.textutils import STOPWORDS, canon, content_terms, direction_of, split_sentences
from app.verifier import NLIScores


_WORD = re.compile(r"[a-z0-9]+")
_NEGATIONS = {"not", "never", "no", "cannot", "n't", "without"}

# Words that describe question wording rather than document facts.
# Values are normalized below to match content_terms() output.
_GENERIC_QUESTION_TERMS = {
    "many", "much", "long", "work", "big", "old", "large", "far",
    "often", "kind", "type", "something", "which", "whether",
    "mention", "describe", "tell", "give", "state",
    "company", "came", "use", "used", "does", "did",
    "what", "how", "who", "where", "when",
}

# Small lexical aliases. These help matching but do NOT establish truth.
_QUESTION_ALIASES = {
    "co2": {"co2", "carbon", "dioxide"},
    "output": {"output", "emission", "carbon"},
    "share": {"share", "percent", "percentage", "proportion"},
    "currency": {"currency", "euro", "dollar", "usd", "gbp"},
    "limitation": {
        "limitation", "limit", "evaluation", "measure",
        "dataset", "energy", "constraint",
    },
    "study": {"study", "paper", "research", "evaluation"},
    "duration": {"duration", "period", "month", "year", "week"},
    "last": {"last", "period", "month", "year", "week"},
    "limit": {"limit", "exceed", "maximum"},
}


def _normalized_terms(text: str) -> set[str]:
    """Return terms in the same normalized form used by the mock matcher."""
    terms = set(content_terms(text, keep_numbers=True))

    normalized = set(terms)

    for term in terms:
        normalized.add(canon(term))

    return {term for term in normalized if term}


# Normalize generic words once so stemming differences do not break filtering.
_GENERIC_NORMALIZED = set()

for _generic_word in _GENERIC_QUESTION_TERMS:
    _GENERIC_NORMALIZED.update(_normalized_terms(_generic_word))


class HashedEmbedder:
    """Hashed bag-of-words vectors. Lexical, not semantic."""

    dim = 256

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []

        for text in texts:
            vector = [0.0] * self.dim

            for word in content_terms(text, keep_numbers=True):
                index = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
                vector[index] += 1.0

            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            out.append([value / norm for value in vector])

        return out


class MockNLI:
    """Overlap-based NLI imitation for deterministic regression tests."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[NLIScores]:
        results = []

        for premise, claim in pairs:
            premise_terms = set(content_terms(premise, keep_numbers=True))
            claim_terms = set(content_terms(claim, keep_numbers=True))

            if not claim_terms:
                results.append(NLIScores(0.05, 0.05, 0.90))
                continue

            coverage = len(premise_terms & claim_terms) / len(claim_terms)
            difference = claim_terms - premise_terms

            premise_words = set(_WORD.findall(premise.lower()))
            claim_words = set(_WORD.findall(claim.lower()))

            premise_negated = bool(premise_words & _NEGATIONS)
            claim_negated = bool(claim_words & _NEGATIONS)

            premise_direction = direction_of(premise)
            claim_direction = direction_of(claim)

            if coverage >= 0.6 and premise_negated != claim_negated:
                results.append(NLIScores(0.02, 0.94, 0.04))

            elif (
                coverage >= 0.6
                and len(premise_direction) == 1
                and len(claim_direction) == 1
                and premise_direction != claim_direction
            ):
                results.append(NLIScores(0.02, 0.94, 0.04))

            elif (
                difference
                and coverage >= 0.5
                and all(term.isdigit() for term in difference)
            ):
                results.append(NLIScores(0.03, 0.92, 0.05))

            elif coverage >= 0.8:
                results.append(NLIScores(0.95, 0.01, 0.04))

            else:
                results.append(NLIScores(0.05, 0.05, 0.90))

        return results


_EXCERPT = re.compile(
    r"<<<EXCERPT (S\d+) page=(\d+)[^>]*>>>\n"
    r"(.*?)\n"
    r"<<<END S\d+ key=\w+>>>",
    re.DOTALL,
)


def _question_groups(question: str) -> list[set[str]]:
    """Create normalized question-concept groups with limited aliases."""

    groups = []
    seen = set()

    for term in content_terms(question, keep_numbers=True):
        normalized = _normalized_terms(term)

        if not normalized:
            continue

        # Remove generic question wording after normalization.
        if normalized & _GENERIC_NORMALIZED:
            continue

        if term in seen:
            continue

        seen.add(term)

        aliases = _QUESTION_ALIASES.get(term, {term})
        group = set()

        for alias in aliases:
            group.update(_normalized_terms(alias))

        group.update(normalized)

        groups.append(group)

    return groups


def _proper_question_terms(question: str) -> set[str]:
    """Find named entities that should occur in the retrieved evidence."""

    words = re.findall(
        r"(?<!^)(?<![.?!] )\b[A-Z][a-z]+\b",
        question,
    )

    ignored = {
        "What", "Which", "Where", "When", "Why", "How",
        "Who", "Does", "Did", "Are", "Was", "Were",
    }

    result = set()

    for word in words:
        if word not in ignored:
            result.update(_normalized_terms(word.lower()))

    return result


class ExtractiveMockLLM:
    """Select document sentences using deterministic lexical matching.

    This is an extractive testing baseline, not a real language model.
    It never invents facts or calculates unsupported answers.
    """

    def __init__(self):
        self.calls = 0

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        self.calls += 1

        excerpts = _EXCERPT.findall(user)

        question = (
            user.rsplit("Question:", 1)[-1]
            .split("\n\nA previous draft", 1)[0]
            .strip()
        )

        if not excerpts or not question:
            return json.dumps({"insufficient": True, "claims": []})

        question_groups = _question_groups(question)

        if not question_groups:
            return json.dumps({"insufficient": True, "claims": []})

        proper_terms = _proper_question_terms(question)

        # Named entities must occur somewhere in the retrieved evidence.
        all_document_terms = set()

        for _, _, excerpt_text in excerpts:
            all_document_terms.update(_normalized_terms(excerpt_text))

        if not proper_terms.issubset(all_document_terms):
            return json.dumps({"insufficient": True, "claims": []})

        # Measure coverage across all retrieved evidence.
        # This permits facts split across sentences or table rows.
        all_matched_groups = set()

        for _, _, excerpt_text in excerpts:
            excerpt_terms = _normalized_terms(excerpt_text)

            for index, group in enumerate(question_groups):
                if excerpt_terms & group:
                    all_matched_groups.add(index)

        evidence_coverage = (
            len(all_matched_groups) / len(question_groups)
        )

        # Strict gate: related words alone must not produce an answer.
        if evidence_coverage < 0.75:
            return json.dumps({"insufficient": True, "claims": []})

        candidates = []

        for source_id, _, excerpt_text in excerpts:
            for sentence in split_sentences(excerpt_text):
                sentence = sentence.strip()

                if len(sentence.split()) < 3:
                    continue

                sentence_terms = _normalized_terms(sentence)

                matched_groups = {
                    index
                    for index, group in enumerate(question_groups)
                    if sentence_terms & group
                }

                # Only keep sentences that contribute to question coverage.
                if not matched_groups:
                    continue

                candidates.append({
                    "score": len(matched_groups) / len(question_groups),
                    "text": sentence,
                    "source": source_id,
                    "matched": len(matched_groups),
                    "groups": matched_groups,
                })

        if not candidates:
            return json.dumps({"insufficient": True, "claims": []})

        # Select only evidence sentences that contribute to the answer.
        candidates.sort(
            key=lambda item: (
                -item["score"],
                -item["matched"],
                len(item["text"]),
            )
        )

        chosen = []
        seen_sentences = set()
        covered_groups = set()

        for candidate in candidates:
            sentence = candidate["text"]

            if sentence in seen_sentences:
                continue

            # Avoid adding sentences that contribute no new question concept.
            new_groups = candidate["groups"] - covered_groups

            if not new_groups:
                continue

            seen_sentences.add(sentence)
            covered_groups.update(candidate["groups"])

            chosen.append({
                "text": sentence,
                "sources": [candidate["source"]],
            })

            if covered_groups >= all_matched_groups:
                break

            if len(chosen) >= 6:
                break

        if not chosen:
            return json.dumps({"insufficient": True, "claims": []})

        return json.dumps({
            "insufficient": False,
            "claims": chosen,
        })


__all__ = [
    "HashedEmbedder",
    "MockNLI",
    "ExtractiveMockLLM",
    "STOPWORDS",
    "canon",
]
