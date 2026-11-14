
"""LangGraph workflow: retrieve -> generate -> verify -> (bounded, purposeful retry) -> finalize/abstain.

Retry policy (bounded by `max_retries`, hard cap 2):
  * a retry is only made when it can plausibly help: incomplete or unsupported claims, a model that said
    "insufficient" while the question's distinctive terms ARE present in the retrieved text, or a saturated
    retrieval that can be widened;
  * no retry when the document clearly lacks the information (the question's names/terms appear nowhere),
    or when the document contradicts itself (another attempt cannot fix that);
  * the best attempt so far is always kept - a worse later answer never replaces a better earlier one.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from .claims import ParsedAnswer, parse_answer
from .evidence import question_terms_found
from .generation import SYSTEM_PROMPT, build_user_prompt
from .llm import LLMClient
from .retrieval import Passage, Retriever
from .verifier import CONTRADICTED, NOT_ESTABLISHED, SUPPORTED, ClaimResult, ClaimVerifier

log = logging.getLogger(__name__)
HARD_RETRY_CAP = 2
MAX_TOP_K = 30

ABSTAIN_MESSAGES = {
    "no_relevant_passages": "No passage in the document is closely related to the question.",
    "model_insufficient": "The document does not contain enough information to answer this question.",
    "no_claims": "The model did not produce a checkable answer.",
    "unverifiable": "None of the answer's claims could be verified against the document, so nothing is shown as verified.",
}


class State(TypedDict, total=False):
    owner_id: str
    doc_id: str
    question: str
    attempt: int
    top_k: int
    passages: list[Passage]
    term_df: dict[str, float]
    id_map: dict[str, Passage]
    feedback: list[str]
    abstain: str | None
    parsed: ParsedAnswer | None
    results: list[ClaimResult]
    attempts: list[dict]
    best: dict | None
    llm_calls: int
    retrieval_ms: float
    retry_reason: str | None


@dataclass
class WorkflowConfig:
    top_k: int = 6
    max_retries: int = 2
    max_tokens: int = 700


def attempt_key(results: list[ClaimResult]) -> tuple[int, int, int, int]:
    """Higher is better: fully verified > more supported claims > fewer contradicted > fewer unestablished."""
    supported = sum(r.status == SUPPORTED for r in results)
    contradicted = sum(r.status == CONTRADICTED for r in results)
    unestablished = sum(r.status == NOT_ESTABLISHED for r in results)
    return (int(bool(results) and supported == len(results)), supported, -contradicted, -unestablished)


def build_graph(retriever: Retriever, llm: LLMClient, verifier: ClaimVerifier, cfg: WorkflowConfig):
    max_retries = max(0, min(cfg.max_retries, HARD_RETRY_CAP))

    def widenable(state: State) -> bool:
        """True when retrieval returned as many passages as asked for, so asking for more could surface new text."""
        return (
            state["attempt"] < max_retries
            and len(state.get("passages", [])) >= state["top_k"]
            and state["top_k"] < MAX_TOP_K
        )

    def retrieve(state: State) -> State:
        t0 = time.perf_counter()

        passages = retriever.retrieve(
            state["owner_id"],
            state["doc_id"],
            state["question"],
            state["top_k"],
        )

        step_ms = (time.perf_counter() - t0) * 1000
        ms = state.get("retrieval_ms", 0.0) + step_ms

        evidence_sufficient = bool(
            passages and retriever.evidence_is_sufficient(
                passages, state["question"]
            )
        )

        update: State = {
            "passages": passages,
            "retrieval_ms": ms,
            "abstain": None,
            "term_df": retriever.term_df(
                state["owner_id"], state["doc_id"]
            ),
        }

        if not evidence_sufficient:
            update["abstain"] = "no_relevant_passages"

        log.info(
            "workflow_retrieve",
            extra={
                "ctx": {
                    "attempt": state["attempt"],
                    "top_k": state["top_k"],
                    "passage_count": len(passages),
                    "retrieval_ms": round(step_ms, 2),
                    "evidence_sufficient": evidence_sufficient,
                    "abstain_reason": update["abstain"],
                }
            },
        )

        return update

    def generate(state: State) -> State:
        t0 = time.perf_counter()

        prompt, id_map = build_user_prompt(
            state["question"],
            state["passages"],
            state.get("feedback") or None,
        )

        raw = llm.complete(SYSTEM_PROMPT, prompt, cfg.max_tokens)
        parsed = parse_answer(raw)

        generation_ms = (time.perf_counter() - t0) * 1000
        incomplete_count = sum(
            1 for claim in parsed.claims if claim.problem
        )

        log.info(
            "workflow_generate",
            extra={
                "ctx": {
                    "attempt": state["attempt"],
                    "generation_ms": round(generation_ms, 2),
                    "output_chars": len(raw),
                    "answer_format": parsed.fmt,
                    "claim_count": len(parsed.claims),
                    "incomplete_claim_count": incomplete_count,
                    "model_insufficient": parsed.insufficient,
                }
            },
        )

        update: State = {
            "id_map": id_map,
            "parsed": parsed,
            "llm_calls": state.get("llm_calls", 0) + 1,
            "retry_reason": None,
        }

        if parsed.insufficient:
            if (
                state["attempt"] == 0
                and widenable(state)
                and question_terms_found(
                    state["question"],
                    state["passages"],
                    state.get("term_df"),
                )
            ):
                update["retry_reason"] = "insufficient_but_terms_present"
            else:
                update["abstain"] = "model_insufficient"

        elif not parsed.claims:
            update["abstain"] = (
                "no_claims" if state["attempt"] >= max_retries else None
            )
            update["retry_reason"] = (
                None if state["attempt"] >= max_retries else "no_claims"
            )

        return update

    def verify(state: State) -> State:
        parsed: ParsedAnswer = state["parsed"]

        results = verifier.verify_claims(
            parsed.claims,
            state["id_map"],
            state["passages"],
            state.get("term_df"),
        )

        record = {
            "attempt": state["attempt"],
            "top_k": state["top_k"],
            "passages": len(state["passages"]),
            "claims": len(results),
            "supported": sum(r.status == SUPPORTED for r in results),
            "contradicted": sum(r.status == CONTRADICTED for r in results),
            "not_established": sum(r.status == NOT_ESTABLISHED for r in results),
            "conflicts": sum(r.conflict for r in results),
            "answer_format": parsed.fmt,
        }

        log.info(
            "workflow_verify",
            extra={"ctx": record},
        )

        best = state.get("best")

        if results and (
            best is None or attempt_key(results) > attempt_key(best["results"])
        ):
            best = {
                "results": results,
                "passages": state["passages"],
                "attempt": state["attempt"],
            }

        return {
            "results": results,
            "attempts": [*state.get("attempts", []), record],
            "best": best,
        }

    def prepare_retry(state: State) -> State:
        failed = [
            f"{r.text} (problem: {r.reason[:160]})"
            for r in state.get("results", [])
            if r.status != SUPPORTED
        ]

        top_k = (
            min(state["top_k"] * 2, MAX_TOP_K)
            if widenable(state)
            or state.get("retry_reason") == "insufficient_but_terms_present"
            else state["top_k"]
        )

        log.info(
            "workflow_retry",
            extra={
                "ctx": {
                    "current_attempt": state["attempt"],
                    "next_attempt": state["attempt"] + 1,
                    "retry_reason": state.get("retry_reason"),
                    "failed_claim_count": len(failed),
                    "old_top_k": state["top_k"],
                    "new_top_k": top_k,
                }
            },
        )

        return {
            "attempt": state["attempt"] + 1,
            "top_k": top_k,
            "feedback": failed,
            "results": [],
        }

    def after_retrieve(state: State) -> str:
        return "finalize" if state.get("abstain") else "generate"

    def after_generate(state: State) -> str:
        if state.get("abstain"):
            return "finalize"
        if state.get("retry_reason"):
            return "prepare_retry"
        return "verify"

    def after_verify(state: State) -> str:
        results = state["results"]

        if (
            results and all(r.status == SUPPORTED for r in results)
        ) or state["attempt"] >= max_retries:
            return "finalize"

        failing = [r for r in results if r.status != SUPPORTED]

        if failing and all(r.conflict for r in failing):
            return "finalize"

        no_support_anywhere = all(
            r.reason_code == "no_relevant_evidence" for r in failing
        )

        if (
            no_support_anywhere
            and not any(r.status == SUPPORTED for r in results)
            and not widenable(state)
        ):
            return "finalize"

        return "prepare_retry"

    g = StateGraph(State)

    for name, fn in [
        ("retrieve", retrieve),
        ("generate", generate),
        ("verify", verify),
        ("prepare_retry", prepare_retry),
        ("finalize", lambda s: {}),
    ]:
        g.add_node(name, fn)

    g.set_entry_point("retrieve")

    g.add_conditional_edges(
        "retrieve",
        after_retrieve,
        {"generate": "generate", "finalize": "finalize"},
    )

    g.add_conditional_edges(
        "generate",
        after_generate,
        {
            "verify": "verify",
            "prepare_retry": "prepare_retry",
            "finalize": "finalize",
        },
    )

    g.add_conditional_edges(
        "verify",
        after_verify,
        {
            "prepare_retry": "prepare_retry",
            "finalize": "finalize",
        },
    )

    g.add_edge("prepare_retry", "retrieve")
    g.add_edge("finalize", END)

    return g.compile()


def _result_dict(r: ClaimResult) -> dict[str, Any]:
    return asdict(r)


def _pages_label(r: ClaimResult) -> str:
    pages = sorted({e.page for e in r.evidence if e.role == "supporting"})
    return ("p." + ", ".join(map(str, pages[:3]))) if pages else ""


def run_question(
    graph,
    owner_id: str,
    doc_id: str,
    question: str,
    cfg: WorkflowConfig,
) -> dict[str, Any]:
    """Run the workflow and shape the API response. Unverified text is never put in `verified_answer`."""
    t0 = time.perf_counter()

    state = graph.invoke(
        {
            "owner_id": owner_id,
            "doc_id": doc_id,
            "question": question,
            "attempt": 0,
            "top_k": cfg.top_k,
            "attempts": [],
            "best": None,
            "llm_calls": 0,
            "retrieval_ms": 0.0,
        },
        {"recursion_limit": 60},
    )

    elapsed = round(time.perf_counter() - t0, 3)
    best = state.get("best")
    attempts = state.get("attempts", [])

    base = {
        "question": question,
        "attempts": attempts,
        "llm_calls": state.get("llm_calls", 0),
        "latency_seconds": elapsed,
        "retrieval_ms": round(state.get("retrieval_ms", 0.0), 1),
        "retries_used": state.get("attempt", 0),
        "conflicts": [],
    }

    if not best:
        code = state.get("abstain") or "no_claims"

        log.info(
            "workflow_final",
            extra={
                "ctx": {
                    "status": "ABSTAINED",
                    "abstain_reason": code,
                    "attempt_count": len(attempts),
                    "llm_calls": state.get("llm_calls", 0),
                    "latency_seconds": elapsed,
                }
            },
        )

        return {
            **base,
            "status": "ABSTAINED",
            "abstain_reason": code,
            "verified_answer": None,
            "claims": [],
            "unverified_claims": [],
            "message": (
                f"VerityGate could not verify an answer. "
                f"{ABSTAIN_MESSAGES.get(code, '')}"
            ).strip(),
            "warnings": [],
            "sources": [],
        }

    results: list[ClaimResult] = best["results"]
    supported = [r for r in results if r.status == SUPPORTED]
    others = [r for r in results if r.status != SUPPORTED]

    conflicts = [
        {
            "claim": r.text,
            "reason": r.reason,
            "evidence": [
                asdict(e)
                for e in r.evidence
                if e.role in ("supporting", "contradicting")
            ],
        }
        for r in results
        if r.conflict
    ]

    if not supported:
        status, message, code = (
            "ABSTAINED",
            ABSTAIN_MESSAGES["unverifiable"],
            "unverifiable",
        )

        if conflicts:
            message = (
                "The document contains conflicting statements about this, "
                "so no answer is shown as verified. See the conflict details."
            )
            code = "document_conflict"

    elif others:
        status, code = "PARTIALLY_VERIFIED", None
        message = (
            f"{len(supported)} of {len(results)} claims are supported by "
            "the document. The rest are listed separately and are NOT verified."
        )

    else:
        status, code = "VERIFIED", None
        message = (
            "Every claim is supported by a passage in the document "
            "(according to the NLI model and the numeric checks)."
        )

    verified_text = (
        " ".join(
            f"{r.text} [{_pages_label(r)}]" if _pages_label(r) else r.text
            for r in supported
        )
        or None
    )

    pages = sorted(
        {
            e.page
            for r in supported
            for e in r.evidence
            if e.role == "supporting"
        }
    )

    log.info(
        "workflow_final",
        extra={
            "ctx": {
                "status": status,
                "abstain_reason": code,
                "best_attempt": best["attempt"],
                "supported_claims": len(supported),
                "unverified_claims": len(others),
                "attempt_count": len(attempts),
                "llm_calls": state.get("llm_calls", 0),
                "latency_seconds": elapsed,
            }
        },
    )

    return {
        **base,
        "status": status,
        "abstain_reason": code,
        "message": message,
        "verified_answer": verified_text,
        "claims": [_result_dict(r) for r in results],
        "unverified_claims": [r.text for r in others],
        "conflicts": conflicts,
        "warnings": list(
            dict.fromkeys(w for r in results for w in r.warnings)
        ),
        "sources": pages,
        "best_attempt": best["attempt"],
    }
