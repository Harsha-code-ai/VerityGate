import { useState } from "react";
import type { AskResult, Claim, ClaimStatus, Evidence, EvidenceRole } from "../types";

const META: Record<ClaimStatus, { label: string; icon: string; text: string; bg: string; bar: string }> = {
  SUPPORTED: { label: "SUPPORTED", icon: "✓", text: "text-ok", bg: "bg-okbg", bar: "bg-ok" },
  CONTRADICTED: { label: "CONTRADICTED", icon: "✕", text: "text-bad", bg: "bg-badbg", bar: "bg-bad" },
  NOT_ESTABLISHED: { label: "NOT_ESTABLISHED", icon: "–", text: "text-warn", bg: "bg-warnbg", bar: "bg-warn" },
};
const ROLE: Record<EvidenceRole, { title: string; border: string }> = {
  supporting: { title: "Supporting evidence", border: "border-ok" },
  contradicting: { title: "Contradicting evidence", border: "border-bad" },
  related: { title: "Related passages that do not settle the claim", border: "border-line" },
};
const HEADLINE = { VERIFIED: "VERIFIED", PARTIALLY_VERIFIED: "PARTIALLY_VERIFIED", ABSTAINED: "ABSTAINED" } as const;

function EvidenceBlock({ e }: { e: Evidence }) {
  return (
    <li className={`border-l-4 ${ROLE[e.role].border} bg-paper px-4 py-3`}>
      <p className="mb-1 text-xs font-semibold text-muted">
        Page {e.page} · entails {(e.entailment * 100).toFixed(0)}% · contradicts {(e.contradiction * 100).toFixed(0)}%{e.cited_by_model ? " · cited by the answer" : ""}
      </p>
      <p className="whitespace-pre-line font-serif text-[0.95rem] leading-relaxed">{e.text}</p>
      {e.note && <p className="mt-1 text-xs text-warn">{e.note}</p>}
      {e.context && <details className="mt-1 text-xs text-muted"><summary className="cursor-pointer">Surrounding text</summary><p className="mt-1 whitespace-pre-line">{e.context}</p></details>}
    </li>
  );
}

function ClaimRow({ claim }: { claim: Claim }) {
  const [open, setOpen] = useState(claim.conflict || claim.status === "CONTRADICTED");
  const m = META[claim.status];
  const groups = (["supporting", "contradicting", "related"] as EvidenceRole[]).map((r) => [r, claim.evidence.filter((e) => e.role === r)] as const).filter(([, l]) => l.length > 0);
  const supportPages = [...new Set(claim.evidence.filter((e) => e.role === "supporting").map((e) => e.page))];
  return (
    <li className="grid grid-cols-1 gap-3 border-b border-line py-4 last:border-b-0 sm:grid-cols-[9.5rem_1fr] sm:gap-4">
      <div className="flex h-fit flex-col gap-1">
        <div className={`flex items-center gap-2 rounded-md px-2 py-1 text-xs font-bold ${m.bg} ${m.text}`}>
          <span aria-hidden className="text-base leading-none">{m.icon}</span>{m.label}
        </div>
        {claim.conflict && <span className="rounded-md bg-badbg px-2 py-1 text-xs font-bold text-bad">DOCUMENT CONFLICT</span>}
      </div>
      <div>
        <p className="font-serif text-[1.05rem] leading-relaxed">{claim.text}</p>
        <p className="mt-1 text-sm text-muted">{claim.reason}</p>
        {claim.status === "SUPPORTED" && supportPages.length > 0 && (
          <p className="mt-1 text-sm">
            Source: <span className="rounded bg-paper px-1.5 py-0.5 font-semibold">page{supportPages.length > 1 ? "s" : ""} {supportPages.join(", ")}</span>
            {!claim.citation_valid && <span className="ml-2 text-warn">The answer cited a different excerpt; this page is the verified source.</span>}
          </p>
        )}
        {claim.status === "NOT_ESTABLISHED" && !claim.conflict && claim.evidence.length === 0 && (
          <p className="mt-1 text-sm font-semibold text-warn">Missing evidence: no passage in the document states this ({claim.considered} sentences were examined).</p>
        )}
        {claim.warnings.filter((w) => !w.includes("did not cite") && !w.includes("no citation")).map((w) => <p key={w} className="mt-1 text-sm text-warn">{w}</p>)}
        {claim.evidence.length > 0 && (
          <button className="mt-2 text-sm font-semibold text-brand underline-offset-2 hover:underline" aria-expanded={open} onClick={() => setOpen(!open)}>
            {open ? "Hide evidence" : `Show evidence (${claim.evidence.length})`}
          </button>
        )}
        {open && groups.map(([role, list]) => (
          <div key={role} className="mt-3">
            <h4 className="mb-1 text-xs font-bold uppercase tracking-wide text-muted">{ROLE[role].title}</h4>
            <ul className="space-y-2">{list.map((e, i) => <EvidenceBlock key={`${e.chunk_id}-${i}`} e={e} />)}</ul>
          </div>
        ))}
      </div>
    </li>
  );
}

export default function AnswerView({ result }: { result: AskResult }) {
  const counts = { SUPPORTED: 0, CONTRADICTED: 0, NOT_ESTABLISHED: 0 } as Record<ClaimStatus, number>;
  result.claims.forEach((c) => counts[c.status]++);
  const total = result.claims.length;
  const tone = result.status === "VERIFIED" ? "text-ok" : result.status === "ABSTAINED" ? "text-bad" : "text-warn";

  return (
    <section aria-live="polite" className="rounded-lg border border-line bg-panel">
      <header className="border-b border-line p-5">
        <h2 className={`text-xl font-bold ${tone}`}>{HEADLINE[result.status]}</h2>
        <p className="mt-1 text-sm">{result.message}</p>
        {result.conflicts.length > 0 && (
          <p role="alert" className="mt-3 rounded-md bg-badbg p-3 text-sm text-bad">
            The document contradicts itself on {result.conflicts.length} point{result.conflicts.length > 1 ? "s" : ""}. Both versions are shown below with their pages; neither is presented as verified.
          </p>
        )}
        {total > 0 && (
          <div className="mt-4">
            <div className="flex h-2.5 overflow-hidden rounded-full bg-paper" role="img" aria-label={`${counts.SUPPORTED} supported, ${counts.CONTRADICTED} contradicted, ${counts.NOT_ESTABLISHED} not established`}>
              {(Object.keys(META) as ClaimStatus[]).map((k) => counts[k] > 0 && <div key={k} className={META[k].bar} style={{ width: `${(counts[k] / total) * 100}%` }} />)}
            </div>
            <p className="mt-2 text-sm text-muted">{counts.SUPPORTED} SUPPORTED · {counts.CONTRADICTED} CONTRADICTED · {counts.NOT_ESTABLISHED} NOT_ESTABLISHED</p>
          </div>
        )}
      </header>

      {result.verified_answer && (
        <div className="border-b border-line p-5">
          <h3 className="mb-2 text-sm font-semibold text-muted">Verified answer (supported claims only; page numbers are the verified sources)</h3>
          <p className="font-serif text-lg leading-relaxed">{result.verified_answer}</p>
        </div>
      )}

      {total > 0 && (
        <div className="p-5">
          <h3 className="mb-1 text-sm font-semibold text-muted">Claim-by-claim check</h3>
          <ul>{result.claims.map((c, i) => <ClaimRow key={i} claim={c} />)}</ul>
        </div>
      )}

      <footer className="border-t border-line bg-paper/60 p-5 text-sm text-muted">
        <details className="mb-2">
          <summary className="cursor-pointer font-semibold text-ink">
            Retry count: {result.retries_used} of 2{result.retries_used === 0 ? " (answered on the first attempt)" : ""}
          </summary>
          <ul className="mt-2 list-disc space-y-1 pl-5">
            {result.attempts.map((a) => (
              <li key={a.attempt}>
                Attempt {a.attempt + 1}: looked at {a.passages} passages (asked for {a.top_k}); {a.supported} of {a.claims} claims supported, {a.contradicted} contradicted, {a.not_established} not established
                {a.conflicts > 0 ? `, ${a.conflicts} document conflict(s)` : ""}.
                {result.best_attempt === a.attempt && result.attempts.length > 1 ? " ← best attempt, the one shown." : ""}
              </li>
            ))}
            {result.attempts.length === 0 && <li>No answer reached the checking stage{result.abstain_reason ? ` (${result.abstain_reason.replace(/_/g, " ")})` : ""}.</li>}
          </ul>
        </details>
        <p>
          {result.llm_calls} model call{result.llm_calls === 1 ? "" : "s"} · {result.latency_seconds.toFixed(1)} s total · retrieval {result.retrieval_ms.toFixed(0)} ms.
          SUPPORTED means a sentence of your PDF entails the claim according to a language-inference model plus number/date checks. It is evidence from the document, not proof that the document is right.
        </p>
      </footer>
    </section>
  );
}
