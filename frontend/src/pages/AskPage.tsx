import { FormEvent, useEffect, useState } from "react";
import { api, errorMessage } from "../api";
import AnswerView from "../components/AnswerView";
import type { AskResult, Doc } from "../types";

export default function AskPage({ docs, selected, onSelect }: { docs: Doc[] | null; selected: string | null; onSelect: (id: string) => void }) {
  const ready = (docs ?? []).filter((d) => d.status === "ready");
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AskResult | null>(null);

  useEffect(() => { if (!selected && ready[0]) onSelect(ready[0].id); }, [selected, ready, onSelect]);
  useEffect(() => { setResult(null); setError(null); }, [selected]);

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!selected) return;
    setBusy(true); setError(null); setResult(null);
    try { setResult(await api.ask(selected, question)); } catch (err) { setError(errorMessage(err)); } finally { setBusy(false); }
  }

  if (docs && ready.length === 0) return <p className="rounded-lg border border-line bg-panel p-6 text-muted">Upload a PDF and wait until it shows “Ready” to ask questions.</p>;

  return (
    <div className="space-y-6">
      <form onSubmit={submit} className="space-y-4 rounded-lg border border-line bg-panel p-5">
        <label className="block text-sm font-medium">Document
          <select className="input mt-1" value={selected ?? ""} onChange={(e) => onSelect(e.target.value)}>
            {ready.map((d) => <option key={d.id} value={d.id}>{d.display_name}</option>)}
          </select>
        </label>
        <label className="block text-sm font-medium">Your question
          <textarea className="input mt-1 min-h-24" required minLength={3} maxLength={1000} value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="For example: What was revenue in 2023?" />
        </label>
        <button className="btn-primary" disabled={busy || !selected}>{busy ? "Retrieving and checking claims…" : "Ask and verify"}</button>
        {busy && <p className="text-sm text-muted">The first question after a restart can take longer while models load.</p>}
      </form>
      {error && <p role="alert" className="rounded-md bg-badbg p-4 text-sm text-bad">{error}</p>}
      {result && <AnswerView result={result} />}
    </div>
  );
}
