import { useCallback, useEffect, useRef, useState } from "react";
import { api, errorMessage } from "../api";
import type { Doc } from "../types";

const MAX_MB = 25;
const STATUS: Record<Doc["status"], { label: string; cls: string }> = {
  pending: { label: "Queued", cls: "bg-warnbg text-warn" },
  processing: { label: "Indexing…", cls: "bg-warnbg text-warn" },
  ready: { label: "Ready", cls: "bg-okbg text-ok" },
  failed: { label: "Failed", cls: "bg-badbg text-bad" },
};

export default function DocumentsPage({ docs, refresh, onAsk }: { docs: Doc[] | null; refresh: () => Promise<void>; onAsk: (id: string) => void }) {
  const [progress, setProgress] = useState<number | null>(null);
  const [message, setMessage] = useState<{ kind: "error" | "info"; text: string } | null>(null);
  const [drag, setDrag] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  // Poll while anything is still being indexed.
  useEffect(() => {
    if (!docs?.some((d) => d.status === "pending" || d.status === "processing")) return;
    const id = setInterval(() => void refresh(), 2000);
    return () => clearInterval(id);
  }, [docs, refresh]);

  const upload = useCallback(async (file: File | undefined) => {
    if (!file) return;
    setMessage(null);
    if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) return setMessage({ kind: "error", text: "Choose a PDF file." });
    if (file.size > MAX_MB * 1024 * 1024) return setMessage({ kind: "error", text: `That file is larger than ${MAX_MB} MB.` });
    setProgress(0);
    try {
      const d = await api.upload(file, setProgress);
      setMessage({ kind: "info", text: d.duplicate ? "You already uploaded this file; the existing index is reused." : d.reprocessing ? "This file failed earlier; indexing has been started again." : "Uploaded. Indexing has started." });
      await refresh();
    } catch (err) { setMessage({ kind: "error", text: errorMessage(err) }); } finally { setProgress(null); if (input.current) input.current.value = ""; }
  }, [refresh]);

  async function remove(d: Doc) {
    if (!window.confirm(`Delete "${d.display_name}"? The file, its search index and its text are removed permanently.`)) return;
    try { await api.remove(d.id); await refresh(); } catch (err) { setMessage({ kind: "error", text: errorMessage(err) }); }
  }

  return (
    <div className="space-y-8">
      <section>
        <div
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); void upload(e.dataTransfer.files[0]); }}
          className={`rounded-lg border-2 border-dashed p-8 text-center ${drag ? "border-brand bg-white" : "border-line bg-panel"}`}
        >
          <p className="font-semibold">Drop a PDF here</p>
          <p className="mt-1 text-sm text-muted">or</p>
          <input ref={input} id="pdf" type="file" accept="application/pdf,.pdf" className="sr-only" onChange={(e) => void upload(e.target.files?.[0])} />
          <label htmlFor="pdf" className="btn-primary mt-2 cursor-pointer">Choose a PDF</label>
          <p className="mt-3 text-xs text-muted">Up to {MAX_MB} MB. Text-based PDFs work best; scanned PDFs need OCR to be enabled on the server.</p>
          {progress !== null && (
            <div className="mx-auto mt-4 max-w-sm" role="progressbar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100} aria-label="Upload progress">
              <div className="h-2 overflow-hidden rounded-full bg-paper"><div className="h-full bg-brand transition-all" style={{ width: `${progress}%` }} /></div>
              <p className="mt-1 text-xs text-muted">{progress < 100 ? `Uploading ${progress}%` : "Upload complete, validating…"}</p>
            </div>
          )}
        </div>
        {message && <p role="alert" className={`mt-3 rounded-md p-3 text-sm ${message.kind === "error" ? "bg-badbg text-bad" : "bg-okbg text-ok"}`}>{message.text}</p>}
      </section>

      <section>
        <h2 className="mb-3 text-lg font-semibold">Your documents</h2>
        {docs === null ? <p className="text-muted">Loading…</p> : docs.length === 0 ? (
          <p className="rounded-lg border border-line bg-panel p-6 text-muted">No documents yet. Upload a PDF to start asking questions.</p>
        ) : (
          <ul className="divide-y divide-line rounded-lg border border-line bg-panel">
            {docs.map((d) => (
              <li key={d.id} className="flex flex-wrap items-center gap-3 p-4">
                <div className="min-w-0 flex-1">
                  <p className="truncate font-semibold">{d.display_name}</p>
                  <p className="text-sm text-muted">
                    {(d.size_bytes / 1024 / 1024).toFixed(2)} MB{d.status === "ready" && ` · ${d.pages} pages · ${d.chunk_count} passages`}
                  </p>
                  {d.error && <p role="alert" className="mt-1 text-sm text-bad">{d.error}</p>}
                  {d.status === "ready" && d.ocr_pages > 0 && <p className="mt-1 text-xs font-semibold text-warn">OCR was used on {d.ocr_pages} page{d.ocr_pages > 1 ? "s" : ""}: check key figures against the original.</p>}
                  {d.status === "ready" && d.warnings.length > 0 && (
                    <details className="mt-1 text-sm text-muted">
                      <summary className="cursor-pointer">{d.warnings.length} note{d.warnings.length > 1 ? "s" : ""} about how this PDF was read</summary>
                      <ul className="mt-1 list-disc pl-5">{d.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
                    </details>
                  )}
                </div>
                <span className={`rounded-full px-3 py-1 text-xs font-semibold ${STATUS[d.status].cls}`}>{STATUS[d.status].label}</span>
                <button className="btn-quiet" disabled={d.status !== "ready"} onClick={() => onAsk(d.id)}>Ask questions</button>
                <button className="btn-quiet text-bad" onClick={() => void remove(d)} aria-label={`Delete ${d.display_name}`}>Delete</button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
