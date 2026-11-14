import { useEffect, useState } from "react";
import { api, errorMessage } from "../api";
import type { AuditRow } from "../types";

export default function HistoryPage() {
  const [rows, setRows] = useState<AuditRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api.audit().then(setRows).catch((e) => setError(errorMessage(e))); }, []);
  if (error) return <p role="alert" className="rounded-md bg-badbg p-4 text-sm text-bad">{error}</p>;
  if (!rows) return <p className="text-muted">Loading…</p>;
  if (rows.length === 0) return <p className="rounded-lg border border-line bg-panel p-6 text-muted">Nothing recorded yet.</p>;
  return (
    <div className="overflow-x-auto rounded-lg border border-line bg-panel">
      <table className="w-full text-left text-sm">
        <caption className="sr-only">Audit history</caption>
        <thead className="border-b border-line text-muted"><tr><th className="p-3">When</th><th className="p-3">Event</th><th className="p-3">Details</th></tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-line last:border-0 align-top">
              <td className="whitespace-nowrap p-3">{new Date(r.ts * 1000).toLocaleString()}</td>
              <td className="p-3 font-semibold">{r.event}</td>
              <td className="p-3 text-muted">{Object.entries(r.detail).map(([k, v]) => `${k}: ${String(v)}`).join(" · ") || "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="border-t border-line p-3 text-xs text-muted">Questions and document text are not stored in the audit log unless the administrator enables it.</p>
    </div>
  );
}
