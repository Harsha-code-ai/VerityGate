import { useEffect, useState } from "react";
import { api } from "../api";
import type { Ready } from "../types";

export default function StatusBadge() {
  const [r, setR] = useState<Ready | null>(null);
  const [down, setDown] = useState(false);
  useEffect(() => {
    let alive = true;
    const tick = () => api.ready().then((x) => alive && (setR(x), setDown(false))).catch(() => alive && setDown(true));
    tick();
    const id = setInterval(tick, 15000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const items: [string, boolean | undefined][] = [
    ["Database", r?.checks.database], ["Qdrant", r?.checks.qdrant], ["Language model", r?.llm_configured],
  ];
  const allGood = !down && r?.ready && r.llm_configured;
  return (
    <details className="relative text-sm">
      <summary className="flex cursor-pointer list-none items-center gap-2 rounded-md border border-line bg-panel px-3 py-1.5">
        <span aria-hidden className={`h-2.5 w-2.5 rounded-full ${down ? "bg-bad" : allGood ? "bg-ok" : "bg-warn"}`} />
        {down ? "Server unreachable" : allGood ? "All systems ready" : "Setup needed"}
      </summary>
      <div className="absolute right-0 z-10 mt-2 w-80 rounded-md border border-line bg-panel p-3 shadow-lg">
        {down ? <p>Cannot reach the backend. Run <code>docker compose ps</code> and check the logs.</p> : (
          <ul className="space-y-1">
            {items.map(([name, ok]) => (
              <li key={name} className="flex justify-between"><span>{name}</span><span className={ok ? "text-ok" : "text-bad"}>{ok ? "ready" : "not ready"}</span></li>
            ))}
          </ul>
        )}
        {r?.models && (
          <div className="mt-2 border-t border-line pt-2 text-xs text-muted">
            <p className="mb-1 font-semibold text-ink">Models in use</p>
            <ul className="space-y-1">
              <li><b>Answers:</b> {r.models.llm.provider ?? "not set"} {r.models.llm.model ?? ""} – {r.models.llm.runs}</li>
              <li><b>Search:</b> {r.models.embedding.model} – {r.models.embedding.runs} ({r.models.embedding.status})</li>
              <li><b>Checking:</b> {r.models.nli.model} – {r.models.nli.runs} ({r.models.nli.status})</li>
              <li><b>OCR:</b> {r.ocr.enabled ? r.ocr.engine : "off"}</li>
            </ul>
            <p className="mt-1">Only the answer model may see document excerpts; facts come from your PDF, never from the model.</p>
          </div>
        )}
        {r && r.llm_problems.length > 0 && (
          <p className="mt-2 border-t border-line pt-2 text-warn">{r.llm_problems.join(" ")} Edit <code>.env</code>, then run <code>docker compose up -d</code>.</p>
        )}
      </div>
    </details>
  );
}
