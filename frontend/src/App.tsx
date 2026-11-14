import { useCallback, useEffect, useState } from "react";
import { api, errorMessage, getToken, setToken, setUnauthorizedHandler } from "./api";
import StatusBadge from "./components/StatusBadge";
import AskPage from "./pages/AskPage";
import AuthPage from "./pages/AuthPage";
import DocumentsPage from "./pages/DocumentsPage";
import HistoryPage from "./pages/HistoryPage";
import type { Doc } from "./types";

type Tab = "documents" | "ask" | "history";
const TABS: [Tab, string][] = [["documents", "Documents"], ["ask", "Ask"], ["history", "History"]];

export default function App() {
  const [signedIn, setSignedIn] = useState(Boolean(getToken()));
  const [tab, setTab] = useState<Tab>("documents");
  const [docs, setDocs] = useState<Doc[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const clearSession = useCallback(() => { setToken(null); setSignedIn(false); setDocs(null); setSelected(null); }, []);
  useEffect(() => setUnauthorizedHandler(clearSession), [clearSession]);
  // Explicit sign-out also invalidates the token on the server (it stops working everywhere).
  const signOut = useCallback(() => { void api.logout().catch(() => undefined).finally(clearSession); }, [clearSession]);

  const refresh = useCallback(async () => {
    try { setDocs(await api.documents()); setLoadError(null); } catch (e) { setLoadError(errorMessage(e)); }
  }, []);
  useEffect(() => { if (signedIn) void refresh(); }, [signedIn, refresh]);

  if (!signedIn) return <AuthPage onSignedIn={() => setSignedIn(true)} />;

  return (
    <div className="min-h-screen">
      <header className="border-b border-line bg-panel">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center gap-4 px-6 py-3">
          <span className="text-lg font-bold tracking-tight">VerityGate</span>
          <nav aria-label="Main" className="flex gap-1">
            {TABS.map(([id, label]) => (
              <button key={id} onClick={() => setTab(id)} aria-current={tab === id ? "page" : undefined}
                className={`rounded-md px-3 py-1.5 text-sm font-semibold ${tab === id ? "bg-brand text-white" : "hover:bg-paper"}`}>{label}</button>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-3">
            <StatusBadge />
            <button className="btn-quiet" onClick={signOut}>Sign out</button>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-5xl px-6 py-8">
        {loadError && <p role="alert" className="mb-4 rounded-md bg-badbg p-4 text-sm text-bad">{loadError}</p>}
        {tab === "documents" && <DocumentsPage docs={docs} refresh={refresh} onAsk={(id) => { setSelected(id); setTab("ask"); }} />}
        {tab === "ask" && <AskPage docs={docs} selected={selected} onSelect={setSelected} />}
        {tab === "history" && <HistoryPage />}
      </main>
    </div>
  );
}
