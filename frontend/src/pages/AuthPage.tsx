import { FormEvent, useState } from "react";
import { api, errorMessage, setToken } from "../api";

export default function AuthPage({ onSignedIn }: { onSignedIn: (email: string) => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true); setError(null);
    try {
      const res = mode === "login" ? await api.login(email, password) : await api.register(email, password);
      setToken(res.token);
      onSignedIn(res.email);
    } catch (err) { setError(errorMessage(err)); } finally { setBusy(false); }
  }

  return (
    <main className="mx-auto grid min-h-screen max-w-5xl items-center gap-12 px-6 md:grid-cols-2">
      <div>
        <h1 className="text-4xl font-bold tracking-tight">VerityGate</h1>
        <p className="mt-4 max-w-md font-serif text-xl leading-relaxed">Ask questions about your PDFs. Every sentence of the answer is checked against the pages it came from.</p>
        <p className="mt-4 max-w-md text-sm text-muted">Claims the document does not support are marked, never passed off as verified.</p>
      </div>
      <form onSubmit={submit} className="space-y-4 rounded-lg border border-line bg-panel p-6">
        <h2 className="text-lg font-semibold">{mode === "login" ? "Sign in" : "Create an account"}</h2>
        <label className="block text-sm font-medium">Email
          <input className="input mt-1" type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
        </label>
        <label className="block text-sm font-medium">Password
          <input className="input mt-1" type="password" minLength={10} autoComplete={mode === "login" ? "current-password" : "new-password"} required value={password} onChange={(e) => setPassword(e.target.value)} />
          {mode === "register" && <span className="mt-1 block text-xs font-normal text-muted">At least 10 characters.</span>}
        </label>
        {error && <p role="alert" className="rounded-md bg-badbg p-3 text-sm text-bad">{error}</p>}
        <button className="btn-primary w-full" disabled={busy}>{busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}</button>
        <button type="button" className="w-full text-sm text-brand hover:underline" onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(null); }}>
          {mode === "login" ? "Need an account? Create one" : "Have an account? Sign in"}
        </button>
      </form>
    </main>
  );
}
