import axios, { AxiosError } from "axios";
import type { AskResult, AuditRow, Doc, Ready } from "./types";

const TOKEN_KEY = "verigate.token";
export const getToken = () => sessionStorage.getItem(TOKEN_KEY);
export const setToken = (t: string | null) => (t ? sessionStorage.setItem(TOKEN_KEY, t) : sessionStorage.removeItem(TOKEN_KEY));

// Relative base URL: nginx (Docker) or the Vite dev proxy forwards /api to the backend.
const http = axios.create({ baseURL: "/api", timeout: 300_000 });
http.interceptors.request.use((cfg) => {
  const t = getToken();
  if (t) cfg.headers.Authorization = `Bearer ${t}`;
  return cfg;
});

let onUnauthorized: () => void = () => {};
export const setUnauthorizedHandler = (fn: () => void) => (onUnauthorized = fn);
http.interceptors.response.use(
  (r) => r,
  (err: AxiosError) => {
    if (err.response?.status === 401 && getToken()) onUnauthorized();
    return Promise.reject(err);
  },
);

/** Turn any failure into a sentence a user can act on. */
export function errorMessage(err: unknown): string {
  const e = err as AxiosError<{ detail?: unknown }>;
  if (e?.response) {
    const d = e.response.data?.detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) return d.map((x: { msg?: string }) => x.msg).filter(Boolean).join("; ") || "The request was not valid.";
    if (d && typeof d === "object" && "message" in d) return String((d as { message: unknown }).message);
    return `The server returned an error (HTTP ${e.response.status}).`;
  }
  if (e?.code === "ECONNABORTED") return "The request timed out. The model may still be loading; try again shortly.";
  return "Cannot reach the VerityGate server. Check that the containers are running (docker compose ps).";
}

export const api = {
  register: (email: string, password: string) => http.post<{ token: string; email: string }>("/auth/register", { email, password }).then((r) => r.data),
  login: (email: string, password: string) => http.post<{ token: string; email: string }>("/auth/login", { email, password }).then((r) => r.data),
  ready: () => http.get<Ready>("/ready", { validateStatus: (s) => s === 200 || s === 503 }).then((r) => r.data),
  documents: () => http.get<Doc[]>("/documents").then((r) => r.data),
  upload: (file: File, onProgress: (pct: number) => void) => {
    const form = new FormData();
    form.append("file", file);
    return http
      .post<Doc & { duplicate: boolean; reprocessing?: boolean }>("/documents", form, { onUploadProgress: (e) => onProgress(e.total ? Math.round((e.loaded / e.total) * 100) : 0) })
      .then((r) => r.data);
  },
  remove: (id: string) => http.delete(`/documents/${id}`),
  logout: () => http.post("/auth/logout"),
  ask: (id: string, question: string) => http.post<AskResult>(`/documents/${id}/ask`, { question }).then((r) => r.data),
  audit: () => http.get<AuditRow[]>("/audit", { params: { limit: 100 } }).then((r) => r.data),
};
