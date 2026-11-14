export interface Doc {
  id: string; display_name: string; size_bytes: number; pages: number; chunk_count: number;
  status: "pending" | "processing" | "ready" | "failed"; error: string | null; extraction_method: string | null; created_at: number;
  ocr_pages: number; warnings: string[];
}
export type EvidenceRole = "supporting" | "contradicting" | "related";
export interface Evidence {
  chunk_id: string; page: number; text: string; entailment: number; contradiction: number; cited_by_model: boolean;
  role: EvidenceRole; relevance: number; context: string; note: string;
}
export type ClaimStatus = "SUPPORTED" | "CONTRADICTED" | "NOT_ESTABLISHED";
export interface Claim {
  text: string; status: ClaimStatus; reason: string; reason_code: string; cited_ids: string[]; evidence: Evidence[];
  best_entailment: number; best_contradiction: number; citation_valid: boolean; warnings: string[];
  conflict: boolean; considered: number;
}
export interface Attempt {
  attempt: number; top_k: number; passages: number; claims: number; supported: number; contradicted: number;
  not_established: number; conflicts: number; answer_format: string;
}
export interface ConflictInfo { claim: string; reason: string; evidence: Evidence[] }
export interface AskResult {
  status: "VERIFIED" | "PARTIALLY_VERIFIED" | "ABSTAINED"; message: string; abstain_reason: string | null; verified_answer: string | null;
  claims: Claim[]; unverified_claims: string[]; conflicts: ConflictInfo[]; attempts: Attempt[]; llm_calls: number; retries_used: number;
  latency_seconds: number; retrieval_ms: number; warnings: string[]; sources?: number[]; best_attempt?: number;
}
export interface ModelInfo { model?: string | null; provider?: string | null; runs: string; status?: string | null }
export interface Ready {
  ready: boolean; checks: Record<string, boolean>; llm_configured: boolean; llm_problems: string[];
  llm_provider: string | null; registration_open: boolean;
  models: { llm: ModelInfo; embedding: ModelInfo; nli: ModelInfo; reranker: ModelInfo | null };
  ocr: { enabled: boolean; engine: string | null };
}
export interface AuditRow { ts: number; event: string; doc_id: string | null; detail: Record<string, unknown> }
