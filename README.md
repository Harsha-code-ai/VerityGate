# VerityGate

**Ask questions about a PDF and get answers that are checked against the PDF itself.**

VerityGate reads the PDF you upload, finds the passages that matter for your question, lets a language model write an answer *from those passages only*, and then checks **every statement** of that answer against the PDF again. Statements that the PDF does not support are never shown as verified. If the PDF does not contain the answer, VerityGate says so instead of guessing.

> Be realistic: no AI system is perfect. VerityGate is built to prefer *"I cannot verify this"* over a confident wrong answer, but the checking models can still make mistakes. The measured results (and their limits) are in [`TESTING_REPORT.md`](TESTING_REPORT.md).

---

## 1. What VerityGate does

1. You upload a PDF (text PDFs, tables, multi-column papers, and scanned PDFs when OCR is on).
2. VerityGate extracts the text page by page, keeps page numbers exact, removes repeated headers/footers, and turns tables into readable row sentences.
3. It splits the text into passages and builds two search indexes: meaning-based (embeddings, stored in Qdrant) and keyword-based (BM25). Results are merged (Reciprocal Rank Fusion).
4. A language model (**your choice of provider**) writes the answer as a list of short, complete statements, each citing the passages it used.
5. Each statement is checked sentence by sentence against the PDF with a natural-language-inference (NLI) model, plus strict number/date/currency/percentage checks.
6. You see the answer, the status of every statement, the exact PDF sentences behind it with page numbers, any contradicting evidence, and how many retries were needed.

**What comes from where**

| Part | Runs | Role |
|---|---|---|
| Embedding model (default `all-MiniLM-L6-v2`) | locally in the backend container | finds relevant passages |
| NLI model (default `cross-encoder/nli-deberta-v3-small`) | locally in the backend container | checks statements against PDF sentences |
| Answer model (Anthropic, OpenAI-compatible, or Ollama) | **external API**, or locally if you use Ollama | writes the wording of the answer |
| Facts | **only your PDF** | the model never supplies facts; unverified statements are not shown as verified |

With an external provider, the retrieved excerpts of your PDF (a few paragraphs per question) are sent to that provider. Use Ollama if that is not acceptable.

## 2. Install (Windows, Docker Desktop)

1. Install **Docker Desktop** and start it (WSL 2 backend). Docker Compose v2.24 or newer is required (Docker Desktop 4.27+).
2. Open PowerShell in the project folder and create your settings file:

```powershell
cd VerityGate
copy .env.example .env
notepad .env
```

`.env` is yours: it is never overwritten and never committed. Do not share it (it holds your API key).

## 3. Configure the language model

Edit `.env`:

| Setting | Meaning | Example |
|---|---|---|
| `LLM_PROVIDER` | `anthropic`, `openai` (any OpenAI-compatible server) or `ollama` | `openai` |
| `LLM_MODEL` | a model name available to **your** account | see your provider's model list |
| `LLM_API_KEY` | your key (leave empty for Ollama) | |
| `LLM_BASE_URL` | only for OpenAI-compatible servers or proxies | `https://your-server/v1` |
| `OLLAMA_BASE_URL` | Ollama address; `http://host.docker.internal:11434` if Ollama runs on Windows itself | |
| `EMBEDDING_MODEL`, `NLI_MODEL`, `RERANKER_ENABLED`/`RERANKER_MODEL` | the local models (defaults are fine) | |
| `SECRET_KEY` | required (32+ random characters) when `APP_ENV=production` | `python -c "import secrets; print(secrets.token_urlsafe(48))"` |

If the model service is unreachable or the key is wrong, VerityGate shows a clear error. It never makes up an answer.

## 4. Start

```powershell
docker compose up --build
```

The first start downloads the local models (a few hundred MB) into a Docker volume; later starts are fast. When the log shows the backend as healthy, open **http://localhost:8080**. The status badge (top right) shows whether everything is ready and which models are in use.

| Task | Command |
|---|---|
| Start in the background | `docker compose up -d --build` |
| **Stop, keeping all data** | `docker compose stop` (start again with `docker compose start`) |
| Logs | `docker compose logs -f backend` |
| Container health | `docker compose ps` |
| Rebuild only the backend | `docker compose up -d --build backend` |
| Run the tests | `docker compose --profile test run --rm tests` |
| Run real-model tests (needs internet for the models) | `docker compose --profile test run --rm tests pytest -m real_models` |
| Optional local LLM (Ollama) | `docker compose --profile ollama up -d`, then `docker compose exec ollama ollama pull <model>` |

> `docker compose down` removes the containers but **keeps your data**. Never add `-v` as a troubleshooting step: it permanently deletes every uploaded PDF, index and downloaded model.

## 5. Upload a PDF

Create an account, open **Documents**, drop a PDF (up to 25 MB; the page limit is a setting). The document goes `Queued → Indexing… → Ready`. If it fails, the card says why (password-protected, corrupted, scanned without OCR, ...). Notes such as "OCR was used on page 3" or "hidden text was ignored" are shown on the document. Uploading the same file again after a failure retries it.

## 6. Ask questions

Open **Ask**, pick the document, type a question, press **Ask and verify**. Short, specific questions work best ("When did the Lisbon office open?"). Each question is answered from the selected document only; other users' documents are never searched.

## 7. How answers are verified

* The answer is split into **claims**: complete, stand-alone statements ("The Lisbon office opened on 15 June 2019."). Fragments such as "15 June 2019." are not accepted as claims.
* For each claim, VerityGate looks at the **individual sentences** of the retrieved passages. Only sentences that are really *about* the claim (same names, same subject) are allowed to vote. A sentence about the Madrid office can never contradict a claim about Lisbon.
* An NLI model judges "does this PDF sentence entail the claim?" and a number checker compares amounts, currencies, percentages, units, versions and dates **exactly** (82 million euros is not 82 million dollars; 12.5 % is not 125 %).
* If the PDF itself says two different things, VerityGate shows **both versions with their pages** and does not call either verified.
* If an attempt fails, VerityGate retries (at most twice) with a wider search or with feedback, but only when a retry can plausibly help. It always keeps the best attempt.

## 8. Reading the statuses

| Claim status | Meaning |
|---|---|
| **SUPPORTED** | A sentence in the PDF states this (page shown). |
| **CONTRADICTED** | A relevant sentence in the PDF says something incompatible (shown). |
| **NOT_ESTABLISHED** | The PDF does not clearly state it, a number/date in it is not in the evidence, or the PDF contradicts itself (marked *DOCUMENT CONFLICT*). |

| Answer status | Meaning |
|---|---|
| **VERIFIED** | Every claim is SUPPORTED. |
| **PARTIALLY_VERIFIED** | Some claims are SUPPORTED; the rest are listed separately and are *not* verified. |
| **ABSTAINED** | Nothing could be verified (or the PDF lacks the information). No answer is shown as verified. |

SUPPORTED means "the PDF entails it according to the models and checks", not "it is true in the world".

## 9. Run the tests

```powershell
docker compose --profile test run --rm tests            # fast suite, deterministic stand-in models
python scripts/evaluate.py --mode mock                  # evaluation harness (stand-in models)
python scripts/evaluate.py --mode real --out evaluation_results/real.json   # real models: the numbers that matter
python scripts/calibrate.py --mode real                 # choose thresholds from data
```

Without Docker: `cd backend`, `pip install -r requirements-dev.txt`, `pytest -q`; frontend: `cd frontend`, `npm ci`, `npx tsc --noEmit`, `npm run build`.

Tests that use stand-in models are labelled as such. Tests marked `real_models` use the real embedding and NLI models and are skipped automatically when those cannot be loaded.

## 10. Troubleshooting

| Problem | What to do |
|---|---|
| `docker compose` is not recognised / `required: false` error | Update Docker Desktop (Compose v2.24+). |
| Status badge says **Setup needed** | Fill `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY` in `.env`, then `docker compose up -d`. |
| Page does not load at http://localhost:8080 | `docker compose ps`; look at `docker compose logs backend`. The frontend waits for a healthy backend. |
| First start is slow | The local models are downloading; follow `docker compose logs -f backend`. They are cached in the `hf_cache` volume. |
| Port already in use | Set `FRONTEND_PORT`/`BACKEND_PORT` in `.env`. |
| "Cannot reach Qdrant" | `docker compose ps` (qdrant should be *healthy*); `docker compose restart qdrant backend`. |
| Document failed: "No usable text" | It is a scan. OCR is on by default in Docker; check the language (`OCR_LANGUAGE`) and image quality. |
| "Embedding size changed" | You changed `EMBEDDING_MODEL`. Set a new `QDRANT_COLLECTION` and re-upload (do not delete volumes). |
| Answers always say "Not established" | Run the calibration (`scripts/calibrate.py --mode real`) and adjust the thresholds; check that the NLI model loaded (status badge). |
| NLI error mentioning labels | Set `NLI_LABEL_ORDER` as the message says; VerityGate refuses to run with an unverified label mapping. |
| LLM errors (401/404/429) | Wrong key, wrong model name, or rate limit; the message tells which. |
| Forgot password | There is no email reset; create a new account (data is per account). |

More: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md), [`docs/HUMAN_ACTIONS.md`](docs/HUMAN_ACTIONS.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/SECURITY.md`](docs/SECURITY.md), [`docs/API.md`](docs/API.md), [`docs/EVALUATION.md`](docs/EVALUATION.md), [`PROJECT_AUDIT_REPORT.md`](PROJECT_AUDIT_REPORT.md), [`TESTING_REPORT.md`](TESTING_REPORT.md).

## Limits you should know

* Verification is only as good as the NLI model; it can miss subtle contradictions and cannot do arithmetic (a percentage is not derived from two amounts).
* Ruled tables are understood; borderless tables are read as plain text. Units stated outside a table ("all amounts in millions") are not attached to the table cells.
* A sentence split over two pages is not joined. Printed page labels (iii, 4) are not used: pages are physical page numbers.
* OCR can misread characters; OCR pages are flagged.
* Small documents and the default thresholds are untested with the real models in the build environment; see the testing report.
